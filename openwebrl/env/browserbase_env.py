"""Browserbase remote browser environment.

Same shape as ``browser_use_env.py``: a ``WebEnv`` that attaches to a remote
Chromium over CDP instead of launching one locally. Two things differ, and both
are the reason this file exists at all:

  * Browserbase sessions are created with ``advanced_stealth`` + residential
    proxies, so the pages we hit are far less likely to serve a 403/CAPTCHA
    interstitial than headless Chromium from a datacenter IP.
  * When a CAPTCHA *is* served, Browserbase solves it in-band and announces the
    fact through two console messages. We must sit completely still while that
    happens -- see ``_wait_for_captcha_if_needed``.

Measured behaviour worth knowing (2026-08-26, advanced stealth on):

  * Browserbase pins the viewport to **1280x720** whenever ``advanced_stealth``
    is on. ``browser_settings.viewport`` and ``fingerprint.screen`` are both
    ignored; a CDP ``Emulation.setDeviceMetricsOverride`` does resize it but
    then breaks ``page.screenshot``. With stealth off the requested viewport
    (e.g. 1280x1000) is honoured. So matching the paper's observation size and
    running stealth are mutually exclusive -- we take stealth, and ``setup()``
    realigns ``screen_size``/``dpr`` so click coordinates stay correct.

Cleanup mirrors the ``.sandboxes/`` / ``.browser_use_sessions/`` marker pattern:
one empty file per live session, removed on release, swept at the start of the
next run or via ``python -m openwebrl.env.browserbase_env --cleanup``.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from playwright.async_api import async_playwright

from openwebrl.env.web_env import WebEnv

logger = logging.getLogger(__name__)

# Browserbase fires solving-started/solving-finished pairs; a page can throw
# several challenges in a row. Resume only once no new event has arrived for
# _CAPTCHA_SETTLE_SECS *and* the last event was a "finished".
_CAPTCHA_SETTLE_SECS = 3.0
_CAPTCHA_TIMEOUT_SECS = 30.0
# Grace polls before concluding "no CAPTCHA": the console event can land a beat
# after the action returns, and moving on early is what trips Kasada.
_CAPTCHA_GRACE_POLLS = 3
_CAPTCHA_POLL_SECS = 0.5

_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".browserbase_sessions"
)


def _save_session_id(session_id: Any) -> None:
    os.makedirs(_MANIFEST_DIR, exist_ok=True)
    with open(os.path.join(_MANIFEST_DIR, str(session_id)), "w") as f:
        f.write("")


def _remove_session_id(session_id: Any) -> None:
    try:
        os.remove(os.path.join(_MANIFEST_DIR, str(session_id)))
    except OSError:
        pass


def _list_session_ids() -> List[str]:
    if not os.path.isdir(_MANIFEST_DIR):
        return []
    try:
        return os.listdir(_MANIFEST_DIR)
    except OSError:
        return []


def _resolve_credentials(cfg: Optional[Dict[str, Any]]) -> "tuple[str, str]":
    cfg = cfg or {}
    api_key = os.environ.get("BROWSERBASE_API_KEY") or cfg.get("api_key")
    project_id = os.environ.get("BROWSERBASE_PROJECT_ID") or cfg.get("project_id")
    if not api_key or not project_id:
        raise ValueError(
            "Set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID env vars, or "
            "browserbase.api_key / browserbase.project_id in config.yaml"
        )
    return str(api_key), str(project_id)


async def cleanup_existing_browserbase_sessions(
    browserbase_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Release any leftover sessions recorded in the manifest directory."""
    session_ids = _list_session_ids()
    if not session_ids:
        logger.info("No leftover Browserbase session markers found.")
        return

    from browserbase import AsyncBrowserbase

    api_key, project_id = _resolve_credentials(browserbase_cfg)
    logger.info(f"Found {len(session_ids)} leftover Browserbase session(s), releasing...")

    client = AsyncBrowserbase(api_key=api_key)
    for sid in session_ids:
        try:
            await client.sessions.update(sid, project_id=project_id, status="REQUEST_RELEASE")
            print(f"♻️  [Released Browserbase session {sid}]")
        except Exception as exc:
            # Already-finished sessions 404 here; same outcome, drop the marker.
            logger.warning("Release failed for %s (may already be gone): %s", sid, exc)
        _remove_session_id(sid)


class BrowserbaseWebEnv(WebEnv):
    """WebEnv variant that drives a remote Browserbase browser via CDP."""

    def __init__(
        self,
        cdp_url: str,
        session_id: str,
        bb_client: Any,
        project_id: str,
        **web_env_kwargs: Any,
    ) -> None:
        super().__init__(**web_env_kwargs)
        self.cdp_url = cdp_url
        self.session_id = session_id
        self._bb = bb_client
        self._project_id = project_id
        self._captcha_events: List[Dict[str, str]] = []
        # WebEnv.reset() re-enters _initialize_context; without this the console
        # handlers stack up and every CAPTCHA event gets counted N times.
        self._listeners_attached = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        self.playwright = await async_playwright().start()
        self.browser_type = self.playwright.chromium
        self.browser = await self.browser_type.connect_over_cdp(self.cdp_url)

        await self._initialize_context(
            enable_recording=False,
            start_url=self.start_url,
            auth_info=self.auth_info,
        )

        # Browserbase sizes the viewport at session-create time and stealth mode
        # may adjust it. WebEnv.execute_single_action scales model coordinates by
        # self.screen_size / self.dpr, so those must describe the REAL viewport or
        # every click lands off-target. Realign to what the remote actually gave us.
        try:
            dims = await self.page.evaluate(
                "() => ({w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio})"
            )
            real_dpr = max(1, int(dims["dpr"] or 1))
            real_w = int(dims["w"]) * real_dpr
            real_h = int(dims["h"]) * real_dpr
            if (real_w, real_h, real_dpr) != (self.screen_size[0], self.screen_size[1], self.dpr):
                logger.warning(
                    "Remote viewport %dx%d @dpr=%d differs from configured "
                    "%dx%d @dpr=%d; using actual dimensions for coord transforms.",
                    real_w, real_h, real_dpr,
                    self.screen_size[0], self.screen_size[1], self.dpr,
                )
                self.screen_size = (real_w, real_h)
                self.dpr = real_dpr
                self.css_width = int(dims["w"])
                self.css_height = int(dims["h"])
        except Exception as exc:
            logger.warning("Could not query remote viewport; keeping configured dims: %s", exc)

        logger.info(
            f"BrowserbaseWebEnv ready (session_id={self.session_id}, "
            f"viewport={self.screen_size}, DPR={self.dpr})"
        )

    async def _initialize_context(
        self,
        enable_recording: bool = False,
        start_url: str = "about:blank",
        auth_info: Optional[dict] = None,
    ) -> None:
        # Always use the Browserbase-provisioned context. Calling new_context()
        # would bypass its fingerprinting and stealth configuration, which is the
        # entire reason we are paying for this browser.
        self.context = (
            self.browser.contexts[0]
            if self.browser.contexts
            else await self.browser.new_context()
        )
        self.context.set_default_timeout(self.timeout)

        # Attach to the initial page and to every future tab/popup, so a CAPTCHA
        # served in a new tab is not missed.
        if not self._listeners_attached:
            for page in self.context.pages:
                self._attach_captcha_listener(page)
            self.context.on("page", self._attach_captcha_listener)
            self._listeners_attached = True

        assert start_url
        start_urls = start_url.split(" |AND| ")
        existing_pages = list(self.context.pages)

        for i, url in enumerate(start_urls):
            page = existing_pages[i] if i < len(existing_pages) else await self.context.new_page()
            for attempt in range(self.max_retries):
                try:
                    await page.goto(url, timeout=self.init_navigation_timeout, wait_until="domcontentloaded")
                    try:
                        await page.wait_for_load_state("load", timeout=5000)
                    except Exception:
                        pass
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise EnvironmentError(
                            f"Failed to navigate to {url} after {self.max_retries} attempts: {e}"
                        )
                    logger.warning(f"Navigate to {url} failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2)

        self.page = self.context.pages[0]
        await self.page.bring_to_front()
        self.setup_global_page_listener()
        self.setup_dialog_interceptor()

        # The landing page itself can be gated; let any interstitial resolve
        # before the first screenshot is taken.
        await self._wait_for_captcha_if_needed()
        await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # CAPTCHA handling
    # ------------------------------------------------------------------

    def _attach_captcha_listener(self, page: Any) -> None:
        def handle_console(msg: Any) -> None:
            text = getattr(msg, "text", "")
            if text == "browserbase-solving-started":
                print("🧩 CAPTCHA solving started")
                self._captcha_events.append(
                    {"event": "solving-started", "timestamp": datetime.now(timezone.utc).isoformat()}
                )
            elif text == "browserbase-solving-finished":
                print("🔓 CAPTCHA solving finished")
                self._captcha_events.append(
                    {"event": "solving-finished", "timestamp": datetime.now(timezone.utc).isoformat()}
                )

        try:
            page.on("console", handle_console)
        except Exception as exc:
            logger.warning("Could not attach CAPTCHA listener to page: %s", exc)

    async def _pump(self) -> None:
        """Cheap round-trip that lets queued CDP console events drain."""
        try:
            await self.context.cookies()
        except Exception:
            pass

    async def _wait_for_captcha_if_needed(self) -> None:
        """
        Freeze all automation while Browserbase solves a CAPTCHA.

        Anti-bot systems (Kasada in particular) watch for automation traffic
        *during* a challenge, so between solving-started and solving-finished we
        must issue nothing but the cookie pump that drains the event queue.
        """
        await self._pump()

        if not self._captcha_events:
            for _ in range(_CAPTCHA_GRACE_POLLS):
                await asyncio.sleep(_CAPTCHA_POLL_SECS)
                await self._pump()
                if self._captcha_events:
                    break

        if not self._captcha_events:
            return

        print("⏳ CAPTCHA activity detected — pausing automation until fully settled...")
        start = time.time()
        last_count = len(self._captcha_events)
        last_change = time.time()
        settled = False

        while time.time() - start < _CAPTCHA_TIMEOUT_SECS:
            await asyncio.sleep(_CAPTCHA_POLL_SECS)
            await self._pump()

            count = len(self._captcha_events)
            if count != last_count:
                last_count = count
                last_change = time.time()
                continue

            if time.time() - last_change >= _CAPTCHA_SETTLE_SECS:
                if self._captcha_events[-1].get("event") == "solving-finished":
                    print(
                        f"✅ CAPTCHA settled in {time.time() - start:.1f}s "
                        f"({last_count} events) — resuming automation"
                    )
                    settled = True
                    break
                # Last event is solving-started: the solver is still working.

        if not settled:
            print(
                f"🔒 Timeout: CAPTCHA not resolved within {_CAPTCHA_TIMEOUT_SECS:.0f}s "
                f"({len(self._captcha_events)} events seen)"
            )

        # Solving usually ends in a redirect; let it land before we look again.
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        await asyncio.sleep(1.0)

        self._captcha_events = []

    async def execute_single_action(self, action: Dict[str, Any]) -> "tuple[bool, str]":
        # The CAPTCHA is triggered BY the action, so the wait has to come after
        # it and before WebEnv.step()'s a11y-tree / screenshot calls.
        result = await super().execute_single_action(action)
        await self._wait_for_captcha_if_needed()
        return result

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def get_screen_size(self) -> "tuple[int, int]":
        # page.viewport_size is None for CDP-attached browsers (Playwright did
        # not launch this one), so fall back to the realigned size.
        viewport = self.page.viewport_size
        if viewport:
            return int(viewport["width"] * self.dpr), int(viewport["height"] * self.dpr)
        return self.screen_size[0], self.screen_size[1]

    async def exit(self) -> None:
        # Never close context/browser -- they live on the remote side. Release
        # the session so billing stops now rather than at the project timeout.
        try:
            # project_id is REQUIRED here. Omitting it raises, the session is
            # never released, and it idles until the project defaultTimeout --
            # which lands mid-episode on long tasks and scores as a failure.
            await self._bb.sessions.update(
                self.session_id, project_id=self._project_id, status="REQUEST_RELEASE"
            )
            print(f"♻️  [Released Browserbase session {self.session_id}]")
        except Exception as exc:
            logger.warning("Release failed for %s: %s", self.session_id, exc)

        _remove_session_id(self.session_id)

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as exc:
                logger.warning("playwright.stop() failed: %s", exc)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def create_browserbase_env(
    browserbase_cfg: Dict[str, Any],
    env_config: Dict[str, Any],
    tool_list: Optional[List[Dict[str, Any]]],
    policy: Optional[str],
    start_url: str,
) -> BrowserbaseWebEnv:
    """Create a Browserbase session and wrap it in a ``BrowserbaseWebEnv``."""
    from browserbase import AsyncBrowserbase

    browserbase_cfg = browserbase_cfg or {}
    api_key, project_id = _resolve_credentials(browserbase_cfg)
    client = AsyncBrowserbase(api_key=api_key)

    # Viewport is requested in CSS pixels (pre-DPR), same convention as Browser-Use.
    dpr = max(1, int(env_config.get("dpr", 1)))
    css_w = int(env_config["width"]) // dpr
    css_h = int(env_config["height"]) // dpr

    browser_settings: Dict[str, Any] = {
        "advanced_stealth": _as_bool(
            os.environ.get("BROWSERBASE_ADVANCED_STEALTH", browserbase_cfg.get("advanced_stealth")), True
        ),
        "solve_captchas": _as_bool(
            os.environ.get("BROWSERBASE_SOLVE_CAPTCHAS", browserbase_cfg.get("solve_captchas")), True
        ),
        "viewport": {"width": css_w, "height": css_h},
    }

    # Proxies are what change the egress IP; advanced stealth alone does not.
    # They bill per GB, hence the explicit switch.
    proxies = _as_bool(os.environ.get("BROWSERBASE_PROXIES", browserbase_cfg.get("proxies")), True)

    create_kwargs: Dict[str, Any] = {
        "project_id": project_id,
        "browser_settings": browser_settings,
        "proxies": proxies,
    }
    # Without this the project defaultTimeout (often 1800s) can end the session
    # mid-episode; let the step budget bind instead.
    session_timeout = os.environ.get("BROWSERBASE_SESSION_TIMEOUT_S") or browserbase_cfg.get("timeout_secs")
    if session_timeout:
        create_kwargs["api_timeout"] = int(session_timeout)
    region = os.environ.get("BROWSERBASE_REGION") or browserbase_cfg.get("region")
    if region:
        create_kwargs["region"] = region

    session = None
    try:
        print(
            f"⏱️  Creating Browserbase session (viewport={css_w}x{css_h}, "
            f"stealth={browser_settings['advanced_stealth']}, "
            f"solve_captchas={browser_settings['solve_captchas']}, "
            f"proxies={proxies}, timeout_s={create_kwargs.get('api_timeout')})..."
        )
        session = await client.sessions.create(**create_kwargs)
        # Marker BEFORE anything that can fail, so a Ctrl+C still leaves a trace.
        _save_session_id(session.id)
        print(f"✅ [Created Browserbase session {session.id}]")

        cdp_url = f"wss://connect.browserbase.com?sessionId={session.id}&apiKey={api_key}"

        try:
            debug = await client.sessions.debug(session.id)
            if getattr(debug, "pages", None):
                print(f"👀 [Browserbase live view: {debug.pages[0].debugger_fullscreen_url}]")
        except Exception:
            pass

        env = BrowserbaseWebEnv(
            cdp_url=cdp_url,
            session_id=session.id,
            bb_client=client,
            project_id=project_id,
            width=int(env_config["width"]),
            height=int(env_config["height"]),
            dpr=dpr,
            max_retries=int(env_config["max_retries"]),
            wait_timeout=int(env_config["wait_timeout"]),
            screenshot_timeout=env_config.get("screenshot_timeout"),
            start_url=start_url,
            resize_output_coords=bool(env_config["resize_output_coords"]),
            resize_scale=int(env_config["resize_scale"]),
            image_patch_size=int(env_config["image_patch_size"]),
            tool_list=tool_list,
            policy=policy,
        )
        await env.setup()
        return env

    except BaseException:
        if session is not None:
            try:
                await client.sessions.update(
                    session.id, project_id=project_id, status="REQUEST_RELEASE"
                )
                print(f"♻️  [Released Browserbase session {session.id} after failure]")
            except Exception as exc:
                logger.warning("Release-on-failure failed: %s", exc)
            _remove_session_id(session.id)
        raise


# ---------------------------------------------------------------------------
# CLI: python -m openwebrl.env.browserbase_env --cleanup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browserbase environment utilities")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Release all leftover sessions recorded in .browserbase_sessions/",
    )
    args = parser.parse_args()

    if not args.cleanup:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ids = _list_session_ids()
    if not ids:
        print("No session markers found in", _MANIFEST_DIR)
        sys.exit(0)

    print(f"Found {len(ids)} Browserbase session(s) to clean up:")
    for sid in ids:
        print(f"  - {sid}")

    asyncio.run(cleanup_existing_browserbase_sessions({}))
    print("Cleanup complete.")
