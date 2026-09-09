# Fast browser step

`env.step()` on Browserbase took about 9 s per action, of which 7.8 s was timers.
This note records what was slow, what changed, and what it measured. All numbers
are Browserbase with advanced stealth at 1280x720, OpenWebRL-4B-RL-s2-iter139
served by sglang on one GPU.

## Where the 9 s went (stock `scroll`, booking.com, median of 8)

| operation | s |
|---|---|
| `wait_for_load_state("networkidle", 5000)` | 5.01 (always the full cap; booking never goes idle) |
| `wait_for_timeout(1000)` after every action | 1.00 |
| CAPTCHA grace polling after every action (3 x 0.5 s + pumps) | 1.83 |
| `page.screenshot()` PNG | 0.66 |
| ~7 CDP round trips at 76 ms (scroll, a11y walk x2, titles) | 0.57 |
| **total** | **9.14**, of which 1.24 s is work |

Same code on a local Chromium: 6.3 s. The remote browser was not the problem.

## What changed (`openwebrl/env/web_env.py`, `browserbase_env.py`)

All of it is behind `SLIME_BROWSER_FAST_STEP` (default on; set `0` for the old path).

1. **Readiness check instead of timers** (`_settle_until_ready`). After an action:
   wait up to `SLIME_BROWSER_NAV_GRACE_MS` (500) for a main-frame navigation
   *request* to start; if one starts, wait for it to *commit* (`framenavigated`),
   then for `load`; then run one in-page check that resolves when
   `document.readyState` is complete, fonts are loaded, every `<img>` intersecting
   the viewport is decoded, no resource finished in the last
   `SLIME_BROWSER_SETTLE_QUIET_MS` (300), and the DOM signature is unchanged for
   two 50 ms polls. Capped at `SLIME_BROWSER_SETTLE_MAX_MS` (3000). Scroll and
   `goto_url` skip the navigation grace. Long-polls never finish, so unlike
   `networkidle` they do not block this.
   The commit wait matters: without it the check passes on the *old* document and
   the model gets a stale screenshot (measured: 7 `wait` calls and 15 extra steps
   over 5 tasks).
   The check also tracks in-flight *content* requests through Playwright's
   request events (documents, scripts, styles, fonts, images from any host, and
   fetch/XHR to the page's own site; beacons, third-party XHR, websockets and
   media are ignored) and keeps waiting while any is younger than
   `SLIME_BROWSER_YOUNG_REQUEST_MS` (1500). Without this, pages that fetch their
   content after `load` (Best Buy search results) passed every DOM check while
   still empty, and an RL rerun's pre-training eval fell from 0.73 to 0.54 task
   success; with it the eval is back at 0.73. While `document.readyState` is
   still loading the wait may run to `SLIME_BROWSER_SETTLE_LOAD_MAX_MS` (10000).
2. **No blind CAPTCHA grace polls.** One pump drains queued console events; the
   wait loop only runs if a `browserbase-solving-*` event actually arrived.
3. **Screenshot over raw CDP** (`Page.captureScreenshot`, `optimizeForSpeed`):
   PNG 0.99 -> 0.38 s remote; `SLIME_BROWSER_FAST_SCREENSHOT_FORMAT=jpeg` gives
   0.17 s and 58 KB but changes the model input format, so PNG stays the default.
4. **Skip the a11y walk.** `step()` walked the DOM twice per action only to build an
   `env_message` diff that nothing reads; the walker is quadratic in DOM size and
   averaged ~2 s per step in training rollouts.
5. **Fetch the observation concurrently.** Screenshot and tab titles go out in one
   `asyncio.gather`; CDP pipelines them.
6. **Episode setup.** `reset()` no longer re-navigates to the start URL that
   `setup()` just loaded, and the two fixed `sleep(2)` become the readiness check.

Also, independent of the flag: **`write` now refuses when focus is not on an
editable element.** It used to press Ctrl+A / Backspace / type regardless, which
selected the whole page and typed nowhere (about 20% of writes in eval). The tool
response now says to click the field first.

`SLIME_BROWSER_DUMP_SHOTS_DIR=<dir>` dumps every step's screenshot plus its
readiness report for auditing. Turn-level rollouts now record `step_env_secs`,
`step_infer_secs`, `env_setup_secs` and per-phase env times in each turn's
metadata; `run_evaluate.py` writes them out as `turn_timings`.

## Measurements with the model in the loop

Online-Mind2Web, all 300 tasks, 20 concurrent episodes, 287 tasks completed in both
arms:

| | control | fast |
|---|---|---|
| env s/step | 11.5 | 4.2 |
| inference s/step | 3.4 | 4.9 |
| total s/step | 14.9 | 9.1 |
| steps / episode | 10.8 | 11.2 |
| episode wall | 183 s | 116 s |
| whole run | 56 min | 35 min |
| success | 147 (51%) | 141 (49%) |

Inference per step rises only because the faster env keeps ~11 requests in flight
at the single GPU instead of ~5 (estimated queue wait 2.4 s vs 0.9 s); GPU work per
step is unchanged. Success is within noise. The model calls `wait` on ~3% of steps
vs ~1% in control.

Screenshot audit of the fast arm (3060 steps): 2894 settled clean; 88 hit the cap
with the document still loading, 63 with resources still finishing, 11 with a
viewport image pending, 4 on fonts. The worst of the 11 was one grey ad panel.

WebVoyager, 45 tasks (3 per site), 29 completed in both arms: 14.1 -> 8.7 s/step,
steps 199 vs 190, success 19 vs 18.

## Remaining cost

Sites that never stop loading (booking.com results pages) run the check to its
3 s cap on every action. Browserbase's CAPTCHA handling waits the full 30 s
whenever a `solving-started` event arrives without a `solving-finished`, which
booking.com does on every session; that is unchanged here. Setting
`SLIME_BROWSER_NETWORKIDLE_TIMEOUT_MS=0` on the old path hangs forever
(Playwright reads 0 as no timeout).

## Reproducing

`junk/bbprof/profile_step.py` (outside the repo) drives the real env classes with
the model stubbed to a scroll and times every Playwright call; `--fast` applies the
same changes as flags. `junk/bbprof/analyze_eval_timing.py <run_dir>` aggregates
`turn_timings` from eval results; `audit_shots.py <dump_dir> <out.html>` builds a
gallery from a screenshot dump.
