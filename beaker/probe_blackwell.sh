#!/usr/bin/env bash
# Cheap capability probe: does the existing Hopper-built image work on holmes?
# Answers, in order: what GPU is this, does torch see it, do the compiled
# extensions (flash-attn, TransformerEngine, sgl-kernel, apex) import and run.
set -x
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())
props = torch.cuda.get_device_properties(0)
print("total_mem_GiB", round(props.total_memory/1024**3, 1))
# a real kernel launch, not just an import
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", (a @ a).sum().item() is not None)
for mod in ("flash_attn", "transformer_engine.pytorch", "sgl_kernel", "apex", "sglang"):
    try:
        __import__(mod); print("IMPORT OK", mod)
    except Exception as e:
        print("IMPORT FAIL", mod, type(e).__name__, str(e)[:160])
# TE is the one that actually OOMed us; exercise a real TE linear
try:
    import transformer_engine.pytorch as te
    lin = te.Linear(2048, 2048, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    y = lin(x); y.sum().backward()
    print("TE linear fwd+bwd OK")
except Exception as e:
    print("TE linear FAIL", type(e).__name__, str(e)[:300])
PY

# Importing a custom-kernel package proves nothing: sgl_kernel ships precompiled
# cubins, so on an arch it was not built for (B300 = sm_103) the import succeeds
# and the first real call dies with "no kernel image is available for execution
# on the device". EXECUTE one kernel from each such package.
python3 - <<'EOF'
import torch
def check(label, fn):
    try:
        fn(); print("EXEC OK", label)
    except Exception as e:
        print("EXEC FAIL", label, type(e).__name__, str(e)[:160])

x = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")

def rms():
    import sgl_kernel
    # Call through the registered op and let torch bind the args, so a signature
    # change in sgl_kernel does not masquerade as a kernel failure.
    out = torch.ops.sgl_kernel.rmsnorm.default(x, w, 1e-6, False) \
          if hasattr(torch.ops.sgl_kernel.rmsnorm, "default") else sgl_kernel.rmsnorm(x, w, 1e-6)
    torch.cuda.synchronize()
    assert out is not None
check("sgl_kernel.rmsnorm", rms)

def fa():
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 8, 128, dtype=torch.bfloat16, device="cuda")
    flash_attn_func(q, q, q); torch.cuda.synchronize()
check("flash_attn_func", fa)
EOF

# Triton JITs at runtime, so it must recognise the arch. @jit functions cannot be
# defined on stdin ("@jit functions should be defined in a Python file"), so write
# the test to a real file before running it.
cat > /tmp/_triton_probe.py <<'EOF'
import torch, triton, triton.language as tl

@triton.jit
def _add1(p, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(p + off, tl.load(p + off, mask=m) + 1.0, mask=m)

print("triton", triton.__version__)
x = torch.zeros(4096, device="cuda", dtype=torch.float32)
_add1[(4,)](x, x.numel(), BLOCK=1024)
torch.cuda.synchronize()
assert torch.allclose(x, torch.ones_like(x)), "triton kernel wrote wrong values"
print("EXEC OK triton jit")
EOF
python3 /tmp/_triton_probe.py 2>&1 | tail -3 || echo "EXEC FAIL triton jit"

# TorchMemorySaver is what --colocate relies on to hand GPU memory back and forth
# between sglang and Megatron. On B300 the RL run completes cuda-graph capture and
# then hangs forever without ever logging "Rollout offload succeeded", with the
# same behaviour under two different attention backends -- so test pause/resume
# directly here rather than inferring it from an 8-GPU hang.
cat > /tmp/_tms_probe.py <<'EOF'
import torch, time
try:
    from torch_memory_saver import torch_memory_saver as tms
except Exception:
    try:
        import torch_memory_saver as _m
        tms = getattr(_m, "torch_memory_saver", _m)
    except Exception as e:
        print("EXEC FAIL torch_memory_saver import", type(e).__name__, str(e)[:120]); raise SystemExit(0)

free0 = torch.cuda.mem_get_info()[0] / 2**30
try:
    with tms.region():
        buf = torch.empty(2 * 1024**3 // 2, dtype=torch.float16, device="cuda")  # ~2 GiB
    torch.cuda.synchronize()
    held = torch.cuda.mem_get_info()[0] / 2**30
    t0 = time.time()
    tms.pause()
    torch.cuda.synchronize()
    after_pause = torch.cuda.mem_get_info()[0] / 2**30
    tms.resume()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"EXEC OK torch_memory_saver pause/resume in {dt:.2f}s "
          f"(free GiB: start {free0:.1f} -> alloc {held:.1f} -> paused {after_pause:.1f})")
except Exception as e:
    print("EXEC FAIL torch_memory_saver", type(e).__name__, str(e)[:160])
EOF
timeout 300 python3 /tmp/_tms_probe.py 2>&1 | tail -4 || echo "EXEC FAIL torch_memory_saver TIMEOUT(300s) -- this is the --colocate blocker"
