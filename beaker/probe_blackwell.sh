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
