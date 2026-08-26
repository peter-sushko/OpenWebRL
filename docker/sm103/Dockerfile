# Rebuild sgl-kernel from source with sm_103a so the image runs on B300 (cc 10.3).
#
# Layered on top of the already-built image rather than rebuilding from the base:
# ENABLE_CUDA_13 is declared near the top of docker/Dockerfile, so changing its
# value invalidates the cache for every layer after it -- including the ~21 min
# flash-attn compile. This keeps everything that image already has (sglang 0.5.9
# with OpenWebRL's megatron+sglang patches applied, TE 2.10, the patched Triton
# fork, apex, Megatron, slime) and replaces only the kernel binary.
#
# Why source: every prebuilt CUDA-12.x sgl-kernel ships cubins for
# sm_80/89/90/100/120 with no PTX (verified with cuobjdump), so B300 fails at
# cuda-graph capture with "RMSNorm failed ... no kernel image is available".
# sgl-kernel's CMakeLists has the compute_103a gencode but gates it behind
# CUDA_VERSION >= 13.0; nvcc 12.9 compiles sm_103a fine and supports the
# --compress-mode=size flag in that same branch, so lowering the gate suffices.
ARG BASE=openwebrl-train-cu13:latest
FROM ${BASE}

RUN cd /sgl-workspace/sglang && \
    (git submodule update --init --recursive sgl-kernel || \
     git submodule update --init --recursive || true) && \
    cd /sgl-workspace/sglang/sgl-kernel && \
    sed -i 's|VERSION_GREATER_EQUAL "13.0"|VERSION_GREATER_EQUAL "12.9"|' CMakeLists.txt && \
    grep -n 'compute_103a' CMakeLists.txt && \
    MAX_JOBS=64 pip install -v . --no-build-isolation --force-reinstall --no-deps && \
    rm -rf /root/.cache/pip

# Fail the build rather than ship an image that cannot run on B300.
RUN python3 - <<'PY'
import glob, subprocess, sys
sos = glob.glob("/usr/local/lib/python3.12/dist-packages/sgl_kernel/**/common_ops*.so", recursive=True)
assert sos, "no sgl_kernel common_ops .so found"
ok = False
for so in sos:
    out = subprocess.run(["cuobjdump", "--list-elf", so], capture_output=True, text=True).stdout
    archs = sorted({t for t in out.split() if t.startswith("sm_")} |
                   {w for w in out.replace(".", " ").split() if w.startswith("sm_")})
    print(so, archs)
    if any("sm_103" in a for a in archs):
        ok = True
if not ok:
    sys.exit("sm_103 cubins missing from rebuilt sgl_kernel -- would fail on B300")
print("sm_103 present")
PY
