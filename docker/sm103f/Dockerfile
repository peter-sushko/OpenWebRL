# Align flashinfer on top of the sm_103 kernel build (docker/sm103e).
#
# The base image ships three flashinfer distributions at two versions:
#   flashinfer-python 0.5.3, flashinfer-cubin 0.5.3, flashinfer-jit-cache 0.6.3+cu129
# The AOT modules in jit-cache 0.6.3 have a different signature than the 0.5.3
# Python layer calls, which is what produced
#   "Mismatched number of arguments when calling trtllm_paged_attention_decode"
# on B300 (the only callers of that op live in flashinfer/decode.py, not in
# sgl-kernel -- the comment in docker/sm103/Dockerfile blamed the wrong package).
#
# No matching flashinfer-jit-cache 0.5.3+cu129 wheel is published, so drop the AOT
# cache and let flashinfer-python JIT its own modules at engine start: same version
# for interface and implementation by construction. flashinfer-cubin 0.5.3 already
# matches, so the cubin path is untouched.
ARG BASE=openwebrl-train-sm103e:latest
FROM ${BASE}

RUN pip uninstall -y flashinfer-jit-cache && \
    pip list 2>/dev/null | grep -iE "^(flashinfer|sgl-kernel|sglang|sglang-kernel) " && \
    python3 -c "import glob,subprocess; \
sos=glob.glob('/usr/local/lib/python3.12/dist-packages/sgl_kernel/**/common_ops*.so',recursive=True); \
a=[subprocess.run(['cuobjdump','--list-elf',s],capture_output=True,text=True).stdout for s in sos]; \
assert any('sm_103' in x for x in a), 'sm_103 cubins lost'; print('sm_103 still present')"
