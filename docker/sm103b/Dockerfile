# Force flashinfer to JIT-compile for the local arch instead of using its
# prebuilt AOT cache, which has no sm_103 kernels.
#
# flashinfer-jit-cache ships precompiled .so files per kernel config (built
# +cu129 for sm_90/100/120). On B300 sglang's flashinfer backend dies at
# cuda-graph capture with "BatchPrefillWithPagedKVCache failed with error no
# kernel image is available": the cache entry exists, so flashinfer uses it
# rather than compiling, and that entry has no sm_103 code.
#
# Removing the cache package makes flashinfer-python fall back to true JIT, which
# compiles for the detected device. The image already sets
# FLASHINFER_DISABLE_VERSION_CHECK=1 because flashinfer-python 0.5.3 and
# flashinfer-jit-cache 0.6.3 are mismatched anyway.
ARG BASE=openwebrl-train-sm103:latest
FROM ${BASE}

RUN pip uninstall -y flashinfer-jit-cache && \
    python3 -c "import importlib.util as u, importlib.metadata as m; print('flashinfer-python', m.version('flashinfer-python')); assert u.find_spec('flashinfer_jit_cache') is None, 'AOT cache still present'; print('flashinfer AOT cache removed; JIT path active')"
