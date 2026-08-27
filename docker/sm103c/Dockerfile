# Rebuild sglang-kernel 0.4.0 from ITS OWN source with sm_103a, for B300.
#
# The earlier attempt (docker/sm103) rebuilt the WRONG package. Two distributions
# install the same `sgl_kernel` module:
#   * sgl-kernel 0.3.21      -- from the base image slimerl/sglang:v0.5.9
#   * sglang-kernel 0.4.0    -- pinned in OpenWebRL's requirements.txt (the rename
#                               happened around sglang v0.5.10; 0.4.0 is commit
#                               15097c5c, between tags v0.5.9 and v0.5.10)
# requirements.txt installs sglang-kernel 0.4.0 last, so that is the ABI the
# installed sglang 0.5.6.post2 expects. Rebuilding sgl-kernel 0.3.21 from the
# base's v0.5.9 tree overwrote it with an older API, which produced BOTH
# "TypeError: Mismatched number of arguments when calling
# trtllm_paged_attention_decode" and a silent qwen3_vl import failure that made
# sglang log "has no SGLang implementation, falling back to Transformers
# implementation" -- neither of which happened on H100.
#
# sglang-kernel 0.4.0's own wheel also lacks sm_103 (verified with cuobjdump:
# sm_80/89/90/100/120, no PTX), so it genuinely needs a source build. Its
# CMakeLists has the compute_103a gencode gated behind CUDA_VERSION >= 13.0;
# nvcc 12.9 compiles sm_103a and supports --compress-mode=size in that branch, so
# lowering the gate is enough and keeps the CUDA 12.9 base.
ARG BASE=openwebrl-train-cu13:latest
FROM ${BASE}

ARG SGLANG_KERNEL_COMMIT=15097c5c3b528d9165ddc935d99c5fadef8cdcb5

# Restore the exact wheel first, so anything we do not rebuild keeps 0.4.0's files.
RUN pip install --force-reinstall --no-deps sglang-kernel==0.4.0

RUN git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git /tmp/sglang-src && \
    cd /tmp/sglang-src && \
    git fetch --depth 1 origin ${SGLANG_KERNEL_COMMIT} && \
    git checkout ${SGLANG_KERNEL_COMMIT} && \
    git submodule update --init --recursive sgl-kernel || true

RUN cd /tmp/sglang-src/sgl-kernel && \
    grep -E 'name|version' pyproject.toml | head -4 && \
    sed -i 's|VERSION_GREATER_EQUAL "13.0"|VERSION_GREATER_EQUAL "12.9"|' CMakeLists.txt && \
    grep -n 'compute_103a' CMakeLists.txt && \
    MAX_JOBS=64 pip install -v . --no-build-isolation --force-reinstall --no-deps && \
    rm -rf /tmp/sglang-src /root/.cache/pip

# Fail the build unless sm_103 is actually present in the result.
RUN python3 -c "import glob,subprocess,sys; sos=glob.glob('/usr/local/lib/python3.12/dist-packages/sgl_kernel/**/common_ops*.so',recursive=True); assert sos,'no common_ops .so'; ok=False; [ok := ok or ('sm_103' in subprocess.run(['cuobjdump','--list-elf',s],capture_output=True,text=True).stdout) for s in sos]; print('sm_103 present' if ok else 'MISSING'); sys.exit(0 if ok else 'sm_103 missing from rebuilt sglang-kernel')"
