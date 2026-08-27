# Build sgl-kernel from the source that MATCHES the installed sglang, with sm_103.
#
# The installed sglang is 0.5.6.post2, and sglang's own tree at tag v0.5.6.post2
# declares sgl-kernel 0.3.19. Pairing those makes the Python layer and the compiled
# .so come from one revision, which is what the earlier attempts got wrong:
#
#   docker/sm103  -> rebuilt sgl-kernel 0.3.21 from the base's v0.5.9 tree, while the
#                    Python layer still came from 0.3.17.post2/0.4.0. (The
#                    "Mismatched number of arguments when calling
#                    trtllm_paged_attention_decode" seen on that image is a separate
#                    flashinfer problem -- that op only has callers in
#                    flashinfer/decode.py -- and is handled in docker/sm103f.)
#   docker/sm103c -> rebuilt sglang-kernel 0.4.0, whose Python layer dropped
#                    sgl_per_tensor_quant_fp8 (removed after 0.3.21), so sglang
#                    could not import it and fell back to Transformers for
#                    Qwen3-VL, then failed outright.
#
# Prebuilt wheels at every version lack sm_103 (cuobjdump: sm_80/89/90/100/120, no
# PTX), so a source build is required regardless. sgl-kernel's CMakeLists has the
# compute_103a gencode gated behind CUDA_VERSION >= 13.0; nvcc 12.9 compiles
# sm_103a and supports --compress-mode=size in that branch, so lowering the gate is
# sufficient and keeps the CUDA 12.9 base.
ARG BASE=openwebrl-train-cu13:latest
FROM ${BASE}

ARG SGLANG_TAG=v0.5.6.post2

RUN git clone --depth 1 --branch ${SGLANG_TAG} https://github.com/sgl-project/sglang.git /tmp/sglang-src && \
    cd /tmp/sglang-src && git submodule update --init --recursive sgl-kernel

RUN cd /tmp/sglang-src/sgl-kernel && \
    grep -E '^(name|version)' pyproject.toml | head -4 && \
    sed -i 's|VERSION_GREATER_EQUAL "13.0"|VERSION_GREATER_EQUAL "12.9"|' CMakeLists.txt && \
    grep -n 'compute_103a' CMakeLists.txt && \
    MAX_JOBS=64 pip install -v . --no-build-isolation --force-reinstall --no-deps && \
    rm -rf /tmp/sglang-src /root/.cache/pip

# Fail the build unless BOTH conditions hold: sm_103 cubins present, and the
# Python layer still exports the symbol sglang 0.5.6.post2 imports at startup.
RUN python3 -c "import glob,subprocess,sys; \
sos=glob.glob('/usr/local/lib/python3.12/dist-packages/sgl_kernel/**/common_ops*.so',recursive=True); \
assert sos,'no common_ops .so'; \
archs=[subprocess.run(['cuobjdump','--list-elf',s],capture_output=True,text=True).stdout for s in sos]; \
assert any('sm_103' in a for a in archs),'sm_103 missing'; print('sm_103 OK')" && \
    grep -rq "sgl_per_tensor_quant_fp8" /usr/local/lib/python3.12/dist-packages/sgl_kernel/ && echo "symbol OK"
