ARG SGLANG_IMAGE_TAG=v0.5.9
FROM slimerl/sglang:${SGLANG_IMAGE_TAG} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862

ARG ENABLE_CUDA_13=0


# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync dnsutils

# ====================================== Python dependencies ============================================

# The compilation is slow, thus should be put at top
# TransformerEngines does not support too high FA2
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# The compilation is slow, thus should be put at top
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && git submodule update --init && cd hopper/ && \
    MAX_JOBS=96 python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    rm -rf flash-attention/

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.1
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# The released Dockerfile pinned nvidia-mathdx==26.6.0, which exists on neither
# PyPI nor pypi.nvidia.com (both top out at 25.6.0), so ENABLE_CUDA_13=1 died at
# "No matching distribution found". Declared HERE, not beside ENABLE_CUDA_13 at the
# top: an ARG invalidates the cache for every layer after it, and declaring it at
# the top forced a full flash-attn rebuild (~21 min).
#
# pybind11/cmake/ninja are added because this branch builds TE from source with
# --no-build-isolation, so pip never installs TE's own build requirements and
# build_tools/build_ext.py fails at "ModuleNotFoundError: No module named
# 'pybind11'". The ENABLE_CUDA_13=0 branch installs a prebuilt wheel and so never
# hits this.
ARG MATHDX_VERSION=25.6.0

# TE does not have wheel on cuda 13 yet, thus need to install from source
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install "nvidia-mathdx==${MATHDX_VERSION}" pybind11 cmake ninja && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    pip install -e .

RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# This patch from masahi will be included in later Triton releases
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    (cd /root && git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git && cd triton && pip install -r python/requirements.txt && pip install --verbose -e .); \
  fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# ---- B300 / sm_103 support without migrating to CUDA 13 (ENABLE_SM103=1) -------
# B300 is compute capability 10.3. Every prebuilt sgl-kernel for CUDA 12.x ships
# cubins for sm_80/89/90/100/120 and NO PTX, so on B300 the first fused kernel
# dies with "RMSNorm failed ... no kernel image is available for execution on the
# device" during sglang cuda-graph capture. Verified with cuobjdump on both the
# base image's 0.3.21 and the plain 0.3.17.post2 wheel.
#
# The only prebuilt wheel WITH sm_103 is the +cu130 one above, but it needs the
# full CUDA 13 runtime (libcudart/libcublas/libnvrtc .so.13), i.e. a cu130 base
# image -- which ships sglang 0.5.15, where OpenWebRL's own sglang.patch (182
# hunks / 57 files, including models/qwen3_vl.py and
# scheduler_update_weights_mixin.py) does NOT apply. Skipping that patch is not an
# option: it modifies the model being trained and the RL weight-sync path.
#
# So instead: rebuild sgl-kernel from source on the CUDA 12.9 base, adding the
# sm_103a gencode. sgl-kernel's CMakeLists already has that gencode but gates it
# behind CUDA_VERSION >= 13.0, which is conservative -- nvcc 12.9 compiles both
# compute_103a/sm_103a and compute_103/sm_103 (verified), and also supports the
# --compress-mode=size flag in the same branch. Lowering the gate to 12.9 is
# therefore sufficient, and leaves the sglang version and all patches untouched.
ARG ENABLE_SM103=0
RUN if [ "$ENABLE_SM103" = "1" ]; then \
      cd /sgl-workspace/sglang && \
      git submodule update --init --recursive sgl-kernel 2>/dev/null || git submodule update --init --recursive || true; \
      cd /sgl-workspace/sglang/sgl-kernel && \
      sed -i 's|VERSION_GREATER_EQUAL "13.0"|VERSION_GREATER_EQUAL "12.9"|' CMakeLists.txt && \
      grep -n 'compute_103a' CMakeLists.txt && \
      MAX_JOBS=64 pip install -v . --no-build-isolation --force-reinstall --no-deps; \
    fi

# https://github.com/pytorch/pytorch/issues/168167
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch

# TODO temporarily skip patching for GB200/GB300 (and require users to bring their own sglang version). should add back later.
ARG ENABLE_SGLANG_PATCH=1
COPY docker/patch/${PATCH_VERSION}/sglang.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
  cd /sgl-workspace/sglang && \
  git update-index --refresh && \
  git apply sglang.patch --3way && \
  if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
  fi && \
  rm sglang.patch; \
fi

# ====================================== Install main package ============================================

ARG SLIME_COMMIT=main
RUN git clone https://github.com/THUDM/slime.git /root/slime && \
    cd /root/slime && \
    git checkout ${SLIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation
