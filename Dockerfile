# syntax=docker/dockerfile:1
# Inference engine (Intel Arc B-series / SYCL) — reproducible build + slim runtime.
#
#   docker build -t ie-engine .
#   ./scripts/ie-docker pull llama8b
#   ./scripts/ie-docker serve <model.gguf> --gpus 1        # then hit :11435/v1
#
# Requires a HOST with Intel Arc (B-series, e.g. B70) + the i915/xe kernel driver
# and /dev/dri render nodes. oneAPI 2026.x is installed from Intel's apt repo (it
# isn't on Docker Hub yet). The engine compiles to SPIR-V (JIT); the bundled
# Level-Zero runtime + IGC specialize it for the actual Arc device on first load.
#
# Runtime is copy-only: just the ~11 oneAPI libs the binary + the Level-Zero UR
# adapter actually need, not the full 1.2 GB oneAPI runtime. See QUICKSTART.md.

ARG ONEAPI_KEY=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

# ============================ build stage ============================
FROM ubuntu:24.04 AS build
ARG ONEAPI_KEY
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates gnupg wget cmake ninja-build git \
 && rm -rf /var/lib/apt/lists/*
# Intel oneAPI apt repo → oneAPI 2026.x (DPC++ compiler + oneDNN headers).
RUN wget -qO- "$ONEAPI_KEY" | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-dpcpp-cpp intel-oneapi-dnnl-devel \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# Build. Source oneAPI setvars DIRECTLY + unconditionally (--force) so the full
# DPC++ toolchain — incl. llvm-foreach — is on PATH. Compile to SPIR-V (spir64 JIT)
# with NO device hint, so the build needs no ocloc/GPU; the Level-Zero runtime
# JIT-compiles for the actual Arc device on first model load (a one-time cost).
RUN bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true ; \
      set -e ; \
      cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DIE_SYCL_TARGET=spir64 -DIE_SYCL_DEVICE_HINT= ; \
      cmake --build build -j --target ie ie-perplexity'

# ===================== oneAPI runtime-lib extract =====================
# Installs the oneAPI runtime packages ONLY so the final stage can copy the few
# libs it needs out of the flat redist/lib layout. This whole stage is discarded.
FROM ubuntu:24.04 AS oneapi
ARG ONEAPI_KEY
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gnupg wget \
 && wget -qO- "$ONEAPI_KEY" | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
 && apt-get update && apt-get install -y --no-install-recommends \
      intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-dnnl intel-oneapi-runtime-tbb \
      intel-oneapi-umf \
 && rm -rf /var/lib/apt/lists/*

# ============================ runtime stage ============================
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# Minimal tools: curl + python3 for `ie pull`; gnupg + software-properties-common
# only to add the Arc graphics PPA, then purged.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl python3 gnupg software-properties-common \
 && add-apt-repository -y ppa:kobuk-team/intel-graphics \
 && apt-get update && apt-get install -y --no-install-recommends \
      libze-intel-gpu1 libze1 intel-opencl-icd \
 && apt-get purge -y --auto-remove software-properties-common gnupg \
 && rm -rf /var/lib/apt/lists/*
# The exact oneAPI libs the binary + the Level-Zero UR adapter need (see ldd
# closure) — ~250 MB instead of the 1.2 GB full runtime. libur_adapter_level_zero
# is the GPU backend; libumf is its hard dependency.
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libdnnl.so*                    /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libsycl.so*                    /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libur_loader.so*              /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libur_adapter_level_zero*.so* /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libtbb.so*                    /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libsvml.so*                   /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libimf.so*                    /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libintlc.so*                  /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libirng.so*                   /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/redist/lib/libhwloc.so*                  /opt/ie/lib/
COPY --from=oneapi /opt/intel/oneapi/umf/1.1/lib/libumf.so*                   /opt/ie/lib/
RUN echo /opt/ie/lib > /etc/ld.so.conf.d/ie.conf && ldconfig
# ie-pull download deps: curl (public GGUFs) + python3 (repo listing) already present.
COPY --from=build /src/build/src/ie               /opt/ie/bin/ie
COPY --from=build /src/build/tools/ie-perplexity  /opt/ie/bin/ie-perplexity
COPY --from=build /src/scripts/ie-pull            /opt/ie/bin/ie-pull
RUN chmod +x /opt/ie/bin/*
# LD_LIBRARY_PATH is REQUIRED: the Unified Runtime loader discovers its GPU adapter
# (libur_adapter_level_zero) via LD_LIBRARY_PATH, not ldconfig.
ENV PATH=/opt/ie/bin:$PATH \
    IE_MODELS=/models \
    LD_LIBRARY_PATH=/opt/ie/lib
VOLUME /models
EXPOSE 11435
ENTRYPOINT ["ie"]
CMD ["--help"]
