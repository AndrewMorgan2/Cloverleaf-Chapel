# Chapel GPU CloverLeaf Makefile
# NOTE: GPU targets must be compiled on a compute node with GPU access.
#       Use 'make cpu' for local CPU-only builds.

# ── GPU build settings (cluster) ────────────────────────────────────────────
# Chapel GPU installation
CHAPEL_GPU_HOME = /home/uobhpc-i3/andrewmorgan.uobhpc-i3/buildit/apps/chapel/gpu/chapel-2.2.0

# CUDA location (from NVIDIA HPC SDK 24.5)
CUDA_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
CUDA_MATH_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4

# GCC 12.3 from module
GCC_PATH = /opt/cray/pe/gcc-native/12/bin

SHELL := /bin/bash

CHPL = $(CHAPEL_GPU_HOME)/bin/linux64-x86_64/chpl

# GPU thread block size — override with: make BLOCKSIZE=128
BLOCKSIZE ?= 32

# Compilation flags - add CUDA include paths for curand headers
CHPL_FLAGS = --fast \
	-I$(CUDA_PATH)/include \
	-I$(CUDA_MATH_PATH)/targets/x86_64-linux/include \
	--ldflags="-ldl" \
	--set Definitions.gpuBlockSize=$(BLOCKSIZE)

TARGET = cloverleaf_gpu
SOURCES = $(wildcard src/*.chpl)

# Environment for GPU Chapel - need to completely override conda
define CHAPEL_ENV
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE; \
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CMAKE_PREFIX_PATH PKG_CONFIG_PATH; \
export PATH=$(GCC_PATH):/usr/local/bin:/usr/bin:/bin:$(CHAPEL_GPU_HOME)/bin/linux64-x86_64:$(CUDA_PATH)/bin; \
export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUDA_MATH_PATH)/lib64:$(CHAPEL_GPU_HOME)/lib/linux64-x86_64; \
export CPATH=$(CUDA_PATH)/include:$(CUDA_MATH_PATH)/targets/x86_64-linux/include; \
export CC=$(GCC_PATH)/gcc; \
export CXX=$(GCC_PATH)/g++; \
export CHPL_HOME=$(CHAPEL_GPU_HOME); \
export CHPL_LLVM=bundled; \
export CHPL_CUDA_PATH=$(CUDA_PATH); \
export CHPL_TARGET_COMPILER=llvm; \
export CHPL_LOCALE_MODEL=gpu; \
export CHPL_GPU=nvidia; \
export CHPL_GPU_ARCH=sm_80; \
export CHPL_GPU_MEM_STRATEGY=array_on_device; \
export CHPL_RE2=bundled; \
export CHPL_HWLOC_PCI=enable; \
export CHPL_TASKS=qthreads
endef

# ── CPU (local) build settings ───────────────────────────────────────────────
# Override LOCAL_CHPL to point at your local Chapel compiler if it is not in PATH:
#   make cpu LOCAL_CHPL=/path/to/chpl
LOCAL_CHPL ?= chpl

CPU_TARGET = cloverleaf_cpu

# --fast: enables -O3, fast-math, bounds/nil/overflow checks removed.
# Use CPU_FLAGS="-g" to get a debug build that surfaces errors.
CPU_FLAGS = --fast

# Strip conda and other env pollution, but keep the system PATH untouched.
define CPU_ENV
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE; \
unset CHPL_LOCALE_MODEL CHPL_GPU CHPL_GPU_ARCH CHPL_GPU_MEM_STRATEGY \
      CHPL_CUDA_PATH CHPL_TARGET_COMPILER
endef

.PHONY: all cpu clean debug help

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "Building GPU-enabled CloverLeaf..."
	@echo "Using Chapel at: $(CHPL)"
	@echo "Using CUDA at: $(CUDA_PATH)"
	@echo "Using CUDA Math at: $(CUDA_MATH_PATH)"
	@echo "Using GCC at: $(GCC_PATH)"
	@$(CHAPEL_ENV); \
	$(CHPL) $(CHPL_FLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"

cpu: $(SOURCES)
	@echo "Building CPU-only CloverLeaf..."
	@echo "Using Chapel at: $(LOCAL_CHPL)"
	@$(CPU_ENV); \
	$(LOCAL_CHPL) $(CPU_FLAGS) -o $(CPU_TARGET) $^
	@echo "Build complete: $(CPU_TARGET)"
	@echo ""
	@echo "Run with:"
	@echo "  ./$(CPU_TARGET) --inputDeck=config/clover_bm2_short.in"
	@echo "  ./$(CPU_TARGET) --inputDeck=config/clover_bm2_short.in --useGpu=false"

debug: CHPL_FLAGS = -g -I$(CUDA_PATH)/include -I$(CUDA_MATH_PATH)/targets/x86_64-linux/include
debug: $(TARGET)

clean:
	rm -f $(TARGET) $(CPU_TARGET)
	rm -rf tmp_c

help:
	@echo "Chapel CloverLeaf Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build optimized GPU binary (requires GPU compute node)"
	@echo "  cpu    - Build CPU-only binary for local development"
	@echo "  debug  - Build GPU binary with debug symbols"
	@echo "  clean  - Remove built files"
	@echo ""
	@echo "CPU build options:"
	@echo "  LOCAL_CHPL=/path/to/chpl  - Path to local Chapel compiler (default: chpl from PATH)"
	@echo ""
	@echo "Run CPU binary:"
	@echo "  ./cloverleaf_cpu --inputDeck=config/clover_bm2_short.in --useGpu=false"
	@echo ""
	@echo "  # Recommended for EPYC 7713 (pin to one NUMA node = 16 cores):"
	@echo "  CHPL_RT_NUM_THREADS_PER_LOCALE=16 numactl --cpunodebind=0 --membind=0 \\"
	@echo "    ./cloverleaf_cpu --inputDeck=config/clover_bm256.in --useGpu=false --reportFreq=10 --profile=false"
	@echo ""
	@echo "  # Full socket (64 cores, 4 NUMA nodes 0-3):"
	@echo "  CHPL_RT_NUM_THREADS_PER_LOCALE=64 numactl --cpunodebind=0-3 --membind=0-3 \\"
	@echo "    ./cloverleaf_cpu --inputDeck=config/clover_bm256.in --useGpu=false --profile=false"
	@echo ""
	@echo "Run GPU binary (on cluster):"
	@echo "  ./cloverleaf_gpu --inputDeck=config/clover_bm.in --numGpus=1"
