#!/bin/bash
#SBATCH --job-name=stream_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --output=stream_bench_%j.out
#SBATCH --error=stream_bench_%j.err

unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS
unset CMAKE_PREFIX_PATH PKG_CONFIG_PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniforge | grep -v conda | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v miniforge | grep -v conda | tr '\n' ':' | sed 's/:$//')
module purge

CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/12.3
CUDA_MATH_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/math_libs/12.3
SPACK_LLVM=/lfs1i3/home/uobhpc-i3/andrewmorgan.uobhpc-i3/buildit/apps/chapel/3/spack/opt/spack/linux-sles15-zen3/gcc-12.3.0/llvm-18.1.8-e7g4loavo3bglmh3ovnrfap5dinwcmw7

export CHPL_HOME=/home/uobhpc-i3/andrewmorgan.uobhpc-i3/buildit/apps/chapel/gpu/chapel-2.2.0
export CHPL_CUDA_PATH=$CUDA_PATH
export PATH=$SPACK_LLVM/bin:$CHPL_HOME/bin/linux64-x86_64:$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$SPACK_LLVM/lib:$CHPL_HOME/lib/linux64-x86_64:$LD_LIBRARY_PATH
export CPATH=$CUDA_PATH/include:$CUDA_MATH_PATH/targets/x86_64-linux/include
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CHPL_LLVM=system
export CHPL_LLVM_CONFIG=$SPACK_LLVM/bin/llvm-config
export CHPL_TARGET_COMPILER=llvm
export CHPL_LOCALE_MODEL=gpu
export CHPL_GPU=nvidia
export CHPL_GPU_ARCH=sm_80
export CHPL_GPU_MEM_STRATEGY=array_on_device
export CHPL_RE2=bundled
export CHPL_TASKS=qthreads

cd /home/uobhpc-i3/andrewmorgan.uobhpc-i3/GPU_Chapel_Cloverleaf/stream_bench

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ---- CUDA STREAM ----
echo "=== Building CUDA STREAM ==="
$CUDA_PATH/bin/nvcc -O3 -arch=sm_80 -ccbin=/usr/bin/gcc-12 -o cuda_stream cuda_stream.cu
echo "Build OK"
echo ""
echo "=== Running CUDA STREAM ==="
./cuda_stream
echo ""

# ---- Chapel STREAM ----
echo "=== Building Chapel STREAM ==="
$CHPL_HOME/bin/linux64-x86_64/chpl --fast --ldflags="-ldl" -o chapel_stream chapel_stream.chpl
echo "Build OK"
echo ""
echo "=== Running Chapel STREAM ==="
./chapel_stream
echo ""

echo "A100 HBM2e theoretical peak: ~2000 GB/s"
