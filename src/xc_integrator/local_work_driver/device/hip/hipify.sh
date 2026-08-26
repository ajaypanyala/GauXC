#!/bin/bash
#
# Regenerate the HIP kernels from their CUDA counterparts.
#
# The HIP backend is a machine translation of the CUDA backend.  This
# script MUST be re-run whenever a CUDA kernel is added or changed,
# otherwise the shared (backend-agnostic) driver in ../scheme1_base.cxx
# calls into kernels that have no HIP definition and the HIP build fails
# to link.  That is exactly what happened between the 2022 port and the
# 1.0 release: mGGA, FXC contraction, EXC gradients, sn-LinK screening
# and the shell-to-task collocation were added on the CUDA side only.
#
# The translation is performed with explicit sed rules rather than
# hipify-perl so that it is reproducible on machines without a ROCm
# installation, and so that the handful of NON-mechanical decisions are
# documented in one place:
#
#   * warp/wavefront size is NOT hardcoded anywhere; kernels take it from
#     GauXC::cuda::warp_size -> GauXC::hip::warp_size (32 vs 64), so the
#     launch geometry adapts.  Kernels that assume a 32-wide reduction
#     must be reviewed by hand -- see CHECK_WAVEFRONT below.
#   * __syncwarp() has no HIP equivalent; wavefronts execute in lockstep
#     on AMD, so it is commented out (the convention already used by the
#     2022 port in this directory).
#   * __shfl_*_sync(mask, ...) -> __shfl_*(...): HIP shuffles take no
#     mask argument.
#   * cub -> hipcub.
#   * nvcc implicitly converts a __global__ function pointer to
#     const void* (used to pass a kernel to cudaFuncSetAttribute);
#     clang/HIP does not, so the call needs an explicit
#     reinterpret_cast<const void*>(...).
#   * symmetrize_mat.cu's block-transpose kernels use a SQUARE
#     block_size x block_size shared tile and launch a matching square
#     block_size x block_size thread block, hardcoding
#     block_size == warp_size == max_warps_per_thread_block (true on
#     CUDA: 32 == 32 == 32). On HIP that triple equality is impossible
#     (warp_size 64 forces max_warps_per_thread_block down to 16, and
#     64x64 would exceed the 1024 threads/block hardware limit anyway),
#     so block_size is decoupled to a fixed 32 for HIP -- independent of
#     hip::warp_size, still square, still <=1024 threads/block and well
#     under the LDS budget.
#   * uvvars_gga.hpp/uvvars_mgga.hpp size a __shared__ tile as
#     [rows][warp_size][SM_BLOCK+1]; warp_size must stay equal to the
#     true warp/wavefront width (it indexes threadIdx.x directly in a
#     warp-synchronous reduction), but at hip::warp_size==64 the CUDA
#     tuning of SM_BLOCK==32 overflows AMD's 64KB LDS limit (rows==4:
#     4*64*33*8 = 67584B > 65536B). SM_BLOCK is otherwise just a
#     performance tile width, so it is halved to 16 for HIP only
#     (4*64*17*8 = 34816B); this trades tiling granularity for fitting
#     in LDS, with no correctness impact.
#   * <cuda.h> (the CUDA driver API header) is dropped rather than mapped
#     to a HIP equivalent; nothing in the translated files uses driver-API
#     symbols, and hip/hip_runtime.h (inserted below) already covers the
#     runtime-API surface these files actually rely on.
#   * CUTLASS has no HIP counterpart; cutlass_wrapper is CUDA-only and is
#     deliberately not translated (it is guarded by GAUXC_ENABLE_CUTLASS,
#     which is a CUDA-only dependent option in the top-level CMakeLists).
#   * `register` is accepted (as a no-op) by nvcc but is ISO-illegal in
#     C++17, which the HIP/clang compiler enforces; the keyword is
#     stripped rather than translated.
#   * The shell-to-task collocation kernels load one primitive
#     coefficient/exponent per lane, in lockstep, on the assumption that
#     shell_nprim_max (32, a Shell data-layout constant, see shell.hpp)
#     equals the warp width -- true for cuda::warp_size (32) but not for
#     hip::warp_size (64, an AMD wavefront). The static_assert is
#     loosened to <= and the two loads are guarded so lanes >=
#     shell_nprim_max (idle on AMD) don't read/write out of bounds.
#
# Usage:  ./hipify.sh          (from this directory)

set -euo pipefail

CUDA_PREFIX=$PWD/../cuda/kernels
HIP_PREFIX=$PWD/kernels

mkdir -p "$HIP_PREFIX/collocation"

hipify_file() {
  local src=$1 dst=$2
  sed \
    -e 's|device_specific/cuda_util\.hpp|device_specific/hip_util.hpp|g' \
    -e 's|device_specific/cuda_device_constants\.hpp|device_specific/hip_device_constants.hpp|g' \
    -e 's|device_specific/cublas_util\.hpp|device_specific/hipblas_util.hpp|g' \
    -e 's|cuda_extensions\.hpp|hip_extensions.hpp|g' \
    -e 's|cuda_aos_scheme1\.hpp|hip_aos_scheme1.hpp|g' \
    -e 's|cuda_ssf_1d\.hpp|hip_ssf_1d.hpp|g' \
    -e 's|\bregister[[:space:]]\+||g' \
    -e 's|#include <cub/\(.*\)\.cuh>|#include <hipcub/\1.hpp>|g' \
    -e '/^#include <cuda\.h>$/d' \
    -e 's|\bcub::|hipcub::|g' \
    -e 's|\bcuda::|hip::|g' \
    -e 's|\bnamespace cuda\b|namespace hip|g' \
    -e 's|\bcudaStream_t\b|hipStream_t|g' \
    -e 's|\bcuda_stream\b|hip_stream|g' \
    -e 's|\bcudaError_t\b|hipError_t|g' \
    -e 's|\bcudaSuccess\b|hipSuccess|g' \
    -e 's|\bcudaGetErrorString\b|hipGetErrorString|g' \
    -e 's|\bcudaGetLastError\b|hipGetLastError|g' \
    -e 's|\bcudaDeviceSynchronize\b|hipDeviceSynchronize|g' \
    -e 's|\bcudaMalloc\b|hipMalloc|g' \
    -e 's|\bcudaFree\b|hipFree|g' \
    -e 's|\bcudaMemcpy|hipMemcpy|g' \
    -e 's|\bcudaMemset|hipMemset|g' \
    -e 's|device/cuda/kernels|device/hip/kernels|g' \
    -e 's|util::cuda_|util::hip_|g' \
    -e 's|\bcuda_kernel_max_threads_per_block\b|hip_kernel_max_threads_per_block|g' \
    -e 's|\bcudaDeviceGetAttribute\b|hipDeviceGetAttribute|g' \
    -e 's|\bcudaDevAttrMaxSharedMemoryPerBlockOptin\b|hipDeviceAttributeMaxSharedMemoryPerBlock|g' \
    -e 's|\bcudaDevAttrMaxSharedMemoryPerBlock\b|hipDeviceAttributeMaxSharedMemoryPerBlock|g' \
    -e 's|\bcudaFuncSetAttribute\b|hipFuncSetAttribute|g' \
    -e 's|hipFuncSetAttribute(&\([A-Za-z_][A-Za-z0-9_]*\),|hipFuncSetAttribute(reinterpret_cast<const void*>(\&\1),|g' \
    -e 's|\bcudaFuncAttribute|hipFuncAttribute|g' \
    -e 's|\bcuda_exception\b|hip_exception|g' \
    -e 's|GAUXC_CUDA|GAUXC_HIP|g' \
    -e 's|GAUXC_CUBLAS|GAUXC_HIPBLAS|g' \
    -e 's|cuda_exception\.hpp|hip_exception.hpp|g' \
    -e 's|\bCUDA_|HIP_|g' \
    -e 's|\bCUBLAS_|HIPBLAS_|g' \
    -e 's|\bcublas|hipblas|g' \
    -e 's|__shfl_\([a-z]*\)_sync *( *[^,]*, *|__shfl_\1(|g' \
    -e 's|\(^[[:space:]]*\)__syncwarp();|\1// __syncwarp();  // lockstep wavefronts on AMD|g' \
    -e 's|static_assert( detail::shell_nprim_max == hip::warp_size );|static_assert( detail::shell_nprim_max <= hip::warp_size );|g' \
    -e 's|my_alpha\[warp_rank\] = alpha_gm\[warp_rank\];|if( warp_rank < detail::shell_nprim_max ) my_alpha[warp_rank] = alpha_gm[warp_rank];|g' \
    -e 's|my_coeff\[warp_rank\] = coeff_gm\[warp_rank\];|if( warp_rank < detail::shell_nprim_max ) my_coeff[warp_rank] = coeff_gm[warp_rank];|g' \
    -e 's|#define VVAR_KERNEL_SM_BLOCK 32|#define VVAR_KERNEL_SM_BLOCK 16|g' \
    -e 's|#define MGGA_KERNEL_SM_BLOCK 32|#define MGGA_KERNEL_SM_BLOCK 16|g' \
    -e 's|constexpr uint32_t block_size = hip::warp_size;|constexpr uint32_t block_size = 32; // fixed, independent of hip::warp_size -- see hipify.sh|g' \
    -e 's|// Warp size must equal max_warps_per_thread_block must equal 32|// block_size is fixed at 32 for HIP, independent of hip::warp_size -- see hipify.sh|g' \
    -e 's|dim3 threads(hip::warp_size, hip::max_warps_per_thread_block), blocks(num_blocks);|dim3 threads(32, 32), blocks(num_blocks);|g' \
    -e 's|const size_t num_blocks = ((N + hip::warp_size - 1) / hip::warp_size);|const size_t num_blocks = ((N + 32 - 1) / 32);|g' \
    "$src" > "$dst"

  # CUDA cache-hint stores have no HIP counterpart: __stcs() is an NVIDIA
  # intrinsic and the pre-CUDA-11 fallback is inline PTX.  CUDART_VERSION
  # is undefined under HIP, so the preprocessor would otherwise select the
  # PTX branch and fail to compile.  Keep the intrinsic branch and lower
  # the store to a plain one (the hint is an optimization, not semantics).
  if grep -q 'CUDART_VERSION' "$dst"; then
    awk '
      /^#if \(CUDART_VERSION/ { skipelse = 1; next }
      /^#else/ && skipelse    { drop = 1; next }
      /^#endif/ && skipelse   { skipelse = 0; drop = 0; next }
      drop                    { next }
      { print }
    ' "$dst" > "$dst.tmp"
    sed -e 's|__stcs( *\([^,]*\), *\([^)]*\));|*(\1) = \2;|g' \
        "$dst.tmp" > "$dst"
    rm -f "$dst.tmp"
  fi

  # bitvector_to_position_list_{shellpair,shells} in
  # exx_ek_screening_bfn_stats.cu declare a local "warp_size" that is
  # really just the (hardcoded, architecture-independent) 32x32 launch
  # geometry from dim3 threads(32,32) at the call site -- the name is a
  # coincidence of CUDA's warp being 32 wide, not a dependence on the
  # true warp/wavefront width (unlike e.g. the warp_reduce_* callers in
  # uvvars_*.hpp, which this pattern does not match). Sizing its
  # collisions_buffer[warp_size][warp_size][...] off hip::warp_size (64)
  # quadruples the tile and overflows AMD's 64KB LDS; pin it back to the
  # actual launch geometry instead.
  if grep -q 'collisions_buffer\[warp_size\]\[warp_size\]\[buffer_size\]' "$dst"; then
    sed -i -e 's|constexpr auto warp_size = hip::warp_size;|constexpr auto warp_size = 32; // matches dim3 threads(32,32) below, independent of hip::warp_size -- see hipify.sh|g' \
        "$dst"
  fi

  # HIP needs its runtime header; insert before the first #include (i.e.
  # after the license comment block).
  if ! grep -q 'hip/hip_runtime.h' "$dst"; then
    awk '
      BEGIN { done = 0 }
      /^#include/ && !done { print "#include \"hip/hip_runtime.h\""; done = 1 }
      { print }
      END { if (!done) print "#include \"hip/hip_runtime.h\"" }
    ' "$dst" > "$dst.tmp"
    mv "$dst.tmp" "$dst"
  fi
}

# ---- collocation ------------------------------------------------------
# Every header directly under collocation/ (per-l cartesian/spherical
# kernels for values/gradient/hessian/laplacian/lapgrad, plus the shared
# angular/radial/constants headers); deprecated/, scripts/ and templates/
# are CUDA-generator-only and are not part of the build.
for f in "$CUDA_PREFIX"/collocation/*.hpp; do
  hipify_file "$f" "$HIP_PREFIX/collocation/$(basename "$f")"
done

hipify_file "$CUDA_PREFIX/collocation_masked_combined_kernels.hpp" \
            "$HIP_PREFIX/collocation_masked_combined_kernels.hpp"
hipify_file "$CUDA_PREFIX/collocation_masked_kernels.hpp" \
            "$HIP_PREFIX/collocation_masked_kernels.hpp"
hipify_file "$CUDA_PREFIX/collocation_shell_to_task_kernels.hpp" \
            "$HIP_PREFIX/collocation_shell_to_task_kernels.hpp"
hipify_file "$CUDA_PREFIX/collocation_device.cu" \
            "$HIP_PREFIX/collocation_device.hip"

# ---- weights ----------------------------------------------------------
hipify_file "$CUDA_PREFIX/grid_to_center.cu"  "$HIP_PREFIX/grid_to_center.hip"
hipify_file "$CUDA_PREFIX/grid_to_center.hpp" "$HIP_PREFIX/grid_to_center.hpp"
hipify_file "$CUDA_PREFIX/cuda_ssf_1d.cu"     "$HIP_PREFIX/hip_ssf_1d.hip"
hipify_file "$CUDA_PREFIX/cuda_ssf_1d.hpp"    "$HIP_PREFIX/hip_ssf_1d.hpp"

# ---- BLAS extensions --------------------------------------------------
hipify_file "$CUDA_PREFIX/cublas_extensions.cu" "$HIP_PREFIX/hipblas_extensions.hip"
hipify_file "$CUDA_PREFIX/cuda_extensions.hpp"  "$HIP_PREFIX/hip_extensions.hpp"

# ---- density / potential / XC assembly --------------------------------
hipify_file "$CUDA_PREFIX/uvvars.cu"          "$HIP_PREFIX/uvvars.hip"
hipify_file "$CUDA_PREFIX/uvvars_lda.hpp"     "$HIP_PREFIX/uvvars_lda.hpp"
hipify_file "$CUDA_PREFIX/uvvars_gga.hpp"     "$HIP_PREFIX/uvvars_gga.hpp"
hipify_file "$CUDA_PREFIX/uvvars_mgga.hpp"    "$HIP_PREFIX/uvvars_mgga.hpp"
hipify_file "$CUDA_PREFIX/zmat_vxc.cu"        "$HIP_PREFIX/zmat_vxc.hip"
hipify_file "$CUDA_PREFIX/zmat_fxc.cu"        "$HIP_PREFIX/zmat_fxc.hip"
hipify_file "$CUDA_PREFIX/pack_submat.cu"     "$HIP_PREFIX/pack_submat.hip"
hipify_file "$CUDA_PREFIX/symmetrize_mat.cu"  "$HIP_PREFIX/symmetrize_mat.hip"
hipify_file "$CUDA_PREFIX/cuda_inc_potential.cu" "$HIP_PREFIX/hip_inc_potential.hip"

# ---- gradients and sn-LinK screening ----------------------------------
hipify_file "$CUDA_PREFIX/increment_exc_grad.cu" "$HIP_PREFIX/increment_exc_grad.hip"
hipify_file "$CUDA_PREFIX/exx_ek_screening_bfn_stats.cu" \
            "$HIP_PREFIX/exx_ek_screening_bfn_stats.hip"

echo "hipify: regenerated $(ls "$HIP_PREFIX"/*.hip "$HIP_PREFIX"/*.hpp | wc -l) files"
echo
echo "CHECK_WAVEFRONT: kernels performing intra-warp reductions were written"
echo "against a 32-lane warp.  On AMD the wavefront is 64 lanes and"
echo "GauXC::hip::warp_size reflects that, but any reduction whose trip"
echo "count is written as a literal must be reviewed.  Grep for '16;' '8;'"
echo "'4;' '2;' '1;' shuffle ladders in the generated files before trusting"
echo "numerical results on AMD hardware."
