#include "cublas_extensions.hpp"
#include <gauxc/util/cublas_util.hpp>
#include <gauxc/util/div_ceil.hpp>
#include <gauxc/exceptions/cublas_exception.hpp>

namespace GauXC {
namespace cuda  {
namespace blas  {

template <typename T>
__global__ void increment_kernel( const T* X, T* Y ) {
  const auto tid = blockIdx.x;
  if( tid < 1 ) (*Y) += (*X);
}

template <typename T>
void increment( const T* X, T* Y, cudaStream_t stream ) {
  increment_kernel<<<1,1,0,stream>>>(X,Y);
}

template <>
void dot( cublasHandle_t handle,
          int            N,
          const double*  X,
          int            INCX,
          const double*  Y,
          int            INCY,
          double*        RES ) {

  auto stat = cublasDdot( handle, N, X, INCX, Y, INCY, RES );
  GAUXC_CUBLAS_ERROR("CUBLAS DDOT FAILED", stat );

}

template <typename T>
void gdot( cublasHandle_t handle,
           int       N,
           const T*  X,
           int       INCX,
           const T*  Y,
           int       INCY,
           T*        SCR,
           T*        RES ) {

  dot( handle, N, X, INCX, Y, INCY, SCR );
  auto stream = util::get_stream(handle);
  increment( SCR, RES, stream );

}

template 
void gdot( cublasHandle_t handle,
           int            N,
           const double*  X,
           int            INCX,
           const double*  Y,
           int            INCY,
           double*        SCR,
           double*        RES );










template <typename T>
void __global__ hadamard_product_kernel( int      M,
                                         int      N,
                                         const T* A,
                                         int      LDA,
                                         T*       B,
                                         int      LDB ) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if( tid_x < M and tid_y < N ) {
    B[ tid_x + tid_y*LDB ] *= A[ tid_x + tid_y*LDA ];
  }

}



template <typename T>
void hadamard_product( cublasHandle_t handle,
                       int            M,
                       int            N,
                       const T*       A,
                       int            LDA,
                       T*             B,
                       int            LDB ) {

  auto stream = util::get_stream(handle);
  dim3 threads( 32, 32 );
  dim3 blocks( util::div_ceil( M, threads.x ),
               util::div_ceil( N, threads.y ) );

  hadamard_product_kernel<<< blocks, threads, 0, stream >>>( M, N, A, LDA, B, LDB );

}
 
template 
void hadamard_product( cublasHandle_t handle,
                       int            M,
                       int            N,
                       const double*  A,
                       int            LDA,
                       double*        B,
                       int            LDB ); 





}
}
}

