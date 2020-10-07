#include "hip/hip_runtime.h"
#include <iostream>
#include <cassert>

#include <gauxc/shell.hpp>
#include <gauxc/xc_task.hpp>

#include "collocation_angular_cartesian.hpp"
#include "collocation_angular_spherical_unnorm.hpp"

namespace GauXC      {
namespace integrator {
namespace hip       {

using namespace GauXC::hip;

template <typename T>
__global__
void collocation_device_petite_combined_kernel(
  size_t           ntasks,
  XCTaskDevice<T>* device_tasks
) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const int batch_id = blockIdx.z;

  if( batch_id < ntasks ) {

    auto& task = device_tasks[ batch_id ];
  
    const auto nshells         = task.nshells;
    const auto nbf             = task.nbe;
    const auto npts            = task.npts;
    const auto*  shells_device = task.shells;
    const auto*  pts_device    = task.points;
    const auto*  offs_device   = task.shell_offs;

    auto* eval_device    = task.bf;

  if( tid_x < npts and tid_y < nshells ) {

    const size_t ipt = tid_x;
    const size_t ish = tid_y;

    const size_t ibf = offs_device[ish];

    const auto& shell = shells_device[ish];
    const auto* pt    = pts_device + 3*ipt;
  

    const auto* O     = shell.O_data();
    const auto* alpha = shell.alpha_data();
    const auto* coeff = shell.coeff_data();

    const auto xc = pt[0] - O[0];
    const auto yc = pt[1] - O[1];
    const auto zc = pt[2] - O[2];
  
    const auto rsq = xc*xc + yc*yc + zc*zc;
  
    const size_t nprim = shell.nprim(); 
    auto tmp = 0.;
    for( size_t i = 0; i < nprim; ++i )
      tmp += coeff[i] * std::exp( - alpha[i] * rsq );

    auto * bf_eval = eval_device + ibf + ipt*nbf;

    const bool do_sph = shell.pure();
    if( do_sph )
      collocation_spherical_unnorm_angular( shell.l(), tmp, xc, yc, zc, bf_eval );
    else
      collocation_cartesian_angular( shell.l(), tmp, xc, yc, zc, bf_eval );

  } // shell / point idx check

  } // Batch idx check

}















template <typename T>
__global__
void collocation_device_petite_combined_kernel_deriv1(
  size_t           ntasks,
  XCTaskDevice<T>* device_tasks
) {

  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  const int batch_id = blockIdx.z;

  if( batch_id < ntasks ) {

    auto& task = device_tasks[ batch_id ];
  
    const auto nshells         = task.nshells;
    const auto nbf             = task.nbe;
    const auto npts            = task.npts;
    const auto*  shells_device = task.shells;
    const auto*  pts_device    = task.points;
    const auto*  offs_device   = task.shell_offs;

    auto* eval_device    = task.bf;
    auto* deval_device_x = task.dbfx;
    auto* deval_device_y = task.dbfy;
    auto* deval_device_z = task.dbfz;


  if( tid_x < npts and tid_y < nshells ) {

    const size_t ipt = tid_x;
    const size_t ish = tid_y;

    const size_t ibf = offs_device[ish];

    const auto& shell = shells_device[ish];
    const auto* pt    = pts_device + 3*ipt;
  

    const auto* O     = shell.O_data();
    const auto* alpha = shell.alpha_data();
    const auto* coeff = shell.coeff_data();

    const auto xc = pt[0] - O[0];
    const auto yc = pt[1] - O[1];
    const auto zc = pt[2] - O[2];
  
    const auto rsq = xc*xc + yc*yc + zc*zc;
  
    const size_t nprim = shell.nprim(); 
    auto tmp = 0.;
    auto tmp_x = 0., tmp_y = 0., tmp_z = 0.;
    for( size_t i = 0; i < nprim; ++i ) {

      const auto a = alpha[i];
      const auto e = coeff[i] * std::exp( - a * rsq );

      const auto ae = 2. * a * e;

      tmp   += e;
      tmp_x -= ae * xc;
      tmp_y -= ae * yc;
      tmp_z -= ae * zc;

    }

    auto * bf_eval = eval_device    + ibf + ipt*nbf;
    auto * dx_eval = deval_device_x + ibf + ipt*nbf;
    auto * dy_eval = deval_device_y + ibf + ipt*nbf;
    auto * dz_eval = deval_device_z + ibf + ipt*nbf;

    const bool do_sph = shell.pure();
    if( do_sph ) 
      collocation_spherical_unnorm_angular_deriv1( shell.l(), tmp, tmp_x, tmp_y, 
                                               tmp_z, xc, yc, zc, bf_eval, dx_eval, 
                                               dy_eval, dz_eval );
    else
      collocation_cartesian_angular_deriv1( shell.l(), tmp, tmp_x, tmp_y, tmp_z, 
                                        xc, yc, zc, bf_eval, dx_eval, 
                                        dy_eval, dz_eval );

  } // shell / point idx check

  } // Batch idx check


}

} // namespace hip
} // namespace integrator
} // namespace GauXC