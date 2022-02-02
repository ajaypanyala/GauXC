#include <math.h>
#include "../include/chebyshev_boys_computation.hpp"
#include "integral_data_types.hpp"
#include "config_obara_saika.hpp"
#include "integral_0.hpp"

#define PI 3.14159265358979323846

#define MIN(a,b)			\
  ({ __typeof__ (a) _a = (a);	        \
  __typeof__ (b) _b = (b);		\
  _a < _b ? _a : _b; })

namespace XCPU {
void integral_0(size_t npts,
               shell_pair shpair,
               double *_points,
               double *Xi,
               int ldX,
               double *Gi,
               int ldG, 
               double *weights,
               double *boys_table) {
   __attribute__((__aligned__(64))) double buffer[1 * NPTS_LOCAL + 3 * NPTS_LOCAL];

   double *temp       = (buffer + 0);
   double *Tval       = (buffer + 1 * NPTS_LOCAL + 0 * NPTS_LOCAL);
   double *Tval_inv_e = (buffer + 1 * NPTS_LOCAL + 1 * NPTS_LOCAL);
   double *FmT        = (buffer + 1 * NPTS_LOCAL + 2 * NPTS_LOCAL);

   size_t npts_upper = NPTS_LOCAL * (npts / NPTS_LOCAL);
   size_t p_outer = 0;
   for(p_outer = 0; p_outer < npts_upper; p_outer += NPTS_LOCAL) {
      double *_point_outer = (_points + p_outer);

      double xA = shpair.rA.x;
      double yA = shpair.rA.y;
      double zA = shpair.rA.z;

      for(int i = 0; i < 1 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

      for(int ij = 0; ij < shpair.nprim_pair; ++ij) {
         double RHO = shpair.prim_pairs[ij].gamma;

         double eval = shpair.prim_pairs[ij].coeff_prod * shpair.prim_pairs[ij].K;

         // Evaluate T Values
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            X_PC = SIMD_MUL(X_PC, X_PC);
            X_PC = SIMD_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SIMD_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SIMD_MUL(SIMD_DUPLICATE(&(RHO)), X_PC);
            SIMD_ALIGNED_STORE((Tval + p_inner), X_PC);
         }

         // Evaluate Boys function
         boys_elements<0>(NPTS_LOCAL, Tval, Tval_inv_e, FmT, boys_table);

         // Evaluate VRR Buffer
         for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
            SIMD_TYPE tx, t00;

            t00 = SIMD_ALIGNED_LOAD((FmT + p_inner));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t00);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
         }
      }

      for(size_t p_inner = 0; p_inner < NPTS_LOCAL; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SIMD_TYPE tx, wg, xik, gik;
         tx  = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
      }
   }

   // cleanup code
   for(; p_outer < npts; p_outer += NPTS_LOCAL) {
      size_t npts_inner = MIN((size_t) NPTS_LOCAL, npts - p_outer);
      double *_point_outer = (_points + p_outer);

      double xA = shpair.rA.x;
      double yA = shpair.rA.y;
      double zA = shpair.rA.z;

      for(int i = 0; i < 1 * NPTS_LOCAL; i += SIMD_LENGTH) SIMD_ALIGNED_STORE((temp + i), SIMD_ZERO());

      for(int ij = 0; ij < shpair.nprim_pair; ++ij) {
         double RHO = shpair.prim_pairs[ij].gamma;

         double eval = shpair.prim_pairs[ij].coeff_prod * shpair.prim_pairs[ij].K;

         // Evaluate T Values
         size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
         size_t p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE xC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 0 * npts));
            SIMD_TYPE yC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 1 * npts));
            SIMD_TYPE zC = SIMD_UNALIGNED_LOAD((_point_outer + p_inner + 2 * npts));

            SIMD_TYPE X_PC = SIMD_SUB(SIMD_DUPLICATE(&(xA)), xC);
            SIMD_TYPE Y_PC = SIMD_SUB(SIMD_DUPLICATE(&(yA)), yC);
            SIMD_TYPE Z_PC = SIMD_SUB(SIMD_DUPLICATE(&(zA)), zC);

            X_PC = SIMD_MUL(X_PC, X_PC);
            X_PC = SIMD_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SIMD_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SIMD_MUL(SIMD_DUPLICATE(&(RHO)), X_PC);
            SIMD_ALIGNED_STORE((Tval + p_inner), X_PC);
         }

         for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
            SCALAR_TYPE xC = SCALAR_LOAD((_point_outer + p_inner + 0 * npts));
            SCALAR_TYPE yC = SCALAR_LOAD((_point_outer + p_inner + 1 * npts));
            SCALAR_TYPE zC = SCALAR_LOAD((_point_outer + p_inner + 2 * npts));

            SCALAR_TYPE X_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(xA)), xC);
            SCALAR_TYPE Y_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(yA)), yC);
            SCALAR_TYPE Z_PC = SCALAR_SUB(SCALAR_DUPLICATE(&(zA)), zC);

            X_PC = SCALAR_MUL(X_PC, X_PC);
            X_PC = SCALAR_FMA(Y_PC, Y_PC, X_PC);
            X_PC = SCALAR_FMA(Z_PC, Z_PC, X_PC);
            X_PC = SCALAR_MUL(SCALAR_DUPLICATE(&(RHO)), X_PC);
            SCALAR_STORE((Tval + p_inner), X_PC);
         }

         // Evaluate Boys function
         boys_elements<0>(NPTS_LOCAL, Tval, Tval_inv_e, FmT, boys_table);

         // Evaluate VRR Buffer
         p_inner = 0;
         for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
            SIMD_TYPE tx, t00;

            t00 = SIMD_ALIGNED_LOAD((FmT + p_inner));

            t00 = SIMD_MUL(SIMD_DUPLICATE(&(eval)), t00);
            tx = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SIMD_ADD(tx, t00);
            SIMD_ALIGNED_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
         }

         for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
            SCALAR_TYPE tx, t00;

            t00 = SCALAR_LOAD((FmT + p_inner));

            t00 = SCALAR_MUL(SCALAR_DUPLICATE(&(eval)), t00);
            tx = SCALAR_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
            tx = SCALAR_ADD(tx, t00);
            SCALAR_STORE((temp + 0 * NPTS_LOCAL + p_inner), tx);
         }
      }

      size_t npts_inner_upper = SIMD_LENGTH * (npts_inner / SIMD_LENGTH);
      size_t p_inner = 0;
      for(p_inner = 0; p_inner < npts_inner_upper; p_inner += SIMD_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SIMD_TYPE tx, wg, xik, gik;
         tx  = SIMD_ALIGNED_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
         wg  = SIMD_UNALIGNED_LOAD((weights + p_outer + p_inner));

         xik = SIMD_UNALIGNED_LOAD((Xik + 0 * ldX));
         gik = SIMD_UNALIGNED_LOAD((Gik + 0 * ldG));

         tx = SIMD_MUL(tx, wg);
         gik = SIMD_FMA(tx, xik, gik);
         SIMD_UNALIGNED_STORE((Gik + 0 * ldG), gik);
      }

      for(; p_inner < npts_inner; p_inner += SCALAR_LENGTH) {
         double *Xik = (Xi + p_outer + p_inner);
         double *Gik = (Gi + p_outer + p_inner);

         SCALAR_TYPE tx, wg, xik, gik;
         tx  = SCALAR_LOAD((temp + 0 * NPTS_LOCAL + p_inner));
         wg  = SCALAR_LOAD((weights + p_outer + p_inner));

         xik = SCALAR_LOAD((Xik + 0 * ldX));
         gik = SCALAR_LOAD((Gik + 0 * ldG));

         tx = SCALAR_MUL(tx, wg);
         gik = SCALAR_FMA(tx, xik, gik);
         SCALAR_STORE((Gik + 0 * ldG), gik);
      }
   }
}
}
