#include <math.h>
#include "boys_computation.h"
#include "integral_data_types.h"

#define PI 3.14159265358979323846

void integral_3_0(int npts,
                  shells shellA,
                  shells shellB,
                  point *_points,
                  double *Xi,
                  double *Xj,
                  int stX,
                  int ldX,
                  double *Gi,
                  double *Gj,
                  int stG, 
                  int ldG, 
                  double *weights) {
   double temp[10];

   for(int point_idx = 0; point_idx < npts; ++point_idx) {
      point C = *(_points + point_idx);

      double xA = shellA.origin.x;
      double yA = shellA.origin.y;
      double zA = shellA.origin.z;

      double xB = shellB.origin.x;
      double yB = shellB.origin.y;
      double zB = shellB.origin.z;

      double X_AB = (xA - xB);
      double Y_AB = (yA - yB);
      double Z_AB = (zA - zB);

      double beta_in = 0.0;
      for(int i = 0; i < shellA.m; ++i) {
         for(int j = 0; j < shellB.m; ++j) {
            double aA = shellA.coeff[i].alpha;
            double cA = shellA.coeff[i].coeff;

            double aB = shellB.coeff[j].alpha;
            double cB = shellB.coeff[j].coeff;

            double RHO = aA + aB;
            double RHO_INV = 1.0 / RHO;

            double xC = C.x;
            double yC = C.y;
            double zC = C.z;

            double xP = (aA * xA + aB * xB) * RHO_INV;
            double yP = (aA * yA + aB * yB) * RHO_INV;
            double zP = (aA * zA + aB * zB) * RHO_INV;

            double X_PA = (xP - xA);
            double Y_PA = (yP - yA);
            double Z_PA = (zP - zA);

            double X_PC = (xP - xC);
            double Y_PC = (yP - yC);
            double Z_PC = (zP - zC);

            double t00, t01, t02, t03, t10, t11, t12, t20, t21, t30;

            double eval = cA * cB * 2 * PI * RHO_INV * exp(-1.0 * (X_AB * X_AB + Y_AB * Y_AB + Z_AB * Z_AB) * aA * aB * RHO_INV);
            double tval = RHO * (X_PC * X_PC + Y_PC * Y_PC + Z_PC * Z_PC);

            t00 = eval * boys_function(0, tval);
            t01 = eval * boys_function(1, tval);
            t02 = eval * boys_function(2, tval);
            t03 = eval * boys_function(3, tval);
            t10 = X_PA * t00 - X_PC * t01;
            t11 = X_PA * t01 - X_PC * t02;
            t12 = X_PA * t02 - X_PC * t03;
            t20 = X_PA * t10 - X_PC * t11 + 0.5 * RHO_INV * 1 * (t00 - t01);
            t21 = X_PA * t11 - X_PC * t12 + 0.5 * RHO_INV * 1 * (t01 - t02);
            t30 = X_PA * t20 - X_PC * t21 + 0.5 * RHO_INV * 2 * (t10 - t11);
            *(temp + 0) = beta_in * (*(temp + 0)) + t30;

            t30 = Y_PA * t20 - Y_PC * t21;
            *(temp + 1) = beta_in * (*(temp + 1)) + t30;

            t30 = Z_PA * t20 - Z_PC * t21;
            *(temp + 2) = beta_in * (*(temp + 2)) + t30;

            t20 = Y_PA * t10 - Y_PC * t11;
            t21 = Y_PA * t11 - Y_PC * t12;
            t30 = Y_PA * t20 - Y_PC * t21 + 0.5 * RHO_INV * 1 * (t10 - t11);
            *(temp + 3) = beta_in * (*(temp + 3)) + t30;

            t30 = Z_PA * t20 - Z_PC * t21;
            *(temp + 4) = beta_in * (*(temp + 4)) + t30;

            t20 = Z_PA * t10 - Z_PC * t11;
            t21 = Z_PA * t11 - Z_PC * t12;
            t30 = Z_PA * t20 - Z_PC * t21 + 0.5 * RHO_INV * 1 * (t10 - t11);
            *(temp + 5) = beta_in * (*(temp + 5)) + t30;

            t10 = Y_PA * t00 - Y_PC * t01;
            t11 = Y_PA * t01 - Y_PC * t02;
            t12 = Y_PA * t02 - Y_PC * t03;
            t20 = Y_PA * t10 - Y_PC * t11 + 0.5 * RHO_INV * 1 * (t00 - t01);
            t21 = Y_PA * t11 - Y_PC * t12 + 0.5 * RHO_INV * 1 * (t01 - t02);
            t30 = Y_PA * t20 - Y_PC * t21 + 0.5 * RHO_INV * 2 * (t10 - t11);
            *(temp + 6) = beta_in * (*(temp + 6)) + t30;

            t30 = Z_PA * t20 - Z_PC * t21;
            *(temp + 7) = beta_in * (*(temp + 7)) + t30;

            t20 = Z_PA * t10 - Z_PC * t11;
            t21 = Z_PA * t11 - Z_PC * t12;
            t30 = Z_PA * t20 - Z_PC * t21 + 0.5 * RHO_INV * 1 * (t10 - t11);
            *(temp + 8) = beta_in * (*(temp + 8)) + t30;

            t10 = Z_PA * t00 - Z_PC * t01;
            t11 = Z_PA * t01 - Z_PC * t02;
            t12 = Z_PA * t02 - Z_PC * t03;
            t20 = Z_PA * t10 - Z_PC * t11 + 0.5 * RHO_INV * 1 * (t00 - t01);
            t21 = Z_PA * t11 - Z_PC * t12 + 0.5 * RHO_INV * 1 * (t01 - t02);
            t30 = Z_PA * t20 - Z_PC * t21 + 0.5 * RHO_INV * 2 * (t10 - t11);
            *(temp + 9) = beta_in * (*(temp + 9)) + t30;

            beta_in = 1.0;
         }
      }

      double *Xik = (Xi + point_idx * stX);
      double *Xjk = (Xj + point_idx * stX);
      double *Gik = (Gi + point_idx * stG);
      double *Gjk = (Gj + point_idx * stG);

      double const_value, X_ABp, Y_ABp, Z_ABp, comb_m_i, comb_n_j, comb_p_k, rcp_i, rcp_j, rcp_k;
      double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;

      X_ABp = 1.0; comb_m_i = 1.0;
      Y_ABp = 1.0; comb_n_j = 1.0;
      Z_ABp = 1.0; comb_p_k = 1.0;
      const_value = comb_m_i * comb_n_j * comb_p_k * X_ABp * Y_ABp * Z_ABp;

      t0 = *(temp + 0) * const_value * (*(weights + point_idx));
      *(Gik + 0 * ldG) += *(Xjk + 0 * ldX) * t0;
      *(Gjk + 0 * ldG) += *(Xik + 0 * ldX) * t0;
      t1 = *(temp + 1) * const_value * (*(weights + point_idx));
      *(Gik + 1 * ldG) += *(Xjk + 0 * ldX) * t1;
      *(Gjk + 0 * ldG) += *(Xik + 1 * ldX) * t1;
      t2 = *(temp + 2) * const_value * (*(weights + point_idx));
      *(Gik + 2 * ldG) += *(Xjk + 0 * ldX) * t2;
      *(Gjk + 0 * ldG) += *(Xik + 2 * ldX) * t2;
      t3 = *(temp + 3) * const_value * (*(weights + point_idx));
      *(Gik + 3 * ldG) += *(Xjk + 0 * ldX) * t3;
      *(Gjk + 0 * ldG) += *(Xik + 3 * ldX) * t3;
      t4 = *(temp + 4) * const_value * (*(weights + point_idx));
      *(Gik + 4 * ldG) += *(Xjk + 0 * ldX) * t4;
      *(Gjk + 0 * ldG) += *(Xik + 4 * ldX) * t4;
      t5 = *(temp + 5) * const_value * (*(weights + point_idx));
      *(Gik + 5 * ldG) += *(Xjk + 0 * ldX) * t5;
      *(Gjk + 0 * ldG) += *(Xik + 5 * ldX) * t5;
      t6 = *(temp + 6) * const_value * (*(weights + point_idx));
      *(Gik + 6 * ldG) += *(Xjk + 0 * ldX) * t6;
      *(Gjk + 0 * ldG) += *(Xik + 6 * ldX) * t6;
      t7 = *(temp + 7) * const_value * (*(weights + point_idx));
      *(Gik + 7 * ldG) += *(Xjk + 0 * ldX) * t7;
      *(Gjk + 0 * ldG) += *(Xik + 7 * ldX) * t7;
      t8 = *(temp + 8) * const_value * (*(weights + point_idx));
      *(Gik + 8 * ldG) += *(Xjk + 0 * ldX) * t8;
      *(Gjk + 0 * ldG) += *(Xik + 8 * ldX) * t8;
      t9 = *(temp + 9) * const_value * (*(weights + point_idx));
      *(Gik + 9 * ldG) += *(Xjk + 0 * ldX) * t9;
      *(Gjk + 0 * ldG) += *(Xik + 9 * ldX) * t9;
   }
}
