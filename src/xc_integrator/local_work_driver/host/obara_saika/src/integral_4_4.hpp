#ifndef __MY_INTEGRAL_4_4
#define __MY_INTEGRAL_4_4

namespace XCPU {
void integral_4_4(size_t npts,
                  shell_pair shpair,
                  double *points,
                  double *Xi,
                  double *Xj,
                  int ldX,
                  double *Gi,
                  double *Gj,
                  int ldG, 
                  double *weights, 
                  double *boys_table);
}

#endif
