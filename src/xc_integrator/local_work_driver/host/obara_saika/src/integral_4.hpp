#ifndef __MY_INTEGRAL_4
#define __MY_INTEGRAL_4

namespace XCPU {
void integral_4(size_t npts,
               shell_pair shpair,
               double *points,
               double *Xi,
               int ldX,
               double *Gi,
               int ldG, 
               double *weights, 
               double *boys_table);
}

#endif
