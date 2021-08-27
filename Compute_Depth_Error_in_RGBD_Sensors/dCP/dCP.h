#ifndef DCP_H
#define DCP_H
#include <cmath>
#include <float.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rt_defines.h"
#include "rt_nonfinite.h"
#include "rtwtypes.h"
#include "omp.h"
#include "dCP_types.h"

extern omp_nest_lock_t emlrtNestLockGlobal;
extern void dCP(const emxArray_uint8_T *I, emxArray_real_T *imagePoints, double
                boardSize[2]);
extern void dCP_initialize();
extern void dCP_terminate();

#endif
