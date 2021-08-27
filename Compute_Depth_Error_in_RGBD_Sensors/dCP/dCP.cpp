#include "rt_nonfinite.h"
#include "dCP.h"
#include "dCP_emxutil.h"

typedef struct {
  boolean_T Neighborhood[9];
  int ImageSize[2];
  double InteriorStart[2];
  int InteriorEnd[2];
  int ImageNeighborLinearOffsets[9];
  double Padding;
  double PadValue;
  boolean_T ProcessBorder;
  double NeighborhoodCenter;
  int NeighborLinearIndices[9];
  int NeighborSubscriptOffsets[18];
} c_images_internal_coder_Neighbo;

static double X[150];
omp_nest_lock_t emlrtNestLockGlobal;
static void Checkerboard_expandBoardDown(const c_vision_internal_calibration_c
  *b_this, const emxArray_real_T *indices, emxArray_real_T *newBoard,
  emxArray_real_T *newBoardCoords);
static void Checkerboard_expandBoardLeft(const c_vision_internal_calibration_c
  *b_this, const emxArray_real_T *indices, emxArray_real_T *newBoard,
  emxArray_real_T *newBoardCoords);
static void Checkerboard_expandBoardUp(const c_vision_internal_calibration_c
  *b_this, const emxArray_real_T *indices, emxArray_real_T *newBoard,
  emxArray_real_T *newBoardCoords);
static void Checkerboard_findClosestIndices(const
  c_vision_internal_calibration_c *b_this, const emxArray_real_T
  *predictedPoints, emxArray_real_T *indices);
static double Checkerboard_findNeighbor(const c_vision_internal_calibration_c
  *b_this, const emxArray_real32_T *pointVectors, const emxArray_real32_T
  *euclideanDists, const float v[2]);
static void Checkerboard_initialize(c_vision_internal_calibration_c *b_this,
  double seedIdx, const emxArray_real32_T *points, const float v1[2], const
  float v2[2]);
static void algbwmorph(emxArray_boolean_T *bw);
static boolean_T any(const emxArray_boolean_T *x);
static void b_abs(const emxArray_real32_T *x, emxArray_real32_T *y);
static void b_hypot(const emxArray_real32_T *x, const emxArray_real32_T *y,
                    emxArray_real32_T *r);
static void b_imfilter(emxArray_real32_T *varargin_1);
static void b_merge(int idx[4], double x[4], int offset, int np, int nq, int
                    iwork[4], double xwork[4]);
static double b_mod(double x);
static void b_rot90(const emxArray_real_T *A, emxArray_real_T *B);
static void b_sort(emxArray_real32_T *x, int dim, emxArray_int32_T *idx);
static void b_squeeze(const emxArray_real_T *a, emxArray_real_T *b);
static void bsxfun(const emxArray_real32_T *a, const float b[2],
                   emxArray_real32_T *c);
static void bwlookup(const emxArray_boolean_T *bwin, emxArray_boolean_T *B);
static float c_Checkerboard_computeNewEnergy(const
  c_vision_internal_calibration_c *b_this, float oldEnergy);
static void c_Checkerboard_expandBoardDirec(c_vision_internal_calibration_c
  *b_this, double direction);
static void c_Checkerboard_predictPointsVer(const
  c_vision_internal_calibration_c *b_this, emxArray_real_T *newPoints);
static void c_hypot(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *r);
static void c_imfilter(emxArray_real32_T *varargin_1);
static void c_sort(double x[4]);
static void cat(const emxArray_real_T *varargin_1, const emxArray_real_T
                *varargin_2, emxArray_real_T *y);
static void cornerOrientations(const emxArray_real32_T *Ix2, const
  emxArray_real32_T *Iy2, const emxArray_real32_T *Ixy, const float p[2], float
  v1[2], float v2[2]);
static float d_Checkerboard_computeNewEnergy(const
  c_vision_internal_calibration_c *b_this, const emxArray_real_T *idx, float
  oldEnergy);
static void d_imfilter(emxArray_real32_T *varargin_1);
static void detectCheckerboard(const emxArray_real32_T *I, emxArray_real_T
  *points, double boardSize[2]);
static int div_s32(int numerator, int denominator);
static int div_s32_floor(int numerator, int denominator);
static float e_Checkerboard_computeNewEnergy(const
  c_vision_internal_calibration_c *b_this, float oldEnergy);
static void find_peaks(const emxArray_real32_T *metric, emxArray_real32_T *loc);
static c_vision_internal_calibration_c *growCheckerboard(const emxArray_real32_T
  *points, const emxArray_real32_T *scores, const emxArray_real32_T *Ix2, const
  emxArray_real32_T *Iy2, const emxArray_real32_T *Ixy, double theta,
  c_vision_internal_calibration_c *iobj_0, c_vision_internal_calibration_c
  *iobj_1, c_vision_internal_calibration_c *iobj_2);
static void imfilter(emxArray_real32_T *varargin_1);
static void imregionalmax(const emxArray_real32_T *varargin_1,
  emxArray_boolean_T *BW);
static boolean_T isequal(const emxArray_boolean_T *varargin_1, const
  emxArray_boolean_T *varargin_2);
static boolean_T isrow(const emxArray_real32_T *x);
static float mean(const emxArray_real32_T *x);
static void merge(emxArray_int32_T *idx, emxArray_real32_T *x, int offset, int
                  np, int nq, emxArray_int32_T *iwork, emxArray_real32_T *xwork);
static void merge_block(emxArray_int32_T *idx, emxArray_real32_T *x, int offset,
  int n, int preSortLevel, emxArray_int32_T *iwork, emxArray_real32_T *xwork);
static double norm(const emxArray_real_T *x);
static void orient(c_vision_internal_calibration_c **board, const
                   emxArray_real32_T *I);
static void padImage(const emxArray_real32_T *a_tmp, const double pad[2],
                     emxArray_real32_T *a);
static void poly2RectMask(double b_X[4], double Y[4], double height, double
  width, emxArray_boolean_T *mask);
static void power(const emxArray_real32_T *a, emxArray_real32_T *y);
static void rdivide(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *z);
static void rot90(const emxArray_real_T *A, emxArray_real_T *B);
static void rot90_checkerboard(c_vision_internal_calibration_c **board);
static float rt_atan2f_snf(float u0, float u1);
static double rt_hypotd_snf(double u0, double u1);
static float rt_hypotf_snf(float u0, float u1);
static double rt_remd_snf(double u0, double u1);
static double rt_roundd_snf(double u);
static void secondDerivCornerMetric(const emxArray_real32_T *I,
  emxArray_real32_T *cxy, emxArray_real32_T *c45, emxArray_real32_T *Ix,
  emxArray_real32_T *Iy, emxArray_real32_T *Ixy, emxArray_real32_T *I_45_45);
static void sort(emxArray_real32_T *x, emxArray_int32_T *idx);
static void sortIdx(emxArray_real32_T *x, emxArray_int32_T *idx);
static void squeeze(const emxArray_real_T *a, emxArray_real_T *b);
static void subPixelLocation(const emxArray_real32_T *metric, emxArray_real_T
  *loc);
static void subPixelLocationImpl_init();
static void toPoints(const c_vision_internal_calibration_c *b_this,
                     emxArray_real_T *points, double boardSize[2]);
static void Checkerboard_expandBoardDown(const c_vision_internal_calibration_c
  *b_this, const emxArray_real_T *indices, emxArray_real_T *newBoard,
  emxArray_real_T *newBoardCoords)
{
  int i18;
  int loop_ub;
  int b_newBoard;
  int i19;
  emxArray_int32_T *r26;
  emxArray_int32_T *r27;
  emxArray_real_T *r28;
  int unnamed_idx_2;
  int b_newBoardCoords;
  i18 = newBoard->size[0] * newBoard->size[1];
  newBoard->size[0] = b_this->BoardIdx->size[0] + 1;
  newBoard->size[1] = b_this->BoardIdx->size[1];
  emxEnsureCapacity_real_T(newBoard, i18);
  loop_ub = (b_this->BoardIdx->size[0] + 1) * b_this->BoardIdx->size[1];
  for (i18 = 0; i18 < loop_ub; i18++) {
    newBoard->data[i18] = 0.0;
  }

  b_newBoard = newBoard->size[0] - 1;
  loop_ub = indices->size[1];
  for (i18 = 0; i18 < loop_ub; i18++) {
    newBoard->data[b_newBoard + newBoard->size[0] * i18] = indices->data
      [indices->size[0] * i18];
  }

  loop_ub = b_this->BoardIdx->size[1];
  for (i18 = 0; i18 < loop_ub; i18++) {
    b_newBoard = b_this->BoardIdx->size[0];
    for (i19 = 0; i19 < b_newBoard; i19++) {
      newBoard->data[i19 + newBoard->size[0] * i18] = b_this->BoardIdx->data[i19
        + b_this->BoardIdx->size[0] * i18];
    }
  }

  i18 = newBoardCoords->size[0] * newBoardCoords->size[1] * newBoardCoords->
    size[2];
  newBoardCoords->size[0] = b_this->BoardCoords->size[0] + 1;
  newBoardCoords->size[1] = b_this->BoardCoords->size[1];
  newBoardCoords->size[2] = b_this->BoardCoords->size[2];
  emxEnsureCapacity_real_T1(newBoardCoords, i18);
  loop_ub = (b_this->BoardCoords->size[0] + 1) * b_this->BoardCoords->size[1] *
    b_this->BoardCoords->size[2];
  for (i18 = 0; i18 < loop_ub; i18++) {
    newBoardCoords->data[i18] = 0.0;
  }

  emxInit_int32_T(&r26, 1);
  loop_ub = newBoardCoords->size[1];
  i18 = r26->size[0];
  r26->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r26, i18);
  for (i18 = 0; i18 < loop_ub; i18++) {
    r26->data[i18] = i18;
  }

  emxInit_int32_T(&r27, 1);
  loop_ub = newBoardCoords->size[2];
  i18 = r27->size[0];
  r27->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r27, i18);
  for (i18 = 0; i18 < loop_ub; i18++) {
    r27->data[i18] = i18;
  }

  emxInit_real_T1(&r28, 2);
  i18 = r28->size[0] * r28->size[1];
  r28->size[0] = indices->size[1];
  r28->size[1] = 2;
  emxEnsureCapacity_real_T(r28, i18);
  loop_ub = indices->size[1];
  for (i18 = 0; i18 < 2; i18++) {
    for (i19 = 0; i19 < loop_ub; i19++) {
      r28->data[i19 + r28->size[0] * i18] = b_this->Points->data[((int)
        indices->data[indices->size[0] * i19] + b_this->Points->size[0] * i18) -
        1];
    }
  }

  b_newBoard = r26->size[0];
  unnamed_idx_2 = r27->size[0];
  b_newBoardCoords = newBoardCoords->size[0] - 1;
  for (i18 = 0; i18 < unnamed_idx_2; i18++) {
    for (i19 = 0; i19 < b_newBoard; i19++) {
      newBoardCoords->data[(b_newBoardCoords + newBoardCoords->size[0] *
                            r26->data[i19]) + newBoardCoords->size[0] *
        newBoardCoords->size[1] * r27->data[i18]] = r28->data[i19 + b_newBoard *
        i18];
    }
  }

  emxFree_real_T(&r28);
  emxFree_int32_T(&r27);
  emxFree_int32_T(&r26);
  loop_ub = b_this->BoardCoords->size[2];
  for (i18 = 0; i18 < loop_ub; i18++) {
    b_newBoard = b_this->BoardCoords->size[1];
    for (i19 = 0; i19 < b_newBoard; i19++) {
      unnamed_idx_2 = b_this->BoardCoords->size[0];
      for (b_newBoardCoords = 0; b_newBoardCoords < unnamed_idx_2;
           b_newBoardCoords++) {
        newBoardCoords->data[(b_newBoardCoords + newBoardCoords->size[0] * i19)
          + newBoardCoords->size[0] * newBoardCoords->size[1] * i18] =
          b_this->BoardCoords->data[(b_newBoardCoords + b_this->
          BoardCoords->size[0] * i19) + b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1] * i18];
      }
    }
  }
}

static void Checkerboard_expandBoardLeft(const c_vision_internal_calibration_c
  *b_this, const emxArray_real_T *indices, emxArray_real_T *newBoard,
  emxArray_real_T *newBoardCoords)
{
  int i22;
  int loop_ub;
  emxArray_int32_T *r35;
  int i23;
  int b_loop_ub;
  int i24;
  emxArray_int32_T *r36;
  emxArray_real_T *r37;
  int c_loop_ub;
  int i25;
  i22 = newBoard->size[0] * newBoard->size[1];
  newBoard->size[0] = b_this->BoardIdx->size[0];
  newBoard->size[1] = b_this->BoardIdx->size[1] + 1;
  emxEnsureCapacity_real_T(newBoard, i22);
  loop_ub = b_this->BoardIdx->size[0] * (b_this->BoardIdx->size[1] + 1);
  for (i22 = 0; i22 < loop_ub; i22++) {
    newBoard->data[i22] = 0.0;
  }

  emxInit_int32_T(&r35, 1);
  loop_ub = newBoard->size[0];
  i22 = r35->size[0];
  r35->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r35, i22);
  for (i22 = 0; i22 < loop_ub; i22++) {
    r35->data[i22] = i22;
  }

  loop_ub = r35->size[0];
  for (i22 = 0; i22 < loop_ub; i22++) {
    newBoard->data[r35->data[i22]] = indices->data[i22];
  }

  i22 = !(2 > newBoard->size[1]);
  loop_ub = b_this->BoardIdx->size[1];
  for (i23 = 0; i23 < loop_ub; i23++) {
    b_loop_ub = b_this->BoardIdx->size[0];
    for (i24 = 0; i24 < b_loop_ub; i24++) {
      newBoard->data[i24 + newBoard->size[0] * (i22 + i23)] = b_this->
        BoardIdx->data[i24 + b_this->BoardIdx->size[0] * i23];
    }
  }

  i22 = newBoardCoords->size[0] * newBoardCoords->size[1] * newBoardCoords->
    size[2];
  newBoardCoords->size[0] = b_this->BoardCoords->size[0];
  newBoardCoords->size[1] = b_this->BoardCoords->size[1] + 1;
  newBoardCoords->size[2] = b_this->BoardCoords->size[2];
  emxEnsureCapacity_real_T1(newBoardCoords, i22);
  loop_ub = b_this->BoardCoords->size[0] * (b_this->BoardCoords->size[1] + 1) *
    b_this->BoardCoords->size[2];
  for (i22 = 0; i22 < loop_ub; i22++) {
    newBoardCoords->data[i22] = 0.0;
  }

  loop_ub = newBoardCoords->size[0];
  i22 = r35->size[0];
  r35->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r35, i22);
  for (i22 = 0; i22 < loop_ub; i22++) {
    r35->data[i22] = i22;
  }

  emxInit_int32_T(&r36, 1);
  loop_ub = newBoardCoords->size[2];
  i22 = r36->size[0];
  r36->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r36, i22);
  for (i22 = 0; i22 < loop_ub; i22++) {
    r36->data[i22] = i22;
  }

  emxInit_real_T1(&r37, 2);
  i22 = r37->size[0] * r37->size[1];
  r37->size[0] = indices->size[1];
  r37->size[1] = 2;
  emxEnsureCapacity_real_T(r37, i22);
  loop_ub = indices->size[1];
  for (i22 = 0; i22 < 2; i22++) {
    for (i23 = 0; i23 < loop_ub; i23++) {
      r37->data[i23 + r37->size[0] * i22] = b_this->Points->data[((int)
        indices->data[indices->size[0] * i23] + b_this->Points->size[0] * i22) -
        1];
    }
  }

  loop_ub = r35->size[0];
  b_loop_ub = r36->size[0];
  for (i22 = 0; i22 < b_loop_ub; i22++) {
    for (i23 = 0; i23 < loop_ub; i23++) {
      newBoardCoords->data[r35->data[i23] + newBoardCoords->size[0] *
        newBoardCoords->size[1] * r36->data[i22]] = r37->data[i23 + loop_ub *
        i22];
    }
  }

  emxFree_real_T(&r37);
  emxFree_int32_T(&r36);
  emxFree_int32_T(&r35);
  i22 = !(2 > newBoardCoords->size[1]);
  loop_ub = b_this->BoardCoords->size[2];
  for (i23 = 0; i23 < loop_ub; i23++) {
    b_loop_ub = b_this->BoardCoords->size[1];
    for (i24 = 0; i24 < b_loop_ub; i24++) {
      c_loop_ub = b_this->BoardCoords->size[0];
      for (i25 = 0; i25 < c_loop_ub; i25++) {
        newBoardCoords->data[(i25 + newBoardCoords->size[0] * (i22 + i24)) +
          newBoardCoords->size[0] * newBoardCoords->size[1] * i23] =
          b_this->BoardCoords->data[(i25 + b_this->BoardCoords->size[0] * i24) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i23];
      }
    }
  }
}

static void Checkerboard_expandBoardUp(const c_vision_internal_calibration_c
  *b_this, const emxArray_real_T *indices, emxArray_real_T *newBoard,
  emxArray_real_T *newBoardCoords)
{
  int i12;
  int loop_ub;
  int i13;
  int b_loop_ub;
  int i14;
  emxArray_int32_T *r17;
  emxArray_int32_T *r18;
  emxArray_real_T *r19;
  int c_loop_ub;
  int i15;
  i12 = newBoard->size[0] * newBoard->size[1];
  newBoard->size[0] = b_this->BoardIdx->size[0] + 1;
  newBoard->size[1] = b_this->BoardIdx->size[1];
  emxEnsureCapacity_real_T(newBoard, i12);
  loop_ub = (b_this->BoardIdx->size[0] + 1) * b_this->BoardIdx->size[1];
  for (i12 = 0; i12 < loop_ub; i12++) {
    newBoard->data[i12] = 0.0;
  }

  loop_ub = indices->size[1];
  for (i12 = 0; i12 < loop_ub; i12++) {
    newBoard->data[newBoard->size[0] * i12] = indices->data[indices->size[0] *
      i12];
  }

  i12 = !(2 > newBoard->size[0]);
  loop_ub = b_this->BoardIdx->size[1];
  for (i13 = 0; i13 < loop_ub; i13++) {
    b_loop_ub = b_this->BoardIdx->size[0];
    for (i14 = 0; i14 < b_loop_ub; i14++) {
      newBoard->data[(i12 + i14) + newBoard->size[0] * i13] = b_this->
        BoardIdx->data[i14 + b_this->BoardIdx->size[0] * i13];
    }
  }

  i12 = newBoardCoords->size[0] * newBoardCoords->size[1] * newBoardCoords->
    size[2];
  newBoardCoords->size[0] = b_this->BoardCoords->size[0] + 1;
  newBoardCoords->size[1] = b_this->BoardCoords->size[1];
  newBoardCoords->size[2] = b_this->BoardCoords->size[2];
  emxEnsureCapacity_real_T1(newBoardCoords, i12);
  loop_ub = (b_this->BoardCoords->size[0] + 1) * b_this->BoardCoords->size[1] *
    b_this->BoardCoords->size[2];
  for (i12 = 0; i12 < loop_ub; i12++) {
    newBoardCoords->data[i12] = 0.0;
  }

  emxInit_int32_T(&r17, 1);
  loop_ub = newBoardCoords->size[1];
  i12 = r17->size[0];
  r17->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r17, i12);
  for (i12 = 0; i12 < loop_ub; i12++) {
    r17->data[i12] = i12;
  }

  emxInit_int32_T(&r18, 1);
  loop_ub = newBoardCoords->size[2];
  i12 = r18->size[0];
  r18->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r18, i12);
  for (i12 = 0; i12 < loop_ub; i12++) {
    r18->data[i12] = i12;
  }

  emxInit_real_T1(&r19, 2);
  i12 = r19->size[0] * r19->size[1];
  r19->size[0] = indices->size[1];
  r19->size[1] = 2;
  emxEnsureCapacity_real_T(r19, i12);
  loop_ub = indices->size[1];
  for (i12 = 0; i12 < 2; i12++) {
    for (i13 = 0; i13 < loop_ub; i13++) {
      r19->data[i13 + r19->size[0] * i12] = b_this->Points->data[((int)
        indices->data[indices->size[0] * i13] + b_this->Points->size[0] * i12) -
        1];
    }
  }

  loop_ub = r17->size[0];
  b_loop_ub = r18->size[0];
  for (i12 = 0; i12 < b_loop_ub; i12++) {
    for (i13 = 0; i13 < loop_ub; i13++) {
      newBoardCoords->data[newBoardCoords->size[0] * r17->data[i13] +
        newBoardCoords->size[0] * newBoardCoords->size[1] * r18->data[i12]] =
        r19->data[i13 + loop_ub * i12];
    }
  }

  emxFree_real_T(&r19);
  emxFree_int32_T(&r18);
  emxFree_int32_T(&r17);
  i12 = !(2 > newBoardCoords->size[0]);
  loop_ub = b_this->BoardCoords->size[2];
  for (i13 = 0; i13 < loop_ub; i13++) {
    b_loop_ub = b_this->BoardCoords->size[1];
    for (i14 = 0; i14 < b_loop_ub; i14++) {
      c_loop_ub = b_this->BoardCoords->size[0];
      for (i15 = 0; i15 < c_loop_ub; i15++) {
        newBoardCoords->data[((i12 + i15) + newBoardCoords->size[0] * i14) +
          newBoardCoords->size[0] * newBoardCoords->size[1] * i13] =
          b_this->BoardCoords->data[(i15 + b_this->BoardCoords->size[0] * i14) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i13];
      }
    }
  }
}

static void Checkerboard_findClosestIndices(const
  c_vision_internal_calibration_c *b_this, const emxArray_real_T
  *predictedPoints, emxArray_real_T *indices)
{
  int n;
  int csz_idx_0;
  int i;
  emxArray_real32_T *diffs;
  emxArray_real32_T *dists;
  emxArray_real32_T *a;
  emxArray_real32_T *b_diffs;
  emxArray_real32_T *c_diffs;
  int ib;
  int szc;
  int acoef;
  int k;
  float mtmp;
  boolean_T exitg1;
  n = indices->size[0] * indices->size[1];
  indices->size[0] = 1;
  indices->size[1] = predictedPoints->size[0];
  emxEnsureCapacity_real_T(indices, n);
  csz_idx_0 = predictedPoints->size[0];
  for (n = 0; n < csz_idx_0; n++) {
    indices->data[n] = 0.0;
  }

  i = 0;
  emxInit_real32_T(&diffs, 2);
  emxInit_real32_T1(&dists, 1);
  emxInit_real32_T(&a, 2);
  emxInit_real32_T1(&b_diffs, 1);
  emxInit_real32_T1(&c_diffs, 1);
  while (i <= predictedPoints->size[0] - 1) {
    n = a->size[0] * a->size[1];
    a->size[0] = b_this->Points->size[0];
    a->size[1] = 2;
    emxEnsureCapacity_real32_T(a, n);
    csz_idx_0 = b_this->Points->size[0] * b_this->Points->size[1];
    for (n = 0; n < csz_idx_0; n++) {
      a->data[n] = b_this->Points->data[n];
    }

    csz_idx_0 = a->size[0];
    n = diffs->size[0] * diffs->size[1];
    diffs->size[0] = csz_idx_0;
    diffs->size[1] = 2;
    emxEnsureCapacity_real32_T(diffs, n);
    if (diffs->size[0] != 0) {
      n = predictedPoints->size[1];
      csz_idx_0 = (n != 1);
      for (n = 0; n < 2; n++) {
        ib = csz_idx_0 * n;
        szc = diffs->size[0];
        acoef = (a->size[0] != 1);
        for (k = 0; k < szc; k++) {
          diffs->data[k + diffs->size[0] * n] = a->data[acoef * k + a->size[0] *
            n] - (float)predictedPoints->data[i + predictedPoints->size[0] * ib];
        }
      }
    }

    csz_idx_0 = diffs->size[0];
    n = b_diffs->size[0];
    b_diffs->size[0] = csz_idx_0;
    emxEnsureCapacity_real32_T1(b_diffs, n);
    for (n = 0; n < csz_idx_0; n++) {
      b_diffs->data[n] = diffs->data[n];
    }

    csz_idx_0 = diffs->size[0];
    n = c_diffs->size[0];
    c_diffs->size[0] = csz_idx_0;
    emxEnsureCapacity_real32_T1(c_diffs, n);
    for (n = 0; n < csz_idx_0; n++) {
      c_diffs->data[n] = diffs->data[n + diffs->size[0]];
    }

    b_hypot(b_diffs, c_diffs, dists);
    csz_idx_0 = indices->size[1];
    for (n = 0; n < csz_idx_0; n++) {
      if ((int)indices->data[n] > 0) {
        dists->data[(int)indices->data[n] - 1] = ((real32_T)rtInf);
      }
    }

    csz_idx_0 = 1;
    n = dists->size[0];
    mtmp = dists->data[0];
    ib = 1;
    if (dists->size[0] > 1) {
      if (rtIsNaNF(dists->data[0])) {
        szc = 2;
        exitg1 = false;
        while ((!exitg1) && (szc <= n)) {
          csz_idx_0 = szc;
          if (!rtIsNaNF(dists->data[szc - 1])) {
            mtmp = dists->data[szc - 1];
            ib = szc;
            exitg1 = true;
          } else {
            szc++;
          }
        }
      }

      if (csz_idx_0 < dists->size[0]) {
        while (csz_idx_0 + 1 <= n) {
          if (dists->data[csz_idx_0] < mtmp) {
            mtmp = dists->data[csz_idx_0];
            ib = csz_idx_0 + 1;
          }

          csz_idx_0++;
        }
      }
    }

    indices->data[i] = ib;
    i++;
  }

  emxFree_real32_T(&c_diffs);
  emxFree_real32_T(&b_diffs);
  emxFree_real32_T(&a);
  emxFree_real32_T(&dists);
  emxFree_real32_T(&diffs);
}

static double Checkerboard_findNeighbor(const c_vision_internal_calibration_c
  *b_this, const emxArray_real32_T *pointVectors, const emxArray_real32_T
  *euclideanDists, const float v[2])
{
  double neighborIdx;
  emxArray_real32_T *angleCosines;
  unsigned int pointVectors_idx_0;
  int n;
  int m;
  int ar;
  float b;
  int br;
  int ic;
  emxArray_real32_T *dists;
  int ia;
  emxArray_boolean_T *r6;
  emxArray_int32_T *r7;
  emxArray_int32_T *r8;
  boolean_T exitg1;
  emxInit_real32_T1(&angleCosines, 1);
  pointVectors_idx_0 = (unsigned int)pointVectors->size[0];
  n = angleCosines->size[0];
  angleCosines->size[0] = (int)pointVectors_idx_0;
  emxEnsureCapacity_real32_T1(angleCosines, n);
  m = pointVectors->size[0];
  ar = angleCosines->size[0];
  n = angleCosines->size[0];
  angleCosines->size[0] = ar;
  emxEnsureCapacity_real32_T1(angleCosines, n);
  for (n = 0; n < ar; n++) {
    angleCosines->data[n] = 0.0F;
  }

  if (pointVectors->size[0] != 0) {
    ar = 0;
    while ((m > 0) && (ar <= 0)) {
      for (ic = 1; ic <= m; ic++) {
        angleCosines->data[ic - 1] = 0.0F;
      }

      ar = m;
    }

    br = 0;
    ar = 0;
    while ((m > 0) && (ar <= 0)) {
      ar = -1;
      for (n = br; n + 1 <= br + 2; n++) {
        if (v[n] != 0.0F) {
          ia = ar;
          for (ic = 0; ic + 1 <= m; ic++) {
            ia++;
            angleCosines->data[ic] += v[n] * pointVectors->data[ia];
          }
        }

        ar += m;
      }

      br += 2;
      ar = m;
    }
  }

  b = rt_hypotf_snf(v[0], v[1]);
  n = angleCosines->size[0];
  emxEnsureCapacity_real32_T1(angleCosines, n);
  ar = angleCosines->size[0];
  for (n = 0; n < ar; n++) {
    angleCosines->data[n] /= euclideanDists->data[n] * b;
  }

  emxInit_real32_T1(&dists, 1);
  n = dists->size[0];
  dists->size[0] = euclideanDists->size[0];
  emxEnsureCapacity_real32_T1(dists, n);
  ar = euclideanDists->size[0];
  for (n = 0; n < ar; n++) {
    dists->data[n] = euclideanDists->data[n] + 1.5F * euclideanDists->data[n] *
      (1.0F - angleCosines->data[n]);
  }

  emxInit_boolean_T(&r6, 2);
  n = r6->size[0] * r6->size[1];
  r6->size[0] = b_this->BoardIdx->size[0];
  r6->size[1] = b_this->BoardIdx->size[1];
  emxEnsureCapacity_boolean_T(r6, n);
  ar = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  for (n = 0; n < ar; n++) {
    r6->data[n] = (b_this->BoardIdx->data[n] > 0.0);
  }

  ic = r6->size[0] * r6->size[1] - 1;
  ar = 0;
  for (n = 0; n <= ic; n++) {
    if (r6->data[n]) {
      ar++;
    }
  }

  emxInit_int32_T(&r7, 1);
  n = r7->size[0];
  r7->size[0] = ar;
  emxEnsureCapacity_int32_T(r7, n);
  ar = 0;
  for (n = 0; n <= ic; n++) {
    if (r6->data[n]) {
      r7->data[ar] = n + 1;
      ar++;
    }
  }

  emxFree_boolean_T(&r6);
  emxInit_int32_T(&r8, 1);
  n = r8->size[0];
  r8->size[0] = r7->size[0];
  emxEnsureCapacity_int32_T(r8, n);
  ar = r7->size[0];
  for (n = 0; n < ar; n++) {
    r8->data[n] = (int)b_this->BoardIdx->data[r7->data[n] - 1];
  }

  emxFree_int32_T(&r7);
  ar = r8->size[0];
  for (n = 0; n < ar; n++) {
    dists->data[r8->data[n] - 1] = ((real32_T)rtInf);
  }

  emxFree_int32_T(&r8);
  ic = angleCosines->size[0];
  for (n = 0; n < ic; n++) {
    if (angleCosines->data[n] < 0.0F) {
      dists->data[n] = ((real32_T)rtInf);
    }
  }

  emxFree_real32_T(&angleCosines);
  ar = 1;
  n = dists->size[0];
  b = dists->data[0];
  ic = 1;
  if (dists->size[0] > 1) {
    if (rtIsNaNF(dists->data[0])) {
      ia = 2;
      exitg1 = false;
      while ((!exitg1) && (ia <= n)) {
        ar = ia;
        if (!rtIsNaNF(dists->data[ia - 1])) {
          b = dists->data[ia - 1];
          ic = ia;
          exitg1 = true;
        } else {
          ia++;
        }
      }
    }

    if (ar < dists->size[0]) {
      while (ar + 1 <= n) {
        if (dists->data[ar] < b) {
          b = dists->data[ar];
          ic = ar + 1;
        }

        ar++;
      }
    }
  }

  emxFree_real32_T(&dists);
  neighborIdx = ic;
  if (rtIsInfF(b)) {
    neighborIdx = -1.0;
  }

  return neighborIdx;
}

static void Checkerboard_initialize(c_vision_internal_calibration_c *b_this,
  double seedIdx, const emxArray_real32_T *points, const float v1[2], const
  float v2[2])
{
  int ixstart;
  int loop_ub;
  emxArray_int32_T *r5;
  float center[2];
  double b_center[2];
  emxArray_real32_T *pointVectors;
  emxArray_real32_T *b_pointVectors;
  emxArray_real32_T *c_pointVectors;
  emxArray_real32_T *euclideanDists;
  float b_v1[2];
  emxArray_boolean_T *c_this;
  float r[2];
  float l[2];
  float d[2];
  float u[2];
  float boardSize;
  float b_l;
  emxArray_boolean_T *x;
  boolean_T y;
  boolean_T exitg1;
  float col1[6];
  float col2[6];
  float row3[6];
  float z1[3];
  ixstart = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  b_this->BoardIdx->size[0] = 1;
  b_this->BoardIdx->size[1] = 1;
  emxEnsureCapacity_real_T(b_this->BoardIdx, ixstart);
  b_this->BoardIdx->data[0] = 0.0;
  ixstart = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  b_this->BoardIdx->size[0] = 3;
  b_this->BoardIdx->size[1] = 3;
  emxEnsureCapacity_real_T(b_this->BoardIdx, ixstart);
  for (ixstart = 0; ixstart < 9; ixstart++) {
    b_this->BoardIdx->data[ixstart] = 0.0;
  }

  for (ixstart = 0; ixstart < 4; ixstart++) {
    b_this->IsDirectionBad[ixstart] = false;
  }

  ixstart = b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
    b_this->BoardCoords->size[2];
  b_this->BoardCoords->size[0] = 1;
  b_this->BoardCoords->size[1] = 1;
  b_this->BoardCoords->size[2] = 1;
  emxEnsureCapacity_real_T1(b_this->BoardCoords, ixstart);
  b_this->BoardCoords->data[0] = 0.0;
  ixstart = b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
    b_this->BoardCoords->size[2];
  b_this->BoardCoords->size[0] = 3;
  b_this->BoardCoords->size[1] = 3;
  b_this->BoardCoords->size[2] = 2;
  emxEnsureCapacity_real_T1(b_this->BoardCoords, ixstart);
  for (ixstart = 0; ixstart < 18; ixstart++) {
    b_this->BoardCoords->data[ixstart] = 0.0;
  }

  ixstart = b_this->Points->size[0] * b_this->Points->size[1];
  b_this->Points->size[0] = points->size[0];
  b_this->Points->size[1] = 2;
  emxEnsureCapacity_real32_T(b_this->Points, ixstart);
  loop_ub = points->size[0] * points->size[1];
  for (ixstart = 0; ixstart < loop_ub; ixstart++) {
    b_this->Points->data[ixstart] = points->data[ixstart];
  }

  for (ixstart = 0; ixstart < 2; ixstart++) {
    center[ixstart] = b_this->Points->data[((int)seedIdx + b_this->Points->size
      [0] * ixstart) - 1];
  }

  emxInit_int32_T(&r5, 1);
  b_this->BoardIdx->data[1 + b_this->BoardIdx->size[0]] = seedIdx;
  loop_ub = b_this->BoardCoords->size[2];
  ixstart = r5->size[0];
  r5->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r5, ixstart);
  for (ixstart = 0; ixstart < loop_ub; ixstart++) {
    r5->data[ixstart] = ixstart;
  }

  for (ixstart = 0; ixstart < 2; ixstart++) {
    b_center[ixstart] = center[ixstart];
  }

  loop_ub = r5->size[0];
  for (ixstart = 0; ixstart < loop_ub; ixstart++) {
    b_this->BoardCoords->data[(b_this->BoardCoords->size[0] +
      b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * r5->
      data[ixstart]) + 1] = b_center[ixstart];
  }

  emxInit_real32_T(&pointVectors, 2);
  emxInit_real32_T1(&b_pointVectors, 1);
  b_this->LastExpandDirection = 1.0;
  b_this->PreviousEnergy = ((real32_T)rtInf);
  b_this->isValid = false;
  bsxfun(b_this->Points, center, pointVectors);
  loop_ub = pointVectors->size[0];
  ixstart = b_pointVectors->size[0];
  b_pointVectors->size[0] = loop_ub;
  emxEnsureCapacity_real32_T1(b_pointVectors, ixstart);
  for (ixstart = 0; ixstart < loop_ub; ixstart++) {
    b_pointVectors->data[ixstart] = pointVectors->data[ixstart];
  }

  emxInit_real32_T1(&c_pointVectors, 1);
  loop_ub = pointVectors->size[0];
  ixstart = c_pointVectors->size[0];
  c_pointVectors->size[0] = loop_ub;
  emxEnsureCapacity_real32_T1(c_pointVectors, ixstart);
  for (ixstart = 0; ixstart < loop_ub; ixstart++) {
    c_pointVectors->data[ixstart] = pointVectors->data[ixstart +
      pointVectors->size[0]];
  }

  emxInit_real32_T1(&euclideanDists, 1);
  b_hypot(b_pointVectors, c_pointVectors, euclideanDists);
  b_this->BoardIdx->data[1 + (b_this->BoardIdx->size[0] << 1)] =
    Checkerboard_findNeighbor(b_this, pointVectors, euclideanDists, v1);
  emxFree_real32_T(&c_pointVectors);
  emxFree_real32_T(&b_pointVectors);
  for (ixstart = 0; ixstart < 2; ixstart++) {
    b_v1[ixstart] = -v1[ixstart];
  }

  b_this->BoardIdx->data[1] = Checkerboard_findNeighbor(b_this, pointVectors,
    euclideanDists, b_v1);
  b_this->BoardIdx->data[2 + b_this->BoardIdx->size[0]] =
    Checkerboard_findNeighbor(b_this, pointVectors, euclideanDists, v2);
  for (ixstart = 0; ixstart < 2; ixstart++) {
    b_v1[ixstart] = -v2[ixstart];
  }

  emxInit_boolean_T1(&c_this, 1);
  b_this->BoardIdx->data[b_this->BoardIdx->size[0]] = Checkerboard_findNeighbor
    (b_this, pointVectors, euclideanDists, b_v1);
  ixstart = c_this->size[0];
  c_this->size[0] = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  emxEnsureCapacity_boolean_T1(c_this, ixstart);
  loop_ub = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  for (ixstart = 0; ixstart < loop_ub; ixstart++) {
    c_this->data[ixstart] = (b_this->BoardIdx->data[ixstart] < 0.0);
  }

  if (any(c_this)) {
    b_this->isValid = false;
  } else {
    loop_ub = (int)b_this->BoardIdx->data[1 + (b_this->BoardIdx->size[0] << 1)];
    for (ixstart = 0; ixstart < 2; ixstart++) {
      r[ixstart] = b_this->Points->data[(loop_ub + b_this->Points->size[0] *
        ixstart) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    ixstart = r5->size[0];
    r5->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r5, ixstart);
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      r5->data[ixstart] = ixstart;
    }

    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_center[ixstart] = r[ixstart];
    }

    loop_ub = r5->size[0];
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      b_this->BoardCoords->data[((b_this->BoardCoords->size[0] << 1) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * r5->
        data[ixstart]) + 1] = b_center[ixstart];
    }

    loop_ub = (int)b_this->BoardIdx->data[1];
    for (ixstart = 0; ixstart < 2; ixstart++) {
      l[ixstart] = b_this->Points->data[(loop_ub + b_this->Points->size[0] *
        ixstart) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    ixstart = r5->size[0];
    r5->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r5, ixstart);
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      r5->data[ixstart] = ixstart;
    }

    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_center[ixstart] = l[ixstart];
    }

    loop_ub = r5->size[0];
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      b_this->BoardCoords->data[1 + b_this->BoardCoords->size[0] *
        b_this->BoardCoords->size[1] * r5->data[ixstart]] = b_center[ixstart];
    }

    loop_ub = (int)b_this->BoardIdx->data[2 + b_this->BoardIdx->size[0]];
    for (ixstart = 0; ixstart < 2; ixstart++) {
      d[ixstart] = b_this->Points->data[(loop_ub + b_this->Points->size[0] *
        ixstart) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    ixstart = r5->size[0];
    r5->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r5, ixstart);
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      r5->data[ixstart] = ixstart;
    }

    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_center[ixstart] = d[ixstart];
    }

    loop_ub = r5->size[0];
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      b_this->BoardCoords->data[(b_this->BoardCoords->size[0] +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * r5->
        data[ixstart]) + 2] = b_center[ixstart];
    }

    loop_ub = (int)b_this->BoardIdx->data[b_this->BoardIdx->size[0]];
    for (ixstart = 0; ixstart < 2; ixstart++) {
      u[ixstart] = b_this->Points->data[(loop_ub + b_this->Points->size[0] *
        ixstart) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    ixstart = r5->size[0];
    r5->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r5, ixstart);
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      r5->data[ixstart] = ixstart;
    }

    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_center[ixstart] = u[ixstart];
    }

    loop_ub = r5->size[0];
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      b_this->BoardCoords->data[b_this->BoardCoords->size[0] +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * r5->
        data[ixstart]] = b_center[ixstart];
    }

    for (ixstart = 0; ixstart < 2; ixstart++) {
      boardSize = u[ixstart] - center[ixstart];
      b_l = l[ixstart] - center[ixstart];
      b_v1[ixstart] = boardSize + b_l;
      u[ixstart] = boardSize;
      d[ixstart] -= center[ixstart];
      r[ixstart] -= center[ixstart];
      l[ixstart] = b_l;
    }

    b_this->BoardIdx->data[0] = Checkerboard_findNeighbor(b_this, pointVectors,
      euclideanDists, b_v1);
    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_v1[ixstart] = d[ixstart] + l[ixstart];
    }

    b_this->BoardIdx->data[2] = Checkerboard_findNeighbor(b_this, pointVectors,
      euclideanDists, b_v1);
    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_v1[ixstart] = d[ixstart] + r[ixstart];
    }

    b_this->BoardIdx->data[2 + (b_this->BoardIdx->size[0] << 1)] =
      Checkerboard_findNeighbor(b_this, pointVectors, euclideanDists, b_v1);
    for (ixstart = 0; ixstart < 2; ixstart++) {
      b_v1[ixstart] = u[ixstart] + r[ixstart];
    }

    emxInit_boolean_T1(&x, 1);
    b_this->BoardIdx->data[b_this->BoardIdx->size[0] << 1] =
      Checkerboard_findNeighbor(b_this, pointVectors, euclideanDists, b_v1);
    ixstart = x->size[0];
    x->size[0] = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    emxEnsureCapacity_boolean_T1(x, ixstart);
    loop_ub = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    for (ixstart = 0; ixstart < loop_ub; ixstart++) {
      x->data[ixstart] = (b_this->BoardIdx->data[ixstart] > 0.0);
    }

    y = true;
    loop_ub = 1;
    exitg1 = false;
    while ((!exitg1) && (loop_ub <= x->size[0])) {
      if (!x->data[loop_ub - 1]) {
        y = false;
        exitg1 = true;
      } else {
        loop_ub++;
      }
    }

    emxFree_boolean_T(&x);
    b_this->isValid = y;
    if (b_this->isValid) {
      loop_ub = b_this->BoardCoords->size[2];
      ixstart = r5->size[0];
      r5->size[0] = loop_ub;
      emxEnsureCapacity_int32_T(r5, ixstart);
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        r5->data[ixstart] = ixstart;
      }

      loop_ub = (int)b_this->BoardIdx->data[0] - 1;
      for (ixstart = 0; ixstart < 2; ixstart++) {
        b_center[ixstart] = b_this->Points->data[loop_ub + b_this->Points->size
          [0] * ixstart];
      }

      loop_ub = r5->size[0];
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        b_this->BoardCoords->data[b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1] * r5->data[ixstart]] = b_center[ixstart];
      }

      loop_ub = b_this->BoardCoords->size[2];
      ixstart = r5->size[0];
      r5->size[0] = loop_ub;
      emxEnsureCapacity_int32_T(r5, ixstart);
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        r5->data[ixstart] = ixstart;
      }

      loop_ub = (int)b_this->BoardIdx->data[2] - 1;
      for (ixstart = 0; ixstart < 2; ixstart++) {
        b_center[ixstart] = b_this->Points->data[loop_ub + b_this->Points->size
          [0] * ixstart];
      }

      loop_ub = r5->size[0];
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        b_this->BoardCoords->data[2 + b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1] * r5->data[ixstart]] = b_center[ixstart];
      }

      loop_ub = b_this->BoardCoords->size[2];
      ixstart = r5->size[0];
      r5->size[0] = loop_ub;
      emxEnsureCapacity_int32_T(r5, ixstart);
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        r5->data[ixstart] = ixstart;
      }

      loop_ub = (int)b_this->BoardIdx->data[2 + (b_this->BoardIdx->size[0] << 1)]
        - 1;
      for (ixstart = 0; ixstart < 2; ixstart++) {
        b_center[ixstart] = b_this->Points->data[loop_ub + b_this->Points->size
          [0] * ixstart];
      }

      loop_ub = r5->size[0];
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        b_this->BoardCoords->data[((b_this->BoardCoords->size[0] << 1) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * r5->
          data[ixstart]) + 2] = b_center[ixstart];
      }

      loop_ub = b_this->BoardCoords->size[2];
      ixstart = r5->size[0];
      r5->size[0] = loop_ub;
      emxEnsureCapacity_int32_T(r5, ixstart);
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        r5->data[ixstart] = ixstart;
      }

      loop_ub = (int)b_this->BoardIdx->data[b_this->BoardIdx->size[0] << 1] - 1;
      for (ixstart = 0; ixstart < 2; ixstart++) {
        b_center[ixstart] = b_this->Points->data[loop_ub + b_this->Points->size
          [0] * ixstart];
      }

      loop_ub = r5->size[0];
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        b_this->BoardCoords->data[(b_this->BoardCoords->size[0] << 1) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * r5->
          data[ixstart]] = b_center[ixstart];
      }

      ixstart = c_this->size[0];
      c_this->size[0] = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
      emxEnsureCapacity_boolean_T1(c_this, ixstart);
      loop_ub = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
      for (ixstart = 0; ixstart < loop_ub; ixstart++) {
        c_this->data[ixstart] = (b_this->BoardIdx->data[ixstart] < 0.0);
      }

      if (any(c_this)) {
        b_l = ((real32_T)rtInf);
      } else {
        for (ixstart = 0; ixstart < 2; ixstart++) {
          for (loop_ub = 0; loop_ub < 3; loop_ub++) {
            col1[loop_ub + 3 * ixstart] = b_this->Points->data[((int)
              b_this->BoardIdx->data[b_this->BoardIdx->size[0] * loop_ub] +
              b_this->Points->size[0] * ixstart) - 1];
          }
        }

        for (ixstart = 0; ixstart < 2; ixstart++) {
          for (loop_ub = 0; loop_ub < 3; loop_ub++) {
            col2[loop_ub + 3 * ixstart] = b_this->Points->data[((int)
              b_this->BoardIdx->data[1 + b_this->BoardIdx->size[0] * loop_ub] +
              b_this->Points->size[0] * ixstart) - 1];
          }
        }

        for (ixstart = 0; ixstart < 2; ixstart++) {
          for (loop_ub = 0; loop_ub < 3; loop_ub++) {
            row3[loop_ub + 3 * ixstart] = b_this->Points->data[((int)
              b_this->BoardIdx->data[2 + b_this->BoardIdx->size[0] * loop_ub] +
              b_this->Points->size[0] * ixstart) - 1];
          }
        }

        for (ixstart = 0; ixstart < 6; ixstart++) {
          boardSize = (col1[ixstart] + row3[ixstart]) - 2.0F * col2[ixstart];
          col1[ixstart] -= row3[ixstart];
          col2[ixstart] = boardSize;
        }

        for (loop_ub = 0; loop_ub < 3; loop_ub++) {
          z1[loop_ub] = rt_hypotf_snf(col2[loop_ub], col2[3 + loop_ub]) /
            rt_hypotf_snf(col1[loop_ub], col1[3 + loop_ub]);
        }

        ixstart = 1;
        boardSize = z1[0];
        if (rtIsNaNF(z1[0])) {
          loop_ub = 2;
          exitg1 = false;
          while ((!exitg1) && (loop_ub < 4)) {
            ixstart = loop_ub;
            if (!rtIsNaNF(z1[loop_ub - 1])) {
              boardSize = z1[loop_ub - 1];
              exitg1 = true;
            } else {
              loop_ub++;
            }
          }
        }

        if (ixstart < 3) {
          while (ixstart + 1 < 4) {
            if (z1[ixstart] > boardSize) {
              boardSize = z1[ixstart];
            }

            ixstart++;
          }
        }

        if ((0.0F > boardSize) || rtIsNaNF(boardSize)) {
          b_l = 0.0F;
        } else {
          b_l = boardSize;
        }

        for (ixstart = 0; ixstart < 2; ixstart++) {
          for (loop_ub = 0; loop_ub < 3; loop_ub++) {
            col1[loop_ub + 3 * ixstart] = b_this->Points->data[((int)
              b_this->BoardIdx->data[loop_ub] + b_this->Points->size[0] *
              ixstart) - 1];
          }
        }

        for (ixstart = 0; ixstart < 2; ixstart++) {
          for (loop_ub = 0; loop_ub < 3; loop_ub++) {
            col2[loop_ub + 3 * ixstart] = b_this->Points->data[((int)
              b_this->BoardIdx->data[loop_ub + b_this->BoardIdx->size[0]] +
              b_this->Points->size[0] * ixstart) - 1];
          }
        }

        for (ixstart = 0; ixstart < 2; ixstart++) {
          for (loop_ub = 0; loop_ub < 3; loop_ub++) {
            row3[loop_ub + 3 * ixstart] = b_this->Points->data[((int)
              b_this->BoardIdx->data[loop_ub + (b_this->BoardIdx->size[0] << 1)]
              + b_this->Points->size[0] * ixstart) - 1];
          }
        }

        for (ixstart = 0; ixstart < 6; ixstart++) {
          boardSize = (col1[ixstart] + row3[ixstart]) - 2.0F * col2[ixstart];
          col1[ixstart] -= row3[ixstart];
          col2[ixstart] = boardSize;
        }

        for (loop_ub = 0; loop_ub < 3; loop_ub++) {
          z1[loop_ub] = rt_hypotf_snf(col2[loop_ub], col2[3 + loop_ub]) /
            rt_hypotf_snf(col1[loop_ub], col1[3 + loop_ub]);
        }

        ixstart = 1;
        boardSize = z1[0];
        if (rtIsNaNF(z1[0])) {
          loop_ub = 2;
          exitg1 = false;
          while ((!exitg1) && (loop_ub < 4)) {
            ixstart = loop_ub;
            if (!rtIsNaNF(z1[loop_ub - 1])) {
              boardSize = z1[loop_ub - 1];
              exitg1 = true;
            } else {
              loop_ub++;
            }
          }
        }

        if (ixstart < 3) {
          while (ixstart + 1 < 4) {
            if (z1[ixstart] > boardSize) {
              boardSize = z1[ixstart];
            }

            ixstart++;
          }
        }

        if (!((b_l > boardSize) || rtIsNaNF(boardSize))) {
          b_l = boardSize;
        }

        boardSize = (float)(b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1]);
        b_l = boardSize * b_l - boardSize;
      }

      b_this->Energy = b_l;
      b_this->isValid = (b_this->Energy < -7.0F);
    }
  }

  emxFree_boolean_T(&c_this);
  emxFree_int32_T(&r5);
  emxFree_real32_T(&euclideanDists);
  emxFree_real32_T(&pointVectors);
}

static void algbwmorph(emxArray_boolean_T *bw)
{
  emxArray_boolean_T *last_aout;
  emxArray_boolean_T *m;
  int i34;
  int loop_ub;
  int i35;
  int i36;
  int i37;
  int i38;
  int i39;
  int b_loop_ub;
  int i40;
  int i41;
  int i42;
  int i43;
  int i44;
  emxInit_boolean_T(&last_aout, 2);
  emxInit_boolean_T(&m, 2);
  do {
    i34 = last_aout->size[0] * last_aout->size[1];
    last_aout->size[0] = bw->size[0];
    last_aout->size[1] = bw->size[1];
    emxEnsureCapacity_boolean_T(last_aout, i34);
    loop_ub = bw->size[0] * bw->size[1];
    for (i34 = 0; i34 < loop_ub; i34++) {
      last_aout->data[i34] = bw->data[i34];
    }

    bwlookup(bw, m);
    i34 = m->size[0] * m->size[1];
    m->size[0] = bw->size[0];
    m->size[1] = bw->size[1];
    emxEnsureCapacity_boolean_T(m, i34);
    loop_ub = bw->size[0] * bw->size[1];
    for (i34 = 0; i34 < loop_ub; i34++) {
      m->data[i34] = (bw->data[i34] && (!m->data[i34]));
    }

    if (1 > m->size[0]) {
      i34 = 1;
      i35 = -1;
    } else {
      i34 = 2;
      i35 = m->size[0] - 1;
    }

    if (1 > m->size[1]) {
      i36 = 1;
      i37 = -1;
    } else {
      i36 = 2;
      i37 = m->size[1] - 1;
    }

    if (1 > bw->size[0]) {
      i38 = 1;
    } else {
      i38 = 2;
    }

    if (1 > bw->size[1]) {
      i39 = 1;
    } else {
      i39 = 2;
    }

    loop_ub = div_s32_floor(i37, i36);
    for (i37 = 0; i37 <= loop_ub; i37++) {
      b_loop_ub = div_s32_floor(i35, i34);
      for (i40 = 0; i40 <= b_loop_ub; i40++) {
        bw->data[i38 * i40 + bw->size[0] * (i39 * i37)] = m->data[i34 * i40 +
          m->size[0] * (i36 * i37)];
      }
    }

    bwlookup(bw, m);
    i34 = m->size[0] * m->size[1];
    m->size[0] = bw->size[0];
    m->size[1] = bw->size[1];
    emxEnsureCapacity_boolean_T(m, i34);
    loop_ub = bw->size[0] * bw->size[1];
    for (i34 = 0; i34 < loop_ub; i34++) {
      m->data[i34] = (bw->data[i34] && (!m->data[i34]));
    }

    if (2 > m->size[0]) {
      i34 = 0;
      i35 = 1;
      i36 = 0;
    } else {
      i34 = 1;
      i35 = 2;
      i36 = m->size[0];
    }

    if (2 > m->size[1]) {
      i37 = 0;
      i38 = 1;
      i39 = 0;
    } else {
      i37 = 1;
      i38 = 2;
      i39 = m->size[1];
    }

    if (2 > bw->size[0]) {
      i40 = 0;
      i41 = 1;
    } else {
      i40 = 1;
      i41 = 2;
    }

    if (2 > bw->size[1]) {
      i42 = 0;
      i43 = 1;
    } else {
      i42 = 1;
      i43 = 2;
    }

    loop_ub = div_s32_floor((i39 - i37) - 1, i38);
    for (i39 = 0; i39 <= loop_ub; i39++) {
      b_loop_ub = div_s32_floor((i36 - i34) - 1, i35);
      for (i44 = 0; i44 <= b_loop_ub; i44++) {
        bw->data[(i40 + i41 * i44) + bw->size[0] * (i42 + i43 * i39)] = m->data
          [(i34 + i35 * i44) + m->size[0] * (i37 + i38 * i39)];
      }
    }

    bwlookup(bw, m);
    i34 = m->size[0] * m->size[1];
    m->size[0] = bw->size[0];
    m->size[1] = bw->size[1];
    emxEnsureCapacity_boolean_T(m, i34);
    loop_ub = bw->size[0] * bw->size[1];
    for (i34 = 0; i34 < loop_ub; i34++) {
      m->data[i34] = (bw->data[i34] && (!m->data[i34]));
    }

    if (1 > m->size[0]) {
      i34 = 1;
      i35 = -1;
    } else {
      i34 = 2;
      i35 = m->size[0] - 1;
    }

    if (2 > m->size[1]) {
      i36 = 0;
      i37 = 1;
      i38 = 0;
    } else {
      i36 = 1;
      i37 = 2;
      i38 = m->size[1];
    }

    if (1 > bw->size[0]) {
      i39 = 1;
    } else {
      i39 = 2;
    }

    if (2 > bw->size[1]) {
      i40 = 0;
      i41 = 1;
    } else {
      i40 = 1;
      i41 = 2;
    }

    loop_ub = div_s32_floor((i38 - i36) - 1, i37);
    for (i38 = 0; i38 <= loop_ub; i38++) {
      b_loop_ub = div_s32_floor(i35, i34);
      for (i42 = 0; i42 <= b_loop_ub; i42++) {
        bw->data[i39 * i42 + bw->size[0] * (i40 + i41 * i38)] = m->data[i34 *
          i42 + m->size[0] * (i36 + i37 * i38)];
      }
    }

    bwlookup(bw, m);
    i34 = m->size[0] * m->size[1];
    m->size[0] = bw->size[0];
    m->size[1] = bw->size[1];
    emxEnsureCapacity_boolean_T(m, i34);
    loop_ub = bw->size[0] * bw->size[1];
    for (i34 = 0; i34 < loop_ub; i34++) {
      m->data[i34] = (bw->data[i34] && (!m->data[i34]));
    }

    if (2 > m->size[0]) {
      i34 = 0;
      i35 = 1;
      i36 = 0;
    } else {
      i34 = 1;
      i35 = 2;
      i36 = m->size[0];
    }

    if (1 > m->size[1]) {
      i37 = 1;
      i38 = -1;
    } else {
      i37 = 2;
      i38 = m->size[1] - 1;
    }

    if (2 > bw->size[0]) {
      i39 = 0;
      i40 = 1;
    } else {
      i39 = 1;
      i40 = 2;
    }

    if (1 > bw->size[1]) {
      i41 = 1;
    } else {
      i41 = 2;
    }

    loop_ub = div_s32_floor(i38, i37);
    for (i38 = 0; i38 <= loop_ub; i38++) {
      b_loop_ub = div_s32_floor((i36 - i34) - 1, i35);
      for (i42 = 0; i42 <= b_loop_ub; i42++) {
        bw->data[(i39 + i40 * i42) + bw->size[0] * (i41 * i38)] = m->data[(i34 +
          i35 * i42) + m->size[0] * (i37 * i38)];
      }
    }
  } while (!isequal(last_aout, bw));

  emxFree_boolean_T(&m);
  emxFree_boolean_T(&last_aout);
}

static boolean_T any(const emxArray_boolean_T *x)
{
  boolean_T y;
  int ix;
  boolean_T exitg1;
  boolean_T b0;
  y = false;
  ix = 1;
  exitg1 = false;
  while ((!exitg1) && (ix <= x->size[0])) {
    b0 = !x->data[ix - 1];
    if (!b0) {
      y = true;
      exitg1 = true;
    } else {
      ix++;
    }
  }

  return y;
}

static void b_abs(const emxArray_real32_T *x, emxArray_real32_T *y)
{
  int nx;
  int k;
  unsigned int uv0[2];
  nx = x->size[0] * x->size[1];
  for (k = 0; k < 2; k++) {
    uv0[k] = (unsigned int)x->size[k];
  }

  k = y->size[0] * y->size[1];
  y->size[0] = (int)uv0[0];
  y->size[1] = (int)uv0[1];
  emxEnsureCapacity_real32_T(y, k);
  for (k = 0; k + 1 <= nx; k++) {
    y->data[k] = std::abs(x->data[k]);
  }
}

static void b_hypot(const emxArray_real32_T *x, const emxArray_real32_T *y,
                    emxArray_real32_T *r)
{
  int c;
  int k;
  if (x->size[0] <= y->size[0]) {
    c = x->size[0];
  } else {
    c = y->size[0];
  }

  k = r->size[0];
  r->size[0] = c;
  emxEnsureCapacity_real32_T1(r, k);
  for (k = 0; k + 1 <= c; k++) {
    r->data[k] = rt_hypotf_snf(x->data[k], y->data[k]);
  }
}

static void b_imfilter(emxArray_real32_T *varargin_1)
{
  int finalSize_idx_0;
  double pad[2];
  int finalSize_idx_1;
  emxArray_real32_T *a;
  emxArray_real_T *b_a;
  int cidx;
  int loop_ub;
  emxArray_real_T *result;
  int iv3[2];
  boolean_T b2;
  int cEnd;
  int cEnd1;
  int ma;
  int na;
  int firstColB;
  int i;
  int lastColB;
  int lastRowA;
  int lastRowB;
  int aidx;
  int lastColA;
  int k;
  int b_firstColB;
  int iC;
  int iA;
  int iB;
  finalSize_idx_0 = varargin_1->size[0];
  pad[0] = 0.0;
  finalSize_idx_1 = varargin_1->size[1];
  pad[1] = 1.0;
  if (!((varargin_1->size[0] == 0) || (varargin_1->size[1] == 0))) {
    emxInit_real32_T(&a, 2);
    emxInit_real_T1(&b_a, 2);
    padImage(varargin_1, pad, a);
    cidx = b_a->size[0] * b_a->size[1];
    b_a->size[0] = a->size[0];
    b_a->size[1] = a->size[1];
    emxEnsureCapacity_real_T(b_a, cidx);
    loop_ub = a->size[0] * a->size[1];
    for (cidx = 0; cidx < loop_ub; cidx++) {
      b_a->data[cidx] = a->data[cidx];
    }

    emxFree_real32_T(&a);
    for (cidx = 0; cidx < 2; cidx++) {
      iv3[cidx] = b_a->size[cidx];
    }

    emxInit_real_T1(&result, 2);
    cidx = result->size[0] * result->size[1];
    result->size[0] = iv3[0];
    result->size[1] = iv3[1];
    emxEnsureCapacity_real_T(result, cidx);
    loop_ub = iv3[0] * iv3[1];
    for (cidx = 0; cidx < loop_ub; cidx++) {
      result->data[cidx] = 0.0;
    }

    if ((b_a->size[0] == 0) || (b_a->size[1] == 0) || ((iv3[0] == 0) || (iv3[1] ==
          0))) {
      b2 = true;
    } else {
      b2 = false;
    }

    if (!b2) {
      cEnd = iv3[1];
      cEnd1 = iv3[0];
      ma = b_a->size[0];
      na = b_a->size[1] - 1;
      if (b_a->size[1] < 1) {
        firstColB = 2;
      } else {
        firstColB = 0;
      }

      if (3 <= iv3[1]) {
        lastColB = 2;
      } else {
        lastColB = iv3[1];
      }

      if (1 <= iv3[0] - 1) {
        lastRowB = 1;
      } else {
        lastRowB = iv3[0];
      }

      while (firstColB <= lastColB) {
        if (firstColB + na < cEnd) {
          lastColA = na;
        } else {
          lastColA = cEnd - firstColB;
        }

        for (k = (firstColB < 1); k <= lastColA; k++) {
          if (firstColB + k > 1) {
            b_firstColB = (firstColB + k) - 1;
          } else {
            b_firstColB = 0;
          }

          iC = b_firstColB * cEnd1;
          iA = k * ma;
          iB = firstColB;
          i = 0;
          while (i <= lastRowB - 1) {
            if (ma <= cEnd1 - 1) {
              lastRowA = ma;
            } else {
              lastRowA = cEnd1;
            }

            aidx = iA;
            cidx = iC;
            for (loop_ub = 1; loop_ub <= lastRowA; loop_ub++) {
              result->data[cidx] += (-1.0 + (double)iB) * b_a->data[aidx];
              aidx++;
              cidx++;
            }

            iB++;
            iC++;
            i = 1;
          }
        }

        firstColB++;
      }
    }

    emxFree_real_T(&b_a);
    if (1 > finalSize_idx_0) {
      loop_ub = 0;
    } else {
      loop_ub = finalSize_idx_0;
    }

    if (2.0 > (double)finalSize_idx_1 + 1.0) {
      cidx = 0;
      i = 0;
    } else {
      cidx = 1;
      i = finalSize_idx_1 + 1;
    }

    lastRowA = varargin_1->size[0] * varargin_1->size[1];
    varargin_1->size[0] = loop_ub;
    varargin_1->size[1] = i - cidx;
    emxEnsureCapacity_real32_T(varargin_1, lastRowA);
    aidx = i - cidx;
    for (i = 0; i < aidx; i++) {
      for (lastRowA = 0; lastRowA < loop_ub; lastRowA++) {
        varargin_1->data[lastRowA + varargin_1->size[0] * i] = (float)
          result->data[lastRowA + result->size[0] * (cidx + i)];
      }
    }

    emxFree_real_T(&result);
  }
}

static void b_merge(int idx[4], double x[4], int offset, int np, int nq, int
                    iwork[4], double xwork[4])
{
  int n;
  int qend;
  int p;
  int iout;
  int exitg1;
  if (nq != 0) {
    n = np + nq;
    for (qend = 0; qend + 1 <= n; qend++) {
      iwork[qend] = idx[offset + qend];
      xwork[qend] = x[offset + qend];
    }

    p = 0;
    n = np;
    qend = np + nq;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork[p] <= xwork[n]) {
        idx[iout] = iwork[p];
        x[iout] = xwork[p];
        if (p + 1 < np) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx[iout] = iwork[n];
        x[iout] = xwork[n];
        if (n + 1 < qend) {
          n++;
        } else {
          n = (iout - p) + 1;
          while (p + 1 <= np) {
            idx[n + p] = iwork[p];
            x[n + p] = xwork[p];
            p++;
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

static double b_mod(double x)
{
  double r;
  if ((!rtIsInf(x)) && (!rtIsNaN(x))) {
    if (x == 0.0) {
      r = 0.0;
    } else {
      r = std::fmod(x, 2.0);
      if (r == 0.0) {
        r = 0.0;
      }
    }
  } else {
    r = rtNaN;
  }

  return r;
}

static void b_rot90(const emxArray_real_T *A, emxArray_real_T *B)
{
  int m;
  int n;
  int j;
  int i;
  int A_idx_0;
  int B_idx_0;
  m = A->size[0];
  n = A->size[1];
  j = B->size[0] * B->size[1];
  B->size[0] = A->size[0];
  B->size[1] = A->size[1];
  emxEnsureCapacity_real_T(B, j);
  for (j = 1; j <= n; j++) {
    for (i = 1; i <= m; i++) {
      A_idx_0 = A->size[0];
      B_idx_0 = B->size[0];
      B->data[(i + B_idx_0 * (j - 1)) - 1] = A->data[(m - i) + A_idx_0 * (n - j)];
    }
  }
}

static void b_sort(emxArray_real32_T *x, int dim, emxArray_int32_T *idx)
{
  int i45;
  emxArray_real32_T *vwork;
  int vstride;
  int x_idx_0;
  int j;
  emxArray_int32_T *iidx;
  if (dim <= 1) {
    i45 = x->size[0];
  } else {
    i45 = 1;
  }

  emxInit_real32_T1(&vwork, 1);
  vstride = vwork->size[0];
  vwork->size[0] = i45;
  emxEnsureCapacity_real32_T1(vwork, vstride);
  x_idx_0 = x->size[0];
  vstride = idx->size[0];
  idx->size[0] = x_idx_0;
  emxEnsureCapacity_int32_T(idx, vstride);
  vstride = 1;
  x_idx_0 = 1;
  while (x_idx_0 <= dim - 1) {
    vstride *= x->size[0];
    x_idx_0 = 2;
  }

  j = 0;
  emxInit_int32_T(&iidx, 1);
  while (j + 1 <= vstride) {
    for (x_idx_0 = 0; x_idx_0 + 1 <= i45; x_idx_0++) {
      vwork->data[x_idx_0] = x->data[j + x_idx_0 * vstride];
    }

    sortIdx(vwork, iidx);
    for (x_idx_0 = 0; x_idx_0 + 1 <= i45; x_idx_0++) {
      x->data[j + x_idx_0 * vstride] = vwork->data[x_idx_0];
      idx->data[j + x_idx_0 * vstride] = iidx->data[x_idx_0];
    }

    j++;
  }

  emxFree_int32_T(&iidx);
  emxFree_real32_T(&vwork);
}

static void b_squeeze(const emxArray_real_T *a, emxArray_real_T *b)
{
  int k;
  int i21;
  int sqsz[3];
  k = 3;
  while ((k > 2) && (a->size[2] == 1)) {
    k = 2;
  }

  if (k <= 2) {
    sqsz[0] = a->size[0];
    i21 = b->size[0] * b->size[1];
    b->size[0] = sqsz[0];
    b->size[1] = 1;
    emxEnsureCapacity_real_T(b, i21);
    i21 = a->size[0] * a->size[2];
    for (k = 0; k + 1 <= i21; k++) {
      b->data[k] = a->data[k];
    }
  } else {
    for (i21 = 0; i21 < 3; i21++) {
      sqsz[i21] = 1;
    }

    k = 0;
    if (a->size[0] != 1) {
      sqsz[0] = a->size[0];
      k = 1;
    }

    if (a->size[2] != 1) {
      sqsz[k] = a->size[2];
    }

    i21 = b->size[0] * b->size[1];
    b->size[0] = sqsz[0];
    b->size[1] = sqsz[1];
    emxEnsureCapacity_real_T(b, i21);
    i21 = a->size[0] * a->size[2];
    for (k = 0; k + 1 <= i21; k++) {
      b->data[k] = a->data[k];
    }
  }
}

static void bsxfun(const emxArray_real32_T *a, const float b[2],
                   emxArray_real32_T *c)
{
  int csz_idx_0;
  int k;
  int szc;
  int b_k;
  csz_idx_0 = a->size[0];
  k = c->size[0] * c->size[1];
  c->size[0] = csz_idx_0;
  c->size[1] = 2;
  emxEnsureCapacity_real32_T(c, k);
  if (c->size[0] != 0) {
    csz_idx_0 = (a->size[0] != 1);
    for (k = 0; k < 2; k++) {
      szc = c->size[0];
      for (b_k = 0; b_k < szc; b_k++) {
        c->data[b_k + c->size[0] * k] = a->data[csz_idx_0 * b_k + a->size[0] * k]
          - b[k];
      }
    }
  }
}

static void bwlookup(const emxArray_boolean_T *bwin, emxArray_boolean_T *B)
{
  emxArray_boolean_T *bw;
  int rowInd;
  int loop_ub;
  unsigned int inDims[2];
  static const boolean_T lut[512] = { false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    true, true, true, true, false, true, true, true, true, true, true, false,
    false, true, true, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, true, false, true,
    true, true, false, true, true, false, false, true, true, false, false, true,
    true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, true, false, false, false, false,
    false, false, false, true, true, true, true, false, false, true, true, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true, true, false, false, true, true, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true, false, false, false, false, false, false, false,
    true, true, true, true, false, false, true, true, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, true, false, true, true, true, false, true, true, true, true, false,
    false, true, true, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,
    false, false, false, false, false, false, false, true, true, true, true,
    false, false, true, true, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true, false,
    true, true, true, false, true, true, true, true, false, false, true, true,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, true, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, true, false, true, true, true,
    false, true, true, false, false, true, true, false, false, true, true, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, true, true, false, false, true, true, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, true, false, false, false, false, false, false, false, true,
    true, true, true, false, false, true, true, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, true, false, true, true, true, false, true, true, true, true, false,
    false, true, true, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,
    false, false, false, false, false, false, false, true, true, true, true,
    false, false, true, true, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true, false,
    true, true, true, false, true, true, true, true, false, false, true, true,
    false, false };

  int lookUpInd;
  int colInd;
  int b_rowInd;
  emxInit_boolean_T(&bw, 2);
  rowInd = bw->size[0] * bw->size[1];
  bw->size[0] = bwin->size[0];
  bw->size[1] = bwin->size[1];
  emxEnsureCapacity_boolean_T(bw, rowInd);
  loop_ub = bwin->size[0] * bwin->size[1];
  for (rowInd = 0; rowInd < loop_ub; rowInd++) {
    bw->data[rowInd] = bwin->data[rowInd];
  }

  for (rowInd = 0; rowInd < 2; rowInd++) {
    inDims[rowInd] = (unsigned int)bwin->size[rowInd];
  }

  rowInd = B->size[0] * B->size[1];
  B->size[0] = (int)inDims[0];
  B->size[1] = (int)inDims[1];
  emxEnsureCapacity_boolean_T(B, rowInd);
  if (!((bwin->size[0] == 0) || (bwin->size[1] == 0))) {
    for (rowInd = 0; rowInd < 2; rowInd++) {
      inDims[rowInd] = (unsigned int)bwin->size[rowInd];
    }

    if (((int)inDims[0] >= 2) && ((int)inDims[1] >= 2)) {
      B->data[0] = lut[(((bwin->data[0] << 4) + (bwin->data[1] << 5)) +
                        (bwin->data[bwin->size[0]] << 7)) + (bwin->data[1 +
        bwin->size[0]] << 8)];
      for (rowInd = 0; rowInd <= (int)inDims[0] - 3; rowInd++) {
        B->data[rowInd + 1] = lut[(((((bwin->data[rowInd] << 3) + (bwin->
          data[rowInd + 1] << 4)) + (bwin->data[rowInd + 2] << 5)) + (bwin->
          data[rowInd + bwin->size[0]] << 6)) + (bwin->data[(rowInd + bwin->
          size[0]) + 1] << 7)) + (bwin->data[(rowInd + bwin->size[0]) + 2] << 8)];
      }

      lookUpInd = ((((bwin->data[(int)inDims[0] - 2] << 3) + (bwin->data[(int)
        inDims[0] - 1] << 4)) + (bwin->data[((int)inDims[0] + bwin->size[0]) - 2]
        << 6)) + (bwin->data[((int)inDims[0] + bwin->size[0]) - 1] << 7)) + 1;
      B->data[(int)inDims[0] - 1] = lut[lookUpInd - 1];
      loop_ub = (int)inDims[1] - 3;

#pragma omp parallel for \
 num_threads(omp_get_max_threads()) \
 private(lookUpInd,b_rowInd)

      for (colInd = 0; colInd <= loop_ub; colInd++) {
        for (b_rowInd = 0; b_rowInd <= (int)inDims[0] - 3; b_rowInd++) {
          lookUpInd = ((((((((bw->data[b_rowInd + bw->size[0] * colInd] +
                              (bw->data[(b_rowInd + bw->size[0] * colInd) + 1] <<
                               1)) + (bw->data[(b_rowInd + bw->size[0] * colInd)
            + 2] << 2)) + (bw->data[b_rowInd + bw->size[0] * (colInd + 1)] << 3))
                           + (bw->data[(b_rowInd + bw->size[0] * (colInd + 1)) +
                              1] << 4)) + (bw->data[(b_rowInd + bw->size[0] *
            (colInd + 1)) + 2] << 5)) + (bw->data[b_rowInd + bw->size[0] *
            (colInd + 2)] << 6)) + (bw->data[(b_rowInd + bw->size[0] * (colInd +
            2)) + 1] << 7)) + (bw->data[(b_rowInd + bw->size[0] * (colInd + 2))
                               + 2] << 8)) + 1;
          B->data[(b_rowInd + B->size[0] * (1 + colInd)) + 1] = lut[lookUpInd -
            1];
        }
      }

      for (loop_ub = 1; loop_ub - 1 <= (int)inDims[1] - 3; loop_ub++) {
        B->data[B->size[0] * loop_ub] = lut[(((((bw->data[bw->size[0] * (loop_ub
          - 1)] << 1) + (bw->data[1 + bw->size[0] * (loop_ub - 1)] << 2)) +
          (bw->data[bw->size[0] * loop_ub] << 4)) + (bw->data[1 + bw->size[0] *
          loop_ub] << 5)) + (bw->data[bw->size[0] * (loop_ub + 1)] << 7)) +
          (bw->data[1 + bw->size[0] * (loop_ub + 1)] << 8)];
        B->data[((int)inDims[0] + B->size[0] * loop_ub) - 1] = lut[((((bw->data
          [((int)inDims[0] + bw->size[0] * (loop_ub - 1)) - 2] + (bw->data[((int)
          inDims[0] + bw->size[0] * (loop_ub - 1)) - 1] << 1)) + (bw->data[((int)
          inDims[0] + bw->size[0] * loop_ub) - 2] << 3)) + (bw->data[((int)
          inDims[0] + bw->size[0] * loop_ub) - 1] << 4)) + (bw->data[((int)
          inDims[0] + bw->size[0] * (loop_ub + 1)) - 2] << 6)) + (bw->data[((int)
          inDims[0] + bw->size[0] * (loop_ub + 1)) - 1] << 7)];
      }

      loop_ub = (int)inDims[1] - 1;
      B->data[B->size[0] * ((int)inDims[1] - 1)] = lut[(((bw->data[bw->size[0] *
        ((int)inDims[1] - 2)] << 1) + (bw->data[1 + bw->size[0] * ((int)inDims[1]
        - 2)] << 2)) + (bw->data[bw->size[0] * ((int)inDims[1] - 1)] << 4)) +
        (bw->data[1 + bw->size[0] * ((int)inDims[1] - 1)] << 5)];
      for (rowInd = 0; rowInd <= (int)inDims[0] - 3; rowInd++) {
        B->data[(rowInd + B->size[0] * loop_ub) + 1] = lut[((((bw->data[rowInd +
          bw->size[0] * (loop_ub - 1)] + (bw->data[(rowInd + bw->size[0] *
          (loop_ub - 1)) + 1] << 1)) + (bw->data[(rowInd + bw->size[0] *
          (loop_ub - 1)) + 2] << 2)) + (bw->data[rowInd + bw->size[0] * loop_ub]
          << 3)) + (bw->data[(rowInd + bw->size[0] * loop_ub) + 1] << 4)) +
          (bw->data[(rowInd + bw->size[0] * loop_ub) + 2] << 5)];
      }

      B->data[((int)inDims[0] + B->size[0] * ((int)inDims[1] - 1)) - 1] = lut
        [((bw->data[((int)inDims[0] + bw->size[0] * ((int)inDims[1] - 2)) - 2] +
           (bw->data[((int)inDims[0] + bw->size[0] * ((int)inDims[1] - 2)) - 1] <<
            1)) + (bw->data[((int)inDims[0] + bw->size[0] * ((int)inDims[1] - 1))
                   - 2] << 3)) + (bw->data[((int)inDims[0] + bw->size[0] * ((int)
        inDims[1] - 1)) - 1] << 4)];
    } else {
      if ((int)inDims[0] == (int)inDims[1]) {
        B->data[0] = lut[bwin->data[0] << 4];
      }

      if ((int)inDims[0] > 1) {
        B->data[0] = lut[(bwin->data[0] << 4) + (bwin->data[1] << 5)];
        for (rowInd = 0; rowInd <= (int)inDims[0] - 3; rowInd++) {
          B->data[rowInd + 1] = lut[((bwin->data[rowInd] << 3) + (bwin->
            data[rowInd + 1] << 4)) + (bwin->data[rowInd + 2] << 5)];
        }

        B->data[(int)inDims[0] - 1] = lut[(bwin->data[(int)inDims[0] - 2] << 3)
          + (bwin->data[(int)inDims[0] - 1] << 4)];
      }

      if ((int)inDims[1] > 1) {
        B->data[0] = lut[(bwin->data[0] << 4) + (bwin->data[bwin->size[0]] << 7)];
        for (loop_ub = 0; loop_ub <= (int)inDims[1] - 3; loop_ub++) {
          B->data[B->size[0] * (loop_ub + 1)] = lut[((bwin->data[bwin->size[0] *
            loop_ub] << 1) + (bwin->data[bwin->size[0] * (loop_ub + 1)] << 4)) +
            (bwin->data[bwin->size[0] * (loop_ub + 2)] << 7)];
        }

        B->data[B->size[0] * ((int)inDims[1] - 1)] = lut[(bwin->data[bwin->size
          [0] * ((int)inDims[1] - 2)] << 1) + (bwin->data[bwin->size[0] * ((int)
          inDims[1] - 1)] << 4)];
      }
    }
  }

  emxFree_boolean_T(&bw);
}

static float c_Checkerboard_computeNewEnergy(const
  c_vision_internal_calibration_c *b_this, float oldEnergy)
{
  float newEnergy;
  emxArray_real_T *r20;
  int loop_ub;
  int ixstart;
  int i16;
  emxArray_real_T *r21;
  int n;
  emxArray_real_T *b;
  emxArray_real_T *r22;
  emxArray_real_T *num;
  emxArray_real_T *denom;
  emxArray_real_T *b_num;
  emxArray_real_T *c_num;
  emxArray_real_T *r23;
  emxArray_real_T *r24;
  double mtmp;
  boolean_T exitg1;
  int i;
  emxArray_real_T *d_num;
  emxArray_real_T *b_denom;
  emxArray_real_T *r25;
  int e_num[1];
  int c_denom[1];
  emxArray_real_T f_num;
  emxArray_real_T d_denom;
  double z;
  emxInit_real_T(&r20, 3);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  i16 = r20->size[0] * r20->size[1] * r20->size[2];
  r20->size[0] = 1;
  r20->size[1] = loop_ub;
  r20->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r20, i16);
  for (i16 = 0; i16 < ixstart; i16++) {
    for (n = 0; n < loop_ub; n++) {
      r20->data[r20->size[0] * n + r20->size[0] * r20->size[1] * i16] =
        b_this->BoardCoords->data[b_this->BoardCoords->size[0] * n +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16];
    }
  }

  emxInit_real_T(&r21, 3);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  i16 = r21->size[0] * r21->size[1] * r21->size[2];
  r21->size[0] = 1;
  r21->size[1] = loop_ub;
  r21->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r21, i16);
  for (i16 = 0; i16 < ixstart; i16++) {
    for (n = 0; n < loop_ub; n++) {
      r21->data[r21->size[0] * n + r21->size[0] * r21->size[1] * i16] =
        b_this->BoardCoords->data[(b_this->BoardCoords->size[0] * n +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16) + 2];
    }
  }

  emxInit_real_T(&b, 3);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  i16 = b->size[0] * b->size[1] * b->size[2];
  b->size[0] = 1;
  b->size[1] = loop_ub;
  b->size[2] = ixstart;
  emxEnsureCapacity_real_T1(b, i16);
  for (i16 = 0; i16 < ixstart; i16++) {
    for (n = 0; n < loop_ub; n++) {
      b->data[b->size[0] * n + b->size[0] * b->size[1] * i16] =
        b_this->BoardCoords->data[(b_this->BoardCoords->size[0] * n +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16) + 1];
    }
  }

  emxInit_real_T(&r22, 3);
  i16 = r22->size[0] * r22->size[1] * r22->size[2];
  r22->size[0] = 1;
  r22->size[1] = r20->size[1];
  r22->size[2] = r20->size[2];
  emxEnsureCapacity_real_T1(r22, i16);
  loop_ub = r20->size[0] * r20->size[1] * r20->size[2];
  for (i16 = 0; i16 < loop_ub; i16++) {
    r22->data[i16] = (r20->data[i16] + r21->data[i16]) - 2.0 * b->data[i16];
  }

  emxFree_real_T(&b);
  emxInit_real_T1(&num, 2);
  squeeze(r22, num);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  i16 = r20->size[0] * r20->size[1] * r20->size[2];
  r20->size[0] = 1;
  r20->size[1] = loop_ub;
  r20->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r20, i16);
  for (i16 = 0; i16 < ixstart; i16++) {
    for (n = 0; n < loop_ub; n++) {
      r20->data[r20->size[0] * n + r20->size[0] * r20->size[1] * i16] =
        b_this->BoardCoords->data[b_this->BoardCoords->size[0] * n +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16];
    }
  }

  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  i16 = r21->size[0] * r21->size[1] * r21->size[2];
  r21->size[0] = 1;
  r21->size[1] = loop_ub;
  r21->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r21, i16);
  for (i16 = 0; i16 < ixstart; i16++) {
    for (n = 0; n < loop_ub; n++) {
      r21->data[r21->size[0] * n + r21->size[0] * r21->size[1] * i16] =
        b_this->BoardCoords->data[(b_this->BoardCoords->size[0] * n +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16) + 2];
    }
  }

  i16 = r22->size[0] * r22->size[1] * r22->size[2];
  r22->size[0] = 1;
  r22->size[1] = r20->size[1];
  r22->size[2] = r20->size[2];
  emxEnsureCapacity_real_T1(r22, i16);
  loop_ub = r20->size[0] * r20->size[1] * r20->size[2];
  for (i16 = 0; i16 < loop_ub; i16++) {
    r22->data[i16] = r20->data[i16] - r21->data[i16];
  }

  emxFree_real_T(&r21);
  emxFree_real_T(&r20);
  emxInit_real_T1(&denom, 2);
  emxInit_real_T2(&b_num, 1);
  squeeze(r22, denom);
  loop_ub = num->size[0];
  i16 = b_num->size[0];
  b_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(b_num, i16);
  emxFree_real_T(&r22);
  for (i16 = 0; i16 < loop_ub; i16++) {
    b_num->data[i16] = num->data[i16];
  }

  emxInit_real_T2(&c_num, 1);
  loop_ub = num->size[0];
  i16 = c_num->size[0];
  c_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(c_num, i16);
  for (i16 = 0; i16 < loop_ub; i16++) {
    c_num->data[i16] = num->data[i16 + num->size[0]];
  }

  emxFree_real_T(&num);
  emxInit_real_T2(&r23, 1);
  c_hypot(b_num, c_num, r23);
  loop_ub = denom->size[0];
  i16 = b_num->size[0];
  b_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(b_num, i16);
  for (i16 = 0; i16 < loop_ub; i16++) {
    b_num->data[i16] = denom->data[i16];
  }

  loop_ub = denom->size[0];
  i16 = c_num->size[0];
  c_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(c_num, i16);
  for (i16 = 0; i16 < loop_ub; i16++) {
    c_num->data[i16] = denom->data[i16 + denom->size[0]];
  }

  emxFree_real_T(&denom);
  emxInit_real_T2(&r24, 1);
  c_hypot(b_num, c_num, r24);
  rdivide(r23, r24, b_num);
  ixstart = 1;
  n = b_num->size[0];
  mtmp = b_num->data[0];
  emxFree_real_T(&c_num);
  emxFree_real_T(&r24);
  emxFree_real_T(&r23);
  if (b_num->size[0] > 1) {
    if (rtIsNaN(b_num->data[0])) {
      loop_ub = 2;
      exitg1 = false;
      while ((!exitg1) && (loop_ub <= n)) {
        ixstart = loop_ub;
        if (!rtIsNaN(b_num->data[loop_ub - 1])) {
          mtmp = b_num->data[loop_ub - 1];
          exitg1 = true;
        } else {
          loop_ub++;
        }
      }
    }

    if (ixstart < b_num->size[0]) {
      while (ixstart + 1 <= n) {
        if (b_num->data[ixstart] > mtmp) {
          mtmp = b_num->data[ixstart];
        }

        ixstart++;
      }
    }
  }

  emxFree_real_T(&b_num);
  if ((oldEnergy > mtmp) || rtIsNaN(mtmp)) {
    newEnergy = oldEnergy;
  } else {
    newEnergy = (float)mtmp;
  }

  mtmp = (double)b_this->BoardCoords->size[1] - 2.0;
  i = 0;
  emxInit_real_T(&d_num, 3);
  emxInit_real_T(&b_denom, 3);
  emxInit_real_T(&r25, 3);
  while (i <= (int)mtmp - 1) {
    loop_ub = b_this->BoardCoords->size[2];
    i16 = d_num->size[0] * d_num->size[1] * d_num->size[2];
    d_num->size[0] = 1;
    d_num->size[1] = 1;
    d_num->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(d_num, i16);
    for (i16 = 0; i16 < loop_ub; i16++) {
      d_num->data[d_num->size[0] * d_num->size[1] * i16] = b_this->
        BoardCoords->data[b_this->BoardCoords->size[0] * i + b_this->
        BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i16 = r25->size[0] * r25->size[1] * r25->size[2];
    r25->size[0] = 1;
    r25->size[1] = 1;
    r25->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(r25, i16);
    for (i16 = 0; i16 < loop_ub; i16++) {
      r25->data[r25->size[0] * r25->size[1] * i16] = b_this->BoardCoords->
        data[b_this->BoardCoords->size[0] * ((int)((1.0 + (double)i) + 2.0) - 1)
        + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i16 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    b_denom->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(b_denom, i16);
    for (i16 = 0; i16 < loop_ub; i16++) {
      b_denom->data[b_denom->size[0] * b_denom->size[1] * i16] =
        b_this->BoardCoords->data[b_this->BoardCoords->size[0] * ((int)((1.0 +
        (double)i) + 1.0) - 1) + b_this->BoardCoords->size[0] *
        b_this->BoardCoords->size[1] * i16];
    }

    i16 = d_num->size[0] * d_num->size[1] * d_num->size[2];
    d_num->size[0] = 1;
    d_num->size[1] = 1;
    emxEnsureCapacity_real_T1(d_num, i16);
    ixstart = d_num->size[0];
    n = d_num->size[1];
    loop_ub = d_num->size[2];
    loop_ub *= ixstart * n;
    for (i16 = 0; i16 < loop_ub; i16++) {
      d_num->data[i16] = (d_num->data[i16] + r25->data[i16]) - 2.0 *
        b_denom->data[i16];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i16 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    b_denom->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(b_denom, i16);
    for (i16 = 0; i16 < loop_ub; i16++) {
      b_denom->data[b_denom->size[0] * b_denom->size[1] * i16] =
        b_this->BoardCoords->data[b_this->BoardCoords->size[0] * i +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i16 = r25->size[0] * r25->size[1] * r25->size[2];
    r25->size[0] = 1;
    r25->size[1] = 1;
    r25->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(r25, i16);
    for (i16 = 0; i16 < loop_ub; i16++) {
      r25->data[r25->size[0] * r25->size[1] * i16] = b_this->BoardCoords->
        data[b_this->BoardCoords->size[0] * ((int)((1.0 + (double)i) + 2.0) - 1)
        + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i16];
    }

    i16 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    emxEnsureCapacity_real_T1(b_denom, i16);
    ixstart = b_denom->size[0];
    n = b_denom->size[1];
    loop_ub = b_denom->size[2];
    loop_ub *= ixstart * n;
    for (i16 = 0; i16 < loop_ub; i16++) {
      b_denom->data[i16] -= r25->data[i16];
    }

    e_num[0] = d_num->size[2];
    c_denom[0] = b_denom->size[2];
    f_num = *d_num;
    f_num.size = (int *)&e_num;
    f_num.numDimensions = 1;
    d_denom = *b_denom;
    d_denom.size = (int *)&c_denom;
    d_denom.numDimensions = 1;
    z = norm(&f_num) / norm(&d_denom);
    if ((newEnergy > z) || rtIsNaN(z)) {
    } else {
      newEnergy = (float)z;
    }

    i++;
  }

  emxFree_real_T(&r25);
  emxFree_real_T(&b_denom);
  emxFree_real_T(&d_num);
  ixstart = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  return newEnergy * (float)ixstart - (float)(b_this->BoardIdx->size[0] *
    b_this->BoardIdx->size[1]);
}

static void c_Checkerboard_expandBoardDirec(c_vision_internal_calibration_c
  *b_this, double direction)
{
  float oldEnergy;
  int numRows;
  emxArray_real_T *predictedPoints;
  emxArray_real_T *newIndices;
  emxArray_real_T *r9;
  emxArray_real_T *idx;
  emxArray_real_T *p2;
  emxArray_real_T *b_p2;
  emxArray_real_T *c_this;
  int loop_ub;
  int n;
  int i8;
  emxArray_real_T *d_this;
  int i;
  int b_idx;
  emxArray_int32_T *r10;
  emxArray_int32_T *r11;
  emxArray_real_T *r12;
  emxArray_real_T *r13;
  emxArray_real_T *b;
  emxArray_real_T *b_predictedPoints;
  emxArray_real_T *c_predictedPoints;
  emxArray_real_T *r14;
  emxArray_real_T *r15;
  double mtmp;
  boolean_T exitg1;
  emxArray_real_T *num;
  emxArray_real_T *denom;
  emxArray_real_T *r16;
  int b_num[1];
  int b_denom[1];
  emxArray_real_T c_num;
  emxArray_real_T c_denom;
  double z;
  oldEnergy = b_this->Energy + (float)(b_this->BoardIdx->size[0] *
    b_this->BoardIdx->size[1]);
  numRows = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  oldEnergy /= (float)numRows;
  emxInit_real_T1(&predictedPoints, 2);
  emxInit_real_T1(&newIndices, 2);
  emxInit_real_T(&r9, 3);
  emxInit_real_T1(&idx, 2);
  emxInit_real_T1(&p2, 2);
  emxInit_real_T1(&b_p2, 2);
  emxInit_real_T(&c_this, 3);
  switch ((int)direction) {
   case 1:
    c_Checkerboard_predictPointsVer(b_this, predictedPoints);
    Checkerboard_findClosestIndices(b_this, predictedPoints, newIndices);
    Checkerboard_expandBoardUp(b_this, newIndices, predictedPoints, r9);
    i8 = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    b_this->BoardIdx->size[0] = predictedPoints->size[0];
    b_this->BoardIdx->size[1] = predictedPoints->size[1];
    emxEnsureCapacity_real_T(b_this->BoardIdx, i8);
    loop_ub = predictedPoints->size[0] * predictedPoints->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardIdx->data[i8] = predictedPoints->data[i8];
    }

    i8 = b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
      b_this->BoardCoords->size[2];
    b_this->BoardCoords->size[0] = r9->size[0];
    b_this->BoardCoords->size[1] = r9->size[1];
    b_this->BoardCoords->size[2] = r9->size[2];
    emxEnsureCapacity_real_T1(b_this->BoardCoords, i8);
    loop_ub = r9->size[0] * r9->size[1] * r9->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardCoords->data[i8] = r9->data[i8];
    }

    oldEnergy = c_Checkerboard_computeNewEnergy(b_this, oldEnergy);
    break;

   case 2:
    numRows = b_this->BoardCoords->size[0];
    if (numRows < (double)numRows - 2.0) {
      i8 = idx->size[0] * idx->size[1];
      idx->size[0] = 1;
      idx->size[1] = 0;
      emxEnsureCapacity_real_T(idx, i8);
    } else {
      i8 = idx->size[0] * idx->size[1];
      idx->size[0] = 1;
      idx->size[1] = (int)-(((double)numRows - 2.0) - (double)numRows) + 1;
      emxEnsureCapacity_real_T(idx, i8);
      loop_ub = (int)-(((double)numRows - 2.0) - (double)numRows);
      for (i8 = 0; i8 <= loop_ub; i8++) {
        idx->data[idx->size[0] * i8] = (double)numRows - (double)i8;
      }
    }

    emxInit_real_T(&d_this, 3);
    loop_ub = b_this->BoardCoords->size[1];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[1];
    i8 = d_this->size[0] * d_this->size[1] * d_this->size[2];
    d_this->size[0] = 1;
    d_this->size[1] = loop_ub;
    d_this->size[2] = n;
    emxEnsureCapacity_real_T1(d_this, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        d_this->data[d_this->size[0] * i + d_this->size[0] * d_this->size[1] *
          i8] = b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0]
          * i) + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
          i8) - 1];
      }
    }

    squeeze(d_this, predictedPoints);
    loop_ub = b_this->BoardCoords->size[1];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i8 = d_this->size[0] * d_this->size[1] * d_this->size[2];
    d_this->size[0] = 1;
    d_this->size[1] = loop_ub;
    d_this->size[2] = n;
    emxEnsureCapacity_real_T1(d_this, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        d_this->data[d_this->size[0] * i + d_this->size[0] * d_this->size[1] *
          i8] = b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0]
          * i) + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
          i8) - 1];
      }
    }

    squeeze(d_this, p2);
    i8 = b_p2->size[0] * b_p2->size[1];
    b_p2->size[0] = p2->size[0];
    b_p2->size[1] = p2->size[1];
    emxEnsureCapacity_real_T(b_p2, i8);
    loop_ub = p2->size[0] * p2->size[1];
    emxFree_real_T(&d_this);
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_p2->data[i8] = (p2->data[i8] + p2->data[i8]) - predictedPoints->data[i8];
    }

    Checkerboard_findClosestIndices(b_this, b_p2, newIndices);
    Checkerboard_expandBoardDown(b_this, newIndices, predictedPoints, r9);
    i8 = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    b_this->BoardIdx->size[0] = predictedPoints->size[0];
    b_this->BoardIdx->size[1] = predictedPoints->size[1];
    emxEnsureCapacity_real_T(b_this->BoardIdx, i8);
    loop_ub = predictedPoints->size[0] * predictedPoints->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardIdx->data[i8] = predictedPoints->data[i8];
    }

    i8 = b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
      b_this->BoardCoords->size[2];
    b_this->BoardCoords->size[0] = r9->size[0];
    b_this->BoardCoords->size[1] = r9->size[1];
    b_this->BoardCoords->size[2] = r9->size[2];
    emxEnsureCapacity_real_T1(b_this->BoardCoords, i8);
    loop_ub = r9->size[0] * r9->size[1] * r9->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardCoords->data[i8] = r9->data[i8];
    }

    i8 = idx->size[0] * idx->size[1];
    idx->size[0] = 1;
    emxEnsureCapacity_real_T(idx, i8);
    b_idx = idx->size[0];
    numRows = idx->size[1];
    loop_ub = b_idx * numRows;
    for (i8 = 0; i8 < loop_ub; i8++) {
      idx->data[i8]++;
    }

    oldEnergy = d_Checkerboard_computeNewEnergy(b_this, idx, oldEnergy);
    break;

   case 3:
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    i8 = c_this->size[0] * c_this->size[1] * c_this->size[2];
    c_this->size[0] = loop_ub;
    c_this->size[1] = 1;
    c_this->size[2] = n;
    emxEnsureCapacity_real_T1(c_this, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        c_this->data[i + c_this->size[0] * c_this->size[1] * i8] =
          b_this->BoardCoords->data[(i + b_this->BoardCoords->size[0]) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    b_squeeze(c_this, predictedPoints);
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    i8 = c_this->size[0] * c_this->size[1] * c_this->size[2];
    c_this->size[0] = loop_ub;
    c_this->size[1] = 1;
    c_this->size[2] = n;
    emxEnsureCapacity_real_T1(c_this, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        c_this->data[i + c_this->size[0] * c_this->size[1] * i8] =
          b_this->BoardCoords->data[i + b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1] * i8];
      }
    }

    b_squeeze(c_this, p2);
    i8 = b_p2->size[0] * b_p2->size[1];
    b_p2->size[0] = p2->size[0];
    b_p2->size[1] = p2->size[1];
    emxEnsureCapacity_real_T(b_p2, i8);
    loop_ub = p2->size[0] * p2->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_p2->data[i8] = (p2->data[i8] + p2->data[i8]) - predictedPoints->data[i8];
    }

    Checkerboard_findClosestIndices(b_this, b_p2, newIndices);
    Checkerboard_expandBoardLeft(b_this, newIndices, predictedPoints, r9);
    i8 = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    b_this->BoardIdx->size[0] = predictedPoints->size[0];
    b_this->BoardIdx->size[1] = predictedPoints->size[1];
    emxEnsureCapacity_real_T(b_this->BoardIdx, i8);
    loop_ub = predictedPoints->size[0] * predictedPoints->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardIdx->data[i8] = predictedPoints->data[i8];
    }

    i8 = b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
      b_this->BoardCoords->size[2];
    b_this->BoardCoords->size[0] = r9->size[0];
    b_this->BoardCoords->size[1] = r9->size[1];
    b_this->BoardCoords->size[2] = r9->size[2];
    emxEnsureCapacity_real_T1(b_this->BoardCoords, i8);
    loop_ub = r9->size[0] * r9->size[1] * r9->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardCoords->data[i8] = r9->data[i8];
    }

    oldEnergy = e_Checkerboard_computeNewEnergy(b_this, oldEnergy);
    break;

   case 4:
    numRows = b_this->BoardCoords->size[1];
    if (numRows < (double)numRows - 2.0) {
      i8 = idx->size[0] * idx->size[1];
      idx->size[0] = 1;
      idx->size[1] = 0;
      emxEnsureCapacity_real_T(idx, i8);
    } else {
      i8 = idx->size[0] * idx->size[1];
      idx->size[0] = 1;
      idx->size[1] = (int)-(((double)numRows - 2.0) - (double)numRows) + 1;
      emxEnsureCapacity_real_T(idx, i8);
      loop_ub = (int)-(((double)numRows - 2.0) - (double)numRows);
      for (i8 = 0; i8 <= loop_ub; i8++) {
        idx->data[idx->size[0] * i8] = (double)numRows - (double)i8;
      }
    }

    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[1];
    i8 = c_this->size[0] * c_this->size[1] * c_this->size[2];
    c_this->size[0] = loop_ub;
    c_this->size[1] = 1;
    c_this->size[2] = n;
    emxEnsureCapacity_real_T1(c_this, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        c_this->data[i + c_this->size[0] * c_this->size[1] * i8] =
          b_this->BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx -
          1)) + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    b_squeeze(c_this, predictedPoints);
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i8 = c_this->size[0] * c_this->size[1] * c_this->size[2];
    c_this->size[0] = loop_ub;
    c_this->size[1] = 1;
    c_this->size[2] = n;
    emxEnsureCapacity_real_T1(c_this, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        c_this->data[i + c_this->size[0] * c_this->size[1] * i8] =
          b_this->BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx -
          1)) + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    b_squeeze(c_this, p2);
    i8 = b_p2->size[0] * b_p2->size[1];
    b_p2->size[0] = p2->size[0];
    b_p2->size[1] = p2->size[1];
    emxEnsureCapacity_real_T(b_p2, i8);
    loop_ub = p2->size[0] * p2->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_p2->data[i8] = (p2->data[i8] + p2->data[i8]) - predictedPoints->data[i8];
    }

    Checkerboard_findClosestIndices(b_this, b_p2, newIndices);
    i8 = predictedPoints->size[0] * predictedPoints->size[1];
    predictedPoints->size[0] = b_this->BoardIdx->size[0];
    predictedPoints->size[1] = b_this->BoardIdx->size[1] + 1;
    emxEnsureCapacity_real_T(predictedPoints, i8);
    loop_ub = b_this->BoardIdx->size[0] * (b_this->BoardIdx->size[1] + 1);
    for (i8 = 0; i8 < loop_ub; i8++) {
      predictedPoints->data[i8] = 0.0;
    }

    emxInit_int32_T(&r10, 1);
    loop_ub = predictedPoints->size[0];
    i8 = r10->size[0];
    r10->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r10, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      r10->data[i8] = i8;
    }

    n = r10->size[0];
    numRows = predictedPoints->size[1] - 1;
    for (i8 = 0; i8 < n; i8++) {
      predictedPoints->data[r10->data[i8] + predictedPoints->size[0] * numRows] =
        newIndices->data[i8];
    }

    loop_ub = b_this->BoardIdx->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      n = b_this->BoardIdx->size[0];
      for (i = 0; i < n; i++) {
        predictedPoints->data[i + predictedPoints->size[0] * i8] =
          b_this->BoardIdx->data[i + b_this->BoardIdx->size[0] * i8];
      }
    }

    i8 = r9->size[0] * r9->size[1] * r9->size[2];
    r9->size[0] = b_this->BoardCoords->size[0];
    r9->size[1] = b_this->BoardCoords->size[1] + 1;
    r9->size[2] = b_this->BoardCoords->size[2];
    emxEnsureCapacity_real_T1(r9, i8);
    loop_ub = b_this->BoardCoords->size[0] * (b_this->BoardCoords->size[1] + 1) *
      b_this->BoardCoords->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      r9->data[i8] = 0.0;
    }

    loop_ub = r9->size[0];
    i8 = r10->size[0];
    r10->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r10, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      r10->data[i8] = i8;
    }

    emxInit_int32_T(&r11, 1);
    loop_ub = r9->size[2];
    i8 = r11->size[0];
    r11->size[0] = loop_ub;
    emxEnsureCapacity_int32_T(r11, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      r11->data[i8] = i8;
    }

    emxInit_real_T1(&d_this, 2);
    i8 = d_this->size[0] * d_this->size[1];
    d_this->size[0] = newIndices->size[1];
    d_this->size[1] = 2;
    emxEnsureCapacity_real_T(d_this, i8);
    for (i8 = 0; i8 < 2; i8++) {
      loop_ub = newIndices->size[1];
      for (i = 0; i < loop_ub; i++) {
        d_this->data[i + d_this->size[0] * i8] = b_this->Points->data[((int)
          newIndices->data[newIndices->size[0] * i] + b_this->Points->size[0] *
          i8) - 1];
      }
    }

    n = r10->size[0];
    numRows = r11->size[0];
    i8 = r9->size[1] - 1;
    for (i = 0; i < numRows; i++) {
      for (b_idx = 0; b_idx < n; b_idx++) {
        r9->data[(r10->data[b_idx] + r9->size[0] * i8) + r9->size[0] * r9->size
          [1] * r11->data[i]] = d_this->data[b_idx + n * i];
      }
    }

    emxFree_real_T(&d_this);
    emxFree_int32_T(&r11);
    emxFree_int32_T(&r10);
    loop_ub = b_this->BoardCoords->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      n = b_this->BoardCoords->size[1];
      for (i = 0; i < n; i++) {
        numRows = b_this->BoardCoords->size[0];
        for (b_idx = 0; b_idx < numRows; b_idx++) {
          r9->data[(b_idx + r9->size[0] * i) + r9->size[0] * r9->size[1] * i8] =
            b_this->BoardCoords->data[(b_idx + b_this->BoardCoords->size[0] * i)
            + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
        }
      }
    }

    i8 = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    b_this->BoardIdx->size[0] = predictedPoints->size[0];
    b_this->BoardIdx->size[1] = predictedPoints->size[1];
    emxEnsureCapacity_real_T(b_this->BoardIdx, i8);
    loop_ub = predictedPoints->size[0] * predictedPoints->size[1];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardIdx->data[i8] = predictedPoints->data[i8];
    }

    i8 = b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
      b_this->BoardCoords->size[2];
    b_this->BoardCoords->size[0] = r9->size[0];
    b_this->BoardCoords->size[1] = r9->size[1];
    b_this->BoardCoords->size[2] = r9->size[2];
    emxEnsureCapacity_real_T1(b_this->BoardCoords, i8);
    loop_ub = r9->size[0] * r9->size[1] * r9->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_this->BoardCoords->data[i8] = r9->data[i8];
    }

    i8 = idx->size[0] * idx->size[1];
    idx->size[0] = 1;
    emxEnsureCapacity_real_T(idx, i8);
    b_idx = idx->size[0];
    numRows = idx->size[1];
    loop_ub = b_idx * numRows;
    for (i8 = 0; i8 < loop_ub; i8++) {
      idx->data[i8]++;
    }

    emxInit_real_T(&r12, 3);
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i8 = r12->size[0] * r12->size[1] * r12->size[2];
    r12->size[0] = loop_ub;
    r12->size[1] = 1;
    r12->size[2] = n;
    emxEnsureCapacity_real_T1(r12, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        r12->data[i + r12->size[0] * r12->size[1] * i8] = b_this->
          BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx - 1)) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    emxInit_real_T(&r13, 3);
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[2];
    i8 = r13->size[0] * r13->size[1] * r13->size[2];
    r13->size[0] = loop_ub;
    r13->size[1] = 1;
    r13->size[2] = n;
    emxEnsureCapacity_real_T1(r13, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        r13->data[i + r13->size[0] * r13->size[1] * i8] = b_this->
          BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx - 1)) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    emxInit_real_T(&b, 3);
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[1];
    i8 = b->size[0] * b->size[1] * b->size[2];
    b->size[0] = loop_ub;
    b->size[1] = 1;
    b->size[2] = n;
    emxEnsureCapacity_real_T1(b, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        b->data[i + b->size[0] * b->size[1] * i8] = b_this->BoardCoords->data[(i
          + b_this->BoardCoords->size[0] * (b_idx - 1)) + b_this->
          BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    i8 = c_this->size[0] * c_this->size[1] * c_this->size[2];
    c_this->size[0] = r12->size[0];
    c_this->size[1] = 1;
    c_this->size[2] = r12->size[2];
    emxEnsureCapacity_real_T1(c_this, i8);
    loop_ub = r12->size[0] * r12->size[1] * r12->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      c_this->data[i8] = (r12->data[i8] + r13->data[i8]) - 2.0 * b->data[i8];
    }

    emxFree_real_T(&b);
    b_squeeze(c_this, predictedPoints);
    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i8 = r12->size[0] * r12->size[1] * r12->size[2];
    r12->size[0] = loop_ub;
    r12->size[1] = 1;
    r12->size[2] = n;
    emxEnsureCapacity_real_T1(r12, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        r12->data[i + r12->size[0] * r12->size[1] * i8] = b_this->
          BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx - 1)) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    loop_ub = b_this->BoardCoords->size[0];
    n = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[2];
    i8 = r13->size[0] * r13->size[1] * r13->size[2];
    r13->size[0] = loop_ub;
    r13->size[1] = 1;
    r13->size[2] = n;
    emxEnsureCapacity_real_T1(r13, i8);
    for (i8 = 0; i8 < n; i8++) {
      for (i = 0; i < loop_ub; i++) {
        r13->data[i + r13->size[0] * r13->size[1] * i8] = b_this->
          BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx - 1)) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }
    }

    i8 = c_this->size[0] * c_this->size[1] * c_this->size[2];
    c_this->size[0] = r12->size[0];
    c_this->size[1] = 1;
    c_this->size[2] = r12->size[2];
    emxEnsureCapacity_real_T1(c_this, i8);
    loop_ub = r12->size[0] * r12->size[1] * r12->size[2];
    for (i8 = 0; i8 < loop_ub; i8++) {
      c_this->data[i8] = r12->data[i8] - r13->data[i8];
    }

    emxFree_real_T(&r13);
    emxFree_real_T(&r12);
    emxInit_real_T2(&b_predictedPoints, 1);
    b_squeeze(c_this, p2);
    loop_ub = predictedPoints->size[0];
    i8 = b_predictedPoints->size[0];
    b_predictedPoints->size[0] = loop_ub;
    emxEnsureCapacity_real_T2(b_predictedPoints, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_predictedPoints->data[i8] = predictedPoints->data[i8];
    }

    emxInit_real_T2(&c_predictedPoints, 1);
    loop_ub = predictedPoints->size[0];
    i8 = c_predictedPoints->size[0];
    c_predictedPoints->size[0] = loop_ub;
    emxEnsureCapacity_real_T2(c_predictedPoints, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      c_predictedPoints->data[i8] = predictedPoints->data[i8 +
        predictedPoints->size[0]];
    }

    emxInit_real_T2(&r14, 1);
    c_hypot(b_predictedPoints, c_predictedPoints, r14);
    loop_ub = p2->size[0];
    i8 = b_predictedPoints->size[0];
    b_predictedPoints->size[0] = loop_ub;
    emxEnsureCapacity_real_T2(b_predictedPoints, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      b_predictedPoints->data[i8] = p2->data[i8];
    }

    loop_ub = p2->size[0];
    i8 = c_predictedPoints->size[0];
    c_predictedPoints->size[0] = loop_ub;
    emxEnsureCapacity_real_T2(c_predictedPoints, i8);
    for (i8 = 0; i8 < loop_ub; i8++) {
      c_predictedPoints->data[i8] = p2->data[i8 + p2->size[0]];
    }

    emxInit_real_T2(&r15, 1);
    c_hypot(b_predictedPoints, c_predictedPoints, r15);
    rdivide(r14, r15, b_predictedPoints);
    numRows = 1;
    n = b_predictedPoints->size[0];
    mtmp = b_predictedPoints->data[0];
    emxFree_real_T(&c_predictedPoints);
    emxFree_real_T(&r15);
    emxFree_real_T(&r14);
    if (b_predictedPoints->size[0] > 1) {
      if (rtIsNaN(b_predictedPoints->data[0])) {
        b_idx = 2;
        exitg1 = false;
        while ((!exitg1) && (b_idx <= n)) {
          numRows = b_idx;
          if (!rtIsNaN(b_predictedPoints->data[b_idx - 1])) {
            mtmp = b_predictedPoints->data[b_idx - 1];
            exitg1 = true;
          } else {
            b_idx++;
          }
        }
      }

      if (numRows < b_predictedPoints->size[0]) {
        while (numRows + 1 <= n) {
          if (b_predictedPoints->data[numRows] > mtmp) {
            mtmp = b_predictedPoints->data[numRows];
          }

          numRows++;
        }
      }
    }

    emxFree_real_T(&b_predictedPoints);
    if ((oldEnergy > mtmp) || rtIsNaN(mtmp)) {
    } else {
      oldEnergy = (float)mtmp;
    }

    mtmp = (double)b_this->BoardCoords->size[0] - 2.0;
    i = 0;
    emxInit_real_T(&num, 3);
    emxInit_real_T(&denom, 3);
    emxInit_real_T(&r16, 3);
    while (i <= (int)mtmp - 1) {
      loop_ub = b_this->BoardCoords->size[2];
      b_idx = (int)idx->data[0];
      i8 = num->size[0] * num->size[1] * num->size[2];
      num->size[0] = 1;
      num->size[1] = 1;
      num->size[2] = loop_ub;
      emxEnsureCapacity_real_T1(num, i8);
      for (i8 = 0; i8 < loop_ub; i8++) {
        num->data[num->size[0] * num->size[1] * i8] = b_this->BoardCoords->data
          [(i + b_this->BoardCoords->size[0] * (b_idx - 1)) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }

      loop_ub = b_this->BoardCoords->size[2];
      b_idx = (int)idx->data[0];
      i8 = r16->size[0] * r16->size[1] * r16->size[2];
      r16->size[0] = 1;
      r16->size[1] = 1;
      r16->size[2] = loop_ub;
      emxEnsureCapacity_real_T1(r16, i8);
      for (i8 = 0; i8 < loop_ub; i8++) {
        r16->data[r16->size[0] * r16->size[1] * i8] = b_this->BoardCoords->data
          [(((int)((1.0 + (double)i) + 2.0) + b_this->BoardCoords->size[0] *
             (b_idx - 1)) + b_this->BoardCoords->size[0] * b_this->
            BoardCoords->size[1] * i8) - 1];
      }

      loop_ub = b_this->BoardCoords->size[2];
      b_idx = (int)idx->data[0];
      i8 = denom->size[0] * denom->size[1] * denom->size[2];
      denom->size[0] = 1;
      denom->size[1] = 1;
      denom->size[2] = loop_ub;
      emxEnsureCapacity_real_T1(denom, i8);
      for (i8 = 0; i8 < loop_ub; i8++) {
        denom->data[denom->size[0] * denom->size[1] * i8] = b_this->
          BoardCoords->data[(((int)((1.0 + (double)i) + 1.0) +
                              b_this->BoardCoords->size[0] * (b_idx - 1)) +
                             b_this->BoardCoords->size[0] * b_this->
                             BoardCoords->size[1] * i8) - 1];
      }

      i8 = num->size[0] * num->size[1] * num->size[2];
      num->size[0] = 1;
      num->size[1] = 1;
      emxEnsureCapacity_real_T1(num, i8);
      numRows = num->size[0];
      n = num->size[1];
      b_idx = num->size[2];
      loop_ub = numRows * n * b_idx;
      for (i8 = 0; i8 < loop_ub; i8++) {
        num->data[i8] = (num->data[i8] + r16->data[i8]) - 2.0 * denom->data[i8];
      }

      loop_ub = b_this->BoardCoords->size[2];
      b_idx = (int)idx->data[0];
      i8 = denom->size[0] * denom->size[1] * denom->size[2];
      denom->size[0] = 1;
      denom->size[1] = 1;
      denom->size[2] = loop_ub;
      emxEnsureCapacity_real_T1(denom, i8);
      for (i8 = 0; i8 < loop_ub; i8++) {
        denom->data[denom->size[0] * denom->size[1] * i8] = b_this->
          BoardCoords->data[(i + b_this->BoardCoords->size[0] * (b_idx - 1)) +
          b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i8];
      }

      loop_ub = b_this->BoardCoords->size[2];
      b_idx = (int)idx->data[0];
      i8 = r16->size[0] * r16->size[1] * r16->size[2];
      r16->size[0] = 1;
      r16->size[1] = 1;
      r16->size[2] = loop_ub;
      emxEnsureCapacity_real_T1(r16, i8);
      for (i8 = 0; i8 < loop_ub; i8++) {
        r16->data[r16->size[0] * r16->size[1] * i8] = b_this->BoardCoords->data
          [(((int)((1.0 + (double)i) + 2.0) + b_this->BoardCoords->size[0] *
             (b_idx - 1)) + b_this->BoardCoords->size[0] * b_this->
            BoardCoords->size[1] * i8) - 1];
      }

      i8 = denom->size[0] * denom->size[1] * denom->size[2];
      denom->size[0] = 1;
      denom->size[1] = 1;
      emxEnsureCapacity_real_T1(denom, i8);
      numRows = denom->size[0];
      n = denom->size[1];
      b_idx = denom->size[2];
      loop_ub = numRows * n * b_idx;
      for (i8 = 0; i8 < loop_ub; i8++) {
        denom->data[i8] -= r16->data[i8];
      }

      b_num[0] = num->size[2];
      b_denom[0] = denom->size[2];
      c_num = *num;
      c_num.size = (int *)&b_num;
      c_num.numDimensions = 1;
      c_denom = *denom;
      c_denom.size = (int *)&b_denom;
      c_denom.numDimensions = 1;
      z = norm(&c_num) / norm(&c_denom);
      if ((oldEnergy > z) || rtIsNaN(z)) {
      } else {
        oldEnergy = (float)z;
      }

      i++;
    }

    emxFree_real_T(&r16);
    emxFree_real_T(&denom);
    emxFree_real_T(&num);
    numRows = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
    oldEnergy = oldEnergy * (float)numRows - (float)(b_this->BoardIdx->size[0] *
      b_this->BoardIdx->size[1]);
    break;

   default:
    oldEnergy = ((real32_T)rtInf);
    break;
  }

  emxFree_real_T(&c_this);
  emxFree_real_T(&b_p2);
  emxFree_real_T(&p2);
  emxFree_real_T(&idx);
  emxFree_real_T(&r9);
  emxFree_real_T(&newIndices);
  emxFree_real_T(&predictedPoints);
  b_this->Energy = oldEnergy;
}

static void c_Checkerboard_predictPointsVer(const
  c_vision_internal_calibration_c *b_this, emxArray_real_T *newPoints)
{
  emxArray_real_T *c_this;
  int loop_ub;
  int b_loop_ub;
  int i9;
  emxArray_real_T *p1;
  int i10;
  emxInit_real_T(&c_this, 3);
  loop_ub = b_this->BoardCoords->size[1];
  b_loop_ub = b_this->BoardCoords->size[2];
  i9 = c_this->size[0] * c_this->size[1] * c_this->size[2];
  c_this->size[0] = 1;
  c_this->size[1] = loop_ub;
  c_this->size[2] = b_loop_ub;
  emxEnsureCapacity_real_T1(c_this, i9);
  for (i9 = 0; i9 < b_loop_ub; i9++) {
    for (i10 = 0; i10 < loop_ub; i10++) {
      c_this->data[c_this->size[0] * i10 + c_this->size[0] * c_this->size[1] *
        i9] = b_this->BoardCoords->data[(b_this->BoardCoords->size[0] * i10 +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i9) + 1];
    }
  }

  emxInit_real_T1(&p1, 2);
  squeeze(c_this, p1);
  loop_ub = b_this->BoardCoords->size[1];
  b_loop_ub = b_this->BoardCoords->size[2];
  i9 = c_this->size[0] * c_this->size[1] * c_this->size[2];
  c_this->size[0] = 1;
  c_this->size[1] = loop_ub;
  c_this->size[2] = b_loop_ub;
  emxEnsureCapacity_real_T1(c_this, i9);
  for (i9 = 0; i9 < b_loop_ub; i9++) {
    for (i10 = 0; i10 < loop_ub; i10++) {
      c_this->data[c_this->size[0] * i10 + c_this->size[0] * c_this->size[1] *
        i9] = b_this->BoardCoords->data[b_this->BoardCoords->size[0] * i10 +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i9];
    }
  }

  squeeze(c_this, newPoints);
  i9 = newPoints->size[0] * newPoints->size[1];
  emxEnsureCapacity_real_T(newPoints, i9);
  loop_ub = newPoints->size[0];
  b_loop_ub = newPoints->size[1];
  loop_ub *= b_loop_ub;
  emxFree_real_T(&c_this);
  for (i9 = 0; i9 < loop_ub; i9++) {
    newPoints->data[i9] = (newPoints->data[i9] + newPoints->data[i9]) - p1->
      data[i9];
  }

  emxFree_real_T(&p1);
}

static void c_hypot(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *r)
{
  int c;
  int k;
  if (x->size[0] <= y->size[0]) {
    c = x->size[0];
  } else {
    c = y->size[0];
  }

  k = r->size[0];
  r->size[0] = c;
  emxEnsureCapacity_real_T2(r, k);
  for (k = 0; k + 1 <= c; k++) {
    r->data[k] = rt_hypotd_snf(x->data[k], y->data[k]);
  }
}

static void c_imfilter(emxArray_real32_T *varargin_1)
{
  int finalSize_idx_0;
  double pad[2];
  int finalSize_idx_1;
  emxArray_real32_T *a;
  emxArray_real_T *b_a;
  int cEnd1;
  int ma;
  emxArray_real_T *result;
  int iv4[2];
  boolean_T b3;
  int cEnd;
  int na;
  int r;
  int firstColB;
  int firstRowA;
  int lastColB;
  int aidx;
  int firstRowB;
  int lastRowB;
  int cidx;
  int lastColA;
  int k;
  int b_firstColB;
  int iC;
  int iA;
  int iB;
  int i;
  int b_i;
  int a_length;
  static const double dv2[49] = { 0.0013419653598432805, 0.0040765308179236169,
    0.00793999784347829, 0.0099158573267036591, 0.00793999784347829,
    0.0040765308179236169, 0.0013419653598432805, 0.0040765308179236169,
    0.012383407207635908, 0.024119583762554284, 0.030121714902657255,
    0.024119583762554284, 0.012383407207635908, 0.0040765308179236169,
    0.00793999784347829, 0.024119583762554284, 0.0469785343503966,
    0.058669089490849473, 0.0469785343503966, 0.024119583762554284,
    0.00793999784347829, 0.0099158573267036591, 0.030121714902657255,
    0.058669089490849473, 0.073268826056005834, 0.058669089490849473,
    0.030121714902657255, 0.0099158573267036591, 0.00793999784347829,
    0.024119583762554284, 0.0469785343503966, 0.058669089490849473,
    0.0469785343503966, 0.024119583762554284, 0.00793999784347829,
    0.0040765308179236169, 0.012383407207635908, 0.024119583762554284,
    0.030121714902657255, 0.024119583762554284, 0.012383407207635908,
    0.0040765308179236169, 0.0013419653598432805, 0.0040765308179236169,
    0.00793999784347829, 0.0099158573267036591, 0.00793999784347829,
    0.0040765308179236169, 0.0013419653598432805 };

  finalSize_idx_0 = varargin_1->size[0];
  pad[0] = 3.0;
  finalSize_idx_1 = varargin_1->size[1];
  pad[1] = 3.0;
  if (!((varargin_1->size[0] == 0) || (varargin_1->size[1] == 0))) {
    emxInit_real32_T(&a, 2);
    emxInit_real_T1(&b_a, 2);
    padImage(varargin_1, pad, a);
    cEnd1 = b_a->size[0] * b_a->size[1];
    b_a->size[0] = a->size[0];
    b_a->size[1] = a->size[1];
    emxEnsureCapacity_real_T(b_a, cEnd1);
    ma = a->size[0] * a->size[1];
    for (cEnd1 = 0; cEnd1 < ma; cEnd1++) {
      b_a->data[cEnd1] = a->data[cEnd1];
    }

    emxFree_real32_T(&a);
    for (cEnd1 = 0; cEnd1 < 2; cEnd1++) {
      iv4[cEnd1] = b_a->size[cEnd1];
    }

    emxInit_real_T1(&result, 2);
    cEnd1 = result->size[0] * result->size[1];
    result->size[0] = iv4[0];
    result->size[1] = iv4[1];
    emxEnsureCapacity_real_T(result, cEnd1);
    ma = iv4[0] * iv4[1];
    for (cEnd1 = 0; cEnd1 < ma; cEnd1++) {
      result->data[cEnd1] = 0.0;
    }

    if ((b_a->size[0] == 0) || (b_a->size[1] == 0) || ((iv4[0] == 0) || (iv4[1] ==
          0))) {
      b3 = true;
    } else {
      b3 = false;
    }

    if (!b3) {
      cEnd = iv4[1];
      cEnd1 = iv4[0];
      ma = b_a->size[0];
      na = b_a->size[1] - 3;
      if (b_a->size[1] < 3) {
        firstColB = 4 - b_a->size[1];
      } else {
        firstColB = 0;
      }

      if (7 <= iv4[1] + 2) {
        lastColB = 6;
      } else {
        lastColB = iv4[1] + 2;
      }

      if (b_a->size[0] < 3) {
        firstRowB = 4 - b_a->size[0];
      } else {
        firstRowB = 0;
      }

      if (7 <= iv4[0] + 2) {
        lastRowB = 6;
      } else {
        lastRowB = iv4[0] + 2;
      }

      while (firstColB <= lastColB) {
        if ((firstColB + na) + 2 < cEnd + 2) {
          lastColA = na;
        } else {
          lastColA = cEnd - firstColB;
        }

        if (firstColB < 3) {
          k = 3 - firstColB;
        } else {
          k = 0;
        }

        while (k <= lastColA + 2) {
          if (firstColB + k > 3) {
            b_firstColB = (firstColB + k) - 3;
          } else {
            b_firstColB = 0;
          }

          iC = b_firstColB * cEnd1;
          iA = k * ma;
          iB = firstRowB + firstColB * 7;
          for (i = firstRowB; i <= lastRowB; i++) {
            if (i < 3) {
              firstRowA = 3 - i;
            } else {
              firstRowA = 0;
            }

            if (i + ma <= cEnd1 + 2) {
              b_i = ma;
            } else {
              b_i = (cEnd1 - i) + 3;
            }

            a_length = b_i - firstRowA;
            aidx = iA + firstRowA;
            cidx = iC;
            for (r = 1; r <= a_length; r++) {
              result->data[cidx] += dv2[iB] * b_a->data[aidx];
              aidx++;
              cidx++;
            }

            iB++;
            if (i >= 3) {
              iC++;
            }
          }

          k++;
        }

        firstColB++;
      }
    }

    emxFree_real_T(&b_a);
    if (4.0 > (double)finalSize_idx_0 + 3.0) {
      cEnd1 = 0;
      r = 0;
    } else {
      cEnd1 = 3;
      r = (int)((double)finalSize_idx_0 + 3.0);
    }

    if (4.0 > (double)finalSize_idx_1 + 3.0) {
      cEnd = 0;
      firstRowA = 0;
    } else {
      cEnd = 3;
      firstRowA = (int)((double)finalSize_idx_1 + 3.0);
    }

    aidx = varargin_1->size[0] * varargin_1->size[1];
    varargin_1->size[0] = r - cEnd1;
    varargin_1->size[1] = firstRowA - cEnd;
    emxEnsureCapacity_real32_T(varargin_1, aidx);
    ma = firstRowA - cEnd;
    for (firstRowA = 0; firstRowA < ma; firstRowA++) {
      cidx = r - cEnd1;
      for (aidx = 0; aidx < cidx; aidx++) {
        varargin_1->data[aidx + varargin_1->size[0] * firstRowA] = (float)
          result->data[(cEnd1 + aidx) + result->size[0] * (cEnd + firstRowA)];
      }
    }

    emxFree_real_T(&result);
  }
}

static void c_sort(double x[4])
{
  int i1;
  int nNaNs;
  int idx[4];
  int ib;
  double x4[4];
  int k;
  int idx4[4];
  double xwork[4];
  signed char perm[4];
  int i2;
  int i3;
  int i4;
  for (i1 = 0; i1 < 4; i1++) {
    idx[i1] = 0;
    x4[i1] = 0.0;
    idx4[i1] = 0;
    xwork[i1] = 0.0;
  }

  nNaNs = 0;
  ib = 0;
  for (k = 0; k < 4; k++) {
    if (rtIsNaN(x[k])) {
      idx[3 - nNaNs] = k + 1;
      xwork[3 - nNaNs] = x[k];
      nNaNs++;
    } else {
      ib++;
      idx4[ib - 1] = k + 1;
      x4[ib - 1] = x[k];
      if (ib == 4) {
        ib = k - nNaNs;
        if (x4[0] <= x4[1]) {
          i1 = 1;
          i2 = 2;
        } else {
          i1 = 2;
          i2 = 1;
        }

        if (x4[2] <= x4[3]) {
          i3 = 3;
          i4 = 4;
        } else {
          i3 = 4;
          i4 = 3;
        }

        if (x4[i1 - 1] <= x4[i3 - 1]) {
          if (x4[i2 - 1] <= x4[i3 - 1]) {
            perm[0] = (signed char)i1;
            perm[1] = (signed char)i2;
            perm[2] = (signed char)i3;
            perm[3] = (signed char)i4;
          } else if (x4[i2 - 1] <= x4[i4 - 1]) {
            perm[0] = (signed char)i1;
            perm[1] = (signed char)i3;
            perm[2] = (signed char)i2;
            perm[3] = (signed char)i4;
          } else {
            perm[0] = (signed char)i1;
            perm[1] = (signed char)i3;
            perm[2] = (signed char)i4;
            perm[3] = (signed char)i2;
          }
        } else if (x4[i1 - 1] <= x4[i4 - 1]) {
          if (x4[i2 - 1] <= x4[i4 - 1]) {
            perm[0] = (signed char)i3;
            perm[1] = (signed char)i1;
            perm[2] = (signed char)i2;
            perm[3] = (signed char)i4;
          } else {
            perm[0] = (signed char)i3;
            perm[1] = (signed char)i1;
            perm[2] = (signed char)i4;
            perm[3] = (signed char)i2;
          }
        } else {
          perm[0] = (signed char)i3;
          perm[1] = (signed char)i4;
          perm[2] = (signed char)i1;
          perm[3] = (signed char)i2;
        }

        idx[ib - 3] = idx4[perm[0] - 1];
        idx[ib - 2] = idx4[perm[1] - 1];
        idx[ib - 1] = idx4[perm[2] - 1];
        idx[ib] = idx4[perm[3] - 1];
        x[ib - 3] = x4[perm[0] - 1];
        x[ib - 2] = x4[perm[1] - 1];
        x[ib - 1] = x4[perm[2] - 1];
        x[ib] = x4[perm[3] - 1];
        ib = 0;
      }
    }
  }

  if (ib > 0) {
    for (i1 = 0; i1 < 4; i1++) {
      perm[i1] = 0;
    }

    if (ib == 1) {
      perm[0] = 1;
    } else if (ib == 2) {
      if (x4[0] <= x4[1]) {
        perm[0] = 1;
        perm[1] = 2;
      } else {
        perm[0] = 2;
        perm[1] = 1;
      }
    } else if (x4[0] <= x4[1]) {
      if (x4[1] <= x4[2]) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 3;
      } else if (x4[0] <= x4[2]) {
        perm[0] = 1;
        perm[1] = 3;
        perm[2] = 2;
      } else {
        perm[0] = 3;
        perm[1] = 1;
        perm[2] = 2;
      }
    } else if (x4[0] <= x4[2]) {
      perm[0] = 2;
      perm[1] = 1;
      perm[2] = 3;
    } else if (x4[1] <= x4[2]) {
      perm[0] = 2;
      perm[1] = 3;
      perm[2] = 1;
    } else {
      perm[0] = 3;
      perm[1] = 2;
      perm[2] = 1;
    }

    for (k = 4; k - 3 <= ib; k++) {
      idx[(k - nNaNs) - ib] = idx4[perm[k - 4] - 1];
      x[(k - nNaNs) - ib] = x4[perm[k - 4] - 1];
    }
  }

  i1 = nNaNs >> 1;
  for (k = 1; k <= i1; k++) {
    ib = idx[(k - nNaNs) + 3];
    idx[(k - nNaNs) + 3] = idx[4 - k];
    idx[4 - k] = ib;
    x[(k - nNaNs) + 3] = xwork[4 - k];
    x[4 - k] = xwork[(k - nNaNs) + 3];
  }

  if ((nNaNs & 1) != 0) {
    x[(i1 - nNaNs) + 4] = xwork[(i1 - nNaNs) + 4];
  }

  if (4 - nNaNs > 1) {
    for (i1 = 0; i1 < 4; i1++) {
      idx4[i1] = 0;
    }

    i3 = (4 - nNaNs) >> 2;
    i2 = 4;
    while (i3 > 1) {
      if ((i3 & 1) != 0) {
        i3--;
        ib = i2 * i3;
        i1 = 4 - (nNaNs + ib);
        if (i1 > i2) {
          b_merge(idx, x, ib, i2, i1 - i2, idx4, xwork);
        }
      }

      ib = i2 << 1;
      i3 >>= 1;
      for (k = 1; k <= i3; k++) {
        b_merge(idx, x, (k - 1) * ib, i2, i2, idx4, xwork);
      }

      i2 = ib;
    }

    if (4 - nNaNs > i2) {
      b_merge(idx, x, 0, i2, 4 - (nNaNs + i2), idx4, xwork);
    }
  }
}

static void cat(const emxArray_real_T *varargin_1, const emxArray_real_T
                *varargin_2, emxArray_real_T *y)
{
  unsigned int ysize_idx_0;
  unsigned int ysize_idx_1;
  int i27;
  int iy;
  int j;
  ysize_idx_0 = (unsigned int)varargin_1->size[0];
  ysize_idx_1 = (unsigned int)varargin_1->size[1];
  i27 = y->size[0] * y->size[1] * y->size[2];
  y->size[0] = (int)ysize_idx_0;
  y->size[1] = (int)ysize_idx_1;
  y->size[2] = 2;
  emxEnsureCapacity_real_T1(y, i27);
  iy = -1;
  i27 = varargin_1->size[0] * varargin_1->size[1];
  for (j = 1; j <= i27; j++) {
    iy++;
    y->data[iy] = varargin_1->data[j - 1];
  }

  i27 = varargin_2->size[0] * varargin_2->size[1];
  for (j = 1; j <= i27; j++) {
    iy++;
    y->data[iy] = varargin_2->data[j - 1];
  }
}

static void cornerOrientations(const emxArray_real32_T *Ix2, const
  emxArray_real32_T *Iy2, const emxArray_real32_T *Ixy, const float p[2], float
  v1[2], float v2[2])
{
  float df;
  float adf;
  float tb;
  float ab;
  float rt;
  int sgn1;
  int sgn2;
  float sn1[2];
  float cs1[2];
  static const float fv0[4] = { 0.707106769F, 0.707106769F, -0.707106769F,
    0.707106769F };

  df = Ix2->data[((int)p[1] + Ix2->size[0] * ((int)p[0] - 1)) - 1] - Iy2->data
    [((int)p[1] + Iy2->size[0] * ((int)p[0] - 1)) - 1];
  adf = std::abs(df);
  tb = Ixy->data[((int)p[1] + Ixy->size[0] * ((int)p[0] - 1)) - 1] + Ixy->data
    [((int)p[1] + Ixy->size[0] * ((int)p[0] - 1)) - 1];
  ab = std::abs(tb);
  if (adf > ab) {
    rt = ab / adf;
    rt = adf * std::sqrt(1.0F + rt * rt);
  } else if (adf < ab) {
    rt = adf / ab;
    rt = ab * std::sqrt(1.0F + rt * rt);
  } else {
    rt = ab * 1.41421354F;
  }

  if (Ix2->data[((int)p[1] + Ix2->size[0] * ((int)p[0] - 1)) - 1] + Iy2->data
      [((int)p[1] + Iy2->size[0] * ((int)p[0] - 1)) - 1] < 0.0F) {
    sgn1 = -1;
  } else {
    sgn1 = 1;
  }

  if (df > 0.0F) {
    rt += df;
    sgn2 = 1;
  } else {
    rt = df - rt;
    sgn2 = -1;
  }

  if (std::abs(rt) > ab) {
    rt = -tb / rt;
    df = 1.0F / std::sqrt(1.0F + rt * rt);
    adf = rt * df;
  } else if (ab == 0.0F) {
    adf = 1.0F;
    df = 0.0F;
  } else {
    rt = -rt / tb;
    adf = 1.0F / std::sqrt(1.0F + rt * rt);
    df = rt * adf;
  }

  if (sgn1 == sgn2) {
    rt = adf;
    adf = -df;
    df = rt;
  }

  sn1[0] = -df;
  sn1[1] = adf;
  cs1[0] = adf;
  cs1[1] = df;
  for (sgn1 = 0; sgn1 < 2; sgn1++) {
    v1[sgn1] = 0.0F;
    v2[sgn1] = 0.0F;
    for (sgn2 = 0; sgn2 < 2; sgn2++) {
      v1[sgn1] += sn1[sgn2] * fv0[sgn2 + (sgn1 << 1)];
      v2[sgn1] += cs1[sgn2] * fv0[sgn2 + (sgn1 << 1)];
    }
  }
}

static float d_Checkerboard_computeNewEnergy(const
  c_vision_internal_calibration_c *b_this, const emxArray_real_T *idx, float
  oldEnergy)
{
  float newEnergy;
  emxArray_real_T *r29;
  int loop_ub;
  int ixstart;
  int b_idx;
  int i20;
  emxArray_real_T *r30;
  int n;
  emxArray_real_T *b;
  emxArray_real_T *r31;
  emxArray_real_T *num;
  emxArray_real_T *denom;
  emxArray_real_T *b_num;
  emxArray_real_T *c_num;
  emxArray_real_T *r32;
  emxArray_real_T *r33;
  double mtmp;
  boolean_T exitg1;
  int i;
  emxArray_real_T *d_num;
  emxArray_real_T *b_denom;
  emxArray_real_T *r34;
  int e_num[1];
  int c_denom[1];
  emxArray_real_T f_num;
  emxArray_real_T d_denom;
  double z;
  emxInit_real_T(&r29, 3);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  b_idx = (int)idx->data[0];
  i20 = r29->size[0] * r29->size[1] * r29->size[2];
  r29->size[0] = 1;
  r29->size[1] = loop_ub;
  r29->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r29, i20);
  for (i20 = 0; i20 < ixstart; i20++) {
    for (n = 0; n < loop_ub; n++) {
      r29->data[r29->size[0] * n + r29->size[0] * r29->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * n) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i20) - 1];
    }
  }

  emxInit_real_T(&r30, 3);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  b_idx = (int)idx->data[2];
  i20 = r30->size[0] * r30->size[1] * r30->size[2];
  r30->size[0] = 1;
  r30->size[1] = loop_ub;
  r30->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r30, i20);
  for (i20 = 0; i20 < ixstart; i20++) {
    for (n = 0; n < loop_ub; n++) {
      r30->data[r30->size[0] * n + r30->size[0] * r30->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * n) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i20) - 1];
    }
  }

  emxInit_real_T(&b, 3);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  b_idx = (int)idx->data[1];
  i20 = b->size[0] * b->size[1] * b->size[2];
  b->size[0] = 1;
  b->size[1] = loop_ub;
  b->size[2] = ixstart;
  emxEnsureCapacity_real_T1(b, i20);
  for (i20 = 0; i20 < ixstart; i20++) {
    for (n = 0; n < loop_ub; n++) {
      b->data[b->size[0] * n + b->size[0] * b->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * n) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i20) - 1];
    }
  }

  emxInit_real_T(&r31, 3);
  i20 = r31->size[0] * r31->size[1] * r31->size[2];
  r31->size[0] = 1;
  r31->size[1] = r29->size[1];
  r31->size[2] = r29->size[2];
  emxEnsureCapacity_real_T1(r31, i20);
  loop_ub = r29->size[0] * r29->size[1] * r29->size[2];
  for (i20 = 0; i20 < loop_ub; i20++) {
    r31->data[i20] = (r29->data[i20] + r30->data[i20]) - 2.0 * b->data[i20];
  }

  emxFree_real_T(&b);
  emxInit_real_T1(&num, 2);
  squeeze(r31, num);
  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  b_idx = (int)idx->data[0];
  i20 = r29->size[0] * r29->size[1] * r29->size[2];
  r29->size[0] = 1;
  r29->size[1] = loop_ub;
  r29->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r29, i20);
  for (i20 = 0; i20 < ixstart; i20++) {
    for (n = 0; n < loop_ub; n++) {
      r29->data[r29->size[0] * n + r29->size[0] * r29->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * n) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i20) - 1];
    }
  }

  loop_ub = b_this->BoardCoords->size[1];
  ixstart = b_this->BoardCoords->size[2];
  b_idx = (int)idx->data[2];
  i20 = r30->size[0] * r30->size[1] * r30->size[2];
  r30->size[0] = 1;
  r30->size[1] = loop_ub;
  r30->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r30, i20);
  for (i20 = 0; i20 < ixstart; i20++) {
    for (n = 0; n < loop_ub; n++) {
      r30->data[r30->size[0] * n + r30->size[0] * r30->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * n) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i20) - 1];
    }
  }

  i20 = r31->size[0] * r31->size[1] * r31->size[2];
  r31->size[0] = 1;
  r31->size[1] = r29->size[1];
  r31->size[2] = r29->size[2];
  emxEnsureCapacity_real_T1(r31, i20);
  loop_ub = r29->size[0] * r29->size[1] * r29->size[2];
  for (i20 = 0; i20 < loop_ub; i20++) {
    r31->data[i20] = r29->data[i20] - r30->data[i20];
  }

  emxFree_real_T(&r30);
  emxFree_real_T(&r29);
  emxInit_real_T1(&denom, 2);
  emxInit_real_T2(&b_num, 1);
  squeeze(r31, denom);
  loop_ub = num->size[0];
  i20 = b_num->size[0];
  b_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(b_num, i20);
  emxFree_real_T(&r31);
  for (i20 = 0; i20 < loop_ub; i20++) {
    b_num->data[i20] = num->data[i20];
  }

  emxInit_real_T2(&c_num, 1);
  loop_ub = num->size[0];
  i20 = c_num->size[0];
  c_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(c_num, i20);
  for (i20 = 0; i20 < loop_ub; i20++) {
    c_num->data[i20] = num->data[i20 + num->size[0]];
  }

  emxFree_real_T(&num);
  emxInit_real_T2(&r32, 1);
  c_hypot(b_num, c_num, r32);
  loop_ub = denom->size[0];
  i20 = b_num->size[0];
  b_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(b_num, i20);
  for (i20 = 0; i20 < loop_ub; i20++) {
    b_num->data[i20] = denom->data[i20];
  }

  loop_ub = denom->size[0];
  i20 = c_num->size[0];
  c_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(c_num, i20);
  for (i20 = 0; i20 < loop_ub; i20++) {
    c_num->data[i20] = denom->data[i20 + denom->size[0]];
  }

  emxFree_real_T(&denom);
  emxInit_real_T2(&r33, 1);
  c_hypot(b_num, c_num, r33);
  rdivide(r32, r33, b_num);
  ixstart = 1;
  n = b_num->size[0];
  mtmp = b_num->data[0];
  emxFree_real_T(&c_num);
  emxFree_real_T(&r33);
  emxFree_real_T(&r32);
  if (b_num->size[0] > 1) {
    if (rtIsNaN(b_num->data[0])) {
      loop_ub = 2;
      exitg1 = false;
      while ((!exitg1) && (loop_ub <= n)) {
        ixstart = loop_ub;
        if (!rtIsNaN(b_num->data[loop_ub - 1])) {
          mtmp = b_num->data[loop_ub - 1];
          exitg1 = true;
        } else {
          loop_ub++;
        }
      }
    }

    if (ixstart < b_num->size[0]) {
      while (ixstart + 1 <= n) {
        if (b_num->data[ixstart] > mtmp) {
          mtmp = b_num->data[ixstart];
        }

        ixstart++;
      }
    }
  }

  emxFree_real_T(&b_num);
  if ((oldEnergy > mtmp) || rtIsNaN(mtmp)) {
    newEnergy = oldEnergy;
  } else {
    newEnergy = (float)mtmp;
  }

  mtmp = (double)b_this->BoardCoords->size[1] - 2.0;
  i = 0;
  emxInit_real_T(&d_num, 3);
  emxInit_real_T(&b_denom, 3);
  emxInit_real_T(&r34, 3);
  while (i <= (int)mtmp - 1) {
    loop_ub = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i20 = d_num->size[0] * d_num->size[1] * d_num->size[2];
    d_num->size[0] = 1;
    d_num->size[1] = 1;
    d_num->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(d_num, i20);
    for (i20 = 0; i20 < loop_ub; i20++) {
      d_num->data[d_num->size[0] * d_num->size[1] * i20] = b_this->
        BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * i) +
                           b_this->BoardCoords->size[0] * b_this->
                           BoardCoords->size[1] * i20) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i20 = r34->size[0] * r34->size[1] * r34->size[2];
    r34->size[0] = 1;
    r34->size[1] = 1;
    r34->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(r34, i20);
    for (i20 = 0; i20 < loop_ub; i20++) {
      r34->data[r34->size[0] * r34->size[1] * i20] = b_this->BoardCoords->data
        [((b_idx + b_this->BoardCoords->size[0] * ((int)((1.0 + (double)i) + 2.0)
            - 1)) + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
          i20) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i20 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    b_denom->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(b_denom, i20);
    for (i20 = 0; i20 < loop_ub; i20++) {
      b_denom->data[b_denom->size[0] * b_denom->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * ((int)
        ((1.0 + (double)i) + 1.0) - 1)) + b_this->BoardCoords->size[0] *
        b_this->BoardCoords->size[1] * i20) - 1];
    }

    i20 = d_num->size[0] * d_num->size[1] * d_num->size[2];
    d_num->size[0] = 1;
    d_num->size[1] = 1;
    emxEnsureCapacity_real_T1(d_num, i20);
    ixstart = d_num->size[0];
    n = d_num->size[1];
    loop_ub = d_num->size[2];
    loop_ub *= ixstart * n;
    for (i20 = 0; i20 < loop_ub; i20++) {
      d_num->data[i20] = (d_num->data[i20] + r34->data[i20]) - 2.0 *
        b_denom->data[i20];
    }

    loop_ub = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i20 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    b_denom->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(b_denom, i20);
    for (i20 = 0; i20 < loop_ub; i20++) {
      b_denom->data[b_denom->size[0] * b_denom->size[1] * i20] =
        b_this->BoardCoords->data[((b_idx + b_this->BoardCoords->size[0] * i) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i20) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    b_idx = (int)idx->data[0];
    i20 = r34->size[0] * r34->size[1] * r34->size[2];
    r34->size[0] = 1;
    r34->size[1] = 1;
    r34->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(r34, i20);
    for (i20 = 0; i20 < loop_ub; i20++) {
      r34->data[r34->size[0] * r34->size[1] * i20] = b_this->BoardCoords->data
        [((b_idx + b_this->BoardCoords->size[0] * ((int)((1.0 + (double)i) + 2.0)
            - 1)) + b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] *
          i20) - 1];
    }

    i20 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    emxEnsureCapacity_real_T1(b_denom, i20);
    ixstart = b_denom->size[0];
    n = b_denom->size[1];
    loop_ub = b_denom->size[2];
    loop_ub *= ixstart * n;
    for (i20 = 0; i20 < loop_ub; i20++) {
      b_denom->data[i20] -= r34->data[i20];
    }

    e_num[0] = d_num->size[2];
    c_denom[0] = b_denom->size[2];
    f_num = *d_num;
    f_num.size = (int *)&e_num;
    f_num.numDimensions = 1;
    d_denom = *b_denom;
    d_denom.size = (int *)&c_denom;
    d_denom.numDimensions = 1;
    z = norm(&f_num) / norm(&d_denom);
    if ((newEnergy > z) || rtIsNaN(z)) {
    } else {
      newEnergy = (float)z;
    }

    i++;
  }

  emxFree_real_T(&r34);
  emxFree_real_T(&b_denom);
  emxFree_real_T(&d_num);
  ixstart = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  return newEnergy * (float)ixstart - (float)(b_this->BoardIdx->size[0] *
    b_this->BoardIdx->size[1]);
}

static void d_imfilter(emxArray_real32_T *varargin_1)
{
  int finalSize_idx_0;
  double pad[2];
  int finalSize_idx_1;
  emxArray_real32_T *a;
  emxArray_real_T *b_a;
  int cEnd1;
  int ma;
  emxArray_real_T *result;
  int iv5[2];
  boolean_T b4;
  int cEnd;
  int na;
  int r;
  int firstColB;
  int firstRowA;
  int lastColB;
  int aidx;
  int firstRowB;
  int lastRowB;
  int cidx;
  int lastColA;
  int k;
  int b_firstColB;
  int iC;
  int iA;
  int iB;
  int i;
  int b_i;
  int a_length;
  static const double dv3[841] = { 4.7624975789886592E-8, 1.1073138628049771E-7,
    2.4185958369660052E-7, 4.96263643636689E-7, 9.5657320969368567E-7,
    1.7321303136116703E-6, 2.9464529177220921E-6, 4.7084183560136709E-6,
    7.0681731091607975E-6, 9.967721737900362E-6, 1.32050858376208E-5,
    1.6433994849696377E-5, 1.9213286523034666E-5, 2.1101667084725587E-5,
    2.1771505901906432E-5, 2.1101667084725587E-5, 1.9213286523034666E-5,
    1.6433994849696377E-5, 1.32050858376208E-5, 9.967721737900362E-6,
    7.0681731091607975E-6, 4.7084183560136709E-6, 2.9464529177220921E-6,
    1.7321303136116703E-6, 9.5657320969368567E-7, 4.96263643636689E-7,
    2.4185958369660052E-7, 1.1073138628049771E-7, 4.7624975789886592E-8,
    1.1073138628049771E-7, 2.5745818668121149E-7, 5.6234038010022088E-7,
    1.1538475413182426E-6, 2.2240993476926933E-6, 4.0273236398247893E-6,
    6.8507082844306369E-6, 1.0947400667459265E-5, 1.6433994849696377E-5,
    2.3175647394876874E-5, 3.070274444240722E-5, 3.8210182822179509E-5,
    4.4672229568919195E-5, 4.9062845919963829E-5, 5.0620267831066929E-5,
    4.9062845919963829E-5, 4.4672229568919195E-5, 3.8210182822179509E-5,
    3.070274444240722E-5, 2.3175647394876874E-5, 1.6433994849696377E-5,
    1.0947400667459265E-5, 6.8507082844306369E-6, 4.0273236398247893E-6,
    2.2240993476926933E-6, 1.1538475413182426E-6, 5.6234038010022088E-7,
    2.5745818668121149E-7, 1.1073138628049771E-7, 2.4185958369660052E-7,
    5.6234038010022088E-7, 1.2282643141692652E-6, 2.5202347353049132E-6,
    4.8578795985648672E-6, 8.79648355952221E-6, 1.4963322589514667E-5,
    2.3911321414187968E-5, 3.5895144875646515E-5, 5.0620267831066929E-5,
    6.7060959305370544E-5, 8.3458712301617483E-5, 9.7573120045374732E-5,
    0.00010716310783930673, 0.00011056483004032011, 0.00010716310783930673,
    9.7573120045374732E-5, 8.3458712301617483E-5, 6.7060959305370544E-5,
    5.0620267831066929E-5, 3.5895144875646515E-5, 2.3911321414187968E-5,
    1.4963322589514667E-5, 8.79648355952221E-6, 4.8578795985648672E-6,
    2.5202347353049132E-6, 1.2282643141692652E-6, 5.6234038010022088E-7,
    2.4185958369660052E-7, 4.96263643636689E-7, 1.1538475413182426E-6,
    2.5202347353049132E-6, 5.1711859147624161E-6, 9.967721737900362E-6,
    1.8049212339316872E-5, 3.070274444240722E-5, 4.9062845919963829E-5,
    7.36520550998763E-5, 0.00010386604562762853, 0.00013760015419691919,
    0.0001712461587297812, 0.00020020704300665807, 0.00021988442031918242,
    0.0002268643010760782, 0.00021988442031918242, 0.00020020704300665807,
    0.0001712461587297812, 0.00013760015419691919, 0.00010386604562762853,
    7.36520550998763E-5, 4.9062845919963829E-5, 3.070274444240722E-5,
    1.8049212339316872E-5, 9.967721737900362E-6, 5.1711859147624161E-6,
    2.5202347353049132E-6, 1.1538475413182426E-6, 4.96263643636689E-7,
    9.5657320969368567E-7, 2.2240993476926933E-6, 4.8578795985648672E-6,
    9.967721737900362E-6, 1.9213286523034666E-5, 3.4790767369819838E-5,
    5.9181088871341108E-5, 9.4571110739527153E-5, 0.00014196805196353137,
    0.00020020704300665807, 0.00026523123916539511, 0.00033008561034905253,
    0.00038590917587419675, 0.00042383831337106966, 0.00043729238566613991,
    0.00042383831337106966, 0.00038590917587419675, 0.00033008561034905253,
    0.00026523123916539511, 0.00020020704300665807, 0.00014196805196353137,
    9.4571110739527153E-5, 5.9181088871341108E-5, 3.4790767369819838E-5,
    1.9213286523034666E-5, 9.967721737900362E-6, 4.8578795985648672E-6,
    2.2240993476926933E-6, 9.5657320969368567E-7, 1.7321303136116703E-6,
    4.0273236398247893E-6, 8.79648355952221E-6, 1.8049212339316872E-5,
    3.4790767369819838E-5, 6.2997941176268E-5, 0.00010716310783930673,
    0.0001712461587297812, 0.00025707093182044473, 0.00036252812087580349,
    0.00048027172914688035, 0.00059770782411488436, 0.000698791242592549,
    0.00076747203791663766, 0.00079183421556039239, 0.00076747203791663766,
    0.000698791242592549, 0.00059770782411488436, 0.00048027172914688035,
    0.00036252812087580349, 0.00025707093182044473, 0.0001712461587297812,
    0.00010716310783930673, 6.2997941176268E-5, 3.4790767369819838E-5,
    1.8049212339316872E-5, 8.79648355952221E-6, 4.0273236398247893E-6,
    1.7321303136116703E-6, 2.9464529177220921E-6, 6.8507082844306369E-6,
    1.4963322589514667E-5, 3.070274444240722E-5, 5.9181088871341108E-5,
    0.00010716310783930673, 0.00018229058707882028, 0.00029129952872078451,
    0.00043729238566613991, 0.00061668110714116519, 0.00081696973173665821,
    0.0010167352585825436, 0.0011886839456797726, 0.0013055139140625692,
    0.0013469553742324059, 0.0013055139140625692, 0.0011886839456797726,
    0.0010167352585825436, 0.00081696973173665821, 0.00061668110714116519,
    0.00043729238566613991, 0.00029129952872078451, 0.00018229058707882028,
    0.00010716310783930673, 5.9181088871341108E-5, 3.070274444240722E-5,
    1.4963322589514667E-5, 6.8507082844306369E-6, 2.9464529177220921E-6,
    4.7084183560136709E-6, 1.0947400667459265E-5, 2.3911321414187968E-5,
    4.9062845919963829E-5, 9.4571110739527153E-5, 0.0001712461587297812,
    0.00029129952872078451, 0.00046549532146857752, 0.000698791242592549,
    0.00098545360328210077, 0.0013055139140625692, 0.0016247383170192853,
    0.0018995115366256426, 0.0020862052945192278, 0.002152428559309937,
    0.0020862052945192278, 0.0018995115366256426, 0.0016247383170192853,
    0.0013055139140625692, 0.00098545360328210077, 0.000698791242592549,
    0.00046549532146857752, 0.00029129952872078451, 0.0001712461587297812,
    9.4571110739527153E-5, 4.9062845919963829E-5, 2.3911321414187968E-5,
    1.0947400667459265E-5, 4.7084183560136709E-6, 7.0681731091607975E-6,
    1.6433994849696377E-5, 3.5895144875646515E-5, 7.36520550998763E-5,
    0.00014196805196353137, 0.00025707093182044473, 0.00043729238566613991,
    0.000698791242592549, 0.0010490099002144347, 0.0014793410721772179,
    0.0019598085053819971, 0.0024390210923189125, 0.0028515045496265343,
    0.0031317650743749826, 0.0032311779693223251, 0.0031317650743749826,
    0.0028515045496265343, 0.0024390210923189125, 0.0019598085053819971,
    0.0014793410721772179, 0.0010490099002144347, 0.000698791242592549,
    0.00043729238566613991, 0.00025707093182044473, 0.00014196805196353137,
    7.36520550998763E-5, 3.5895144875646515E-5, 1.6433994849696377E-5,
    7.0681731091607975E-6, 9.967721737900362E-6, 2.3175647394876874E-5,
    5.0620267831066929E-5, 0.00010386604562762853, 0.00020020704300665807,
    0.00036252812087580349, 0.00061668110714116519, 0.00098545360328210077,
    0.0014793410721772179, 0.0020862052945192278, 0.0027637729777585367,
    0.0034395710441210759, 0.0040212659545924524, 0.0044164966431546543,
    0.0045566912958167321, 0.0044164966431546543, 0.0040212659545924524,
    0.0034395710441210759, 0.0027637729777585367, 0.0020862052945192278,
    0.0014793410721772179, 0.00098545360328210077, 0.00061668110714116519,
    0.00036252812087580349, 0.00020020704300665807, 0.00010386604562762853,
    5.0620267831066929E-5, 2.3175647394876874E-5, 9.967721737900362E-6,
    1.32050858376208E-5, 3.070274444240722E-5, 6.7060959305370544E-5,
    0.00013760015419691919, 0.00026523123916539511, 0.00048027172914688035,
    0.00081696973173665821, 0.0013055139140625692, 0.0019598085053819971,
    0.0027637729777585367, 0.0036614043175212017, 0.0045566912958167321,
    0.0053273118474393638, 0.0058509074398284842, 0.0060366351789304913,
    0.0058509074398284842, 0.0053273118474393638, 0.0045566912958167321,
    0.0036614043175212017, 0.0027637729777585367, 0.0019598085053819971,
    0.0013055139140625692, 0.00081696973173665821, 0.00048027172914688035,
    0.00026523123916539511, 0.00013760015419691919, 6.7060959305370544E-5,
    3.070274444240722E-5, 1.32050858376208E-5, 1.6433994849696377E-5,
    3.8210182822179509E-5, 8.3458712301617483E-5, 0.0001712461587297812,
    0.00033008561034905253, 0.00059770782411488436, 0.0010167352585825436,
    0.0016247383170192853, 0.0024390210923189125, 0.0034395710441210759,
    0.0045566912958167321, 0.0056708939425266667, 0.0066299467144788314,
    0.0072815719575447923, 0.007512713863427176, 0.0072815719575447923,
    0.0066299467144788314, 0.0056708939425266667, 0.0045566912958167321,
    0.0034395710441210759, 0.0024390210923189125, 0.0016247383170192853,
    0.0010167352585825436, 0.00059770782411488436, 0.00033008561034905253,
    0.0001712461587297812, 8.3458712301617483E-5, 3.8210182822179509E-5,
    1.6433994849696377E-5, 1.9213286523034666E-5, 4.4672229568919195E-5,
    9.7573120045374732E-5, 0.00020020704300665807, 0.00038590917587419675,
    0.000698791242592549, 0.0011886839456797726, 0.0018995115366256426,
    0.0028515045496265343, 0.0040212659545924524, 0.0053273118474393638,
    0.0066299467144788314, 0.0077511930010181031, 0.0085130200926762049,
    0.0087832523585260752, 0.0085130200926762049, 0.0077511930010181031,
    0.0066299467144788314, 0.0053273118474393638, 0.0040212659545924524,
    0.0028515045496265343, 0.0018995115366256426, 0.0011886839456797726,
    0.000698791242592549, 0.00038590917587419675, 0.00020020704300665807,
    9.7573120045374732E-5, 4.4672229568919195E-5, 1.9213286523034666E-5,
    2.1101667084725587E-5, 4.9062845919963829E-5, 0.00010716310783930673,
    0.00021988442031918242, 0.00042383831337106966, 0.00076747203791663766,
    0.0013055139140625692, 0.0020862052945192278, 0.0031317650743749826,
    0.0044164966431546543, 0.0058509074398284842, 0.0072815719575447923,
    0.0085130200926762049, 0.009349723466928226, 0.0096465155489428526,
    0.009349723466928226, 0.0085130200926762049, 0.0072815719575447923,
    0.0058509074398284842, 0.0044164966431546543, 0.0031317650743749826,
    0.0020862052945192278, 0.0013055139140625692, 0.00076747203791663766,
    0.00042383831337106966, 0.00021988442031918242, 0.00010716310783930673,
    4.9062845919963829E-5, 2.1101667084725587E-5, 2.1771505901906432E-5,
    5.0620267831066929E-5, 0.00011056483004032011, 0.0002268643010760782,
    0.00043729238566613991, 0.00079183421556039239, 0.0013469553742324059,
    0.002152428559309937, 0.0032311779693223251, 0.0045566912958167321,
    0.0060366351789304913, 0.007512713863427176, 0.0087832523585260752,
    0.0096465155489428526, 0.0099527288229593752, 0.0096465155489428526,
    0.0087832523585260752, 0.007512713863427176, 0.0060366351789304913,
    0.0045566912958167321, 0.0032311779693223251, 0.002152428559309937,
    0.0013469553742324059, 0.00079183421556039239, 0.00043729238566613991,
    0.0002268643010760782, 0.00011056483004032011, 5.0620267831066929E-5,
    2.1771505901906432E-5, 2.1101667084725587E-5, 4.9062845919963829E-5,
    0.00010716310783930673, 0.00021988442031918242, 0.00042383831337106966,
    0.00076747203791663766, 0.0013055139140625692, 0.0020862052945192278,
    0.0031317650743749826, 0.0044164966431546543, 0.0058509074398284842,
    0.0072815719575447923, 0.0085130200926762049, 0.009349723466928226,
    0.0096465155489428526, 0.009349723466928226, 0.0085130200926762049,
    0.0072815719575447923, 0.0058509074398284842, 0.0044164966431546543,
    0.0031317650743749826, 0.0020862052945192278, 0.0013055139140625692,
    0.00076747203791663766, 0.00042383831337106966, 0.00021988442031918242,
    0.00010716310783930673, 4.9062845919963829E-5, 2.1101667084725587E-5,
    1.9213286523034666E-5, 4.4672229568919195E-5, 9.7573120045374732E-5,
    0.00020020704300665807, 0.00038590917587419675, 0.000698791242592549,
    0.0011886839456797726, 0.0018995115366256426, 0.0028515045496265343,
    0.0040212659545924524, 0.0053273118474393638, 0.0066299467144788314,
    0.0077511930010181031, 0.0085130200926762049, 0.0087832523585260752,
    0.0085130200926762049, 0.0077511930010181031, 0.0066299467144788314,
    0.0053273118474393638, 0.0040212659545924524, 0.0028515045496265343,
    0.0018995115366256426, 0.0011886839456797726, 0.000698791242592549,
    0.00038590917587419675, 0.00020020704300665807, 9.7573120045374732E-5,
    4.4672229568919195E-5, 1.9213286523034666E-5, 1.6433994849696377E-5,
    3.8210182822179509E-5, 8.3458712301617483E-5, 0.0001712461587297812,
    0.00033008561034905253, 0.00059770782411488436, 0.0010167352585825436,
    0.0016247383170192853, 0.0024390210923189125, 0.0034395710441210759,
    0.0045566912958167321, 0.0056708939425266667, 0.0066299467144788314,
    0.0072815719575447923, 0.007512713863427176, 0.0072815719575447923,
    0.0066299467144788314, 0.0056708939425266667, 0.0045566912958167321,
    0.0034395710441210759, 0.0024390210923189125, 0.0016247383170192853,
    0.0010167352585825436, 0.00059770782411488436, 0.00033008561034905253,
    0.0001712461587297812, 8.3458712301617483E-5, 3.8210182822179509E-5,
    1.6433994849696377E-5, 1.32050858376208E-5, 3.070274444240722E-5,
    6.7060959305370544E-5, 0.00013760015419691919, 0.00026523123916539511,
    0.00048027172914688035, 0.00081696973173665821, 0.0013055139140625692,
    0.0019598085053819971, 0.0027637729777585367, 0.0036614043175212017,
    0.0045566912958167321, 0.0053273118474393638, 0.0058509074398284842,
    0.0060366351789304913, 0.0058509074398284842, 0.0053273118474393638,
    0.0045566912958167321, 0.0036614043175212017, 0.0027637729777585367,
    0.0019598085053819971, 0.0013055139140625692, 0.00081696973173665821,
    0.00048027172914688035, 0.00026523123916539511, 0.00013760015419691919,
    6.7060959305370544E-5, 3.070274444240722E-5, 1.32050858376208E-5,
    9.967721737900362E-6, 2.3175647394876874E-5, 5.0620267831066929E-5,
    0.00010386604562762853, 0.00020020704300665807, 0.00036252812087580349,
    0.00061668110714116519, 0.00098545360328210077, 0.0014793410721772179,
    0.0020862052945192278, 0.0027637729777585367, 0.0034395710441210759,
    0.0040212659545924524, 0.0044164966431546543, 0.0045566912958167321,
    0.0044164966431546543, 0.0040212659545924524, 0.0034395710441210759,
    0.0027637729777585367, 0.0020862052945192278, 0.0014793410721772179,
    0.00098545360328210077, 0.00061668110714116519, 0.00036252812087580349,
    0.00020020704300665807, 0.00010386604562762853, 5.0620267831066929E-5,
    2.3175647394876874E-5, 9.967721737900362E-6, 7.0681731091607975E-6,
    1.6433994849696377E-5, 3.5895144875646515E-5, 7.36520550998763E-5,
    0.00014196805196353137, 0.00025707093182044473, 0.00043729238566613991,
    0.000698791242592549, 0.0010490099002144347, 0.0014793410721772179,
    0.0019598085053819971, 0.0024390210923189125, 0.0028515045496265343,
    0.0031317650743749826, 0.0032311779693223251, 0.0031317650743749826,
    0.0028515045496265343, 0.0024390210923189125, 0.0019598085053819971,
    0.0014793410721772179, 0.0010490099002144347, 0.000698791242592549,
    0.00043729238566613991, 0.00025707093182044473, 0.00014196805196353137,
    7.36520550998763E-5, 3.5895144875646515E-5, 1.6433994849696377E-5,
    7.0681731091607975E-6, 4.7084183560136709E-6, 1.0947400667459265E-5,
    2.3911321414187968E-5, 4.9062845919963829E-5, 9.4571110739527153E-5,
    0.0001712461587297812, 0.00029129952872078451, 0.00046549532146857752,
    0.000698791242592549, 0.00098545360328210077, 0.0013055139140625692,
    0.0016247383170192853, 0.0018995115366256426, 0.0020862052945192278,
    0.002152428559309937, 0.0020862052945192278, 0.0018995115366256426,
    0.0016247383170192853, 0.0013055139140625692, 0.00098545360328210077,
    0.000698791242592549, 0.00046549532146857752, 0.00029129952872078451,
    0.0001712461587297812, 9.4571110739527153E-5, 4.9062845919963829E-5,
    2.3911321414187968E-5, 1.0947400667459265E-5, 4.7084183560136709E-6,
    2.9464529177220921E-6, 6.8507082844306369E-6, 1.4963322589514667E-5,
    3.070274444240722E-5, 5.9181088871341108E-5, 0.00010716310783930673,
    0.00018229058707882028, 0.00029129952872078451, 0.00043729238566613991,
    0.00061668110714116519, 0.00081696973173665821, 0.0010167352585825436,
    0.0011886839456797726, 0.0013055139140625692, 0.0013469553742324059,
    0.0013055139140625692, 0.0011886839456797726, 0.0010167352585825436,
    0.00081696973173665821, 0.00061668110714116519, 0.00043729238566613991,
    0.00029129952872078451, 0.00018229058707882028, 0.00010716310783930673,
    5.9181088871341108E-5, 3.070274444240722E-5, 1.4963322589514667E-5,
    6.8507082844306369E-6, 2.9464529177220921E-6, 1.7321303136116703E-6,
    4.0273236398247893E-6, 8.79648355952221E-6, 1.8049212339316872E-5,
    3.4790767369819838E-5, 6.2997941176268E-5, 0.00010716310783930673,
    0.0001712461587297812, 0.00025707093182044473, 0.00036252812087580349,
    0.00048027172914688035, 0.00059770782411488436, 0.000698791242592549,
    0.00076747203791663766, 0.00079183421556039239, 0.00076747203791663766,
    0.000698791242592549, 0.00059770782411488436, 0.00048027172914688035,
    0.00036252812087580349, 0.00025707093182044473, 0.0001712461587297812,
    0.00010716310783930673, 6.2997941176268E-5, 3.4790767369819838E-5,
    1.8049212339316872E-5, 8.79648355952221E-6, 4.0273236398247893E-6,
    1.7321303136116703E-6, 9.5657320969368567E-7, 2.2240993476926933E-6,
    4.8578795985648672E-6, 9.967721737900362E-6, 1.9213286523034666E-5,
    3.4790767369819838E-5, 5.9181088871341108E-5, 9.4571110739527153E-5,
    0.00014196805196353137, 0.00020020704300665807, 0.00026523123916539511,
    0.00033008561034905253, 0.00038590917587419675, 0.00042383831337106966,
    0.00043729238566613991, 0.00042383831337106966, 0.00038590917587419675,
    0.00033008561034905253, 0.00026523123916539511, 0.00020020704300665807,
    0.00014196805196353137, 9.4571110739527153E-5, 5.9181088871341108E-5,
    3.4790767369819838E-5, 1.9213286523034666E-5, 9.967721737900362E-6,
    4.8578795985648672E-6, 2.2240993476926933E-6, 9.5657320969368567E-7,
    4.96263643636689E-7, 1.1538475413182426E-6, 2.5202347353049132E-6,
    5.1711859147624161E-6, 9.967721737900362E-6, 1.8049212339316872E-5,
    3.070274444240722E-5, 4.9062845919963829E-5, 7.36520550998763E-5,
    0.00010386604562762853, 0.00013760015419691919, 0.0001712461587297812,
    0.00020020704300665807, 0.00021988442031918242, 0.0002268643010760782,
    0.00021988442031918242, 0.00020020704300665807, 0.0001712461587297812,
    0.00013760015419691919, 0.00010386604562762853, 7.36520550998763E-5,
    4.9062845919963829E-5, 3.070274444240722E-5, 1.8049212339316872E-5,
    9.967721737900362E-6, 5.1711859147624161E-6, 2.5202347353049132E-6,
    1.1538475413182426E-6, 4.96263643636689E-7, 2.4185958369660052E-7,
    5.6234038010022088E-7, 1.2282643141692652E-6, 2.5202347353049132E-6,
    4.8578795985648672E-6, 8.79648355952221E-6, 1.4963322589514667E-5,
    2.3911321414187968E-5, 3.5895144875646515E-5, 5.0620267831066929E-5,
    6.7060959305370544E-5, 8.3458712301617483E-5, 9.7573120045374732E-5,
    0.00010716310783930673, 0.00011056483004032011, 0.00010716310783930673,
    9.7573120045374732E-5, 8.3458712301617483E-5, 6.7060959305370544E-5,
    5.0620267831066929E-5, 3.5895144875646515E-5, 2.3911321414187968E-5,
    1.4963322589514667E-5, 8.79648355952221E-6, 4.8578795985648672E-6,
    2.5202347353049132E-6, 1.2282643141692652E-6, 5.6234038010022088E-7,
    2.4185958369660052E-7, 1.1073138628049771E-7, 2.5745818668121149E-7,
    5.6234038010022088E-7, 1.1538475413182426E-6, 2.2240993476926933E-6,
    4.0273236398247893E-6, 6.8507082844306369E-6, 1.0947400667459265E-5,
    1.6433994849696377E-5, 2.3175647394876874E-5, 3.070274444240722E-5,
    3.8210182822179509E-5, 4.4672229568919195E-5, 4.9062845919963829E-5,
    5.0620267831066929E-5, 4.9062845919963829E-5, 4.4672229568919195E-5,
    3.8210182822179509E-5, 3.070274444240722E-5, 2.3175647394876874E-5,
    1.6433994849696377E-5, 1.0947400667459265E-5, 6.8507082844306369E-6,
    4.0273236398247893E-6, 2.2240993476926933E-6, 1.1538475413182426E-6,
    5.6234038010022088E-7, 2.5745818668121149E-7, 1.1073138628049771E-7,
    4.7624975789886592E-8, 1.1073138628049771E-7, 2.4185958369660052E-7,
    4.96263643636689E-7, 9.5657320969368567E-7, 1.7321303136116703E-6,
    2.9464529177220921E-6, 4.7084183560136709E-6, 7.0681731091607975E-6,
    9.967721737900362E-6, 1.32050858376208E-5, 1.6433994849696377E-5,
    1.9213286523034666E-5, 2.1101667084725587E-5, 2.1771505901906432E-5,
    2.1101667084725587E-5, 1.9213286523034666E-5, 1.6433994849696377E-5,
    1.32050858376208E-5, 9.967721737900362E-6, 7.0681731091607975E-6,
    4.7084183560136709E-6, 2.9464529177220921E-6, 1.7321303136116703E-6,
    9.5657320969368567E-7, 4.96263643636689E-7, 2.4185958369660052E-7,
    1.1073138628049771E-7, 4.7624975789886592E-8 };

  finalSize_idx_0 = varargin_1->size[0];
  pad[0] = 14.0;
  finalSize_idx_1 = varargin_1->size[1];
  pad[1] = 14.0;
  if (!((varargin_1->size[0] == 0) || (varargin_1->size[1] == 0))) {
    emxInit_real32_T(&a, 2);
    emxInit_real_T1(&b_a, 2);
    padImage(varargin_1, pad, a);
    cEnd1 = b_a->size[0] * b_a->size[1];
    b_a->size[0] = a->size[0];
    b_a->size[1] = a->size[1];
    emxEnsureCapacity_real_T(b_a, cEnd1);
    ma = a->size[0] * a->size[1];
    for (cEnd1 = 0; cEnd1 < ma; cEnd1++) {
      b_a->data[cEnd1] = a->data[cEnd1];
    }

    emxFree_real32_T(&a);
    for (cEnd1 = 0; cEnd1 < 2; cEnd1++) {
      iv5[cEnd1] = b_a->size[cEnd1];
    }

    emxInit_real_T1(&result, 2);
    cEnd1 = result->size[0] * result->size[1];
    result->size[0] = iv5[0];
    result->size[1] = iv5[1];
    emxEnsureCapacity_real_T(result, cEnd1);
    ma = iv5[0] * iv5[1];
    for (cEnd1 = 0; cEnd1 < ma; cEnd1++) {
      result->data[cEnd1] = 0.0;
    }

    if ((b_a->size[0] == 0) || (iv5[0] == 0)) {
      b4 = true;
    } else {
      b4 = false;
    }

    if (!b4) {
      cEnd = iv5[1];
      cEnd1 = iv5[0];
      ma = b_a->size[0];
      na = b_a->size[1] - 14;
      if (b_a->size[1] < 14) {
        firstColB = 15 - b_a->size[1];
      } else {
        firstColB = 0;
      }

      if (29 <= iv5[1] + 13) {
        lastColB = 28;
      } else {
        lastColB = iv5[1] + 13;
      }

      if (b_a->size[0] < 14) {
        firstRowB = 15 - b_a->size[0];
      } else {
        firstRowB = 0;
      }

      if (29 <= iv5[0] + 13) {
        lastRowB = 28;
      } else {
        lastRowB = iv5[0] + 13;
      }

      while (firstColB <= lastColB) {
        if ((firstColB + na) + 13 < cEnd + 13) {
          lastColA = na;
        } else {
          lastColA = cEnd - firstColB;
        }

        if (firstColB < 14) {
          k = 14 - firstColB;
        } else {
          k = 0;
        }

        while (k <= lastColA + 13) {
          if (firstColB + k > 14) {
            b_firstColB = (firstColB + k) - 14;
          } else {
            b_firstColB = 0;
          }

          iC = b_firstColB * cEnd1;
          iA = k * ma;
          iB = firstRowB + firstColB * 29;
          for (i = firstRowB; i <= lastRowB; i++) {
            if (i < 14) {
              firstRowA = 14 - i;
            } else {
              firstRowA = 0;
            }

            if (i + ma <= cEnd1 + 13) {
              b_i = ma;
            } else {
              b_i = (cEnd1 - i) + 14;
            }

            a_length = b_i - firstRowA;
            aidx = iA + firstRowA;
            cidx = iC;
            for (r = 1; r <= a_length; r++) {
              result->data[cidx] += dv3[iB] * b_a->data[aidx];
              aidx++;
              cidx++;
            }

            iB++;
            if (i >= 14) {
              iC++;
            }
          }

          k++;
        }

        firstColB++;
      }
    }

    emxFree_real_T(&b_a);
    if (15.0 > (double)finalSize_idx_0 + 14.0) {
      cEnd1 = 0;
      r = 0;
    } else {
      cEnd1 = 14;
      r = (int)((double)finalSize_idx_0 + 14.0);
    }

    if (15.0 > (double)finalSize_idx_1 + 14.0) {
      cEnd = 0;
      firstRowA = 0;
    } else {
      cEnd = 14;
      firstRowA = (int)((double)finalSize_idx_1 + 14.0);
    }

    aidx = varargin_1->size[0] * varargin_1->size[1];
    varargin_1->size[0] = r - cEnd1;
    varargin_1->size[1] = firstRowA - cEnd;
    emxEnsureCapacity_real32_T(varargin_1, aidx);
    ma = firstRowA - cEnd;
    for (firstRowA = 0; firstRowA < ma; firstRowA++) {
      cidx = r - cEnd1;
      for (aidx = 0; aidx < cidx; aidx++) {
        varargin_1->data[aidx + varargin_1->size[0] * firstRowA] = (float)
          result->data[(cEnd1 + aidx) + result->size[0] * (cEnd + firstRowA)];
      }
    }

    emxFree_real_T(&result);
  }
}

static void detectCheckerboard(const emxArray_real32_T *I, emxArray_real_T
  *points, double boardSize[2])
{
  c_vision_internal_calibration_c lobj_5;
  c_vision_internal_calibration_c lobj_4;
  c_vision_internal_calibration_c lobj_3;
  c_vision_internal_calibration_c lobj_2;
  c_vision_internal_calibration_c lobj_1;
  c_vision_internal_calibration_c lobj_0;
  emxArray_real32_T *Ig;
  int i;
  int loop_ub;
  emxArray_real32_T *Iy;
  emxArray_real32_T *I_45;
  emxArray_real32_T *Ixy;
  emxArray_real32_T *I_45_x;
  emxArray_real32_T *I_45_y;
  int psiz;
  emxArray_real32_T *r44;
  emxArray_real32_T *r45;
  emxArray_real32_T *cxy;
  emxArray_real_T *r46;
  int end;
  emxArray_int32_T *r47;
  emxArray_real32_T *c45;
  emxArray_int32_T *r48;
  emxArray_real32_T *points0;
  emxArray_int32_T *r49;
  emxArray_real32_T *b_cxy;
  c_vision_internal_calibration_c *board0;
  c_vision_internal_calibration_c *board45;
  c_emxInitStruct_vision_internal(&lobj_5);
  c_emxInitStruct_vision_internal(&lobj_4);
  c_emxInitStruct_vision_internal(&lobj_3);
  c_emxInitStruct_vision_internal(&lobj_2);
  c_emxInitStruct_vision_internal(&lobj_1);
  c_emxInitStruct_vision_internal(&lobj_0);
  emxInit_real32_T(&Ig, 2);
  i = Ig->size[0] * Ig->size[1];
  Ig->size[0] = I->size[0];
  Ig->size[1] = I->size[1];
  emxEnsureCapacity_real32_T(Ig, i);
  loop_ub = I->size[0] * I->size[1];
  for (i = 0; i < loop_ub; i++) {
    Ig->data[i] = I->data[i];
  }

  emxInit_real32_T(&Iy, 2);
  d_imfilter(Ig);
  i = Iy->size[0] * Iy->size[1];
  Iy->size[0] = Ig->size[0];
  Iy->size[1] = Ig->size[1];
  emxEnsureCapacity_real32_T(Iy, i);
  loop_ub = Ig->size[0] * Ig->size[1];
  for (i = 0; i < loop_ub; i++) {
    Iy->data[i] = Ig->data[i];
  }

  emxInit_real32_T(&I_45, 2);
  imfilter(Iy);
  b_imfilter(Ig);
  i = I_45->size[0] * I_45->size[1];
  I_45->size[0] = Ig->size[0];
  I_45->size[1] = Ig->size[1];
  emxEnsureCapacity_real32_T(I_45, i);
  loop_ub = Ig->size[0] * Ig->size[1];
  for (i = 0; i < loop_ub; i++) {
    I_45->data[i] = Ig->data[i] * 0.707106769F + Iy->data[i] * 0.707106769F;
  }

  emxInit_real32_T(&Ixy, 2);
  i = Ixy->size[0] * Ixy->size[1];
  Ixy->size[0] = Ig->size[0];
  Ixy->size[1] = Ig->size[1];
  emxEnsureCapacity_real32_T(Ixy, i);
  loop_ub = Ig->size[0] * Ig->size[1];
  for (i = 0; i < loop_ub; i++) {
    Ixy->data[i] = Ig->data[i];
  }

  emxInit_real32_T(&I_45_x, 2);
  imfilter(Ixy);
  i = I_45_x->size[0] * I_45_x->size[1];
  I_45_x->size[0] = I_45->size[0];
  I_45_x->size[1] = I_45->size[1];
  emxEnsureCapacity_real32_T(I_45_x, i);
  loop_ub = I_45->size[0] * I_45->size[1];
  for (i = 0; i < loop_ub; i++) {
    I_45_x->data[i] = I_45->data[i];
  }

  emxInit_real32_T(&I_45_y, 2);
  b_imfilter(I_45_x);
  i = I_45_y->size[0] * I_45_y->size[1];
  I_45_y->size[0] = I_45->size[0];
  I_45_y->size[1] = I_45->size[1];
  emxEnsureCapacity_real32_T(I_45_y, i);
  loop_ub = I_45->size[0] * I_45->size[1];
  for (i = 0; i < loop_ub; i++) {
    I_45_y->data[i] = I_45->data[i];
  }

  imfilter(I_45_y);
  i = I_45_x->size[0] * I_45_x->size[1];
  emxEnsureCapacity_real32_T(I_45_x, i);
  loop_ub = I_45_x->size[0];
  psiz = I_45_x->size[1];
  loop_ub *= psiz;
  for (i = 0; i < loop_ub; i++) {
    I_45_x->data[i] = I_45_x->data[i] * 0.707106769F + I_45_y->data[i] *
      -0.707106769F;
  }

  emxInit_real32_T(&r44, 2);
  b_abs(Ixy, r44);
  i = r44->size[0] * r44->size[1];
  emxEnsureCapacity_real32_T(r44, i);
  i = r44->size[0];
  loop_ub = r44->size[1];
  loop_ub *= i;
  for (i = 0; i < loop_ub; i++) {
    r44->data[i] *= 16.0F;
  }

  emxInit_real32_T(&r45, 2);
  b_abs(I_45, r45);
  i = I_45->size[0] * I_45->size[1];
  I_45->size[0] = Ig->size[0];
  I_45->size[1] = Ig->size[1];
  emxEnsureCapacity_real32_T(I_45, i);
  loop_ub = Ig->size[0] * Ig->size[1];
  for (i = 0; i < loop_ub; i++) {
    I_45->data[i] = Ig->data[i] * 0.707106769F + Iy->data[i] * -0.707106769F;
  }

  b_abs(I_45, I_45_y);
  i = r45->size[0] * r45->size[1];
  emxEnsureCapacity_real32_T(r45, i);
  i = r45->size[0];
  loop_ub = r45->size[1];
  loop_ub *= i;
  for (i = 0; i < loop_ub; i++) {
    r45->data[i] = 6.0F * (r45->data[i] + I_45_y->data[i]);
  }

  emxInit_real32_T(&cxy, 2);
  i = cxy->size[0] * cxy->size[1];
  cxy->size[0] = r44->size[0];
  cxy->size[1] = r44->size[1];
  emxEnsureCapacity_real32_T(cxy, i);
  loop_ub = r44->size[0] * r44->size[1];
  for (i = 0; i < loop_ub; i++) {
    cxy->data[i] = r44->data[i] - r45->data[i];
  }

  emxInit_real_T1(&r46, 2);
  i = r46->size[0] * r46->size[1];
  r46->size[0] = r44->size[0];
  r46->size[1] = r44->size[1];
  emxEnsureCapacity_real_T(r46, i);
  loop_ub = r44->size[0] * r44->size[1];
  for (i = 0; i < loop_ub; i++) {
    r46->data[i] = r44->data[i] - r45->data[i];
  }

  end = r46->size[0] * r46->size[1] - 1;
  loop_ub = 0;
  for (i = 0; i <= end; i++) {
    if (r44->data[i] - r45->data[i] < 0.0F) {
      loop_ub++;
    }
  }

  emxInit_int32_T(&r47, 1);
  i = r47->size[0];
  r47->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r47, i);
  psiz = 0;
  for (i = 0; i <= end; i++) {
    if (r44->data[i] - r45->data[i] < 0.0F) {
      r47->data[psiz] = i + 1;
      psiz++;
    }
  }

  loop_ub = r47->size[0];
  for (i = 0; i < loop_ub; i++) {
    cxy->data[r47->data[i] - 1] = 0.0F;
  }

  emxFree_int32_T(&r47);
  b_abs(I_45_x, r44);
  i = r44->size[0] * r44->size[1];
  emxEnsureCapacity_real32_T(r44, i);
  i = r44->size[0];
  loop_ub = r44->size[1];
  loop_ub *= i;
  for (i = 0; i < loop_ub; i++) {
    r44->data[i] *= 16.0F;
  }

  b_abs(Ig, r45);
  b_abs(Iy, I_45_y);
  i = r45->size[0] * r45->size[1];
  emxEnsureCapacity_real32_T(r45, i);
  i = r45->size[0];
  loop_ub = r45->size[1];
  loop_ub *= i;
  for (i = 0; i < loop_ub; i++) {
    r45->data[i] = 6.0F * (r45->data[i] + I_45_y->data[i]);
  }

  emxInit_real32_T(&c45, 2);
  i = c45->size[0] * c45->size[1];
  c45->size[0] = r44->size[0];
  c45->size[1] = r44->size[1];
  emxEnsureCapacity_real32_T(c45, i);
  loop_ub = r44->size[0] * r44->size[1];
  for (i = 0; i < loop_ub; i++) {
    c45->data[i] = r44->data[i] - r45->data[i];
  }

  i = r46->size[0] * r46->size[1];
  r46->size[0] = r44->size[0];
  r46->size[1] = r44->size[1];
  emxEnsureCapacity_real_T(r46, i);
  loop_ub = r44->size[0] * r44->size[1];
  for (i = 0; i < loop_ub; i++) {
    r46->data[i] = r44->data[i] - r45->data[i];
  }

  end = r46->size[0] * r46->size[1] - 1;
  loop_ub = 0;
  emxFree_real_T(&r46);
  for (i = 0; i <= end; i++) {
    if (r44->data[i] - r45->data[i] < 0.0F) {
      loop_ub++;
    }
  }

  emxInit_int32_T(&r48, 1);
  i = r48->size[0];
  r48->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r48, i);
  psiz = 0;
  for (i = 0; i <= end; i++) {
    if (r44->data[i] - r45->data[i] < 0.0F) {
      r48->data[psiz] = i + 1;
      psiz++;
    }
  }

  emxFree_real32_T(&r45);
  emxFree_real32_T(&r44);
  loop_ub = r48->size[0];
  for (i = 0; i < loop_ub; i++) {
    c45->data[r48->data[i] - 1] = 0.0F;
  }

  emxFree_int32_T(&r48);
  power(Ig, I_45_y);
  power(Iy, I_45);
  c_imfilter(I_45_y);
  c_imfilter(I_45);
  i = Ig->size[0] * Ig->size[1];
  emxEnsureCapacity_real32_T(Ig, i);
  loop_ub = Ig->size[0];
  psiz = Ig->size[1];
  loop_ub *= psiz;
  for (i = 0; i < loop_ub; i++) {
    Ig->data[i] *= Iy->data[i];
  }

  emxFree_real32_T(&Iy);
  emxInit_real32_T(&points0, 2);
  emxInit_int32_T(&r49, 1);
  c_imfilter(Ig);
  find_peaks(cxy, points0);
  psiz = cxy->size[0];
  loop_ub = points0->size[0];
  i = r49->size[0];
  r49->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r49, i);
  for (i = 0; i < loop_ub; i++) {
    r49->data[i] = (int)points0->data[i + points0->size[0]] + psiz * ((int)
      points0->data[i] - 1);
  }

  emxInit_real32_T1(&b_cxy, 1);
  i = b_cxy->size[0];
  b_cxy->size[0] = r49->size[0];
  emxEnsureCapacity_real32_T1(b_cxy, i);
  loop_ub = r49->size[0];
  for (i = 0; i < loop_ub; i++) {
    b_cxy->data[i] = cxy->data[r49->data[i] - 1];
  }

  emxFree_real32_T(&cxy);
  board0 = growCheckerboard(points0, b_cxy, I_45_y, I_45, Ig, 0.0, &lobj_0,
    &lobj_1, &lobj_2);
  find_peaks(c45, points0);
  psiz = c45->size[0];
  loop_ub = points0->size[0];
  i = r49->size[0];
  r49->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r49, i);
  for (i = 0; i < loop_ub; i++) {
    r49->data[i] = (int)points0->data[i + points0->size[0]] + psiz * ((int)
      points0->data[i] - 1);
  }

  i = b_cxy->size[0];
  b_cxy->size[0] = r49->size[0];
  emxEnsureCapacity_real32_T1(b_cxy, i);
  loop_ub = r49->size[0];
  for (i = 0; i < loop_ub; i++) {
    b_cxy->data[i] = c45->data[r49->data[i] - 1];
  }

  emxFree_int32_T(&r49);
  emxFree_real32_T(&c45);
  board45 = growCheckerboard(points0, b_cxy, I_45_y, I_45, Ig,
    0.78539816339744828, &lobj_3, &lobj_4, &lobj_5);
  i = points->size[0] * points->size[1];
  points->size[0] = 0;
  points->size[1] = 0;
  emxEnsureCapacity_real_T(points, i);
  emxFree_real32_T(&b_cxy);
  emxFree_real32_T(&I_45_y);
  emxFree_real32_T(&I_45);
  emxFree_real32_T(&Ig);
  emxFree_real32_T(&points0);
  for (i = 0; i < 2; i++) {
    boardSize[i] = 0.0;
  }

  if (board0->isValid && (board0->Energy < board45->Energy)) {
    board45 = board0;
    orient(&board45, I);
    toPoints(board45, points, boardSize);
    subPixelLocation(Ixy, points);
  } else {
    if (board45->isValid) {
      orient(&board45, I);
      toPoints(board45, points, boardSize);
      subPixelLocation(I_45_x, points);
    }
  }

  emxFree_real32_T(&I_45_x);
  emxFree_real32_T(&Ixy);
  c_emxFreeStruct_vision_internal(&lobj_0);
  c_emxFreeStruct_vision_internal(&lobj_1);
  c_emxFreeStruct_vision_internal(&lobj_2);
  c_emxFreeStruct_vision_internal(&lobj_3);
  c_emxFreeStruct_vision_internal(&lobj_4);
  c_emxFreeStruct_vision_internal(&lobj_5);
}

static int div_s32(int numerator, int denominator)
{
  int quotient;
  unsigned int absNumerator;
  unsigned int absDenominator;
  boolean_T quotientNeedsNegation;
  if (denominator == 0) {
    if (numerator >= 0) {
      quotient = MAX_int32_T;
    } else {
      quotient = MIN_int32_T;
    }
  } else {
    if (numerator < 0) {
      absNumerator = ~(unsigned int)numerator + 1U;
    } else {
      absNumerator = (unsigned int)numerator;
    }

    if (denominator < 0) {
      absDenominator = ~(unsigned int)denominator + 1U;
    } else {
      absDenominator = (unsigned int)denominator;
    }

    quotientNeedsNegation = ((numerator < 0) != (denominator < 0));
    absNumerator /= absDenominator;
    if (quotientNeedsNegation) {
      quotient = -(int)absNumerator;
    } else {
      quotient = (int)absNumerator;
    }
  }

  return quotient;
}

static int div_s32_floor(int numerator, int denominator)
{
  int quotient;
  unsigned int absNumerator;
  unsigned int absDenominator;
  boolean_T quotientNeedsNegation;
  unsigned int tempAbsQuotient;
  if (denominator == 0) {
    if (numerator >= 0) {
      quotient = MAX_int32_T;
    } else {
      quotient = MIN_int32_T;
    }
  } else {
    if (numerator < 0) {
      absNumerator = ~(unsigned int)numerator + 1U;
    } else {
      absNumerator = (unsigned int)numerator;
    }

    if (denominator < 0) {
      absDenominator = ~(unsigned int)denominator + 1U;
    } else {
      absDenominator = (unsigned int)denominator;
    }

    quotientNeedsNegation = ((numerator < 0) != (denominator < 0));
    tempAbsQuotient = absNumerator / absDenominator;
    if (quotientNeedsNegation) {
      absNumerator %= absDenominator;
      if (absNumerator > 0U) {
        tempAbsQuotient++;
      }

      quotient = -(int)tempAbsQuotient;
    } else {
      quotient = (int)tempAbsQuotient;
    }
  }

  return quotient;
}

static float e_Checkerboard_computeNewEnergy(const
  c_vision_internal_calibration_c *b_this, float oldEnergy)
{
  float newEnergy;
  emxArray_real_T *r38;
  int loop_ub;
  int ixstart;
  int i26;
  emxArray_real_T *r39;
  int n;
  emxArray_real_T *b;
  emxArray_real_T *r40;
  emxArray_real_T *num;
  emxArray_real_T *denom;
  emxArray_real_T *b_num;
  emxArray_real_T *c_num;
  emxArray_real_T *r41;
  emxArray_real_T *r42;
  double mtmp;
  boolean_T exitg1;
  int i;
  emxArray_real_T *d_num;
  emxArray_real_T *b_denom;
  emxArray_real_T *r43;
  int e_num[1];
  int c_denom[1];
  emxArray_real_T f_num;
  emxArray_real_T d_denom;
  double z;
  emxInit_real_T(&r38, 3);
  loop_ub = b_this->BoardCoords->size[0];
  ixstart = b_this->BoardCoords->size[2];
  i26 = r38->size[0] * r38->size[1] * r38->size[2];
  r38->size[0] = loop_ub;
  r38->size[1] = 1;
  r38->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r38, i26);
  for (i26 = 0; i26 < ixstart; i26++) {
    for (n = 0; n < loop_ub; n++) {
      r38->data[n + r38->size[0] * r38->size[1] * i26] = b_this->
        BoardCoords->data[n + b_this->BoardCoords->size[0] * b_this->
        BoardCoords->size[1] * i26];
    }
  }

  emxInit_real_T(&r39, 3);
  loop_ub = b_this->BoardCoords->size[0];
  ixstart = b_this->BoardCoords->size[2];
  i26 = r39->size[0] * r39->size[1] * r39->size[2];
  r39->size[0] = loop_ub;
  r39->size[1] = 1;
  r39->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r39, i26);
  for (i26 = 0; i26 < ixstart; i26++) {
    for (n = 0; n < loop_ub; n++) {
      r39->data[n + r39->size[0] * r39->size[1] * i26] = b_this->
        BoardCoords->data[(n + (b_this->BoardCoords->size[0] << 1)) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i26];
    }
  }

  emxInit_real_T(&b, 3);
  loop_ub = b_this->BoardCoords->size[0];
  ixstart = b_this->BoardCoords->size[2];
  i26 = b->size[0] * b->size[1] * b->size[2];
  b->size[0] = loop_ub;
  b->size[1] = 1;
  b->size[2] = ixstart;
  emxEnsureCapacity_real_T1(b, i26);
  for (i26 = 0; i26 < ixstart; i26++) {
    for (n = 0; n < loop_ub; n++) {
      b->data[n + b->size[0] * b->size[1] * i26] = b_this->BoardCoords->data[(n
        + b_this->BoardCoords->size[0]) + b_this->BoardCoords->size[0] *
        b_this->BoardCoords->size[1] * i26];
    }
  }

  emxInit_real_T(&r40, 3);
  i26 = r40->size[0] * r40->size[1] * r40->size[2];
  r40->size[0] = r38->size[0];
  r40->size[1] = 1;
  r40->size[2] = r38->size[2];
  emxEnsureCapacity_real_T1(r40, i26);
  loop_ub = r38->size[0] * r38->size[1] * r38->size[2];
  for (i26 = 0; i26 < loop_ub; i26++) {
    r40->data[i26] = (r38->data[i26] + r39->data[i26]) - 2.0 * b->data[i26];
  }

  emxFree_real_T(&b);
  emxInit_real_T1(&num, 2);
  b_squeeze(r40, num);
  loop_ub = b_this->BoardCoords->size[0];
  ixstart = b_this->BoardCoords->size[2];
  i26 = r38->size[0] * r38->size[1] * r38->size[2];
  r38->size[0] = loop_ub;
  r38->size[1] = 1;
  r38->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r38, i26);
  for (i26 = 0; i26 < ixstart; i26++) {
    for (n = 0; n < loop_ub; n++) {
      r38->data[n + r38->size[0] * r38->size[1] * i26] = b_this->
        BoardCoords->data[n + b_this->BoardCoords->size[0] * b_this->
        BoardCoords->size[1] * i26];
    }
  }

  loop_ub = b_this->BoardCoords->size[0];
  ixstart = b_this->BoardCoords->size[2];
  i26 = r39->size[0] * r39->size[1] * r39->size[2];
  r39->size[0] = loop_ub;
  r39->size[1] = 1;
  r39->size[2] = ixstart;
  emxEnsureCapacity_real_T1(r39, i26);
  for (i26 = 0; i26 < ixstart; i26++) {
    for (n = 0; n < loop_ub; n++) {
      r39->data[n + r39->size[0] * r39->size[1] * i26] = b_this->
        BoardCoords->data[(n + (b_this->BoardCoords->size[0] << 1)) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i26];
    }
  }

  i26 = r40->size[0] * r40->size[1] * r40->size[2];
  r40->size[0] = r38->size[0];
  r40->size[1] = 1;
  r40->size[2] = r38->size[2];
  emxEnsureCapacity_real_T1(r40, i26);
  loop_ub = r38->size[0] * r38->size[1] * r38->size[2];
  for (i26 = 0; i26 < loop_ub; i26++) {
    r40->data[i26] = r38->data[i26] - r39->data[i26];
  }

  emxFree_real_T(&r39);
  emxFree_real_T(&r38);
  emxInit_real_T1(&denom, 2);
  emxInit_real_T2(&b_num, 1);
  b_squeeze(r40, denom);
  loop_ub = num->size[0];
  i26 = b_num->size[0];
  b_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(b_num, i26);
  emxFree_real_T(&r40);
  for (i26 = 0; i26 < loop_ub; i26++) {
    b_num->data[i26] = num->data[i26];
  }

  emxInit_real_T2(&c_num, 1);
  loop_ub = num->size[0];
  i26 = c_num->size[0];
  c_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(c_num, i26);
  for (i26 = 0; i26 < loop_ub; i26++) {
    c_num->data[i26] = num->data[i26 + num->size[0]];
  }

  emxFree_real_T(&num);
  emxInit_real_T2(&r41, 1);
  c_hypot(b_num, c_num, r41);
  loop_ub = denom->size[0];
  i26 = b_num->size[0];
  b_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(b_num, i26);
  for (i26 = 0; i26 < loop_ub; i26++) {
    b_num->data[i26] = denom->data[i26];
  }

  loop_ub = denom->size[0];
  i26 = c_num->size[0];
  c_num->size[0] = loop_ub;
  emxEnsureCapacity_real_T2(c_num, i26);
  for (i26 = 0; i26 < loop_ub; i26++) {
    c_num->data[i26] = denom->data[i26 + denom->size[0]];
  }

  emxFree_real_T(&denom);
  emxInit_real_T2(&r42, 1);
  c_hypot(b_num, c_num, r42);
  rdivide(r41, r42, b_num);
  ixstart = 1;
  n = b_num->size[0];
  mtmp = b_num->data[0];
  emxFree_real_T(&c_num);
  emxFree_real_T(&r42);
  emxFree_real_T(&r41);
  if (b_num->size[0] > 1) {
    if (rtIsNaN(b_num->data[0])) {
      loop_ub = 2;
      exitg1 = false;
      while ((!exitg1) && (loop_ub <= n)) {
        ixstart = loop_ub;
        if (!rtIsNaN(b_num->data[loop_ub - 1])) {
          mtmp = b_num->data[loop_ub - 1];
          exitg1 = true;
        } else {
          loop_ub++;
        }
      }
    }

    if (ixstart < b_num->size[0]) {
      while (ixstart + 1 <= n) {
        if (b_num->data[ixstart] > mtmp) {
          mtmp = b_num->data[ixstart];
        }

        ixstart++;
      }
    }
  }

  emxFree_real_T(&b_num);
  if ((oldEnergy > mtmp) || rtIsNaN(mtmp)) {
    newEnergy = oldEnergy;
  } else {
    newEnergy = (float)mtmp;
  }

  mtmp = (double)b_this->BoardCoords->size[0] - 2.0;
  i = 0;
  emxInit_real_T(&d_num, 3);
  emxInit_real_T(&b_denom, 3);
  emxInit_real_T(&r43, 3);
  while (i <= (int)mtmp - 1) {
    loop_ub = b_this->BoardCoords->size[2];
    i26 = d_num->size[0] * d_num->size[1] * d_num->size[2];
    d_num->size[0] = 1;
    d_num->size[1] = 1;
    d_num->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(d_num, i26);
    for (i26 = 0; i26 < loop_ub; i26++) {
      d_num->data[d_num->size[0] * d_num->size[1] * i26] = b_this->
        BoardCoords->data[i + b_this->BoardCoords->size[0] * b_this->
        BoardCoords->size[1] * i26];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i26 = r43->size[0] * r43->size[1] * r43->size[2];
    r43->size[0] = 1;
    r43->size[1] = 1;
    r43->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(r43, i26);
    for (i26 = 0; i26 < loop_ub; i26++) {
      r43->data[r43->size[0] * r43->size[1] * i26] = b_this->BoardCoords->data
        [((int)((1.0 + (double)i) + 2.0) + b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1] * i26) - 1];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i26 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    b_denom->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(b_denom, i26);
    for (i26 = 0; i26 < loop_ub; i26++) {
      b_denom->data[b_denom->size[0] * b_denom->size[1] * i26] =
        b_this->BoardCoords->data[((int)((1.0 + (double)i) + 1.0) +
        b_this->BoardCoords->size[0] * b_this->BoardCoords->size[1] * i26) - 1];
    }

    i26 = d_num->size[0] * d_num->size[1] * d_num->size[2];
    d_num->size[0] = 1;
    d_num->size[1] = 1;
    emxEnsureCapacity_real_T1(d_num, i26);
    ixstart = d_num->size[0];
    n = d_num->size[1];
    loop_ub = d_num->size[2];
    loop_ub *= ixstart * n;
    for (i26 = 0; i26 < loop_ub; i26++) {
      d_num->data[i26] = (d_num->data[i26] + r43->data[i26]) - 2.0 *
        b_denom->data[i26];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i26 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    b_denom->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(b_denom, i26);
    for (i26 = 0; i26 < loop_ub; i26++) {
      b_denom->data[b_denom->size[0] * b_denom->size[1] * i26] =
        b_this->BoardCoords->data[i + b_this->BoardCoords->size[0] *
        b_this->BoardCoords->size[1] * i26];
    }

    loop_ub = b_this->BoardCoords->size[2];
    i26 = r43->size[0] * r43->size[1] * r43->size[2];
    r43->size[0] = 1;
    r43->size[1] = 1;
    r43->size[2] = loop_ub;
    emxEnsureCapacity_real_T1(r43, i26);
    for (i26 = 0; i26 < loop_ub; i26++) {
      r43->data[r43->size[0] * r43->size[1] * i26] = b_this->BoardCoords->data
        [((int)((1.0 + (double)i) + 2.0) + b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1] * i26) - 1];
    }

    i26 = b_denom->size[0] * b_denom->size[1] * b_denom->size[2];
    b_denom->size[0] = 1;
    b_denom->size[1] = 1;
    emxEnsureCapacity_real_T1(b_denom, i26);
    ixstart = b_denom->size[0];
    n = b_denom->size[1];
    loop_ub = b_denom->size[2];
    loop_ub *= ixstart * n;
    for (i26 = 0; i26 < loop_ub; i26++) {
      b_denom->data[i26] -= r43->data[i26];
    }

    e_num[0] = d_num->size[2];
    c_denom[0] = b_denom->size[2];
    f_num = *d_num;
    f_num.size = (int *)&e_num;
    f_num.numDimensions = 1;
    d_denom = *b_denom;
    d_denom.size = (int *)&c_denom;
    d_denom.numDimensions = 1;
    z = norm(&f_num) / norm(&d_denom);
    if ((newEnergy > z) || rtIsNaN(z)) {
    } else {
      newEnergy = (float)z;
    }

    i++;
  }

  emxFree_real_T(&r43);
  emxFree_real_T(&b_denom);
  emxFree_real_T(&d_num);
  ixstart = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  return newEnergy * (float)ixstart - (float)(b_this->BoardIdx->size[0] *
    b_this->BoardIdx->size[1]);
}

static void find_peaks(const emxArray_real32_T *metric, emxArray_real32_T *loc)
{
  int ixstart;
  int n;
  float threshold;
  int idx;
  emxArray_boolean_T *bw;
  int i2;
  boolean_T exitg1;
  emxArray_int32_T *ii;
  emxArray_int32_T *b_idx;
  unsigned int siz[2];
  emxArray_int32_T *vk;
  ixstart = 1;
  n = metric->size[0] * metric->size[1];
  threshold = metric->data[0];
  if (metric->size[0] * metric->size[1] > 1) {
    if (rtIsNaNF(metric->data[0])) {
      idx = 2;
      exitg1 = false;
      while ((!exitg1) && (idx <= n)) {
        ixstart = idx;
        if (!rtIsNaNF(metric->data[idx - 1])) {
          threshold = metric->data[idx - 1];
          exitg1 = true;
        } else {
          idx++;
        }
      }
    }

    if (ixstart < metric->size[0] * metric->size[1]) {
      while (ixstart + 1 <= n) {
        if (metric->data[ixstart] > threshold) {
          threshold = metric->data[ixstart];
        }

        ixstart++;
      }
    }
  }

  if (threshold <= 4.94065645841247E-324) {
    i2 = loc->size[0] * loc->size[1];
    loc->size[0] = 0;
    loc->size[1] = 2;
    emxEnsureCapacity_real32_T(loc, i2);
  } else {
    emxInit_boolean_T(&bw, 2);
    imregionalmax(metric, bw);
    threshold *= 0.15F;
    idx = metric->size[0] * metric->size[1] - 1;
    ixstart = 0;
    for (n = 0; n <= idx; n++) {
      if (metric->data[n] < threshold) {
        ixstart++;
      }
    }

    emxInit_int32_T(&ii, 1);
    i2 = ii->size[0];
    ii->size[0] = ixstart;
    emxEnsureCapacity_int32_T(ii, i2);
    ixstart = 0;
    for (n = 0; n <= idx; n++) {
      if (metric->data[n] < threshold) {
        ii->data[ixstart] = n + 1;
        ixstart++;
      }
    }

    ixstart = ii->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      bw->data[ii->data[i2] - 1] = false;
    }

    if (!((bw->size[0] == 0) || (bw->size[1] == 0))) {
      algbwmorph(bw);
    }

    ixstart = bw->size[1];
    for (i2 = 0; i2 < ixstart; i2++) {
      bw->data[bw->size[0] * i2] = false;
    }

    ixstart = bw->size[1];
    n = bw->size[0] - 1;
    for (i2 = 0; i2 < ixstart; i2++) {
      bw->data[n + bw->size[0] * i2] = false;
    }

    ixstart = bw->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      bw->data[i2] = false;
    }

    ixstart = bw->size[0];
    n = bw->size[1] - 1;
    for (i2 = 0; i2 < ixstart; i2++) {
      bw->data[i2 + bw->size[0] * n] = false;
    }

    n = bw->size[0] * bw->size[1];
    idx = 0;
    i2 = ii->size[0];
    ii->size[0] = n;
    emxEnsureCapacity_int32_T(ii, i2);
    ixstart = 1;
    exitg1 = false;
    while ((!exitg1) && (ixstart <= n)) {
      if (bw->data[ixstart - 1]) {
        idx++;
        ii->data[idx - 1] = ixstart;
        if (idx >= n) {
          exitg1 = true;
        } else {
          ixstart++;
        }
      } else {
        ixstart++;
      }
    }

    emxFree_boolean_T(&bw);
    if (n == 1) {
      if (idx == 0) {
        i2 = ii->size[0];
        ii->size[0] = 0;
        emxEnsureCapacity_int32_T(ii, i2);
      }
    } else {
      i2 = ii->size[0];
      if (1 > idx) {
        ii->size[0] = 0;
      } else {
        ii->size[0] = idx;
      }

      emxEnsureCapacity_int32_T(ii, i2);
    }

    emxInit_int32_T(&b_idx, 1);
    i2 = b_idx->size[0];
    b_idx->size[0] = ii->size[0];
    emxEnsureCapacity_int32_T(b_idx, i2);
    ixstart = ii->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      b_idx->data[i2] = ii->data[i2];
    }

    ixstart = b_idx->size[0];
    i2 = loc->size[0] * loc->size[1];
    loc->size[0] = ixstart;
    loc->size[1] = 2;
    emxEnsureCapacity_real32_T(loc, i2);
    ixstart <<= 1;
    for (i2 = 0; i2 < ixstart; i2++) {
      loc->data[i2] = 0.0F;
    }

    for (i2 = 0; i2 < 2; i2++) {
      siz[i2] = (unsigned int)metric->size[i2];
    }

    i2 = ii->size[0];
    ii->size[0] = b_idx->size[0];
    emxEnsureCapacity_int32_T(ii, i2);
    ixstart = b_idx->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      ii->data[i2] = b_idx->data[i2] - 1;
    }

    emxFree_int32_T(&b_idx);
    emxInit_int32_T(&vk, 1);
    i2 = vk->size[0];
    vk->size[0] = ii->size[0];
    emxEnsureCapacity_int32_T(vk, i2);
    ixstart = ii->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      vk->data[i2] = div_s32(ii->data[i2], (int)siz[0]);
    }

    i2 = ii->size[0];
    emxEnsureCapacity_int32_T(ii, i2);
    ixstart = ii->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      ii->data[i2] -= vk->data[i2] * (int)siz[0];
    }

    ixstart = ii->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      loc->data[i2 + loc->size[0]] = (float)(ii->data[i2] + 1);
    }

    emxFree_int32_T(&ii);
    ixstart = vk->size[0];
    for (i2 = 0; i2 < ixstart; i2++) {
      loc->data[i2] = (float)(vk->data[i2] + 1);
    }

    emxFree_int32_T(&vk);
  }
}

static c_vision_internal_calibration_c *growCheckerboard(const emxArray_real32_T
  *points, const emxArray_real32_T *scores, const emxArray_real32_T *Ix2, const
  emxArray_real32_T *Iy2, const emxArray_real32_T *Ixy, double theta,
  c_vision_internal_calibration_c *iobj_0, c_vision_internal_calibration_c
  *iobj_1, c_vision_internal_calibration_c *iobj_2)
{
  c_vision_internal_calibration_c *board;
  emxArray_real_T *seedIdx;
  int i4;
  int i5;
  int loop_ub;
  emxArray_real32_T *x;
  emxArray_int32_T *sortedIdx;
  emxArray_int32_T *iidx;
  emxArray_uint32_T *b_seedIdx;
  int b_x;
  c_vision_internal_calibration_c *currentBoard;
  int i;
  emxArray_real_T *b_currentBoard;
  emxArray_real_T *c_currentBoard;
  float c_x[2];
  float v1[2];
  float v2[2];
  boolean_T hasExpanded;
  c_vision_internal_calibration_c *tmpBoard;
  int b_i;
  int exitg1;
  double d0;
  int i6;
  int b_loop_ub;
  int d_currentBoard;
  int i7;
  if (scores->size[0] == 0) {
    iobj_2->isValid = false;
    iobj_2->Energy = ((real32_T)rtInf);
    board = iobj_2;
    i4 = iobj_2->BoardIdx->size[0] * iobj_2->BoardIdx->size[1];
    iobj_2->BoardIdx->size[0] = 1;
    iobj_2->BoardIdx->size[1] = 1;
    emxEnsureCapacity_real_T(iobj_2->BoardIdx, i4);
    iobj_2->BoardIdx->data[0] = 0.0;
    i4 = iobj_2->BoardIdx->size[0] * iobj_2->BoardIdx->size[1];
    iobj_2->BoardIdx->size[0] = 3;
    iobj_2->BoardIdx->size[1] = 3;
    emxEnsureCapacity_real_T(iobj_2->BoardIdx, i4);
    for (i4 = 0; i4 < 9; i4++) {
      iobj_2->BoardIdx->data[i4] = 0.0;
    }

    i4 = iobj_2->BoardCoords->size[0] * iobj_2->BoardCoords->size[1] *
      iobj_2->BoardCoords->size[2];
    iobj_2->BoardCoords->size[0] = 1;
    iobj_2->BoardCoords->size[1] = 1;
    iobj_2->BoardCoords->size[2] = 1;
    emxEnsureCapacity_real_T1(iobj_2->BoardCoords, i4);
    iobj_2->BoardCoords->data[0] = 0.0;
    i4 = iobj_2->BoardCoords->size[0] * iobj_2->BoardCoords->size[1] *
      iobj_2->BoardCoords->size[2];
    iobj_2->BoardCoords->size[0] = 3;
    iobj_2->BoardCoords->size[1] = 3;
    iobj_2->BoardCoords->size[2] = 2;
    emxEnsureCapacity_real_T1(iobj_2->BoardCoords, i4);
    for (i4 = 0; i4 < 18; i4++) {
      iobj_2->BoardCoords->data[i4] = 0.0;
    }
  } else {
    emxInit_real_T1(&seedIdx, 2);
    if (points->size[0] < 1) {
      i4 = seedIdx->size[0] * seedIdx->size[1];
      seedIdx->size[0] = 1;
      seedIdx->size[1] = 0;
      emxEnsureCapacity_real_T(seedIdx, i4);
    } else {
      i4 = points->size[0];
      i5 = seedIdx->size[0] * seedIdx->size[1];
      seedIdx->size[0] = 1;
      seedIdx->size[1] = (int)((double)i4 - 1.0) + 1;
      emxEnsureCapacity_real_T(seedIdx, i5);
      loop_ub = (int)((double)i4 - 1.0);
      for (i4 = 0; i4 <= loop_ub; i4++) {
        seedIdx->data[seedIdx->size[0] * i4] = 1.0 + (double)i4;
      }
    }

    emxInit_real32_T1(&x, 1);
    i4 = x->size[0];
    x->size[0] = seedIdx->size[1];
    emxEnsureCapacity_real32_T1(x, i4);
    loop_ub = seedIdx->size[1];
    for (i4 = 0; i4 < loop_ub; i4++) {
      x->data[i4] = scores->data[(int)seedIdx->data[seedIdx->size[0] * i4] - 1];
    }

    emxInit_int32_T(&sortedIdx, 1);
    emxInit_int32_T(&iidx, 1);
    sort(x, iidx);
    i4 = sortedIdx->size[0];
    sortedIdx->size[0] = iidx->size[0];
    emxEnsureCapacity_int32_T(sortedIdx, i4);
    loop_ub = iidx->size[0];
    emxFree_real32_T(&x);
    for (i4 = 0; i4 < loop_ub; i4++) {
      sortedIdx->data[i4] = iidx->data[i4];
    }

    emxFree_int32_T(&iidx);
    emxInit_uint32_T(&b_seedIdx, 2);
    i4 = b_seedIdx->size[0] * b_seedIdx->size[1];
    b_seedIdx->size[0] = 1;
    b_seedIdx->size[1] = seedIdx->size[1];
    emxEnsureCapacity_uint32_T(b_seedIdx, i4);
    loop_ub = seedIdx->size[1];
    for (i4 = 0; i4 < loop_ub; i4++) {
      b_seedIdx->data[b_seedIdx->size[0] * i4] = (unsigned int)seedIdx->
        data[seedIdx->size[0] * i4];
    }

    i4 = seedIdx->size[0] * seedIdx->size[1];
    seedIdx->size[0] = 1;
    seedIdx->size[1] = sortedIdx->size[0];
    emxEnsureCapacity_real_T(seedIdx, i4);
    loop_ub = sortedIdx->size[0];
    for (i4 = 0; i4 < loop_ub; i4++) {
      seedIdx->data[seedIdx->size[0] * i4] = b_seedIdx->data[sortedIdx->data[i4]
        - 1];
    }

    if (sortedIdx->size[0] > 2000) {
      b_x = (int)rt_roundd_snf((double)seedIdx->size[1] / 2.0);
      if (2000 < b_x) {
        b_x = 2000;
      }

      if (1 > b_x) {
        loop_ub = 0;
      } else {
        loop_ub = b_x;
      }

      i4 = b_seedIdx->size[0] * b_seedIdx->size[1];
      b_seedIdx->size[0] = 1;
      b_seedIdx->size[1] = seedIdx->size[1];
      emxEnsureCapacity_uint32_T(b_seedIdx, i4);
      b_x = seedIdx->size[1];
      for (i4 = 0; i4 < b_x; i4++) {
        b_seedIdx->data[b_seedIdx->size[0] * i4] = (unsigned int)seedIdx->
          data[seedIdx->size[0] * i4];
      }

      i4 = seedIdx->size[0] * seedIdx->size[1];
      seedIdx->size[0] = 1;
      seedIdx->size[1] = loop_ub;
      emxEnsureCapacity_real_T(seedIdx, i4);
      for (i4 = 0; i4 < loop_ub; i4++) {
        seedIdx->data[seedIdx->size[0] * i4] = b_seedIdx->data[i4];
      }
    }

    emxFree_int32_T(&sortedIdx);
    iobj_0->isValid = false;
    iobj_0->Energy = ((real32_T)rtInf);
    board = iobj_0;
    i4 = iobj_0->BoardIdx->size[0] * iobj_0->BoardIdx->size[1];
    iobj_0->BoardIdx->size[0] = 1;
    iobj_0->BoardIdx->size[1] = 1;
    emxEnsureCapacity_real_T(iobj_0->BoardIdx, i4);
    iobj_0->BoardIdx->data[0] = 0.0;
    i4 = iobj_0->BoardIdx->size[0] * iobj_0->BoardIdx->size[1];
    iobj_0->BoardIdx->size[0] = 3;
    iobj_0->BoardIdx->size[1] = 3;
    emxEnsureCapacity_real_T(iobj_0->BoardIdx, i4);
    for (i4 = 0; i4 < 9; i4++) {
      iobj_0->BoardIdx->data[i4] = 0.0;
    }

    i4 = iobj_0->BoardCoords->size[0] * iobj_0->BoardCoords->size[1] *
      iobj_0->BoardCoords->size[2];
    iobj_0->BoardCoords->size[0] = 1;
    iobj_0->BoardCoords->size[1] = 1;
    iobj_0->BoardCoords->size[2] = 1;
    emxEnsureCapacity_real_T1(iobj_0->BoardCoords, i4);
    iobj_0->BoardCoords->data[0] = 0.0;
    i4 = iobj_0->BoardCoords->size[0] * iobj_0->BoardCoords->size[1] *
      iobj_0->BoardCoords->size[2];
    iobj_0->BoardCoords->size[0] = 3;
    iobj_0->BoardCoords->size[1] = 3;
    iobj_0->BoardCoords->size[2] = 2;
    emxEnsureCapacity_real_T1(iobj_0->BoardCoords, i4);
    for (i4 = 0; i4 < 18; i4++) {
      iobj_0->BoardCoords->data[i4] = 0.0;
    }

    iobj_1->isValid = false;
    iobj_1->Energy = ((real32_T)rtInf);
    currentBoard = iobj_1;
    i4 = iobj_1->BoardIdx->size[0] * iobj_1->BoardIdx->size[1];
    iobj_1->BoardIdx->size[0] = 1;
    iobj_1->BoardIdx->size[1] = 1;
    emxEnsureCapacity_real_T(iobj_1->BoardIdx, i4);
    iobj_1->BoardIdx->data[0] = 0.0;
    i4 = iobj_1->BoardIdx->size[0] * iobj_1->BoardIdx->size[1];
    iobj_1->BoardIdx->size[0] = 3;
    iobj_1->BoardIdx->size[1] = 3;
    emxEnsureCapacity_real_T(iobj_1->BoardIdx, i4);
    for (i4 = 0; i4 < 9; i4++) {
      iobj_1->BoardIdx->data[i4] = 0.0;
    }

    i4 = iobj_1->BoardCoords->size[0] * iobj_1->BoardCoords->size[1] *
      iobj_1->BoardCoords->size[2];
    iobj_1->BoardCoords->size[0] = 1;
    iobj_1->BoardCoords->size[1] = 1;
    iobj_1->BoardCoords->size[2] = 1;
    emxEnsureCapacity_real_T1(iobj_1->BoardCoords, i4);
    iobj_1->BoardCoords->data[0] = 0.0;
    i4 = iobj_1->BoardCoords->size[0] * iobj_1->BoardCoords->size[1] *
      iobj_1->BoardCoords->size[2];
    iobj_1->BoardCoords->size[0] = 3;
    iobj_1->BoardCoords->size[1] = 3;
    iobj_1->BoardCoords->size[2] = 2;
    emxEnsureCapacity_real_T1(iobj_1->BoardCoords, i4);
    for (i4 = 0; i4 < 18; i4++) {
      iobj_1->BoardCoords->data[i4] = 0.0;
    }

    i = 0;
    emxInit_real_T1(&b_currentBoard, 2);
    emxInit_real_T(&c_currentBoard, 3);
    while (i <= seedIdx->size[1] - 1) {
      i4 = b_seedIdx->size[0] * b_seedIdx->size[1];
      b_seedIdx->size[0] = 1;
      b_seedIdx->size[1] = seedIdx->size[1];
      emxEnsureCapacity_uint32_T(b_seedIdx, i4);
      loop_ub = seedIdx->size[1];
      for (i4 = 0; i4 < loop_ub; i4++) {
        b_seedIdx->data[b_seedIdx->size[0] * i4] = (unsigned int)seedIdx->
          data[seedIdx->size[0] * i4];
      }

      b_x = (int)b_seedIdx->data[i];
      for (i4 = 0; i4 < 2; i4++) {
        c_x[i4] = points->data[(b_x + points->size[0] * i4) - 1];
      }

      cornerOrientations(Ix2, Iy2, Ixy, c_x, v1, v2);
      if ((std::abs(std::abs(std::abs(rt_atan2f_snf(v1[1], v1[0])) - 3.14159274F)
                    - (float)theta) > 0.58904862254808621) && (std::abs(std::abs
            (std::abs(rt_atan2f_snf(v2[1], v2[0])) - 3.14159274F) - (float)theta)
           > 0.58904862254808621)) {
      } else {
        Checkerboard_initialize(currentBoard, (double)(unsigned int)
          seedIdx->data[i], points, v1, v2);
        if (currentBoard->isValid) {
          hasExpanded = true;
          while (hasExpanded) {
            currentBoard->PreviousEnergy = currentBoard->Energy;
            b_i = 0;
            do {
              exitg1 = 0;
              if (b_i < 4) {
                if (!currentBoard->IsDirectionBad[b_i]) {
                  currentBoard->LastExpandDirection = 1.0 + (double)b_i;
                  c_Checkerboard_expandBoardDirec(currentBoard, 1.0 + (double)
                    b_i);
                  if (currentBoard->Energy < currentBoard->PreviousEnergy) {
                    exitg1 = 1;
                  } else {
                    currentBoard->Energy = currentBoard->PreviousEnergy;
                    switch ((int)currentBoard->LastExpandDirection) {
                     case 1:
                      i4 = currentBoard->BoardIdx->size[0];
                      if (2 > i4) {
                        i5 = 0;
                        i4 = 0;
                      } else {
                        i5 = 1;
                      }

                      b_x = currentBoard->BoardIdx->size[1];
                      i6 = b_currentBoard->size[0] * b_currentBoard->size[1];
                      b_currentBoard->size[0] = i4 - i5;
                      b_currentBoard->size[1] = b_x;
                      emxEnsureCapacity_real_T(b_currentBoard, i6);
                      for (i6 = 0; i6 < b_x; i6++) {
                        loop_ub = i4 - i5;
                        for (b_loop_ub = 0; b_loop_ub < loop_ub; b_loop_ub++) {
                          b_currentBoard->data[b_loop_ub + b_currentBoard->size
                            [0] * i6] = currentBoard->BoardIdx->data[(i5 +
                            b_loop_ub) + currentBoard->BoardIdx->size[0] * i6];
                        }
                      }

                      i4 = currentBoard->BoardIdx->size[0] *
                        currentBoard->BoardIdx->size[1];
                      currentBoard->BoardIdx->size[0] = b_currentBoard->size[0];
                      currentBoard->BoardIdx->size[1] = b_currentBoard->size[1];
                      emxEnsureCapacity_real_T(currentBoard->BoardIdx, i4);
                      loop_ub = b_currentBoard->size[1];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = b_currentBoard->size[0];
                        for (i5 = 0; i5 < b_x; i5++) {
                          currentBoard->BoardIdx->data[i5 +
                            currentBoard->BoardIdx->size[0] * i4] =
                            b_currentBoard->data[i5 + b_currentBoard->size[0] *
                            i4];
                        }
                      }

                      i4 = currentBoard->BoardCoords->size[0];
                      if (2 > i4) {
                        i5 = 0;
                        i4 = 0;
                      } else {
                        i5 = 1;
                      }

                      b_x = currentBoard->BoardCoords->size[1];
                      d_currentBoard = currentBoard->BoardCoords->size[2];
                      i6 = c_currentBoard->size[0] * c_currentBoard->size[1] *
                        c_currentBoard->size[2];
                      c_currentBoard->size[0] = i4 - i5;
                      c_currentBoard->size[1] = b_x;
                      c_currentBoard->size[2] = d_currentBoard;
                      emxEnsureCapacity_real_T1(c_currentBoard, i6);
                      for (i6 = 0; i6 < d_currentBoard; i6++) {
                        for (b_loop_ub = 0; b_loop_ub < b_x; b_loop_ub++) {
                          loop_ub = i4 - i5;
                          for (i7 = 0; i7 < loop_ub; i7++) {
                            c_currentBoard->data[(i7 + c_currentBoard->size[0] *
                                                  b_loop_ub) +
                              c_currentBoard->size[0] * c_currentBoard->size[1] *
                              i6] = currentBoard->BoardCoords->data[((i5 + i7) +
                              currentBoard->BoardCoords->size[0] * b_loop_ub) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i6];
                          }
                        }
                      }

                      i4 = currentBoard->BoardCoords->size[0] *
                        currentBoard->BoardCoords->size[1] *
                        currentBoard->BoardCoords->size[2];
                      currentBoard->BoardCoords->size[0] = c_currentBoard->size
                        [0];
                      currentBoard->BoardCoords->size[1] = c_currentBoard->size
                        [1];
                      currentBoard->BoardCoords->size[2] = c_currentBoard->size
                        [2];
                      emxEnsureCapacity_real_T1(currentBoard->BoardCoords, i4);
                      loop_ub = c_currentBoard->size[2];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = c_currentBoard->size[1];
                        for (i5 = 0; i5 < b_x; i5++) {
                          b_loop_ub = c_currentBoard->size[0];
                          for (i6 = 0; i6 < b_loop_ub; i6++) {
                            currentBoard->BoardCoords->data[(i6 +
                              currentBoard->BoardCoords->size[0] * i5) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i4] =
                              c_currentBoard->data[(i6 + c_currentBoard->size[0]
                              * i5) + c_currentBoard->size[0] *
                              c_currentBoard->size[1] * i4];
                          }
                        }
                      }
                      break;

                     case 2:
                      d0 = (double)currentBoard->BoardIdx->size[0] - 1.0;
                      if (1.0 > d0) {
                        loop_ub = 0;
                      } else {
                        loop_ub = (int)d0;
                      }

                      b_x = currentBoard->BoardIdx->size[1];
                      i4 = b_currentBoard->size[0] * b_currentBoard->size[1];
                      b_currentBoard->size[0] = loop_ub;
                      b_currentBoard->size[1] = b_x;
                      emxEnsureCapacity_real_T(b_currentBoard, i4);
                      for (i4 = 0; i4 < b_x; i4++) {
                        for (i5 = 0; i5 < loop_ub; i5++) {
                          b_currentBoard->data[i5 + b_currentBoard->size[0] * i4]
                            = currentBoard->BoardIdx->data[i5 +
                            currentBoard->BoardIdx->size[0] * i4];
                        }
                      }

                      i4 = currentBoard->BoardIdx->size[0] *
                        currentBoard->BoardIdx->size[1];
                      currentBoard->BoardIdx->size[0] = b_currentBoard->size[0];
                      currentBoard->BoardIdx->size[1] = b_currentBoard->size[1];
                      emxEnsureCapacity_real_T(currentBoard->BoardIdx, i4);
                      loop_ub = b_currentBoard->size[1];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = b_currentBoard->size[0];
                        for (i5 = 0; i5 < b_x; i5++) {
                          currentBoard->BoardIdx->data[i5 +
                            currentBoard->BoardIdx->size[0] * i4] =
                            b_currentBoard->data[i5 + b_currentBoard->size[0] *
                            i4];
                        }
                      }

                      d0 = (double)currentBoard->BoardCoords->size[0] - 1.0;
                      if (1.0 > d0) {
                        loop_ub = 0;
                      } else {
                        loop_ub = (int)d0;
                      }

                      b_x = currentBoard->BoardCoords->size[1];
                      d_currentBoard = currentBoard->BoardCoords->size[2];
                      i4 = c_currentBoard->size[0] * c_currentBoard->size[1] *
                        c_currentBoard->size[2];
                      c_currentBoard->size[0] = loop_ub;
                      c_currentBoard->size[1] = b_x;
                      c_currentBoard->size[2] = d_currentBoard;
                      emxEnsureCapacity_real_T1(c_currentBoard, i4);
                      for (i4 = 0; i4 < d_currentBoard; i4++) {
                        for (i5 = 0; i5 < b_x; i5++) {
                          for (i6 = 0; i6 < loop_ub; i6++) {
                            c_currentBoard->data[(i6 + c_currentBoard->size[0] *
                                                  i5) + c_currentBoard->size[0] *
                              c_currentBoard->size[1] * i4] =
                              currentBoard->BoardCoords->data[(i6 +
                              currentBoard->BoardCoords->size[0] * i5) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i4];
                          }
                        }
                      }

                      i4 = currentBoard->BoardCoords->size[0] *
                        currentBoard->BoardCoords->size[1] *
                        currentBoard->BoardCoords->size[2];
                      currentBoard->BoardCoords->size[0] = c_currentBoard->size
                        [0];
                      currentBoard->BoardCoords->size[1] = c_currentBoard->size
                        [1];
                      currentBoard->BoardCoords->size[2] = c_currentBoard->size
                        [2];
                      emxEnsureCapacity_real_T1(currentBoard->BoardCoords, i4);
                      loop_ub = c_currentBoard->size[2];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = c_currentBoard->size[1];
                        for (i5 = 0; i5 < b_x; i5++) {
                          b_loop_ub = c_currentBoard->size[0];
                          for (i6 = 0; i6 < b_loop_ub; i6++) {
                            currentBoard->BoardCoords->data[(i6 +
                              currentBoard->BoardCoords->size[0] * i5) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i4] =
                              c_currentBoard->data[(i6 + c_currentBoard->size[0]
                              * i5) + c_currentBoard->size[0] *
                              c_currentBoard->size[1] * i4];
                          }
                        }
                      }
                      break;

                     case 3:
                      i4 = currentBoard->BoardIdx->size[1];
                      if (2 > i4) {
                        i5 = 0;
                        i4 = 0;
                      } else {
                        i5 = 1;
                      }

                      b_x = currentBoard->BoardIdx->size[0];
                      i6 = b_currentBoard->size[0] * b_currentBoard->size[1];
                      b_currentBoard->size[0] = b_x;
                      b_currentBoard->size[1] = i4 - i5;
                      emxEnsureCapacity_real_T(b_currentBoard, i6);
                      loop_ub = i4 - i5;
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        for (i6 = 0; i6 < b_x; i6++) {
                          b_currentBoard->data[i6 + b_currentBoard->size[0] * i4]
                            = currentBoard->BoardIdx->data[i6 +
                            currentBoard->BoardIdx->size[0] * (i5 + i4)];
                        }
                      }

                      i4 = currentBoard->BoardIdx->size[0] *
                        currentBoard->BoardIdx->size[1];
                      currentBoard->BoardIdx->size[0] = b_currentBoard->size[0];
                      currentBoard->BoardIdx->size[1] = b_currentBoard->size[1];
                      emxEnsureCapacity_real_T(currentBoard->BoardIdx, i4);
                      loop_ub = b_currentBoard->size[1];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = b_currentBoard->size[0];
                        for (i5 = 0; i5 < b_x; i5++) {
                          currentBoard->BoardIdx->data[i5 +
                            currentBoard->BoardIdx->size[0] * i4] =
                            b_currentBoard->data[i5 + b_currentBoard->size[0] *
                            i4];
                        }
                      }

                      i4 = currentBoard->BoardCoords->size[1];
                      if (2 > i4) {
                        i5 = 0;
                        i4 = 0;
                      } else {
                        i5 = 1;
                      }

                      b_x = currentBoard->BoardCoords->size[0];
                      d_currentBoard = currentBoard->BoardCoords->size[2];
                      i6 = c_currentBoard->size[0] * c_currentBoard->size[1] *
                        c_currentBoard->size[2];
                      c_currentBoard->size[0] = b_x;
                      c_currentBoard->size[1] = i4 - i5;
                      c_currentBoard->size[2] = d_currentBoard;
                      emxEnsureCapacity_real_T1(c_currentBoard, i6);
                      for (i6 = 0; i6 < d_currentBoard; i6++) {
                        loop_ub = i4 - i5;
                        for (b_loop_ub = 0; b_loop_ub < loop_ub; b_loop_ub++) {
                          for (i7 = 0; i7 < b_x; i7++) {
                            c_currentBoard->data[(i7 + c_currentBoard->size[0] *
                                                  b_loop_ub) +
                              c_currentBoard->size[0] * c_currentBoard->size[1] *
                              i6] = currentBoard->BoardCoords->data[(i7 +
                              currentBoard->BoardCoords->size[0] * (i5 +
                              b_loop_ub)) + currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i6];
                          }
                        }
                      }

                      i4 = currentBoard->BoardCoords->size[0] *
                        currentBoard->BoardCoords->size[1] *
                        currentBoard->BoardCoords->size[2];
                      currentBoard->BoardCoords->size[0] = c_currentBoard->size
                        [0];
                      currentBoard->BoardCoords->size[1] = c_currentBoard->size
                        [1];
                      currentBoard->BoardCoords->size[2] = c_currentBoard->size
                        [2];
                      emxEnsureCapacity_real_T1(currentBoard->BoardCoords, i4);
                      loop_ub = c_currentBoard->size[2];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = c_currentBoard->size[1];
                        for (i5 = 0; i5 < b_x; i5++) {
                          b_loop_ub = c_currentBoard->size[0];
                          for (i6 = 0; i6 < b_loop_ub; i6++) {
                            currentBoard->BoardCoords->data[(i6 +
                              currentBoard->BoardCoords->size[0] * i5) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i4] =
                              c_currentBoard->data[(i6 + c_currentBoard->size[0]
                              * i5) + c_currentBoard->size[0] *
                              c_currentBoard->size[1] * i4];
                          }
                        }
                      }
                      break;

                     case 4:
                      d0 = (double)currentBoard->BoardIdx->size[1] - 1.0;
                      if (1.0 > d0) {
                        loop_ub = 0;
                      } else {
                        loop_ub = (int)d0;
                      }

                      b_x = currentBoard->BoardIdx->size[0];
                      i4 = b_currentBoard->size[0] * b_currentBoard->size[1];
                      b_currentBoard->size[0] = b_x;
                      b_currentBoard->size[1] = loop_ub;
                      emxEnsureCapacity_real_T(b_currentBoard, i4);
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        for (i5 = 0; i5 < b_x; i5++) {
                          b_currentBoard->data[i5 + b_currentBoard->size[0] * i4]
                            = currentBoard->BoardIdx->data[i5 +
                            currentBoard->BoardIdx->size[0] * i4];
                        }
                      }

                      i4 = currentBoard->BoardIdx->size[0] *
                        currentBoard->BoardIdx->size[1];
                      currentBoard->BoardIdx->size[0] = b_currentBoard->size[0];
                      currentBoard->BoardIdx->size[1] = b_currentBoard->size[1];
                      emxEnsureCapacity_real_T(currentBoard->BoardIdx, i4);
                      loop_ub = b_currentBoard->size[1];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = b_currentBoard->size[0];
                        for (i5 = 0; i5 < b_x; i5++) {
                          currentBoard->BoardIdx->data[i5 +
                            currentBoard->BoardIdx->size[0] * i4] =
                            b_currentBoard->data[i5 + b_currentBoard->size[0] *
                            i4];
                        }
                      }

                      d0 = (double)currentBoard->BoardCoords->size[1] - 1.0;
                      if (1.0 > d0) {
                        loop_ub = 0;
                      } else {
                        loop_ub = (int)d0;
                      }

                      b_x = currentBoard->BoardCoords->size[0];
                      d_currentBoard = currentBoard->BoardCoords->size[2];
                      i4 = c_currentBoard->size[0] * c_currentBoard->size[1] *
                        c_currentBoard->size[2];
                      c_currentBoard->size[0] = b_x;
                      c_currentBoard->size[1] = loop_ub;
                      c_currentBoard->size[2] = d_currentBoard;
                      emxEnsureCapacity_real_T1(c_currentBoard, i4);
                      for (i4 = 0; i4 < d_currentBoard; i4++) {
                        for (i5 = 0; i5 < loop_ub; i5++) {
                          for (i6 = 0; i6 < b_x; i6++) {
                            c_currentBoard->data[(i6 + c_currentBoard->size[0] *
                                                  i5) + c_currentBoard->size[0] *
                              c_currentBoard->size[1] * i4] =
                              currentBoard->BoardCoords->data[(i6 +
                              currentBoard->BoardCoords->size[0] * i5) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i4];
                          }
                        }
                      }

                      i4 = currentBoard->BoardCoords->size[0] *
                        currentBoard->BoardCoords->size[1] *
                        currentBoard->BoardCoords->size[2];
                      currentBoard->BoardCoords->size[0] = c_currentBoard->size
                        [0];
                      currentBoard->BoardCoords->size[1] = c_currentBoard->size
                        [1];
                      currentBoard->BoardCoords->size[2] = c_currentBoard->size
                        [2];
                      emxEnsureCapacity_real_T1(currentBoard->BoardCoords, i4);
                      loop_ub = c_currentBoard->size[2];
                      for (i4 = 0; i4 < loop_ub; i4++) {
                        b_x = c_currentBoard->size[1];
                        for (i5 = 0; i5 < b_x; i5++) {
                          b_loop_ub = c_currentBoard->size[0];
                          for (i6 = 0; i6 < b_loop_ub; i6++) {
                            currentBoard->BoardCoords->data[(i6 +
                              currentBoard->BoardCoords->size[0] * i5) +
                              currentBoard->BoardCoords->size[0] *
                              currentBoard->BoardCoords->size[1] * i4] =
                              c_currentBoard->data[(i6 + c_currentBoard->size[0]
                              * i5) + c_currentBoard->size[0] *
                              c_currentBoard->size[1] * i4];
                          }
                        }
                      }
                      break;
                    }

                    currentBoard->IsDirectionBad[b_i] = true;
                    b_i++;
                  }
                } else {
                  b_i++;
                }
              } else {
                hasExpanded = false;
                exitg1 = 1;
              }
            } while (exitg1 == 0);
          }
        }

        if (currentBoard->Energy < board->Energy) {
          tmpBoard = board;
          board = currentBoard;
          currentBoard = tmpBoard;
        }
      }

      i++;
    }

    emxFree_real_T(&c_currentBoard);
    emxFree_real_T(&b_currentBoard);
    emxFree_uint32_T(&b_seedIdx);
    emxFree_real_T(&seedIdx);
  }

  return board;
}

static void imfilter(emxArray_real32_T *varargin_1)
{
  int finalSize_idx_0;
  double pad[2];
  int finalSize_idx_1;
  emxArray_real32_T *a;
  emxArray_real_T *b_a;
  int r;
  int iC;
  emxArray_real_T *result;
  int iv2[2];
  boolean_T b1;
  int cEnd;
  int cEnd1;
  int ma;
  int na;
  int cidx;
  int lastColB;
  int iA;
  int lastColA;
  int k;
  int firstRowA;
  int aidx;
  int iB;
  int i;
  int b_i;
  int a_length;
  finalSize_idx_0 = varargin_1->size[0];
  pad[0] = 1.0;
  finalSize_idx_1 = varargin_1->size[1];
  pad[1] = 0.0;
  if (!((varargin_1->size[0] == 0) || (varargin_1->size[1] == 0))) {
    emxInit_real32_T(&a, 2);
    emxInit_real_T1(&b_a, 2);
    padImage(varargin_1, pad, a);
    r = b_a->size[0] * b_a->size[1];
    b_a->size[0] = a->size[0];
    b_a->size[1] = a->size[1];
    emxEnsureCapacity_real_T(b_a, r);
    iC = a->size[0] * a->size[1];
    for (r = 0; r < iC; r++) {
      b_a->data[r] = a->data[r];
    }

    emxFree_real32_T(&a);
    for (r = 0; r < 2; r++) {
      iv2[r] = b_a->size[r];
    }

    emxInit_real_T1(&result, 2);
    r = result->size[0] * result->size[1];
    result->size[0] = iv2[0];
    result->size[1] = iv2[1];
    emxEnsureCapacity_real_T(result, r);
    iC = iv2[0] * iv2[1];
    for (r = 0; r < iC; r++) {
      result->data[r] = 0.0;
    }

    if ((b_a->size[1] == 0) || (iv2[1] == 0)) {
      b1 = true;
    } else {
      b1 = false;
    }

    if (!b1) {
      cEnd = iv2[1];
      cEnd1 = iv2[0];
      ma = b_a->size[0];
      na = b_a->size[1];
      if (1 <= iv2[1] - 1) {
        lastColB = 1;
      } else {
        lastColB = iv2[1];
      }

      iA = 0;
      while (iA <= lastColB - 1) {
        if (na - 1 < cEnd - 1) {
          lastColA = na;
        } else {
          lastColA = cEnd;
        }

        for (k = 0; k < lastColA; k++) {
          if (k > 0) {
            firstRowA = k;
          } else {
            firstRowA = 0;
          }

          iC = firstRowA * cEnd1;
          iA = k * ma;
          iB = 0;
          for (i = 0; i < 3; i++) {
            firstRowA = (i < 1);
            if (i + ma <= cEnd1) {
              b_i = ma;
            } else {
              b_i = (cEnd1 - i) + 1;
            }

            a_length = b_i - firstRowA;
            aidx = iA + firstRowA;
            cidx = iC;
            for (r = 1; r <= a_length; r++) {
              result->data[cidx] += (-1.0 + (double)iB) * b_a->data[aidx];
              aidx++;
              cidx++;
            }

            iB++;
            if (i >= 1) {
              iC++;
            }
          }
        }

        iA = 1;
      }
    }

    emxFree_real_T(&b_a);
    if (2.0 > (double)finalSize_idx_0 + 1.0) {
      r = 0;
      cidx = 0;
    } else {
      r = 1;
      cidx = finalSize_idx_0 + 1;
    }

    if (1 > finalSize_idx_1) {
      iC = 0;
    } else {
      iC = finalSize_idx_1;
    }

    iA = varargin_1->size[0] * varargin_1->size[1];
    varargin_1->size[0] = cidx - r;
    varargin_1->size[1] = iC;
    emxEnsureCapacity_real32_T(varargin_1, iA);
    for (iA = 0; iA < iC; iA++) {
      firstRowA = cidx - r;
      for (aidx = 0; aidx < firstRowA; aidx++) {
        varargin_1->data[aidx + varargin_1->size[0] * iA] = (float)result->data
          [(r + aidx) + result->size[0] * iA];
      }
    }

    emxFree_real_T(&result);
  }
}

static void imregionalmax(const emxArray_real32_T *varargin_1,
  emxArray_boolean_T *BW)
{
  emxArray_boolean_T *out_;
  int i;
  c_images_internal_coder_Neighbo np;
  double centerPixelSub[2];
  int indx;
  boolean_T continuePropagation;
  emxArray_boolean_T *bwpre;
  emxArray_boolean_T *imParams_bw;
  emxArray_real32_T *in;
  struct_T fparams;
  int secondIndRange[2];
  boolean_T nhConn[9];
  int imnhInds_[9];
  int imnhSubs[18];
  int imageNeighborLinearOffsets[9];
  int pixelsPerImPage[2];
  double y;
  int pind;
  signed char subs[2];
  int r;
  signed char b_subs[2];
  int pixelSub[2];
  int a[18];
  c_images_internal_coder_Neighbo *nhoodObj;
  int k;
  int secondInd;
  float padValue;
  int varargin_2;
  float imnh_data[81];
  int u0;
  signed char imnh_size[2];
  int b_varargin_2;
  int imnhInds[9];
  int b_pind;
  int b_secondInd;
  int firstInd;
  int i3;
  int b_i;
  int b_firstInd;
  int b_centerPixelSub[2];
  float pixel;
  int b_y;
  boolean_T pixelout;
  int b_pixelSub[2];
  boolean_T exitg1;
  int trueCount;
  int imnhInds_data[9];
  int tmp_data[9];
  int c_pixelSub[2];
  float b_imnh_data[81];
  float b_pixel;
  int b_k[2];
  int b_tmp_data[9];
  int b_trueCount;
  int d_pixelSub[2];
  int b_imnhInds_data[9];
  int c_trueCount;
  int c_tmp_data[9];
  int c_imnhInds_data[9];
  int e_pixelSub[2];
  int d_tmp_data[9];
  int d_trueCount;
  int d_imnhInds_data[9];
  int e_tmp_data[9];
  emxInit_boolean_T1(&out_, 1);
  for (i = 0; i < 2; i++) {
    np.ImageSize[i] = varargin_1->size[i];
  }

  for (i = 0; i < 9; i++) {
    np.Neighborhood[i] = true;
  }

  np.Padding = 1.0;
  np.ProcessBorder = true;
  np.NeighborhoodCenter = 1.0;
  np.PadValue = 0.0;
  for (i = 0; i < 2; i++) {
    centerPixelSub[i] = varargin_1->size[i];
  }

  i = BW->size[0] * BW->size[1];
  BW->size[0] = (int)centerPixelSub[0];
  emxEnsureCapacity_boolean_T(BW, i);
  i = BW->size[0] * BW->size[1];
  BW->size[1] = (int)centerPixelSub[1];
  emxEnsureCapacity_boolean_T(BW, i);
  indx = (int)centerPixelSub[0] * (int)centerPixelSub[1];
  for (i = 0; i < indx; i++) {
    BW->data[i] = true;
  }

  continuePropagation = true;
  emxInit_boolean_T(&bwpre, 2);
  emxInit_boolean_T(&imParams_bw, 2);
  emxInit_real32_T(&in, 2);
  emxInitStruct_struct_T(&fparams);
  while (continuePropagation) {
    i = bwpre->size[0] * bwpre->size[1];
    bwpre->size[0] = BW->size[0];
    bwpre->size[1] = BW->size[1];
    emxEnsureCapacity_boolean_T(bwpre, i);
    indx = BW->size[0] * BW->size[1];
    for (i = 0; i < indx; i++) {
      bwpre->data[i] = BW->data[i];
    }

    i = imParams_bw->size[0] * imParams_bw->size[1];
    imParams_bw->size[0] = BW->size[0];
    imParams_bw->size[1] = BW->size[1];
    emxEnsureCapacity_boolean_T(imParams_bw, i);
    indx = BW->size[0] * BW->size[1];
    for (i = 0; i < indx; i++) {
      imParams_bw->data[i] = BW->data[i];
    }

    for (i = 0; i < 2; i++) {
      secondIndRange[i] = np.ImageSize[i];
    }

    for (i = 0; i < 9; i++) {
      nhConn[i] = np.Neighborhood[i];
    }

    for (i = 0; i < 9; i++) {
      imnhInds_[i] = np.ImageNeighborLinearOffsets[i];
    }

    for (i = 0; i < 9; i++) {
      imageNeighborLinearOffsets[i] = np.NeighborLinearIndices[i];
    }

    memcpy(&imnhSubs[0], &np.NeighborSubscriptOffsets[0], 18U * sizeof(int));
    pixelsPerImPage[0] = 1;
    pixelsPerImPage[1] = secondIndRange[0];
    for (i = 0; i < 2; i++) {
      secondIndRange[i]--;
    }

    y = nhConn[0];
    for (indx = 0; indx < 8; indx++) {
      y += (double)nhConn[indx + 1];
    }

    if (!(y == 0.0)) {
      indx = 0;
      for (pind = 0; pind < 9; pind++) {
        if (nhConn[pind]) {
          r = (int)rt_remd_snf((1.0 + (double)pind) - 1.0, 3.0) + 1;
          b_subs[1] = (signed char)((int)(((double)(pind - r) + 1.0) / 3.0) + 1);
          b_subs[0] = (signed char)r;
          subs[0] = (signed char)r;
          subs[1] = (signed char)(b_subs[1] - 1);
          for (i = 0; i < 2; i++) {
            imnhSubs[indx + 9 * i] = b_subs[i];
            pixelSub[i] = subs[i] * (1 + (i << 1));
          }

          imageNeighborLinearOffsets[indx] = pixelSub[0] + pixelSub[1];
          subs[0] = (signed char)r;
          subs[1] = (signed char)(b_subs[1] - 1);
          for (i = 0; i < 2; i++) {
            pixelSub[i] = subs[i] * pixelsPerImPage[i];
          }

          imnhInds_[indx] = (int)((double)pixelSub[0] + (double)pixelSub[1]);
          indx++;
        }
      }

      subs[0] = 2;
      subs[1] = 1;
      for (i = 0; i < 2; i++) {
        pixelSub[i] = subs[i] * pixelsPerImPage[i];
      }

      for (i = 0; i < 9; i++) {
        imnhInds_[i] -= (int)(2.0 + (double)pixelSub[1]);
      }

      memcpy(&a[0], &imnhSubs[0], 18U * sizeof(int));
      for (indx = 0; indx < 2; indx++) {
        for (k = 0; k < 9; k++) {
          imnhSubs[k + 9 * indx] = a[k + 9 * indx] - 2;
        }
      }
    }

    for (i = 0; i < 9; i++) {
      np.ImageNeighborLinearOffsets[i] = imnhInds_[i];
    }

    for (i = 0; i < 9; i++) {
      np.NeighborLinearIndices[i] = imageNeighborLinearOffsets[i];
    }

    memcpy(&np.NeighborSubscriptOffsets[0], &imnhSubs[0], 18U * sizeof(int));
    for (i = 0; i < 2; i++) {
      np.InteriorStart[i] = 2.0;
    }

    for (i = 0; i < 2; i++) {
      np.InteriorEnd[i] = secondIndRange[i];
    }

    nhoodObj = &np;
    i = in->size[0] * in->size[1];
    in->size[0] = varargin_1->size[0];
    in->size[1] = varargin_1->size[1];
    emxEnsureCapacity_real32_T(in, i);
    indx = varargin_1->size[0] * varargin_1->size[1];
    for (i = 0; i < indx; i++) {
      in->data[i] = varargin_1->data[i];
    }

    i = fparams.bw->size[0] * fparams.bw->size[1];
    fparams.bw->size[0] = imParams_bw->size[0];
    fparams.bw->size[1] = imParams_bw->size[1];
    emxEnsureCapacity_boolean_T(fparams.bw, i);
    indx = imParams_bw->size[0] * imParams_bw->size[1];
    for (i = 0; i < indx; i++) {
      fparams.bw->data[i] = imParams_bw->data[i];
    }

    secondIndRange[0] = (int)np.InteriorStart[1];
    secondIndRange[1] = np.InteriorEnd[1];
    pixelsPerImPage[0] = (int)np.InteriorStart[0];
    pixelsPerImPage[1] = np.InteriorEnd[0];
    for (i = 0; i < 9; i++) {
      imageNeighborLinearOffsets[i] = np.ImageNeighborLinearOffsets[i];
    }

    indx = np.ImageSize[0];
    k = secondIndRange[1];
    i = secondIndRange[0];

#pragma omp parallel \
 num_threads(omp_get_max_threads()) \
 private(out_,imnh_data,imnh_size,imnhInds,b_pind,i3,b_i,b_firstInd,pixel,pixelout,exitg1)

    {
      emxInit_boolean_T1(&out_, 1);

#pragma omp for nowait

      for (secondInd = i; secondInd <= k; secondInd++) {
        i3 = BW->size[0];
        b_i = out_->size[0];
        out_->size[0] = i3;
        emxEnsureCapacity_boolean_T1(out_, b_i);
        for (b_firstInd = pixelsPerImPage[0]; b_firstInd <= pixelsPerImPage[1];
             b_firstInd++) {
          b_pind = (secondInd - 1) * indx + b_firstInd;
          for (b_i = 0; b_i < 9; b_i++) {
            imnhInds[b_i] = imageNeighborLinearOffsets[b_i] + b_pind;
          }

          if (isrow(in)) {
            imnh_size[0] = 1;
            imnh_size[1] = 9;
            for (i3 = 0; i3 < 9; i3++) {
              imnh_data[i3] = in->data[imnhInds[i3] - 1];
            }
          } else {
            imnh_size[0] = 9;
            imnh_size[1] = 1;
            for (i3 = 0; i3 < 9; i3++) {
              imnh_data[i3] = in->data[imnhInds[i3] - 1];
            }
          }

          pixel = in->data[b_pind - 1];
          pixelout = fparams.bw->data[b_pind - 1];
          if (fparams.bw->data[b_pind - 1]) {
            b_i = 0;
            exitg1 = false;
            while ((!exitg1) && (b_i <= imnh_size[0] * imnh_size[1] - 1)) {
              if (imnh_data[b_i] > pixel) {
                pixelout = false;
                exitg1 = true;
              } else if ((imnh_data[b_i] == pixel) && (!fparams.bw->
                          data[imnhInds[b_i] - 1])) {
                pixelout = false;
                exitg1 = true;
              } else {
                b_i++;
              }
            }
          }

          out_->data[b_firstInd - 1] = pixelout;
        }

        b_i = out_->size[0];
        for (i3 = 0; i3 < b_i; i3++) {
          BW->data[i3 + BW->size[0] * (secondInd - 1)] = out_->data[i3];
        }
      }

      emxFree_boolean_T(&out_);
    }

    if (nhoodObj->ProcessBorder) {
      secondIndRange[1] = nhoodObj->ImageSize[0];
      centerPixelSub[1] = nhoodObj->InteriorStart[1] - 1.0;
      padValue = (float)nhoodObj->PadValue;
      varargin_2 = nhoodObj->ImageSize[0];
      u0 = secondIndRange[1];
      if (u0 < varargin_2) {
        varargin_2 = u0;
      }

      b_varargin_2 = nhoodObj->ImageSize[1];
      if (b_varargin_2 > centerPixelSub[1]) {
        b_varargin_2 = (int)centerPixelSub[1];
      }

      for (b_secondInd = 0; b_secondInd < b_varargin_2; b_secondInd++) {
        for (firstInd = 0; firstInd + 1 <= varargin_2; firstInd++) {
          indx = nhoodObj->ImageSize[0];
          pind = b_secondInd * indx + firstInd;
          for (i = 0; i < 9; i++) {
            imnhInds_[i] = nhoodObj->ImageNeighborLinearOffsets[i];
          }

          for (i = 0; i < 9; i++) {
            imnhInds_[i] = (imnhInds_[i] + pind) + 1;
          }

          for (i = 0; i < 2; i++) {
            b_centerPixelSub[i] = nhoodObj->ImageSize[i];
          }

          if (b_centerPixelSub[0] == 0) {
            r = 0;
          } else {
            r = pind - b_centerPixelSub[0] * div_s32(pind, b_centerPixelSub[0]);
          }

          k = pind - r;
          if (b_centerPixelSub[0] == 0) {
            if (k == 0) {
              b_y = 0;
            } else if (k < 0) {
              b_y = MIN_int32_T;
            } else {
              b_y = MAX_int32_T;
            }
          } else if (b_centerPixelSub[0] == 1) {
            b_y = k;
          } else if (b_centerPixelSub[0] == -1) {
            b_y = -k;
          } else {
            if (k >= 0) {
              indx = k;
            } else if (k == MIN_int32_T) {
              indx = MAX_int32_T;
            } else {
              indx = -k;
            }

            if (b_centerPixelSub[0] >= 0) {
              i = b_centerPixelSub[0];
            } else if (b_centerPixelSub[0] == MIN_int32_T) {
              i = MAX_int32_T;
            } else {
              i = -b_centerPixelSub[0];
            }

            b_y = div_s32(indx, i);
            indx -= b_y * i;
            if ((indx > 0) && (indx >= (i >> 1) + (i & 1))) {
              b_y++;
            }

            if ((k < 0) != (b_centerPixelSub[0] < 0)) {
              b_y = -b_y;
            }
          }

          b_pixelSub[1] = b_y + 1;
          b_pixelSub[0] = r + 1;
          for (i = 0; i < 2; i++) {
            pixelSub[i] = b_pixelSub[i];
          }

          for (i = 0; i < 18; i++) {
            a[i] = nhoodObj->NeighborSubscriptOffsets[i];
          }

          for (indx = 0; indx < 2; indx++) {
            for (k = 0; k < 9; k++) {
              imnhSubs[k + 9 * indx] = a[k + 9 * indx] + pixelSub[indx];
            }
          }

          for (i = 0; i < 9; i++) {
            nhConn[i] = true;
          }

          switch ((int)nhoodObj->Padding) {
           case 1:
            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }

            trueCount = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                trueCount++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                imnhInds_data[indx] = imnhInds_[i];
                indx++;
              }
            }
            break;

           case 2:
            trueCount = 9;
            for (i = 0; i < 9; i++) {
              imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  imnhInds_data[b_y] = 1;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }
            break;

           case 3:
            trueCount = 9;
            for (i = 0; i < 9; i++) {
              imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  i = 1;
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  i = nhoodObj->ImageSize[k];
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                imnhInds_data[b_y] = indx;
              }
            }
            break;

           case 4:
            trueCount = 9;
            for (i = 0; i < 9; i++) {
              imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = pixelSub[k] + (indx << 1);
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = ((indx << 1) - i) + 1;
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                imnhInds_data[b_y] = indx;
              }
            }
            break;
          }

          continuePropagation = (in->size[0] == 1);
          if (continuePropagation) {
            if (0 <= trueCount - 1) {
              memcpy(&tmp_data[0], &imnhInds_data[0], (unsigned int)(trueCount *
                      (int)sizeof(int)));
            }

            b_y = 1;
            r = trueCount;
            for (i = 0; i < trueCount; i++) {
              b_imnh_data[i] = in->data[tmp_data[i] - 1];
            }
          } else {
            b_y = trueCount;
            r = 1;
            for (i = 0; i < trueCount; i++) {
              b_imnh_data[i] = in->data[imnhInds_data[i] - 1];
            }
          }

          if (nhoodObj->Padding == 2.0) {
            k = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                k++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                b_tmp_data[indx] = i + 1;
                indx++;
              }
            }

            for (i = 0; i < k; i++) {
              b_imnh_data[b_tmp_data[i] - 1] = padValue;
            }
          }

          b_pixel = in->data[pind];
          continuePropagation = fparams.bw->data[pind];
          if (fparams.bw->data[pind]) {
            indx = 0;
            exitg1 = false;
            while ((!exitg1) && (indx <= b_y * r - 1)) {
              if (b_imnh_data[indx] > b_pixel) {
                continuePropagation = false;
                exitg1 = true;
              } else if ((b_imnh_data[indx] == b_pixel) && (!fparams.bw->
                          data[imnhInds_data[indx] - 1])) {
                continuePropagation = false;
                exitg1 = true;
              } else {
                indx++;
              }
            }
          }

          BW->data[pind] = continuePropagation;
        }
      }

      secondIndRange[1] = nhoodObj->ImageSize[0];
      pixelsPerImPage[0] = nhoodObj->InteriorEnd[1] + 1;
      pixelsPerImPage[1] = nhoodObj->ImageSize[1];
      padValue = (float)nhoodObj->PadValue;
      varargin_2 = nhoodObj->ImageSize[0];
      u0 = secondIndRange[1];
      if (u0 < varargin_2) {
        varargin_2 = u0;
      }

      b_varargin_2 = nhoodObj->ImageSize[1];
      u0 = pixelsPerImPage[1];
      if (u0 < b_varargin_2) {
        b_varargin_2 = u0;
      }

      u0 = pixelsPerImPage[0];
      if (!(u0 > 1)) {
        u0 = 1;
      }

      while (u0 <= b_varargin_2) {
        for (firstInd = 0; firstInd + 1 <= varargin_2; firstInd++) {
          indx = nhoodObj->ImageSize[0];
          pind = (u0 - 1) * indx + firstInd;
          for (i = 0; i < 9; i++) {
            imnhInds_[i] = nhoodObj->ImageNeighborLinearOffsets[i];
          }

          for (i = 0; i < 9; i++) {
            imnhInds_[i] = (imnhInds_[i] + pind) + 1;
          }

          for (i = 0; i < 2; i++) {
            b_centerPixelSub[i] = nhoodObj->ImageSize[i];
          }

          if (b_centerPixelSub[0] == 0) {
            r = 0;
          } else {
            r = pind - b_centerPixelSub[0] * div_s32(pind, b_centerPixelSub[0]);
          }

          k = pind - r;
          if (b_centerPixelSub[0] == 0) {
            if (k == 0) {
              b_y = 0;
            } else if (k < 0) {
              b_y = MIN_int32_T;
            } else {
              b_y = MAX_int32_T;
            }
          } else if (b_centerPixelSub[0] == 1) {
            b_y = k;
          } else if (b_centerPixelSub[0] == -1) {
            b_y = -k;
          } else {
            if (k >= 0) {
              indx = k;
            } else if (k == MIN_int32_T) {
              indx = MAX_int32_T;
            } else {
              indx = -k;
            }

            if (b_centerPixelSub[0] >= 0) {
              i = b_centerPixelSub[0];
            } else if (b_centerPixelSub[0] == MIN_int32_T) {
              i = MAX_int32_T;
            } else {
              i = -b_centerPixelSub[0];
            }

            b_y = div_s32(indx, i);
            indx -= b_y * i;
            if ((indx > 0) && (indx >= (i >> 1) + (i & 1))) {
              b_y++;
            }

            if ((k < 0) != (b_centerPixelSub[0] < 0)) {
              b_y = -b_y;
            }
          }

          c_pixelSub[1] = b_y + 1;
          c_pixelSub[0] = r + 1;
          for (i = 0; i < 2; i++) {
            pixelSub[i] = c_pixelSub[i];
          }

          for (i = 0; i < 18; i++) {
            a[i] = nhoodObj->NeighborSubscriptOffsets[i];
          }

          for (indx = 0; indx < 2; indx++) {
            for (k = 0; k < 9; k++) {
              imnhSubs[k + 9 * indx] = a[k + 9 * indx] + pixelSub[indx];
            }
          }

          for (i = 0; i < 9; i++) {
            nhConn[i] = true;
          }

          switch ((int)nhoodObj->Padding) {
           case 1:
            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }

            b_trueCount = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                b_trueCount++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                b_imnhInds_data[indx] = imnhInds_[i];
                indx++;
              }
            }
            break;

           case 2:
            b_trueCount = 9;
            for (i = 0; i < 9; i++) {
              b_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  b_imnhInds_data[b_y] = 1;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }
            break;

           case 3:
            b_trueCount = 9;
            for (i = 0; i < 9; i++) {
              b_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  i = 1;
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  i = nhoodObj->ImageSize[k];
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                b_imnhInds_data[b_y] = indx;
              }
            }
            break;

           case 4:
            b_trueCount = 9;
            for (i = 0; i < 9; i++) {
              b_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = pixelSub[k] + (indx << 1);
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = ((indx << 1) - i) + 1;
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                b_imnhInds_data[b_y] = indx;
              }
            }
            break;
          }

          continuePropagation = (in->size[0] == 1);
          if (continuePropagation) {
            if (0 <= b_trueCount - 1) {
              memcpy(&tmp_data[0], &b_imnhInds_data[0], (unsigned int)
                     (b_trueCount * (int)sizeof(int)));
            }

            b_y = 1;
            r = b_trueCount;
            for (i = 0; i < b_trueCount; i++) {
              b_imnh_data[i] = in->data[tmp_data[i] - 1];
            }
          } else {
            b_y = b_trueCount;
            r = 1;
            for (i = 0; i < b_trueCount; i++) {
              b_imnh_data[i] = in->data[b_imnhInds_data[i] - 1];
            }
          }

          if (nhoodObj->Padding == 2.0) {
            k = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                k++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                c_tmp_data[indx] = i + 1;
                indx++;
              }
            }

            for (i = 0; i < k; i++) {
              b_imnh_data[c_tmp_data[i] - 1] = padValue;
            }
          }

          b_pixel = in->data[pind];
          continuePropagation = fparams.bw->data[pind];
          if (fparams.bw->data[pind]) {
            indx = 0;
            exitg1 = false;
            while ((!exitg1) && (indx <= b_y * r - 1)) {
              if (b_imnh_data[indx] > b_pixel) {
                continuePropagation = false;
                exitg1 = true;
              } else if ((b_imnh_data[indx] == b_pixel) && (!fparams.bw->
                          data[b_imnhInds_data[indx] - 1])) {
                continuePropagation = false;
                exitg1 = true;
              } else {
                indx++;
              }
            }
          }

          BW->data[pind] = continuePropagation;
        }

        u0++;
      }

      centerPixelSub[1] = nhoodObj->InteriorStart[0] - 1.0;
      pixelsPerImPage[1] = nhoodObj->ImageSize[1];
      padValue = (float)nhoodObj->PadValue;
      varargin_2 = nhoodObj->ImageSize[0];
      if (varargin_2 > centerPixelSub[1]) {
        b_varargin_2 = (int)centerPixelSub[1];
      } else {
        b_varargin_2 = varargin_2;
      }

      varargin_2 = nhoodObj->ImageSize[1];
      u0 = pixelsPerImPage[1];
      if (u0 < varargin_2) {
        varargin_2 = u0;
      }

      for (b_secondInd = 1; b_secondInd <= varargin_2; b_secondInd++) {
        for (firstInd = 0; firstInd < b_varargin_2; firstInd++) {
          indx = nhoodObj->ImageSize[0];
          pind = (b_secondInd - 1) * indx + firstInd;
          for (i = 0; i < 9; i++) {
            imnhInds_[i] = nhoodObj->ImageNeighborLinearOffsets[i];
          }

          for (i = 0; i < 9; i++) {
            imnhInds_[i] = (imnhInds_[i] + pind) + 1;
          }

          for (i = 0; i < 2; i++) {
            b_centerPixelSub[i] = nhoodObj->ImageSize[i];
          }

          if (b_centerPixelSub[0] == 0) {
            r = 0;
          } else {
            r = pind - b_centerPixelSub[0] * div_s32(pind, b_centerPixelSub[0]);
          }

          k = pind - r;
          if (b_centerPixelSub[0] == 0) {
            if (k == 0) {
              b_y = 0;
            } else if (k < 0) {
              b_y = MIN_int32_T;
            } else {
              b_y = MAX_int32_T;
            }
          } else if (b_centerPixelSub[0] == 1) {
            b_y = k;
          } else if (b_centerPixelSub[0] == -1) {
            b_y = -k;
          } else {
            if (k >= 0) {
              indx = k;
            } else if (k == MIN_int32_T) {
              indx = MAX_int32_T;
            } else {
              indx = -k;
            }

            if (b_centerPixelSub[0] >= 0) {
              i = b_centerPixelSub[0];
            } else if (b_centerPixelSub[0] == MIN_int32_T) {
              i = MAX_int32_T;
            } else {
              i = -b_centerPixelSub[0];
            }

            b_y = div_s32(indx, i);
            indx -= b_y * i;
            if ((indx > 0) && (indx >= (i >> 1) + (i & 1))) {
              b_y++;
            }

            if ((k < 0) != (b_centerPixelSub[0] < 0)) {
              b_y = -b_y;
            }
          }

          d_pixelSub[1] = b_y + 1;
          d_pixelSub[0] = r + 1;
          for (i = 0; i < 2; i++) {
            pixelSub[i] = d_pixelSub[i];
          }

          for (i = 0; i < 18; i++) {
            a[i] = nhoodObj->NeighborSubscriptOffsets[i];
          }

          for (indx = 0; indx < 2; indx++) {
            for (k = 0; k < 9; k++) {
              imnhSubs[k + 9 * indx] = a[k + 9 * indx] + pixelSub[indx];
            }
          }

          for (i = 0; i < 9; i++) {
            nhConn[i] = true;
          }

          switch ((int)nhoodObj->Padding) {
           case 1:
            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }

            c_trueCount = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                c_trueCount++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                c_imnhInds_data[indx] = imnhInds_[i];
                indx++;
              }
            }
            break;

           case 2:
            c_trueCount = 9;
            for (i = 0; i < 9; i++) {
              c_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  c_imnhInds_data[b_y] = 1;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }
            break;

           case 3:
            c_trueCount = 9;
            for (i = 0; i < 9; i++) {
              c_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  i = 1;
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  i = nhoodObj->ImageSize[k];
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                c_imnhInds_data[b_y] = indx;
              }
            }
            break;

           case 4:
            c_trueCount = 9;
            for (i = 0; i < 9; i++) {
              c_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = pixelSub[k] + (indx << 1);
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = ((indx << 1) - i) + 1;
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                c_imnhInds_data[b_y] = indx;
              }
            }
            break;
          }

          continuePropagation = (in->size[0] == 1);
          if (continuePropagation) {
            if (0 <= c_trueCount - 1) {
              memcpy(&tmp_data[0], &c_imnhInds_data[0], (unsigned int)
                     (c_trueCount * (int)sizeof(int)));
            }

            b_y = 1;
            r = c_trueCount;
            for (i = 0; i < c_trueCount; i++) {
              b_imnh_data[i] = in->data[tmp_data[i] - 1];
            }
          } else {
            b_y = c_trueCount;
            r = 1;
            for (i = 0; i < c_trueCount; i++) {
              b_imnh_data[i] = in->data[c_imnhInds_data[i] - 1];
            }
          }

          if (nhoodObj->Padding == 2.0) {
            k = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                k++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                d_tmp_data[indx] = i + 1;
                indx++;
              }
            }

            for (i = 0; i < k; i++) {
              b_imnh_data[d_tmp_data[i] - 1] = padValue;
            }
          }

          b_pixel = in->data[pind];
          continuePropagation = fparams.bw->data[pind];
          if (fparams.bw->data[pind]) {
            indx = 0;
            exitg1 = false;
            while ((!exitg1) && (indx <= b_y * r - 1)) {
              if (b_imnh_data[indx] > b_pixel) {
                continuePropagation = false;
                exitg1 = true;
              } else if ((b_imnh_data[indx] == b_pixel) && (!fparams.bw->
                          data[c_imnhInds_data[indx] - 1])) {
                continuePropagation = false;
                exitg1 = true;
              } else {
                indx++;
              }
            }
          }

          BW->data[pind] = continuePropagation;
        }
      }

      secondIndRange[0] = nhoodObj->InteriorEnd[0] + 1;
      secondIndRange[1] = nhoodObj->ImageSize[0];
      pixelsPerImPage[1] = nhoodObj->ImageSize[1];
      padValue = (float)nhoodObj->PadValue;
      u0 = secondIndRange[0];
      if (!(u0 > 1)) {
        u0 = 1;
      }

      varargin_2 = nhoodObj->ImageSize[0];
      indx = secondIndRange[1];
      if (indx < varargin_2) {
        varargin_2 = indx;
      }

      b_varargin_2 = nhoodObj->ImageSize[1];
      indx = pixelsPerImPage[1];
      if (indx < b_varargin_2) {
        b_varargin_2 = indx;
      }

      for (b_secondInd = 1; b_secondInd <= b_varargin_2; b_secondInd++) {
        for (firstInd = u0; firstInd <= varargin_2; firstInd++) {
          indx = nhoodObj->ImageSize[0];
          pind = ((b_secondInd - 1) * indx + firstInd) - 1;
          for (i = 0; i < 9; i++) {
            imnhInds_[i] = nhoodObj->ImageNeighborLinearOffsets[i];
          }

          for (i = 0; i < 9; i++) {
            imnhInds_[i] = (imnhInds_[i] + pind) + 1;
          }

          for (i = 0; i < 2; i++) {
            b_centerPixelSub[i] = nhoodObj->ImageSize[i];
          }

          if (b_centerPixelSub[0] == 0) {
            r = 0;
          } else {
            r = pind - b_centerPixelSub[0] * div_s32(pind, b_centerPixelSub[0]);
          }

          k = pind - r;
          if (b_centerPixelSub[0] == 0) {
            if (k == 0) {
              b_y = 0;
            } else if (k < 0) {
              b_y = MIN_int32_T;
            } else {
              b_y = MAX_int32_T;
            }
          } else if (b_centerPixelSub[0] == 1) {
            b_y = k;
          } else if (b_centerPixelSub[0] == -1) {
            b_y = -k;
          } else {
            if (k >= 0) {
              indx = k;
            } else if (k == MIN_int32_T) {
              indx = MAX_int32_T;
            } else {
              indx = -k;
            }

            if (b_centerPixelSub[0] >= 0) {
              i = b_centerPixelSub[0];
            } else if (b_centerPixelSub[0] == MIN_int32_T) {
              i = MAX_int32_T;
            } else {
              i = -b_centerPixelSub[0];
            }

            b_y = div_s32(indx, i);
            indx -= b_y * i;
            if ((indx > 0) && (indx >= (i >> 1) + (i & 1))) {
              b_y++;
            }

            if ((k < 0) != (b_centerPixelSub[0] < 0)) {
              b_y = -b_y;
            }
          }

          e_pixelSub[1] = b_y + 1;
          e_pixelSub[0] = r + 1;
          for (i = 0; i < 2; i++) {
            pixelSub[i] = e_pixelSub[i];
          }

          for (i = 0; i < 18; i++) {
            a[i] = nhoodObj->NeighborSubscriptOffsets[i];
          }

          for (indx = 0; indx < 2; indx++) {
            for (k = 0; k < 9; k++) {
              imnhSubs[k + 9 * indx] = a[k + 9 * indx] + pixelSub[indx];
            }
          }

          for (i = 0; i < 9; i++) {
            nhConn[i] = true;
          }

          switch ((int)nhoodObj->Padding) {
           case 1:
            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }

            d_trueCount = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                d_trueCount++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (nhConn[i]) {
                d_imnhInds_data[indx] = imnhInds_[i];
                indx++;
              }
            }
            break;

           case 2:
            d_trueCount = 9;
            for (i = 0; i < 9; i++) {
              d_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              k = 0;
              exitg1 = false;
              while ((!exitg1) && (k < 2)) {
                if ((imnhSubs[b_y + 9 * k] < 1) || (imnhSubs[b_y + 9 * k] >
                     nhoodObj->ImageSize[k])) {
                  nhConn[b_y] = false;
                  d_imnhInds_data[b_y] = 1;
                  exitg1 = true;
                } else {
                  k++;
                }
              }
            }
            break;

           case 3:
            d_trueCount = 9;
            for (i = 0; i < 9; i++) {
              d_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  i = 1;
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  i = nhoodObj->ImageSize[k];
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                d_imnhInds_data[b_y] = indx;
              }
            }
            break;

           case 4:
            d_trueCount = 9;
            for (i = 0; i < 9; i++) {
              d_imnhInds_data[i] = imnhInds_[i];
            }

            for (b_y = 0; b_y < 9; b_y++) {
              for (i = 0; i < 2; i++) {
                pixelSub[i] = imnhSubs[b_y + 9 * i];
              }

              for (k = 0; k < 2; k++) {
                i = pixelSub[k];
                if (pixelSub[k] < 1) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = pixelSub[k] + (indx << 1);
                }

                if (i > nhoodObj->ImageSize[k]) {
                  nhConn[b_y] = false;
                  indx = nhoodObj->ImageSize[k];
                  i = ((indx << 1) - i) + 1;
                }

                pixelSub[k] = i;
              }

              if (!nhConn[b_y]) {
                for (i = 0; i < 2; i++) {
                  b_centerPixelSub[i] = nhoodObj->ImageSize[i];
                }

                b_k[0] = 1;
                b_k[1] = b_centerPixelSub[0];
                indx = 1;
                for (i = 0; i < 2; i++) {
                  indx += (pixelSub[i] - 1) * b_k[i];
                }

                d_imnhInds_data[b_y] = indx;
              }
            }
            break;
          }

          continuePropagation = (in->size[0] == 1);
          if (continuePropagation) {
            if (0 <= d_trueCount - 1) {
              memcpy(&tmp_data[0], &d_imnhInds_data[0], (unsigned int)
                     (d_trueCount * (int)sizeof(int)));
            }

            b_y = 1;
            r = d_trueCount;
            for (i = 0; i < d_trueCount; i++) {
              b_imnh_data[i] = in->data[tmp_data[i] - 1];
            }
          } else {
            b_y = d_trueCount;
            r = 1;
            for (i = 0; i < d_trueCount; i++) {
              b_imnh_data[i] = in->data[d_imnhInds_data[i] - 1];
            }
          }

          if (nhoodObj->Padding == 2.0) {
            k = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                k++;
              }
            }

            indx = 0;
            for (i = 0; i < 9; i++) {
              if (!nhConn[i]) {
                e_tmp_data[indx] = i + 1;
                indx++;
              }
            }

            for (i = 0; i < k; i++) {
              b_imnh_data[e_tmp_data[i] - 1] = padValue;
            }
          }

          b_pixel = in->data[pind];
          continuePropagation = fparams.bw->data[pind];
          if (fparams.bw->data[pind]) {
            indx = 0;
            exitg1 = false;
            while ((!exitg1) && (indx <= b_y * r - 1)) {
              if (b_imnh_data[indx] > b_pixel) {
                continuePropagation = false;
                exitg1 = true;
              } else if ((b_imnh_data[indx] == b_pixel) && (!fparams.bw->
                          data[d_imnhInds_data[indx] - 1])) {
                continuePropagation = false;
                exitg1 = true;
              } else {
                indx++;
              }
            }
          }

          BW->data[pind] = continuePropagation;
        }
      }
    }

    continuePropagation = !isequal(bwpre, BW);
  }

  emxFreeStruct_struct_T(&fparams);
  emxFree_real32_T(&in);
  emxFree_boolean_T(&imParams_bw);
  emxFree_boolean_T(&bwpre);
  emxFree_boolean_T(&out_);
}

static boolean_T isequal(const emxArray_boolean_T *varargin_1, const
  emxArray_boolean_T *varargin_2)
{
  boolean_T p;
  boolean_T b_p;
  int k;
  boolean_T exitg1;
  p = false;
  b_p = false;
  if ((varargin_1->size[0] != varargin_2->size[0]) || (varargin_1->size[1] !=
       varargin_2->size[1])) {
  } else {
    b_p = true;
  }

  if (b_p && (!((varargin_1->size[0] == 0) || (varargin_1->size[1] == 0))) &&
      (!((varargin_2->size[0] == 0) || (varargin_2->size[1] == 0)))) {
    k = 0;
    exitg1 = false;
    while ((!exitg1) && (k <= varargin_2->size[0] * varargin_2->size[1] - 1)) {
      if (varargin_1->data[k] != varargin_2->data[k]) {
        b_p = false;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }

  if (b_p) {
    p = true;
  }

  return p;
}

static boolean_T isrow(const emxArray_real32_T *x)
{
  return x->size[0] == 1;
}

static float mean(const emxArray_real32_T *x)
{
  float y;
  int firstBlockLength;
  int nblocks;
  int lastBlockLength;
  int k;
  int xblockoffset;
  float bsum;
  int hi;
  if (x->size[0] == 0) {
    y = 0.0F;
  } else {
    if (x->size[0] <= 1024) {
      firstBlockLength = x->size[0];
      lastBlockLength = 0;
      nblocks = 1;
    } else {
      firstBlockLength = 1024;
      nblocks = x->size[0] / 1024;
      lastBlockLength = x->size[0] - (nblocks << 10);
      if (lastBlockLength > 0) {
        nblocks++;
      } else {
        lastBlockLength = 1024;
      }
    }

    y = x->data[0];
    for (k = 2; k <= firstBlockLength; k++) {
      y += x->data[k - 1];
    }

    for (firstBlockLength = 2; firstBlockLength <= nblocks; firstBlockLength++)
    {
      xblockoffset = (firstBlockLength - 1) << 10;
      bsum = x->data[xblockoffset];
      if (firstBlockLength == nblocks) {
        hi = lastBlockLength;
      } else {
        hi = 1024;
      }

      for (k = 2; k <= hi; k++) {
        bsum += x->data[(xblockoffset + k) - 1];
      }

      y += bsum;
    }
  }

  y /= (float)x->size[0];
  return y;
}

static void merge(emxArray_int32_T *idx, emxArray_real32_T *x, int offset, int
                  np, int nq, emxArray_int32_T *iwork, emxArray_real32_T *xwork)
{
  int n;
  int qend;
  int p;
  int iout;
  int exitg1;
  if (nq != 0) {
    n = np + nq;
    for (qend = 0; qend + 1 <= n; qend++) {
      iwork->data[qend] = idx->data[offset + qend];
      xwork->data[qend] = x->data[offset + qend];
    }

    p = 0;
    n = np;
    qend = np + nq;
    iout = offset - 1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork->data[p] >= xwork->data[n]) {
        idx->data[iout] = iwork->data[p];
        x->data[iout] = xwork->data[p];
        if (p + 1 < np) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx->data[iout] = iwork->data[n];
        x->data[iout] = xwork->data[n];
        if (n + 1 < qend) {
          n++;
        } else {
          n = (iout - p) + 1;
          while (p + 1 <= np) {
            idx->data[n + p] = iwork->data[p];
            x->data[n + p] = xwork->data[p];
            p++;
          }

          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

static void merge_block(emxArray_int32_T *idx, emxArray_real32_T *x, int offset,
  int n, int preSortLevel, emxArray_int32_T *iwork, emxArray_real32_T *xwork)
{
  int nPairs;
  int bLen;
  int tailOffset;
  int nTail;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
    if ((nPairs & 1) != 0) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        merge(idx, x, offset + tailOffset, bLen, nTail - bLen, iwork, xwork);
      }
    }

    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 1; nTail <= nPairs; nTail++) {
      merge(idx, x, offset + (nTail - 1) * tailOffset, bLen, bLen, iwork, xwork);
    }

    bLen = tailOffset;
  }

  if (n > bLen) {
    merge(idx, x, offset, bLen, n - bLen, iwork, xwork);
  }
}

static double norm(const emxArray_real_T *x)
{
  double y;
  double scale;
  int k;
  double absxk;
  double t;
  if (x->size[0] == 0) {
    y = 0.0;
  } else {
    y = 0.0;
    if (x->size[0] == 1) {
      y = std::abs(x->data[0]);
    } else {
      scale = 3.3121686421112381E-170;
      for (k = 1; k <= x->size[0]; k++) {
        absxk = std::abs(x->data[k - 1]);
        if (absxk > scale) {
          t = scale / absxk;
          y = 1.0 + y * t * t;
          scale = absxk;
        } else {
          t = absxk / scale;
          y += t * t;
        }
      }

      y = scale * std::sqrt(y);
    }
  }

  return y;
}

static void orient(c_vision_internal_calibration_c **board, const
                   emxArray_real32_T *I)
{
  float x;
  emxArray_real_T *b_board;
  emxArray_boolean_T *upperLeftMask;
  emxArray_boolean_T *nextSquareMask;
  int j;
  double upperLeftPolyX[4];
  int i1;
  double upperLeftPolyY[4];
  int k;
  emxArray_real_T *newBoardCoords1;
  int vspread;
  emxArray_int32_T *r50;
  emxArray_real_T *newBoardCoords2;
  emxArray_real_T *r51;
  emxArray_int32_T *r52;
  emxArray_real32_T *b_I;
  float b_x;
  c_vision_internal_calibration_c *r53;
  boolean_T s;
  boolean_T t;
  emxArray_real_T *r54;
  emxArray_real_T *r55;
  emxArray_boolean_T *c_x;
  int ix;
  boolean_T exitg1;
  x = (*board)->Energy;
  if (!rtIsInfF(x)) {
    if ((*board)->BoardCoords->size[0] < (*board)->BoardCoords->size[1]) {
      emxInit_real_T1(&b_board, 2);
      rot90((*board)->BoardIdx, b_board);
      j = (*board)->BoardIdx->size[0] * (*board)->BoardIdx->size[1];
      (*board)->BoardIdx->size[0] = b_board->size[0];
      (*board)->BoardIdx->size[1] = b_board->size[1];
      emxEnsureCapacity_real_T((*board)->BoardIdx, j);
      i1 = b_board->size[0] * b_board->size[1];
      for (j = 0; j < i1; j++) {
        (*board)->BoardIdx->data[j] = b_board->data[j];
      }

      i1 = (*board)->BoardCoords->size[0];
      k = (*board)->BoardCoords->size[1];
      j = b_board->size[0] * b_board->size[1];
      b_board->size[0] = i1;
      b_board->size[1] = k;
      emxEnsureCapacity_real_T(b_board, j);
      for (j = 0; j < k; j++) {
        for (vspread = 0; vspread < i1; vspread++) {
          b_board->data[vspread + b_board->size[0] * j] = (*board)->
            BoardCoords->data[vspread + (*board)->BoardCoords->size[0] * j];
        }
      }

      emxInit_real_T1(&newBoardCoords1, 2);
      rot90(b_board, newBoardCoords1);
      i1 = (*board)->BoardCoords->size[0];
      k = (*board)->BoardCoords->size[1];
      j = b_board->size[0] * b_board->size[1];
      b_board->size[0] = i1;
      b_board->size[1] = k;
      emxEnsureCapacity_real_T(b_board, j);
      for (j = 0; j < k; j++) {
        for (vspread = 0; vspread < i1; vspread++) {
          b_board->data[vspread + b_board->size[0] * j] = (*board)->
            BoardCoords->data[(vspread + (*board)->BoardCoords->size[0] * j) + (*
            board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1]];
        }
      }

      emxInit_real_T1(&newBoardCoords2, 2);
      emxInit_real_T(&r51, 3);
      rot90(b_board, newBoardCoords2);
      cat(newBoardCoords1, newBoardCoords2, r51);
      j = (*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1] *
        (*board)->BoardCoords->size[2];
      (*board)->BoardCoords->size[0] = r51->size[0];
      (*board)->BoardCoords->size[1] = r51->size[1];
      (*board)->BoardCoords->size[2] = 2;
      emxEnsureCapacity_real_T1((*board)->BoardCoords, j);
      i1 = r51->size[0] * r51->size[1] * r51->size[2];
      emxFree_real_T(&b_board);
      emxFree_real_T(&newBoardCoords2);
      emxFree_real_T(&newBoardCoords1);
      for (j = 0; j < i1; j++) {
        (*board)->BoardCoords->data[j] = r51->data[j];
      }

      emxFree_real_T(&r51);
    }

    emxInit_boolean_T(&upperLeftMask, 2);
    emxInit_boolean_T(&nextSquareMask, 2);
    upperLeftPolyX[0] = (*board)->BoardCoords->data[0];
    upperLeftPolyX[1] = (*board)->BoardCoords->data[(*board)->BoardCoords->size
      [0]];
    upperLeftPolyX[2] = (*board)->BoardCoords->data[1 + (*board)->
      BoardCoords->size[0]];
    upperLeftPolyX[3] = (*board)->BoardCoords->data[1];
    upperLeftPolyY[0] = (*board)->BoardCoords->data[(*board)->BoardCoords->size
      [0] * (*board)->BoardCoords->size[1]];
    upperLeftPolyY[1] = (*board)->BoardCoords->data[(*board)->BoardCoords->size
      [0] + (*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1]];
    upperLeftPolyY[2] = (*board)->BoardCoords->data[((*board)->BoardCoords->
      size[0] + (*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1])
      + 1];
    upperLeftPolyY[3] = (*board)->BoardCoords->data[1 + (*board)->
      BoardCoords->size[0] * (*board)->BoardCoords->size[1]];
    poly2RectMask(upperLeftPolyX, upperLeftPolyY, (double)I->size[0], (double)
                  I->size[1], upperLeftMask);
    upperLeftPolyX[0] = (*board)->BoardCoords->data[(*board)->BoardCoords->size
      [0]];
    upperLeftPolyX[1] = (*board)->BoardCoords->data[(*board)->BoardCoords->size
      [0] << 1];
    upperLeftPolyX[2] = (*board)->BoardCoords->data[1 + ((*board)->
      BoardCoords->size[0] << 1)];
    upperLeftPolyX[3] = (*board)->BoardCoords->data[1 + (*board)->
      BoardCoords->size[0]];
    upperLeftPolyY[0] = (*board)->BoardCoords->data[(*board)->BoardCoords->size
      [0] + (*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1]];
    upperLeftPolyY[1] = (*board)->BoardCoords->data[((*board)->BoardCoords->
      size[0] << 1) + (*board)->BoardCoords->size[0] * (*board)->
      BoardCoords->size[1]];
    upperLeftPolyY[2] = (*board)->BoardCoords->data[(((*board)->
      BoardCoords->size[0] << 1) + (*board)->BoardCoords->size[0] * (*board)
      ->BoardCoords->size[1]) + 1];
    upperLeftPolyY[3] = (*board)->BoardCoords->data[((*board)->BoardCoords->
      size[0] + (*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1])
      + 1];
    poly2RectMask(upperLeftPolyX, upperLeftPolyY, (double)I->size[0], (double)
                  I->size[1], nextSquareMask);
    i1 = upperLeftMask->size[0] * upperLeftMask->size[1] - 1;
    k = 0;
    for (vspread = 0; vspread <= i1; vspread++) {
      if (upperLeftMask->data[vspread]) {
        k++;
      }
    }

    emxInit_int32_T(&r50, 1);
    j = r50->size[0];
    r50->size[0] = k;
    emxEnsureCapacity_int32_T(r50, j);
    k = 0;
    for (vspread = 0; vspread <= i1; vspread++) {
      if (upperLeftMask->data[vspread]) {
        r50->data[k] = vspread + 1;
        k++;
      }
    }

    emxFree_boolean_T(&upperLeftMask);
    i1 = nextSquareMask->size[0] * nextSquareMask->size[1] - 1;
    k = 0;
    for (vspread = 0; vspread <= i1; vspread++) {
      if (nextSquareMask->data[vspread]) {
        k++;
      }
    }

    emxInit_int32_T(&r52, 1);
    j = r52->size[0];
    r52->size[0] = k;
    emxEnsureCapacity_int32_T(r52, j);
    k = 0;
    for (vspread = 0; vspread <= i1; vspread++) {
      if (nextSquareMask->data[vspread]) {
        r52->data[k] = vspread + 1;
        k++;
      }
    }

    emxFree_boolean_T(&nextSquareMask);
    emxInit_real32_T1(&b_I, 1);
    j = b_I->size[0];
    b_I->size[0] = r50->size[0];
    emxEnsureCapacity_real32_T1(b_I, j);
    i1 = r50->size[0];
    for (j = 0; j < i1; j++) {
      b_I->data[j] = I->data[r50->data[j] - 1];
    }

    emxFree_int32_T(&r50);
    x = mean(b_I);
    j = b_I->size[0];
    b_I->size[0] = r52->size[0];
    emxEnsureCapacity_real32_T1(b_I, j);
    i1 = r52->size[0];
    for (j = 0; j < i1; j++) {
      b_I->data[j] = I->data[r52->data[j] - 1];
    }

    emxFree_int32_T(&r52);
    b_x = mean(b_I);
    emxFree_real32_T(&b_I);
    if (!(x < b_x)) {
      r53 = *board;
      rot90_checkerboard(&r53);
      *board = r53;
    }

    s = (b_mod((double)(*board)->BoardCoords->size[0]) == 0.0);
    t = (b_mod((double)(*board)->BoardCoords->size[1]) == 0.0);
    if (!((s ^ t) != 0)) {
      emxInit_real_T(&r54, 3);
      i1 = (*board)->BoardCoords->size[2];
      j = r54->size[0] * r54->size[1] * r54->size[2];
      r54->size[0] = 1;
      r54->size[1] = 1;
      r54->size[2] = i1;
      emxEnsureCapacity_real_T1(r54, j);
      for (j = 0; j < i1; j++) {
        r54->data[r54->size[0] * r54->size[1] * j] = (*board)->BoardCoords->
          data[(*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1] *
          j];
      }

      emxInit_real_T(&r55, 3);
      i1 = (*board)->BoardCoords->size[2];
      k = (*board)->BoardCoords->size[1];
      vspread = (*board)->BoardCoords->size[0];
      j = r55->size[0] * r55->size[1] * r55->size[2];
      r55->size[0] = 1;
      r55->size[1] = 1;
      r55->size[2] = i1;
      emxEnsureCapacity_real_T1(r55, j);
      for (j = 0; j < i1; j++) {
        r55->data[r55->size[0] * r55->size[1] * j] = (*board)->BoardCoords->
          data[((vspread + (*board)->BoardCoords->size[0] * (k - 1)) + (*board
                )->BoardCoords->size[0] * (*board)->BoardCoords->size[1] * j) -
          1];
      }

      emxInit_boolean_T2(&c_x, 3);
      j = c_x->size[0] * c_x->size[1] * c_x->size[2];
      c_x->size[0] = 1;
      c_x->size[1] = 1;
      c_x->size[2] = r54->size[2];
      emxEnsureCapacity_boolean_T2(c_x, j);
      i1 = r54->size[0] * r54->size[1] * r54->size[2];
      for (j = 0; j < i1; j++) {
        c_x->data[j] = (r54->data[j] > r55->data[j]);
      }

      emxFree_real_T(&r55);
      emxFree_real_T(&r54);
      s = false;
      k = 3;
      while ((k > 2) && (c_x->size[2] == 1)) {
        k = 2;
      }

      if (3 > k) {
        k = c_x->size[2];
      } else {
        k = 1;
      }

      vspread = (c_x->size[2] - 1) * k;
      i1 = 0;
      for (j = 1; j <= k; j++) {
        i1++;
        vspread++;
        ix = i1;
        exitg1 = false;
        while ((!exitg1) && ((k > 0) && (ix <= vspread))) {
          t = !c_x->data[ix - 1];
          if (!t) {
            s = true;
            exitg1 = true;
          } else {
            ix += k;
          }
        }
      }

      emxFree_boolean_T(&c_x);
      if (s) {
        r53 = *board;
        rot90_checkerboard(&r53);
        *board = r53;
      }
    }
  }
}

static void padImage(const emxArray_real32_T *a_tmp, const double pad[2],
                     emxArray_real32_T *a)
{
  int i0;
  double sizeB[2];
  int j;
  int i;
  int i1;
  if ((a_tmp->size[0] == 0) || (a_tmp->size[1] == 0)) {
    for (i0 = 0; i0 < 2; i0++) {
      sizeB[i0] = (double)a_tmp->size[i0] + 2.0 * pad[i0];
    }

    i0 = a->size[0] * a->size[1];
    a->size[0] = (int)sizeB[0];
    a->size[1] = (int)sizeB[1];
    emxEnsureCapacity_real32_T(a, i0);
    j = (int)sizeB[0] * (int)sizeB[1];
    for (i0 = 0; i0 < j; i0++) {
      a->data[i0] = 0.0F;
    }
  } else {
    for (i0 = 0; i0 < 2; i0++) {
      sizeB[i0] = (double)a_tmp->size[i0] + 2.0 * pad[i0];
    }

    i0 = a->size[0] * a->size[1];
    a->size[0] = (int)sizeB[0];
    a->size[1] = (int)sizeB[1];
    emxEnsureCapacity_real32_T(a, i0);
    for (j = 0; j < (int)pad[1]; j++) {
      i0 = a->size[0];
      for (i = 0; i < i0; i++) {
        a->data[i + a->size[0] * j] = 0.0F;
      }
    }

    i0 = a->size[1];
    for (j = a_tmp->size[1] + (int)pad[1]; j + 1 <= i0; j++) {
      i1 = a->size[0];
      for (i = 0; i < i1; i++) {
        a->data[i + a->size[0] * j] = 0.0F;
      }
    }

    for (j = 0; j < a_tmp->size[1]; j++) {
      for (i = 0; i < (int)pad[0]; i++) {
        a->data[i + a->size[0] * (j + (int)pad[1])] = 0.0F;
      }
    }

    for (j = 0; j < a_tmp->size[1]; j++) {
      i0 = a->size[0];
      for (i = (int)pad[0] + a_tmp->size[0]; i + 1 <= i0; i++) {
        a->data[i + a->size[0] * (j + (int)pad[1])] = 0.0F;
      }
    }

    for (j = 0; j < a_tmp->size[1]; j++) {
      for (i = 0; i < a_tmp->size[0]; i++) {
        a->data[(i + (int)pad[0]) + a->size[0] * (j + (int)pad[1])] =
          a_tmp->data[i + a_tmp->size[0] * j];
      }
    }
  }
}

static void poly2RectMask(double b_X[4], double Y[4], double height, double
  width, emxArray_boolean_T *mask)
{
  int i28;
  int loop_ub;
  int i29;
  int i30;
  int i31;
  int unnamed_idx_0;
  c_sort(b_X);
  c_sort(Y);
  i28 = mask->size[0] * mask->size[1];
  mask->size[0] = (int)height;
  mask->size[1] = (int)width;
  emxEnsureCapacity_boolean_T(mask, i28);
  loop_ub = (int)height * (int)width;
  for (i28 = 0; i28 < loop_ub; i28++) {
    mask->data[i28] = false;
  }

  if (Y[1] > Y[2]) {
    i28 = 1;
    i29 = 1;
  } else {
    i28 = (int)Y[1];
    i29 = (int)Y[2] + 1;
  }

  if (b_X[1] > b_X[2]) {
    i30 = 1;
    i31 = 1;
  } else {
    i30 = (int)b_X[1];
    i31 = (int)b_X[2] + 1;
  }

  unnamed_idx_0 = i29 - i28;
  loop_ub = i31 - i30;
  for (i29 = 0; i29 < loop_ub; i29++) {
    for (i31 = 0; i31 < unnamed_idx_0; i31++) {
      mask->data[((i28 + i31) + mask->size[0] * ((i30 + i29) - 1)) - 1] = true;
    }
  }
}

static void power(const emxArray_real32_T *a, emxArray_real32_T *y)
{
  int nx;
  emxArray_real32_T *ztemp;
  unsigned int uv1[2];
  int k;
  for (nx = 0; nx < 2; nx++) {
    uv1[nx] = (unsigned int)a->size[nx];
  }

  emxInit_real32_T(&ztemp, 2);
  nx = ztemp->size[0] * ztemp->size[1];
  ztemp->size[0] = (int)uv1[0];
  ztemp->size[1] = (int)uv1[1];
  emxEnsureCapacity_real32_T(ztemp, nx);
  nx = y->size[0] * y->size[1];
  y->size[0] = (int)uv1[0];
  y->size[1] = (int)uv1[1];
  emxEnsureCapacity_real32_T(y, nx);
  nx = ztemp->size[0] * ztemp->size[1];
  k = 0;
  emxFree_real32_T(&ztemp);
  while (k + 1 <= nx) {
    y->data[k] = a->data[k] * a->data[k];
    k++;
  }
}

static void rdivide(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *z)
{
  int i17;
  int loop_ub;
  i17 = z->size[0];
  z->size[0] = x->size[0];
  emxEnsureCapacity_real_T2(z, i17);
  loop_ub = x->size[0];
  for (i17 = 0; i17 < loop_ub; i17++) {
    z->data[i17] = x->data[i17] / y->data[i17];
  }
}

static void rot90(const emxArray_real_T *A, emxArray_real_T *B)
{
  int n;
  int sizeB_idx_0;
  int sizeB_idx_1;
  int A_idx_0;
  int B_idx_0;
  n = A->size[1];
  sizeB_idx_0 = A->size[1];
  sizeB_idx_1 = A->size[0];
  A_idx_0 = B->size[0] * B->size[1];
  B->size[0] = sizeB_idx_0;
  B->size[1] = sizeB_idx_1;
  emxEnsureCapacity_real_T(B, A_idx_0);
  for (sizeB_idx_0 = 1; sizeB_idx_0 <= n; sizeB_idx_0++) {
    for (sizeB_idx_1 = 0; sizeB_idx_1 + 1 <= A->size[0]; sizeB_idx_1++) {
      A_idx_0 = A->size[0];
      B_idx_0 = B->size[0];
      B->data[(sizeB_idx_0 + B_idx_0 * sizeB_idx_1) - 1] = A->data[sizeB_idx_1 +
        A_idx_0 * (n - sizeB_idx_0)];
    }
  }
}

static void rot90_checkerboard(c_vision_internal_calibration_c **board)
{
  emxArray_real_T *b_board;
  int i46;
  int loop_ub;
  int b_loop_ub;
  emxArray_real_T *newBoardCoords1;
  int i47;
  emxArray_real_T *newBoardCoords2;
  emxArray_real_T *r56;
  emxInit_real_T1(&b_board, 2);
  b_rot90((*board)->BoardIdx, b_board);
  i46 = (*board)->BoardIdx->size[0] * (*board)->BoardIdx->size[1];
  (*board)->BoardIdx->size[0] = b_board->size[0];
  (*board)->BoardIdx->size[1] = b_board->size[1];
  emxEnsureCapacity_real_T((*board)->BoardIdx, i46);
  loop_ub = b_board->size[0] * b_board->size[1];
  for (i46 = 0; i46 < loop_ub; i46++) {
    (*board)->BoardIdx->data[i46] = b_board->data[i46];
  }

  loop_ub = (*board)->BoardCoords->size[0];
  b_loop_ub = (*board)->BoardCoords->size[1];
  i46 = b_board->size[0] * b_board->size[1];
  b_board->size[0] = loop_ub;
  b_board->size[1] = b_loop_ub;
  emxEnsureCapacity_real_T(b_board, i46);
  for (i46 = 0; i46 < b_loop_ub; i46++) {
    for (i47 = 0; i47 < loop_ub; i47++) {
      b_board->data[i47 + b_board->size[0] * i46] = (*board)->BoardCoords->
        data[i47 + (*board)->BoardCoords->size[0] * i46];
    }
  }

  emxInit_real_T1(&newBoardCoords1, 2);
  b_rot90(b_board, newBoardCoords1);
  loop_ub = (*board)->BoardCoords->size[0];
  b_loop_ub = (*board)->BoardCoords->size[1];
  i46 = b_board->size[0] * b_board->size[1];
  b_board->size[0] = loop_ub;
  b_board->size[1] = b_loop_ub;
  emxEnsureCapacity_real_T(b_board, i46);
  for (i46 = 0; i46 < b_loop_ub; i46++) {
    for (i47 = 0; i47 < loop_ub; i47++) {
      b_board->data[i47 + b_board->size[0] * i46] = (*board)->BoardCoords->data
        [(i47 + (*board)->BoardCoords->size[0] * i46) + (*board)->
        BoardCoords->size[0] * (*board)->BoardCoords->size[1]];
    }
  }

  emxInit_real_T1(&newBoardCoords2, 2);
  emxInit_real_T(&r56, 3);
  b_rot90(b_board, newBoardCoords2);
  cat(newBoardCoords1, newBoardCoords2, r56);
  i46 = (*board)->BoardCoords->size[0] * (*board)->BoardCoords->size[1] *
    (*board)->BoardCoords->size[2];
  (*board)->BoardCoords->size[0] = r56->size[0];
  (*board)->BoardCoords->size[1] = r56->size[1];
  (*board)->BoardCoords->size[2] = 2;
  emxEnsureCapacity_real_T1((*board)->BoardCoords, i46);
  loop_ub = r56->size[0] * r56->size[1] * r56->size[2];
  emxFree_real_T(&b_board);
  emxFree_real_T(&newBoardCoords2);
  emxFree_real_T(&newBoardCoords1);
  for (i46 = 0; i46 < loop_ub; i46++) {
    (*board)->BoardCoords->data[i46] = r56->data[i46];
  }

  emxFree_real_T(&r56);
}

static float rt_atan2f_snf(float u0, float u1)
{
  float y;
  int b_u0;
  int b_u1;
  if (rtIsNaNF(u0) || rtIsNaNF(u1)) {
    y = ((real32_T)rtNaN);
  } else if (rtIsInfF(u0) && rtIsInfF(u1)) {
    if (u0 > 0.0F) {
      b_u0 = 1;
    } else {
      b_u0 = -1;
    }

    if (u1 > 0.0F) {
      b_u1 = 1;
    } else {
      b_u1 = -1;
    }

    y = std::atan2((float)b_u0, (float)b_u1);
  } else if (u1 == 0.0F) {
    if (u0 > 0.0F) {
      y = RT_PIF / 2.0F;
    } else if (u0 < 0.0F) {
      y = -(RT_PIF / 2.0F);
    } else {
      y = 0.0F;
    }
  } else {
    y = std::atan2(u0, u1);
  }

  return y;
}

static double rt_hypotd_snf(double u0, double u1)
{
  double y;
  double a;
  double b;
  a = std::abs(u0);
  b = std::abs(u1);
  if (a < b) {
    a /= b;
    y = b * std::sqrt(a * a + 1.0);
  } else if (a > b) {
    b /= a;
    y = a * std::sqrt(b * b + 1.0);
  } else if (rtIsNaN(b)) {
    y = b;
  } else {
    y = a * 1.4142135623730951;
  }

  return y;
}

static float rt_hypotf_snf(float u0, float u1)
{
  float y;
  float a;
  float b;
  a = std::abs(u0);
  b = std::abs(u1);
  if (a < b) {
    a /= b;
    y = b * std::sqrt(a * a + 1.0F);
  } else if (a > b) {
    b /= a;
    y = a * std::sqrt(b * b + 1.0F);
  } else if (rtIsNaNF(b)) {
    y = b;
  } else {
    y = a * 1.41421354F;
  }

  return y;
}

static double rt_remd_snf(double u0, double u1)
{
  double y;
  double b_u1;
  double q;
  if (!((!rtIsNaN(u0)) && (!rtIsInf(u0)) && ((!rtIsNaN(u1)) && (!rtIsInf(u1)))))
  {
    y = rtNaN;
  } else {
    if (u1 < 0.0) {
      b_u1 = std::ceil(u1);
    } else {
      b_u1 = std::floor(u1);
    }

    if ((u1 != 0.0) && (u1 != b_u1)) {
      q = std::abs(u0 / u1);
      if (std::abs(q - std::floor(q + 0.5)) <= DBL_EPSILON * q) {
        y = 0.0 * u0;
      } else {
        y = std::fmod(u0, u1);
      }
    } else {
      y = std::fmod(u0, u1);
    }
  }

  return y;
}

static double rt_roundd_snf(double u)
{
  double y;
  if (std::abs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = std::floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = std::ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

static void secondDerivCornerMetric(const emxArray_real32_T *I,
  emxArray_real32_T *cxy, emxArray_real32_T *c45, emxArray_real32_T *Ix,
  emxArray_real32_T *Iy, emxArray_real32_T *Ixy, emxArray_real32_T *I_45_45)
{
  unsigned int finalSize_idx_0;
  double pad[2];
  unsigned int finalSize_idx_1;
  emxArray_real32_T *a;
  emxArray_real_T *b_a;
  int ma;
  int aidx;
  emxArray_real_T *result;
  int iv0[2];
  emxArray_real32_T *I_45;
  int cEnd;
  int cEnd1;
  int na;
  int j;
  int lastColA;
  int firstRowA;
  int k;
  int cidx;
  int r;
  int b_j;
  int iC;
  int iA;
  int iB;
  int i;
  int b_i;
  int a_length;
  static const double dv0[225] = { 1.9045144150126372E-7, 9.67192226178406E-7,
    3.8253194603479479E-6, 1.1782813454257782E-5, 2.8265500088842114E-5,
    5.2806906275779393E-5, 7.6833595263807038E-5, 8.7063869616745574E-5,
    7.6833595263807038E-5, 5.2806906275779393E-5, 2.8265500088842114E-5,
    1.1782813454257782E-5, 3.8253194603479479E-6, 9.67192226178406E-7,
    1.9045144150126372E-7, 9.67192226178406E-7, 4.9118074140370011E-6,
    1.9426575170726453E-5, 5.98380641576443E-5, 0.00014354405374659103,
    0.00026817559812550236, 0.00039019319288270693, 0.00044214681291224515,
    0.00039019319288270693, 0.00026817559812550236, 0.00014354405374659103,
    5.98380641576443E-5, 1.9426575170726453E-5, 4.9118074140370011E-6,
    9.67192226178406E-7, 3.8253194603479479E-6, 1.9426575170726453E-5,
    7.6833595263807038E-5, 0.00023666413469452708, 0.00056772774568680253,
    0.0010606550658014775, 0.0015432440146124532, 0.0017487245678627402,
    0.0015432440146124532, 0.0010606550658014775, 0.00056772774568680253,
    0.00023666413469452708, 7.6833595263807038E-5, 1.9426575170726453E-5,
    3.8253194603479479E-6, 1.1782813454257782E-5, 5.98380641576443E-5,
    0.00023666413469452708, 0.00072897685522068864, 0.0017487245678627402,
    0.0032670476045719744, 0.0047535262158011835, 0.0053864508780477162,
    0.0047535262158011835, 0.0032670476045719744, 0.0017487245678627402,
    0.00072897685522068864, 0.00023666413469452708, 5.98380641576443E-5,
    1.1782813454257782E-5, 2.8265500088842114E-5, 0.00014354405374659103,
    0.00056772774568680253, 0.0017487245678627402, 0.0041949721617992172,
    0.0078372397828220981, 0.01140311659831037, 0.012921423933516047,
    0.01140311659831037, 0.0078372397828220981, 0.0041949721617992172,
    0.0017487245678627402, 0.00056772774568680253, 0.00014354405374659103,
    2.8265500088842114E-5, 5.2806906275779393E-5, 0.00026817559812550236,
    0.0010606550658014775, 0.0032670476045719744, 0.0078372397828220981,
    0.014641891541684373, 0.021303826486921626, 0.024140398028059319,
    0.021303826486921626, 0.014641891541684373, 0.0078372397828220981,
    0.0032670476045719744, 0.0010606550658014775, 0.00026817559812550236,
    5.2806906275779393E-5, 7.6833595263807038E-5, 0.00039019319288270693,
    0.0015432440146124532, 0.0047535262158011835, 0.01140311659831037,
    0.021303826486921626, 0.0309968846369868, 0.035124071876292469,
    0.0309968846369868, 0.021303826486921626, 0.01140311659831037,
    0.0047535262158011835, 0.0015432440146124532, 0.00039019319288270693,
    7.6833595263807038E-5, 8.7063869616745574E-5, 0.00044214681291224515,
    0.0017487245678627402, 0.0053864508780477162, 0.012921423933516047,
    0.024140398028059319, 0.035124071876292469, 0.039800787712028829,
    0.035124071876292469, 0.024140398028059319, 0.012921423933516047,
    0.0053864508780477162, 0.0017487245678627402, 0.00044214681291224515,
    8.7063869616745574E-5, 7.6833595263807038E-5, 0.00039019319288270693,
    0.0015432440146124532, 0.0047535262158011835, 0.01140311659831037,
    0.021303826486921626, 0.0309968846369868, 0.035124071876292469,
    0.0309968846369868, 0.021303826486921626, 0.01140311659831037,
    0.0047535262158011835, 0.0015432440146124532, 0.00039019319288270693,
    7.6833595263807038E-5, 5.2806906275779393E-5, 0.00026817559812550236,
    0.0010606550658014775, 0.0032670476045719744, 0.0078372397828220981,
    0.014641891541684373, 0.021303826486921626, 0.024140398028059319,
    0.021303826486921626, 0.014641891541684373, 0.0078372397828220981,
    0.0032670476045719744, 0.0010606550658014775, 0.00026817559812550236,
    5.2806906275779393E-5, 2.8265500088842114E-5, 0.00014354405374659103,
    0.00056772774568680253, 0.0017487245678627402, 0.0041949721617992172,
    0.0078372397828220981, 0.01140311659831037, 0.012921423933516047,
    0.01140311659831037, 0.0078372397828220981, 0.0041949721617992172,
    0.0017487245678627402, 0.00056772774568680253, 0.00014354405374659103,
    2.8265500088842114E-5, 1.1782813454257782E-5, 5.98380641576443E-5,
    0.00023666413469452708, 0.00072897685522068864, 0.0017487245678627402,
    0.0032670476045719744, 0.0047535262158011835, 0.0053864508780477162,
    0.0047535262158011835, 0.0032670476045719744, 0.0017487245678627402,
    0.00072897685522068864, 0.00023666413469452708, 5.98380641576443E-5,
    1.1782813454257782E-5, 3.8253194603479479E-6, 1.9426575170726453E-5,
    7.6833595263807038E-5, 0.00023666413469452708, 0.00056772774568680253,
    0.0010606550658014775, 0.0015432440146124532, 0.0017487245678627402,
    0.0015432440146124532, 0.0010606550658014775, 0.00056772774568680253,
    0.00023666413469452708, 7.6833595263807038E-5, 1.9426575170726453E-5,
    3.8253194603479479E-6, 9.67192226178406E-7, 4.9118074140370011E-6,
    1.9426575170726453E-5, 5.98380641576443E-5, 0.00014354405374659103,
    0.00026817559812550236, 0.00039019319288270693, 0.00044214681291224515,
    0.00039019319288270693, 0.00026817559812550236, 0.00014354405374659103,
    5.98380641576443E-5, 1.9426575170726453E-5, 4.9118074140370011E-6,
    9.67192226178406E-7, 1.9045144150126372E-7, 9.67192226178406E-7,
    3.8253194603479479E-6, 1.1782813454257782E-5, 2.8265500088842114E-5,
    5.2806906275779393E-5, 7.6833595263807038E-5, 8.7063869616745574E-5,
    7.6833595263807038E-5, 5.2806906275779393E-5, 2.8265500088842114E-5,
    1.1782813454257782E-5, 3.8253194603479479E-6, 9.67192226178406E-7,
    1.9045144150126372E-7 };

  emxArray_real32_T *r1;
  emxArray_real32_T *r2;
  emxArray_int32_T *r3;
  emxArray_int32_T *r4;
  finalSize_idx_0 = (unsigned int)I->size[0];
  pad[0] = 7.0;
  finalSize_idx_1 = (unsigned int)I->size[1];
  pad[1] = 7.0;
  emxInit_real32_T(&a, 2);
  emxInit_real_T1(&b_a, 2);
  if ((I->size[0] == 0) || (I->size[1] == 0)) {
    ma = Ix->size[0] * Ix->size[1];
    Ix->size[0] = I->size[0];
    Ix->size[1] = I->size[1];
    emxEnsureCapacity_real32_T(Ix, ma);
    aidx = I->size[0] * I->size[1];
    for (ma = 0; ma < aidx; ma++) {
      Ix->data[ma] = I->data[ma];
    }
  } else {
    padImage(I, pad, a);
    ma = b_a->size[0] * b_a->size[1];
    b_a->size[0] = a->size[0];
    b_a->size[1] = a->size[1];
    emxEnsureCapacity_real_T(b_a, ma);
    aidx = a->size[0] * a->size[1];
    for (ma = 0; ma < aidx; ma++) {
      b_a->data[ma] = a->data[ma];
    }

    for (ma = 0; ma < 2; ma++) {
      iv0[ma] = b_a->size[ma];
    }

    emxInit_real_T1(&result, 2);
    ma = result->size[0] * result->size[1];
    result->size[0] = iv0[0];
    result->size[1] = iv0[1];
    emxEnsureCapacity_real_T(result, ma);
    aidx = iv0[0] * iv0[1];
    for (ma = 0; ma < aidx; ma++) {
      result->data[ma] = 0.0;
    }

    cEnd = iv0[1] + 6;
    cEnd1 = iv0[0];
    ma = b_a->size[0];
    na = b_a->size[1] - 1;
    for (j = 0; j < 15; j++) {
      if (j + na < cEnd) {
        lastColA = na;
      } else {
        lastColA = cEnd - j;
      }

      if (j < 7) {
        k = 7 - j;
      } else {
        k = 0;
      }

      while (k <= lastColA) {
        if (j + k > 7) {
          b_j = (j + k) - 7;
        } else {
          b_j = 0;
        }

        iC = b_j * cEnd1;
        iA = k * ma;
        iB = j * 15;
        for (i = 0; i < 15; i++) {
          if (i < 7) {
            firstRowA = 7 - i;
          } else {
            firstRowA = 0;
          }

          if (i + ma <= cEnd1 + 6) {
            b_i = ma;
          } else {
            b_i = (cEnd1 - i) + 7;
          }

          a_length = b_i - firstRowA;
          aidx = iA + firstRowA;
          cidx = iC;
          for (r = 1; r <= a_length; r++) {
            result->data[cidx] += dv0[iB] * b_a->data[aidx];
            aidx++;
            cidx++;
          }

          iB++;
          if (i >= 7) {
            iC++;
          }
        }

        k++;
      }
    }

    if (8U > finalSize_idx_0 + 7U) {
      ma = 0;
      firstRowA = 0;
    } else {
      ma = 7;
      firstRowA = (int)(finalSize_idx_0 + 7U);
    }

    if (8U > finalSize_idx_1 + 7U) {
      cidx = 0;
      r = 0;
    } else {
      cidx = 7;
      r = (int)(finalSize_idx_1 + 7U);
    }

    cEnd = Ix->size[0] * Ix->size[1];
    Ix->size[0] = firstRowA - ma;
    Ix->size[1] = r - cidx;
    emxEnsureCapacity_real32_T(Ix, cEnd);
    aidx = r - cidx;
    for (r = 0; r < aidx; r++) {
      cEnd1 = firstRowA - ma;
      for (cEnd = 0; cEnd < cEnd1; cEnd++) {
        Ix->data[cEnd + Ix->size[0] * r] = (float)result->data[(ma + cEnd) +
          result->size[0] * (cidx + r)];
      }
    }

    emxFree_real_T(&result);
  }

  ma = Iy->size[0] * Iy->size[1];
  Iy->size[0] = Ix->size[0];
  Iy->size[1] = Ix->size[1];
  emxEnsureCapacity_real32_T(Iy, ma);
  aidx = Ix->size[0] * Ix->size[1];
  for (ma = 0; ma < aidx; ma++) {
    Iy->data[ma] = Ix->data[ma];
  }

  emxInit_real32_T(&I_45, 2);
  imfilter(Iy);
  b_imfilter(Ix);
  ma = I_45->size[0] * I_45->size[1];
  I_45->size[0] = Ix->size[0];
  I_45->size[1] = Ix->size[1];
  emxEnsureCapacity_real32_T(I_45, ma);
  aidx = Ix->size[0] * Ix->size[1];
  for (ma = 0; ma < aidx; ma++) {
    I_45->data[ma] = Ix->data[ma] * 0.707106769F + Iy->data[ma] * 0.707106769F;
  }

  ma = Ixy->size[0] * Ixy->size[1];
  Ixy->size[0] = Ix->size[0];
  Ixy->size[1] = Ix->size[1];
  emxEnsureCapacity_real32_T(Ixy, ma);
  aidx = Ix->size[0] * Ix->size[1];
  for (ma = 0; ma < aidx; ma++) {
    Ixy->data[ma] = Ix->data[ma];
  }

  imfilter(Ixy);
  ma = I_45_45->size[0] * I_45_45->size[1];
  I_45_45->size[0] = I_45->size[0];
  I_45_45->size[1] = I_45->size[1];
  emxEnsureCapacity_real32_T(I_45_45, ma);
  aidx = I_45->size[0] * I_45->size[1];
  for (ma = 0; ma < aidx; ma++) {
    I_45_45->data[ma] = I_45->data[ma];
  }

  b_imfilter(I_45_45);
  ma = a->size[0] * a->size[1];
  a->size[0] = I_45->size[0];
  a->size[1] = I_45->size[1];
  emxEnsureCapacity_real32_T(a, ma);
  aidx = I_45->size[0] * I_45->size[1];
  for (ma = 0; ma < aidx; ma++) {
    a->data[ma] = I_45->data[ma];
  }

  imfilter(a);
  ma = I_45_45->size[0] * I_45_45->size[1];
  emxEnsureCapacity_real32_T(I_45_45, ma);
  aidx = I_45_45->size[0];
  firstRowA = I_45_45->size[1];
  aidx *= firstRowA;
  for (ma = 0; ma < aidx; ma++) {
    I_45_45->data[ma] = I_45_45->data[ma] * 0.707106769F + a->data[ma] *
      -0.707106769F;
  }

  emxInit_real32_T(&r1, 2);
  b_abs(Ixy, r1);
  ma = r1->size[0] * r1->size[1];
  emxEnsureCapacity_real32_T(r1, ma);
  ma = r1->size[0];
  firstRowA = r1->size[1];
  aidx = ma * firstRowA;
  for (ma = 0; ma < aidx; ma++) {
    r1->data[ma] *= 4.0F;
  }

  emxInit_real32_T(&r2, 2);
  b_abs(I_45, r2);
  ma = I_45->size[0] * I_45->size[1];
  I_45->size[0] = Ix->size[0];
  I_45->size[1] = Ix->size[1];
  emxEnsureCapacity_real32_T(I_45, ma);
  aidx = Ix->size[0] * Ix->size[1];
  for (ma = 0; ma < aidx; ma++) {
    I_45->data[ma] = Ix->data[ma] * 0.707106769F + Iy->data[ma] * -0.707106769F;
  }

  b_abs(I_45, a);
  ma = r2->size[0] * r2->size[1];
  emxEnsureCapacity_real32_T(r2, ma);
  ma = r2->size[0];
  firstRowA = r2->size[1];
  aidx = ma * firstRowA;
  emxFree_real32_T(&I_45);
  for (ma = 0; ma < aidx; ma++) {
    r2->data[ma] = 3.0F * (r2->data[ma] + a->data[ma]);
  }

  ma = cxy->size[0] * cxy->size[1];
  cxy->size[0] = r1->size[0];
  cxy->size[1] = r1->size[1];
  emxEnsureCapacity_real32_T(cxy, ma);
  aidx = r1->size[0] * r1->size[1];
  for (ma = 0; ma < aidx; ma++) {
    cxy->data[ma] = r1->data[ma] - r2->data[ma];
  }

  ma = b_a->size[0] * b_a->size[1];
  b_a->size[0] = r1->size[0];
  b_a->size[1] = r1->size[1];
  emxEnsureCapacity_real_T(b_a, ma);
  aidx = r1->size[0] * r1->size[1];
  for (ma = 0; ma < aidx; ma++) {
    b_a->data[ma] = r1->data[ma] - r2->data[ma];
  }

  firstRowA = b_a->size[0] * b_a->size[1] - 1;
  aidx = 0;
  for (i = 0; i <= firstRowA; i++) {
    if (r1->data[i] - r2->data[i] < 0.0F) {
      aidx++;
    }
  }

  emxInit_int32_T(&r3, 1);
  ma = r3->size[0];
  r3->size[0] = aidx;
  emxEnsureCapacity_int32_T(r3, ma);
  aidx = 0;
  for (i = 0; i <= firstRowA; i++) {
    if (r1->data[i] - r2->data[i] < 0.0F) {
      r3->data[aidx] = i + 1;
      aidx++;
    }
  }

  aidx = r3->size[0];
  for (ma = 0; ma < aidx; ma++) {
    cxy->data[r3->data[ma] - 1] = 0.0F;
  }

  emxFree_int32_T(&r3);
  b_abs(I_45_45, r1);
  ma = r1->size[0] * r1->size[1];
  emxEnsureCapacity_real32_T(r1, ma);
  ma = r1->size[0];
  firstRowA = r1->size[1];
  aidx = ma * firstRowA;
  for (ma = 0; ma < aidx; ma++) {
    r1->data[ma] *= 4.0F;
  }

  b_abs(Ix, r2);
  b_abs(Iy, a);
  ma = r2->size[0] * r2->size[1];
  emxEnsureCapacity_real32_T(r2, ma);
  ma = r2->size[0];
  firstRowA = r2->size[1];
  aidx = ma * firstRowA;
  for (ma = 0; ma < aidx; ma++) {
    r2->data[ma] = 3.0F * (r2->data[ma] + a->data[ma]);
  }

  emxFree_real32_T(&a);
  ma = c45->size[0] * c45->size[1];
  c45->size[0] = r1->size[0];
  c45->size[1] = r1->size[1];
  emxEnsureCapacity_real32_T(c45, ma);
  aidx = r1->size[0] * r1->size[1];
  for (ma = 0; ma < aidx; ma++) {
    c45->data[ma] = r1->data[ma] - r2->data[ma];
  }

  ma = b_a->size[0] * b_a->size[1];
  b_a->size[0] = r1->size[0];
  b_a->size[1] = r1->size[1];
  emxEnsureCapacity_real_T(b_a, ma);
  aidx = r1->size[0] * r1->size[1];
  for (ma = 0; ma < aidx; ma++) {
    b_a->data[ma] = r1->data[ma] - r2->data[ma];
  }

  firstRowA = b_a->size[0] * b_a->size[1] - 1;
  aidx = 0;
  emxFree_real_T(&b_a);
  for (i = 0; i <= firstRowA; i++) {
    if (r1->data[i] - r2->data[i] < 0.0F) {
      aidx++;
    }
  }

  emxInit_int32_T(&r4, 1);
  ma = r4->size[0];
  r4->size[0] = aidx;
  emxEnsureCapacity_int32_T(r4, ma);
  aidx = 0;
  for (i = 0; i <= firstRowA; i++) {
    if (r1->data[i] - r2->data[i] < 0.0F) {
      r4->data[aidx] = i + 1;
      aidx++;
    }
  }

  emxFree_real32_T(&r2);
  emxFree_real32_T(&r1);
  aidx = r4->size[0];
  for (ma = 0; ma < aidx; ma++) {
    c45->data[r4->data[ma] - 1] = 0.0F;
  }

  emxFree_int32_T(&r4);
}

static void sort(emxArray_real32_T *x, emxArray_int32_T *idx)
{
  int dim;
  dim = 2;
  if (x->size[0] != 1) {
    dim = 1;
  }

  b_sort(x, dim, idx);
}

static void sortIdx(emxArray_real32_T *x, emxArray_int32_T *idx)
{
  int ib;
  int i;
  int n;
  int b_n;
  emxArray_int32_T *iwork;
  float x4[4];
  int idx4[4];
  emxArray_real32_T *xwork;
  int nNaNs;
  int k;
  int wOffset;
  signed char perm[4];
  int nNonNaN;
  int i3;
  int i4;
  int nBlocks;
  int bLen2;
  int nPairs;
  int b_iwork[256];
  float b_xwork[256];
  int exitg1;
  ib = x->size[0];
  i = idx->size[0];
  idx->size[0] = ib;
  emxEnsureCapacity_int32_T(idx, i);
  for (i = 0; i < ib; i++) {
    idx->data[i] = 0;
  }

  n = x->size[0];
  b_n = x->size[0];
  for (i = 0; i < 4; i++) {
    x4[i] = 0.0F;
    idx4[i] = 0;
  }

  emxInit_int32_T(&iwork, 1);
  i = iwork->size[0];
  iwork->size[0] = ib;
  emxEnsureCapacity_int32_T(iwork, i);
  for (i = 0; i < ib; i++) {
    iwork->data[i] = 0;
  }

  emxInit_real32_T1(&xwork, 1);
  ib = x->size[0];
  i = xwork->size[0];
  xwork->size[0] = ib;
  emxEnsureCapacity_real32_T1(xwork, i);
  for (i = 0; i < ib; i++) {
    xwork->data[i] = 0.0F;
  }

  nNaNs = 0;
  ib = 0;
  for (k = 0; k + 1 <= b_n; k++) {
    if (rtIsNaNF(x->data[k])) {
      idx->data[(b_n - nNaNs) - 1] = k + 1;
      xwork->data[(b_n - nNaNs) - 1] = x->data[k];
      nNaNs++;
    } else {
      ib++;
      idx4[ib - 1] = k + 1;
      x4[ib - 1] = x->data[k];
      if (ib == 4) {
        i = k - nNaNs;
        if (x4[0] >= x4[1]) {
          ib = 1;
          wOffset = 2;
        } else {
          ib = 2;
          wOffset = 1;
        }

        if (x4[2] >= x4[3]) {
          i3 = 3;
          i4 = 4;
        } else {
          i3 = 4;
          i4 = 3;
        }

        if (x4[ib - 1] >= x4[i3 - 1]) {
          if (x4[wOffset - 1] >= x4[i3 - 1]) {
            perm[0] = (signed char)ib;
            perm[1] = (signed char)wOffset;
            perm[2] = (signed char)i3;
            perm[3] = (signed char)i4;
          } else if (x4[wOffset - 1] >= x4[i4 - 1]) {
            perm[0] = (signed char)ib;
            perm[1] = (signed char)i3;
            perm[2] = (signed char)wOffset;
            perm[3] = (signed char)i4;
          } else {
            perm[0] = (signed char)ib;
            perm[1] = (signed char)i3;
            perm[2] = (signed char)i4;
            perm[3] = (signed char)wOffset;
          }
        } else if (x4[ib - 1] >= x4[i4 - 1]) {
          if (x4[wOffset - 1] >= x4[i4 - 1]) {
            perm[0] = (signed char)i3;
            perm[1] = (signed char)ib;
            perm[2] = (signed char)wOffset;
            perm[3] = (signed char)i4;
          } else {
            perm[0] = (signed char)i3;
            perm[1] = (signed char)ib;
            perm[2] = (signed char)i4;
            perm[3] = (signed char)wOffset;
          }
        } else {
          perm[0] = (signed char)i3;
          perm[1] = (signed char)i4;
          perm[2] = (signed char)ib;
          perm[3] = (signed char)wOffset;
        }

        idx->data[i - 3] = idx4[perm[0] - 1];
        idx->data[i - 2] = idx4[perm[1] - 1];
        idx->data[i - 1] = idx4[perm[2] - 1];
        idx->data[i] = idx4[perm[3] - 1];
        x->data[i - 3] = x4[perm[0] - 1];
        x->data[i - 2] = x4[perm[1] - 1];
        x->data[i - 1] = x4[perm[2] - 1];
        x->data[i] = x4[perm[3] - 1];
        ib = 0;
      }
    }
  }

  wOffset = (b_n - nNaNs) - 1;
  if (ib > 0) {
    for (i = 0; i < 4; i++) {
      perm[i] = 0;
    }

    if (ib == 1) {
      perm[0] = 1;
    } else if (ib == 2) {
      if (x4[0] >= x4[1]) {
        perm[0] = 1;
        perm[1] = 2;
      } else {
        perm[0] = 2;
        perm[1] = 1;
      }
    } else if (x4[0] >= x4[1]) {
      if (x4[1] >= x4[2]) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 3;
      } else if (x4[0] >= x4[2]) {
        perm[0] = 1;
        perm[1] = 3;
        perm[2] = 2;
      } else {
        perm[0] = 3;
        perm[1] = 1;
        perm[2] = 2;
      }
    } else if (x4[0] >= x4[2]) {
      perm[0] = 2;
      perm[1] = 1;
      perm[2] = 3;
    } else if (x4[1] >= x4[2]) {
      perm[0] = 2;
      perm[1] = 3;
      perm[2] = 1;
    } else {
      perm[0] = 3;
      perm[1] = 2;
      perm[2] = 1;
    }

    for (k = 1; k <= ib; k++) {
      idx->data[(wOffset - ib) + k] = idx4[perm[k - 1] - 1];
      x->data[(wOffset - ib) + k] = x4[perm[k - 1] - 1];
    }
  }

  i = (nNaNs >> 1) + 1;
  for (k = 1; k < i; k++) {
    ib = idx->data[wOffset + k];
    idx->data[wOffset + k] = idx->data[b_n - k];
    idx->data[b_n - k] = ib;
    x->data[wOffset + k] = xwork->data[b_n - k];
    x->data[b_n - k] = xwork->data[wOffset + k];
  }

  if ((nNaNs & 1) != 0) {
    x->data[wOffset + i] = xwork->data[wOffset + i];
  }

  nNonNaN = n - nNaNs;
  ib = 2;
  if (nNonNaN > 1) {
    if (n >= 256) {
      nBlocks = nNonNaN >> 8;
      if (nBlocks > 0) {
        for (i3 = 1; i3 <= nBlocks; i3++) {
          i4 = ((i3 - 1) << 8) - 1;
          for (b_n = 0; b_n < 6; b_n++) {
            n = 1 << (b_n + 2);
            bLen2 = n << 1;
            nPairs = 256 >> (b_n + 3);
            for (k = 1; k <= nPairs; k++) {
              ib = i4 + (k - 1) * bLen2;
              for (i = 1; i <= bLen2; i++) {
                b_iwork[i - 1] = idx->data[ib + i];
                b_xwork[i - 1] = x->data[ib + i];
              }

              wOffset = 0;
              i = n;
              do {
                exitg1 = 0;
                ib++;
                if (b_xwork[wOffset] >= b_xwork[i]) {
                  idx->data[ib] = b_iwork[wOffset];
                  x->data[ib] = b_xwork[wOffset];
                  if (wOffset + 1 < n) {
                    wOffset++;
                  } else {
                    exitg1 = 1;
                  }
                } else {
                  idx->data[ib] = b_iwork[i];
                  x->data[ib] = b_xwork[i];
                  if (i + 1 < bLen2) {
                    i++;
                  } else {
                    i = ib - wOffset;
                    while (wOffset + 1 <= n) {
                      idx->data[(i + wOffset) + 1] = b_iwork[wOffset];
                      x->data[(i + wOffset) + 1] = b_xwork[wOffset];
                      wOffset++;
                    }

                    exitg1 = 1;
                  }
                }
              } while (exitg1 == 0);
            }
          }
        }

        ib = nBlocks << 8;
        i = nNonNaN - ib;
        if (i > 0) {
          merge_block(idx, x, ib, i, 2, iwork, xwork);
        }

        ib = 8;
      }
    }

    merge_block(idx, x, 0, nNonNaN, ib, iwork, xwork);
  }

  if ((nNaNs > 0) && (nNonNaN > 0)) {
    for (k = 0; k + 1 <= nNaNs; k++) {
      xwork->data[k] = x->data[nNonNaN + k];
      iwork->data[k] = idx->data[nNonNaN + k];
    }

    for (k = nNonNaN - 1; k + 1 > 0; k--) {
      x->data[nNaNs + k] = x->data[k];
      idx->data[nNaNs + k] = idx->data[k];
    }

    for (k = 0; k + 1 <= nNaNs; k++) {
      x->data[k] = xwork->data[k];
      idx->data[k] = iwork->data[k];
    }
  }

  emxFree_real32_T(&xwork);
  emxFree_int32_T(&iwork);
}

static void squeeze(const emxArray_real_T *a, emxArray_real_T *b)
{
  int k;
  int i11;
  int sqsz[3];
  k = 3;
  while ((k > 2) && (a->size[2] == 1)) {
    k = 2;
  }

  if (k <= 2) {
    sqsz[1] = a->size[1];
    i11 = b->size[0] * b->size[1];
    b->size[0] = 1;
    b->size[1] = sqsz[1];
    emxEnsureCapacity_real_T(b, i11);
    i11 = a->size[1] * a->size[2];
    for (k = 0; k + 1 <= i11; k++) {
      b->data[k] = a->data[k];
    }
  } else {
    for (i11 = 0; i11 < 3; i11++) {
      sqsz[i11] = 1;
    }

    k = 0;
    if (a->size[1] != 1) {
      sqsz[0] = a->size[1];
      k = 1;
    }

    if (a->size[2] != 1) {
      sqsz[k] = a->size[2];
    }

    i11 = b->size[0] * b->size[1];
    b->size[0] = sqsz[0];
    b->size[1] = sqsz[1];
    emxEnsureCapacity_real_T(b, i11);
    i11 = a->size[1] * a->size[2];
    for (k = 0; k + 1 <= i11; k++) {
      b->data[k] = a->data[k];
    }
  }
}

static void subPixelLocation(const emxArray_real32_T *metric, emxArray_real_T
  *loc)
{
  int i48;
  int id;
  emxArray_real32_T *b_metric;
  int loop_ub;
  int i49;
  int b_loop_ub;
  double loc_data[2];
  int loc_size[1];
  emxArray_boolean_T b_loc_data;
  boolean_T c_loc_data[2];
  int i50;
  float subPixelLoc_data[2];
  int i51;
  int i52;
  int i53;
  int c_loop_ub;
  float x;
  float beta[6];
  float y;
  float b_x[2];
  float d_loc_data[2];
  i48 = loc->size[0];
  id = 0;
  emxInit_real32_T(&b_metric, 2);
  while (id <= i48 - 1) {
    loop_ub = loc->size[1];
    for (i49 = 0; i49 < loop_ub; i49++) {
      loc_data[i49] = loc->data[id + loc->size[0] * i49];
    }

    b_loop_ub = loc->size[1];
    loc_size[0] = b_loop_ub;
    for (i49 = 0; i49 < b_loop_ub; i49++) {
      c_loc_data[i49] = (loc_data[i49] < 3.0);
    }

    b_loc_data.data = (boolean_T *)&c_loc_data;
    b_loc_data.size = (int *)&loc_size;
    b_loc_data.allocatedSize = 2;
    b_loc_data.numDimensions = 1;
    b_loc_data.canFreeData = false;
    if (any(&b_loc_data) || (loc->data[id] > ((double)metric->size[1] - 2.0) -
         1.0) || (loc->data[id + loc->size[0]] > ((double)metric->size[0] - 2.0)
                  - 1.0)) {
      for (i49 = 0; i49 < loop_ub; i49++) {
        subPixelLoc_data[i49] = (float)loc_data[i49];
      }
    } else {
      if (loc->data[id + loc->size[0]] - 2.0 > loc->data[id + loc->size[0]] +
          2.0) {
        i49 = 0;
        i50 = 0;
      } else {
        i49 = (int)(loc->data[id + loc->size[0]] - 2.0) - 1;
        i50 = (int)(loc->data[id + loc->size[0]] + 2.0);
      }

      if (loc->data[id] - 2.0 > loc->data[id] + 2.0) {
        i51 = 0;
        i52 = 0;
      } else {
        i51 = (int)(loc->data[id] - 2.0) - 1;
        i52 = (int)(loc->data[id] + 2.0);
      }

      i53 = b_metric->size[0] * b_metric->size[1];
      b_metric->size[0] = i50 - i49;
      b_metric->size[1] = i52 - i51;
      emxEnsureCapacity_real32_T(b_metric, i53);
      b_loop_ub = i52 - i51;
      for (i52 = 0; i52 < b_loop_ub; i52++) {
        c_loop_ub = i50 - i49;
        for (i53 = 0; i53 < c_loop_ub; i53++) {
          b_metric->data[i53 + b_metric->size[0] * i52] = metric->data[(i49 +
            i53) + metric->size[0] * (i51 + i52)];
        }
      }

      for (i49 = 0; i49 < 6; i49++) {
        beta[i49] = 0.0F;
        for (i50 = 0; i50 < 25; i50++) {
          x = beta[i49] + (float)X[i49 + 6 * i50] * b_metric->data[i50];
          beta[i49] = x;
        }
      }

      x = -(2.0F * beta[1] * beta[2] - beta[3] * beta[4]) / (4.0F * beta[0] *
        beta[1] - beta[4] * beta[4]);
      y = -(2.0F * beta[0] * beta[3] - beta[2] * beta[4]) / (4.0F * beta[0] *
        beta[1] - beta[4] * beta[4]);
      if ((!((!rtIsInfF(x)) && (!rtIsNaNF(x)))) || (std::abs(x) > 2.0F) ||
          (!((!rtIsInfF(y)) && (!rtIsNaNF(y)))) || (std::abs(y) > 2.0F)) {
        x = 0.0F;
        y = 0.0F;
      }

      for (i49 = 0; i49 < loop_ub; i49++) {
        d_loc_data[i49] = (float)loc_data[i49];
      }

      b_x[0] = x;
      b_x[1] = y;
      loop_ub = 2;
      for (i49 = 0; i49 < 2; i49++) {
        subPixelLoc_data[i49] = d_loc_data[i49] + b_x[i49];
      }
    }

    for (i49 = 0; i49 < loop_ub; i49++) {
      loc->data[id + loc->size[0] * i49] = subPixelLoc_data[i49];
    }

    id++;
  }

  emxFree_real32_T(&b_metric);
}

static void subPixelLocationImpl_init()
{
  static const double dv1[150] = { 0.028571428571428574, 0.028571428571428574,
    -0.04, -0.04, 0.04, -0.0742857142857143, 0.028571428571428574,
    -0.014285714285714285, -0.04, -0.02, 0.02, 0.01142857142857142,
    0.028571428571428574, -0.028571428571428571, -0.04, 0.0, 0.0,
    0.039999999999999994, 0.028571428571428574, -0.014285714285714285, -0.04,
    0.02, -0.02, 0.01142857142857142, 0.028571428571428574, 0.028571428571428574,
    -0.04, 0.04, -0.04, -0.0742857142857143, -0.014285714285714287,
    0.028571428571428571, -0.02, -0.04, 0.02, 0.011428571428571429,
    -0.014285714285714285, -0.014285714285714284, -0.02, -0.02, 0.01,
    0.097142857142857142, -0.01428571428571429, -0.028571428571428574, -0.02,
    0.0, 0.0, 0.12571428571428572, -0.014285714285714285, -0.014285714285714284,
    -0.02, 0.02, -0.01, 0.097142857142857142, -0.014285714285714287,
    0.028571428571428571, -0.02, 0.04, -0.02, 0.011428571428571429,
    -0.028571428571428574, 0.028571428571428571, 0.0, -0.04, 0.0,
    0.040000000000000008, -0.028571428571428574, -0.014285714285714287, 0.0,
    -0.02, 0.0, 0.12571428571428572, -0.028571428571428574,
    -0.028571428571428574, 0.0, 0.0, 0.0, 0.1542857142857143,
    -0.028571428571428574, -0.014285714285714287, 0.0, 0.02, 0.0,
    0.12571428571428572, -0.028571428571428574, 0.028571428571428571, 0.0, 0.04,
    0.0, 0.040000000000000008, -0.014285714285714287, 0.028571428571428571, 0.02,
    -0.04, -0.02, 0.011428571428571429, -0.014285714285714285,
    -0.014285714285714284, 0.02, -0.02, -0.01, 0.097142857142857142,
    -0.01428571428571429, -0.028571428571428574, 0.02, 0.0, 0.0,
    0.12571428571428572, -0.014285714285714285, -0.014285714285714284, 0.02,
    0.02, 0.01, 0.097142857142857142, -0.014285714285714287,
    0.028571428571428571, 0.02, 0.04, 0.02, 0.011428571428571429,
    0.028571428571428574, 0.028571428571428574, 0.04, -0.04, -0.04,
    -0.0742857142857143, 0.028571428571428574, -0.014285714285714285, 0.04,
    -0.02, -0.02, 0.01142857142857142, 0.028571428571428574,
    -0.028571428571428571, 0.04, 0.0, 0.0, 0.039999999999999994,
    0.028571428571428574, -0.014285714285714285, 0.04, 0.02, 0.02,
    0.01142857142857142, 0.028571428571428574, 0.028571428571428574, 0.04, 0.04,
    0.04, -0.0742857142857143 };

  memcpy(&X[0], &dv1[0], 150U * sizeof(double));
}

static void toPoints(const c_vision_internal_calibration_c *b_this,
                     emxArray_real_T *points, double boardSize[2])
{
  emxArray_boolean_T *c_this;
  int i32;
  int loop_ub;
  double numPoints;
  emxArray_real_T *x;
  int b_loop_ub;
  int i33;
  emxInit_boolean_T1(&c_this, 1);
  i32 = c_this->size[0];
  c_this->size[0] = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  emxEnsureCapacity_boolean_T1(c_this, i32);
  loop_ub = b_this->BoardIdx->size[0] * b_this->BoardIdx->size[1];
  for (i32 = 0; i32 < loop_ub; i32++) {
    c_this->data[i32] = (b_this->BoardIdx->data[i32] == 0.0);
  }

  if (any(c_this)) {
    i32 = points->size[0] * points->size[1];
    points->size[0] = 0;
    points->size[1] = 0;
    emxEnsureCapacity_real_T(points, i32);
    for (i32 = 0; i32 < 2; i32++) {
      boardSize[i32] = 0.0;
    }
  } else {
    numPoints = (double)b_this->BoardCoords->size[0] * (double)
      b_this->BoardCoords->size[1];
    i32 = points->size[0] * points->size[1];
    points->size[0] = (int)numPoints;
    points->size[1] = 2;
    emxEnsureCapacity_real_T(points, i32);
    loop_ub = (int)numPoints << 1;
    for (i32 = 0; i32 < loop_ub; i32++) {
      points->data[i32] = 0.0;
    }

    emxInit_real_T1(&x, 2);
    loop_ub = b_this->BoardCoords->size[0];
    b_loop_ub = b_this->BoardCoords->size[1];
    i32 = x->size[0] * x->size[1];
    x->size[0] = b_loop_ub;
    x->size[1] = loop_ub;
    emxEnsureCapacity_real_T(x, i32);
    for (i32 = 0; i32 < loop_ub; i32++) {
      for (i33 = 0; i33 < b_loop_ub; i33++) {
        x->data[i33 + x->size[0] * i32] = b_this->BoardCoords->data[i32 +
          b_this->BoardCoords->size[0] * i33];
      }
    }

    loop_ub = x->size[0] * x->size[1];
    for (i32 = 0; i32 < loop_ub; i32++) {
      points->data[i32] = x->data[i32];
    }

    loop_ub = b_this->BoardCoords->size[0];
    b_loop_ub = b_this->BoardCoords->size[1];
    i32 = x->size[0] * x->size[1];
    x->size[0] = b_loop_ub;
    x->size[1] = loop_ub;
    emxEnsureCapacity_real_T(x, i32);
    for (i32 = 0; i32 < loop_ub; i32++) {
      for (i33 = 0; i33 < b_loop_ub; i33++) {
        x->data[i33 + x->size[0] * i32] = b_this->BoardCoords->data[(i32 +
          b_this->BoardCoords->size[0] * i33) + b_this->BoardCoords->size[0] *
          b_this->BoardCoords->size[1]];
      }
    }

    loop_ub = x->size[0] * x->size[1];
    for (i32 = 0; i32 < loop_ub; i32++) {
      points->data[i32 + points->size[0]] = x->data[i32];
    }

    emxFree_real_T(&x);
    boardSize[0] = (double)b_this->BoardCoords->size[1] + 1.0;
    boardSize[1] = (double)b_this->BoardCoords->size[0] + 1.0;
  }

  emxFree_boolean_T(&c_this);
}

void dCP(const emxArray_uint8_T *I, emxArray_real_T *imagePoints, double
         boardSize[2])
{
  emxArray_real32_T *b_I;
  c_vision_internal_calibration_c lobj_5;
  c_vision_internal_calibration_c lobj_4;
  c_vision_internal_calibration_c lobj_3;
  c_vision_internal_calibration_c lobj_2;
  c_vision_internal_calibration_c lobj_1;
  c_vision_internal_calibration_c lobj_0;
  int Ix;
  int loop_ub;
  emxArray_real32_T *cxy;
  emxArray_real32_T *c45;
  emxArray_real32_T *b_Ix;
  emxArray_real32_T *Iy;
  emxArray_real32_T *Ixy;
  emxArray_real32_T *I_45_45;
  emxArray_real32_T *Ix2;
  emxArray_real32_T *Iy2;
  emxArray_real32_T *points0;
  emxArray_int32_T *r0;
  int psiz;
  emxArray_real32_T *b_cxy;
  c_vision_internal_calibration_c *board0;
  c_vision_internal_calibration_c *board45;
  emxInit_real32_T(&b_I, 2);
  c_emxInitStruct_vision_internal(&lobj_5);
  c_emxInitStruct_vision_internal(&lobj_4);
  c_emxInitStruct_vision_internal(&lobj_3);
  c_emxInitStruct_vision_internal(&lobj_2);
  c_emxInitStruct_vision_internal(&lobj_1);
  c_emxInitStruct_vision_internal(&lobj_0);
  Ix = b_I->size[0] * b_I->size[1];
  b_I->size[0] = I->size[0];
  b_I->size[1] = I->size[1];
  emxEnsureCapacity_real32_T(b_I, Ix);
  loop_ub = I->size[0] * I->size[1];
  for (Ix = 0; Ix < loop_ub; Ix++) {
    b_I->data[Ix] = (float)I->data[Ix] / 255.0F;
  }

  emxInit_real32_T(&cxy, 2);
  emxInit_real32_T(&c45, 2);
  emxInit_real32_T(&b_Ix, 2);
  emxInit_real32_T(&Iy, 2);
  emxInit_real32_T(&Ixy, 2);
  emxInit_real32_T(&I_45_45, 2);
  emxInit_real32_T(&Ix2, 2);
  emxInit_real32_T(&Iy2, 2);
  secondDerivCornerMetric(b_I, cxy, c45, b_Ix, Iy, Ixy, I_45_45);
  power(b_Ix, Ix2);
  power(Iy, Iy2);
  c_imfilter(Ix2);
  c_imfilter(Iy2);
  Ix = b_Ix->size[0] * b_Ix->size[1];
  emxEnsureCapacity_real32_T(b_Ix, Ix);
  Ix = b_Ix->size[0];
  loop_ub = b_Ix->size[1];
  loop_ub *= Ix;
  for (Ix = 0; Ix < loop_ub; Ix++) {
    b_Ix->data[Ix] *= Iy->data[Ix];
  }

  emxFree_real32_T(&Iy);
  emxInit_real32_T(&points0, 2);
  emxInit_int32_T(&r0, 1);
  c_imfilter(b_Ix);
  find_peaks(cxy, points0);
  psiz = cxy->size[0];
  loop_ub = points0->size[0];
  Ix = r0->size[0];
  r0->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r0, Ix);
  for (Ix = 0; Ix < loop_ub; Ix++) {
    r0->data[Ix] = (int)points0->data[Ix + points0->size[0]] + psiz * ((int)
      points0->data[Ix] - 1);
  }

  emxInit_real32_T1(&b_cxy, 1);
  Ix = b_cxy->size[0];
  b_cxy->size[0] = r0->size[0];
  emxEnsureCapacity_real32_T1(b_cxy, Ix);
  loop_ub = r0->size[0];
  for (Ix = 0; Ix < loop_ub; Ix++) {
    b_cxy->data[Ix] = cxy->data[r0->data[Ix] - 1];
  }

  emxFree_real32_T(&cxy);
  board0 = growCheckerboard(points0, b_cxy, Ix2, Iy2, b_Ix, 0.0, &lobj_0,
    &lobj_1, &lobj_2);
  find_peaks(c45, points0);
  psiz = c45->size[0];
  loop_ub = points0->size[0];
  Ix = r0->size[0];
  r0->size[0] = loop_ub;
  emxEnsureCapacity_int32_T(r0, Ix);
  for (Ix = 0; Ix < loop_ub; Ix++) {
    r0->data[Ix] = (int)points0->data[Ix + points0->size[0]] + psiz * ((int)
      points0->data[Ix] - 1);
  }

  Ix = b_cxy->size[0];
  b_cxy->size[0] = r0->size[0];
  emxEnsureCapacity_real32_T1(b_cxy, Ix);
  loop_ub = r0->size[0];
  for (Ix = 0; Ix < loop_ub; Ix++) {
    b_cxy->data[Ix] = c45->data[r0->data[Ix] - 1];
  }

  emxFree_int32_T(&r0);
  emxFree_real32_T(&c45);
  board45 = growCheckerboard(points0, b_cxy, Ix2, Iy2, b_Ix, 0.78539816339744828,
    &lobj_3, &lobj_4, &lobj_5);
  Ix = imagePoints->size[0] * imagePoints->size[1];
  imagePoints->size[0] = 0;
  imagePoints->size[1] = 0;
  emxEnsureCapacity_real_T(imagePoints, Ix);
  emxFree_real32_T(&b_cxy);
  emxFree_real32_T(&Iy2);
  emxFree_real32_T(&Ix2);
  emxFree_real32_T(&b_Ix);
  emxFree_real32_T(&points0);
  for (Ix = 0; Ix < 2; Ix++) {
    boardSize[Ix] = 0.0;
  }

  if (board0->isValid && (board0->Energy < board45->Energy)) {
    board45 = board0;
    orient(&board45, b_I);
    toPoints(board45, imagePoints, boardSize);
    subPixelLocation(Ixy, imagePoints);
  } else {
    if (board45->isValid) {
      orient(&board45, b_I);
      toPoints(board45, imagePoints, boardSize);
      subPixelLocation(I_45_45, imagePoints);
    }
  }

  emxFree_real32_T(&I_45_45);
  emxFree_real32_T(&Ixy);
  if ((imagePoints->size[0] == 0) || (imagePoints->size[1] == 0)) {
    detectCheckerboard(b_I, imagePoints, boardSize);
  }

  emxFree_real32_T(&b_I);
  c_emxFreeStruct_vision_internal(&lobj_0);
  c_emxFreeStruct_vision_internal(&lobj_1);
  c_emxFreeStruct_vision_internal(&lobj_2);
  c_emxFreeStruct_vision_internal(&lobj_3);
  c_emxFreeStruct_vision_internal(&lobj_4);
  c_emxFreeStruct_vision_internal(&lobj_5);
}

void dCP_initialize()
{
  rt_InitInfAndNaN(8U);
  //omp_init_nest_lock(&emlrtNestLockGlobal);
  subPixelLocationImpl_init();
}

void dCP_terminate()
{
  //omp_destroy_nest_lock(&emlrtNestLockGlobal);
}
