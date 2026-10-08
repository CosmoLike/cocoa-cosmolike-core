#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h> 

#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>

#include "structs.h"
#include "basics.h"

#include "log.c/src/log.h"
#include <complex.h>


// ---------------------------------------------------------------------------
// Detect uniform or piecewise-uniform structure in a 1D grid.
//
// A "segment" is a maximal contiguous range of x[] with constant spacing
// (within relative tolerance rtol). A perfectly uniform linspace gives 1
// segment; np.concatenate of two linspaces with different spacings gives 2.
//
// Returns the number of segments found (>= 1). Aborts if x is not strictly
// increasing, or if the number of segments would exceed max_seg.
//
// Caller provides 4 output arrays of length >= max_seg. On return, the first
// nseg entries describe each segment for direct-index lookup:
//
//     start[s]   = index in x[] where segment s begins
//     len[s]     = number of points in segment s
//     xmin[s]    = x[start[s]]
//     inv_dx[s]  = 1 / spacing within segment s     (multiply, don't divide)
//
// To find the bucket of a query value q in segment s:
//     idx = start[s] + (int)((q - xmin[s]) * inv_dx[s]);
// ---------------------------------------------------------------------------
int detect_uniform_segments(const double *x, int n, double rtol, int max_seg,
                            int *start, int *len, double *xmin, double *inv_dx,
                            const char *name)
{
  if (n < 2) {
    log_fatal("%s: need at least two grid points, got n=%d", name, n);
    exit(EXIT_FAILURE);
  }

  int nseg  = 0; // number of segments closed out so far
  int begin = 0; // index where the current segment started
  
  double dx = x[1] - x[0]; // reference spacing of the current segment
  if (dx <= 0.0) {
    log_fatal("%s: not strictly increasing at first interval", name);
    exit(EXIT_FAILURE);
  }

  // Walk pairs (x[i], x[i+1]) and close out a segment whenever the spacing
  // changes, or we hit the end of the array. The (i == n-1) guard handles
  // the final segment without a duplicated close-out block after the loop.
  for (int i = 1; i < n; i++) {
    // At i == n-1, x[i+1] doesn't exist; reuse dx so is_break fires from
    // the end-of-array condition, not from a phantom spacing comparison.
    const double d = (i < n - 1) ? x[i+1] - x[i] : dx;

    if (d <= 0.0) {
        log_fatal("%s: not strictly increasing at i=%d", name, i);
        exit(EXIT_FAILURE);
    }

    // Break the segment if (a) we've reached the end of the array, or
    // (b) the spacing has changed by more than rtol relative to the
    // segment's reference spacing dx.
    const int is_break = (i == n - 1) || (fabs(d - dx) > rtol * fabs(dx));

    if (is_break) {
      if (nseg >= max_seg) {
        log_fatal("%s: more than %d segments detected at i=%d "
                  "(d=%.6e, ref=%.6e)", name, max_seg, i, d, dx);
        exit(EXIT_FAILURE);
      }

      // Record the segment that just ended.
      // Note: when i == n-1 we extend through index n-1 (n - begin
      // points); otherwise we stop at index i (i - begin + 1 points,
      // because index i is shared with the next segment as its start).
      start[nseg]  = begin;
      len[nseg]    = (i == n - 1) ? n - begin : i - begin + 1;
      xmin[nseg]   = x[begin];
      inv_dx[nseg] = 1.0 / dx;
      nseg++;

      // Begin the next segment at i, with the just-measured spacing
      // as its new reference. (Irrelevant if i == n-1, harmless.)
      begin = i;
      dx    = d;
    }
  }

  return nseg;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// SIMD FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
  #if defined(__aarch64__) || defined(_M_ARM64)
    #ifndef SIMDE_ARM_NEON_A64V8_NATIVE
      #warning "SIMDe: NEON is being EMULATED — something is wrong"
    #endif
  #else
    #ifndef SIMDE_X86_AVX2_NATIVE
      #warning "SIMDe: AVX2 is being EMULATED (no -mavx2 flag?)"
    #endif
    #ifndef SIMDE_X86_FMA_NATIVE
      #warning "SIMDe: FMA is being EMULATED (no -mfma flag?)"
    #endif
  #endif

// -----------------------------------------------------------------------------
// What is SIMD? How is the basic building block of SIMD?
// A normal double variable holds 1 number (64 bits).
// A simde__m256d holds 4 doubles side-by-side (256 bits = 4 x 64).
//
// Think of it as a box with 4 slots ("lanes"):
//
//   simde__m256d box = [ slot0 | slot1 | slot2 | slot3 ]
//                        64 bit  64 bit  64 bit  64 bit
//                      <----------- 256 bits ---------->
//
// When you add two such boxes, all 4 slots are added in parallel:
//
//   box_a = [ 1.0 | 2.0 | 3.0 | 4.0 ]
//   box_b = [ 5.0 | 6.0 | 7.0 | 8.0 ]
//   result = [ 6.0 | 8.0 | 10.0 | 12.0 ]   (one AVX2 instruction on
//   x86-64; two NEON instructions, one per two-lane half, on arm64)
// -----------------------------------------------------------------------------
double simd_horizontal_sum(simde__m256d four_lanes)
{ // Takes a 4-lane register and sums all 4 values into a single double
  double tmp[4]; // Store the 4 lanes into a regular C array
  // scalar: for (int l = 0; l < 4; l++) { tmp[l] = four_lanes[l]; }
  //
  // storeu writes lane l of four_lanes to tmp[l], l = 0..3 (the u means
  // tmp needs no 32-byte alignment); the return then adds the four values
  // left to right, ((tmp[0] + tmp[1]) + tmp[2]) + tmp[3]
  simde_mm256_storeu_pd(tmp, four_lanes);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

// ---------------------------------------------------------------------------
// SIMD-accelerated horizontal sum of a double array using AVX2.
//
// Computes: result = a[0] + a[1] + ... + a[n-1]
//
// Uses two independent 256-bit accumulators (4 doubles each) to exploit
// instruction-level parallelism — the CPU can issue adds to both accumulators
// simultaneously since they have no data dependency. This halves the
// effective latency of the reduction chain compared to a single accumulator.
//
// The main loop processes 8 elements per iteration (2 × 4-wide loads).
// Then simd_horizontal_sum adds each accumulator's four lanes in lane
// order, the two partial sums are added (accum_A's first), and a scalar
// tail adds the remaining n % 8 elements one at a time. This order
// differs from a left-to-right loop over a, so the two results can
// differ in the last bits.
//
// scalar: double sum_A[4] = {0.0}, sum_B[4] = {0.0};
//         int q = 0;
//         for (; q <= n - 8; q += 8) {
//           for (int l = 0; l < 4; l++) {
//             sum_A[l] += a[q + l];      // lane l of accum_A
//             sum_B[l] += a[q + 4 + l];  // lane l of accum_B
//           }
//         }
//         double result = (((sum_A[0] + sum_A[1]) + sum_A[2]) + sum_A[3])
//                       + (((sum_B[0] + sum_B[1]) + sum_B[2]) + sum_B[3]);
//         for (; q < n; q++) { result += a[q]; }
// ---------------------------------------------------------------------------
double simd_array_sum(
    const double* restrict a,  // input array, length n (need not be aligned)
    const int n                // number of elements to sum
  )
{
  // setzero: all four lanes of accum_A start at 0.0; lane l collects
  // a[q + l] over the main-loop steps (scalar: sum_A[l] = 0.0)
  simde__m256d accum_A = simde_mm256_setzero_pd();
  // setzero: accum_B likewise; lane l collects a[q + 4 + l]
  // (scalar: sum_B[l] = 0.0)
  simde__m256d accum_B = simde_mm256_setzero_pd();
 
  int q = 0;
  for (; q <= n - 8; q += 8) { // Main loop: process 8 doubles per iteration
    // loadu reads a[q..q+3] into lanes 0..3 (no 32-byte alignment
    // needed; q <= n - 8 keeps this step's eight reads inside a), and
    // add_pd adds lane by lane: sum_A[l] += a[q + l], l = 0..3
    accum_A = simde_mm256_add_pd(accum_A, simde_mm256_loadu_pd(a + q));
    // loadu reads a[q+4..q+7]; add_pd: sum_B[l] += a[q + 4 + l]
    accum_B = simde_mm256_add_pd(accum_B, simde_mm256_loadu_pd(a + q + 4));
  }
  double result = simd_horizontal_sum(accum_A) + simd_horizontal_sum(accum_B);
  for (; q < n; q++) { // Scalar tail: remaining 0-7 elements, one at a time
    result += a[q];
  }
  return result;
}

// ---------------------------------------------------------------------------
// Allocate a GSL interpolation object using the globally configured
// interpolation scheme.
//
// The interpolation type is selected via Ntable.photoz_interpolation_type:
//   - 0: cubic spline (gsl_interp_cspline)
//   - 1: linear (gsl_interp_linear)
//   - 2 (or any other value): Steffen monotone interpolation (gsl_interp_steffen)
//
// @param n  Number of data points the interpolation object must support.
//           Must satisfy the minimum size requirement of the chosen method
//           (e.g., >= 2 for linear, >= 3 for cubic spline).
// @return   Pointer to the newly allocated gsl_interp. Never returns NULL;
//           terminates the program on allocation failure.
// ---------------------------------------------------------------------------
gsl_interp* malloc_gsl_interp(const int n)
{
  gsl_interp* result;
  if (0 == Ntable.photoz_interpolation_type) {
    result = gsl_interp_alloc(gsl_interp_cspline, n);
  }
  else if (1 == Ntable.photoz_interpolation_type) {
    result = gsl_interp_alloc(gsl_interp_linear, n);
  }
  else {
    result = gsl_interp_alloc(gsl_interp_steffen, n);
  }
  if (result == NULL) {
    log_fatal("array allocation failed"); exit(1);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Allocate a GSL spline object using the globally configured
// interpolation scheme.
//
// Behaves identically to malloc_gsl_interp() but returns a gsl_spline,
// which bundles the interpolation object together with copies of the
// data arrays for a more convenient evaluation interface.
//
// @param n  Number of data points the spline must support.
//           Must satisfy the minimum size requirement of the chosen method.
// @return   Pointer to the newly allocated gsl_spline. Never returns NULL;
//           terminates the program on allocation failure.
//
// @see malloc_gsl_interp
// ---------------------------------------------------------------------------
gsl_spline* malloc_gsl_spline(const int n)
{
  gsl_spline* result;
  if (0 == Ntable.photoz_interpolation_type) {
    result = gsl_spline_alloc(gsl_interp_cspline, n);
  }
  else if (1 == Ntable.photoz_interpolation_type) {
    result = gsl_spline_alloc(gsl_interp_linear, n);
  }
  else {
    result = gsl_spline_alloc(gsl_interp_steffen, n);
  }
  if (result == NULL) {
    log_fatal("array allocation failed"); exit(1);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Allocate a Gauss-Legendre fixed-point integration table.
//
// Wraps gsl_integration_glfixed_table_alloc() with a fatal-on-failure
// guarantee, consistent with the other allocation helpers in this module.
//
// @param n  Number of quadrature points (order of the integration rule).
//           Higher values increase accuracy at the cost of more function
//           evaluations per integration call.
// @return   Pointer to the newly allocated table. Never returns NULL;
//           terminates the program on allocation failure.
// ---------------------------------------------------------------------------
gsl_integration_glfixed_table* malloc_gslint_glfixed(const int n)
{
  gsl_integration_glfixed_table* w = gsl_integration_glfixed_table_alloc(n);
  if (w == NULL) {
    log_fatal("array allocation failed"); exit(1);
  }
  return w;
}

// ---------------------------------------------------------------------------
// The malloc*d allocators: layout and ownership rules for all of them
//
// One block. Each allocator below makes a single posix_memalign call and
// returns the start of that block: the pointer tables (2D and higher)
// followed by the data.
//
// Padded rows. The data are stored as rows of the last index. Each row
// starts on a 64-byte boundary and is padded up to a multiple of 64 bytes:
// a row of ny doubles occupies nyp = 8*ceil(ny/8) doubles, so row i+1
// starts nyp (not ny) doubles after row i. The ny values of a row are
// contiguous; the block is not one flat array of nx*ny values. Pointer
// tables are padded as a whole, so each table and the data start on a
// 64-byte boundary. Exceptions: malloc2d_fftwp and malloc2d_ptr pad only
// the row-pointer table, and calloc1d does not round its size up.
//
// Padding stays uninitialized. posix_memalign does not zero memory, and
// zero2d/zero3d/zero4d zero only the logical values of each row, so no
// allocator or zeroing function writes the padding; code must not read
// it. A flat memset or memcpy over a 2D+ block is therefore wrong (see
// zero2d).
//
// One free. free(p) on the returned pointer releases the block, row
// pointers included. Rows are addresses inside the block, not separate
// allocations: never free a row, and free each block exactly once.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Allocate a 4D array as a single 64-byte-aligned contiguous block with
// pointer indirection for convenient multi-index access (result[i][j][k][l]).
//
// All internal pointer arrays and the data region are padded to 64-byte
// boundaries, suitable for SIMD and cache-friendly access patterns.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer arrays and the data block).
// ---------------------------------------------------------------------------
void**** malloc4d(
    const long nx,  // extent of the 1st dimension
    const long ny,  // extent of the 2nd dimension
    const long nz,  // extent of the 3rd dimension
    const long nw   // extent of the 4th dimension
  )
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(double***);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(double***);

  size_t nxyp = nx * ny * sizeof(double**);
  if (nxyp % align != 0) nxyp = nxyp + (align - nxyp % align);
  nxyp = nxyp / sizeof(double**);

  size_t nxyzp = nx * ny * nz * sizeof(double*);
  if (nxyzp % align != 0) nxyzp = nxyzp + (align - nxyzp % align);
  nxyzp = nxyzp / sizeof(double*);

  // each row of nw doubles, padded
  size_t nwp = nw * sizeof(double);
  if (nwp % align != 0) nwp = nwp + (align - nwp % align);
  nwp = nwp / sizeof(double);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     nxp * sizeof(double***) +
                     nxyp * sizeof(double**) +
                     nxyzp * sizeof(double*) +
                     nx * ny * nz * nwp * sizeof(double)) != 0) {
    log_fatal("posix_memalign failed in malloc4d");
    exit(EXIT_FAILURE);
  }

  double**** tab = (double****) raw_block;

  double*** lvl3 = (double***) ((char*) raw_block +
                                nxp * sizeof(double***));
  double**  lvl2 = (double**)  ((char*) raw_block + 
                                nxp * sizeof(double***) +
                                nxyp * sizeof(double**));
  double*   data = (double*)   ((char*) raw_block +
                                nxp * sizeof(double***) +
                                nxyp * sizeof(double**) +
                                nxyzp * sizeof(double*));
  #pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    tab[i] = lvl3 + i * ny;
    for (int j = 0; j < ny; ++j) {
      tab[i][j] = lvl2 + (ny * i + j) * nz;
      for (int k = 0; k < nz; ++k)
        tab[i][j][k] = data + ((ny * i + j) * nz + k) * nwp;
    }
  }
  return (void****) tab;
}

// ---------------------------------------------------------------------------
// Allocate a 3D array as a single 64-byte-aligned contiguous block with
// pointer indirection for convenient multi-index access (result[i][j][k]).
//
// All internal pointer arrays and the data region are padded to 64-byte
// boundaries, suitable for SIMD and cache-friendly access patterns.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer arrays and the data block).
// ---------------------------------------------------------------------------
void*** malloc3d(
    const int nx,  // extent of the 1st dimension
    const int ny,  // extent of the 2nd dimension
    const int nz   // extent of the 3rd dimension
  )
{
  const size_t align = 64;

  // first pointer table: nx double** pointers
  size_t nxp = nx * sizeof(double**);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(double**);

  // second pointer table: nx*ny double* pointers
  size_t nxyp = nx * ny * sizeof(double*);
  if (nxyp % align != 0) nxyp = nxyp + (align - nxyp % align);
  nxyp = nxyp / sizeof(double*);

  // each row of nz doubles, padded
  size_t nzp = nz * sizeof(double);
  if (nzp % align != 0) nzp = nzp + (align - nzp % align);
  nzp = nzp / sizeof(double);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     nxp * sizeof(double**) +
                     nxyp * sizeof(double*) +
                     nx * ny * nzp * sizeof(double)) != 0) {
    log_fatal("posix_memalign failed in malloc3d"); exit(EXIT_FAILURE);
  }

  double*** tab = (double***) raw_block;
  #pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    tab[i] = (double**) ((char*) raw_block +
             nxp * sizeof(double**)) + ny * i;
    for (int j = 0; j < ny; ++j) {
      tab[i][j] = (double*) ((char*) raw_block +
                  nxp * sizeof(double**) +
                  nxyp * sizeof(double*)) + nzp * (ny * i + j);
    }
  }
  return (void***) tab;
}

// ---------------------------------------------------------------------------
// Allocate a 2D array of int as a single 64-byte-aligned contiguous block
// with pointer indirection for convenient multi-index access (result[i][j]).
//
// Both the row-pointer array and each row's data region are padded to
// 64-byte boundaries, suitable for SIMD and cache-friendly OpenMP access.
// Row pointers are wired up in parallel.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer array and the data block).
// ---------------------------------------------------------------------------
void** malloc2d_int(
    const int nx,  // number of rows
    const int ny   // number of columns (ints per row, before padding)
  )
{
  const size_t align = 64;
  size_t nxp = nx * sizeof(int*);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(int*);
  size_t nyp = ny * sizeof(int);
  if (nyp % align != 0) nyp = nyp + (align - nyp % align);
  nyp = nyp / sizeof(int);
  void* raw_block = NULL;
  if (posix_memalign(&raw_block,
                     align,
                     sizeof(int*)*nxp + sizeof(int)*nx*nyp) != 0) {
    log_fatal("array allocation failed (malloc2d_int)"); exit(EXIT_FAILURE);
  }
  int** tab = (int**) raw_block;
  #pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    // with padding, we need to use the byte-level cast 
    // (char*) raw_block + nxp*sizeof(int*) to get the exact padded offset
    tab[i] = (int*) ((char*) raw_block + nxp*sizeof(int*)) + i * nyp;
  }
  return (void**) tab;
}

// ---------------------------------------------------------------------------
// Allocate a 2D array of double as a single 64-byte-aligned contiguous
// block with pointer indirection for convenient multi-index access
// (result[i][j]).
//
// Both the row-pointer array and each row's data region are padded to
// 64-byte boundaries, suitable for SIMD and cache-friendly access patterns.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer array and the data block).
// ---------------------------------------------------------------------------
void** malloc2d(
    const int nx,  // number of rows
    const int ny   // number of columns (doubles per row, before padding)
  )
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(double*);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp/sizeof(double*);

  size_t nyp = ny * sizeof(double);
  if (nyp % align != 0) nyp = nyp + (align - nyp % align);
  nyp = nyp/sizeof(double);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, 
                     align, 
                     sizeof(double*)*nxp+sizeof(double)*nx*nyp) != 0) {
    log_fatal("posix_memalign failed for malloc2d"); exit(EXIT_FAILURE);
  }

  double** tab = (double**) raw_block;
  #pragma omp parallel for
  for (int i = 0; i < nx; ++i) {
    // with padding, the byte-level cast (char*) raw_block plus
    // nxp*sizeof(double*) bytes gives the exact padded start of the data
    tab[i] = (double*) ((char*) raw_block + nxp*sizeof(double*)) + i * nyp;
  }
  return (void**) tab;
}


// ---------------------------------------------------------------------------
// Allocate a 1D array of int as a single 64-byte-aligned contiguous block.
//
// The total allocation is padded to a 64-byte boundary, suitable for
// SIMD and cache-friendly access patterns.
//
// The caller is responsible for freeing the returned pointer.
// ---------------------------------------------------------------------------
void* malloc1d_int(
    const int nx   // number of int elements to allocate
  )
{
  const size_t align = 64;
  size_t nxp = nx * sizeof(int);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  void* vec = NULL;
  if (posix_memalign(&vec, align, nxp) != 0) {
    log_fatal("array allocation failed (malloc1d_int)"); exit(EXIT_FAILURE);
  }
  return vec;
}

// ---------------------------------------------------------------------------
// Allocate a 1D array of double as a single 64-byte-aligned contiguous
// block.
//
// The total allocation is padded to a 64-byte boundary, suitable for
// SIMD and cache-friendly access patterns.
//
// The caller is responsible for freeing the returned pointer.
// ---------------------------------------------------------------------------
void* malloc1d(
    const int nx   // number of double elements to allocate
  )
{
  const size_t align = 64;
  size_t nxp = nx * sizeof(double);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  void* vec = NULL;
  if (posix_memalign(&vec, align, nxp) != 0) {
    log_fatal("array allocation failed (malloc1d)"); exit(EXIT_FAILURE);
  }
  return vec;
}

// ---------------------------------------------------------------------------
// Allocate a 1D array of double as a single 64-byte-aligned contiguous
// block, zero-initialized.
//
// The block starts on a 64-byte boundary, but unlike malloc1d its size
// is not rounded up: exactly nx doubles, all set to zero before
// returning.
//
// The caller is responsible for freeing the returned pointer.
// ---------------------------------------------------------------------------
void* calloc1d(
    const int nx   // number of double elements to allocate
  )
{
  void* vec = NULL;
  if (posix_memalign(&vec, 64, sizeof(double) * nx) != 0) {
    log_fatal("array allocation failed (calloc1d)"); exit(EXIT_FAILURE);
  }
  memset(vec, 0, sizeof(double) * nx);
  return vec;
}

// ---------------------------------------------------------------------------
// Allocate a 2D array of fftw_complex as a single 64-byte-aligned
// contiguous block with pointer indirection for convenient multi-index
// access (result[i][j]).
//
// Both the row-pointer array and each row's data region are padded to
// 64-byte boundaries, suitable for SIMD, cache-friendly access, and
// FFTW alignment requirements.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer array and the data block).
// ---------------------------------------------------------------------------
void** malloc2d_fftwc(
    const long nx,  // number of rows
    const long ny   // number of columns (fftw_complex per row, before padding)
  )
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(fftw_complex*);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(fftw_complex*);

  size_t nyp = ny * sizeof(fftw_complex);
  if (nyp % align != 0) nyp = nyp + (align - nyp % align);
  nyp = nyp / sizeof(fftw_complex);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(fftw_complex*) * nxp +
                     sizeof(fftw_complex)  * nx * nyp) != 0) {
    log_fatal("array allocation failed (malloc2d_fftwc)");
    exit(1);
  }

  fftw_complex** tab = (fftw_complex**) raw_block;
  fftw_complex*  data = (fftw_complex*)
    ((char*) raw_block + sizeof(fftw_complex*) * nxp);

  for (int i = 0; i < nx; i++) {
    tab[i] = data + i * nyp;
  }
  return (void**) tab;
}

// ---------------------------------------------------------------------------
// Allocate a 2D array of fftw_plan as a single 64-byte-aligned contiguous
// block with pointer indirection for convenient multi-index access
// (result[i][j]).
//
// The row-pointer array is padded to a 64-byte boundary. The data region
// is not padded per-row, so plans are stored densely after the pointer
// block.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer array and the data block). Note that
// individual fftw_plan handles may need to be destroyed with
// fftw_destroy_plan() before freeing the container.
// ---------------------------------------------------------------------------
void** malloc2d_fftwp(
    const long nx,  // number of rows
    const long ny   // number of columns (fftw_plan per row)
  )
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(fftw_plan*);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(fftw_plan*);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(fftw_plan*) * nxp +
                     sizeof(fftw_plan)  * nx * ny) != 0) {
    log_fatal("array allocation failed (malloc2d_fftwp)");
    exit(1);
  }

  fftw_plan** tab = (fftw_plan**) raw_block;
  fftw_plan*  data = (fftw_plan*)
    ((char*) raw_block + sizeof(fftw_plan*) * nxp);

  for (int i = 0; i < nx; i++) {
    tab[i] = data + i * ny;
  }
  return (void**) tab;
}

// ---------------------------------------------------------------------------
// Allocate a 2D array of double* pointers as a single 64-byte-aligned
// contiguous block with pointer indirection for convenient multi-index
// access (result[i][j]).
//
// The row-pointer array is padded to a 64-byte boundary. Each entry
// result[i][j] is a double* that the caller can later point at an
// independently allocated data buffer.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases both the pointer array and the data block of double*
// entries). Buffers that the caller attaches to the entries are separate
// allocations, freed separately by their owner.
// ---------------------------------------------------------------------------
void*** malloc2d_ptr(
    const long nx,  // number of rows
    const long ny   // number of columns (double* pointers per row)
  )
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(double**);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(double**);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(double**) * nxp +
                     sizeof(double*)  * nx * ny) != 0) {
    log_fatal("array allocation failed (malloc2d_ptr)");
    exit(1);
  }

  double*** tab = (double***) raw_block;
  double**  data = (double**)
    ((char*) raw_block + sizeof(double**) * nxp);

  for (int i = 0; i < nx; i++) {
    tab[i] = data + i * ny;
  }
  return (void***) tab;
}

// ---------------------------------------------------------------------------
// Allocate a 3D array of double complex as a single 64-byte-aligned
// contiguous block with pointer indirection for convenient multi-index
// access (result[i][j][k]).
//
// All three indirection levels (the row-pointer array, the second-level
// pointer array, and each row's data region) are independently padded to
// 64-byte boundaries, suitable for SIMD and cache-friendly access
// patterns.
//
// The caller is responsible for freeing the returned pointer (a single
// free() releases all pointer arrays and the data block).
// ---------------------------------------------------------------------------
void*** malloc3d_complex(
    const long nx,  // extent of the 1st dimension
    const long ny,  // extent of the 2nd dimension
    const long nz   // extent of the 3rd dimension (complex elements, before padding)
  )
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(double complex**);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(double complex**);

  size_t s2p = nx * ny * sizeof(double complex*);
  if (s2p % align != 0) s2p = s2p + (align - s2p % align);
  s2p = s2p / sizeof(double complex*);

  size_t nzp = nz * sizeof(double complex);
  if (nzp % align != 0) nzp = nzp + (align - nzp % align);
  nzp = nzp / sizeof(double complex);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(double complex**) * nxp +
                     sizeof(double complex*)  * s2p +
                     sizeof(double complex)   * nx * ny * nzp) != 0) {
    log_fatal("array allocation failed (malloc3d_complex)");
    exit(1);
  }

  double complex*** tab = (double complex***) raw_block;
  double complex**  lvl2 = (double complex**)
    ((char*) raw_block + sizeof(double complex**) * nxp);
  double complex*   data = (double complex*)
    ((char*) raw_block + sizeof(double complex**) * nxp +
                         sizeof(double complex*)  * s2p);

  for (int i = 0; i < nx; i++) {
    tab[i] = lvl2 + i * ny;
    for (int j = 0; j < ny; j++) {
      tab[i][j] = data + ((long)(ny * i) + j) * nzp;
    }
  }
  return (void***) tab;
}

// ---------------------------------------------------------------------------
// Return the smaller of two doubles.
// ---------------------------------------------------------------------------
double fmin(
    const double a,  // first value
    const double b   // second value
  )
{
  return a < b ? a : b;
}

// ---------------------------------------------------------------------------
// Return the larger of two doubles.
// ---------------------------------------------------------------------------
double fmax(
    const double a,  // first value
    const double b   // second value
  )
{
  return a > b ? a : b;
}

// ---------------------------------------------------------------------------
// Return precomputed Legendre polynomial values and derivatives at the
// angular bin edges for a given angular bin and multipole.
//
// On first call (or when Ntable.Ntheta or Ntable.random changes), this
// function allocates and caches Legendre polynomials P_l(x) and their
// derivatives dP_l(x)/dx evaluated at the cosines of the log-spaced
// angular bin edges [theta_min, theta_max] for all bins and multipoles
// up to Ntable.LMAX. Subsequent calls with unchanged parameters return
// cached values without recomputation.
//
// The angular bins are log-spaced between Ntable.vt[RANGE_MIN] and Ntable.vt[RANGE_MAX].
// The range is not a cache key here: init_binning_real_space
// (generic_interface.cpp) redraws Ntable.random when Ntheta or the range
// changes, which rebuilds this table and every kernel built from it.
// ---------------------------------------------------------------------------
bin_avg set_bin_average(
    const int i_theta,  // angular bin index, must be in [0, Ntable.Ntheta)
    const int j_L       // multipole index, must be in [0, Ntable.LMAX]
  )
{
  static double*** P  = NULL;
  static double** xminmax = NULL;
  static int ntheta = 0;
  static uint64_t cache [MAX_SIZE_ARRAYS];

  if (Ntable.Ntheta == 0) {
    log_fatal("Ntable.Ntheta not initialized"); exit(EXIT_FAILURE);
  }
  if (P == NULL || (ntheta != Ntable.Ntheta) || fdiff2(cache[0], Ntable.random))
  {
    if (P != NULL) {
      free(P);
    }
    if (xminmax != NULL) {
      free(xminmax);
    }

    // Legendre computes l=0,...,lmax (inclusive)
    P  = (double***) malloc3d(4, Ntable.Ntheta, Ntable.LMAX+1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    double** dPmin = P[2]; double** dPmax = P[3];

    xminmax = (double**) malloc2d(2, Ntable.Ntheta);

    const double logdt = (log(Ntable.vt[RANGE_MAX])-log(Ntable.vt[RANGE_MIN]))/ Ntable.Ntheta;
    for(int i=0; i<Ntable.Ntheta ; i++) {
      xminmax[0][i] = cos(exp(log(Ntable.vt[RANGE_MIN]) + (i + 0.)*logdt));
      xminmax[1][i] = cos(exp(log(Ntable.vt[RANGE_MIN]) + (i + 1.)*logdt));
    }

    #pragma omp parallel for
    for (int i=0; i<Ntable.Ntheta; i++) {
      if (fabs(xminmax[0][i]) > 1) {
        log_fatal("logical error: Legendre argument xmin = %.3e>1", xminmax[0][i]);
        exit(EXIT_FAILURE);
      }
      if (fabs(xminmax[1][i]) > 1) {
        log_fatal("logical error: Legendre argument xmax = %.3e>1", xminmax[1][i]);
        exit(EXIT_FAILURE);
      }
      
      int status = 
      gsl_sf_legendre_Pl_deriv_array(Ntable.LMAX, xminmax[0][i], Pmin[i], dPmin[i]);
      if (status) {
        log_fatal(gsl_strerror(status)); exit(EXIT_FAILURE);
      }
      status = 
      gsl_sf_legendre_Pl_deriv_array(Ntable.LMAX, xminmax[1][i], Pmax[i], dPmax[i]);
      if (status) {
        log_fatal(gsl_strerror(status)); exit(EXIT_FAILURE);
      } 
    }
    ntheta = Ntable.Ntheta;
    cache[0] = Ntable.random;
  }
  if (!(i_theta < Ntable.Ntheta)) {
    log_fatal("bad i_theta index");
    exit(1);
  }
  if (j_L > Ntable.LMAX) {
    log_fatal("bad j_L index");
    exit(1);
  }
  bin_avg r;
  r.xmin = xminmax[0][i_theta];
  r.xmax = xminmax[1][i_theta];
  r.Pmin = P[0][i_theta][j_L];
  r.Pmax = P[1][i_theta][j_L];
  r.dPmin = P[2][i_theta][j_L];
  r.dPmax = P[3][i_theta][j_L];
  return r;
}

// ---------------------------------------------------------------------------
// Perform 1D linear interpolation on a uniformly spaced grid.
//
// Values outside the grid domain are handled by constant extrapolation
// (clamped to the nearest boundary value).
// ---------------------------------------------------------------------------
double interpol1d(
    const double* const f,  // data array of length n
    const int n,            // number of grid points
    const double a,         // grid lower bound (x value of f[0])
    const double b,         // grid upper bound (unused; kept for API symmetry)
    const double dx,        // uniform grid spacing
    const double x          // query point at which to interpolate
  )
{
  double ans;
  if (x < a) {  
    ans = f[0]; // constant extrapolation
  }
  else {
    const double r = (x - a) / dx;
    const int i = (int) floor(r);
    if (i + 1 >= n) {
      ans = f[n-1]; // constant extrapolation
    }
    else {
      ans = (r - i) * (f[i + 1] - f[i]) + f[i];
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// Natural cubic spline coefficients on a uniform grid.
//
// DERIVATION:
//   A cubic spline S_i(x) = y_i + b_i·δ + c_i·δ^2 + d_i·δ^3 on each
//   interval [x_i, x_{i+1}] (where δ = x − x_i) must satisfy:
//     (1) interpolation:  S_i(x_i) = y_i,  S_i(x_{i+1}) = y_{i+1}
//     (2) C1 continuity:  S_i'(x_{i+1}) = S_{i+1}'(x_{i+1})
//     (3) C2 continuity:  S_i''(x_{i+1}) = S_{i+1}''(x_{i+1})
//
//   Conditions (1) and (3) give b_i and d_i in terms of the c's; then
//   condition (2) yields a tridiagonal system for the c_i coefficients
//   (second derivatives / 2). For general spacing h_i = x_{i+1} − x_i:
//
//     h_{i-1} c_{i-1} + 2(h_{i-1} + h_i) c_i + h_i c_{i+1}
//       = 3 [(y_{i+1} − y_i)/h_i − (y_i − y_{i-1})/h_{i-1}]
//
//   For a uniform grid (h_i = dx for all i), this simplifies to:
//
//     dx · c_{i-1} + 4·dx · c_i + dx · c_{i+1} = (3/dx)(y_{i-1} − 2y_i + y_{i+1})
//
//   Dividing through by dx gives the symmetric tridiagonal system:
//
//     [1  4  1] [c_1, ..., c_{n-2}]^T = (3/dx^2) [y_0−2y_1+y_2, ..., y_{n-3}−2y_{n-2}+y_{n-1}]^T
//
//   with natural boundary conditions c_0 = c_{n-1} = 0.
//
// ALGORITHM:
//   Thomas algorithm (forward elimination + back substitution) for
//   symmetric tridiagonal systems. Subdiagonal = superdiagonal = 1,
//   diagonal = 4. The multiplier m_i = 1/(4 − m_{i-1}) converges
//   quickly to 1/(4 − 1/(4 − ...)) ≈ 0.268 (the continued fraction).
//
//   Forward sweep:  c_i = (rhs_i − c_{i-1}) · m_i
//   Back substitution:  c_i -= m_i · c_{i+1}
//
//   Cost: O(n) time, O(n) scratch space. Called once per cache rebuild.
//
// PARAMETERS:
//   y  — function values on the uniform grid (length n)
//   n  — number of grid points
//   dx — uniform grid spacing
//   c  — output: spline coefficients (length n), with c[0] = c[n-1] = 0
// ---------------------------------------------------------------------------
void spline_coeffs_uniform(
    const double* restrict y,
    const int n,
    const double dx,
    double* restrict c
  )
{
  // Thomas algorithm on the system derived in the header: every
  // interior row reads c_{i-1} + 4 c_i + c_{i+1} = rhs_i with
  // rhs_i = (3/dx^2)(y_{i-1} - 2 y_i + y_{i+1}), and the natural
  // boundaries pin c_0 = c_{n-1} = 0.
  double* scratch = (double*) malloc(n * sizeof(double));
  const double inv_dx2 = 3.0 / (dx * dx); // the right side's scale

  c[0] = 0.0;       // natural boundary: S'' = 0 at the first node
  scratch[0] = 0.0; // row 1 has no subdiagonal term to eliminate

  // Forward elimination. After this loop, scratch[i] holds the
  // multiplier m_i = 1/(4 - m_{i-1}) (the off-diagonals are 1, so
  // m_{i-1} is also the eliminated subdiagonal's weight), and c[i]
  // holds the partially solved value (rhs_i - c_{i-1}) m_i.
  for (int i = 1; i < n - 1; i++) {
    // second difference of y: the discrete curvature driving S''
    const double rhs = inv_dx2 * (y[i-1] - 2.0 * y[i] + y[i+1]);
    const double m = 1.0 / (4.0 - scratch[i-1]);
    c[i] = (rhs - c[i-1]) * m;
    scratch[i] = m;
  }
  c[n-1] = 0.0; // natural boundary: S'' = 0 at the last node

  // Back substitution: remove each row's superdiagonal term (weight
  // m_i after elimination), last interior row first.
  for (int i = n - 2; i > 0; i--) {
    c[i] -= scratch[i] * c[i+1];
  }

  free(scratch);
}

// ---------------------------------------------------------------------------
// Natural bicubic upsampling between two uniform 2D grids.
//
// The 2D version of the 1D strategy (spline_coeffs_uniform + the
// direct-index Horner evaluation): a coarse table zc, exact at its
// nxc x nyc nodes, fills a dense table zf at nxf x nyf nodes. Both
// grids are uniform along each axis and share their endpoints - that
// is the contract that makes every interval lookup pure arithmetic
// (one multiply + one cast, no search).
//
// A tensor-product bicubic spline separates into two 1D passes:
//
//   pass 1 (along y): each coarse row -> 1D natural cubic spline
//     -> evaluated at the nyf fine columns -> tmp[nxc][nyf]
//   pass 2 (along x): each fine column of tmp -> 1D natural cubic
//     spline -> evaluated at the nxf fine rows -> zf[nxf][nyf]
//
// The passes commute in exact arithmetic (both orders give the unique
// tensor-product interpolant), so the order is a convention. Each 1D
// piece is the house machinery: c_q = S''(x_q)/2 from the [1 4 1]
// tridiagonal system (derived and solved by forward elimination +
// back substitution in spline_coeffs_uniform's header), then the cubic
//
//   S(x_q + dx) = y_q + b dx + c_q dx^2 + d dx^3
//     with d = (c_{q+1} - c_q) / (3 h)
//     and  b = (y_{q+1} - y_q)/h - h (c_{q+1} + 2 c_q)/3
//
// in Horner form, b and d computed once per coarse interval. The fine
// spacings follow from the shared endpoints (dxf = dxc (nxc-1)/(nxf-1))
// and the interval index is clamped onto the last interval against a
// 1-ulp overshoot of the shared top endpoint, as in the 1D consumers.
//
// tmp is row-major [x][y], so the x direction is strided in memory:
// pass 2 runs row-wise, every step one contiguous loop over the nyf
// fine columns (SIMDe, 4 doubles per operation, plus a scalar tail).
//
// Cost: O(nxc (nyc + nyf) + nyf (nxc + nxf)) time; O(nxc nyf) scratch
// as one 4 x nxc x nyf block plus O(nxc + nyc + nyf) small arrays.
// Called once per cache rebuild.
//
// PARAMETERS:
//   zc       - coarse table [nxc][nyc] (malloc2d layout)
//   nxc, nyc - coarse node counts (>= 4 each)
//   dxc, dyc - coarse grid spacings along x and y
//   zf       - output fine table [nxf][nyf] (malloc2d layout)
//   nxf, nyf - fine node counts
// ---------------------------------------------------------------------------
void spline2d_upsample_uniform(
    double** zc,
    const int nxc,
    const int nyc,
    const double dxc,
    const double dyc,
    double** zf,
    const int nxf,
    const int nyf
  )
{
  // shared endpoints fix the fine spacings: (nxf - 1) dxf = (nxc - 1) dxc
  // and likewise along y
  const double dxf = dxc * ((double) (nxc - 1)) / ((double) (nxf - 1));
  const double dyf = dyc * ((double) (nyc - 1)) / ((double) (nyf - 1));

  // four [nxc][nyf] workspaces in one block: tmp (exact in x, fine in
  // y) and the three pass-2 coefficient tables cx, bx, dx3 (below)
  double*** w = (double***) malloc3d(4, nxc, nyf);
  double** tmp = w[0];
  const int ncmax = (nxc > nyc) ? nxc : nyc;
  double* cbuf = (double*) malloc(ncmax * sizeof(double)); // 1D spline c

  // pass 1 (along y): each coarse row onto the fine columns. Fine
  // column j has the same interval q and offset del in every row
  // (qy, ey), and b, d are per coarse interval (by, dy), so no
  // division runs per fine node.
  int* qy = (int*) malloc(nyf * sizeof(int));
  double* ey = (double*) malloc(nyf * sizeof(double));
  for (int j=0; j<nyf; j++) {
    // position of fine column j in coarse spacings: uniform grids with
    // shared endpoints, so the interval index is a cast - no search
    const double r = (double) j * dyf / dyc;
    int q = (int) r;      // left node of the spline interval [q, q+1]
    if (q > nyc - 2) { // shared endpoint (up to 1 ulp overshoot)
      q = nyc - 2;
    }
    qy[j] = q;
    ey[j] = (r - q) * dyc; // offset inside the interval
  }
  double* by = (double*) malloc((nyc - 1) * sizeof(double));
  double* dy = (double*) malloc((nyc - 1) * sizeof(double));
  for (int i=0; i<nxc; i++) {
    // natural cubic spline through this row: cbuf[q] = S''(y_q)/2
    spline_coeffs_uniform(zc[i], nyc, dyc, cbuf);
    const double* restrict y = zc[i];
    const double* restrict cc = cbuf;
    // cubic S(y_q + del) = y_q + b del + c_q del^2 + d del^3 with
    //   S(y_{q+1}) = y_{q+1} (interpolate the right node)  -> b
    //   S'' linear from 2 c_q to 2 c_{q+1}                 -> d
    for (int q=0; q<nyc-1; q++) {
      by[q] = (y[q+1] - y[q])/dyc - dyc*(cc[q+1] + 2.0*cc[q])/3.0;
      dy[q] = (cc[q+1] - cc[q])/(3.0*dyc);
    }
    double* restrict out = tmp[i];
    for (int j=0; j<nyf; j++) {
      const int q = qy[j];
      const double del = ey[j];
      out[j] = y[q] + del*(by[q] + del*(cc[q] + del*dy[q])); // Horner form
    }
  }
  free(dy);
  free(by);
  free(ey);
  free(qy);

  // pass 2 (along x), row-wise. Every column of tmp has the same
  // [1 4 1] system along x (spline_coeffs_uniform's header), so the
  // multipliers m_q = 1/(4 - m_{q-1}) are shared (mq) and each solve
  // step runs over whole rows, with the per-element arithmetic of a
  // per-column 1D solve. cx[q][j] = S''/2 at coarse x node q of
  // column j; bx, dx3 = per-interval b and d (rows 0 .. nxc-2 used).
  double** cx = w[1];
  double** bx = w[2];
  double** dx3 = w[3];
  double* mq = (double*) malloc(nxc * sizeof(double));
  const double inv_dx2 = 3.0 / (dxc * dxc); // the right side's scale
  mq[0] = 0.0; // row 1 has no subdiagonal term to eliminate
  // natural boundaries: S'' = 0 at the first and last x node
  for (int j=0; j<nyf; j++) {
    cx[0][j] = 0.0;
    cx[nxc-1][j] = 0.0;
  }
  // forward elimination, row by row: c_q = (rhs_q - c_{q-1}) m_q with
  // rhs_q = (3/dxc^2)(tmp_{q-1} - 2 tmp_q + tmp_{q+1}) in each column
  for (int q=1; q<nxc-1; q++) {
    const double m = 1.0 / (4.0 - mq[q-1]);
    mq[q] = m;
    const double* restrict t0 = tmp[q-1];
    const double* restrict t1 = tmp[q];
    const double* restrict t2 = tmp[q+1];
    const double* restrict c0 = cx[q-1];
    double* restrict c1 = cx[q];
    int j = 0;
    // SIMDe body, 4 columns per operation (AVX2 on x86, two NEON
    // registers on Apple Silicon); the scalar loop after it finishes
    // the last nyf % 4 columns. Every row loop below has the same shape:
    // lane l holds column j + l, so the four lanes are four independent
    // column splines that share m and never mix.
    //
    // scalar: for (j = 0; j < nyf; j++) {
    //           const double rhs = inv_dx2*(t0[j] - 2.0*t1[j] + t2[j]);
    //           c1[j] = (rhs - c0[j])*m;
    //         }
    //
    // set1 copies one scalar into all four lanes: inv_dx2 = 3/dxc^2, the
    // constant 2.0 and this row's multiplier m = m_q
    const simde__m256d vinv = simde_mm256_set1_pd(inv_dx2);
    const simde__m256d vtwo = simde_mm256_set1_pd(2.0);
    const simde__m256d vm = simde_mm256_set1_pd(m);
    for (; j <= nyf - 4; j += 4) {
      // s = (t0[j] - 2.0*t1[j]) + t2[j] for columns j..j+3 (lanes 0..3),
      // the second difference at coarse x node q. loadu reads
      // tmp[q-1][j..j+3] (t0), tmp[q][j..j+3] (t1) and tmp[q+1][j..j+3]
      // (t2) without needing 32-byte alignment; mul forms 2.0*t1 (exact),
      // sub gives t0 - 2.0*t1, and add includes t2
      const simde__m256d s = simde_mm256_add_pd(
        simde_mm256_sub_pd(simde_mm256_loadu_pd(t0 + j),
          simde_mm256_mul_pd(vtwo, simde_mm256_loadu_pd(t1 + j))),
        simde_mm256_loadu_pd(t2 + j));
      // c1[j..j+3] = (inv_dx2*s - c0[j..j+3])*m: mul forms the tail's rhs,
      // loadu reads cx[q-1][j..j+3] (c0), and sub and the second mul
      // finish the step, the same three operations as the tail's two
      // statements; storeu writes the four lanes to cx[q][j..j+3]
      simde_mm256_storeu_pd(c1 + j, simde_mm256_mul_pd(simde_mm256_sub_pd(
        simde_mm256_mul_pd(vinv, s), simde_mm256_loadu_pd(c0 + j)), vm));
    }
    for (; j<nyf; j++) {
      const double rhs = inv_dx2 * (t0[j] - 2.0 * t1[j] + t2[j]);
      c1[j] = (rhs - c0[j]) * m;
    }
  }
  // back substitution, last interior row first: c_q -= m_q c_{q+1} in
  // every column (lane l holds column j + l; lanes never mix).
  //
  // scalar: for (j = 0; j < nyf; j++) { c1[j] -= m*c2[j]; }
  //
  // Rounding: on x86 with FMA, simde_mm256_fnmadd_pd is the native fused
  // instruction, and the compiler fuses the scalar tail's c1[j] - m*c2[j]
  // too (contraction is on by default). On arm64 there is no NEON branch
  // at 256 bits, but unlike the four-lane fmadd (a mul, then an add: two
  // roundings) SIMDe writes fnmadd as the per-lane expression
  // -(a*b) + c, the tail's own expression, so lanes and tail round alike
  // whether or not the compiler contracts them (the arm64 clang build
  // fuses both). Vector and tail columns agree bitwise here; the Horner
  // rows below do not on arm64.
  for (int q=nxc-2; q>0; q--) {
    const double m = mq[q];
    const double* restrict c2 = cx[q+1];
    double* restrict c1 = cx[q];
    int j = 0;
    // set1: this row's multiplier m = m_q in all four lanes
    const simde__m256d vm = simde_mm256_set1_pd(m);
    for (; j <= nyf - 4; j += 4) {
      // c1[j..j+3] = -(m*c2[j..j+3]) + c1[j..j+3], rounded as the tail
      // (see above): loadu reads cx[q+1][j..j+3] (c2) and cx[q][j..j+3]
      // (c1), fnmadd forms -(a*b) + c lane by lane, and storeu writes the
      // result back to cx[q][j..j+3] (the loads complete before the store)
      simde_mm256_storeu_pd(c1 + j, simde_mm256_fnmadd_pd(vm,
        simde_mm256_loadu_pd(c2 + j), simde_mm256_loadu_pd(c1 + j)));
    }
    for (; j<nyf; j++) {
      c1[j] -= m * c2[j];
    }
  }
  // per-interval b and d, one row per coarse x interval (the pass-1
  // formulas, across the columns)
  for (int q=0; q<nxc-1; q++) {
    const double* restrict t0 = tmp[q];
    const double* restrict t1 = tmp[q+1];
    const double* restrict c0 = cx[q];
    const double* restrict c1 = cx[q+1];
    double* restrict bq = bx[q];
    double* restrict dq = dx3[q];
    int j = 0;
    // scalar: for (j = 0; j < nyf; j++) {
    //           bq[j] = (t1[j] - t0[j])/dxc - dxc*(c1[j] + 2.0*c0[j])/3.0;
    //           dq[j] = (c1[j] - c0[j])/(3.0*dxc);
    //         }
    // Lane l holds column j + l; lanes never mix.
    //
    // set1 copies one scalar into all four lanes: dxc, 2.0, 3.0 and the
    // tail's product 3.0*dxc (computed once, the same double)
    const simde__m256d vdx = simde_mm256_set1_pd(dxc);
    const simde__m256d vtwo = simde_mm256_set1_pd(2.0);
    const simde__m256d vthree = simde_mm256_set1_pd(3.0);
    const simde__m256d v3dx = simde_mm256_set1_pd(3.0*dxc);
    for (; j <= nyf - 4; j += 4) {
      // loadu reads four consecutive columns into lanes 0..3, no 32-byte
      // alignment needed: a0 = tmp[q][j..j+3], a1 = tmp[q+1][j..j+3],
      // k0 = cx[q][j..j+3], k1 = cx[q+1][j..j+3]
      const simde__m256d a0 = simde_mm256_loadu_pd(t0 + j);
      const simde__m256d a1 = simde_mm256_loadu_pd(t1 + j);
      const simde__m256d k0 = simde_mm256_loadu_pd(c0 + j);
      const simde__m256d k1 = simde_mm256_loadu_pd(c1 + j);
      // bq[j..j+3] = (a1 - a0)/dxc - (dxc*(k1 + 2.0*k0))/3.0, the tail's
      // order (C reads dxc*(..)/3.0 as (dxc*(..))/3.0): sub and div form
      // (a1 - a0)/dxc; mul gives 2.0*k0 (exact, so k1 + 2.0*k0 is the
      // same double fused or not), add k1, mul by dxc and div by 3.0 form
      // the curvature term; the outer sub combines the two, and storeu
      // writes bx[q][j..j+3]
      simde_mm256_storeu_pd(bq + j, simde_mm256_sub_pd(
        simde_mm256_div_pd(simde_mm256_sub_pd(a1, a0), vdx),
        simde_mm256_div_pd(simde_mm256_mul_pd(vdx, simde_mm256_add_pd(k1,
          simde_mm256_mul_pd(vtwo, k0))), vthree)));
      // dq[j..j+3] = (k1 - k0)/(3.0*dxc): sub, then div by v3dx; storeu
      // writes dx3[q][j..j+3]
      simde_mm256_storeu_pd(dq + j, simde_mm256_div_pd(
        simde_mm256_sub_pd(k1, k0), v3dx));
    }
    for (; j<nyf; j++) {
      bq[j] = (t1[j] - t0[j])/dxc - dxc*(c1[j] + 2.0*c0[j])/3.0;
      dq[j] = (c1[j] - c0[j])/(3.0*dxc);
    }
  }
  // each fine row: one interval q and offset del, then a contiguous
  // Horner evaluation over the columns
  for (int i=0; i<nxf; i++) {
    const double r = (double) i * dxf / dxc;
    int q = (int) r;      // left node of the spline interval [q, q+1]
    if (q > nxc - 2) { // shared endpoint (up to 1 ulp overshoot)
      q = nxc - 2;
    }
    const double del = (r - q) * dxc; // offset inside the interval
    const double* restrict t0 = tmp[q];
    const double* restrict c0 = cx[q];
    const double* restrict bq = bx[q];
    const double* restrict dq = dx3[q];
    double* restrict out = zf[i];
    int j = 0;
    // scalar: for (j = 0; j < nyf; j++) {
    //           out[j] = t0[j] + del*(bq[j] + del*(c0[j] + del*dq[j]));
    //         }
    // Lane l holds column j + l; lanes never mix.
    //
    // Rounding: the compiler contracts the scalar Horner form into three
    // fused multiply-adds. simde_mm256_fmadd_pd matches that on x86 with
    // FMA (one instruction, one rounding), but on arm64 SIMDe writes the
    // four-lane fmadd as simde_mm256_add_pd(simde_mm256_mul_pd(a, b), c),
    // two roundings (the two-lane simde_mm_fmadd_pd would be fused; see
    // limber_fmadd4 in cosmo2D_cluster.c). On arm64 the first 4*(nyf/4)
    // columns of zf can therefore differ from the last nyf % 4 in the
    // last bit.
    //
    // set1: the offset del in all four lanes
    const simde__m256d vdel = simde_mm256_set1_pd(del);
    for (; j <= nyf - 4; j += 4) {
      // h = del*dq[j..j+3] + c0[j..j+3]: loadu reads dx3[q][j..j+3] and
      // cx[q][j..j+3], and fmadd computes a*b + c lane by lane
      const simde__m256d h = simde_mm256_fmadd_pd(vdel,
        simde_mm256_loadu_pd(dq + j), simde_mm256_loadu_pd(c0 + j));
      // g = del*h + bq[j..j+3]: loadu reads bx[q][j..j+3], then fmadd
      const simde__m256d g = simde_mm256_fmadd_pd(vdel, h,
        simde_mm256_loadu_pd(bq + j));
      // out[j..j+3] = del*g + t0[j..j+3]: loadu reads tmp[q][j..j+3],
      // fmadd completes the Horner form, and storeu writes zf[i][j..j+3]
      simde_mm256_storeu_pd(out + j, simde_mm256_fmadd_pd(vdel, g,
        simde_mm256_loadu_pd(t0 + j)));
    }
    for (; j<nyf; j++) {
      out[j] = t0[j] + del*(bq[j] + del*(c0[j] + del*dq[j])); // Horner form
    }
  }
  free(mq);
  free(cbuf);
  free(w);
}



// ---------------------------------------------------------------------------
// Count the lines of a text file.
//
// Counts the newline characters, empty lines included, and adds one for
// a last line without a trailing newline, provided the file contains at
// least one newline (a single line without a newline counts as 0).
// Terminates the program if the file cannot be opened.
// ---------------------------------------------------------------------------
int line_count(
    char* filename  // path to the text file
  )
{  
  FILE* ein = fopen(filename, "r");
  if (ein == NULL) 
  {
    log_fatal("File not open (%s)", filename);
    exit(1);
  }
  
  int ch = 0; 
  int prev = 0; 
  int nlines = 0;

  do 
  {
    prev = ch;
    
    ch = fgetc(ein);
    
    if (ch == '\n')
    {
      nlines++;
    }
  } while (ch != EOF);
  
  fclose(ein);
  
  // last line might not end with "\n". 
  // However, if previous character does, then the last line is empty
  if (ch != '\n' && prev != '\n' && nlines != 0) 
  {
    nlines++;
  }
  return nlines;
}

// ---------------------------------------------------------------------------
// Perform 2D bilinear interpolation on a uniformly spaced grid.
//
// Out-of-range behavior differs by axis:
//   - x out of [ax, bx]: returns 0
//   - y below ay: the edge value f(x, ay), interpolated in x, plus
//     (y - ay): a straight line of unit slope in y, not the table's own
//     slope at the edge
//   - y above by: likewise f(x, by) + (y - by)
//
// Inside [ay, by], boundary cases where the query falls on the last grid
// index in either dimension are handled by dropping the out-of-bounds
// terms from the bilinear formula. The two y extrapolations always read
// row i+1, so they need x < bx.
// ---------------------------------------------------------------------------
double interpol2d(
    double** f,   // 2D data array of shape [nx][ny]
    int nx,       // number of grid points along x
    double ax,    // x-axis lower bound
    double bx,    // x-axis upper bound
    double dx,    // uniform x grid spacing
    double x,     // x query point
    int ny,       // number of grid points along y
    double ay,    // y-axis lower bound
    double by,    // y-axis upper bound
    double dy,    // uniform y grid spacing
    double y      // y query point
  )
{
  double t, dt, s, ds;
  int i, j;
  
  if (x < ax) 
    return 0.;
  if (x > bx) 
    return 0.;

  t = (x - ax) / dx;
  i = (int)(floor(t));
  dt = t - i;
  
  if (y < ay) 
  {
    return ((1. - dt) * f[i][0] + dt * f[i + 1][0]) + (y - ay);
  } 
  else if (y > by) 
  {
    return ((1. - dt) * f[i][ny - 1] + dt * f[i + 1][ny - 1]) + (y - by);
  }
  s = (y - ay) / dy;
  j = (int)(floor(s));
  ds = s - j;
  
  if ((i + 1 == nx) && (j + 1 == ny)) 
  {
    return (1. - dt) * (1. - ds) * f[i][j];
  }
  if (i + 1 == nx) 
  {
    return (1. - dt) * (1. - ds) * f[i][j] + (1. - dt) * ds * f[i][j + 1];
  }
  if (j + 1 == ny) 
  {
    return (1. - dt) * (1. - ds) * f[i][j] + dt * (1. - ds) * f[i + 1][j];
  }
  
  return (1. - dt) * (1. - ds) * f[i][j] + (1. - dt) * ds * f[i][j + 1] +
         dt * (1. - ds) * f[i + 1][j] + dt * ds * f[i + 1][j + 1];
}


// ---------------------------------------------------------------------------
// Safe zeroing functions for padded multi-dimensional arrays.
//
// PROBLEM:
//   malloc2d/3d/4d pad the innermost dimension to 64-byte boundaries
//   for SIMD alignment. For example, malloc3d(11, 100, 100) allocates
//   rows of 104 doubles (nzp = 104), not 100. The data layout is:
//
//     row (0,0): [100 logical doubles | 4 padding doubles]
//     row (0,1): [100 logical doubles | 4 padding doubles]
//     ...
//
//   The naive idiom
//     memset(arr[0][0], 0, nx*ny*nz*sizeof(double))
//   treats the data as a flat contiguous block of nx*ny*nz doubles.
//   But the actual stride is nzp, not nz. The flat memset zeros
//   nx*ny*nz doubles starting from arr[0][0], which undershoots the
//   true allocation (nx*ny*nzp doubles). The result:
//     - early rows: logical data and padding zeroed (the flat memset runs
//       straight through the padding between rows)
//     - late rows: logical data left uninitialized (dangerous)
//
//   Example: malloc3d(11, 100, 100), nzp = 104, rows 0-1099
//     memset zeros:  11*100*100 = 110,000 doubles
//     actual data:   11*100*104 = 114,400 doubles
//     last 4,400 doubles uninitialized: rows 1058-1099 entirely, and row
//     1057 from its value 72 on
//
//
// SOLUTION:
//   Zero through the pointer indirection, one innermost row at a time.
//   Each memset follows the actual row pointer (which accounts for
//   padding) and zeros exactly the logical element count, skipping the
//   padding (left uninitialized, so it must not be read). Safe for any
//   dimension size, regardless of 64-byte alignment.
// ---------------------------------------------------------------------------
void zero2d(double** a, const int nx, const int ny)
{
  for (int i = 0; i < nx; i++) {
    memset(a[i], 0, ny * sizeof(double));
  }
}

void zero3d(double*** a, const int nx, const int ny, const int nz)
{
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      memset(a[i][j], 0, nz * sizeof(double));
    }
  }
}

void zero4d(double**** a, const int nx, const int ny, const int nz,
            const int nw)
{
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        memset(a[i][j][k], 0, nw * sizeof(double));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Compute the Hankel-transform kernel in Fourier space.
//
// Evaluates the FFTLog kernel of Hamilton (2000, Appendix B) at the
// complex argument q + ix and stores it in the caller-provided
// fftw_complex:
//
//   U_mu(q + ix) = int_0^inf t^(q + ix) J_mu(t) dt
//                = 2^(q + ix) Gamma((1+mu+q)/2 + ix/2)
//                             / Gamma((1+mu-q)/2 - ix/2),
//
// i.e. the Gamma ratio times 2^q and the phase 2^(ix) = exp(i x ln 2).
// The factor (k0 r0)^(-ix) of Hamilton's u_m is not included.
//
// The Bessel order is converted with (int)(mu + 0.1): an integer order
// stored slightly below its value maps to that integer, while a
// fractional order is truncated (1.5 -> 1).
// ---------------------------------------------------------------------------
void hankel_kernel_FT(
    double x,           // Fourier-space frequency variable
    fftw_complex* res,  // output: computed kernel value (real, imaginary)
    double* arg,        // parameter array: arg[0] = bias q, arg[1] = Bessel order mu
    int argc            // length of arg (unused, kept for callback signature)
  )
{
  fftw_complex a1, a2, g1, g2;

  // arguments for complex gamma
  const double q = arg[0];
  const int mu = (int)(arg[1] + 0.1);
  a1[0] = 0.5 * (1.0 + mu + q);
  a2[0] = 0.5 * (1.0 + mu - q);
  a1[1] = 0.5 * x;
  a2[1] = -a1[1];

  cdgamma(a1, &g1);
  cdgamma(a2, &g2);

  const double xln2 = x * M_LN2;
  const double si = sin(xln2);
  const double co = cos(xln2);
  const double d1 = g1[0] * g2[0] + g1[1] * g2[1]; /* Re */
  const double d2 = g1[1] * g2[0] - g1[0] * g2[1]; /* Im */
  const double mod = g2[0] * g2[0] + g2[1] * g2[1];
  const double pref = exp(M_LN2 * q) / mod;

  (*res)[0] = pref * (co * d1 - si * d2);
  (*res)[1] = pref * (si * d1 + co * d2);
}

// ---------------------------------------------------------------------------
// Evaluate the complex gamma function Gamma(z) using a Lanczos-type
// rational approximation.
//
// For Re(z) >= 0 the approximation is applied directly. For Re(z) < 0
// the reflection formula
//   Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
// is used to map into the right half-plane first.
//
// The approximation coefficients are tuned for double precision and
// the method is based on the Lanczos decomposition with g ~ 7.
// ---------------------------------------------------------------------------
void cdgamma(
    fftw_complex x,     // input: complex argument z = (Re, Im)
    fftw_complex* res   // output: Gamma(z) = (Re, Im)
  )
{
  double xr, xi, wr, wi, ur, ui, vr, vi, yr, yi, t;

  xr = (double) x[0];
  xi = (double) x[1];

  if (xr < 0) 
  {
    wr = 1 - xr;
    wi = -xi;
  } else 
  {
    wr = xr;
    wi = xi;
  }

  ur = wr + 6.00009857740312429;
  vr = ur * (wr + 4.99999857982434025) - wi * wi;
  vi = wi * (wr + 4.99999857982434025) + ur * wi;
  yr = ur * 13.2280130755055088 + vr * 66.2756400966213521 +
       0.293729529320536228;
  yi = wi * 13.2280130755055088 + vi * 66.2756400966213521;
  ur = vr * (wr + 4.00000003016801681) - vi * wi;
  ui = vi * (wr + 4.00000003016801681) + vr * wi;
  vr = ur * (wr + 2.99999999944915534) - ui * wi;
  vi = ui * (wr + 2.99999999944915534) + ur * wi;
  yr += ur * 91.1395751189899762 + vr * 47.3821439163096063;
  yi += ui * 91.1395751189899762 + vi * 47.3821439163096063;
  ur = vr * (wr + 2.00000000000603851) - vi * wi;
  ui = vi * (wr + 2.00000000000603851) + vr * wi;
  vr = ur * (wr + 0.999999999999975753) - ui * wi;
  vi = ui * (wr + 0.999999999999975753) + ur * wi;
  yr += ur * 10.5400280458730808 + vr;
  yi += ui * 10.5400280458730808 + vi;
  ur = vr * wr - vi * wi;
  ui = vi * wr + vr * wi;
  t = ur * ur + ui * ui;
  vr = yr * ur + yi * ui + t * 0.0327673720261526849;
  vi = yi * ur - yr * ui;
  yr = wr + 7.31790632447016203;
  ur = log(yr * yr + wi * wi) * 0.5 - 1;
  ui = atan2(wi, yr);
  yr = exp(ur * (wr - 0.5) - ui * wi - 3.48064577727581257) / t;
  yi = ui * (wr - 0.5) + ur * wi;
  ur = yr * cos(yi);
  ui = yr * sin(yi);
  yr = ur * vr - ui * vi;
  yi = ui * vr + ur * vi;
  if (xr < 0) {
    wr = xr * 3.14159265358979324;
    wi = exp(xi * 3.14159265358979324);
    vi = 1 / wi;
    ur = (vi + wi) * sin(wr);
    ui = (vi - wi) * cos(wr);
    vr = ur * yr + ui * yi;
    vi = ui * yr - ur * yi;
    ur = 6.2831853071795862 / (vr * vr + vi * vi);
    yr = ur * vr;
    yi = ur * vi;
  }

  (*res)[0] = yr;
  (*res)[1] = yi;
}

// ---------------------------------------------------------------------------
// Compute the 3D Hankel-transform kernel in Fourier space.
//
// Identical to hankel_kernel_FT() except that the Bessel order mu is
// treated as a continuous real value rather than being converted to an
// integer. This is appropriate for 3D spherical Bessel
// transforms where half-integer orders arise naturally.
//
// @see hankel_kernel_FT
// ---------------------------------------------------------------------------
void hankel_kernel_FT_3D(
    double x,           // Fourier-space frequency variable
    fftw_complex* res,  // output: computed kernel value (real, imaginary)
    double* arg,        // parameter array: arg[0] = bias q, arg[1] = Bessel order mu
    int argc            // length of arg (unused, kept for callback signature)
  )
{
  fftw_complex a1, a2, g1, g2;
  double           mu;
  double        mod, xln2, si, co, d1, d2, pref, q;
  q = arg[0];
  mu = arg[1];

  /* arguments for complex gamma */
  a1[0] = 0.5*(1.0+mu+q);
  a2[0] = 0.5*(1.0+mu-q);
  a1[1] = 0.5*x; a2[1]=-a1[1];
  cdgamma(a1,&g1);
  cdgamma(a2,&g2);
  xln2 = x*M_LN2;
  si   = sin(xln2);
  co   = cos(xln2);
  d1   = g1[0]*g2[0]+g1[1]*g2[1]; /* Re */
  d2   = g1[1]*g2[0]-g1[0]*g2[1]; /* Im */
  mod  = g2[0]*g2[0]+g2[1]*g2[1];
  pref = exp(M_LN2*q)/mod;

  (*res)[0] = pref*(co*d1-si*d2);
  (*res)[1] = pref*(si*d1+co*d2);
}
