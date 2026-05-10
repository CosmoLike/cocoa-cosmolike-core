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

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM

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
#endif

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// SIMD FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#ifndef COSMO2D_NOT_USE_SIMD
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
//   result = [ 6.0 | 8.0 | 10.0 | 12.0 ]   (one instruction!)
// -----------------------------------------------------------------------------
double simd_horizontal_sum(simde__m256d four_lanes)
{ // Takes a 4-lane register and sums all 4 values into a single double
  double tmp[4]; // Store the 4 lanes into a regular C array
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
// A scalar tail handles the remaining n % 8 elements. The final reduction
// adds the two vector accumulators, then horizontally sums the 4 lanes
// via 128-bit extract + add + shuffle + add.
// ---------------------------------------------------------------------------
double simd_array_sum(
    const double* restrict a,  // input array, length n (need not be aligned)
    const int n                // number of elements to sum
  )
{
  simde__m256d accum_A = simde_mm256_setzero_pd();
  simde__m256d accum_B = simde_mm256_setzero_pd();
 
  int q = 0;
  for (; q <= n - 8; q += 8) { // Main loop: process 8 doubles per iteration
    accum_A = simde_mm256_add_pd(accum_A, simde_mm256_loadu_pd(a + q));
    accum_B = simde_mm256_add_pd(accum_B, simde_mm256_loadu_pd(a + q + 4));
  }
  double result = simd_horizontal_sum(accum_A) + simd_horizontal_sum(accum_B);
  for (; q < n; q++) { // Scalar tail: remaining 0-7 elements, one at a time
    result += a[q];
  }
  return result;
}
#endif

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
  result = gsl_interp_alloc(gsl_interp_cspline, n);
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
  result = gsl_spline_alloc(gsl_interp_cspline, n);
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
    // with padding, we need to use the byte-level cast 
    // (char*) raw_block + nxp*sizeof(int*) to get the exact padded offset
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
// The total allocation is padded to a 64-byte boundary, suitable for
// SIMD and cache-friendly access patterns. All bytes are set to zero
// before returning.
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
// free() releases both the pointer array and the data block).
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
// The angular bins are log-spaced between Ntable.vtmin and Ntable.vtmax.
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

    const double logdt = (log(Ntable.vtmax)-log(Ntable.vtmin))/ Ntable.Ntheta;
    for(int i=0; i<Ntable.Ntheta ; i++) {
      xminmax[0][i] = cos(exp(log(Ntable.vtmin) + (i + 0.)*logdt));
      xminmax[1][i] = cos(exp(log(Ntable.vtmin) + (i + 1.)*logdt));
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
//     (1) interpolation:  S_i(x_i) = y_i
//     (2) C1 continuity:  S_i'(x_{i+1}) = S_{i+1}'(x_i+1)
//     (3) C2 continuity:  S_i''(x_{i+1}) = S_{i+1}''(x_{i+1})
//
//   Condition (3) yields a tridiagonal system for the c_i coefficients
//   (second derivatives / 2). For general spacing h_i = x_{i+1} − x_i:
//
//     h_{i-1} c_{i-1} + 2(h_{i-1} + h_i) c_i + h_i c_{i+1}
//       = 3 [(y_{i+1} − y_i)/h_i − (y_i − y_{i-1})/h_{i-1}]
//
//   For a UNIFORM grid (h_i = dx for all i), this simplifies to:
//
//     dx · c_{i-1} + 4·dx · c_i + dx · c_{i+1} = (3/dx)(y_{i-1} − 2y_i + y_{i+1})
//
//   Dividing through by dx gives the symmetric tridiagonal system:
//
//     [1  4  1] [c_1, ..., c_{n-2}]^T = (6/dx^2) [y_0−2y_1+y_2, ..., y_{n-3}−2y_{n-2}+y_{n-1}]^T
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
  double* scratch = (double*) malloc(n * sizeof(double));
  const double inv_dx2 = 3.0 / (dx * dx);

  c[0] = 0.0;
  scratch[0] = 0.0;

  for (int i = 1; i < n - 1; i++) {
    const double rhs = inv_dx2 * (y[i-1] - 2.0 * y[i] + y[i+1]);
    const double m = 1.0 / (4.0 - scratch[i-1]);
    c[i] = (rhs - c[i-1]) * m;
    scratch[i] = m;
  }
  c[n-1] = 0.0;

  for (int i = n - 2; i > 0; i--) {
    c[i] -= scratch[i] * c[i+1];
  }

  free(scratch);
}


// ---------------------------------------------------------------------------
// Count the number of non-empty lines in a text file.
//
// Opens the file, scans for newline characters, and accounts for a
// possible missing trailing newline on the last line. Terminates the
// program if the file cannot be opened.
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
//   - y below ay: linearly extrapolates from the y=ay edge
//   - y above by: linearly extrapolates from the y=by edge
//
// Boundary cases where the query falls on the last grid index in either
// dimension are handled by dropping the out-of-bounds terms from the
// bilinear formula.
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
//     - early rows: logical data zeroed, padding left dirty (harmless)
//     - late rows: logical data left UNINITIALIZED (dangerous)
//
//   Example: malloc3d(11, 100, 100), nzp = 104
//     memset zeros:  11*100*100 = 110,000 doubles
//     actual data:   11*100*104 = 114,400 doubles
//     last ~4,400 doubles uninitialized — affects rows 1058-1099
//
//
// SOLUTION:
//   Zero through the pointer indirection, one innermost row at a time.
//   Each memset follows the actual row pointer (which accounts for
//   padding) and zeros exactly the logical element count. Safe for
//   any dimension size, regardless of 64-byte alignment.
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
// Evaluates the ratio of complex gamma functions that appears in the
// analytic Fourier transform of the Hankel kernel r^(q-1) J_mu(kr),
// multiplied by a phase factor from the FFTLog decomposition. The
// result is stored in the caller-provided fftw_complex.
//
// The computation follows the FFTLog formalism (Hamilton 2000), where
// the kernel is expressed as
//   u(x) = 2^q * Gamma((1+mu+q)/2 + ix/2) / Gamma((1+mu-q)/2 - ix/2)
//
// The Bessel order mu is rounded to the nearest integer (via the +0.1
// offset before truncation).
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
// treated as a continuous real value rather than being rounded to the
// nearest integer. This is appropriate for 3D spherical Bessel
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
