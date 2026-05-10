#if !defined(__APPLE__)
#include <malloc.h>
#endif
#include <stdint.h>
#include <fftw3.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix.h>
#include "structs.h"
#ifndef COSMO2D_NOT_USE_SIMD
#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"
#endif
#ifndef __COSMOLIKE_BASICS_H
#define __COSMOLIKE_BASICS_H
#ifdef __cplusplus
extern "C" {
#endif
#define NR_END 1
#define FREE_ARG char *

// ---------------------------------------------------------------------------
// Linear interpolation between two adjacent elements of an array.
//
// Computes arr[b] + dr * (arr[b+1] - arr[b]), where dr is the
// fractional distance in [0, 1) between indices b and b+1.
// ---------------------------------------------------------------------------
#define LERP(arr, b, dr) ((dr)*((arr)[(b)+1] - (arr)[(b)]) + (arr)[(b)])

// ---------------------------------------------------------------------------
// Portable wrapper for simultaneous sine and cosine computation.
//
// On macOS the standard sincos() is unavailable; the Apple-specific
// __sincos() is used instead. Both variants compute sin(x) and cos(x)
// in a single call, storing results through the pointers s and c.
// ---------------------------------------------------------------------------
#ifdef __APPLE__
  #define cosmo_sincos(x, s, c) __sincos((x), (s), (c))
#else
  #define cosmo_sincos(x, s, c) sincos((x), (s), (c))
#endif

// ---------------------------------------------------------------------------
// Portable alias for the restrict qualifier.
//
// C++ does not define restrict as a keyword; __restrict__ is the
// compiler-specific equivalent supported by GCC and Clang.
// ---------------------------------------------------------------------------
#ifdef __cplusplus
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif


#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Detect contiguous segments of a sorted array that are each uniformly
// spaced to within a relative tolerance.
//
// Scans the input array x[0..n-1] and partitions it into up to max_seg
// runs where consecutive spacings differ by no more than rtol relative
// to the first spacing in each run. For each detected segment the
// function records the starting index, length, minimum value, and
// reciprocal spacing (1/dx), enabling fast index lookups via
//   i = (int)((x - xmin) * inv_dx)
// without a full binary search.
//
// Terminates the program if the array cannot be covered by max_seg
// segments (diagnostic output includes name for identification).
//
// @return  Number of uniform segments found (<= max_seg).
// ---------------------------------------------------------------------------
int detect_uniform_segments(
    const double* x,   // sorted input array of length n
    int n,             // number of elements in x
    double rtol,       // relative tolerance for spacing uniformity
    int max_seg,       // maximum number of segments allowed
    int* start,        // output array [max_seg]: starting index of each segment
    int* len,          // output array [max_seg]: number of points in each segment
    double* xmin,      // output array [max_seg]: x value at start of each segment
    double* inv_dx,    // output array [max_seg]: reciprocal spacing of each segment
    const char* name   // descriptive label used in error messages
  );
#endif

// ---------------------------------------------------------------------------
// Precomputed Legendre polynomial data at the edges of a single angular
// bin, returned by set_bin_average().
//
// Stores the cosines of the bin edges (xmin, xmax) together with the
// Legendre polynomial P_l and its derivative dP_l/dx evaluated at each
// edge, for use in the bin-averaged angular power spectrum projection.
// ---------------------------------------------------------------------------
typedef struct
{
  double xmin;   // cos(theta_max) — upper edge cosine (note reversed order)
  double xmax;   // cos(theta_min) — lower edge cosine
  double Pmin;   // P_l(xmin)
  double Pmax;   // P_l(xmax)
  double dPmin;  // dP_l/dx evaluated at xmin
  double dPmax;  // dP_l/dx evaluated at xmax
} bin_avg;

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
gsl_interp* malloc_gsl_interp(const int n);

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
gsl_spline* malloc_gsl_spline(const int n);

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
gsl_integration_glfixed_table* malloc_gslint_glfixed(const int n);

#ifndef COSMO2D_NOT_USE_SIMD
// ---------------------------------------------------------------------------
// Sum all elements of a double array using AVX2 SIMD intrinsics (via
// SIMDe for portability).
//
// Processes four doubles per iteration with 256-bit vector adds,
// falling back to scalar arithmetic for any trailing elements that
// don't fill a full lane. The input array need not be 32-byte aligned.
// ---------------------------------------------------------------------------
double simd_array_sum(
    const double* RESTRICT a,  // input array, length n (need not be aligned)
    const int n                // number of elements to sum
  );

// ---------------------------------------------------------------------------
// Reduce a 256-bit SIMD register containing four packed doubles to a
// single scalar sum.
//
// Performs a horizontal add across all four lanes of the input register
// and returns the result as a plain double.
// ---------------------------------------------------------------------------
double simd_horizontal_sum(
    simde__m256d four_lanes  // 256-bit register holding four doubles to sum
  );
#endif

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
  );

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
  );

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
  );

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
  );

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
  );

// ---------------------------------------------------------------------------
// Allocate a 4D array as a single 64-byte-aligned contiguous block with
// pointer indirection for convenient multi-index access
// (result[i][j][k][l]).
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
  );

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
  );

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
  );

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
  );

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
  );

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
  );

static inline int fdiff2(const uint64_t a, const uint64_t b)
{
  return (a == b) ? 0 : 1;
}

static inline int fdiff(const double a, const double b)
{
  return (fabs(a-b) < 1.0e-13 * fabs(a+b) || fabs(a-b) < 2.0e-38) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Return the smaller of two doubles.
// ---------------------------------------------------------------------------
double fmin(
    const double a,  // first value
    const double b   // second value
  );

// ---------------------------------------------------------------------------
// Return the larger of two doubles.
// ---------------------------------------------------------------------------
double fmax(
    const double a,  // first value
    const double b   // second value
  );

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
  );

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
  ) __attribute__((pure));

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
  ) __attribute__((pure));

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
void zero2d(double** a, const int nx, const int ny);

void zero3d(double*** a, const int nx, const int ny, const int nz);

void zero4d(double**** a, const int nx, const int ny, const int nz,
            const int nw);

// ---------------------------------------------------------------------------
// Count the number of non-empty lines in a text file.
//
// Opens the file, scans for newline characters, and accounts for a
// possible missing trailing newline on the last line. Terminates the
// program if the file cannot be opened.
// ---------------------------------------------------------------------------
int line_count(
    char* filename  // path to the text file
  );


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
//     dx · c_{i-1} + 4·dx · c_i + dx · c_{i+1} = (6/dx)(y_{i-1} − 2y_i + y_{i+1})
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
    const double* RESTRICT y,
    const int n,
    const double dx,
    double* RESTRICT c
  );

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
  );

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
  );

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
    double* arg,        // parameter array: arg[0] = bias q, arg[1] = Bessel order mu (real-valued)
    int argc            // length of arg (unused, kept for callback signature)
  );

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
