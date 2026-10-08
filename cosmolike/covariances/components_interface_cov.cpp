#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include "production_interface_cov.hpp"
#include "gaussian_cov.h"
#include "halo_cov.h"
#include "mask_cov.h"
#include "non_gaussian_cov.h"
#include "operators_cov.h"
#include "perturbation_cov.h"
#include "ssc_cov.h"
#include "spectra_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Production bindings to the individual covariance components.
//
// NumPy owns all arrays. A C-style array has adjacent elements within each
// row, so its rows can be passed directly to the existing C components.
// The small pointer vectors below describe those rows; they copy no data.
//
// Inputs are borrowed: every array argument is registered with noconvert(),
// so pybind11 accepts only C-contiguous float64 (cov_array) or int32
// (cov_int_array) arrays and never makes a converted copy. C reads them
// and never writes to them. Each output is a cov_array allocated in its
// binding and owned by Python once returned.
//
// Units are the core ones: distances in c/H0, wavenumbers in (c/H0)^-1,
// power in (c/H0)^3, angles in radians and areas in steradians.
//
// Threads: these conversions open no OpenMP region and call no BLAS
// routine. Each C component distributes its own loop over the OpenMP
// team. Only covariance_halo_moments and covariance_power read core state
// that is initialized on first use (lazy halo tables; the Pdelta run-mode
// latch). Their C routines initialize it serially before any worker reads
// it, and the vector power path is serial throughout.
// ---------------------------------------------------------------------------
using cov_array = py::array_t<double, py::array::c_style>;
using cov_int_array = py::array_t<int, py::array::c_style>;

// Reject NaN and infinity anywhere in an array. C arithmetic would carry
// them silently into every covariance entry that reads the value.
static void finite_cov(const cov_array& values, const char* name)
{
  // size() multiplies the array dimensions; do it once before the scan.
  const py::ssize_t count = values.size(); // number of elements to check
  const double* data = values.data(); // borrowed contiguous array storage

  for (py::ssize_t index=0; index<count; index++) {
    if (!std::isfinite(data[index])) {
      throw std::invalid_argument(std::string(name)+" must be finite");
    }
  }
}

// A nonempty, finite 1D input.
static void vector_cov(const cov_array& values, const char* name)
{
  if (values.ndim() != 1
      || values.size() < 1) {
    throw std::invalid_argument(
        std::string(name)+" must be a nonempty 1D array");
  }
  finite_cov(values, name);
}

// A finite 2D input with at least one row and one column.
static void matrix_cov(const cov_array& values, const char* name)
{
  if (values.ndim() != 2
      || values.shape(0) < 1
      || values.shape(1) < 1) {
    throw std::invalid_argument(
        std::string(name)+" must be a nonempty 2D array");
  }
  finite_cov(values, name);
}

// Collect the address of each row of a C-contiguous 2D array for C
// routines that take row pointers (const double* const*). The vector owns
// only the addresses; the values stay in the NumPy array, which outlives
// the synchronous C call. output_rows_cov does the same for an output.
static std::vector<const double*> input_rows_cov(const cov_array& values)
{
  std::vector<const double*> rows(values.shape(0));
  for (py::ssize_t row=0; row<values.shape(0); row++) {
    rows[row] = values.data(row, 0);
  }
  return rows;
}

static std::vector<double*> output_rows_cov(cov_array& values)
{
  std::vector<double*> rows(values.shape(0));
  for (py::ssize_t row=0; row<values.shape(0); row++) {
    rows[row] = values.mutable_data(row, 0);
  }
  return rows;
}

// GSL tabulates the Gauss-Legendre nodes for these sizes; the C kernels
// accept no other rule, so reject the rest before calling them.
static void quadrature_cov(const int nquad)
{
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    throw std::invalid_argument("nquad must be 64,96,128,256,512 or 1024");
  }
}

// Angular-bin edges in radians: at least two finite values, inside
// [0,pi] (the possible separations on the sphere), strictly increasing.
static void angles_cov(const cov_array& edges)
{
  vector_cov(edges, "edges_rad");
  if (edges.size() < 2
      || edges.data()[0] < 0.0
      || edges.data()[edges.size()-1] > M_PI) {
    throw std::invalid_argument("need at least two angle edges inside [0,pi]");
  }
  for (py::ssize_t edge=1; edge<edges.size(); edge++) {
    if (edges.data()[edge] <= edges.data()[edge-1]) {
      throw std::invalid_argument("angle edges must increase strictly");
    }
  }
}

// Supply the same precomputed GSL rule to Python-prepared integrals as to
// the C radial, mass and angular kernels. Row 0 contains nodes on [-1,1];
// row 1 contains positive weights for integration over that interval.
// Python maps these onto its physical panels without generating another
// rule: on [lo,hi] a node x becomes (lo+hi)/2 + (hi-lo)/2 x and a weight w
// becomes w (hi-lo)/2. The supported sizes all have at least 64 nodes.
// NumPy owns the output [2,nquad], so releasing GSL's descriptor cannot
// invalidate the returned data. Only nquad is validated; the GSL calls
// are serial and read no cosmology state.
static cov_array covariance_integration_rule(
    const int nquad // precomputed rule size: 64,96,128,256,512,1024
  )
{
  quadrature_cov(nquad);
  cov_array output({2, nquad});
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);

  // Copy each node and its matching weight before releasing the rule.
  for (int node=0; node<nquad; node++) {
    gsl_integration_glfixed_point(-1.0, 1.0, node,
        output.mutable_data(0, node), output.mutable_data(1, node), rule);
  }

  gsl_integration_glfixed_table_free(rule);
  return output;
}

// ---------------------------------------------------------------------------
// Project supplied rows through a shared weighted sum.
//
// Every output is sum_node left[i,node]*weight[node]*right[j,node].
// For Gaussian covariance the nodes are multipoles. For density SSC they
// can instead be radial shells. In the latter case, using the same
// responses on both sides and positive weights preserves nonnegative
// variance. The caller can request a rectangular matrix subblock without
// changing the summation order. All parallel work remains in the C code.
//
// Arrays: left[nleft,nnode], right[nright,nnode] and weight[nnode] in,
// output[nleft,nright] out; the units are those of the caller's rows.
// Validation: both matrices and the weights are finite and nonempty, and
// share the node count. gaussian_project_cov first forms the weighted
// left rows in the scratch array (allocated here, discarded at return),
// then one ordered dot product per entry. A standalone call distributes
// output tiles over the OpenMP team; no core table is read.
// ---------------------------------------------------------------------------
static cov_array covariance_project(
    const cov_array& left,   // [nleft,nnode], left response/operator rows
    const cov_array& right,  // [nright,nnode], right rows
    const cov_array& weight  // [nnode], common integration weights
  )
{
  matrix_cov(left, "left");
  matrix_cov(right, "right");
  vector_cov(weight, "weight");
  if (left.shape(1) != weight.size()
      || right.shape(1) != weight.size()) {
    throw std::invalid_argument("left/right columns must match weight length");
  }

  cov_array scratch({left.shape(0), weight.size()});
  cov_array output({left.shape(0), right.shape(0)});
  auto left_rows = input_rows_cov(left);
  auto right_rows = input_rows_cov(right);
  auto scratch_rows = output_rows_cov(scratch);
  auto output_rows = output_rows_cov(output);

  gaussian_project_cov(left.shape(0), right.shape(0), weight.size(),
      left_rows.data(), right_rows.data(), weight.data(),
      scratch_rows.data(), output_rows.data());
  return output;
}

// ---------------------------------------------------------------------------
// Gaussian covariance of two spectra C_AB and C_CD at each integer ell.
//
// The four rows are the AC, BD, AD and BC pairings of an AB-by-CD block.
// Noise is supplied separately so real-space calculations can replace the
// pure noise product by the analytic number of available galaxy pairs.
// For ell = ell_min, ..., ell_min+nell-1 the output is
//
//   G(ell) = [(C_AC+N_AC)(C_BD+N_BD) + (C_AD+N_AD)(C_BC+N_BC)]
//            / [(2 ell+1) fsky],
//
// with the pure noise products N*N omitted unless include_noise_noise.
// Arrays: cross_spectra[4,nell] signal spectra, cross_noise[4] white
// noise 1/n or sigma_component^2/n with n per steradian (zero unless the
// two catalogs coincide), output G[nell]. Validation: finite inputs, four
// rows, ell_min >= 0 and 0 < fsky <= 1. gaussian_wick_cov distributes
// the multipoles over the OpenMP team; it reads no global state.
// ---------------------------------------------------------------------------
static cov_array covariance_gaussian_wick(
    const cov_array& cross_spectra, // [4,nell], signal only
    const cov_array& cross_noise,   // [4], matching white-noise powers
    const int ell_min,              // first consecutive integer multipole
    const double fsky,              // survey area / (4*pi)
    const bool include_noise_noise  // retain the pure noise product
  )
{
  matrix_cov(cross_spectra, "cross_spectra");
  vector_cov(cross_noise, "cross_noise");
  if (cross_spectra.shape(0) != 4
      || cross_noise.size() != 4
      || ell_min < 0
      || !std::isfinite(fsky)
      || fsky <= 0.0
      || fsky > 1.0) {
    throw std::invalid_argument("need four pairings, ell_min>=0 and 0<fsky<=1");
  }

  auto rows = input_rows_cov(cross_spectra);
  cov_array output(cross_spectra.shape(1));
  gaussian_wick_cov(ell_min, output.size(), fsky, rows.data(),
      cross_noise.data(), include_noise_noise, output.mutable_data());
  return output;
}

// ---------------------------------------------------------------------------
// Full-sky operators K[probe,bin,ell] that turn a harmonic spectrum into
// an angular-bin average: X_bin = sum_ell K[probe,bin,ell] C_ell.
//
// Build all four estimator operators together. The flattened C row order
// is probe*nbin+bin; reshape on return without copying the owned data.
// Probes are xi+, xi-, gamma_t and w (0..3), and each row already holds
// the (2 ell+1)/(4 pi) factor and the area average over its bin. The
// output [4,nbin,ell_max+1] is dimensionless and covers ell = 0..ell_max;
// spin rows are zero below ell=2. The shear rows act on unit-normalized
// observed-shear spectra; operators_cov.c gives the factor that converts
// each source leg of a core-convention spectrum. Validation:
// edges_rad[nbin+1] finite, strictly increasing inside [0,pi]; tabulated
// nquad; ell_max >= 2.
// realspace_operator_cov gives each OpenMP worker one (probe,bin) row;
// it reads no cosmology, so the result can be kept across cosmologies.
// ---------------------------------------------------------------------------
static cov_array covariance_realspace_operator(
    const cov_array& edges_rad, // angular-bin boundaries in radians
    const int ell_max,          // last integer multipole, inclusive
    const int nquad             // integration nodes per angular bin
  )
{
  angles_cov(edges_rad);
  quadrature_cov(nquad);
  if (ell_max < 2) {
    throw std::invalid_argument("ell_max must be at least 2");
  }
  const py::ssize_t nbin = edges_rad.size()-1;
  cov_array output({4*nbin, (py::ssize_t) ell_max+1});
  auto rows = output_rows_cov(output);
  realspace_operator_cov(nbin, edges_rad.data(), ell_max, nquad, rows.data());
  output.resize({(py::ssize_t) 4, nbin, (py::ssize_t) ell_max+1});
  return output;
}

// ---------------------------------------------------------------------------
// Mode-weighted Fourier band operators on a consecutive integer ell grid.
//
// Each ell carries 2 ell+1 independent modes, so a band average weights
// it by (2 ell+1)/N_band, where N_band = sum over the band of (2 ell+1).
// Arrays: first[nband] and last[nband], int32 inclusive absolute
// multipoles; output [nband,nell] dimensionless, column j at ell_min+j,
// zero outside each band. Validation: equal nonempty 1D bounds, ell_min
// >= 0, nell >= 1 and every band inside the grid. bandpower_operator_cov
// gives each OpenMP worker one band; no cosmology state is read.
// ---------------------------------------------------------------------------
static cov_array covariance_bandpower_operator(
    const cov_int_array& first, // inclusive lower multipole of each band
    const cov_int_array& last,  // inclusive upper multipole
    const int ell_min,          // first multipole of the shared output grid
    const int nell             // number of consecutive output multipoles
  )
{
  if (first.ndim() != 1
      || last.ndim() != 1
      || first.size() < 1
      || first.size() != last.size()
      || ell_min < 0
      || nell < 1) {
    throw std::invalid_argument(
        "need equal 1D band bounds and a valid ell grid");
  }

  // Each band must be nonempty and lie on the supplied grid, whose last
  // multipole is ell_min+nell-1; the C routine would stop otherwise.
  for (py::ssize_t band=0; band<first.size(); band++) {
    if (first.data()[band] < ell_min
        || last.data()[band] < first.data()[band]
        || last.data()[band] >= ell_min+nell) {
      throw std::invalid_argument("band bounds must lie inside the ell grid");
    }
  }

  cov_array output({first.size(), (py::ssize_t) nell});
  auto rows = output_rows_cov(output);
  bandpower_operator_cov(first.size(), ell_min, nell, first.data(),
      last.data(), rows.data());
  return output;
}

// ---------------------------------------------------------------------------
// Pure shot/shape-noise covariance of two estimators in one angular bin.
//
// For w_AB and w_CD the noise pairings give
//   (delta_AC delta_BD + delta_AD delta_BC) N_A N_B / pair_area,
// where delta compares catalog IDs and N = 1/n or sigma_component^2/n,
// with n per steradian. gamma_t keeps only the direct pairing; xi+ and
// xi- double the w expression; different estimators give zero.
// Inputs: two probe IDs (0 xi+, 1 xi-, 2 gamma_t, 3 w), fields[4] int32
// catalog IDs A,B,C,D (one unique ID per catalog across lenses and
// sources), noise_ab[2] = N_A, N_B, and the ordered-pair area in sr^2.
// Validation: probe IDs in 0..3, four nonnegative IDs, two finite
// nonnegative noise powers, finite positive pair area. The scalar result
// comes from gaussian_noise_pair_cov; there is no threading or state.
// ---------------------------------------------------------------------------
static double covariance_noise_pair(
    const int probe_left,       // xi+, xi-, gamma_t, w: 0,1,2,3
    const int probe_right,      // right estimator in the same convention
    const cov_int_array& fields,// [4], A,B,C,D global catalog indices
    const cov_array& noise_ab,  // [2], powers of catalogs A and B
    const double pair_area_sr2 // ordered-pair area within the angular bin
  )
{
  vector_cov(noise_ab, "noise_ab");
  if (fields.ndim() != 1
      || fields.size() != 4
      || noise_ab.size() != 2
      || probe_left < 0
      || probe_left > 3
      || probe_right < 0
      || probe_right > 3
      || !std::isfinite(pair_area_sr2)
      || pair_area_sr2 <= 0.0) {
    throw std::invalid_argument(
        "invalid estimator, catalog or pair-area input");
  }
  for (int index=0; index<4; index++) {
    if (fields.data()[index] < 0) {
      throw std::invalid_argument("catalog indices must be nonnegative");
    }
  }
  if (noise_ab.data()[0] < 0.0
      || noise_ab.data()[1] < 0.0) {
    throw std::invalid_argument("noise powers must be nonnegative");
  }
  return gaussian_noise_pair_cov((probe_cov) probe_left,
      (probe_cov) probe_right, fields.data(), noise_ab.data(), pair_area_sr2);
}


// A raw mask power is nonnegative and keeps C0=area^2/(4*pi). Checking
// that convention here prevents interpreting a normalized mask as raw.
static void raw_mask_cov(const cov_array& mask, const double area)
{
  vector_cov(mask, "mask_cl");
  if (!std::isfinite(area)
      || area <= 0.0
      || area > 4.0*M_PI) {
    throw std::invalid_argument("area_sr must lie in (0,4*pi]");
  }
  for (py::ssize_t ell=0; ell<mask.size(); ell++) {
    if (mask.data()[ell] < 0.0) {
      throw std::invalid_argument("mask_cl must be nonnegative");
    }
  }
  const double monopole = area*area/(4.0*M_PI);
  if (std::fabs(mask.data()[0]/monopole-1.0) > 1.e-8) {
    throw std::invalid_argument("raw mask C0 must equal area_sr^2/(4*pi)");
  }
}

// ---------------------------------------------------------------------------
// Ordered-pair angular area of a common footprint in each angular bin.
//
// Two unclustered positions inside the mask W, separated by an angle in
// the bin, give n_A n_B A_pair expected ordered pairs. With the raw mask
// power C_L^W and the scalar-bin operator K[bin,L] of w (including its
// (2L+1)/(4 pi) factor and bin average), mask_pair_area_cov evaluates
//   A_pair = 8 pi^2 Delta_x sum_L C_L^W K[bin,L],
// Delta_x = cos(theta_low)-cos(theta_high). The result feeds the pure
// noise term of covariance_noise_pair. Arrays: edges_rad[nbin+1] radians,
// mask_cl[nmask] raw C_L^W from L=0, scalar_kernel[nbin,nmask]; output
// [nbin] in sr^2. Validation: edges, raw-mask convention and shapes.
// C stops the process if a reconstructed area is not positive (check the
// footprint and its band limit). Each OpenMP worker owns two bins.
// ---------------------------------------------------------------------------
static cov_array covariance_mask_pair_area(
    const cov_array& edges_rad,    // angular-bin edges in radians
    const cov_array& mask_cl,      // raw footprint spectrum, L=0 onward
    const double area_sr,          // footprint area
    const cov_array& scalar_kernel// [nbin,nmask], bin-averaged w operator
  )
{
  angles_cov(edges_rad);
  raw_mask_cov(mask_cl, area_sr);
  matrix_cov(scalar_kernel, "scalar_kernel");
  if (scalar_kernel.shape(0) != edges_rad.size()-1
      || scalar_kernel.shape(1) != mask_cl.size()) {
    throw std::invalid_argument("scalar_kernel must have shape [nbin,nmask]");
  }

  cov_array output(edges_rad.size()-1);
  auto rows = input_rows_cov(scalar_kernel);
  mask_pair_area_cov(output.size(), mask_cl.size(), area_sr,
      edges_rad.data(), mask_cl.data(), rows.data(), output.mutable_data());
  return output;
}

// ---------------------------------------------------------------------------
// Variance of the survey-averaged background mode delta_b at each shell.
//
// The output describes the background power of a radial shell in the
// long-mode Limber approximation. It has units of length; the radial
// quadrature weight is applied later when the observable responses meet:
//
//   sigma_b^2(chi) = sum_L (2L+1) C_L^W P_lin((L+1/2)/f_K, a)
//                    / [area_sr^2 f_K^2].
//
// Arrays: mask_cl[nmask] raw mask power, distance[nnode] f_K in c/H0,
// power[nnode,nmask] = P_lin((L+1/2)/f_K, a) in (c/H0)^3 at each node and
// L; output sigma_b^2[nnode] in c/H0. Validation: raw-mask convention,
// shapes, finite values and positive distances. ssc_mask_variance_cov
// gives each OpenMP worker two radial nodes; it reads no core table.
// ---------------------------------------------------------------------------
static cov_array covariance_ssc_mask_variance(
    const cov_array& mask_cl, // raw footprint spectrum
    const double area_sr,    // footprint area in steradians
    const cov_array& distance,// positive transverse distances, c/H0
    const cov_array& power   // [nnode,nmask], linear power, (c/H0)^3
  )
{
  raw_mask_cov(mask_cl, area_sr);
  vector_cov(distance, "distance");
  matrix_cov(power, "power");
  if (power.shape(0) != distance.size()
      || power.shape(1) != mask_cl.size()) {
    throw std::invalid_argument("power must have shape [nnode,nmask]");
  }
  for (py::ssize_t node=0; node<distance.size(); node++) {
    if (distance.data()[node] <= 0.0) {
      throw std::invalid_argument("distance must be positive");
    }
  }

  cov_array output(distance.size());
  auto rows = input_rows_cov(power);
  ssc_mask_variance_cov(distance.size(), mask_cl.size(), area_sr,
      mask_cl.data(), distance.data(), rows.data(), output.mutable_data());
  return output;
}

// ---------------------------------------------------------------------------
// Response of each projected spectrum to the background mode of a shell.
//
// Each row describes one observable at one multipole. Its shell response
// includes the change of matter clustering and, when present, the change
// in the catalog mean used to define observed galaxy density:
//
//   Phi(chi) = W_A W_B D((ell+1/2)/f_K, chi)/f_K^2 - (U_A+U_B) C_AB(ell),
//
// where D = dP/d(delta_b) and U describes the catalog-mean response.
// SSC is then sum_node dchi sigma_b^2 Phi_i Phi_j. Arrays: distance[nnode]
// f_K, signal[nrow] dimensionless C_AB, pair_window[nrow,nnode] W_A W_B in
// (c/H0)^-2, mean_window[nrow,nnode] U_A+U_B in (c/H0)^-1 (zero without a
// catalog-mean normalization), power_response[nrow,nnode] in (c/H0)^3;
// output Phi[nrow,nnode] in (c/H0)^-1, without the dchi weight.
// Validation: finite values, matching shapes, positive distances.
// ssc_shell_response_cov gives each OpenMP worker one row; no core table.
// ---------------------------------------------------------------------------
static cov_array covariance_ssc_shell_response(
    const cov_array& distance,       // [nnode], transverse distances
    const cov_array& signal,         // [nrow], complete projected spectra
    const cov_array& pair_window,    // [nrow,nnode], W_A*W_B
    const cov_array& mean_window,    // [nrow,nnode], U_A+U_B
    const cov_array& power_response  // [nrow,nnode], dP/d(delta_b)
  )
{
  vector_cov(distance, "distance");
  vector_cov(signal, "signal");
  matrix_cov(pair_window, "pair_window");
  matrix_cov(mean_window, "mean_window");
  matrix_cov(power_response, "power_response");
  const py::ssize_t nrow = signal.size();
  const py::ssize_t nnode = distance.size();

  if (pair_window.shape(0) != nrow
      || mean_window.shape(0) != nrow
      || power_response.shape(0) != nrow
      || pair_window.shape(1) != nnode
      || mean_window.shape(1) != nnode
      || power_response.shape(1) != nnode) {
    throw std::invalid_argument("response inputs must have shape [nrow,nnode]");
  }
  for (py::ssize_t node=0; node<nnode; node++) {
    if (distance.data()[node] <= 0.0) {
      throw std::invalid_argument("distance must be positive");
    }
  }

  cov_array output({nrow, nnode});
  auto pair_rows = input_rows_cov(pair_window);
  auto mean_rows = input_rows_cov(mean_window);
  auto power_rows = input_rows_cov(power_response);
  auto rows = output_rows_cov(output);
  ssc_shell_response_cov(nrow, nnode, distance.data(), signal.data(),
      pair_rows.data(), mean_rows.data(), power_rows.data(), rows.data());
  return output;
}

// ---------------------------------------------------------------------------
// Halo-model mass integrals of the cold-matter-plus-baryon (cb) field.
//
// Return both the one-profile moment and the five pair moments. Their
// NumPy arrays own the values after the C routine releases its workspace.
// The temporary pointers merely give C access to [role][a][pair] rows;
// no moment is copied and no covariance approximation is chosen here.
// A response slope can request I11 alone: its other moments are already
// available at the central k. Return None for the omitted pair array.
//
//   I_mu^beta = integral dlnM (dn/dlnM) b_beta (M/rho_cb)^mu
//               product_i u(k_i|M),
// with b_0 = 1 and b_1 the linear halo bias; array names write beta first
// (I11, I02, ...). I11 receives the completion that restores I11(k->0)=1
// for halos below the lowest mass edge.
// Arrays: a[na]; k[na,nk] in (c/H0)^-1; lnm_edges[npanel+1] = ln(M/[Msun/h]).
// Outputs: i11[na,nk], dimensionless; moments[5,na,nk*(nk+1)/2] for the
// roles I02(K,Q), I12(K,Q), I13(K,Q,Q), I13(K,K,Q), I04(K,K,Q,Q), in
// (c/H0)^3,^3,^6,^6,^9, pairs in (0,0),(0,1),...,(1,1),... order of k.
// Validation: linear power and growth tables set; NFW profiles; finite
// inputs; limits.a_min <= a < 1; k >= 0; edges increasing inside
// [ln halo_sigma_min, ln halo_m[max]]; tabulated nquad. The default
// eleven four-decade panels from 10^-40 to 10^4 Msun/h switch on the
// C routine's Wynn extrapolation of I11; other layouts are finite sums.
// halo_moments_cov warms its lazy core tables serially at entry, then
// runs its OpenMP loops over (a,mass), (a,k) and (a,k-pair) work.
// ---------------------------------------------------------------------------
static py::tuple covariance_halo_moments(
    const cov_array& a,         // scale factors
    const cov_array& k,         // [na,nk], inverse c/H0
    const cov_array& lnm_edges, // logarithmic halo-mass panel boundaries
    const int nquad,           // Gaussian mass nodes per panel
    const bool pair_moments    // also compute the five pair-moment roles
  )
{
  vector_cov(a, "a");
  matrix_cov(k, "k");
  vector_cov(lnm_edges, "lnm_edges");
  quadrature_cov(nquad);
  if (k.shape(0) != a.size()
      || lnm_edges.size() < 2) {
    throw std::invalid_argument(
        "k needs na rows and lnm_edges needs two edges");
  }
  if (cosmology.lnPL == nullptr
      || cosmology.G == nullptr
      || like.halo_model[3] != HALO_PROFILE_NFW) {
    throw std::invalid_argument("initialize cosmology and NFW halo profiles");
  }
  for (py::ssize_t row=0; row<a.size(); row++) {
    if (a.data()[row] < limits.a_min
        || a.data()[row] >= 1.0) {
      throw std::invalid_argument("a must lie in the initialized halo range");
    }
  }
  for (py::ssize_t index=0; index<k.size(); index++) {
    if (k.data()[index] < 0.0) {
      throw std::invalid_argument("halo wavenumbers must be nonnegative");
    }
  }
  for (py::ssize_t edge=0; edge<lnm_edges.size(); edge++) {
    if (lnm_edges.data()[edge] < std::log(limits.halo_sigma_min)
        || lnm_edges.data()[edge] > std::log(limits.halo_m[RANGE_MAX])
        || (edge > 0
            && lnm_edges.data()[edge] <= lnm_edges.data()[edge-1])) {
      throw std::invalid_argument(
          "mass edges must increase inside halo limits");
    }
  }

  const py::ssize_t na = a.size();
  const py::ssize_t nk = k.shape(1);
  const py::ssize_t npair = nk*(nk+1)/2;
  cov_array i11({na, nk});
  auto k_rows = input_rows_cov(k);
  auto i11_rows = output_rows_cov(i11);

  if (!pair_moments) {
    halo_moments_cov(na, a.data(), nk, k_rows.data(), lnm_edges.size()-1,
        lnm_edges.data(), nquad, i11_rows.data(), nullptr);
    return py::make_tuple(i11, py::none());
  }

  cov_array moments({(py::ssize_t) 5, na, npair});
  std::vector<std::vector<double*>> moment_rows(5);
  std::vector<double**> roles(5);

  // Build the [role][a] row map that C expects: moment_rows[role][row]
  // points at moments[role,row,0], the first of its npair pair entries,
  // and roles[role] points at that role's list of rows.
  for (int role=0; role<5; role++) {
    moment_rows[role].resize(na);
    for (py::ssize_t row=0; row<na; row++) {
      moment_rows[role][row] = moments.mutable_data(role, row, 0);
    }
    roles[role] = moment_rows[role].data();
  }

  halo_moments_cov(na, a.data(), nk, k_rows.data(), lnm_edges.size()-1,
      lnm_edges.data(), nquad, i11_rows.data(), roles.data());
  return py::make_tuple(i11, moments);
}

// ---------------------------------------------------------------------------
// Matter power P(k,a) at one scale factor, read from the core tables.
//
// Read one vector or a matrix of physical wavenumbers at a shared a.
// Matrix rows are independent batches of table reads, distributed by C
// (power_rows_cov) over OpenMP workers after it touches the Pdelta
// run-mode latch serially. A vector retains the original serial
// core-reader path (p_lin_at_a or Pdelta_at_a). The output has the
// input's shape; the caller chooses the k grid and no core grid changes.
// k is in (c/H0)^-1 and P in (c/H0)^3. linear selects p_lin(k,a); else
// the run-mode Pdelta. Validation: finite k > 0, 1D or 2D and nonempty;
// limits.a_min <= a < 1; linear (and, if needed, nonlinear) tables set.
// ---------------------------------------------------------------------------
static cov_array covariance_power(
    const double a,      // scale factor inside the initialized range
    const cov_array& k,  // physical wavenumbers in inverse c/H0
    const bool linear   // linear total-matter P or the configured Pdelta
  )
{
  if (k.ndim() == 1) {
    vector_cov(k, "k");
  } else {
    matrix_cov(k, "k");
  }
  if (!std::isfinite(a)
      || a < limits.a_min
      || a >= 1.0
      || cosmology.lnPL == nullptr
      || (!linear
          && cosmology.lnP == nullptr)) {
    throw std::invalid_argument("initialize power tables and use a_min<=a<1");
  }
  // size() multiplies the array dimensions; do it once before the scan.
  const py::ssize_t count = k.size(); // number of wavenumbers to check
  const double* data = k.data(); // borrowed contiguous wavenumber storage

  for (py::ssize_t index=0; index<count; index++) {
    if (data[index] <= 0.0) {
      throw std::invalid_argument("power wavenumbers must be positive");
    }
  }

  cov_array output(k.request().shape);
  if (k.ndim() == 2) {
    auto k_rows = input_rows_cov(k);
    auto rows = output_rows_cov(output);
    power_rows_cov(a, k.shape(0), k.shape(1), k_rows.data(), linear,
                    rows.data());
  } else if (linear) {
    p_lin_at_a(a, k.data(), k.size(), output.mutable_data());
  } else {
    Pdelta_at_a(a, k.data(), k.size(), output.mutable_data());
  }
  return output;
}


// ---------------------------------------------------------------------------
// Linear power for base-10 log wavenumbers plus one scalar shift.
//
// Same lnPL table read as covariance_power with linear=true, minus the
// per-sample log10: the caller supplies the log wavenumbers once and
// moves the shell dependence into the scalar shift (the
// connected-covariance angle grid is shared by every radial shell up to
// -log10(f_K)). The physical wavenumber of each sample is
// 10^(log10k+shift) in (c/H0)^-1. Not bitwise the physical-k reader: the
// shifted sum rounds differently in the last bits (spectra_cov.h). log10k
// is a 2D array; rows are distributed over OpenMP workers by C.
// ---------------------------------------------------------------------------
static cov_array covariance_power_logk(
    const double a,          // scale factor inside the initialized range
    const cov_array& log10k, // [nrow,ncol] base-10 logs before the shift
    const double shift       // common addend to every sample
  )
{
  matrix_cov(log10k, "log10k");
  if (!std::isfinite(a)
      || a < limits.a_min
      || a >= 1.0
      || !std::isfinite(shift)
      || cosmology.lnPL == nullptr) {
    throw std::invalid_argument("initialize power tables and use a_min<=a<1");
  }
  cov_array output(log10k.request().shape);
  auto logk_rows = input_rows_cov(log10k);
  auto rows = output_rows_cov(output);
  linear_power_logk_rows_cov(a, log10k.shape(0), log10k.shape(1),
      logk_rows.data(), shift, rows.data());
  return output;
}


// ---------------------------------------------------------------------------
// Planar angular averages of tree-level P, B and T for the covariance.
//
// The angular rule and its power samples are supplied together. The
// arrays must describe the same K,Q pairs and the same angular nodes;
// otherwise the cancellations between perturbation diagrams are lost.
// The average over the angle theta between k and q uses the measure
// dtheta/pi. Arrays: k[2,npair] = K,Q in (c/H0)^-1; pk[2,npair] linear
// P(K), P(Q) in (c/H0)^3; corner[nangle] = 1+cos(theta), computed as
// 2 sin^2((pi-theta)/2) near pi; weight[nangle]; ps[npair,nangle] linear
// P(|k+q|). Output [3,npair]: <P>, <B_tree>, <T_tree> in (c/H0)^3,^6,^9.
// Validation: finite values and matching shapes; K,Q > 0; 0 < corner
// <= 2; positive weights summing to one within 1e-10. tree_averages_cov
// gives each OpenMP worker two (K,Q) pairs; it reads no core table.
// ---------------------------------------------------------------------------
static cov_array covariance_tree_averages(
    const cov_array& k,      // [2,npair], positive K and Q
    const cov_array& pk,     // [2,npair], matching linear power
    const cov_array& corner, // [nangle], stable 1+cos(theta)
    const cov_array& weight, // [nangle], normalized dtheta/pi weights
    const cov_array& ps      // [npair,nangle], P(|K+Q|)
  )
{
  matrix_cov(k, "k");
  matrix_cov(pk, "pk");
  matrix_cov(ps, "ps");
  vector_cov(corner, "corner");
  vector_cov(weight, "weight");
  if (k.shape(0) != 2
      || pk.shape(0) != 2
      || pk.shape(1) != k.shape(1)
      || ps.shape(0) != k.shape(1)
      || ps.shape(1) != corner.size()
      || corner.size() != weight.size()) {
    throw std::invalid_argument(
        "tree input pair and angle dimensions disagree");
  }

  // The stable kernel formulas divide by K and Q. The angular nodes must
  // exclude theta=pi (corner=0), where |k+q| can vanish, and their weights
  // must form a normalized average.
  for (py::ssize_t point=0; point<k.size(); point++) {
    if (k.data()[point] <= 0.0) {
      throw std::invalid_argument("tree wavenumbers must be positive");
    }
  }
  double normalization = 0.0;
  for (py::ssize_t node=0; node<corner.size(); node++) {
    if (corner.data()[node] <= 0.0
        || corner.data()[node] > 2.0
        || weight.data()[node] <= 0.0) {
      throw std::invalid_argument("need 0<corner<=2 and positive weights");
    }
    normalization += weight.data()[node];
  }
  if (std::fabs(normalization-1.0) > 1.e-10) {
    throw std::invalid_argument("angular weights must sum to one");
  }

  cov_array output({(py::ssize_t) 3, k.shape(1)});
  auto k_rows = input_rows_cov(k);
  auto pk_rows = input_rows_cov(pk);
  auto ps_rows = input_rows_cov(ps);
  auto rows = output_rows_cov(output);
  tree_averages_cov(k.shape(1), corner.size(), k_rows.data(),
      pk_rows.data(), corner.data(), weight.data(), ps_rows.data(),
      rows.data());
  return output;
}

// ---------------------------------------------------------------------------
// Angle-averaged halo-model trispectrum T(K,Q) of the connected covariance.
//
// Keep the five halo contributions separate in the returned array so
// callers can examine their scale dependence before projecting their sum.
// All supplied moments and powers must refer to the same density field.
// Arrays (one column per (K,Q,a) point): pk[2,npoint] linear P(K), P(Q);
// i11[2,npoint] I11(K), I11(Q); moments[5,npoint] in halo_cov order
// I02, I12, I13(K,Q,Q), I13(K,K,Q), I04 (I02 unused); tree[3,npoint]
// <P>, <B>, <T> from covariance_tree_averages. Output [5,npoint]: 1h,
// 2h(1+3), 2h(2+2), 3h, 4h, each in (c/H0)^9. Validation: finite values
// and the 2,2,5,3 row counts with a common point count; signs are kept.
// halo_trispectrum_cov splits points over the OpenMP team; no core table.
// ---------------------------------------------------------------------------
static cov_array covariance_halo_trispectrum(
    const cov_array& pk,      // [2,npoint], linear power at K,Q
    const cov_array& i11,     // [2,npoint], one-profile moments
    const cov_array& moments, // [5,npoint], pair moments in halo_cov order
    const cov_array& tree     // [3,npoint], angular P/B/T averages
  )
{
  matrix_cov(pk, "pk");
  matrix_cov(i11, "i11");
  matrix_cov(moments, "moments");
  matrix_cov(tree, "tree");
  const py::ssize_t npoint = pk.shape(1);
  if (pk.shape(0) != 2
      || i11.shape(0) != 2
      || moments.shape(0) != 5
      || tree.shape(0) != 3
      || i11.shape(1) != npoint
      || moments.shape(1) != npoint
      || tree.shape(1) != npoint) {
    throw std::invalid_argument(
        "trispectrum inputs need 2,2,5,3 matching rows");
  }

  cov_array output({(py::ssize_t) 5, npoint});
  auto pk_rows = input_rows_cov(pk);
  auto i11_rows = input_rows_cov(i11);
  auto moment_rows = input_rows_cov(moments);
  auto tree_rows = input_rows_cov(tree);
  auto rows = output_rows_cov(output);
  halo_trispectrum_cov(npoint, pk_rows.data(), i11_rows.data(),
      moment_rows.data(), tree_rows.data(), rows.data());
  return output;
}

// ---------------------------------------------------------------------------
// Halo power and its response to a background density mode delta_b.
//
// The two coefficients and the differentiated spectrum are explicit:
// the Python workflow selects a response prescription, not the binding.
// Row 0 returns halo power; row 1 returns its dimensional response or
// the fractional response transferred to the supplied target power:
//
//   P_halo = I11^2 P_lin + I02,
//   D_halo = (growth - dilation * slope) I11^2 P_lin + I12,
//   D      = (D_halo/P_halo) P_target if fractional, else D_halo.
//
// Input rows [6,npoint]: P_lin, P_target, I11, I02(k,k), I12(k,k) and the
// slope dlnP_X/dlnk, at independent (k,a) points; powers and moments in
// (c/H0)^3, I11 and slope dimensionless. Output [2,npoint] in (c/H0)^3.
// Validation: finite inputs and coefficients, six rows, P_halo > 0.
// halo_response_cov splits points over the OpenMP team; no core table.
// ---------------------------------------------------------------------------
static cov_array covariance_halo_response(
    const cov_array& inputs,            // [6,npoint], documented halo inputs
    const double growth_coefficient,   // constant growth contribution
    const double dilation_coefficient, // coefficient of logarithmic slope
    const bool fractional              // transfer D_halo/P_halo to P_target
  )
{
  matrix_cov(inputs, "inputs");
  if (inputs.shape(0) != 6
      || !std::isfinite(growth_coefficient)
      || !std::isfinite(dilation_coefficient)) {
    throw std::invalid_argument(
        "need six response rows and finite coefficients");
  }

  // The fractional response divides by P_halo, and C stops the process
  // when P_halo <= 0. Repeat its check at every point so Python receives
  // an exception instead.
  for (py::ssize_t point=0; point<inputs.shape(1); point++) {
    const double i11 = *inputs.data(2, point);
    const double phalo = i11*i11*(*inputs.data(0, point))
                         +*inputs.data(3, point);
    if (!std::isfinite(phalo)
        || phalo <= 0.0) {
      throw std::invalid_argument("supplied moments must give positive halo P");
    }
  }

  cov_array output({(py::ssize_t) 2, inputs.shape(1)});
  auto input_rows = input_rows_cov(inputs);
  auto rows = output_rows_cov(output);
  halo_response_cov(inputs.shape(1), growth_coefficient,
      dilation_coefficient, fractional, input_rows.data(), rows.data());
  return output;
}

// Register every component above on the production submodule. Each
// string is the Python help text; noconvert() on an array argument makes
// pybind11 reject, rather than copy, a non-contiguous or mistyped array.
void bind_production_components_cov(py::module_& module)
{

  module.def("covariance_integration_rule", &covariance_integration_rule,
      "Precomputed GSL nodes and weights [2,nquad] on [-1,1]. "
      "Allowed sizes: 64,96,128,256,512,1024; smaller rules are rejected.",
      py::arg("nquad"));

  module.def("covariance_mask_pair_area", &covariance_mask_pair_area,
      "Ordered-pair area [nbin] in sr^2 from a raw mask and bin kernel.",
      py::arg("edges_rad").noconvert(), py::arg("mask_cl").noconvert(),
      py::arg("area_sr"), py::arg("scalar_kernel").noconvert());

  module.def("covariance_ssc_mask_variance", &covariance_ssc_mask_variance,
      "Long-mode Limber background strength [nnode], in c/H0 length units.",
      py::arg("mask_cl").noconvert(), py::arg("area_sr"),
      py::arg("distance").noconvert(), py::arg("power").noconvert());

  module.def("covariance_ssc_shell_response", &covariance_ssc_shell_response,
      "Observable response [nrow,nnode], including projected mean subtraction.",
      py::arg("distance").noconvert(), py::arg("signal").noconvert(),
      py::arg("pair_window").noconvert(), py::arg("mean_window").noconvert(),
      py::arg("power_response").noconvert());

  module.def("covariance_halo_moments", &covariance_halo_moments,
      "Return I11 [na,nk] and moments [5,na,nk*(nk+1)/2]. "
      "pair_moments=False omits pair sums and returns (I11, None). "
      "Both use the cb halo convention.",
      py::arg("a").noconvert(), py::arg("k").noconvert(),
      py::arg("lnm_edges").noconvert(), py::arg("nquad"),
      py::arg("pair_moments") = true);

  module.def("covariance_power", &covariance_power,
      "Read power for a k vector or matrix at one a; retain its shape. "
      "Matrix rows use OpenMP. Input/output use the core c/H0 units.",
      py::arg("a"), py::arg("k").noconvert(), py::arg("linear") = true);

  module.def("covariance_power_logk", &covariance_power_logk,
      "Linear power for [nrow,ncol] base-10 log wavenumbers plus one "
      "scalar shift: k = 10^(log10k+shift) in (c/H0)^-1. Rows use OpenMP. "
      "Not bitwise covariance_power at the same k (last bits).",
      py::arg("a"), py::arg("log10k").noconvert(), py::arg("shift"));

  module.def("covariance_tree_averages", &covariance_tree_averages,
      "Planar P/B/T averages [3,npair] from supplied linear-power inputs.",
      py::arg("k").noconvert(), py::arg("pk").noconvert(),
      py::arg("corner").noconvert(), py::arg("weight").noconvert(),
      py::arg("ps").noconvert());

  module.def("covariance_halo_trispectrum", &covariance_halo_trispectrum,
      "Return five halo terms [5,npoint]: 1h,2h(13),2h(22),3h,4h.",
      py::arg("pk").noconvert(), py::arg("i11").noconvert(),
      py::arg("moments").noconvert(), py::arg("tree").noconvert());

  module.def("covariance_halo_response", &covariance_halo_response,
      "Return halo power and dimensional response [2,npoint].",
      py::arg("inputs").noconvert(), py::arg("growth_coefficient"),
      py::arg("dilation_coefficient"), py::arg("fractional") = true);

  module.def("covariance_project", &covariance_project,
      "Contract rows through common weights; return an owned matrix.",
      py::arg("left").noconvert(), py::arg("right").noconvert(),
      py::arg("weight").noconvert());

  module.def("covariance_gaussian_wick", &covariance_gaussian_wick,
      "Gaussian AB,CD covariance from AC,BD,AD,BC rows on consecutive ell.",
      py::arg("cross_spectra").noconvert(), py::arg("cross_noise").noconvert(),
      py::arg("ell_min"), py::arg("fsky"),
      py::arg("include_noise_noise") = false);

  module.def("covariance_realspace_operator", &covariance_realspace_operator,
      "Return [4,nbin,ell_max+1] full-sky xi+,xi-,gamma_t,w operators.",
      py::arg("edges_rad").noconvert(), py::arg("ell_max"), py::arg("nquad"));

  module.def("covariance_bandpower_operator", &covariance_bandpower_operator,
      "Return normalized integer-mode weights [nband,nell]; bounds inclusive.",
      py::arg("first").noconvert(), py::arg("last").noconvert(),
      py::arg("ell_min"), py::arg("nell"));

  module.def("covariance_noise_pair", &covariance_noise_pair,
      "Pure real-space count/shape noise for one angular-bin pair area.",
      py::arg("probe_left"), py::arg("probe_right"),
      py::arg("fields").noconvert(), py::arg("noise_ab").noconvert(),
      py::arg("pair_area_sr2"));
}
}
