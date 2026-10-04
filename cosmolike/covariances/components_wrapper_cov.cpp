#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <carma.h>
#include <armadillo>
#include "covariance_wrapper_cov.hpp"
#include "gaussian_cov.h"
#include "halo_cov.h"
#include "mask_cov.h"
#include "non_gaussian_cov.h"
#include "operators_cov.h"
#include "perturbation_cov.h"
#include "ssc_cov.h"
#include "spectra_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/IA.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Inspect covariance spectra and their radial inputs in a notebook.
//
// Armadillo vectors describe the requested grids. CARMA converts the
// resulting matrices and cubes at the Python boundary. A later call or
// cosmology change cannot alter an earlier result. The C snapshot is
// temporary and its unique_ptr releases it even if Python allocation fails.
// No Ntable field, data-vector mask, covariance or likelihood state changes.
//
// The C routine stores one row per unordered field pair. Python receives
// a full symmetric [ell][field][field] array, plus copies of the radial
// geometry and window tables so it can audit the integration inputs.
// ---------------------------------------------------------------------------
py::dict covariance_limber_spectra_cpp(
    const arma::Col<double>& ell,     // multipole samples
    const arma::Col<double>& a_edges, // scale-factor panel edges
    const int nquad,        // Gaussian nodes per scale-factor panel
    const int nwindow,      // uniform-a lensing-efficiency samples
    const bool include_ia,  // include the signed NLA window
    const bool include_rsd, // include the lens redshift-distortion window
    const bool linear      // select linear rather than nonlinear matter P
  )
{
  if (ell.n_elem < 1
      || a_edges.n_elem < 2) {
    throw std::invalid_argument(
        "covariance_limber_spectra needs 1D ell and a_edges arrays, "
        "with at least one multipole and two scale-factor edges");
  }
  if (nwindow < 2) {
    throw std::invalid_argument("nwindow must be at least 2");
  }
  if (cosmology.chi == nullptr
      || cosmology.G == nullptr
      || cosmology.lnPL == nullptr
      || (!linear
          && cosmology.lnP == nullptr)
      || redshift.clustering_nbin < 1
      || redshift.shear_nbin < 1) {
    throw std::invalid_argument(
        "initialize lens/source samples and set_cosmology before spectra");
  }
  if (include_ia
      && nuisance.IA_MODEL != IA_MODEL_NLA) {
    throw std::invalid_argument(
        "covariance_limber_spectra supports NLA; initialize IA model 0");
  }
  for (arma::uword index=0; index<ell.n_elem; index++) {
    if (!std::isfinite(ell(index))
        || ell(index) < 1.0) {
      throw std::invalid_argument("ell must contain finite values >= 1");
    }
  }
  for (arma::uword edge=0; edge<a_edges.n_elem; edge++) {
    if (!std::isfinite(a_edges(edge))
        || a_edges(edge) <= 0.0
        || a_edges(edge) >= 1.0
        || (edge > 0
            && a_edges(edge) <= a_edges(edge-1))) {
      throw std::invalid_argument(
          "a_edges must increase strictly inside (0,1)");
    }
  }
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    throw std::invalid_argument(
        "nquad must be a tabulated rule: 64,96,128,256,512,1024");
  }

  // --- 1. ALLOCATE ONE SPECTRUM PER UNORDERED FIELD PAIR ---

  const int nfield = redshift.clustering_nbin+redshift.shear_nbin;
  const int npair = nfield*(nfield+1)/2;
  const int nell = ell.n_elem;

  // The C batch returns one contiguous row per unordered field pair.
  // Keep this temporary C workspace separate from the public Armadillo
  // cube, whose axes are (multipole, first field, second field).
  arma::Cube<double> spectra(nell, nfield, nfield);
  double** triangular = (double**) malloc2d(npair, nell);

  // --- 2. BUILD THE RADIAL SNAPSHOT AND INTEGRATE ALL SPECTRA ---

  // unique_ptr is the sole owner of this temporary C snapshot. Its
  // specified cleanup function, free_radial_cov, runs when the owner
  // leaves scope, including if a later Python array allocation fails.
  std::unique_ptr<radial_cov, decltype(&free_radial_cov)> radial(
      radial_inputs_cov(a_edges.n_elem-1, a_edges.memptr(), nquad,
                        nwindow, include_ia), &free_radial_cov);

  limber_spectra_cov(radial.get(), nell, ell.memptr(), linear, include_rsd,
                     triangular);

  // --- 3. EXPAND THE FIELD-PAIR TRIANGLE FOR PYTHON ---

  // Store both triangles from the same computed number. This makes the
  // returned field matrix exactly symmetric, independent of thread count.
  int pair = 0;

  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      for (int node=0; node<nell; node++) {
        spectra(node, first, second) = triangular[pair][node];
        spectra(node, second, first) = triangular[pair][node];
      }
      pair++;
    }
  }

  free(triangular);

  // --- 4. COPY THE INTEGRATION INPUTS BEFORE RELEASING THE SNAPSHOT ---

  // Geometry uses roles a, chi, f_K, dchi weight. Windows use density,
  // lensing/magnification and signed NLA roles, then field and radial node.
  arma::Mat<double> geometry(4, radial->nnode);
  arma::Cube<double> windows(3, nfield, radial->nnode);

  for (int node=0; node<radial->nnode; node++) {
    for (int role=0; role<4; role++) {
      geometry(role, node) = radial->geometry[role][node];
    }
    for (int role=0; role<3; role++) {
      for (int field=0; field<nfield; field++) {
        windows(role, field, node) =
            radial->window[role][field][node];
      }
    }
  }

  // All returned arrays now own their values independently of the C state.
  py::dict result;
  result["spectra"] = carma::cube_to_arr(spectra);
  result["geometry"] = carma::mat_to_arr(geometry);
  result["windows"] = carma::cube_to_arr(windows);
  result["nlens"] = radial->nlens;
  result["nsource"] = radial->nsource;
  return result;
}


// Numeric notebook arguments have fixed ranks: columns for grids and
// matrices for tabulated functions. CARMA handles the Python conversion;
// the calculations below use ordinary Armadillo indexing and ownership.
static void vector_cov(const arma::Col<double>& values, const char* name)
{
  if (values.is_empty()
      || !values.is_finite()) {
    throw std::invalid_argument(
        std::string(name)+" must be a nonempty finite vector");
  }
}

static void matrix_cov(const arma::Mat<double>& values, const char* name)
{
  if (values.is_empty()
      || !values.is_finite()) {
    throw std::invalid_argument(
        std::string(name)+" must be a nonempty finite matrix");
  }
}

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

static void angles_cov(const arma::Col<double>& edges)
{
  vector_cov(edges, "edges_rad");
  if (edges.n_elem < 2
      || edges(0) < 0.0
      || edges(edges.n_elem-1) > M_PI) {
    throw std::invalid_argument("need at least two angle edges inside [0,pi]");
  }
  for (arma::uword edge=1; edge<edges.n_elem; edge++) {
    if (edges(edge) <= edges(edge-1)) {
      throw std::invalid_argument("angle edges must increase strictly");
    }
  }
}

// Supply the same precomputed GSL rule to Python-prepared integrals as to
// the C radial, mass and angular kernels. Row 0 contains nodes on [-1,1];
// row 1 contains positive weights for integration over that interval.
// Python maps these onto its physical panels without generating another
// rule. The supported sizes all have at least 64 nodes. Armadillo owns the
// copy, so releasing GSL's descriptor cannot invalidate the returned data.
arma::Mat<double> covariance_integration_rule_cpp(
    const int nquad // precomputed rule size: 64,96,128,256,512,1024
  )
{
  quadrature_cov(nquad);
  arma::Mat<double> output(2, nquad);
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);

  // Node and weight belong to the same column. They integrate a function
  // on [-1,1]; the caller supplies the map to its physical interval.
  for (int node=0; node<nquad; node++) {
    gsl_integration_glfixed_point(-1.0, 1.0, node,
        &output(0, node), &output(1, node), rule);
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
// ---------------------------------------------------------------------------
arma::Mat<double> covariance_project_cpp(
    const arma::Mat<double>& left,   // [nleft,nnode], left operators
    const arma::Mat<double>& right,  // [nright,nnode], right rows
    const arma::Col<double>& weight  // [nnode], common integration weights
  )
{
  matrix_cov(left, "left");
  matrix_cov(right, "right");
  vector_cov(weight, "weight");
  if (left.n_cols != weight.n_elem
      || right.n_cols != weight.n_elem) {
    throw std::invalid_argument("left/right columns must match weight length");
  }

  arma::Mat<double> output(left.n_rows, right.n_rows);
  double** left_c = (double**) malloc2d(left.n_rows, weight.n_elem);
  double** right_c = (double**) malloc2d(right.n_rows, weight.n_elem);
  double** scratch = (double**) malloc2d(left.n_rows, weight.n_elem);
  double** result = (double**) malloc2d(left.n_rows, right.n_rows);

  // C sums along contiguous node rows. Copy by physical indices because
  // Armadillo stores columns contiguously; reinterpreting its memory
  // would silently interchange observables and integration nodes.
  for (arma::uword row=0; row<left.n_rows; row++) {
    for (arma::uword node=0; node<weight.n_elem; node++) {
      left_c[row][node] = left(row, node);
    }
  }
  for (arma::uword row=0; row<right.n_rows; row++) {
    for (arma::uword node=0; node<weight.n_elem; node++) {
      right_c[row][node] = right(row, node);
    }
  }

  gaussian_project_cov(left.n_rows, right.n_rows, weight.n_elem,
      left_c, right_c, weight.memptr(), scratch, result);

  for (arma::uword row=0; row<left.n_rows; row++) {
    for (arma::uword col=0; col<right.n_rows; col++) {
      output(row, col) = result[row][col];
    }
  }
  free(left_c);
  free(right_c);
  free(scratch);
  free(result);
  return output;
}

// The four rows are the AC, BD, AD and BC pairings of an AB-by-CD block.
// Noise is supplied separately so real-space calculations can replace the
// pure noise product by the analytic number of available galaxy pairs.
arma::Col<double> covariance_gaussian_wick_cpp(
    const arma::Mat<double>& cross_spectra, // [4,nell], signal only
    const arma::Col<double>& cross_noise,   // [4], matching white-noise powers
    const int ell_min,              // first consecutive integer multipole
    const double fsky,              // survey area / (4*pi)
    const bool include_noise_noise  // retain the pure noise product
  )
{
  matrix_cov(cross_spectra, "cross_spectra");
  vector_cov(cross_noise, "cross_noise");
  if (cross_spectra.n_rows != 4
      || cross_noise.n_elem != 4
      || ell_min < 0
      || !std::isfinite(fsky)
      || fsky <= 0.0
      || fsky > 1.0) {
    throw std::invalid_argument("need four pairings, ell_min>=0 and 0<fsky<=1");
  }

  arma::Col<double> output(cross_spectra.n_cols);
  double** spectra = (double**) malloc2d(4, output.n_elem);

  // Each C row is one Wick pairing, with multipole varying within it.
  for (int pair=0; pair<4; pair++) {
    for (arma::uword ell=0; ell<output.n_elem; ell++) {
      spectra[pair][ell] = cross_spectra(pair, ell);
    }
  }
  gaussian_wick_cov(ell_min, output.n_elem, fsky, spectra,
      cross_noise.memptr(), include_noise_noise, output.memptr());
  free(spectra);
  return output;
}

// Build all four estimator operators together. The notebook cube exposes
// probe, angular bin and multipole as separate axes.
arma::Cube<double> covariance_realspace_operator_cpp(
    const arma::Col<double>& edges_rad, // angular-bin boundaries in radians
    const int ell_max,          // last integer multipole, inclusive
    const int nquad             // integration nodes per angular bin
  )
{
  angles_cov(edges_rad);
  quadrature_cov(nquad);
  if (ell_max < 2) {
    throw std::invalid_argument("ell_max must be at least 2");
  }
  const arma::uword nbin = edges_rad.n_elem-1;
  arma::Cube<double> output(4, nbin, ell_max+1);
  double** kernels = (double**) malloc2d(4*nbin, ell_max+1);
  realspace_operator_cov(nbin, edges_rad.memptr(), ell_max, nquad, kernels);

  // The C batch combines probe and angular bin into one row index.
  // Expose them as separate notebook axes so each spin kernel is visible.
  for (int probe=0; probe<4; probe++) {
    for (arma::uword bin=0; bin<nbin; bin++) {
      for (int ell=0; ell<=ell_max; ell++) {
        output(probe, bin, ell) = kernels[probe*nbin+bin][ell];
      }
    }
  }
  free(kernels);
  return output;
}

arma::Mat<double> covariance_bandpower_operator_cpp(
    const arma::Col<int>& first, // inclusive lower multipole of each band
    const arma::Col<int>& last,  // inclusive upper multipole
    const int ell_min,          // first multipole of the shared output grid
    const int nell             // number of consecutive output multipoles
  )
{
  if (first.n_elem < 1
      || first.n_elem != last.n_elem
      || ell_min < 0
      || nell < 1) {
    throw std::invalid_argument(
        "need equal 1D band bounds and a valid ell grid");
  }
  for (arma::uword band=0; band<first.n_elem; band++) {
    if (first(band) < ell_min
        || last(band) < first(band)
        || last(band) >= ell_min+nell) {
      throw std::invalid_argument("band bounds must lie inside the ell grid");
    }
  }

  arma::Mat<double> output(first.n_elem, nell);
  double** kernels = (double**) malloc2d(first.n_elem, nell);
  bandpower_operator_cov(first.n_elem, ell_min, nell, first.memptr(),
      last.memptr(), kernels);

  // Rows label measured bands; columns retain the shared integer ell grid.
  for (arma::uword band=0; band<first.n_elem; band++) {
    for (int ell=0; ell<nell; ell++) {
      output(band, ell) = kernels[band][ell];
    }
  }
  free(kernels);
  return output;
}

double covariance_noise_pair_cpp(
    const int probe_left,       // xi+, xi-, gamma_t, w: 0,1,2,3
    const int probe_right,      // right estimator in the same convention
    const arma::Col<int>& fields,// [4], A,B,C,D global catalog indices
    const arma::Col<double>& noise_ab,  // [2], powers of catalogs A and B
    const double pair_area_sr2 // ordered-pair area within the angular bin
  )
{
  vector_cov(noise_ab, "noise_ab");
  if (fields.n_elem != 4
      || noise_ab.n_elem != 2
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
    if (fields(index) < 0) {
      throw std::invalid_argument("catalog indices must be nonnegative");
    }
  }
  if (noise_ab(0) < 0.0
      || noise_ab(1) < 0.0) {
    throw std::invalid_argument("noise powers must be nonnegative");
  }
  return gaussian_noise_pair_cov((probe_cov) probe_left,
      (probe_cov) probe_right, fields.memptr(), noise_ab.memptr(),
      pair_area_sr2);
}


// A raw mask power is nonnegative and keeps C0=area^2/(4*pi). Checking
// that convention here prevents interpreting a normalized mask as raw.
static void raw_mask_cov(const arma::Col<double>& mask, const double area)
{
  vector_cov(mask, "mask_cl");
  if (!std::isfinite(area)
      || area <= 0.0
      || area > 4.0*M_PI) {
    throw std::invalid_argument("area_sr must lie in (0,4*pi]");
  }
  for (arma::uword ell=0; ell<mask.n_elem; ell++) {
    if (mask(ell) < 0.0) {
      throw std::invalid_argument("mask_cl must be nonnegative");
    }
  }
  const double monopole = area*area/(4.0*M_PI);
  if (std::fabs(mask(0)/monopole-1.0) > 1.e-8) {
    throw std::invalid_argument("raw mask C0 must equal area_sr^2/(4*pi)");
  }
}

arma::Col<double> covariance_mask_pair_area_cpp(
    const arma::Col<double>& edges_rad,    // angular-bin edges in radians
    const arma::Col<double>& mask_cl,      // raw footprint spectrum, L=0 onward
    const double area_sr,          // footprint area
    const arma::Mat<double>& scalar_kernel // [nbin,nmask], w kernel
  )
{
  angles_cov(edges_rad);
  raw_mask_cov(mask_cl, area_sr);
  matrix_cov(scalar_kernel, "scalar_kernel");
  if (scalar_kernel.n_rows != edges_rad.n_elem-1
      || scalar_kernel.n_cols != mask_cl.n_elem) {
    throw std::invalid_argument("scalar_kernel must have shape [nbin,nmask]");
  }

  arma::Col<double> output(edges_rad.n_elem-1);
  double** kernels = (double**) malloc2d(output.n_elem, mask_cl.n_elem);
  for (arma::uword bin=0; bin<output.n_elem; bin++) {
    for (arma::uword ell=0; ell<mask_cl.n_elem; ell++) {
      kernels[bin][ell] = scalar_kernel(bin, ell);
    }
  }
  mask_pair_area_cov(output.n_elem, mask_cl.n_elem, area_sr,
      edges_rad.memptr(), mask_cl.memptr(), kernels, output.memptr());
  free(kernels);
  return output;
}

// The output describes the background power of a radial shell in the
// long-mode Limber approximation. It has units of length; the radial
// quadrature weight is applied later when the observable responses meet.
arma::Col<double> covariance_ssc_mask_variance_cpp(
    const arma::Col<double>& mask_cl, // raw footprint spectrum
    const double area_sr,    // footprint area in steradians
    const arma::Col<double>& distance,// positive transverse distances, c/H0
    const arma::Mat<double>& power   // [nnode,nmask], linear power, (c/H0)^3
  )
{
  raw_mask_cov(mask_cl, area_sr);
  vector_cov(distance, "distance");
  matrix_cov(power, "power");
  if (power.n_rows != distance.n_elem
      || power.n_cols != mask_cl.n_elem) {
    throw std::invalid_argument("power must have shape [nnode,nmask]");
  }
  for (arma::uword node=0; node<distance.n_elem; node++) {
    if (distance(node) <= 0.0) {
      throw std::invalid_argument("distance must be positive");
    }
  }

  arma::Col<double> output(distance.n_elem);
  double** powers = (double**) malloc2d(distance.n_elem, mask_cl.n_elem);
  for (arma::uword node=0; node<distance.n_elem; node++) {
    for (arma::uword ell=0; ell<mask_cl.n_elem; ell++) {
      powers[node][ell] = power(node, ell);
    }
  }
  ssc_mask_variance_cov(distance.n_elem, mask_cl.n_elem, area_sr,
      mask_cl.memptr(), distance.memptr(), powers, output.memptr());
  free(powers);
  return output;
}

// Each row describes one observable at one multipole. Its shell response
// includes the change of matter clustering and, when present, the change
// in the catalog mean used to define observed galaxy density.
arma::Mat<double> covariance_ssc_shell_response_cpp(
    const arma::Col<double>& distance,       // [nnode], transverse distances
    const arma::Col<double>& signal, // [nrow], full projected spectra
    const arma::Mat<double>& pair_window,    // [nrow,nnode], W_A*W_B
    const arma::Mat<double>& mean_window,    // [nrow,nnode], U_A+U_B
    const arma::Mat<double>& power_response  // [nrow,nnode], dP/d(delta_b)
  )
{
  vector_cov(distance, "distance");
  vector_cov(signal, "signal");
  matrix_cov(pair_window, "pair_window");
  matrix_cov(mean_window, "mean_window");
  matrix_cov(power_response, "power_response");
  const arma::uword nrow = signal.n_elem;
  const arma::uword nnode = distance.n_elem;

  if (pair_window.n_rows != nrow
      || mean_window.n_rows != nrow
      || power_response.n_rows != nrow
      || pair_window.n_cols != nnode
      || mean_window.n_cols != nnode
      || power_response.n_cols != nnode) {
    throw std::invalid_argument("response inputs must have shape [nrow,nnode]");
  }
  for (arma::uword node=0; node<nnode; node++) {
    if (distance(node) <= 0.0) {
      throw std::invalid_argument("distance must be positive");
    }
  }

  arma::Mat<double> output(nrow, nnode);
  double*** work = (double***) malloc3d(4, nrow, nnode);

  // Group the three supplied functions and the C result in one workspace.
  // Their rows refer to the same observable and radial shell throughout.
  for (arma::uword row=0; row<nrow; row++) {
    for (arma::uword node=0; node<nnode; node++) {
      work[0][row][node] = pair_window(row, node);
      work[1][row][node] = mean_window(row, node);
      work[2][row][node] = power_response(row, node);
    }
  }
  ssc_shell_response_cov(nrow, nnode, distance.memptr(), signal.memptr(),
      work[0], work[1], work[2], work[3]);

  for (arma::uword row=0; row<nrow; row++) {
    for (arma::uword node=0; node<nnode; node++) {
      output(row, node) = work[3][row][node];
    }
  }
  free(work);
  return output;
}

// Return the one-profile moment and the five pair moments as a matrix and
// a cube. A response slope may need I11 alone; pair_moments=false omits
// the pair integration and returns None as the second tuple member.
py::tuple covariance_halo_moments_cpp(
    const arma::Col<double>& a,         // scale factors
    const arma::Mat<double>& k,         // [na,nk], inverse c/H0
    const arma::Col<double>& lnm_edges, // log halo-mass panel edges
    const int nquad,           // Gaussian mass nodes per panel
    const bool pair_moments    // also compute the five pair-moment roles
  )
{
  vector_cov(a, "a");
  matrix_cov(k, "k");
  vector_cov(lnm_edges, "lnm_edges");
  quadrature_cov(nquad);
  if (k.n_rows != a.n_elem
      || lnm_edges.n_elem < 2) {
    throw std::invalid_argument(
        "k needs na rows and lnm_edges needs two edges");
  }
  if (cosmology.lnPL == nullptr
      || cosmology.G == nullptr
      || like.halo_model[3] != HALO_PROFILE_NFW) {
    throw std::invalid_argument("initialize cosmology and NFW halo profiles");
  }
  for (arma::uword row=0; row<a.n_elem; row++) {
    if (a(row) < limits.a_min
        || a(row) >= 1.0) {
      throw std::invalid_argument("a must lie in the initialized halo range");
    }
  }
  for (arma::uword index=0; index<k.n_elem; index++) {
    if (k(index) < 0.0) {
      throw std::invalid_argument("halo wavenumbers must be nonnegative");
    }
  }
  for (arma::uword edge=0; edge<lnm_edges.n_elem; edge++) {
    if (lnm_edges(edge) < std::log(limits.halo_m[RANGE_MIN])
        || lnm_edges(edge) > std::log(limits.halo_m[RANGE_MAX])
        || (edge > 0
            && lnm_edges(edge) <= lnm_edges(edge-1))) {
      throw std::invalid_argument(
          "mass edges must increase inside halo limits");
    }
  }

  const arma::uword na = a.n_elem;
  const arma::uword nk = k.n_cols;
  const arma::uword npair = nk*(nk+1)/2;
  arma::Mat<double> i11(na, nk);
  arma::Cube<double> moments;
  double*** pairs = nullptr; // omitted when only I11 is requested
  if (pair_moments) {
    moments.set_size(5, na, npair);
    pairs = (double***) malloc3d(5, na, npair);
  }
  double*** work = (double***) malloc3d(2, na, nk);

  // Work role 0 contains the requested k values; role 1 receives I11.
  // The five pair moments use a separate triangular k-pair axis.
  for (arma::uword row=0; row<na; row++) {
    for (arma::uword mode=0; mode<nk; mode++) {
      work[0][row][mode] = k(row, mode);
    }
  }
  halo_moments_cov(na, a.memptr(), nk, work[0], lnm_edges.n_elem-1,
      lnm_edges.memptr(), nquad, work[1], pairs);

  for (arma::uword row=0; row<na; row++) {
    for (arma::uword mode=0; mode<nk; mode++) {
      i11(row, mode) = work[1][row][mode];
    }
  }
  if (pair_moments) {
    for (int role=0; role<5; role++) {
      for (arma::uword row=0; row<na; row++) {
        for (arma::uword pair=0; pair<npair; pair++) {
          moments(role, row, pair) = pairs[role][row][pair];
        }
      }
    }
    free(pairs);
  }
  free(work);

  if (!pair_moments) {
    return py::make_tuple(carma::mat_to_arr(i11), py::none());
  }
  return py::make_tuple(carma::mat_to_arr(i11), carma::cube_to_arr(moments));
}

// Read one vector or a matrix of physical wavenumbers at a shared a.
// Matrix rows are independent integration batches distributed by C over
// OpenMP workers. A vector retains the original serial core-reader path.
// The output has the input's shape; the covariance owns the input grid.
arma::Mat<double> covariance_power_cpp(
    const double a,      // scale factor inside the initialized range
    const arma::Mat<double>& k,  // physical wavenumbers in inverse c/H0
    const bool linear   // linear total-matter P or the configured Pdelta
  )
{
  matrix_cov(k, "k");
  if (!std::isfinite(a)
      || a < limits.a_min
      || a >= 1.0
      || cosmology.lnPL == nullptr
      || (!linear
          && cosmology.lnP == nullptr)) {
    throw std::invalid_argument("initialize power tables and use a_min<=a<1");
  }
  for (arma::uword index=0; index<k.n_elem; index++) {
    if (k(index) <= 0.0) {
      throw std::invalid_argument("power wavenumbers must be positive");
    }
  }

  arma::Mat<double> output(k.n_rows, k.n_cols);
  double*** work = (double***) malloc3d(2, k.n_rows, k.n_cols);
  for (arma::uword row=0; row<k.n_rows; row++) {
    for (arma::uword col=0; col<k.n_cols; col++) {
      work[0][row][col] = k(row, col);
    }
  }
  power_rows_cov(a, k.n_rows, k.n_cols, work[0], linear, work[1]);
  for (arma::uword row=0; row<k.n_rows; row++) {
    for (arma::uword col=0; col<k.n_cols; col++) {
      output(row, col) = work[1][row][col];
    }
  }
  free(work);
  return output;
}


// A column vector is one requested k grid. Reuse the same validation and
// core batch as the matrix overload, then return that single column.
arma::Col<double> covariance_power_vector_cpp(
    const double a,             // scale factor
    const arma::Col<double>& k, // wavenumbers in inverse c/H0
    const bool linear          // linear or configured nonlinear power
  )
{
  const arma::Mat<double> grid = k.t();
  const arma::Mat<double> power = covariance_power_cpp(a, grid, linear);
  return power.row(0).t();
}

// The angular rule and its power samples are supplied together. The
// arrays must describe the same K,Q pairs and the same angular nodes;
// otherwise the cancellations between perturbation diagrams are lost.
arma::Mat<double> covariance_tree_averages_cpp(
    const arma::Mat<double>& k,      // [2,npair], positive K and Q
    const arma::Mat<double>& pk,     // [2,npair], matching linear power
    const arma::Col<double>& corner, // [nangle], stable 1+cos(theta)
    const arma::Col<double>& weight, // [nangle], normalized dtheta/pi weights
    const arma::Mat<double>& ps      // [npair,nangle], P(|K+Q|)
  )
{
  matrix_cov(k, "k");
  matrix_cov(pk, "pk");
  matrix_cov(ps, "ps");
  vector_cov(corner, "corner");
  vector_cov(weight, "weight");
  if (k.n_rows != 2
      || pk.n_rows != 2
      || pk.n_cols != k.n_cols
      || ps.n_rows != k.n_cols
      || ps.n_cols != corner.n_elem
      || corner.n_elem != weight.n_elem) {
    throw std::invalid_argument(
        "tree input pair and angle dimensions disagree");
  }
  for (arma::uword point=0; point<k.n_elem; point++) {
    if (k(point) <= 0.0) {
      throw std::invalid_argument("tree wavenumbers must be positive");
    }
  }
  double normalization = 0.0;
  for (arma::uword node=0; node<corner.n_elem; node++) {
    if (corner(node) <= 0.0
        || corner(node) > 2.0
        || weight(node) <= 0.0) {
      throw std::invalid_argument("need 0<corner<=2 and positive weights");
    }
    normalization += weight(node);
  }
  if (std::fabs(normalization-1.0) > 1.e-10) {
    throw std::invalid_argument("angular weights must sum to one");
  }

  arma::Mat<double> output(3, k.n_cols);
  double*** pair_inputs = (double***) malloc3d(2, 2, k.n_cols);
  double** angular_power = (double**) malloc2d(ps.n_rows, ps.n_cols);
  double** averages = (double**) malloc2d(3, k.n_cols);

  // K and Q label the two external modes, while ps samples their vector
  // sum as its relative angle changes. These are different physical axes.
  for (arma::uword pair=0; pair<k.n_cols; pair++) {
    for (int role=0; role<2; role++) {
      pair_inputs[0][role][pair] = k(role, pair);
      pair_inputs[1][role][pair] = pk(role, pair);
    }
    for (arma::uword node=0; node<corner.n_elem; node++) {
      angular_power[pair][node] = ps(pair, node);
    }
  }
  tree_averages_cov(k.n_cols, corner.n_elem, pair_inputs[0],
      pair_inputs[1], corner.memptr(), weight.memptr(), angular_power,
      averages);

  for (int role=0; role<3; role++) {
    for (arma::uword pair=0; pair<k.n_cols; pair++) {
      output(role, pair) = averages[role][pair];
    }
  }
  free(pair_inputs);
  free(angular_power);
  free(averages);
  return output;
}

// Keep the five halo contributions separate in the returned array so
// callers can examine their scale dependence before projecting their sum.
// All supplied moments and powers must refer to the same density field.
arma::Mat<double> covariance_halo_trispectrum_cpp(
    const arma::Mat<double>& pk,      // [2,npoint], linear power at K,Q
    const arma::Mat<double>& i11,     // [2,npoint], one-profile moments
    const arma::Mat<double>& moments, // [5,npoint], halo pair moments
    const arma::Mat<double>& tree     // [3,npoint], angular P/B/T averages
  )
{
  matrix_cov(pk, "pk");
  matrix_cov(i11, "i11");
  matrix_cov(moments, "moments");
  matrix_cov(tree, "tree");
  const arma::uword npoint = pk.n_cols;
  if (pk.n_rows != 2
      || i11.n_rows != 2
      || moments.n_rows != 5
      || tree.n_rows != 3
      || i11.n_cols != npoint
      || moments.n_cols != npoint
      || tree.n_cols != npoint) {
    throw std::invalid_argument(
        "trispectrum inputs need 2,2,5,3 matching rows");
  }

  arma::Mat<double> output(5, npoint);
  double*** work = (double***) malloc3d(5, 5, npoint);

  // The C kernel consumes four tables with 2, 2, 5 and 3 physical roles.
  // Group their workspace with the five output terms; unused rows are
  // neither read nor included in the halo sums.
  for (arma::uword point=0; point<npoint; point++) {
    for (int role=0; role<2; role++) {
      work[0][role][point] = pk(role, point);
      work[1][role][point] = i11(role, point);
    }
    for (int role=0; role<5; role++) {
      work[2][role][point] = moments(role, point);
    }
    for (int role=0; role<3; role++) {
      work[3][role][point] = tree(role, point);
    }
  }
  halo_trispectrum_cov(npoint, work[0], work[1], work[2], work[3], work[4]);
  for (int role=0; role<5; role++) {
    for (arma::uword point=0; point<npoint; point++) {
      output(role, point) = work[4][role][point];
    }
  }
  free(work);
  return output;
}

// The two coefficients and the differentiated spectrum are explicit:
// the Python workflow selects a response prescription, not the binding.
// Row 0 returns halo power; row 1 returns its dimensional response or
// the fractional response transferred to the supplied target power.
arma::Mat<double> covariance_halo_response_cpp(
    const arma::Mat<double>& inputs, // [6,npoint], halo inputs
    const double growth_coefficient,   // constant growth contribution
    const double dilation_coefficient, // coefficient of logarithmic slope
    const bool fractional              // transfer D_halo/P_halo to P_target
  )
{
  matrix_cov(inputs, "inputs");
  if (inputs.n_rows != 6
      || !std::isfinite(growth_coefficient)
      || !std::isfinite(dilation_coefficient)) {
    throw std::invalid_argument(
        "need six response rows and finite coefficients");
  }
  for (arma::uword point=0; point<inputs.n_cols; point++) {
    const double i11 = inputs(2, point);
    const double phalo = i11*i11*(inputs(0, point))
                         +inputs(3, point);
    if (!std::isfinite(phalo)
        || phalo <= 0.0) {
      throw std::invalid_argument("supplied moments must give positive halo P");
    }
  }

  arma::Mat<double> output(2, inputs.n_cols);
  double*** work = (double***) malloc3d(2, 6, inputs.n_cols);
  for (int role=0; role<6; role++) {
    for (arma::uword point=0; point<inputs.n_cols; point++) {
      work[0][role][point] = inputs(role, point);
    }
  }
  halo_response_cov(inputs.n_cols, growth_coefficient,
      dilation_coefficient, fractional, work[0], work[1]);
  for (int role=0; role<2; role++) {
    for (arma::uword point=0; point<inputs.n_cols; point++) {
      output(role, point) = work[1][role][point];
    }
  }
  free(work);
  return output;
}

} // namespace cosmolike_interface
