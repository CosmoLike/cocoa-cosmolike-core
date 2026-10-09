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
#include "nonlimber_cov.h"
#include "ia_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/IA.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Armadillo notebook wrappers of the covariance components.
//
// Each function below exposes one step of the covariance calculation:
//
//   NumPy argument (C order, Fortran order, a sliced view or read-only)
//     -> notebook_input_cov, called by the binding: a private copy in an
//        owning arma::Col, arma::Mat or arma::Cube; the caller's array is
//        never written or aliased
//     -> wrapper here: shape and physical-domain checks, then a short copy
//        into a padded malloc2d/malloc3d C workspace with contiguous rows
//     -> shared C routine: every integral, SIMD loop and OpenMP team, the
//        same routine that the production interface calls
//     -> copy back into Armadillo; CARMA exports the result to NumPy,
//        where it owns its memory independently of later calls.
//
// Checks throw std::invalid_argument, which pybind11 raises as a Python
// ValueError. The C routines stop the whole Python process (log_fatal and
// exit) on the inputs they check and trust the rest, so the wrappers test
// shapes, physical domains and initialization first. Core units: distances
// in c/H0, wavenumbers in (c/H0)^-1, matter power in (c/H0)^3 and white
// noise in steradians.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Inspect covariance spectra and their radial inputs in a notebook.
//
// PHYSICAL QUANTITY
// A Gaussian covariance pairs the fields of two measured spectra AB and
// CD as AC*BD + AD*BC, so it needs the angular power spectrum of every
// pair of catalogs, including pairs absent from the data vector. In the
// Limber approximation each one is a radial integral,
//
//   C_AB(ell) = integral dchi W_A W_B P((ell+1/2)/f_K, a) / f_K^2,
//
// evaluated on one common Gauss-Legendre rule over the supplied panels in
// scale factor (radial_inputs_cov, limber_spectra_cov). include_rsd adds
// one common RSD window to each lens wherever it appears. include_ia adds
// the signed NLA window or, for TATT, its further E terms and B spectra
// (tatt_spectra_cov). nonlimber_lmax > 0 adds the exact-minus-matched
// separable linear spectra to every row containing a lens field, at
// integer ell in [2,nonlimber_lmax] (apply_nonlimber_cov). The fields are
// the lens samples followed by the source samples.
//
// RETURNED DICT
//   spectra    arma::Cube [nell,nfield,nfield], dimensionless E spectra
//              in the core C_ell convention, no noise, exactly symmetric
//   b_spectra  the same axes for TATT B modes; None for NLA or no IA
//   geometry   arma::Mat [4,nnode]: a, chi, f_K and the dchi weight, the
//              last three in c/H0
//   windows    arma::Cube [3,nfield,nnode]: density, lensing/magnification
//              and signed NLA windows, in (c/H0)^-1
//   nlens, nsource  field counts; nnode = npanel*nquad radial nodes
//
// OWNERSHIP AND STATE
// Armadillo vectors describe the requested grids. CARMA converts the
// resulting matrices and cubes at the Python boundary. A later call or
// cosmology change cannot alter an earlier result. The C snapshot is
// temporary and its unique_ptr releases it even if Python allocation fails.
// No Ntable field, data-vector mask, covariance or likelihood state changes;
// only lazily built core tables are filled.
//
// The C routine stores one row per unordered field pair. Python receives
// a full symmetric [ell][field][field] array, plus copies of the radial
// geometry and window tables so it can audit the integration inputs.
// ---------------------------------------------------------------------------
py::dict covariance_limber_spectra_cpp(
    const arma::Col<double>& ell,     // [nell], multipoles >= 1
    const arma::Col<double>& a_edges, // [npanel+1], scale-factor panel edges
    const int nquad,        // Gauss-Legendre nodes per scale-factor panel
    const int nwindow,      // uniform-a lensing-efficiency samples
    const bool include_ia,  // NLA window; with TATT also its E and B terms
    const bool include_rsd, // include the lens redshift-distortion window
    const bool linear,     // 1: linear p_lin(k,a); 0: run-mode Pdelta(k,a)
    const int nonlimber_lmax, // gg/gs correction through this ell; 0 disables
    const int nonlimber_nchi, // logarithmic radial samples, 2^n+1
    const double nonlimber_chi_min // positive near distance in c/H0
  )
{
  // Check the inputs before any allocation or C call. A C fatal check
  // would stop the whole Python process; an exception here only reports
  // the mistake to the caller. Flat geometry is not tested here:
  // radial_inputs_cov itself stops the process for a curved cosmology.
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
      && nuisance.IA_MODEL != IA_MODEL_NLA
      && nuisance.IA_MODEL != IA_MODEL_TATT) {
    throw std::invalid_argument(
        "covariance_spectra supports NLA or TATT; initialize IA model 0 or 1");
  }

  // Limber reads P at k=(ell+1/2)/f_K, and the shear spin factor
  // sqrt[(ell-1)ell(ell+1)(ell+2)] is real only for ell >= 1.
  for (arma::uword index=0; index<ell.n_elem; index++) {
    if (!std::isfinite(ell(index))
        || ell(index) < 1.0) {
      throw std::invalid_argument("ell must contain finite values >= 1");
    }
  }

  // Each consecutive pair of edges is one Gauss-Legendre panel in scale
  // factor. Strictly increasing edges inside (0,1) give panels of positive
  // width between the far boundary and the observer at a=1.
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

  // Only GSL's precomputed Gauss-Legendre tables are accepted.
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    throw std::invalid_argument(
        "nquad must be a tabulated rule: 64,96,128,256,512,1024");
  }

  // The hybrid non-Limber correction starts at ell=2, so lmax is either 0
  // (disabled) or at least 2. FFTLog needs a power-of-two number of
  // log-distance intervals, m = nchi-1 = 2^n >= 64. A power of two has a
  // single set bit; subtracting one clears it and sets every lower bit, so
  // the bit test m&(m-1) is zero exactly when m is a power of two.
  if (nonlimber_lmax < 0
      || nonlimber_lmax == 1
      || nonlimber_nchi < 65
      || ((nonlimber_nchi-1) & (nonlimber_nchi-2)) != 0
      || !std::isfinite(nonlimber_chi_min)
      || nonlimber_chi_min <= 0.0) {
    throw std::invalid_argument(
        "non-Limber requires lmax=0 or >=2, nchi=2^n+1 >=65, chi_min>0");
  }
  if (nonlimber_lmax > 0) {
    // The non-Limber transfers have no RSD term, and the separable field
    // D(a) delta(k,1) does not describe the scale-dependent growth that
    // massive neutrinos introduce.
    if (include_rsd
        || cosmology.Omega_nu != 0.0) {
      throw std::invalid_argument(
          "non-Limber covariance currently requires no RSD and mnu=0");
    }

    // The correction is tabulated at integer multipoles 2..lmax and C
    // reads the entry (int) ell - 2. A fractional ell in that range would
    // silently receive the correction of a neighboring integer.
    for (int index=0; index<ell.n_elem; index++) {
      const double value = ell(index);
      if (value <= nonlimber_lmax
          && value >= 2.0
          && value != std::floor(value)) {
        throw std::invalid_argument(
            "non-Limber correction requires integer ell below its cutoff");
      }
    }
  }

  // The TATT galaxy-alignment terms use only the lens density and
  // magnification windows, so they cannot be combined with lens RSD.
  if (include_ia
      && nuisance.IA_MODEL == IA_MODEL_TATT
      && include_rsd) {
    throw std::invalid_argument("Gaussian TATT currently requires no RSD");
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

  // radial_inputs_cov samples a, chi, f_K, the dchi weight and every field
  // window on npanel*nquad Gauss-Legendre nodes. limber_spectra_cov then
  // integrates C_AB for all pairs into triangular, in the i-major order
  // (0,0),(0,1),...,(1,1),... of the field indices.
  limber_spectra_cov(radial.get(), nell, ell.memptr(), linear, include_rsd,
                     triangular);

  // TATT: add the E-mode terms beyond NLA to triangular in place, and fill
  // the B modes from zero. Parity gives B only to source-source pairs;
  // other rows stay zero. NLA has no B mode, so b_triangular stays null.
  double** b_triangular = nullptr;
  if (include_ia
      && nuisance.IA_MODEL == IA_MODEL_TATT) {
    b_triangular = (double**) malloc2d(npair, nell);
    tatt_spectra_cov(radial.get(), nell, ell.memptr(), triangular, b_triangular);
  }

  // Non-Limber: add exact-minus-matched separable linear spectra to every
  // row containing a lens field, at integer ell in [2,lmax]. Source-source
  // rows stay Limber. a_edges(0) is the far radial boundary of FFTLog.
  if (nonlimber_lmax > 0) {
    apply_nonlimber_cov(radial.get(), a_edges(0), nwindow, include_ia,
        nonlimber_lmax, nonlimber_nchi, nonlimber_chi_min,
        nell, ell.memptr(), triangular);
  }

  // --- 3. EXPAND THE FIELD-PAIR TRIANGLE FOR PYTHON ---

  // Store both triangles from the same computed number. This makes the
  // returned field matrix exactly symmetric, independent of thread count.
  // A default-constructed b_spectra is empty; without TATT it stays empty
  // and is returned as None below.
  arma::Cube<double> b_spectra;
  if (b_triangular != nullptr) {
    b_spectra.set_size(nell, nfield, nfield);
  }
  int pair = 0;

  // Visit the field pairs in the same i-major order that C used, so
  // triangular[pair] holds the spectrum of (first,second). The innermost
  // loop copies each multipole to (first,second) and to its mirror.
  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      for (int node=0; node<nell; node++) {
        spectra(node, first, second) = triangular[pair][node];
        spectra(node, second, first) = triangular[pair][node];
        if (b_triangular != nullptr) {
          b_spectra(node, first, second) = b_triangular[pair][node];
          b_spectra(node, second, first) = b_triangular[pair][node];
        }
      }
      pair++;
    }
  }

  free(triangular);

  // Without TATT b_triangular is null, and free(nullptr) does nothing.
  free(b_triangular);

  // --- 4. COPY THE INTEGRATION INPUTS BEFORE RELEASING THE SNAPSHOT ---

  // Geometry uses roles a, chi, f_K, dchi weight. Windows use density,
  // lensing/magnification and signed NLA roles, then field and radial node.
  // Distances and the dchi weight are in c/H0; windows are in (c/H0)^-1.
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
  // These local Armadillo objects are not const, so CARMA moves their
  // storage into a heap object that the NumPy array owns through a
  // capsule, without copying the elements.
  py::dict result;
  result["spectra"] = carma::cube_to_arr(spectra);
  result["b_spectra"] = py::none();
  if (!b_spectra.is_empty()) {
    result["b_spectra"] = carma::cube_to_arr(b_spectra);
  }
  result["geometry"] = carma::mat_to_arr(geometry);
  result["windows"] = carma::cube_to_arr(windows);
  result["nlens"] = radial->nlens;
  result["nsource"] = radial->nsource;
  return result;
}


// Numeric notebook arguments have fixed ranks: columns for grids and
// matrices for tabulated functions. notebook_input_cov has already copied
// each NumPy argument into Armadillo; CARMA only exports the results. The
// calculations below use ordinary Armadillo indexing and ownership. These
// two helpers reject empty arrays and NaN or infinite entries.
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

// Only GSL's precomputed Gauss-Legendre tables with at least 64 nodes are
// accepted: 64 for low-level tests and the 96 to 1024 accuracy ladder.
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

// Angular-bin edges in radians: at least one bin, strictly increasing,
// inside [0,pi].
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
// rule: on [lo,hi] the node x becomes (hi+lo)/2 + (hi-lo)/2 x and its
// weight is multiplied by (hi-lo)/2. The supported sizes all have at least
// 64 nodes. The arma::Mat [2,nquad] owns the copy, so releasing GSL's
// descriptor cannot invalidate the returned data.
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
//
// Arrays: left arma::Mat [nleft,nnode], right arma::Mat [nright,nnode]
// and weight arma::Col [nnode]; the result is an arma::Mat [nleft,nright]
// whose units are the product of the three inputs' units. The C routine
// gaussian_project_cov forms left*weight once, then one dot product with
// each right row, always summing the nodes in increasing order.
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

  // scratch receives left*weight once; C reuses it for every right row.
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

// ---------------------------------------------------------------------------
// Gaussian covariance of two spectra C_AB and C_CD at each multipole.
//
// The four rows are the AC, BD, AD and BC pairings of an AB-by-CD block.
// Noise is supplied separately so real-space calculations can replace the
// pure noise product by the analytic number of available galaxy pairs.
// The C routine gaussian_wick_cov evaluates, at ell = ell_min+index,
//
//   G(ell) = [(C_AC+N_AC)(C_BD+N_BD) + (C_AD+N_AD)(C_BC+N_BC)]
//            / [(2 ell+1) fsky],
//
// counting the 2 ell+1 modes of each multipole within the fraction fsky
// of the sky. include_noise_noise=false omits the N*N products. N is in
// steradians: 1/n or sigma_e^2/n, with n per steradian. The result is an
// arma::Col [nell] in the squared units of the spectra.
// ---------------------------------------------------------------------------
arma::Col<double> covariance_gaussian_wick_cpp(
    const arma::Mat<double>& cross_spectra, // [4,nell], signal only
    const arma::Col<double>& cross_noise,   // [4], matching white noise, sr
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

// ---------------------------------------------------------------------------
// Bin-averaged full-sky operators of the four real-space estimators.
//
// A correlation is X(theta) = sum_ell (2 ell+1)/(4 pi) d_ell(theta) C_ell,
// with spin kernels d^ell_(2,2), d^ell_(2,-2), d^ell_(2,0) and d^ell_(0,0)
// for xi+, xi-, gamma_t and w. realspace_operator_cov averages each kernel
// over the area of every angular bin, using Gauss-Legendre panels and the
// Jacobi-polynomial recursion, and builds all four estimators together.
// The notebook cube exposes probe, angular bin and multipole as separate
// axes: a dimensionless arma::Cube [4,nbin,ell_max+1] for xi+, xi-,
// gamma_t, w and ell = 0..ell_max; spin rows vanish for ell < 2. These
// operators act on unit-normalized observed-shear spectra; operators_cov.c
// gives the conversion of each source leg from the core C_ell convention.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Mode-weighted Fourier band operator.
//
// Each integer multipole carries 2 ell+1 modes, so an unbiased band average
// is C_band = sum_(ell=first)^last (2 ell+1) C_ell / N_band with
// N_band = (last+1)^2 - first^2. bandpower_operator_cov writes these
// weights; the result is a dimensionless arma::Mat [nband,nell] whose
// column c is the multipole ell_min+c, zero outside each band. Bounds are
// inclusive absolute multipoles; bands may overlap.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Pure real-space noise of two estimators in the same angular bin.
//
// White noise correlates only repeated catalogs. For w_AB and w_CD the C
// routine gaussian_noise_pair_cov returns
//
//   (delta_AC delta_BD + delta_AD delta_BC) N_A N_B / pair_area.
//
// xi+ with xi+, or xi- with xi-, doubles this (two shear components);
// gamma_t with gamma_t keeps only the direct term delta_AC delta_BD; and
// different estimators, including xi+ with xi-, share no pure noise. N_A
// and N_B are in steradians and pair_area_sr2, the ordered-pair area of
// the bin, is in sr^2, so the returned covariance entry is dimensionless.
// ---------------------------------------------------------------------------
double covariance_noise_pair_cpp(
    const int probe_left,       // xi+, xi-, gamma_t, w: 0,1,2,3
    const int probe_right,      // right estimator in the same convention
    const arma::Col<int>& fields,// [4], A,B,C,D global catalog indices
    const arma::Col<double>& noise_ab,  // [2], white noise of A and B, sr
    const double pair_area_sr2 // ordered-pair area of the bin, sr^2
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
// that convention here, to a relative tolerance of 1e-8, prevents
// interpreting a normalized mask as raw.
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

// ---------------------------------------------------------------------------
// Ordered-pair area of each angular bin inside one common binary footprint.
//
// n_A n_B A_pair is the expected number of ordered pairs in a bin, which
// sets the pure real-space noise. With the raw mask spectrum C_L^W, the C
// routine mask_pair_area_cov evaluates
//
//   A_pair = 8 pi^2 Delta_x sum_L C_L^W K_bin,L,
//
// where Delta_x = cos(theta_low) - cos(theta_high) and K_bin,L are the w
// operator rows of covariance_realspace_operator_cpp with ell_max =
// nmask-1, including (2L+1)/(4 pi). The mask monopole and dipole are kept:
// they describe the footprint, not the signal. The result is an arma::Col
// [nbin] in sr^2; edges_rad is an arma::Col [nbin+1] in radians.
// ---------------------------------------------------------------------------
arma::Col<double> covariance_mask_pair_area_cpp(
    const arma::Col<double>& edges_rad,    // angular-bin edges in radians
    const arma::Col<double>& mask_cl,      // raw footprint spectrum, L=0 onward
    const double area_sr,          // footprint area, sr
    const arma::Mat<double>& scalar_kernel // [nbin,nmask], w operator rows
  )
{
  angles_cov(edges_rad);
  raw_mask_cov(mask_cl, area_sr);
  matrix_cov(scalar_kernel, "scalar_kernel");
  if (scalar_kernel.n_rows != edges_rad.n_elem-1
      || scalar_kernel.n_cols != mask_cl.n_elem) {
    throw std::invalid_argument("scalar_kernel must have shape [nbin,nmask]");
  }

  // C reads one contiguous row of mask multipoles per angular bin.
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

// ---------------------------------------------------------------------------
// Long-mode background strength of each radial shell for SSC.
//
// The output describes the background power of a radial shell in the
// long-mode Limber approximation. It has units of length; the radial
// quadrature weight is applied later when the observable responses meet.
// The C routine ssc_mask_variance_cov sums the footprint's angular modes,
//
//   sigma_b^2(chi) = sum_L (2L+1) C_L^W P_lin((L+1/2)/f_K, a)
//                    / (area_sr^2 f_K^2),
//
// with mask_cl an arma::Col [nmask] of raw C_L^W from L=0, distance an
// arma::Col [nnode] of f_K, and power an arma::Mat [nnode,nmask] of P_lin
// at k=(L+1/2)/f_K and the shell's a. The result is an arma::Col [nnode]
// in the length unit of distance (c/H0), not a dimensionless variance.
// ---------------------------------------------------------------------------
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

  // C reads one contiguous row of mask multipoles per radial shell.
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

// ---------------------------------------------------------------------------
// Radial SSC response of projected spectra.
//
// Each row describes one observable at one multipole. Its shell response
// includes the change of matter clustering and, when present, the change
// in the catalog mean used to define observed galaxy density. With
// D = dP/d(delta_b), the C routine ssc_shell_response_cov returns
//
//   Phi(row,chi) = W_A W_B D((ell+1/2)/f_K)/f_K^2 - (U_A+U_B) C_AB(ell),
//
// so that delta C_AB = integral dchi Phi delta_b(chi). Inputs: distance
// arma::Col [nnode] (f_K, c/H0); signal arma::Col [nrow], the full C_AB
// of each row; pair_window, mean_window and power_response arma::Mat
// [nrow,nnode] in (c/H0)^-2, (c/H0)^-1 and (c/H0)^3. pair_window includes
// the spin convention of the spectra; mean_window is zero for a field not
// normalized by its catalog mean. The result is an arma::Mat [nrow,nnode]
// in (c/H0)^-1.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Halo-model mass moments of the cold-matter-plus-baryon (cb) field.
//
// The C routine halo_moments_cov integrates, on the supplied mass panels,
//
//   I_mu^beta(k1..k_mu; a) = integral dlnM (dn/dlnM) b_beta
//                            (M/rho_cb)^mu u(k1|M) ... u(k_mu|M),
//
// with b_0 = 1 and b_1 the linear halo bias; halo_cov.c derives it and
// cites its sources. I11 receives the low-mass completion that gives
// I11(k->0) = 1.
//
// Return the one-profile moment and the five pair moments as a matrix and
// a cube. A response slope may need I11 alone; pair_moments=false omits
// the pair integration and returns None as the second tuple member.
//
// Tuple: I11 arma::Mat [na,nk], dimensionless; moments arma::Cube
// [5,na,nk(nk+1)/2] with roles I02(K,Q), I12(K,Q), I13(K,Q,Q),
// I13(K,K,Q) and I04(K,K,Q,Q), in (c/H0)^3, ^3, ^6, ^6, ^9. Its last axis
// lists the k pairs (0,0),(0,1),...,(1,1),... of each row of k.
// ---------------------------------------------------------------------------
py::tuple covariance_halo_moments_cpp(
    const arma::Col<double>& a,         // [na], scale factors
    const arma::Mat<double>& k,         // [na,nk], inverse c/H0
    const arma::Col<double>& lnm_edges, // ln(M/[Msun/h]) mass-panel edges
    const int nquad,           // Gauss-Legendre mass nodes per panel
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

  // The C routine reads the linear power, growth, sigma(M) and NFW
  // profile tables: a must lie in their initialized range [a_min,1), k is
  // nonnegative (k=0 is the large-scale limit, u=1), and the mass panels
  // increase inside [halo_sigma_min, halo_m[RANGE_MAX]].
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
    if (lnm_edges(edge) < std::log(limits.halo_sigma_min)
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

  // make_tuple builds the Python tuple (I11, moments), with None in place
  // of the moments when only I11 was requested.
  if (!pair_moments) {
    return py::make_tuple(carma::mat_to_arr(i11), py::none());
  }
  return py::make_tuple(carma::mat_to_arr(i11), carma::cube_to_arr(moments));
}

// ---------------------------------------------------------------------------
// Matter power spectrum at one scale factor on a caller-chosen k grid.
//
// Read one vector or a matrix of physical wavenumbers at a shared a.
// Matrix rows are independent integration batches distributed by C over
// OpenMP workers; a vector is a single row, read in order by one worker.
// The C routine power_rows_cov calls the core readers: linear selects the
// linear total-matter p_lin(k,a), otherwise the configured nonlinear
// Pdelta(k,a). k is an arma::Mat in (c/H0)^-1; the result is an arma::Mat
// of the same shape in (c/H0)^3. The k grid belongs to the caller; no
// Ntable setting changes.
// ---------------------------------------------------------------------------
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

  // Workspace role 0 holds the wavenumbers and role 1 receives the power;
  // each row is contiguous, as the C reader requires.
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


// A column vector is one requested k grid. Transpose it into a one-row
// matrix, reuse the same validation and core batch as the matrix
// overload, then return that single row as an arma::Col [nk], (c/H0)^3.
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


// ---------------------------------------------------------------------------
// Linear power for a matrix of base-10 log wavenumbers plus one shift.
//
// The connected-covariance angle grid keeps the same log wavenumbers at
// every radial shell; only the scalar shift -log10(f_K) changes with the
// shell. This wrapper reaches linear_power_logk_rows_cov, which skips the
// per-sample log10 of the standard reader (spectra_cov.h documents the
// mechanism). The result is NOT bitwise the standard reader at
// k = 10^(log10k+shift): the shifted sum rounds differently in the last
// bits. log10k is an arma::Mat of finite values of either sign; the
// result has its shape, in (c/H0)^3.
// ---------------------------------------------------------------------------
arma::Mat<double> covariance_power_logk_cpp(
    const double a,                   // scale factor
    const arma::Mat<double>& log10k, // base-10 logs before the shift
    const double shift               // common addend to every sample
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

  // Workspace role 0 holds the log wavenumbers and role 1 receives the
  // power; each row is contiguous, as the C reader requires.
  arma::Mat<double> output(log10k.n_rows, log10k.n_cols);
  double*** work = (double***) malloc3d(2, log10k.n_rows, log10k.n_cols);
  for (arma::uword row=0; row<log10k.n_rows; row++) {
    for (arma::uword col=0; col<log10k.n_cols; col++) {
      work[0][row][col] = log10k(row, col);
    }
  }
  linear_power_logk_rows_cov(a, log10k.n_rows, log10k.n_cols, work[0],
      shift, work[1]);
  for (arma::uword row=0; row<log10k.n_rows; row++) {
    for (arma::uword col=0; col<log10k.n_cols; col++) {
      output(row, col) = work[1][row][col];
    }
  }
  free(work);
  return output;
}


// ---------------------------------------------------------------------------
// Tree-level averages fed by the log-domain reader, block by block.
//
// Bit-for-bit covariance_power_logk_cpp followed by
// covariance_tree_averages_cpp, but the npair x nangle power table never
// exists in full: tree_averages_logk_cov (spectra_cov.c, whose header
// carries the full argument) evaluates one even block of pairs at a time
// into a cache-resident buffer. log10s is the run-constant [npair,nangle]
// base-10 log of the internal momenta and shift is -log10(f_K); the
// result is an arma::Mat [3,npair] with rows AvgP, AvgB, AvgT.
// ---------------------------------------------------------------------------
arma::Mat<double> covariance_tree_averages_logk_cpp(
    const arma::Mat<double>& k,      // [2,npair], positive K and Q
    const arma::Mat<double>& pk,     // [2,npair], matching linear power
    const arma::Col<double>& corner, // [nangle], stable 1+cos(theta)
    const arma::Col<double>& weight, // [nangle], normalized dtheta/pi weights
    const double a,                   // scale factor of the shell
    const arma::Mat<double>& log10s, // [npair,nangle], logs before shift
    const double shift               // common addend to every sample
  )
{
  matrix_cov(k, "k");
  matrix_cov(pk, "pk");
  matrix_cov(log10s, "log10s");
  vector_cov(corner, "corner");
  vector_cov(weight, "weight");
  if (k.n_rows != 2
      || pk.n_rows != 2
      || pk.n_cols != k.n_cols
      || log10s.n_rows != k.n_cols
      || log10s.n_cols != corner.n_elem
      || corner.n_elem != weight.n_elem) {
    throw std::invalid_argument(
        "tree input pair and angle dimensions disagree");
  }
  if (!std::isfinite(a)
      || a < limits.a_min
      || a >= 1.0
      || !std::isfinite(shift)
      || cosmology.lnPL == nullptr) {
    throw std::invalid_argument("initialize power tables and use a_min<=a<1");
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

  // Contiguous copies for the C driver: pair inputs, the log table and
  // the three output rows. The copies are the notebook wrapper's known
  // exploration overhead; the production binding borrows instead.
  const arma::uword npair = k.n_cols;
  const arma::uword nangle = corner.n_elem;
  double** pair_inputs = (double**) malloc2d(4, npair);
  double** logs = (double**) malloc2d(npair, nangle);
  double** averages = (double**) malloc2d(3, npair);
  for (arma::uword pair=0; pair<npair; pair++) {
    pair_inputs[0][pair] = k(0, pair);
    pair_inputs[1][pair] = k(1, pair);
    pair_inputs[2][pair] = pk(0, pair);
    pair_inputs[3][pair] = pk(1, pair);
    for (arma::uword node=0; node<nangle; node++) {
      logs[pair][node] = log10s(pair, node);
    }
  }
  const double* k_rows[2] = {pair_inputs[0], pair_inputs[1]};
  const double* pk_rows[2] = {pair_inputs[2], pair_inputs[3]};
  tree_averages_logk_cov(npair, nangle, k_rows, pk_rows,
      corner.memptr(), weight.memptr(), a,
      (const double* const*) logs, shift, averages);

  arma::Mat<double> output(3, npair);
  for (int role=0; role<3; role++) {
    for (arma::uword pair=0; pair<npair; pair++) {
      output(role, pair) = averages[role][pair];
    }
  }
  free(pair_inputs);
  free(logs);
  free(averages);
  return output;
}

// ---------------------------------------------------------------------------
// Planar tree-level averages used by the halo trispectrum.
//
// For each pair of wavenumber magnitudes K and Q, the C routine
// tree_averages_cov averages over the angle theta between k and q, with
// weights for integral dtheta/pi, the tree-level quantities
// AvgP = <P_lin(|k+q|)>, AvgB and AvgT built from the F2 and F3 kernels.
// corner = 1+cos(theta) avoids cancellation near theta = pi.
// The angular rule and its power samples are supplied together. The
// arrays must describe the same K,Q pairs and the same angular nodes;
// otherwise the cancellations between perturbation diagrams are lost.
//
// Arrays: k and pk arma::Mat [2,npair] (K,Q in (c/H0)^-1; P_lin(K),
// P_lin(Q) in (c/H0)^3); corner and weight arma::Col [nangle]; ps
// arma::Mat [npair,nangle], P_lin(|k+q|). The result is an arma::Mat
// [3,npair] with rows AvgP, AvgB, AvgT in (c/H0)^3, ^6, ^9.
// ---------------------------------------------------------------------------
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

  // The weights integrate dtheta/pi over [0,pi], so they must sum to one
  // (to 1e-10). corner=0 (theta=pi) is excluded: |k+q| vanishes there
  // when K=Q, and the tree kernels divide by |k+q|^2.
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

// ---------------------------------------------------------------------------
// The five halo-model contributions to the angle-averaged trispectrum.
//
// Four density legs k,-k,q,-q can sit in one to four halos. The C routine
// halo_trispectrum_cov returns, at each point (K,Q,a),
//
//   T_1h = I04(K,K,Q,Q),
//   T_13 = 2 [P_K I11(K) I13(K,Q,Q) + P_Q I11(Q) I13(K,K,Q)],
//   T_22 = 2 I12(K,Q)^2 AvgP,
//   T_3h = 4 I12(K,Q) I11(K) I11(Q) AvgB,
//   T_4h = [I11(K) I11(Q)]^2 AvgT.
//
// Keep the five halo contributions separate in the returned array so
// callers can examine their scale dependence before projecting their sum.
// All supplied moments and powers must refer to the same density field.
// Inputs: pk and i11 arma::Mat [2,npoint] (K and Q rows), moments
// arma::Mat [5,npoint] in the roles of covariance_halo_moments_cpp, tree
// arma::Mat [3,npoint]. The result is an arma::Mat [5,npoint] with rows
// 1h, 2h(1+3), 2h(2+2), 3h, 4h, each in (c/H0)^9.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Halo-model power and its response to a background density mode.
//
// The C routine halo_response_cov evaluates, with P_2h = I11^2 P_lin,
//
//   P_halo = P_2h + I02,
//   D_halo = (growth_coefficient - dilation_coefficient * slope) P_2h + I12,
//
// where slope = dlnP_X/dlnk of the spectrum the caller chose. fractional
// returns D = (D_halo/P_halo) P_target instead of D_halo.
// The two coefficients and the differentiated spectrum are explicit:
// the Python workflow selects a response prescription, not the binding.
// Row 0 returns halo power; row 1 returns its dimensional response or
// the fractional response transferred to the supplied target power.
//
// inputs is an arma::Mat [6,npoint] with rows P_lin, P_target, I11,
// I02(k,k), I12(k,k) and slope. The result is an arma::Mat [2,npoint];
// powers and responses are in (c/H0)^3.
// ---------------------------------------------------------------------------
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

  // halo_response_cov requires a positive P_halo = I11^2 P_lin + I02 (the
  // fractional response divides by it), so check that here first.
  for (arma::uword point=0; point<inputs.n_cols; point++) {
    const double i11 = inputs(2, point);
    const double phalo = i11*i11*(inputs(0, point))
                         +inputs(3, point);
    if (!std::isfinite(phalo)
        || phalo <= 0.0) {
      throw std::invalid_argument("supplied moments must give positive halo P");
    }
  }

  // Workspace plane 0 holds the six input rows; rows 0 and 1 of plane 1
  // receive P_halo and D.
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
