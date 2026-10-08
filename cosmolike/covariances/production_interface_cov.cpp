#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include "production_interface_cov.hpp"
#include "spectra_cov.h"
#include "nonlimber_cov.h"
#include "ia_cov.h"
#include "cosmolike/IA.h"
#include "cosmolike/basics.h"
#include "cosmolike/structs.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Expose covariance-owned spectra and their radial inputs as NumPy arrays.
//
// PHYSICAL QUANTITY
// A Gaussian covariance pairs the fields of two measured spectra AB and
// CD as AC*BD + AD*BC. It therefore needs the angular power spectrum of
// every pair of catalogs, including pairs absent from the data vector.
// In the Limber approximation each one is a radial integral,
//
//   C_AB(ell) = integral dchi W_A W_B P((ell+1/2)/f_K, a) / f_K^2,
//
// evaluated here on one common Gauss-Legendre rule over the supplied
// scale-factor panels. The fields are the lens samples followed by the
// source samples of the initialized redshift setup.
//
// ARRAYS CROSSING THE BOUNDARY
// Inputs, borrowed C-contiguous float64 (noconvert rejects a copy):
//   ell[nell]            finite multipoles >= 1
//   a_edges[npanel+1]    scale-factor panel edges, increasing in (0,1)
// Outputs, a dict of newly allocated arrays owned by Python:
//   spectra[nell,nfield,nfield]    dimensionless signal spectra (E modes
//                                  for shear) in the core C_ell
//                                  convention, without noise
//   b_spectra                      TATT B modes with the same axes, or
//                                  None for NLA or disabled IA
//   geometry[4,nnode]              a, chi, f_K (c/H0), dchi weight (c/H0)
//   windows[3,nfield,nnode]        density, lensing/magnification and
//                                  signed NLA windows, in (c/H0)^-1
//   nlens, nsource                 field counts; nfield is their sum
// Here nnode = npanel*nquad is the number of common radial nodes.
//
// VALIDATION BEFORE C
// Array ranks and sizes; finite ell >= 1; a_edges strictly increasing
// inside (0,1); a tabulated nquad and nwindow >= 2; distance, growth and
// linear-power tables set (nonlinear too unless linear); at least one
// lens and one source bin; IA model NLA or TATT when include_ia; lmax 0
// or >= 2, nchi=2^n+1 >= 65 and finite chi_min > 0. With lmax > 0 the
// ell in [2,lmax] must be integers, with no RSD and massless neutrinos.
// TATT spectra (include_ia with the TATT model) also require no RSD.
// Flat geometry is not checked here: radial_inputs_cov stops the process
// for a non-flat cosmology.
//
// C WORK AND THREADS
// radial_inputs_cov samples the windows, limber_spectra_cov integrates
// all field pairs, tatt_spectra_cov adds the TATT terms beyond NLA, and
// apply_nonlimber_cov replaces the linear Limber part of gg/gs spectra.
// Each routine prepares its lazy core tables on the calling thread before
// its own OpenMP loops: radial_inputs_cov the distance, window and NLA
// readers, limber_spectra_cov the power and RSD tables, tatt_spectra_cov
// FAST-PT, apply_nonlimber_cov its logarithmic snapshot and power reader.
// Call this binding outside any OpenMP region, as Python does. The C++
// conversion itself calls no BLAS routine.
//
// OWNERSHIP AND STATE
// The C snapshot is temporary and its unique_ptr releases it even if
// Python allocation fails. No Ntable field, data-vector mask, covariance
// or likelihood state changes; only lazily built core tables are filled.
//
// The C routine stores one row per unordered field pair. Python receives
// a full symmetric [ell][field][field] array, plus copies of the radial
// geometry and window tables so it can audit the integration inputs.
// ---------------------------------------------------------------------------
static py::dict covariance_limber_spectra(
    const py::array_t<double, py::array::c_style>& ell, // [nell] multipoles
    const py::array_t<double, py::array::c_style>& a_edges, // [npanel+1], a
    const int nquad,        // Gaussian nodes per scale-factor panel
    const int nwindow,      // uniform-a lensing-efficiency samples
    const bool include_ia,  // NLA window; with TATT also its E and B terms
    const bool include_rsd, // include the lens redshift-distortion window
    const bool linear,     // 1: linear p_lin(k,a); 0: run-mode Pdelta(k,a)
    const int nonlimber_lmax, // gg/gs correction through this ell; 0 disables
    const int nonlimber_nchi, // logarithmic radial samples, 2^n+1
    const double nonlimber_chi_min // positive near distance in c/H0
  )
{
  // Check the inputs and initialized state before any allocation or C
  // call. A C fatal check would stop the whole Python process; an
  // exception here only reports the mistake to the caller.
  if (ell.ndim() != 1
      || ell.size() < 1
      || a_edges.ndim() != 1
      || a_edges.size() < 2) {
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

  // C accepts only finite ell >= 1, where the shear spin factor
  // sqrt[(ell-1)ell(ell+1)(ell+2)] is real; check every entry here.
  for (py::ssize_t index=0; index<ell.size(); index++) {
    if (!std::isfinite(ell.data()[index])
        || ell.data()[index] < 1.0) {
      throw std::invalid_argument("ell must contain finite values >= 1");
    }
  }

  // Each consecutive pair of edges is one Gauss-Legendre panel in scale
  // factor. Strictly increasing edges inside (0,1) give panels of positive
  // width between the far boundary and the observer at a=1.
  for (py::ssize_t edge=0; edge<a_edges.size(); edge++) {
    if (!std::isfinite(a_edges.data()[edge])
        || a_edges.data()[edge] <= 0.0
        || a_edges.data()[edge] >= 1.0
        || (edge > 0
            && a_edges.data()[edge] <= a_edges.data()[edge-1])) {
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

  // The hybrid non-Limber correction starts at ell=2, so lmax is either 0
  // (disabled) or at least 2. FFTLog needs a power-of-two number of
  // log-distance intervals, nchi-1 = 2^n >= 64: for m=nchi-1 the bit test
  // m&(m-1) is zero exactly when m is a power of two.
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
    if (include_rsd
        || cosmology.Omega_nu != 0.0) {
      throw std::invalid_argument(
          "non-Limber covariance currently requires no RSD and mnu=0");
    }

    // The correction is tabulated at integer multipoles 2..lmax and C
    // reads the entry (int) ell - 2. A fractional ell in that range would
    // silently receive the correction of a neighboring integer.
    for (int index=0; index<ell.size(); index++) {
      const double value = ell.data()[index];
      if (value <= nonlimber_lmax
          && value >= 2.0
          && value != std::floor(value)) {
        throw std::invalid_argument(
            "non-Limber correction requires integer ell below its cutoff");
      }
    }
  }

  if (include_ia
      && nuisance.IA_MODEL == IA_MODEL_TATT
      && include_rsd) {
    throw std::invalid_argument("Gaussian TATT currently requires no RSD");
  }

  // --- 1. ALLOCATE ONE SPECTRUM PER UNORDERED FIELD PAIR ---

  const int nfield = redshift.clustering_nbin+redshift.shear_nbin;
  const int npair = nfield*(nfield+1)/2;
  const int nell = ell.size();

  // NumPy owns the values; rows only points to their starting addresses
  // for the C interface. The array stays alive throughout the calculation.
  py::array_t<double> triangular({npair, nell});
  std::vector<double*> rows(npair);

  for (int pair=0; pair<npair; pair++) {
    rows[pair] = triangular.mutable_data(pair, 0);
  }

  // --- 2. BUILD THE RADIAL SNAPSHOT AND INTEGRATE ALL SPECTRA ---

  // radial_inputs_cov samples a, chi, f_K, the dchi weight and every field
  // window on npanel*nquad Gauss-Legendre nodes. limber_spectra_cov then
  // integrates C_AB for all pairs into rows, in the i-major triangular
  // order (0,0),(0,1),...,(1,1),... of the field indices.
  //
  // unique_ptr is the sole owner of this temporary C snapshot. Its
  // specified cleanup function, free_radial_cov, runs when the owner
  // leaves scope, including if a later Python array allocation fails.
  std::unique_ptr<radial_cov, decltype(&free_radial_cov)> radial(
      radial_inputs_cov(a_edges.size()-1, a_edges.data(), nquad,
                        nwindow, include_ia), &free_radial_cov);

  limber_spectra_cov(radial.get(), nell, ell.data(), linear, include_rsd,
                     rows.data());

  // TATT: add the E-mode terms beyond NLA to rows in place, and fill the
  // B modes from zero. Parity gives B only to source-source pairs; other
  // rows stay zero. b_triangular is one house malloc2d block [npair][nell]
  // (row pointers and values together), so a single free releases it.
  double** b_triangular = nullptr;
  if (include_ia
      && nuisance.IA_MODEL == IA_MODEL_TATT) {
    b_triangular = (double**) malloc2d(npair, nell);
    tatt_spectra_cov(radial.get(), nell, ell.data(), rows.data(), b_triangular);
  }

  // Non-Limber: add exact-minus-matched separable linear spectra to every
  // row containing a lens field, at integer ell in [2,lmax]. Source-source
  // rows stay Limber. a_edges[0] is the far radial boundary of FFTLog.
  if (nonlimber_lmax > 0) {
    apply_nonlimber_cov(radial.get(), a_edges.data()[0], nwindow, include_ia,
        nonlimber_lmax, nonlimber_nchi, nonlimber_chi_min,
        nell, ell.data(), rows.data());
  }

  // --- 3. EXPAND THE FIELD-PAIR TRIANGLE FOR PYTHON ---

  // Store both triangles from the same computed number. This makes the
  // returned field matrix exactly symmetric, independent of thread count.
  // A default-constructed b_spectra has size zero; it stays empty without
  // TATT and is returned as None below.
  py::array_t<double> spectra({nell, nfield, nfield});
  py::array_t<double> b_spectra;
  if (b_triangular != nullptr) {
    b_spectra = py::array_t<double>({nell, nfield, nfield});
  }
  int pair = 0;

  // Visit the field pairs in the same i-major order that C used for its
  // rows, so rows[pair] holds the spectrum of (first,second). One
  // iteration of the inner loop copies one multipole of that pair into
  // the [ell][field][field] layout, at both (first,second) and its mirror.
  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      for (int node=0; node<nell; node++) {
        *spectra.mutable_data(node, first, second) = rows[pair][node];
        *spectra.mutable_data(node, second, first) = rows[pair][node];
        if (b_triangular != nullptr) {
          *b_spectra.mutable_data(node, first, second) = b_triangular[pair][node];
          *b_spectra.mutable_data(node, second, first) = b_triangular[pair][node];
        }
      }
      pair++;
    }
  }

  free(b_triangular);

  // --- 4. COPY THE INTEGRATION INPUTS BEFORE RELEASING THE SNAPSHOT ---

  // Geometry uses roles a, chi, f_K, dchi weight. Windows use density,
  // lensing/magnification and signed NLA roles, then field and radial node.
  // Distances and the dchi weight are in c/H0; windows are in (c/H0)^-1.
  // The signed NLA role is zero for lenses and whenever include_ia is off.
  py::array_t<double> geometry({4, radial->nnode});
  py::array_t<double> windows({3, nfield, radial->nnode});

  // Each iteration copies every geometry and window role at one radial
  // node from the C snapshot, which free_radial_cov releases at return.
  for (int node=0; node<radial->nnode; node++) {
    for (int role=0; role<4; role++) {
      *geometry.mutable_data(role, node) = radial->geometry[role][node];
    }
    for (int role=0; role<3; role++) {
      for (int field=0; field<nfield; field++) {
        *windows.mutable_data(role, field, node) =
            radial->window[role][field][node];
      }
    }
  }

  // All returned arrays now own their values independently of the C state.
  py::dict result;
  result["spectra"] = spectra;
  result["b_spectra"] = py::none();
  if (b_spectra.size() > 0) {
    result["b_spectra"] = b_spectra;
  }
  result["geometry"] = geometry;
  result["windows"] = windows;
  result["nlens"] = radial->nlens;
  result["nsource"] = radial->nsource;
  return result;
}


// ---------------------------------------------------------------------------
// Create the production submodule parent.covariance and fill it.
//
// The component and matrix bindings come from components_interface_cov.cpp
// and matrix_interface_cov.cpp. The spectrum builder above is registered
// here under covariance_limber_spectra, with covariance_spectra as a second
// name for the same Python function object. Every array argument is
// noconvert(): only C-contiguous float64 arrays (int32 for index maps)
// are accepted, and they are borrowed rather than copied. bind_covariance
// (generic_interface_cov.cpp) calls this function first, then registers
// the Armadillo notebook functions with the same names on parent itself.
// ---------------------------------------------------------------------------
void bind_covariance_production(py::module_& parent)
{
  py::module_ module = parent.def_submodule("covariance",
      "Contiguous-array production bindings to the shared covariance C code.");
  bind_production_components_cov(module);
  bind_production_matrices_cov(module);

  module.def("covariance_limber_spectra", &covariance_limber_spectra,
      R"doc(Build all lens/source spectra, optionally correcting Gaussian gg/gs.

Arguments:
    ell: float64 1D multipoles >= 1; integer below a non-Limber cutoff.
    a_edges: float64 increasing panel edges inside (0,1). Include the
        full source/lens support and the foreground to a close to 1.
    nquad: nodes per panel from 64,96,128,256,512,1024.
    nwindow: uniform-a nodes for covariance-owned lensing efficiencies.
    include_ia: include the configured NLA or TATT Gaussian spectra.
    include_rsd: use the same lens RSD window in every spectrum.
    linear: use linear total-matter P instead of the current Pdelta mode.
    nonlimber_lmax: 0 keeps Limber; >=2 corrects all gg/gs pairs through
        that integer multipole. Shear-shear remains Limber.
    nonlimber_nchi: 2^n+1 log-distance samples, default 4097.
    nonlimber_chi_min: positive near distance in c/H0, default 1e-6.
        The far edge is a_edges[0]. Refine support and resolution separately.
        Non-Limber currently requires massless neutrinos and no RSD.

Returns a dict of owned arrays:
    spectra [nell,nfield,nfield], dimensionless E, core C_ell convention;
    b_spectra: same axes for TATT B; None for NLA or disabled IA;
    geometry [4,nnode]: a, chi, f_K, positive dchi quadrature weights;
    windows [3,nfield,nnode]: density, lensing, signed NLA contributions;
    nlens, nsource: field counts. Lenses precede sources in nfield.
Distances use c/H0 and windows its inverse. Spectra contain no noise,
mask or pair exclusions. Bias is linear. The radial panels and node count
belong to covariance and do not modify the data-vector accuracy settings.
The non-Limber correction adds exact separable linear minus its matched
Limber approximation. Both use D(a)^2*P(k,1). Nonlinear residuals remain
Limber. covariance_spectra is the preferred name; covariance_limber_spectra
is retained for existing callers. This function does not compute SSC/cNG.
)doc",
      py::arg("ell").noconvert(),
      py::arg("a_edges").noconvert(),
      py::arg("nquad"),
      py::arg("nwindow") = 4097,
      py::arg("include_ia") = true,
      py::arg("include_rsd") = true,
      py::arg("linear") = false,
      py::arg("nonlimber_lmax") = 0,
      py::arg("nonlimber_nchi") = 4097,
      py::arg("nonlimber_chi_min") = 1.e-6);
  module.attr("covariance_spectra") = module.attr("covariance_limber_spectra");
}
}
