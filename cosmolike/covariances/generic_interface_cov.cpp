#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include "generic_interface_cov.hpp"
#include "spectra_cov.h"
#include "cosmolike/IA.h"
#include "cosmolike/structs.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Expose covariance-owned spectra and their radial inputs as NumPy arrays.
//
// The Python boundary validates array shapes before handing raw pointers
// to C. The returned arrays own their memory; a later covariance call or
// cosmology change cannot alter an earlier result. The C snapshot is
// temporary and its unique_ptr releases it even if Python allocation fails.
// No Ntable field, data-vector mask, covariance or likelihood state changes.
//
// The C routine stores one row per unordered field pair. Python receives
// a full symmetric [ell][field][field] array, plus copies of the radial
// geometry and window tables so it can audit the integration inputs.
// ---------------------------------------------------------------------------
static py::dict covariance_limber_spectra(
    const py::array_t<double, py::array::c_style>& ell, // multipole samples
    const py::array_t<double, py::array::c_style>& a_edges, // radial panels
    const int nquad,        // Gaussian nodes per scale-factor panel
    const int nwindow,      // uniform-a lensing-efficiency samples
    const bool include_ia,  // include the signed NLA window
    const bool include_rsd, // include the lens redshift-distortion window
    const bool linear      // select linear rather than nonlinear matter P
  )
{
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
      && nuisance.IA_MODEL != IA_MODEL_NLA) {
    throw std::invalid_argument(
        "covariance_limber_spectra supports NLA; initialize IA model 0");
  }
  for (py::ssize_t index=0; index<ell.size(); index++) {
    if (!std::isfinite(ell.data()[index])
        || ell.data()[index] < 1.0) {
      throw std::invalid_argument("ell must contain finite values >= 1");
    }
  }
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

  // unique_ptr is the sole owner of this temporary C snapshot. Its
  // specified cleanup function, free_radial_cov, runs when the owner
  // leaves scope, including if a later Python array allocation fails.
  std::unique_ptr<radial_cov, decltype(&free_radial_cov)> radial(
      radial_inputs_cov(a_edges.size()-1, a_edges.data(), nquad,
                        nwindow, include_ia), &free_radial_cov);

  limber_spectra_cov(radial.get(), nell, ell.data(), linear, include_rsd,
                     rows.data());

  // --- 3. EXPAND THE FIELD-PAIR TRIANGLE FOR PYTHON ---

  // Store both triangles from the same computed number. This makes the
  // returned field matrix exactly symmetric, independent of thread count.
  py::array_t<double> spectra({nell, nfield, nfield});
  int pair = 0;

  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      for (int node=0; node<nell; node++) {
        *spectra.mutable_data(node, first, second) = rows[pair][node];
        *spectra.mutable_data(node, second, first) = rows[pair][node];
      }
      pair++;
    }
  }

  // --- 4. COPY THE INTEGRATION INPUTS BEFORE RELEASING THE SNAPSHOT ---

  // Geometry uses roles a, chi, f_K, dchi weight. Windows use density,
  // lensing/magnification and signed NLA roles, then field and radial node.
  py::array_t<double> geometry({4, radial->nnode});
  py::array_t<double> windows({3, nfield, radial->nnode});

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
  result["geometry"] = geometry;
  result["windows"] = windows;
  result["nlens"] = radial->nlens;
  result["nsource"] = radial->nsource;
  return result;
}


void bind_covariance(py::module_& module)
{
  module.def("covariance_limber_spectra", &covariance_limber_spectra,
      R"doc(Build all lens/source Limber spectra on common radial nodes.

Arguments:
    ell: float64 1D multipoles >= 1. Small ell are still Limber here.
    a_edges: float64 increasing panel edges inside (0,1). Include the
        full source/lens support and the foreground to a close to 1.
    nquad: nodes per panel from 64,96,128,256,512,1024.
    nwindow: uniform-a nodes for covariance-owned lensing efficiencies.
    include_ia: include NLA in the source windows; TATT is unsupported.
    include_rsd: use the same lens RSD window in every spectrum.
    linear: use linear total-matter P instead of the current Pdelta mode.

Returns a dict of owned arrays:
    spectra [nell,nfield,nfield], dimensionless, core C_ell convention;
    geometry [4,nnode]: a, chi, f_K, positive dchi quadrature weights;
    windows [3,nfield,nnode]: density, lensing, signed NLA contributions;
    nlens, nsource: field counts. Lenses precede sources in nfield.
Distances use c/H0 and windows its inverse. Spectra contain no noise,
mask or pair exclusions. Bias is linear. The radial panels and node count
belong to covariance and do not modify the data-vector accuracy settings.
This computes spectra, not a full covariance or a non-Limber correction.
)doc",
      py::arg("ell").noconvert(),
      py::arg("a_edges").noconvert(),
      py::arg("nquad"),
      py::arg("nwindow") = 4097,
      py::arg("include_ia") = true,
      py::arg("include_rsd") = true,
      py::arg("linear") = false);
}
}
