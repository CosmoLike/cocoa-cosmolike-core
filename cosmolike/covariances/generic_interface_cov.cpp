#include <carma.h>
#include "production_interface_cov.hpp"
#include "generic_interface_cov.hpp"
#include "covariance_wrapper_cov.hpp"
#include "notebook_bindings_cov.hpp"

namespace py = pybind11;
namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Register the two covariance layers on a project's Python module.
//
// 1. bind_covariance_production creates module.covariance, the production
//    layer used by command-line runs: it borrows C-contiguous NumPy inputs
//    and returns owned NumPy outputs (production_interface_cov.cpp and the
//    files it registers).
// 2. bind_covariance_components and bind_covariance_wrappers
//    (python_components_cov.cpp) register the notebook layer directly on
//    module, with the same function names. Each notebook binding checks
//    the rank of its array inputs and copies them, with notebook_input_cov,
//    from NumPy arrays of any numeric dtype or memory layout into owning
//    Armadillo containers. It then calls a *_cpp wrapper; CARMA exports
//    the Armadillo results as NumPy arrays.
// 3. The notebook spectrum builder is registered below: ell and a_edges
//    become arma::Col<double> copies, and covariance_limber_spectra_cpp
//    (components_wrapper_cov.cpp) returns the same dict as the production
//    function, with spectra[nell,nfield,nfield], b_spectra or None,
//    geometry[4,nnode], windows[3,nfield,nnode], nlens and nsource.
//
// Both layers call the same C routines, so their numerical results and
// threading agree: C routines that read lazy core tables warm them on the
// calling thread, then run their own OpenMP loops. The copies make
// notebook inputs independent of the caller's arrays, and no output
// aliases an input.
// ---------------------------------------------------------------------------
void bind_covariance(py::module_& module)
{
  bind_covariance_production(module);

  bind_covariance_components(module);
  bind_covariance_wrappers(module);

  // Notebook covariance_limber_spectra: the lambda converts the Python
  // objects and forwards every scalar unchanged. The docstring and the
  // defaults repeat those of the production function, and the second
  // name covariance_spectra refers to the same function object.
  module.def("covariance_limber_spectra", [](
          const py::object& ell,
          const py::object& a_edges,
          const int nquad,
          const int nwindow,
          const bool include_ia,
          const bool include_rsd,
          const bool linear,
          const int nonlimber_lmax,
          const int nonlimber_nchi,
          const double nonlimber_chi_min) {
        const arma::Col<double> ell_input =
            notebook_input_cov<arma::Col<double>>(ell, 1);
        const arma::Col<double> a_edges_input =
            notebook_input_cov<arma::Col<double>>(a_edges, 1);
        return covariance_limber_spectra_cpp(
            ell_input,
            a_edges_input, nquad, nwindow, include_ia, include_rsd, linear,
            nonlimber_lmax, nonlimber_nchi, nonlimber_chi_min);
      },
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
      py::arg("ell"),
      py::arg("a_edges"),
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
