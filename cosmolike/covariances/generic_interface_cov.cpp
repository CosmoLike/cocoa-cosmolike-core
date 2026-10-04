#include <carma.h>
#include "generic_interface_cov.hpp"
#include "covariance_wrapper_cov.hpp"
#include "notebook_bindings_cov.hpp"

namespace py = pybind11;
namespace cosmolike_interface {

void bind_covariance(py::module_& module)
{
  bind_covariance_components(module);
  bind_covariance_wrappers(module);

  module.def("covariance_limber_spectra", [](
          const py::object& ell,
          const py::object& a_edges,
          const int nquad,
          const int nwindow,
          const bool include_ia,
          const bool include_rsd,
          const bool linear) {
        const arma::Col<double> ell_input =
            notebook_input_cov<arma::Col<double>>(ell, 1);
        const arma::Col<double> a_edges_input =
            notebook_input_cov<arma::Col<double>>(a_edges, 1);
        return covariance_limber_spectra_cpp(
            ell_input,
            a_edges_input, nquad, nwindow, include_ia, include_rsd, linear);
      },
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
      py::arg("ell"),
      py::arg("a_edges"),
      py::arg("nquad"),
      py::arg("nwindow") = 4097,
      py::arg("include_ia") = true,
      py::arg("include_rsd") = true,
      py::arg("linear") = false);
}
}
