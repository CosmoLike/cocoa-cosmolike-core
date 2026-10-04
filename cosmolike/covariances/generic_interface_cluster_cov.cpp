#include <cmath>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include "generic_interface_cluster_cov.hpp"
#include "counts_cluster_cov.h"

namespace py = pybind11;

namespace cosmolike_interface {

using cluster_cov_array = py::array_t<double, py::array::c_style>;

// ---------------------------------------------------------------------------
// Expose count-shell quantities without assigning a mass-selection model.
//
// NumPy supplies selected abundances and their background-density
// derivatives. C converts them to counts per radial distance, using the
// shell volume. Shape and value checks precede C, so a malformed notebook
// input raises a Python exception instead of stopping the Python process.
// Returned arrays own their data and remain valid after subsequent calls.
// No likelihood, nuisance parameter or cosmology state is read or changed.
// ---------------------------------------------------------------------------
static py::dict covariance_counts_shell(
    const cluster_cov_array& distance, // [nnode], transverse distances
    const cluster_cov_array& density,  // [ncount,nnode], selected n_i
    const cluster_cov_array& derivative, // matching dn_i/d(delta_b)
    const double area_sr               // survey solid angle
  )
{
  if (distance.ndim() != 1
      || distance.size() < 1
      || density.ndim() != 2
      || derivative.ndim() != 2
      || density.shape(0) < 1
      || density.shape(1) != distance.size()
      || derivative.shape(0) != density.shape(0)
      || derivative.shape(1) != density.shape(1)) {
    throw std::invalid_argument(
        "distance must be nonempty [nnode]; density and derivative "
        "must both have shape [ncount,nnode], with ncount > 0");
  }
  if (!std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    throw std::invalid_argument("area_sr must be finite and in (0,4*pi]");
  }

  // A positive distance gives a physical shell volume. Zero abundances
  // are allowed where a bin's selection vanishes; signed responses permit
  // a selection model whose environmental change reduces the abundance.
  for (py::ssize_t node=0; node<distance.size(); node++) {
    if (!std::isfinite(distance.data()[node])
        || distance.data()[node] <= 0.0) {
      throw std::invalid_argument("distance must be finite and positive");
    }
  }
  for (py::ssize_t entry=0; entry<density.size(); entry++) {
    if (!std::isfinite(density.data()[entry])
        || density.data()[entry] < 0.0
        || !std::isfinite(derivative.data()[entry])) {
      throw std::invalid_argument(
          "density must be finite and nonnegative; derivative must "
          "be finite and may have either sign");
    }
  }

  // The pointer vectors describe NumPy's rows without copying values.
  // Separate output arrays ensure that C never overwrites an input.
  const py::ssize_t ncount = density.shape(0);
  const py::ssize_t nnode = distance.size();
  cluster_cov_array shell({ncount, nnode});
  cluster_cov_array response({ncount, nnode});
  std::vector<const double*> density_rows(ncount);
  std::vector<const double*> derivative_rows(ncount);
  std::vector<double*> shell_rows(ncount);
  std::vector<double*> response_rows(ncount);

  for (py::ssize_t bin=0; bin<ncount; bin++) {
    density_rows[bin] = density.data(bin, 0);
    derivative_rows[bin] = derivative.data(bin, 0);
    shell_rows[bin] = shell.mutable_data(bin, 0);
    response_rows[bin] = response.mutable_data(bin, 0);
  }

  counts_shell_cluster_cov(ncount, nnode, area_sr, distance.data(),
      density_rows.data(), derivative_rows.data(), shell_rows.data(),
      response_rows.data());

  py::dict result;
  result["shell_density"] = shell;
  result["shell_response"] = response;
  return result;
}


void bind_covariance_cluster(py::module_& module)
{
  module.def("covariance_counts_shell", &covariance_counts_shell,
      R"doc(Convert selected abundances into count densities and responses.

Arguments:
    distance: float64 [nnode], positive f_K in one chosen length unit L.
    density: float64 [ncount,nnode], selected abundance in L^-3,
        including richness probability, completeness and redshift selection.
    derivative: same shape, d(density)/d(delta_b), in L^-3. For a fixed
        selection this is the selected, bias-weighted halo abundance.
    area_sr: angular area, in steradians, inside (0,4*pi].
Returns:
    Dict with owned shell_density and shell_response arrays, both
    [ncount,nnode] in L^-1. Neither contains radial quadrature weights.
    Integrating shell_density over dchi gives the expected counts.
    Use shell_response with the same background kernel and two-point
    responses to assemble count-count and count-two-point SSC.
Scope:
    No mass-selection model, Poisson noise, non-SSC cross term or radial
    integration is added. No global interface state is changed.
)doc",
      py::arg("distance").noconvert(),
      py::arg("density").noconvert(),
      py::arg("derivative").noconvert(),
      py::arg("area_sr"));
}

} // namespace cosmolike_interface
