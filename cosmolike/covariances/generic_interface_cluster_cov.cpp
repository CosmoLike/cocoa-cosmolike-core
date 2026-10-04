#include <cmath>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include "generic_interface_cluster_cov.hpp"
#include "counts_cluster_cov.h"
#include "spectra_cluster_cov.h"

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


// ---------------------------------------------------------------------------
// Project supplied cluster ingredients without importing a survey model.
//
// All arrays share the same radial nodes. The caller obtains normalized
// cluster windows, selected biases and one-halo profiles from its model.
// This wrapper checks shapes before passing row pointers to C; returned
// cross and auto spectra own their data. The galaxy/shear block is supplied
// separately by the ordinary covariance spectrum builder.
// ---------------------------------------------------------------------------
static py::dict covariance_cluster_spectra(
    const cluster_cov_array& ell,       // multipole samples
    const cluster_cov_array& distance,  // common transverse distances
    const cluster_cov_array& dchi,      // radial integration weights
    const cluster_cov_array& base,      // galaxy and lensing windows
    const cluster_cov_array& window,    // normalized cluster windows
    const cluster_cov_array& bias,      // selected cluster bias
    const cluster_cov_array& power,     // nonlinear matter power
    const cluster_cov_array& profile,   // selected one-halo spectra
    const py::array_t<int, py::array::c_style>& richness, // profile map
    const int nlens                     // leading galaxy fields in base
  )
{
  if (ell.ndim() != 1
      || ell.size() < 1
      || distance.ndim() != 1
      || distance.size() < 1
      || dchi.ndim() != 1
      || dchi.size() != distance.size()
      || base.ndim() != 2
      || window.ndim() != 2
      || bias.ndim() != 2
      || power.ndim() != 2
      || profile.ndim() != 3
      || richness.ndim() != 1) {
    throw std::invalid_argument(
        "ell/distance/dchi/richness must be vectors; base/window/bias/"
        "power matrices; profile a 3D array; ell/distance nonempty");
  }
  const py::ssize_t nell = ell.size();
  const py::ssize_t nnode = distance.size();
  const py::ssize_t nbase = base.shape(0);
  const py::ssize_t ncluster = window.shape(0);
  const py::ssize_t nrichness = profile.shape(0);
  if (nbase < 1
      || ncluster < 1
      || nrichness < 1
      || nlens < 0
      || nlens > nbase
      || base.shape(1) != nnode
      || window.shape(1) != nnode
      || bias.shape(0) != ncluster
      || bias.shape(1) != nnode
      || power.shape(0) != nell
      || power.shape(1) != nnode
      || profile.shape(1) != nell
      || profile.shape(2) != nnode
      || richness.size() != ncluster) {
    throw std::invalid_argument(
        "need base[nbase,nnode], window/bias[ncluster,nnode], "
        "power[nell,nnode], profile[nrichness,nell,nnode], "
        "richness[ncluster], positive field counts and 0<=nlens<=nbase");
  }

  // Reject nonfinite values before any output allocation or C operation.
  // Signed windows and profiles can be supplied; they are never clipped.
  for (const auto* input : {&ell, &distance, &dchi, &base, &window,
                            &bias, &power, &profile}) {
    for (py::ssize_t entry=0; entry<input->size(); entry++) {
      if (!std::isfinite(input->data()[entry])) {
        throw std::invalid_argument("cluster spectrum inputs must be finite");
      }
    }
  }
  for (py::ssize_t index=0; index<nell; index++) {
    if (ell.data()[index] < 2.0) {
      throw std::invalid_argument("cluster spectrum ell must be >= 2");
    }
  }
  for (py::ssize_t node=0; node<nnode; node++) {
    if (distance.data()[node] <= 0.0
        || dchi.data()[node] <= 0.0) {
      throw std::invalid_argument("distance and dchi must be positive");
    }
  }
  for (py::ssize_t field=0; field<ncluster; field++) {
    if (richness.data()[field] < 0
        || richness.data()[field] >= nrichness) {
      throw std::invalid_argument("richness indices must be in [0,nrichness)");
    }
  }

  // Row-pointer vectors describe the supplied contiguous arrays without
  // copying their values. They remain alive until C finishes reading them.
  std::vector<const double*> base_rows(nbase);
  std::vector<const double*> window_rows(ncluster);
  std::vector<const double*> bias_rows(ncluster);
  std::vector<const double*> power_rows(nell);
  std::vector<const double*> profile_rows(nrichness*nell);
  std::vector<const double* const*> profile_planes(nrichness);
  for (py::ssize_t field=0; field<nbase; field++) {
    base_rows[field] = base.data(field, 0);
  }
  for (py::ssize_t field=0; field<ncluster; field++) {
    window_rows[field] = window.data(field, 0);
    bias_rows[field] = bias.data(field, 0);
  }
  for (py::ssize_t index=0; index<nell; index++) {
    power_rows[index] = power.data(index, 0);
  }
  for (py::ssize_t bin=0; bin<nrichness; bin++) {
    profile_planes[bin] = profile_rows.data()+bin*nell;
    for (py::ssize_t index=0; index<nell; index++) {
      profile_rows[bin*nell+index] = profile.data(bin, index, 0);
    }
  }

  const py::ssize_t ncross = ncluster*nbase;
  const py::ssize_t npair = ncross+ncluster*(ncluster+1)/2;
  cluster_cov_array triangular({npair, nell});
  std::vector<double*> output_rows(npair);
  for (py::ssize_t pair=0; pair<npair; pair++) {
    output_rows[pair] = triangular.mutable_data(pair, 0);
  }
  limber_cluster_cov(nell, ell.data(), nnode, distance.data(), dchi.data(),
      nbase, nlens, base_rows.data(), ncluster, window_rows.data(),
      bias_rows.data(), power_rows.data(), profile_planes.data(),
      richness.data(), output_rows.data());

  // Python receives explicit field axes, not the C triangular row map.
  // Copy both auto-spectrum triangles from the same integrated value.
  cluster_cov_array cross({nell, ncluster, nbase});
  cluster_cov_array auto_spectra({nell, ncluster, ncluster});
  py::ssize_t pair = 0;
  for (py::ssize_t field=0; field<ncluster; field++) {
    for (py::ssize_t other=0; other<nbase; other++) {
      for (py::ssize_t index=0; index<nell; index++) {
        *cross.mutable_data(index, field, other) = output_rows[pair][index];
      }
      pair++;
    }
  }
  for (py::ssize_t field=0; field<ncluster; field++) {
    for (py::ssize_t other=field; other<ncluster; other++) {
      for (py::ssize_t index=0; index<nell; index++) {
        const double value = output_rows[pair][index];
        *auto_spectra.mutable_data(index, field, other) = value;
        *auto_spectra.mutable_data(index, other, field) = value;
      }
      pair++;
    }
  }
  py::dict result;
  result["cluster_base"] = cross;
  result["cluster_cluster"] = auto_spectra;
  return result;
}


void bind_covariance_cluster(py::module_& module)
{
  module.def("covariance_cluster_spectra", &covariance_cluster_spectra,
      R"doc(Project every cluster-galaxy, cluster-source and cluster pair.

Arguments (contiguous float64, except richness as int32):
    ell: [nell], multipoles >= 2.
    distance, dchi: [nnode], positive transverse distances/radial weights.
    base: [nbase,nnode], biased galaxy windows followed by lensing windows.
    window, bias: [ncluster,nnode], normalized q_c and selected bias b_c.
    power: [nell,nnode], nonlinear matter P at k=(ell+1/2)/f_K.
    profile: [nrichness,nell,nnode], selected P_cm^1h on the same grid.
    richness: [ncluster], index of each category's selected profile.
    nlens: number of leading galaxy fields in base, from 0 to nbase.
Units:
    One length unit L for distance/dchi; windows L^-1; power/profile L^3.
    Bias and the returned angular spectra are dimensionless.
Returns:
    Owned arrays cluster_base[nell,ncluster,nbase] and
    cluster_cluster[nell,ncluster,ncluster], the latter exactly symmetric.
Model:
    cc and cg use biased nonlinear matter power. Only cluster lensing
    adds the selected halo's own mass profile. Source spectra include the
    core harmonic spin factor. No noise, IA, magnification, RSD, non-Limber
    correction or estimator transform is included. No global state changes.
    Validate field positivity with catalog noise before Gaussian assembly.
)doc",
      py::arg("ell").noconvert(), py::arg("distance").noconvert(),
      py::arg("dchi").noconvert(), py::arg("base").noconvert(),
      py::arg("window").noconvert(), py::arg("bias").noconvert(),
      py::arg("power").noconvert(), py::arg("profile").noconvert(),
      py::arg("richness").noconvert(), py::arg("nlens"));

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
