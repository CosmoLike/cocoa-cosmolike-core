#include <carma.h>
#include "generic_interface_cluster_cov.hpp"
#include "cluster_wrapper_cov.hpp"
#include "notebook_bindings_cov.hpp"

namespace py = pybind11;
namespace cosmolike_interface {

void bind_covariance_cluster(py::module_& module)
{
  module.def("covariance_cluster_halo_samples",
      [](
          const py::object& a,
          const py::object& k,
          const py::object& lnm,
          const py::object& dlnm) {
        const arma::Col<double> a_input =
            notebook_input_cov<arma::Col<double>>(a, 1);
        const arma::Mat<double> k_input =
            notebook_input_cov<arma::Mat<double>>(k, 2);
        const arma::Col<double> lnm_input =
            notebook_input_cov<arma::Col<double>>(lnm, 1);
        const arma::Col<double> dlnm_input =
            notebook_input_cov<arma::Col<double>>(dlnm, 1);
        return covariance_cluster_halo_samples_cpp(
            a_input,
            k_input, lnm_input, dlnm_input);
      },
      R"doc(Sample the initialized halo/richness model on covariance nodes.

Arguments (float64):
    a: [state], limits.a_min <= a < 1.
    k: [state,k], nonnegative wavenumbers in inverse c/H0.
    lnm: [mass], ln(M/[Msun/h]) inside the core sigma mass range.
    dlnm: [mass], positive quadrature weights for dlnM.
Returns:
    Owned weight[state,richness,mass], selected dn in (c/H0)^-3;
    bias[state,mass], dimensionless linear halo bias;
    profile[state,k,mass], (M/rho_m)*u_NFW in (c/H0)^3.
    Pass these arrays to covariance_cluster_moments for mass integration.
Scope:
    Requires initialized massless cosmology, NFW, lognormal richness
    selection and selection_model=0. The HMF amplitude follows the
    initialized cluster hmf_alpha_mode. No redshift-bin selection, low-mass
    completion, catalog normalization or environmental response is added.
    Core reader tables are warmed serially; model settings are not changed.
)doc",
      py::arg("a"), py::arg("k"),
      py::arg("lnm"), py::arg("dlnm"));

  module.def("covariance_cluster_moments", [](
          const py::object& weight,
          const py::object& bias,
          const py::object& profile) {
        const arma::Cube<double> weight_input =
            notebook_input_cov<arma::Cube<double>>(weight, 3);
        const arma::Mat<double> bias_input =
            notebook_input_cov<arma::Mat<double>>(bias, 2);
        const arma::Cube<double> profile_input =
            notebook_input_cov<arma::Cube<double>>(profile, 3);
        return covariance_cluster_moments_cpp(
            weight_input,
            bias_input, profile_input);
      },
      R"doc(Integrate selected halo moments with supplied mass quadrature.

Arguments (float64; one consistent length unit L):
    weight: [state,selection,mass], dlnM*(dn/dlnM)*S_i, in L^-3.
        S_i is the membership probability, included ONCE, even for a
        same-halo correlation with multiple cluster legs. Weight >= 0.
    bias: [state,mass], finite linear halo bias, dimensionless.
    profile: [state,k,mass], finite (M/rho)*u(k|M), in L^3.
Returns:
    density[state,selection]: n_i; biased_density[state,selection]:
        integral dn S_i b. Both have units L^-3.
    J01[state,selection,k] and J11[state,selection,k], dimensionless.
    J02[state,selection,kpair], J03_KKQ[state,selection,kpair] and
        J03_KQQ[state,selection,kpair], with units L^3,L^6,L^6.
        Each named array owns its values. Pair order is the k-grid's
        upper triangle: (0,0),(0,1),...,(1,1),... .
    J_beta_mu = integral dn S_i b_beta product(profile), b_0=1, b_1=b.
Scope:
    These are unnormalized ingredients, not a covariance. No mass function,
    low-mass completion, survey mask, shot noise or catalog normalization is
    chosen. The fixed-selection interpretation requires S_i not to respond
    to the background overdensity; an environmental derivative is separate.
    Distinct exclusive observed categories share no same-halo term, even
    when their true-mass distributions overlap. Their different-halo and
    super-sample correlations must still be computed.
)doc",
      py::arg("weight"), py::arg("bias"),
      py::arg("profile"));

  module.def("covariance_cluster_spectra", [](
          const py::object& ell,
          const py::object& distance,
          const py::object& dchi,
          const py::object& base,
          const py::object& window,
          const py::object& bias,
          const py::object& power,
          const py::object& profile,
          const py::object& richness,
          const int nlens) {
        const arma::Col<double> ell_input =
            notebook_input_cov<arma::Col<double>>(ell, 1);
        const arma::Col<double> distance_input =
            notebook_input_cov<arma::Col<double>>(distance, 1);
        const arma::Col<double> dchi_input =
            notebook_input_cov<arma::Col<double>>(dchi, 1);
        const arma::Mat<double> base_input =
            notebook_input_cov<arma::Mat<double>>(base, 2);
        const arma::Mat<double> window_input =
            notebook_input_cov<arma::Mat<double>>(window, 2);
        const arma::Mat<double> bias_input =
            notebook_input_cov<arma::Mat<double>>(bias, 2);
        const arma::Mat<double> power_input =
            notebook_input_cov<arma::Mat<double>>(power, 2);
        const arma::Cube<double> profile_input =
            notebook_input_cov<arma::Cube<double>>(profile, 3);
        const arma::Col<int> richness_input =
            notebook_input_cov<arma::Col<int>>(richness, 1);
        return covariance_cluster_spectra_cpp(
            ell_input,
            distance_input, dchi_input, base_input, window_input, bias_input,
            power_input, profile_input, richness_input, nlens);
      },
      R"doc(Project every cluster-galaxy, cluster-source and cluster pair.

Arguments (float64, except richness as int32):
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
      py::arg("ell"), py::arg("distance"),
      py::arg("dchi"), py::arg("base"),
      py::arg("window"), py::arg("bias"),
      py::arg("power"), py::arg("profile"),
      py::arg("richness"), py::arg("nlens"));

  module.def("covariance_counts_shell", [](
          const py::object& distance,
          const py::object& density,
          const py::object& derivative,
          const double area_sr) {
        const arma::Col<double> distance_input =
            notebook_input_cov<arma::Col<double>>(distance, 1);
        const arma::Mat<double> density_input =
            notebook_input_cov<arma::Mat<double>>(density, 2);
        const arma::Mat<double> derivative_input =
            notebook_input_cov<arma::Mat<double>>(derivative, 2);
        return covariance_counts_shell_cpp(
            distance_input,
            density_input, derivative_input, area_sr);
      },
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
      py::arg("distance"),
      py::arg("density"),
      py::arg("derivative"),
      py::arg("area_sr"));
}

} // namespace cosmolike_interface
