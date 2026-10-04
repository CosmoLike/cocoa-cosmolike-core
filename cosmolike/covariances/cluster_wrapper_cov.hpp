#ifndef COSMOLIKE_CLUSTER_WRAPPER_COV_HPP
#define COSMOLIKE_CLUSTER_WRAPPER_COV_HPP

#include <armadillo>
#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Notebook numeric inputs and outputs use Armadillo's physical axes.
// CARMA performs the NumPy conversion only at the binding boundary.
// Dict/tuple results group the named matrices and cubes documented below.

pybind11::dict covariance_counts_shell_cpp(
    const arma::Col<double>& distance, // [nnode], transverse distances
    const arma::Mat<double>& density,  // [ncount,nnode], selected n_i
    const arma::Mat<double>& derivative, // matching dn_i/d(delta_b)
    const double area_sr               // survey solid angle
  );

pybind11::dict covariance_cluster_spectra_cpp(
    const arma::Col<double>& ell,       // multipole samples
    const arma::Col<double>& distance,  // common transverse distances
    const arma::Col<double>& dchi,      // radial integration weights
    const arma::Mat<double>& base,      // galaxy and lensing windows
    const arma::Mat<double>& window,    // normalized cluster windows
    const arma::Mat<double>& bias,      // selected cluster bias
    const arma::Mat<double>& power,     // nonlinear matter power
    const arma::Cube<double>& profile,   // selected one-halo spectra
    const arma::Col<int>& richness, // profile map
    const int nlens                     // leading galaxy fields in base
  );

pybind11::dict covariance_cluster_moments_cpp(
    const arma::Cube<double>& weight,  // [state,selection,mass], selected dn
    const arma::Mat<double>& bias,    // [state,mass], linear halo bias
    const arma::Cube<double>& profile  // [state,k,mass], (M/rho)*u(k|M)
  );

pybind11::dict covariance_cluster_halo_samples_cpp(
    const arma::Col<double>& a,       // scale factors [state]
    const arma::Mat<double>& k,       // core wavenumbers [state,k]
    const arma::Col<double>& lnm,     // log masses [mass]
    const arma::Col<double>& dlnm     // positive quadrature measures [mass]
  );

}
#endif
