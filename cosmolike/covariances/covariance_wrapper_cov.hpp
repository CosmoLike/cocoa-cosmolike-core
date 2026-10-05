#ifndef COSMOLIKE_COVARIANCE_WRAPPER_COV_HPP
#define COSMOLIKE_COVARIANCE_WRAPPER_COV_HPP

#include <carma.h>
#include <armadillo>
#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Notebook numeric inputs and outputs use Armadillo's physical axes.
// Bindings copy Python inputs and use CARMA to export numeric results.
// Dict/tuple results group the named matrices and cubes documented below.

pybind11::dict covariance_limber_spectra_cpp(
    const arma::Col<double>& ell,     // multipole samples
    const arma::Col<double>& a_edges, // scale-factor panel edges
    const int nquad,        // Gaussian nodes per scale-factor panel
    const int nwindow,      // uniform-a lensing-efficiency samples
    const bool include_ia,  // include the signed NLA window
    const bool include_rsd, // include the lens redshift-distortion window
    const bool linear,     // select linear rather than nonlinear matter P
    const int nonlimber_lmax, // gg/gs correction through this ell; 0 disables
    const int nonlimber_nchi, // logarithmic radial samples, 2^n+1
    const double nonlimber_chi_min // positive near distance in c/H0
  );

arma::Mat<double> covariance_integration_rule_cpp(
    const int nquad // precomputed rule size: 64,96,128,256,512,1024
  );

arma::Mat<double> covariance_project_cpp(
    const arma::Mat<double>& left,   // [nleft,nnode], left operators
    const arma::Mat<double>& right,  // [nright,nnode], right rows
    const arma::Col<double>& weight  // [nnode], common integration weights
  );

arma::Col<double> covariance_gaussian_wick_cpp(
    const arma::Mat<double>& cross_spectra, // [4,nell], signal only
    const arma::Col<double>& cross_noise,   // [4], matching white-noise powers
    const int ell_min,              // first consecutive integer multipole
    const double fsky,              // survey area / (4*pi)
    const bool include_noise_noise  // retain the pure noise product
  );

arma::Cube<double> covariance_realspace_operator_cpp(
    const arma::Col<double>& edges_rad, // angular-bin boundaries in radians
    const int ell_max,          // last integer multipole, inclusive
    const int nquad             // integration nodes per angular bin
  );

arma::Mat<double> covariance_bandpower_operator_cpp(
    const arma::Col<int>& first, // inclusive lower multipole of each band
    const arma::Col<int>& last,  // inclusive upper multipole
    const int ell_min,          // first multipole of the shared output grid
    const int nell             // number of consecutive output multipoles
  );

arma::Col<double> covariance_mask_pair_area_cpp(
    const arma::Col<double>& edges_rad,    // angular-bin edges in radians
    const arma::Col<double>& mask_cl,      // raw footprint spectrum, L=0 onward
    const double area_sr,          // footprint area
    const arma::Mat<double>& scalar_kernel // [nbin,nmask], w kernel
  );

arma::Col<double> covariance_ssc_mask_variance_cpp(
    const arma::Col<double>& mask_cl, // raw footprint spectrum
    const double area_sr,    // footprint area in steradians
    const arma::Col<double>& distance,// positive transverse distances, c/H0
    const arma::Mat<double>& power   // [nnode,nmask], linear power, (c/H0)^3
  );

arma::Mat<double> covariance_ssc_shell_response_cpp(
    const arma::Col<double>& distance,       // [nnode], transverse distances
    const arma::Col<double>& signal, // [nrow], full projected spectra
    const arma::Mat<double>& pair_window,    // [nrow,nnode], W_A*W_B
    const arma::Mat<double>& mean_window,    // [nrow,nnode], U_A+U_B
    const arma::Mat<double>& power_response  // [nrow,nnode], dP/d(delta_b)
  );

pybind11::tuple covariance_halo_moments_cpp(
    const arma::Col<double>& a,         // scale factors
    const arma::Mat<double>& k,         // [na,nk], inverse c/H0
    const arma::Col<double>& lnm_edges, // log halo-mass panel edges
    const int nquad,           // Gaussian mass nodes per panel
    const bool pair_moments    // also compute the five pair-moment roles
  );

arma::Mat<double> covariance_power_cpp(
    const double a,      // scale factor inside the initialized range
    const arma::Mat<double>& k,  // physical wavenumbers in inverse c/H0
    const bool linear   // linear total-matter P or the configured Pdelta
  );

arma::Col<double> covariance_power_vector_cpp(
    const double a,             // scale factor
    const arma::Col<double>& k, // wavenumbers in inverse c/H0
    const bool linear          // linear or configured nonlinear power
  );

arma::Mat<double> covariance_tree_averages_cpp(
    const arma::Mat<double>& k,      // [2,npair], positive K and Q
    const arma::Mat<double>& pk,     // [2,npair], matching linear power
    const arma::Col<double>& corner, // [nangle], stable 1+cos(theta)
    const arma::Col<double>& weight, // [nangle], normalized dtheta/pi weights
    const arma::Mat<double>& ps      // [npair,nangle], P(|K+Q|)
  );

arma::Mat<double> covariance_halo_trispectrum_cpp(
    const arma::Mat<double>& pk,      // [2,npoint], linear power at K,Q
    const arma::Mat<double>& i11,     // [2,npoint], one-profile moments
    const arma::Mat<double>& moments, // [5,npoint], halo pair moments
    const arma::Mat<double>& tree     // [3,npoint], angular P/B/T averages
  );

arma::Mat<double> covariance_halo_response_cpp(
    const arma::Mat<double>& inputs, // [6,npoint], halo inputs
    const double growth_coefficient,   // constant growth contribution
    const double dilation_coefficient, // coefficient of logarithmic slope
    const bool fractional              // transfer D_halo/P_halo to P_target
  );

arma::Mat<double> covariance_gaussian_real_cpp(
    const arma::Cube<double>& spectra,
    const arma::Col<double>& noise,
    const arma::Mat<int>& rows,
    const arma::Cube<double>& operators,
    const int ell_min,
    const double area_sr,
    const arma::Col<double>& pair_area_sr2,
    const arma::Cube<double>& b_spectra
  );

arma::Mat<double> covariance_gaussian_fourier_cpp(
    const arma::Cube<double>& spectra,
    const arma::Col<double>& noise,
    const arma::Mat<int>& pairs,
    const arma::Mat<double>& operators,
    const int ell_min,
    const double area_sr
  );

arma::Mat<double> covariance_project_connected_cpp(
    const arma::Col<int>& probes,        // (observable), xi+,xi-,gamma_t,w
    const arma::Mat<double>& pair_window,// (observable,node), W_A W_B
    const arma::Cube<double>& projected, // (4*bin,4*bin,node), transformed T
    const arma::Col<double>& measure     // (node), dchi/(area*f_K^6)
  );

double covariance_noise_pair_cpp(
    const int probe_left,       // xi+, xi-, gamma_t, w: 0,1,2,3
    const int probe_right,      // right estimator in the same convention
    const arma::Col<int>& fields,// [4], A,B,C,D global catalog indices
    const arma::Col<double>& noise_ab,  // [2], powers of catalogs A and B
    const double pair_area_sr2 // ordered-pair area within the angular bin
  );

void bind_covariance_wrappers(pybind11::module_& module);
}
#endif
