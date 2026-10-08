#ifndef COSMOLIKE_COVARIANCE_WRAPPER_COV_HPP
#define COSMOLIKE_COVARIANCE_WRAPPER_COV_HPP

#include <carma.h>
#include <armadillo>
#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// ---------------------------------------------------------------------------
// Armadillo notebook wrappers of the covariance components and matrices.
//
// The notebook bindings, python_components_cov.cpp and
// generic_interface_cov.cpp, copy each NumPy argument with
// notebook_input_cov: C-order, Fortran-order and sliced arrays are all
// accepted, and the caller's array is never modified. The wrappers in
// components_wrapper_cov.cpp and covariance_wrapper_cov.cpp check shapes
// and physical domains, copy into row-major C workspaces and call the
// shared covariance C routines, which own every integral, SIMD loop and
// OpenMP team. Results return as owning Armadillo objects that CARMA
// exports to NumPy; a dict or tuple groups several named results.
// The production interface (production_interface_cov.cpp,
// components_interface_cov.cpp, matrix_interface_cov.cpp) calls the same
// C routines on borrowed contiguous NumPy arrays instead.
//
// Axes are listed [first,second,...] in Armadillo index order, which is
// also the NumPy shape. Core units: distances and f_K in c/H0, wavenumbers
// in (c/H0)^-1, matter power in (c/H0)^3, angles in radians, solid angles
// in sr. White-noise powers are in steradians: 1/n for galaxy counts and
// sigma_e^2/n for shear, with n per steradian and sigma_e per ellipticity
// component. A steradian is dimensionless, so this noise adds directly to
// the dimensionless C_ell; n per arcmin^2 would be a unit error.
// ---------------------------------------------------------------------------

// All lens/source field-pair spectra on one radial rule (radial_inputs_cov,
// limber_spectra_cov, tatt_spectra_cov, apply_nonlimber_cov). Dict:
// spectra arma::Cube [nell,nfield,nfield] in the core C_ell convention;
// b_spectra with the same axes for TATT B, else None; geometry arma::Mat
// [4,nnode] (a, chi, f_K, dchi weight); windows arma::Cube
// [3,nfield,nnode] (density, lensing/magnification, signed NLA) in
// (c/H0)^-1; nlens and nsource. Lenses precede sources on field axes.
pybind11::dict covariance_limber_spectra_cpp(
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
  );

// GSL Gauss-Legendre rule on [-1,1]: arma::Mat [2,nquad], nodes in row 0
// and positive weights, summing to 2, in row 1.
arma::Mat<double> covariance_integration_rule_cpp(
    const int nquad // precomputed rule size: 64,96,128,256,512,1024
  );

// out(i,j) = sum_node left(i,node) weight(node) right(j,node), computed by
// gaussian_project_cov: arma::Mat [nleft,nright], in the product units.
arma::Mat<double> covariance_project_cpp(
    const arma::Mat<double>& left,   // [nleft,nnode], left operators
    const arma::Mat<double>& right,  // [nright,nnode], right rows
    const arma::Col<double>& weight  // [nnode], common integration weights
  );

// Gaussian covariance of C_AB and C_CD at each multipole (gaussian_wick_cov)
//   [(C_AC+N_AC)(C_BD+N_BD) + (C_AD+N_AD)(C_BC+N_BC)] / ((2 ell+1) fsky):
// arma::Col [nell] at ell = ell_min, ell_min+1, ... . Without
// include_noise_noise the N*N products are omitted (real-space split).
arma::Col<double> covariance_gaussian_wick_cpp(
    const arma::Mat<double>& cross_spectra, // [4,nell], AC,BD,AD,BC signal
    const arma::Col<double>& cross_noise,   // [4], matching white noise, sr
    const int ell_min,              // first consecutive integer multipole
    const double fsky,              // survey area / (4*pi)
    const bool include_noise_noise  // retain the pure noise product
  );

// Bin-averaged full-sky estimator operators (realspace_operator_cov):
// arma::Cube [4,nbin,ell_max+1] for probes xi+, xi-, gamma_t, w, each
// entry (2 ell+1)/(4 pi) times the bin-averaged d^ell kernel. They act
// on unit-normalized observed-shear spectra and are dimensionless.
arma::Cube<double> covariance_realspace_operator_cpp(
    const arma::Col<double>& edges_rad, // angular-bin boundaries in radians
    const int ell_max,          // last integer multipole, inclusive
    const int nquad             // integration nodes per angular bin
  );

// Mode-weighted Fourier bands (bandpower_operator_cov): arma::Mat
// [nband,nell] with (2 ell+1)/N_band inside [first,last], zero elsewhere;
// column 0 is ell_min.
arma::Mat<double> covariance_bandpower_operator_cpp(
    const arma::Col<int>& first, // inclusive lower multipole of each band
    const arma::Col<int>& last,  // inclusive upper multipole
    const int ell_min,          // first multipole of the shared output grid
    const int nell             // number of consecutive output multipoles
  );

// Ordered-pair area of each angular bin for one common binary footprint
// (mask_pair_area_cov): arma::Col [nbin] in sr^2.
arma::Col<double> covariance_mask_pair_area_cpp(
    const arma::Col<double>& edges_rad,    // angular-bin edges in radians
    const arma::Col<double>& mask_cl,      // raw footprint spectrum, L=0 onward
    const double area_sr,          // footprint area, sr
    const arma::Mat<double>& scalar_kernel // [nbin,nmask], w operator rows
  );

// Long-mode Limber background strength (ssc_mask_variance_cov)
//   sigma_b^2(chi) = sum_L (2L+1) C_L^W P_lin((L+1/2)/f_K) / (area^2 f_K^2):
// arma::Col [nnode], a length in c/H0, not a dimensionless variance.
arma::Col<double> covariance_ssc_mask_variance_cpp(
    const arma::Col<double>& mask_cl, // [nmask], raw footprint spectrum
    const double area_sr,    // footprint area in steradians
    const arma::Col<double>& distance,// positive transverse distances, c/H0
    const arma::Mat<double>& power   // [nnode,nmask], linear power, (c/H0)^3
  );

// Radial SSC response of each (field pair, multipole) row
// (ssc_shell_response_cov):
//   Phi = W_A W_B D((ell+1/2)/f_K)/f_K^2 - (U_A+U_B) C_AB,
// arma::Mat [nrow,nnode] in (c/H0)^-1.
arma::Mat<double> covariance_ssc_shell_response_cpp(
    const arma::Col<double>& distance,       // [nnode], f_K in c/H0
    const arma::Col<double>& signal, // [nrow], full projected spectra
    const arma::Mat<double>& pair_window,    // [nrow,nnode], W_A*W_B
    const arma::Mat<double>& mean_window,    // [nrow,nnode], U_A+U_B
    const arma::Mat<double>& power_response  // [nrow,nnode], dP/d(delta_b)
  );

// cb-field halo moments (halo_moments_cov). Tuple: I11 arma::Mat [na,nk],
// dimensionless, with the low-mass completion; then arma::Cube
// [5,na,nk(nk+1)/2] with roles I02(K,Q), I12(K,Q), I13(K,Q,Q),
// I13(K,K,Q), I04(K,K,Q,Q) in (c/H0)^3, ^3, ^6, ^6, ^9, or None.
pybind11::tuple covariance_halo_moments_cpp(
    const arma::Col<double>& a,         // [na], scale factors
    const arma::Mat<double>& k,         // [na,nk], inverse c/H0
    const arma::Col<double>& lnm_edges, // ln(M/[Msun/h]) mass-panel edges
    const int nquad,           // Gauss-Legendre mass nodes per panel
    const bool pair_moments    // also compute the five pair-moment roles
  );

// Matter power at one scale factor (power_rows_cov): arma::Mat with the
// shape of k, in (c/H0)^3. Rows are independent OpenMP batches.
arma::Mat<double> covariance_power_cpp(
    const double a,      // scale factor inside the initialized range
    const arma::Mat<double>& k,  // physical wavenumbers in inverse c/H0
    const bool linear   // linear total-matter P or the configured Pdelta
  );

// The same matter power for one k vector: arma::Col [nk], in (c/H0)^3.
arma::Col<double> covariance_power_vector_cpp(
    const double a,             // scale factor
    const arma::Col<double>& k, // wavenumbers in inverse c/H0
    const bool linear          // linear or configured nonlinear power
  );

// Planar tree-level averages <P>, <B_tree>, <T_tree> (tree_averages_cov):
// arma::Mat [3,npair] in (c/H0)^3, ^6, ^9.
arma::Mat<double> covariance_tree_averages_cpp(
    const arma::Mat<double>& k,      // [2,npair], K and Q, (c/H0)^-1
    const arma::Mat<double>& pk,     // [2,npair], P_lin(K), P_lin(Q)
    const arma::Col<double>& corner, // [nangle], stable 1+cos(theta)
    const arma::Col<double>& weight, // [nangle], normalized dtheta/pi weights
    const arma::Mat<double>& ps      // [npair,nangle], P_lin(|K+Q|)
  );

// The five separated halo trispectrum terms (halo_trispectrum_cov):
// arma::Mat [5,npoint], rows 1h, 2h(1+3), 2h(2+2), 3h, 4h, in (c/H0)^9.
arma::Mat<double> covariance_halo_trispectrum_cpp(
    const arma::Mat<double>& pk,      // [2,npoint], linear power at K,Q
    const arma::Mat<double>& i11,     // [2,npoint], I11(K), I11(Q)
    const arma::Mat<double>& moments, // [5,npoint], halo pair-moment roles
    const arma::Mat<double>& tree     // [3,npoint], angular P/B/T averages
  );

// Halo power and its background response (halo_response_cov). inputs rows:
// P_lin, P_target, I11, I02(k,k), I12(k,k), dlnP_X/dlnk. Returns arma::Mat
// [2,npoint]: P_halo = I11^2 P_lin + I02 and D = dP/d(delta_b), (c/H0)^3.
arma::Mat<double> covariance_halo_response_cpp(
    const arma::Mat<double>& inputs, // [6,npoint], the six rows above
    const double growth_coefficient,   // constant growth contribution
    const double dilation_coefficient, // coefficient of logarithmic slope
    const bool fractional              // transfer D_halo/P_halo to P_target
  );

// Complete real-space Gaussian matrix (gaussian_matrix_cov): arma::Mat
// [nobs*nbin,nobs*nbin], bin inside observable, both triangles filled.
// spectra carry the spin-operator convention on source legs; noise does not.
arma::Mat<double> covariance_gaussian_real_cpp(
    const arma::Cube<double>& spectra,   // [nell,nfield,nfield], signal
    const arma::Col<double>& noise,      // [nfield], white noise, sr
    const arma::Mat<int>& rows,          // [nobs,3], (probe,A,B)
    const arma::Cube<double>& operators, // [4,nbin,nell], probe kernels
    const int ell_min,                   // first multipole, >= 2
    const double area_sr,                // survey solid angle, sr
    const arma::Col<double>& pair_area_sr2, // [nbin], ordered pairs, sr^2
    const arma::Cube<double>& b_spectra  // like spectra, BB; empty: E only
  );

// Complete Fourier-band Gaussian matrix (gaussian_matrix_cov, all noise in
// the harmonic sum): arma::Mat [nobs*nband,nobs*nband], band inside
// observable. Bands average the core C_ell directly.
arma::Mat<double> covariance_gaussian_fourier_cpp(
    const arma::Cube<double>& spectra,   // [nell,nfield,nfield], signal
    const arma::Col<double>& noise,      // [nfield], white noise, sr
    const arma::Mat<int>& pairs,         // [nobs,2], (A,B)
    const arma::Mat<double>& operators,  // [nband,nell], band weights
    const int ell_min,                   // first multipole, >= 0
    const double area_sr                 // survey solid angle, sr
  );

// Connected (cNG) matrix from angularly transformed matter trispectra
// (connected_matrix_cov), with W_A W_B in (c/H0)^-2, T in (c/H0)^9 and
// the measure in (c/H0)^-5 per sr: arma::Mat [nobs*nbin,nobs*nbin], bin
// inside observable, dimensionless.
arma::Mat<double> covariance_project_connected_cpp(
    const arma::Col<int>& probes,        // (observable), xi+,xi-,gamma_t,w
    const arma::Mat<double>& pair_window,// (observable,node), W_A W_B
    const arma::Cube<double>& projected, // (4*bin,4*bin,node), transformed T
    const arma::Col<double>& measure     // (node), dchi/(area*f_K^6)
  );

// Pure real-space pair-count noise of two estimators in one angular bin
// (gaussian_noise_pair_cov): one dimensionless covariance entry.
double covariance_noise_pair_cpp(
    const int probe_left,       // xi+, xi-, gamma_t, w: 0,1,2,3
    const int probe_right,      // right estimator in the same convention
    const arma::Col<int>& fields,// [4], A,B,C,D global catalog indices
    const arma::Col<double>& noise_ab,  // [2], white noise of A and B, sr
    const double pair_area_sr2 // ordered-pair area of the bin, sr^2
  );

// Register the Gaussian and connected matrix bindings above on a notebook
// module; the definition is in python_components_cov.cpp.
void bind_covariance_wrappers(pybind11::module_& module);
}
#endif
