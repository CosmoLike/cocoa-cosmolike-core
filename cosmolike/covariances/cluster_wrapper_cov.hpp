#ifndef COSMOLIKE_CLUSTER_WRAPPER_COV_HPP
#define COSMOLIKE_CLUSTER_WRAPPER_COV_HPP

#include <carma.h>
#include <armadillo>
#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// ---------------------------------------------------------------------------
// Armadillo notebook wrappers of the cluster covariance components.
//
// generic_interface_cluster_cov.cpp copies each NumPy argument with
// notebook_input_cov: C-order, Fortran-order and sliced arrays are all
// accepted, and the caller's array is never modified. The wrappers in
// cluster_wrapper_cov.cpp check shapes and domains, copy into C row
// workspaces and call the shared *_cluster_cov.c routines, which perform
// every integral and OpenMP loop. Each result is an Armadillo matrix or
// cube exported by CARMA into a Python dict; every NumPy array owns its
// memory, independent of later calls. The production interface
// (cluster_interface_cov.cpp) calls the same C routines directly on
// borrowed contiguous NumPy arrays.
//
// Axes are listed [first,second,third] in Armadillo index order, which is
// also the NumPy shape. L is one consistent length unit (c/H0 in the
// survey workflow). A selected quantity carries the probability S_i that
// a halo enters observed category i.
// ---------------------------------------------------------------------------

// Count shells (counts_shell_cluster_cov). Dict of arma::Mat [ncount,nnode]
// in L^-1: shell_density S_i = area_sr f_K^2 n_i = dN_i/dchi and
// shell_response Phi_i = area_sr f_K^2 dn_i/d(delta_b).
pybind11::dict covariance_counts_shell_cpp(
    const arma::Col<double>& distance, // [nnode], f_K in L
    const arma::Mat<double>& density,  // [ncount,nnode], selected n_i, L^-3
    const arma::Mat<double>& derivative, // matching dn_i/d(delta_b), L^-3
    const double area_sr               // survey solid angle, sr
  );

// Limber cluster spectra (limber_cluster_cov), with P_NL and P_cm^1h in
// L^3: cluster-cluster with b_c b_c' P_NL, cluster-galaxy with b_c W_g
// P_NL, and cluster-source with b_c P_NL + P_cm^1h and the core spin
// factor. Dict of dimensionless arma::Cube: cluster_base
// [nell,ncluster,nbase] and cluster_cluster [nell,ncluster,ncluster],
// the latter exactly symmetric.
pybind11::dict covariance_cluster_spectra_cpp(
    const arma::Col<double>& ell,       // [nell], multipoles >= 2
    const arma::Col<double>& distance,  // [nnode], f_K in L
    const arma::Col<double>& dchi,      // [nnode], radial weights in L
    const arma::Mat<double>& base,      // [nbase,nnode], W_g then W_s, L^-1
    const arma::Mat<double>& window,    // [ncluster,nnode], q_c, L^-1
    const arma::Mat<double>& bias,      // [ncluster,nnode], selected b_c
    const arma::Mat<double>& power,     // [nell,nnode], P_NL at (ell+1/2)/f_K
    const arma::Cube<double>& profile,   // [nrichness,nell,nnode], P_cm^1h, L^3
    const arma::Col<int>& richness, // [ncluster], profile row of each q_c
    const int nlens                     // leading galaxy fields in base
  );

// Selected halo-mass moments (moments_cluster_cov), S_i entering once.
// Dict: density and biased_density, arma::Mat [state,selection] in L^-3;
// J01 and J11, arma::Cube [state,selection,k], dimensionless; J02,
// J03_KKQ and J03_KQQ, arma::Cube [state,selection,kpair] in L^3, L^6,
// L^6, with kpair the nk(nk+1)/2 upper-triangle pairs (0,0),(0,1),...
pybind11::dict covariance_cluster_moments_cpp(
    const arma::Cube<double>& weight,  // [state,selection,mass], dn S_i, L^-3
    const arma::Mat<double>& bias,    // [state,mass], linear halo bias
    const arma::Cube<double>& profile  // [state,k,mass], (M/rho)u(k|M), L^3
  );

// Samples of the initialized halo and richness model on caller mass nodes
// (halo_samples_cluster_cov). Dict: weight arma::Cube [state,richness,mass]
// = dlnM dn/dlnM S in (c/H0)^-3; bias arma::Mat [state,mass], the linear
// halo bias; profile arma::Cube [state,k,mass] = (M/rho_m) u_NFW in
// (c/H0)^3. These are the inputs of covariance_cluster_moments_cpp.
pybind11::dict covariance_cluster_halo_samples_cpp(
    const arma::Col<double>& a,       // scale factors [state]
    const arma::Mat<double>& k,       // wavenumbers [state,k], (c/H0)^-1
    const arma::Col<double>& lnm,     // ln(M/[Msun/h]) [mass]
    const arma::Col<double>& dlnm     // positive quadrature measures [mass]
  );

}
#endif
