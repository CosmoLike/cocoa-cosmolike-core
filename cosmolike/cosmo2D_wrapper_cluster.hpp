#include <carma.h>
#include <armadillo>
#include <map>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_COSMO2D_WRAPPER_CLUSTER_HPP
#define __COSMOLIKE_COSMO2D_WRAPPER_CLUSTER_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Notebook bindings of the cluster 2D statistics (cosmo2D_cluster.c),
// implemented in cosmo2D_wrapper_cluster.cpp: the cluster analog of
// cosmo2D_wrapper.hpp.
//
// One Python call travels
//
//   ci.w_gammat_cluster_tomo()                 (Python)
//     -> m.def("w_gammat_cluster_tomo")        (project interface.cpp)
//     -> w_gammat_cluster_tomo_cpp()           (this layer: checks the
//                                               state, warms the tables
//                                               on one thread, loops)
//     -> w_gammat_cluster_tomo(nt, nl, ni, ns) (cosmo2D_cluster.c: reads
//                                               a cached block)
//
// Names, as in cosmo2D_wrapper.hpp: each function is the C function's
// name plus _cpp, and its Python name is the C name itself
// (cosmo2D_cluster.c w_cc_tomo -> w_cc_tomo_cpp -> ci.w_cc_tomo). The
// arrays are indexed by the bins themselves,
//
//   (theta or ell, richness bin, cluster z bin, source or lens bin),
//
// the argument order of the C functions.
//
// Index names (cosmo2D_cluster.h): nt = theta bin, nl = richness bin,
// ni = cluster redshift bin, ns = source bin, ng = lens bin.
//
// Return type: the 4d arrays are Fortran-ordered numpy arrays built by
// to_np4d (cosmo2D_wrapper.hpp).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tomographic pair lists (the pair maps of redshift_spline_cluster.c)
// ---------------------------------------------------------------------------

// row n = (ZC_cs(n), ZS_cs(n)): cluster bin and source bin of cs pair n
arma::Mat<double> cs_bins();

// row n = (ZC_cg(n), ZG_cg(n)): cluster bin and lens bin of cg pair n
arma::Mat<double> cg_bins();

// row n = (NL1_cc(n), NL2_cc(n)): richness bins of w_cc richness pair n
arma::Mat<double> cc_richness_bins();

// ---------------------------------------------------------------------------
// Real-space (theta-binned) statistics and number counts
// ---------------------------------------------------------------------------

// cluster gamma_t BEFORE the Y transform, the selection bias and the
// shear calibration: (Ntheta, richness_nbin, zdist_nbin, shear_nbin)
pybind11::array_t<double,pybind11::array::f_style>
w_gammat_cluster_tomo_cpp();

// cluster lensing as the data vector holds it (unmasked): Sigma = T gamma_t
// (gamma_t when cluster.ytransform = 0) times the selection bias and
// (1 + m): (Ntheta, richness_nbin, zdist_nbin, shear_nbin)
pybind11::array_t<double,pybind11::array::f_style>
w_sigma_cluster_tomo_cpp();

// w_cc before the selection bias:
// (Ntheta, richness_nbin, richness_nbin, zdist_nbin)
pybind11::array_t<double,pybind11::array::f_style> w_cc_tomo_cpp(
    const int limber
  );

// w_cg before the selection bias:
// (Ntheta, richness_nbin, zdist_nbin, clustering_nbin)
pybind11::array_t<double,pybind11::array::f_style> w_cg_tomo_cpp(
    const int limber
  );

// expected counts: (richness_nbin, zdist_nbin)
arma::Mat<double> N_cluster_tomo_cpp();

// ---------------------------------------------------------------------------
// Fourier-space C_l: cluster lensing
// ---------------------------------------------------------------------------

double C_cs_tomo_limber_cpp(
    const double l,
    const int nl,
    const int ni,
    const int ns
  );

// (nell, richness_nbin, zdist_nbin, shear_nbin)
pybind11::array_t<double,pybind11::array::f_style> C_cs_tomo_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// Fourier-space C_l: cluster clustering (auto z bin, richness pairs)
// ---------------------------------------------------------------------------

double C_cc_tomo_limber_cpp(
    const double l,
    const int nl1,
    const int nl2,
    const int ni
  );

// (nell, richness_nbin, richness_nbin, zdist_nbin)
pybind11::array_t<double,pybind11::array::f_style> C_cc_tomo_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// Fourier-space C_l: cluster x galaxy clustering
// ---------------------------------------------------------------------------

double C_cg_tomo_limber_cpp(
    const double l,
    const int nl,
    const int ni,
    const int ng
  );

// (nell, richness_nbin, zdist_nbin, clustering_nbin)
pybind11::array_t<double,pybind11::array::f_style> C_cg_tomo_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
