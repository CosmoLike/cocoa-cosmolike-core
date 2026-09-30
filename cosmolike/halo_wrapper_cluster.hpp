#include <carma.h>
#include <armadillo>
#include <map>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_HALO_WRAPPER_CLUSTER_HPP
#define __COSMOLIKE_HALO_WRAPPER_CLUSTER_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Cluster halo-model and radial-kernel bindings (halo_cluster.c,
// redshift_spline_cluster.c, radial_weights_cluster.c), implemented in
// halo_wrapper_cluster.cpp: the cluster analog of halo_wrapper.hpp.
//
// Those files compute the ingredients of every cluster statistic: the
// probability that a halo lands in a richness bin (the mass-observable
// relation), the number density, bias and one-halo spectrum of the
// clusters of a richness bin, the redshift selection of a cluster bin
// and the radial kernels built from it. This layer makes them callable
// from Python one number at a time (tests) or on whole arrays
// (notebooks).
//
// One Python call travels
//
//   ci.ncl_richness(a)                    (Python; a = numpy array)
//     -> m.def("ncl_richness", ...)       (project interface.cpp)
//     -> ncl_richness_cpp(a)              (this layer: warms the tables
//                                          on one thread, loops)
//     -> ncl_richness(a, nl)              (halo_cluster.c: reads a
//                                          cached table)
//
// Names: each function below is the C function's name plus _cpp, and
// its Python name is the C name itself: halo_cluster.c ncl_richness ->
// ncl_richness_cpp -> ci.ncl_richness.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Overloads: the scalar overload takes the C function's arguments (one
// bin, one point) and returns one number; the array overload takes the
// continuous arguments as arrays and returns every bin at once, with
// the axes in the order of the C function's arguments:
//
//   prob_richness_bin_given_m(lnM, z, nl) -> (nlnM, nz, richness_nbin)
//   ncl_richness(a, nl), bcl_richness     -> (na, richness_nbin)
//   pcm_1h_richness(k, a, nl)             -> (nk, na, richness_nbin)
//   phi_cluster(z, ni)                    -> (nz, zdist_nbin)
//   nz_cluster(z, ni, nl)                 -> (nz, zdist_nbin, richness_nbin)
//   g_cluster, W_cluster, W_mag_cluster(a, ni, nl)
//                                         -> (na, zdist_nbin, richness_nbin)
//
// Index names: nl = richness bin, ni = cluster redshift bin.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Units: the C files work in cosmolike code units and this layer passes
// them through unconverted:
//
//   lnM = natural log of the halo mass in M_sun/h (M200m)
//   k   = wavenumber in (c/H0)^-1: k = k[h/Mpc] * coverH0, with
//         coverH0 = c/H0 = 2997.92458 Mpc/h
//   P   = power spectra in (c/H0)^3: P = P[(Mpc/h)^3] / coverH0^3
//   n   = comoving cluster number density in (c/H0)^-3
//   a   = scale factor, z = redshift
//
// Dimensionless: prob_richness_bin_given_m, bcl_richness, phi_cluster,
// g_cluster; nz_cluster is per unit z, W_cluster per unit comoving
// distance (c/H0).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Warm-up and threading: the cluster tables are filled lazily and a
// refill is not thread-safe, so every wrapper of a cosmology-dependent
// function calls cluster_warmup() (halo_cluster.h) on the calling
// thread and then loops serially over its inputs. phi_cluster and
// prob_richness_bin_given_m need no cosmology and skip the warm-up
// (the selection-kernel table is built by the first, serial, read).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// MASS-OBSERVABLE RELATION
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// probability that a halo of mass M at redshift z has observed richness
// in bin nl (closed-form erf difference)
double prob_richness_bin_given_m_cpp(
    const double lnM,
    const double z,
    const int nl
  );

arma::Cube<double> prob_richness_bin_given_m_cpp(
    const arma::Col<double> lnM,
    const arma::Col<double> z
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL TABLES OF A RICHNESS BIN (cached tables in a)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// comoving number density of the clusters of richness bin nl
double ncl_richness_cpp(const double a, const int nl);

arma::Mat<double> ncl_richness_cpp(const arma::Col<double> a);

// -----------------------------------------------------------------------------

// richness-weighted linear bias of richness bin nl
double bcl_richness_cpp(const double a, const int nl);

arma::Mat<double> bcl_richness_cpp(const arma::Col<double> a);

// -----------------------------------------------------------------------------

// one-halo cluster-matter power spectrum of richness bin nl
double pcm_1h_richness_cpp(const double k, const double a, const int nl);

arma::Cube<double> pcm_1h_richness_cpp(
    const arma::Col<double> k,
    const arma::Col<double> a
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// REDSHIFT SELECTION AND RADIAL KERNELS OF A CLUSTER BIN
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// selection kernel <phi_ni|z> at true redshift z
double phi_cluster_cpp(const double z, const int ni);

arma::Mat<double> phi_cluster_cpp(const arma::Col<double> z);

// -----------------------------------------------------------------------------

// normalized true-redshift distribution of the clusters of bin ni
// (richness bin nl for the abundance-weighted kernel)
double nz_cluster_cpp(const double z, const int ni, const int nl);

arma::Cube<double> nz_cluster_cpp(const arma::Col<double> z);

// -----------------------------------------------------------------------------

// lensing efficiency of the cluster distribution (cluster magnification)
double g_cluster_cpp(const double a, const int ni, const int nl);

arma::Cube<double> g_cluster_cpp(const arma::Col<double> a);

// -----------------------------------------------------------------------------

// cluster density kernel W_cluster = nz_cluster H/H0
double W_cluster_cpp(const double a, const int ni, const int nl);

arma::Cube<double> W_cluster_cpp(const arma::Col<double> a);

// -----------------------------------------------------------------------------

// cluster magnification kernel 1.5 Omega_m f_K/a g_cluster (without the
// coefficient cluster.magnification)
double W_mag_cluster_cpp(const double a, const int ni, const int nl);

arma::Cube<double> W_mag_cluster_cpp(const arma::Col<double> a);

// -----------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
