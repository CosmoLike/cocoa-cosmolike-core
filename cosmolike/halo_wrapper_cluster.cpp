#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <variant>

// SPDLOG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/structs.h"

// cosmolike: the cluster contract (every header carries its extern "C" guard)
#include "cosmolike/structs_cluster.h"
#include "cosmolike/redshift_spline_cluster.h"
#include "cosmolike/radial_weights_cluster.h"
#include "cosmolike/halo_cluster.h"
#include "cosmolike/halo_wrapper_cluster.hpp"

// ---------------------------------------------------------------------------
// Pybind wrappers of the cluster halo model (halo_cluster.c), the
// cluster redshift distributions (redshift_spline_cluster.c) and the
// cluster radial kernels (radial_weights_cluster.c). The header
// (halo_wrapper_cluster.hpp) documents the call chain, the units and
// the warm-up once; each function below says which C function it calls
// and what comes back.
//
// Conventions shared by every wrapper here:
//
//   scalar overload = the C function's arguments (one point, one bin),
//                     one C call, one number back
//   array overload  = the continuous arguments as arrays; the same C
//                     call in a serial loop over the arrays and over
//                     every bin, returned as arma::Mat or arma::Cube
//                     with the axes in the C argument order
//   bad input       = spdlog::critical + exit(1), the same way the C
//                     code itself fails (it validates the bins; the
//                     array overloads validate what would otherwise
//                     take log(0) or size an array with zero bins)
//
// carma converts at the pybind11 boundary: a numpy array handed to an
// arma::Col<double> parameter arrives as an armadillo column (a copy),
// and a returned arma::Mat / arma::Cube reaches Python as a numpy array
// of the same shape, (rows, cols) or (rows, cols, slices).
// ---------------------------------------------------------------------------
namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INPUT CHECKS (PRIVATE)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the richness bins are set.
//
// The array overloads size their output with cluster.richness_nbin
// before any C function (and its own bin check) runs.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//
// Returns:
//   nothing; no richness bins abort (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_richness_bins(const char* fname)
{
  if (!(cluster.richness_nbin > 0)) {
    spdlog::critical("{}: richness bins not set (call "
                     "init_cluster_richness_bins first)", fname);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the cluster redshift bins (the selection kernels
// <phi_i|z>) are set.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//
// Returns:
//   nothing; no cluster bins abort (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_cluster_redshift_bins(const char* fname)
{
  if (!(cluster.zdist_nbin > 0) || NULL == cluster.zdist_table) {
    spdlog::critical("{}: cluster selection kernels not set (call "
                     "set_cluster_zdist first)", fname);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the input array is non-empty.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   name  - name of the argument, for the message
//   x     - the array
//
// Returns:
//   nothing; an empty array aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_not_empty(
    const char* fname,
    const char* name,
    const arma::Col<double>& x
  )
{
  if (0 == x.n_elem) {
    spdlog::critical("{}: {} array size = 0", fname, name);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless k is a positive wavenumber (the one-halo table is read
// at ln k).
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   k     - wavenumber in (c/H0)^-1
//
// Returns:
//   nothing; k <= 0 (or NaN) aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_wavenumber(const char* fname, const double k)
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive", fname, k);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless a is a scale factor strictly inside (0, 1).
//
// The radial kernels are defined there: W_cluster and W_mag_cluster
// abort on anything else, and H(a)/H0 and chi(a) are table reads at
// z = 1/a - 1.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   a     - scale factor
//
// Returns:
//   nothing; a outside (0, 1) (or NaN) aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_scale_factor(const char* fname, const double a)
{
  if (!(a > 0 && a < 1)) {
    spdlog::critical("{}: a = {} outside (0, 1)", fname, a);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// MASS-OBSERVABLE RELATION
//
// A cluster is a halo of mass M (M200m) whose observed richness lambda
// scatters around a mean set by M and z, the lognormal relation of eqs
// (18)-(19) of arXiv 2503.13631:
//
//   <ln lambda|M, z> = mor[0] + mor[1] ln(M/M_piv)
//                      + mor[3] ln((1 + z)/(1 + z_piv))
//   sigma^2          = mor[2]^2 + (e^<ln lambda> - 1)/e^(2 <ln lambda>)
//
// with cluster.mor = {ln lambda_0, A_lambda, sigma_int, B_lambda}
// (set_nuisance_cluster_mor) and the pivots cluster.mor_pivot_mass,
// cluster.mor_pivot_1pz. The probability that the halo lands in the
// richness bin [lambda_min, lambda_max) is a difference of two error
// functions.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Probability that a halo of mass M at redshift z has observed richness
// in bin nl.
//
// Calls halo_cluster.c prob_richness_bin_given_m: a closed form, no
// table and no cosmology (so no warm-up).
//
// Parameters:
//   lnM - natural log of the halo mass in M_sun/h
//   z   - redshift
//   nl  - richness bin (the C function aborts outside
//         [0, richness_nbin))
//
// Returns:
//   P(nl|M, z), in [0, 1]
// ---------------------------------------------------------------------------
double prob_richness_bin_given_m_cpp(
    const double lnM,   // ln of the halo mass in M_sun/h
    const double z,     // redshift
    const int nl        // richness bin
  )
{
  return prob_richness_bin_given_m(lnM, z, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P(nl|M, z) of every richness bin on a grid of masses and redshifts:
// the scalar overload in a serial loop.
//
// Parameters:
//   lnM - natural logs of the halo masses in M_sun/h; an empty array
//         aborts (spdlog::critical + exit)
//   z   - redshifts; an empty array aborts
//
// Returns:
//   arma::Cube (nlnM, nz, richness_nbin): entry (i, j, nl) =
//   P(nl|M_i, z_j); the sum over nl is the probability of any bin
// ---------------------------------------------------------------------------
arma::Cube<double> prob_richness_bin_given_m_cpp(
    const arma::Col<double> lnM,   // ln of the halo masses in M_sun/h
    const arma::Col<double> z      // redshifts
  )
{
  check_richness_bins("prob_richness_bin_given_m_cpp");
  check_not_empty("prob_richness_bin_given_m_cpp", "lnM", lnM);
  check_not_empty("prob_richness_bin_given_m_cpp", "z", z);

  arma::Cube<double> result(lnM.n_elem,
                            z.n_elem,
                            cluster.richness_nbin,
                            arma::fill::zeros);
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    for (arma::uword j=0; j<z.n_elem; j++) {
      for (arma::uword i=0; i<lnM.n_elem; i++) {
        result(i, j, nl) = prob_richness_bin_given_m(lnM(i), z(j), nl);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL TABLES OF A RICHNESS BIN
//
// Weighting the halo mass function dn/dlnM (Tinker et al. 2010, M200m)
// with P(nl|M, z) gives the three quantities every cluster statistic is
// built from (eqs 16, 21, 22 of arXiv 2503.13631):
//
//   n_nl(a)      = int dlnM dn/dlnM P(nl|M)               (number density)
//   b_nl(a)      = int dlnM dn/dlnM P(nl|M) b_h / n_nl    (linear bias)
//   P1h_nl(k, a) = int dlnM dn/dlnM P(nl|M) (M/rho_m) u_NFW(k|M) / n_nl
//                                            (one-halo cluster-matter)
//
// over [cluster.m[RANGE_MIN], cluster.m[RANGE_MAX]]. halo_cluster.c
// tabulates them on an a grid that covers the support of every cluster
// redshift bin and returns 0 outside it; the tables refill when the
// cosmology, the precision settings (Ntable), the cluster model, the
// selection kernels, the mass-observable relation or, under
// CLUSTER_SELECTION_Y1, the selection-bias parameters change.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comoving number density of the clusters of richness bin nl.
//
// Calls halo_cluster.c ncl_richness (cached table in a, spline in a)
// after cluster_warmup on the calling thread.
//
// Parameters:
//   a  - scale factor
//   nl - richness bin (the C function aborts outside
//        [0, richness_nbin))
//
// Returns:
//   n_nl(a) in (c/H0)^-3 (multiply by coverH0^-3 for (h/Mpc)^3); 0
//   outside the a grid of the cluster bins
// ---------------------------------------------------------------------------
double ncl_richness_cpp(
    const double a,   // scale factor
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return ncl_richness(a, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// n_nl(a) of every richness bin at many scale factors: one warm-up,
// then halo_cluster.c ncl_richness in a serial loop.
//
// Parameters:
//   a - scale factors; an empty array aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Mat (na, richness_nbin): rows = scale factor, columns =
//   richness bin, in (c/H0)^-3; 0 outside the a grid of the cluster bins
// ---------------------------------------------------------------------------
arma::Mat<double> ncl_richness_cpp(
    const arma::Col<double> a   // scale factors
  )
{
  check_richness_bins("ncl_richness_cpp");
  check_not_empty("ncl_richness_cpp", "a", a);
  cluster_warmup();

  arma::Mat<double> result(a.n_elem, cluster.richness_nbin, arma::fill::zeros);
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    for (arma::uword i=0; i<a.n_elem; i++) {
      result(i, nl) = ncl_richness(a(i), nl);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Richness-weighted linear bias of the clusters of richness bin nl
// (with the Y1 mass-dependent selection bias inside the integral when
// cluster.selection_model = CLUSTER_SELECTION_Y1).
//
// Calls halo_cluster.c bcl_richness (cached table in a, spline in a)
// after cluster_warmup on the calling thread.
//
// Parameters:
//   a  - scale factor
//   nl - richness bin (the C function aborts outside
//        [0, richness_nbin))
//
// Returns:
//   b_nl(a), dimensionless; 0 outside the a grid of the cluster bins
// ---------------------------------------------------------------------------
double bcl_richness_cpp(
    const double a,   // scale factor
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return bcl_richness(a, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// b_nl(a) of every richness bin at many scale factors: one warm-up,
// then halo_cluster.c bcl_richness in a serial loop.
//
// Parameters:
//   a - scale factors; an empty array aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Mat (na, richness_nbin): rows = scale factor, columns =
//   richness bin; 0 outside the a grid of the cluster bins
// ---------------------------------------------------------------------------
arma::Mat<double> bcl_richness_cpp(
    const arma::Col<double> a   // scale factors
  )
{
  check_richness_bins("bcl_richness_cpp");
  check_not_empty("bcl_richness_cpp", "a", a);
  cluster_warmup();

  arma::Mat<double> result(a.n_elem, cluster.richness_nbin, arma::fill::zeros);
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    for (arma::uword i=0; i<a.n_elem; i++) {
      result(i, nl) = bcl_richness(a(i), nl);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One-halo cluster-matter power spectrum of richness bin nl: the NFW
// profile of the clusters' own halos, the small-scale part of cluster
// lensing (no bias, no magnification).
//
// Calls halo_cluster.c pcm_1h_richness (cached table in (a, ln k):
// spline in ln k, linear in a, flat below the lowest tabulated k and a
// power law above the highest) after cluster_warmup on the calling
// thread. cluster_warmup builds this table only when cluster lensing is
// on (cluster.probe[CLUSTER_PROBE_CS]); otherwise the first read builds
// it, here, on the calling thread.
//
// Parameters:
//   k  - wavenumber in (c/H0)^-1
//   a  - scale factor
//   nl - richness bin (the C function aborts outside
//        [0, richness_nbin))
//
// Returns:
//   P1h_nl(k, a) in (c/H0)^3; 0 outside the a grid of the cluster bins
// ---------------------------------------------------------------------------
double pcm_1h_richness_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a,   // scale factor
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return pcm_1h_richness(k, a, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P1h_nl(k, a) of every richness bin on a grid of wavenumbers and scale
// factors: one warm-up, then halo_cluster.c pcm_1h_richness in a serial
// loop.
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0
//       aborts (spdlog::critical + exit)
//   a - scale factors; an empty array aborts
//
// Returns:
//   arma::Cube (nk, na, richness_nbin): entry (i, j, nl) =
//   P1h_nl(k_i, a_j) in (c/H0)^3; 0 outside the a grid of the cluster
//   bins
// ---------------------------------------------------------------------------
arma::Cube<double> pcm_1h_richness_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const arma::Col<double> a    // scale factors
  )
{
  check_richness_bins("pcm_1h_richness_cpp");
  check_not_empty("pcm_1h_richness_cpp", "k", k);
  check_not_empty("pcm_1h_richness_cpp", "a", a);
  for (arma::uword i=0; i<k.n_elem; i++) {
    check_wavenumber("pcm_1h_richness_cpp", k(i));
  }
  cluster_warmup();

  arma::Cube<double> result(k.n_elem,
                            a.n_elem,
                            cluster.richness_nbin,
                            arma::fill::zeros);
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    for (arma::uword j=0; j<a.n_elem; j++) {
      for (arma::uword i=0; i<k.n_elem; i++) {
        result(i, j, nl) = pcm_1h_richness(k(i), a(j), nl);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// REDSHIFT SELECTION AND RADIAL KERNELS OF A CLUSTER BIN
//
// A cluster redshift bin is a bin in the photometric redshift z_lambda.
// Its selection kernel <phi_ni|z> is the probability that a cluster at
// true redshift z lands in the bin (a table from Python,
// set_cluster_zdist). The normalized true-redshift distribution follows
// as
//
//   n(z) = dV/dz <phi_ni|z> [n_nl(z)] / norm
//
// (the n_nl factor only for cluster.kernel_mode =
// CLUSTER_KERNEL_ABUNDANCE; the volume kernel is the same for every
// richness bin), and from it the radial kernels of the Limber
// integrals:
//
//   W_cluster(a)     = n(z(a)) H(a)/H0          (density, per unit chi)
//   g_cluster(a)     = int_{z(a)}^{zmax} dz' n(z') [1 - chi(a)/chi(z')]
//   W_mag_cluster(a) = 1.5 Omega_m f_K(chi(a))/a g_cluster(a)
//
// W_mag_cluster enters the spectra times the coefficient
// cluster.magnification (C_c = -2).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Selection kernel <phi_ni|z> of cluster bin ni at true redshift z.
//
// Calls redshift_spline_cluster.c phi_cluster: the piecewise-linear
// interpolant of the Python table inside the bin's support, read from a
// fine z grid that the first call builds (on the calling thread). No
// cosmology enters, so no warm-up.
//
// Parameters:
//   z  - true redshift
//   ni - cluster redshift bin (the C function aborts outside
//        [0, zdist_nbin))
//
// Returns:
//   <phi_ni|z>, a probability; 0 outside the bin's support
// ---------------------------------------------------------------------------
double phi_cluster_cpp(
    const double z,   // true redshift
    const int ni      // cluster redshift bin
  )
{
  return phi_cluster(z, ni);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// <phi_ni|z> of every cluster bin at many redshifts:
// redshift_spline_cluster.c phi_cluster in a serial loop.
//
// Parameters:
//   z - true redshifts; an empty array aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Mat (nz, zdist_nbin): rows = redshift, columns = cluster
//   redshift bin; 0 outside each bin's support
// ---------------------------------------------------------------------------
arma::Mat<double> phi_cluster_cpp(
    const arma::Col<double> z   // true redshifts
  )
{
  check_cluster_redshift_bins("phi_cluster_cpp");
  check_not_empty("phi_cluster_cpp", "z", z);

  arma::Mat<double> result(z.n_elem, cluster.zdist_nbin, arma::fill::zeros);
  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    for (arma::uword i=0; i<z.n_elem; i++) {
      result(i, ni) = phi_cluster(z(i), ni);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Normalized true-redshift distribution of the clusters of redshift bin
// ni (and richness bin nl for the abundance-weighted kernel).
//
// Calls redshift_spline_cluster.c nz_cluster (cached table on the fine
// z grid of the selection kernel; cosmology dependent through dV/dz)
// after cluster_warmup on the calling thread.
//
// Parameters:
//   z  - true redshift
//   ni - cluster redshift bin (the C function aborts outside
//        [0, zdist_nbin))
//   nl - richness bin (validated by the C function; unused by the
//        volume kernel)
//
// Returns:
//   n(z) per unit z (integrates to 1 over z); 0 outside the bin's
//   support
// ---------------------------------------------------------------------------
double nz_cluster_cpp(
    const double z,   // true redshift
    const int ni,     // cluster redshift bin
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return nz_cluster(z, ni, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// n(z) of every cluster bin and richness bin at many redshifts: one
// warm-up, then redshift_spline_cluster.c nz_cluster in a serial loop.
//
// Parameters:
//   z - true redshifts; an empty array aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Cube (nz, zdist_nbin, richness_nbin): entry (i, ni, nl) =
//   n(z_i) of cluster bin ni and richness bin nl (the same for every nl
//   with the volume kernel); 0 outside each bin's support
// ---------------------------------------------------------------------------
arma::Cube<double> nz_cluster_cpp(
    const arma::Col<double> z   // true redshifts
  )
{
  check_cluster_redshift_bins("nz_cluster_cpp");
  check_richness_bins("nz_cluster_cpp");
  check_not_empty("nz_cluster_cpp", "z", z);
  cluster_warmup();

  arma::Cube<double> result(z.n_elem,
                            cluster.zdist_nbin,
                            cluster.richness_nbin,
                            arma::fill::zeros);
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    for (int ni=0; ni<cluster.zdist_nbin; ni++) {
      for (arma::uword i=0; i<z.n_elem; i++) {
        result(i, ni, nl) = nz_cluster(z(i), ni, nl);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lensing efficiency of the clusters of redshift bin ni (the g_lens
// convention): the kernel of cluster magnification.
//
// Calls redshift_spline_cluster.c g_cluster (cached table on a uniform
// a grid, read linearly) after cluster_warmup on the calling thread.
//
// Parameters:
//   a  - scale factor, 0 < a <= 1 (the C function aborts otherwise)
//   ni - cluster redshift bin (the C function aborts outside
//        [0, zdist_nbin))
//   nl - richness bin (validated by the C function; unused by the
//        volume kernel)
//
// Returns:
//   g(a), dimensionless: nonzero over the whole foreground of the bin,
//   0 behind the far edge of its support
// ---------------------------------------------------------------------------
double g_cluster_cpp(
    const double a,   // scale factor
    const int ni,     // cluster redshift bin
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return g_cluster(a, ni, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// g(a) of every cluster bin and richness bin at many scale factors: one
// warm-up, then redshift_spline_cluster.c g_cluster in a serial loop.
//
// Parameters:
//   a - scale factors, 0 < a <= 1 (the C function aborts otherwise); an
//       empty array aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Cube (na, zdist_nbin, richness_nbin): entry (i, ni, nl) =
//   g(a_i) of cluster bin ni and richness bin nl
// ---------------------------------------------------------------------------
arma::Cube<double> g_cluster_cpp(
    const arma::Col<double> a   // scale factors
  )
{
  check_cluster_redshift_bins("g_cluster_cpp");
  check_richness_bins("g_cluster_cpp");
  check_not_empty("g_cluster_cpp", "a", a);
  cluster_warmup();

  arma::Cube<double> result(a.n_elem,
                            cluster.zdist_nbin,
                            cluster.richness_nbin,
                            arma::fill::zeros);
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    for (int ni=0; ni<cluster.zdist_nbin; ni++) {
      for (arma::uword i=0; i<a.n_elem; i++) {
        result(i, ni, nl) = g_cluster(a(i), ni, nl);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Radial density kernel of the clusters of redshift bin ni,
// W_cluster(a) = n(z(a)) H(a)/H0 (it integrates to 1 over chi).
//
// Calls radial_weights_cluster.c W_cluster with H(a)/H0 = hoverh0(a)
// (the C signature takes it from the caller, whose Limber nodes hold
// it) after cluster_warmup on the calling thread.
//
// Parameters:
//   a  - scale factor, 0 < a < 1 (the C function aborts otherwise)
//   ni - cluster redshift bin (the C function aborts outside
//        [0, zdist_nbin))
//   nl - richness bin (validated by the C function; unused by the
//        volume kernel)
//
// Returns:
//   W_cluster(a) per unit comoving distance (chi in c/H0); 0 outside
//   the bin's support
// ---------------------------------------------------------------------------
double W_cluster_cpp(
    const double a,   // scale factor
    const int ni,     // cluster redshift bin
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return W_cluster(a, ni, nl, hoverh0(a));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// W_cluster(a) of every cluster bin and richness bin at many scale
// factors: one warm-up, then radial_weights_cluster.c W_cluster in a
// serial loop, with H(a)/H0 = hoverh0(a) computed once per a.
//
// Parameters:
//   a - scale factors, each inside (0, 1); an empty array or an entry
//       outside (0, 1) aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Cube (na, zdist_nbin, richness_nbin): entry (i, ni, nl) =
//   W_cluster(a_i) of cluster bin ni and richness bin nl
// ---------------------------------------------------------------------------
arma::Cube<double> W_cluster_cpp(
    const arma::Col<double> a   // scale factors
  )
{
  check_cluster_redshift_bins("W_cluster_cpp");
  check_richness_bins("W_cluster_cpp");
  check_not_empty("W_cluster_cpp", "a", a);
  for (arma::uword i=0; i<a.n_elem; i++) {
    check_scale_factor("W_cluster_cpp", a(i));
  }
  cluster_warmup();

  arma::Cube<double> result(a.n_elem,
                            cluster.zdist_nbin,
                            cluster.richness_nbin,
                            arma::fill::zeros);
  for (arma::uword i=0; i<a.n_elem; i++) {
    const double hoverh0_a = hoverh0(a(i));
    for (int nl=0; nl<cluster.richness_nbin; nl++) {
      for (int ni=0; ni<cluster.zdist_nbin; ni++) {
        result(i, ni, nl) = W_cluster(a(i), ni, nl, hoverh0_a);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Magnification kernel of the clusters of redshift bin ni,
// W_mag_cluster(a) = 1.5 Omega_m f_K(chi(a))/a g_cluster(a), without
// the coefficient cluster.magnification the spectra multiply it by.
//
// Calls radial_weights_cluster.c W_mag_cluster with fK = f_K(chi(a))
// (the C signature takes it from the caller, whose Limber nodes hold
// it) after cluster_warmup on the calling thread.
//
// Parameters:
//   a  - scale factor, 0 < a < 1 (the C function aborts otherwise)
//   ni - cluster redshift bin (the C function aborts outside
//        [0, zdist_nbin))
//   nl - richness bin (validated by the C function; unused by the
//        volume kernel)
//
// Returns:
//   W_mag_cluster(a): nonzero over the whole foreground of the bin
// ---------------------------------------------------------------------------
double W_mag_cluster_cpp(
    const double a,   // scale factor
    const int ni,     // cluster redshift bin
    const int nl      // richness bin
  )
{
  cluster_warmup();
  return W_mag_cluster(a, f_K(chi(a)), ni, nl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// W_mag_cluster(a) of every cluster bin and richness bin at many scale
// factors: one warm-up, then radial_weights_cluster.c W_mag_cluster in
// a serial loop, with fK = f_K(chi(a)) computed once per a.
//
// Parameters:
//   a - scale factors, each inside (0, 1); an empty array or an entry
//       outside (0, 1) aborts (spdlog::critical + exit)
//
// Returns:
//   arma::Cube (na, zdist_nbin, richness_nbin): entry (i, ni, nl) =
//   W_mag_cluster(a_i) of cluster bin ni and richness bin nl
// ---------------------------------------------------------------------------
arma::Cube<double> W_mag_cluster_cpp(
    const arma::Col<double> a   // scale factors
  )
{
  check_cluster_redshift_bins("W_mag_cluster_cpp");
  check_richness_bins("W_mag_cluster_cpp");
  check_not_empty("W_mag_cluster_cpp", "a", a);
  for (arma::uword i=0; i<a.n_elem; i++) {
    check_scale_factor("W_mag_cluster_cpp", a(i));
  }
  cluster_warmup();

  arma::Cube<double> result(a.n_elem,
                            cluster.zdist_nbin,
                            cluster.richness_nbin,
                            arma::fill::zeros);
  for (arma::uword i=0; i<a.n_elem; i++) {
    const double fK = f_K(chi(a(i)));
    for (int nl=0; nl<cluster.richness_nbin; nl++) {
      for (int ni=0; ni<cluster.zdist_nbin; ni++) {
        result(i, ni, nl) = W_mag_cluster(a(i), fK, ni, nl);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
