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
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

// cosmolike: to_np4d (the 4d numpy stacking of the galaxy wrappers)
#include "cosmolike/cosmo2D_wrapper.hpp"

// cosmolike: the cluster contract (every cluster C header, the Y transform
// matrix and the selection-bias factor of the joint data vector)
#include "cosmolike/generic_interface_cluster.hpp"
#include "cosmolike/cosmo2D_wrapper_cluster.hpp"

// ---------------------------------------------------------------------------
// Pybind-facing batch evaluators of the cluster 2D statistics: the
// Python side asks for whole data products, not per-point C calls.
//
//   Python -> *_bins_cpp overload (scalar diagnostic or array batch)
//     -> *_nointerp_ells / w_*_tomo / N_cluster_tomo engines
//        (cosmo2D_cluster.c)
//     -> numpy arrays (ell-or-theta, bin, bin, bin), stacked by to_np4d
//
// Layout conventions: rows = angular bin or multipole; the trailing
// axes are the bins in the argument order of the C engines,
//
//   cs (cluster lensing)    (nt or l, nl, ni, ns)
//   cc (cluster clustering) (nt or l, nl1, nl2, ni)
//   cg (cluster x galaxy)   (nt or l, nl, ni, ng)
//   N  (counts)             (nl, ni)
//
// with nl = richness bin, ni = cluster redshift bin, ns = source bin,
// ng = lens bin. Only the enumerated pairs are filled: every
// (ZC_cs(n), ZS_cs(n)) for cs, the (ZC_cg(n), ZG_cg(n)) set by
// init_cluster_pairs for cg, and both orderings of every richness pair
// for cc (the statistic is symmetric in nl1 <-> nl2) - all other
// entries stay zero. The scalar overloads are point diagnostics and
// pay the full batch cost per call (see each header).
//
// Cluster lensing comes in two forms. w_gammat_cluster_tomo_bins_cpp is
// the tangential shear gamma_t of the C engine: BEFORE the Y transform
// (eq 15 of arXiv 2503.13631), the selection bias (eq 23) and the
// shear calibration (1 + m), which generic_interface_cluster.cpp
// applies on the data vector. w_sigma_cluster_tomo_bins_cpp applies
// those three steps with the interface's own matrices, so it is what
// the cs block of the data vector holds (without the mask). w_cc and
// w_cg are returned before the selection bias too
// (compute_cluster_selection_factor gives the factor).
//
// Thread safety: the cluster tables are filled lazily and a refill is
// not thread-safe (halo_cluster.h, redshift_spline_cluster.h). Every
// evaluator below first warms the pair maps and calls cluster_warmup()
// on the calling thread, then loops serially over the engine, whose
// first call computes and caches the whole block (its OpenMP loops only
// read the warmed tables).
// ---------------------------------------------------------------------------
namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// STATE CHECKS AND WARM-UP (PRIVATE)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the cluster redshift bins and the richness bins are set.
//
// The wrappers size their arrays with cluster.zdist_nbin and
// cluster.richness_nbin before any engine (and its own checks) runs.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//
// Returns:
//   nothing; missing bins abort (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_cluster_bins(const char* fname)
{
  if (!(cluster.zdist_nbin > 0) || NULL == cluster.zdist_table) {
    spdlog::critical("{}: cluster selection kernels not set (call "
                     "set_cluster_zdist first)", fname);
    exit(1);
  }
  if (!(cluster.richness_nbin > 0)) {
    spdlog::critical("{}: richness bins not set (call "
                     "init_cluster_richness_bins first)", fname);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Build the pair maps of redshift_spline_cluster.c on the calling
// thread (private copy of warmup_cluster_pair_maps of
// generic_interface_cluster.cpp, which is static there).
//
// Any accessor runs the builder, which rebuilds on a new
// cluster.random_pairs or bin count and writes
// cluster.cs/cg/cc_npowerspectra; every reader of those counts calls
// this first. The calls respect each accessor's range checks.
//
// Parameters:
//   none (reads the bin counts of cluster and redshift)
//
// Returns:
//   nothing; the pair maps and the pair counts are up to date
// ---------------------------------------------------------------------------
static void warmup_pair_maps()
{
  if (cluster.zdist_nbin > 0 && redshift.shear_nbin > 0) {
    (void) N_cs(0, 0);
  }
  if (cluster.richness_nbin > 0) {
    (void) N_cc_richness(0, 0);
  }
  if (cluster.cg_npowerspectra > 0) {  // count written by the builder above
    (void) ZC_cg(0);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Everything an evaluator needs before its first engine call: the bin
// checks, the pair maps, and every lazily filled cluster table
// (cluster_warmup: the n_nl, b_nl and P1h tables, the selection kernel,
// the cluster n(z) and its lensing efficiency), all on the calling
// thread. The cosmology and the cluster nuisance parameters must be set.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//
// Returns:
//   nothing; an unset cluster sample aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void warmup_cluster_state(const char* fname)
{
  check_cluster_bins(fname);
  warmup_pair_maps();
  cluster_warmup();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the angular binning is set.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//
// Returns:
//   nothing; Ntable.Ntheta = 0 aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_binning_real_space(const char* fname)
{
  if (!(Ntable.Ntheta > 0)) {
    spdlog::critical("{}: angular binning not set (call init_binning "
                     "first)", fname);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless limber is 0 or 1.
//
// The flag is an argument because the interface's cc / cg Limber
// switches (init_cluster_adopt_limber) are private to
// generic_interface_cluster.cpp. It is passed through to the engine:
// limber = 0 aborts there until the non-Limber w_cc / w_cg exist.
//
// Parameters:
//   fname  - name of the calling wrapper, for the message
//   limber - 1 = Limber, 0 = non-Limber
//
// Returns:
//   nothing; any other value aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_limber_flag(const char* fname, const int limber)
{
  if (!(0 == limber || 1 == limber)) {
    spdlog::critical("{}: limber = {} not supported (0 or 1)", fname, limber);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the multipole array is non-empty.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   l     - multipole values
//
// Returns:
//   nothing; an empty array aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_multipoles(const char* fname, const arma::Col<double>& l)
{
  if (!(l.n_elem > 0)) {
    spdlog::critical("{}: l array size = {}", fname, l.n_elem);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One zero cube per row (angular bin or multipole): the work layout
// that to_np4d stacks into a (nrows, n1, n2, n3) numpy array.
//
// Parameters:
//   nrows      - number of angular bins or multipoles
//   n1, n2, n3 - the three bin counts of the trailing axes
//
// Returns:
//   field of nrows cubes of shape (n1, n2, n3), all zeros
// ---------------------------------------------------------------------------
static arma::field<arma::Cube<double>> zero_cubes(
    const int nrows,
    const int n1,
    const int n2,
    const int n3
  )
{
  arma::field<arma::Cube<double>> result(nrows);
  for (int i=0; i<nrows; i++) {
    result(i).zeros(n1, n2, n3);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// TOMOGRAPHIC PAIR LISTS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The (cluster, source) bin indices of every cluster-lensing pair,
// stored as doubles (the cluster analog of gs_bins).
//
// Parameters:
//   none (reads cluster.cs_npowerspectra and the ZC_cs/ZS_cs maps)
//
// Returns:
//   arma::Mat (cs_npowerspectra, 2) with row n = (ZC_cs(n), ZS_cs(n))
// ---------------------------------------------------------------------------
arma::Mat<double> cs_bins()
{
  check_cluster_bins("cs_bins");
  warmup_pair_maps();
  arma::Mat<double> result(cluster.cs_npowerspectra, 2);
  for (int n=0; n<cluster.cs_npowerspectra; n++) {
    result(n,0) = ZC_cs(n);
    result(n,1) = ZS_cs(n);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The (cluster, lens) bin indices of every cluster x galaxy pair,
// stored as doubles: cluster bin ni with the lens bin
// cluster.cg_lens_bin[ni] set by init_cluster_pairs.
//
// Parameters:
//   none (reads cluster.cg_npowerspectra and the ZC_cg/ZG_cg maps)
//
// Returns:
//   arma::Mat (cg_npowerspectra, 2) with row n = (ZC_cg(n), ZG_cg(n))
// ---------------------------------------------------------------------------
arma::Mat<double> cg_bins()
{
  check_cluster_bins("cg_bins");
  warmup_pair_maps();
  arma::Mat<double> result(cluster.cg_npowerspectra, 2);
  for (int n=0; n<cluster.cg_npowerspectra; n++) {
    result(n,0) = ZC_cg(n);
    result(n,1) = ZG_cg(n);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The richness bins of every richness pair nl1 <= nl2 of the w_cc block
// of one cluster redshift bin, stored as doubles.
//
// Parameters:
//   none (reads cluster.richness_nbin and the NL1_cc/NL2_cc maps)
//
// Returns:
//   arma::Mat (R(R+1)/2, 2), R = richness_nbin, with row
//   n = (NL1_cc(n), NL2_cc(n))
// ---------------------------------------------------------------------------
arma::Mat<double> cc_richness_bins()
{
  check_cluster_bins("cc_richness_bins");
  warmup_pair_maps();
  const int npairs = cluster.richness_nbin*(cluster.richness_nbin + 1)/2;
  arma::Mat<double> result(npairs, 2);
  for (int n=0; n<npairs; n++) {
    result(n,0) = NL1_cc(n);
    result(n,1) = NL2_cc(n);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// REAL-SPACE STATISTICS AND NUMBER COUNTS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster tangential shear gamma_t at every angular bin, richness bin
// and (cluster, source) pair.
//
// This is gamma_t BEFORE the Y transform (eq 15 of arXiv 2503.13631),
// the selection bias (eq 23) and the shear calibration (1 + m): the
// interface applies the three on the data vector
// (w_sigma_cluster_tomo_bins_cpp returns that form).
//
// Engine: w_gammat_cluster_tomo(nt, nl, ni, ns) (full-sky, bin-averaged
// spin-2 Legendre sum of the Limber C_cs; Limber is the only option).
// Serial loop: the first engine call computes and caches the whole
// (pair, richness, theta) block.
//
// Parameters:
//   none (reads Ntable.Ntheta, cluster.richness_nbin, cluster.zdist_nbin,
//   redshift.shear_nbin, cluster.cs_npowerspectra)
//
// Returns:
//   numpy array (Ntheta, richness_nbin, zdist_nbin, shear_nbin): rows =
//   angular bin, entry (i, nl, ZC_cs(n), ZS_cs(n)) for every cs pair n
//   (every (cluster, source) pair is enumerated)
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> w_gammat_cluster_tomo_bins_cpp()
{
  check_binning_real_space("w_gammat_cluster_tomo_bins_cpp");
  warmup_cluster_state("w_gammat_cluster_tomo_bins_cpp");

  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;
  arma::field<arma::Cube<double>> result = zero_cubes(ntheta,
                                                      nrichness,
                                                      cluster.zdist_nbin,
                                                      redshift.shear_nbin);
  for (int n=0; n<cluster.cs_npowerspectra; n++) {
    const int ni = ZC_cs(n);
    const int ns = ZS_cs(n);
    for (int nl=0; nl<nrichness; nl++) {
      for (int i=0; i<ntheta; i++) {
        result(i)(nl, ni, ns) = w_gammat_cluster_tomo(i, nl, ni, ns);
      }
    }
  }
  return to_np4d(result);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster lensing as the cs block of the joint data vector holds it, at
// every angular bin, richness bin and (cluster, source) pair, without
// the mask.
//
// The steps of compute_cs_block_masked (generic_interface_cluster.cpp),
// row by row (one row = one (pair, richness bin)), with the interface's
// own matrices (the Y transform is not re-implemented here):
//
//   gamma_t(theta_k)                          w_gammat_cluster_tomo
//     -> Sigma_i = sum_k T_ik gamma_t(theta_k)   when cluster.ytransform = 1
//        (T = compute_cluster_ytransform_matrix, eq 15 of arXiv
//        2503.13631); Sigma = gamma_t otherwise (the Y1 choice)
//     -> times B(ni, theta_i)                  compute_cluster_selection_factor
//        (eq 23; ones unless CLUSTER_SELECTION_Y6)
//     -> times (1 + m_ns)                      nuisance.shear_calibration_m
//
// The last row of T is zero, so with the Y transform the last angular
// bin is identically zero (the interface always masks it). The matrix
// needs at least five angular bins (compute_cluster_ytransform_matrix
// aborts below that).
//
// Parameters:
//   none (reads the state of w_gammat_cluster_tomo_bins_cpp,
//   cluster.ytransform, the selection-bias and shear-calibration
//   parameters)
//
// Returns:
//   numpy array (Ntheta, richness_nbin, zdist_nbin, shear_nbin): rows =
//   angular bin, entry (i, nl, ZC_cs(n), ZS_cs(n)) for every cs pair n
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> w_sigma_cluster_tomo_bins_cpp()
{
  check_binning_real_space("w_sigma_cluster_tomo_bins_cpp");
  warmup_cluster_state("w_sigma_cluster_tomo_bins_cpp");

  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;
  const bool ytransform = (1 == cluster.ytransform);

  const arma::Mat<double> selection = compute_cluster_selection_factor();
  arma::Mat<double> T;
  if (ytransform) {
    T = compute_cluster_ytransform_matrix();
  }

  arma::field<arma::Cube<double>> result = zero_cubes(ntheta,
                                                      nrichness,
                                                      cluster.zdist_nbin,
                                                      redshift.shear_nbin);
  arma::Col<double> gammat(ntheta, arma::fill::zeros);
  arma::Col<double> sigma(ntheta, arma::fill::zeros);

  for (int n=0; n<cluster.cs_npowerspectra; n++) {
    const int ni = ZC_cs(n);
    const int ns = ZS_cs(n);
    const double shear_calib = 1.0 + nuisance.shear_calibration_m[ns];
    for (int nl=0; nl<nrichness; nl++) {
      for (int i=0; i<ntheta; i++) {
        gammat(i) = w_gammat_cluster_tomo(i, nl, ni, ns);
      }
      // same expressions as compute_cs_block_masked, so unmasked entries
      // of the data vector are reproduced bit by bit
      if (ytransform) {
        sigma = T*gammat;
      }
      else {
        sigma = gammat;
      }
      for (int i=0; i<ntheta; i++) {
        result(i)(nl, ni, ns) = selection(ni, i)*sigma(i)*shear_calib;
      }
    }
  }
  return to_np4d(result);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster-cluster angular correlation w_cc at every angular bin,
// richness pair and cluster redshift bin (auto z bin), before the
// selection bias (the data vector multiplies by B(theta)^2, eq 23).
//
// Engine: w_cc_tomo(nt, nl1, nl2, ni, limber) (full-sky, bin-averaged
// spin-0 Legendre sum of C_cc). Serial loop over the richness pairs
// (NL1_cc(n), NL2_cc(n)) with nl1 <= nl2; both orderings are filled
// (w_cc is symmetric in nl1 <-> nl2).
//
// Parameters:
//   limber - 1 = Limber at every multipole; 0 = non-Limber, which the
//            engine does not implement yet (it aborts); any other value
//            aborts here (spdlog::critical + exit)
//
// Returns:
//   numpy array (Ntheta, richness_nbin, richness_nbin, zdist_nbin):
//   rows = angular bin, entries (i, nl1, nl2, ni) and (i, nl2, nl1, ni)
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> w_cc_tomo_bins_cpp(
    const int limber   // 1 = Limber; 0 = non-Limber
  )
{
  check_limber_flag("w_cc_tomo_bins_cpp", limber);
  check_binning_real_space("w_cc_tomo_bins_cpp");
  warmup_cluster_state("w_cc_tomo_bins_cpp");

  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;
  const int npairs = nrichness*(nrichness + 1)/2;
  arma::field<arma::Cube<double>> result = zero_cubes(ntheta,
                                                      nrichness,
                                                      nrichness,
                                                      cluster.zdist_nbin);
  for (int ni=0; ni<cluster.cc_npowerspectra; ni++) {
    for (int n=0; n<npairs; n++) {
      const int nl1 = NL1_cc(n);
      const int nl2 = NL2_cc(n);
      for (int i=0; i<ntheta; i++) {
        result(i)(nl1, nl2, ni) = w_cc_tomo(i, nl1, nl2, ni, limber);
        result(i)(nl2, nl1, ni) = result(i)(nl1, nl2, ni);
      }
    }
  }
  return to_np4d(result);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster x galaxy angular correlation w_cg at every angular bin,
// richness bin and (cluster, lens) pair, before the selection bias (the
// data vector multiplies by B(theta), eq 23).
//
// Engine: w_cg_tomo(nt, nl, ni, ng, limber) (full-sky, bin-averaged
// spin-0 Legendre sum of C_cg). Serial loop over the pairs
// (ZC_cg(n), ZG_cg(n)): cluster bin ni with the lens bin
// cluster.cg_lens_bin[ni] of init_cluster_pairs.
//
// Parameters:
//   limber - 1 = Limber at every multipole; 0 = non-Limber, which the
//            engine does not implement yet (it aborts); any other value
//            aborts here (spdlog::critical + exit)
//
// Returns:
//   numpy array (Ntheta, richness_nbin, zdist_nbin, clustering_nbin):
//   rows = angular bin, entry (i, nl, ZC_cg(n), ZG_cg(n)) filled for
//   the enumerated cg pairs only, everything else stays zero
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> w_cg_tomo_bins_cpp(
    const int limber   // 1 = Limber; 0 = non-Limber
  )
{
  check_limber_flag("w_cg_tomo_bins_cpp", limber);
  check_binning_real_space("w_cg_tomo_bins_cpp");
  warmup_cluster_state("w_cg_tomo_bins_cpp");

  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;
  arma::field<arma::Cube<double>> result = zero_cubes(ntheta,
                                                      nrichness,
                                                      cluster.zdist_nbin,
                                                      redshift.clustering_nbin);
  for (int n=0; n<cluster.cg_npowerspectra; n++) {
    const int ni = ZC_cg(n);
    const int ng = ZG_cg(n);
    for (int nl=0; nl<nrichness; nl++) {
      for (int i=0; i<ntheta; i++) {
        result(i)(nl, ni, ng) = w_cg_tomo(i, nl, ni, ng, limber);
      }
    }
  }
  return to_np4d(result);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Expected number of clusters in every richness bin and cluster
// redshift bin (eq 16 of arXiv 2503.13631): no selection bias and no
// calibration, the N block of the data vector as it is.
//
// Engine: N_cluster_tomo(nl, ni), a cached table of every (ni, nl)
// (survey.area in deg^2 must be set: init_survey_parameters). Serial
// loop: the first engine call fills the table.
//
// Parameters:
//   none (reads cluster.richness_nbin, cluster.zdist_nbin, survey.area)
//
// Returns:
//   arma::Mat (richness_nbin, zdist_nbin): rows = richness bin, columns
//   = cluster redshift bin (the argument order of the engine; the data
//   vector stores the transpose, [cluster z bin][richness])
// ---------------------------------------------------------------------------
arma::Mat<double> N_cluster_tomo_bins_cpp()
{
  warmup_cluster_state("N_cluster_tomo_bins_cpp");

  arma::Mat<double> result(cluster.richness_nbin,
                           cluster.zdist_nbin,
                           arma::fill::zeros);
  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    for (int nl=0; nl<cluster.richness_nbin; nl++) {
      result(nl, ni) = N_cluster_tomo(nl, ni);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// FOURIER SPACE: CLUSTER LENSING
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two C_cs_tomo_limber_bins_cpp overloads: a
// single C_cs_tomo_limber_nointerp_ells call fills every cs pair and
// richness bin at every multipole (out[n][nl][i], pair
// n = (ZC_cs(n), ZS_cs(n))), and the values are scattered into one
// (nl, ni, ns) cube per multipole.
//
// Parameters:
//   fname - name of the calling wrapper, for the messages
//   l     - multipole values (need not be integers)
//
// Returns:
//   field of nell cubes (richness_nbin, zdist_nbin, shear_nbin); all
//   zeros when no cs pair exists (no source bins)
// ---------------------------------------------------------------------------
static arma::field<arma::Cube<double>> C_cs_tomo_limber_cubes(
    const char* fname,            // calling wrapper
    const arma::Col<double>& l    // multipole values
  )
{
  warmup_cluster_state(fname);

  const int nell = static_cast<int>(l.n_elem);
  const int npairs = cluster.cs_npowerspectra;
  const int nrichness = cluster.richness_nbin;
  arma::field<arma::Cube<double>> result = zero_cubes(nell,
                                                      nrichness,
                                                      cluster.zdist_nbin,
                                                      redshift.shear_nbin);
  if (0 == npairs) {
    return result;
  }
  double*** tmp = (double***) malloc3d(npairs, nrichness, nell);
  C_cs_tomo_limber_nointerp_ells(l.memptr(), nell, tmp);
  for (int n=0; n<npairs; n++) {
    const int ni = ZC_cs(n);
    const int ns = ZS_cs(n);
    for (int nl=0; nl<nrichness; nl++) {
      for (int i=0; i<nell; i++) {
        result(i)(nl, ni, ns) = tmp[n][nl][i];
      }
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster-lensing Limber C_l at many multipoles, filled by
// C_cs_tomo_limber_cubes (one batched C_cs_tomo_limber_nointerp_ells
// call: the exact Limber quadrature at each multipole, no table
// interpolation). The spectrum of the gamma_t above: no selection bias
// and no shear calibration.
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   numpy array (nell, richness_nbin, zdist_nbin, shear_nbin): rows =
//   multipole, entry (i, nl, ZC_cs(n), ZS_cs(n)) for every cs pair n
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> C_cs_tomo_limber_bins_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  check_multipoles("C_cs_tomo_limber_bins_cpp", l);
  return to_np4d(C_cs_tomo_limber_cubes("C_cs_tomo_limber_bins_cpp", l));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster-lensing Limber C_l at one multipole for richness bin nl,
// cluster bin ni and source bin ns.
//
// Point diagnostic: runs the full batch of C_cs_tomo_limber_cubes at a
// single multipole and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (l, nl, ni, ns) should call the array
// overload once and index the returned array instead.
//
// Parameters:
//   l  - multipole
//   nl - richness bin; outside [0, richness_nbin) aborts
//        (spdlog::critical + exit)
//   ni - cluster redshift bin; outside [0, zdist_nbin) aborts
//   ns - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   C_l^cs of the (nl, ni, ns) entry
// ---------------------------------------------------------------------------
double C_cs_tomo_limber_bins_cpp(
    const double l,   // multipole
    const int nl,     // richness bin
    const int ni,     // cluster redshift bin
    const int ns      // source redshift bin
  )
{
  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1 ||
      ns < 0 || ns > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (nl, ni, ns) = ({}, {}, {})",
                     "C_cs_tomo_limber_bins_cpp", nl, ni, ns);
    exit(1);
  }
  arma::Col<double> ell(1);
  ell(0) = l;
  const arma::field<arma::Cube<double>> res =
    C_cs_tomo_limber_cubes("C_cs_tomo_limber_bins_cpp", ell);
  return res(0)(nl, ni, ns);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// FOURIER SPACE: CLUSTER CLUSTERING
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two C_cc_tomo_limber_bins_cpp overloads: a
// single C_cc_tomo_limber_nointerp_ells call fills every cluster bin
// and richness pair at every multipole (out[ni][n][i], richness pair
// n = (NL1_cc(n), NL2_cc(n)) with nl1 <= nl2), and the values are
// scattered into one (nl1, nl2, ni) cube per multipole, both orderings
// of the richness pair filled.
//
// Parameters:
//   fname - name of the calling wrapper, for the messages
//   l     - multipole values (need not be integers)
//
// Returns:
//   field of nell cubes (richness_nbin, richness_nbin, zdist_nbin)
// ---------------------------------------------------------------------------
static arma::field<arma::Cube<double>> C_cc_tomo_limber_cubes(
    const char* fname,            // calling wrapper
    const arma::Col<double>& l    // multipole values
  )
{
  warmup_cluster_state(fname);

  const int nell = static_cast<int>(l.n_elem);
  const int nbins = cluster.cc_npowerspectra;
  const int nrichness = cluster.richness_nbin;
  const int npairs = nrichness*(nrichness + 1)/2;
  arma::field<arma::Cube<double>> result = zero_cubes(nell,
                                                      nrichness,
                                                      nrichness,
                                                      cluster.zdist_nbin);
  if (0 == nbins) {
    return result;
  }
  double*** tmp = (double***) malloc3d(nbins, npairs, nell);
  C_cc_tomo_limber_nointerp_ells(l.memptr(), nell, tmp);
  for (int ni=0; ni<nbins; ni++) {
    for (int n=0; n<npairs; n++) {
      const int nl1 = NL1_cc(n);
      const int nl2 = NL2_cc(n);
      for (int i=0; i<nell; i++) {
        result(i)(nl1, nl2, ni) = tmp[ni][n][i];
        result(i)(nl2, nl1, ni) = tmp[ni][n][i];
      }
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster-clustering Limber C_l at many multipoles (auto z bin, every
// richness pair), filled by C_cc_tomo_limber_cubes (one batched
// C_cc_tomo_limber_nointerp_ells call: the exact Limber quadrature at
// each multipole, no table interpolation). No selection bias.
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   numpy array (nell, richness_nbin, richness_nbin, zdist_nbin): rows
//   = multipole, entries (i, nl1, nl2, ni) and (i, nl2, nl1, ni)
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> C_cc_tomo_limber_bins_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  check_multipoles("C_cc_tomo_limber_bins_cpp", l);
  return to_np4d(C_cc_tomo_limber_cubes("C_cc_tomo_limber_bins_cpp", l));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster-clustering Limber C_l at one multipole for the richness pair
// (nl1, nl2), in either ordering, of cluster bin ni.
//
// Point diagnostic: runs the full batch of C_cc_tomo_limber_cubes at a
// single multipole and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (l, nl1, nl2, ni) should call the
// array overload once and index the returned array instead.
//
// Parameters:
//   l   - multipole
//   nl1 - first richness bin; outside [0, richness_nbin) aborts
//         (spdlog::critical + exit)
//   nl2 - second richness bin; same validation as nl1
//   ni  - cluster redshift bin; outside [0, zdist_nbin) aborts
//
// Returns:
//   C_l^cc of the (nl1, nl2, ni) entry
// ---------------------------------------------------------------------------
double C_cc_tomo_limber_bins_cpp(
    const double l,   // multipole
    const int nl1,    // first richness bin
    const int nl2,    // second richness bin
    const int ni      // cluster redshift bin
  )
{
  if (nl1 < 0 || nl1 > cluster.richness_nbin - 1 ||
      nl2 < 0 || nl2 > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1) {
    spdlog::critical("{}: invalid bin input (nl1, nl2, ni) = ({}, {}, {})",
                     "C_cc_tomo_limber_bins_cpp", nl1, nl2, ni);
    exit(1);
  }
  arma::Col<double> ell(1);
  ell(0) = l;
  const arma::field<arma::Cube<double>> res =
    C_cc_tomo_limber_cubes("C_cc_tomo_limber_bins_cpp", ell);
  return res(0)(nl1, nl2, ni);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// FOURIER SPACE: CLUSTER X GALAXY CLUSTERING
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two C_cg_tomo_limber_bins_cpp overloads: a
// single C_cg_tomo_limber_nointerp_ells call fills every cg pair and
// richness bin at every multipole (out[n][nl][i], pair
// n = (ZC_cg(n), ZG_cg(n))), and the values are scattered into one
// (nl, ni, ng) cube per multipole; pairs outside the enumeration stay
// zero.
//
// Parameters:
//   fname - name of the calling wrapper, for the messages
//   l     - multipole values (need not be integers)
//
// Returns:
//   field of nell cubes (richness_nbin, zdist_nbin, clustering_nbin);
//   all zeros when no cg pair exists (init_cluster_pairs paired no
//   cluster bin with a lens bin)
// ---------------------------------------------------------------------------
static arma::field<arma::Cube<double>> C_cg_tomo_limber_cubes(
    const char* fname,            // calling wrapper
    const arma::Col<double>& l    // multipole values
  )
{
  warmup_cluster_state(fname);

  const int nell = static_cast<int>(l.n_elem);
  const int npairs = cluster.cg_npowerspectra;
  const int nrichness = cluster.richness_nbin;
  arma::field<arma::Cube<double>> result = zero_cubes(nell,
                                                      nrichness,
                                                      cluster.zdist_nbin,
                                                      redshift.clustering_nbin);
  if (0 == npairs) {
    return result;
  }
  double*** tmp = (double***) malloc3d(npairs, nrichness, nell);
  C_cg_tomo_limber_nointerp_ells(l.memptr(), nell, tmp);
  for (int n=0; n<npairs; n++) {
    const int ni = ZC_cg(n);
    const int ng = ZG_cg(n);
    for (int nl=0; nl<nrichness; nl++) {
      for (int i=0; i<nell; i++) {
        result(i)(nl, ni, ng) = tmp[n][nl][i];
      }
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster x galaxy Limber C_l at many multipoles, filled by
// C_cg_tomo_limber_cubes (one batched C_cg_tomo_limber_nointerp_ells
// call: the exact Limber quadrature at each multipole, no table
// interpolation). No selection bias.
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   numpy array (nell, richness_nbin, zdist_nbin, clustering_nbin):
//   rows = multipole, entry (i, nl, ZC_cg(n), ZG_cg(n)) filled for the
//   enumerated cg pairs only
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> C_cg_tomo_limber_bins_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  check_multipoles("C_cg_tomo_limber_bins_cpp", l);
  return to_np4d(C_cg_tomo_limber_cubes("C_cg_tomo_limber_bins_cpp", l));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cluster x galaxy Limber C_l at one multipole for richness bin nl,
// cluster bin ni and lens bin ng.
//
// Point diagnostic: runs the full batch of C_cg_tomo_limber_cubes at a
// single multipole and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (l, nl, ni, ng) should call the array
// overload once and index the returned array instead.
//
// Parameters:
//   l  - multipole
//   nl - richness bin; outside [0, richness_nbin) aborts
//        (spdlog::critical + exit)
//   ni - cluster redshift bin; outside [0, zdist_nbin) aborts
//   ng - lens redshift bin; outside [0, clustering_nbin) aborts
//
// Returns:
//   C_l^cg of the (nl, ni, ng) entry; 0 for a (ni, ng) outside the
//   enumerated cg list
// ---------------------------------------------------------------------------
double C_cg_tomo_limber_bins_cpp(
    const double l,   // multipole
    const int nl,     // richness bin
    const int ni,     // cluster redshift bin
    const int ng      // lens redshift bin
  )
{
  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1 ||
      ng < 0 || ng > redshift.clustering_nbin - 1) {
    spdlog::critical("{}: invalid bin input (nl, ni, ng) = ({}, {}, {})",
                     "C_cg_tomo_limber_bins_cpp", nl, ni, ng);
    exit(1);
  }
  arma::Col<double> ell(1);
  ell(0) = l;
  const arma::field<arma::Cube<double>> res =
    C_cg_tomo_limber_cubes("C_cg_tomo_limber_bins_cpp", ell);
  return res(0)(nl, ni, ng);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
