#include <stdint.h>
#include "structs.h"

#ifndef __COSMOLIKE_STRUCTS_CLUSTER_H
#define __COSMOLIKE_STRUCTS_CLUSTER_H
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// [SECTION] CLUSTER STATE FOR THE 4x2pt + N ANALYSIS
// ============================================================================
//
// Every piece of cluster state lives in the single global `cluster` below,
// so the core structs (structs.h) carry no cluster fields.
//
// Model: DES Y6 methods paper, arXiv 2503.13631 (equation numbers below),
// with switches that recover the DES Y1 choices of arXiv 2008.10757.
//
// Units (the library's): comoving distance chi in c/H0, wavenumber k in
// (c/H0)^-1, halo mass M in Msun/h (M200m), number densities in (c/H0)^-3,
// survey area (survey.area) in deg^2.
//
// Index names used by every cluster function:
//   ni, nj = cluster redshift (z_lambda) bin   (0 .. zdist_nbin - 1)
//   nl     = observed-richness (lambda_obs) bin (0 .. richness_nbin - 1)
//   ns     = source redshift bin, ng = lens (galaxy) redshift bin

// mass-observable relation models (cluster.mor_model)
#define CLUSTER_MOR_LOGNORMAL 0   // eqs (18)-(19), Buzzard/Cardinal form

// radial kernel of clusters in the 2pt functions (cluster.kernel_mode)
#define CLUSTER_KERNEL_VOLUME 0   // q_i(z) ~ dV/dz <phi_i|z> (Y1 eq 15)
#define CLUSTER_KERNEL_ABUNDANCE 1 // q_iA(z) ~ dV/dz <phi_i|z> n_A(z)

// selection-bias models (cluster.selection_model)
#define CLUSTER_SELECTION_NONE 0
#define CLUSTER_SELECTION_Y1 1     // b_s0 (M/M_piv)^b_s1 ((1+z)/1.45)^b_s2
                                   // inside the bias mass integral (Y1 eq 1)
#define CLUSTER_SELECTION_Y6 2     // b_s1 + b_s2 exp(-theta chi(zbar)/r0)
                                   // on the data vector (eq 23)

// amplitude alpha of the Tinker et al. 2010 multiplicity f(nu) in the
// cluster mass function (cluster.hmf_alpha_mode). Both modes use the same
// shape, 1001.3162 Eqs. 8-12 with the Table 4 parameters at Delta = 200
// (mean), evolved in z and frozen at z = 3 (a floored at 0.25):
//   beta = 0.589 a^-0.2, gamma = 0.864 a^0.01, phi = -0.729 a^0.08,
//   eta = -0.243 a^-0.27
// They differ in alpha only, so n_nl and the counts scale with it while
// b_nl and P1h_nl (ratios over the mass function) do not.
#define CLUSTER_HMF_ALPHA_FIXED 0  // alpha = 0.368 at every z, Table 4 at
                                   // Delta = 200: the convention of the DES
                                   // cluster analyses (lighthouse code)
#define CLUSTER_HMF_ALPHA_NORMALIZED 1 // alpha(a) from int b(nu) f(nu) dnu
                                   // = 1 (1001.3162 Eq. 7), halo.c's fnu:
                                   // 0.3684 at z = 0, falling with z;
                                   // alpha/0.368 = 0.967, 0.951, 0.936,
                                   // 0.923, 0.909 at z = 0.2, 0.3, 0.4,
                                   // 0.5, 0.6 (counts 3-9% below mode 0)

// Slots of cluster.probe[]: 1 = the probe is part of the data vector
#define CLUSTER_PROBE_N 0    // cluster counts
#define CLUSTER_PROBE_CS 1   // cluster lensing (gamma_t or Sigma)
#define CLUSTER_PROBE_CC 2   // cluster-cluster clustering w_cc
#define CLUSTER_PROBE_CG 3   // cluster-galaxy clustering w_cg
#define NCLUSTER_PROBES 4

typedef struct
{
  // ---------------------------------------------------------------------------
  // CACHE KEYS
  // ---------------------------------------------------------------------------
  // uint64 counters drawn from RandomNumber (generic_interface.hpp) by the
  // cluster setters, and only when a value actually changed (fdiff). A table
  // stores the keys it was built with and refills when any differs.
  uint64_t random_model;     // model choices and richness binning
  uint64_t random_zdist;     // selection kernels <phi_i|z> and bin edges
  uint64_t random_mor;       // mass-observable relation parameters
  uint64_t random_selection; // selection-bias parameters
  uint64_t random_pairs;     // cs / cg / cc pair lists

  // ---------------------------------------------------------------------------
  // MODEL CHOICES
  // ---------------------------------------------------------------------------
  int mor_model;             // CLUSTER_MOR_*
  int kernel_mode;           // CLUSTER_KERNEL_*
  int selection_model;       // CLUSTER_SELECTION_*
  int hmf_alpha_mode;        // CLUSTER_HMF_ALPHA_*
  int ytransform;          // 1: cluster lensing is Sigma = Y gamma_t
                             //    (eq 15, Park+2021); 0: gamma_t (Y1)
  int include_ia;            // 1: intrinsic alignments of the sources in
                             //    the 2-halo cluster-lensing term
  double magnification;      // cluster magnification coefficient C_c
                             //    (eq 28: -2); 0 switches it off
  double mor_pivot_mass;     // M_piv of eq (19), Msun/h
  double mor_pivot_1pz;      // (1 + z_piv) of eq (19)

  // ---------------------------------------------------------------------------
  // PROBES IN THE DATA VECTOR
  // ---------------------------------------------------------------------------
  int probe[NCLUSTER_PROBES]; // 1 = in the data vector; slots CLUSTER_PROBE_*

  // ---------------------------------------------------------------------------
  // OBSERVED-RICHNESS BINS
  // ---------------------------------------------------------------------------
  int richness_nbin;
  double richness[2][MAX_SIZE_ARRAYS];  // [RANGE_MIN|RANGE_MAX][nl]: the
                                        // lambda_obs edges of bin nl

  // ---------------------------------------------------------------------------
  // CLUSTER REDSHIFT BINS: SELECTION KERNELS <phi_i|z_true>
  // ---------------------------------------------------------------------------
  // <phi_i|z> = probability that a cluster at true redshift z has its
  // photometric redshift z_lambda inside bin i (Y1 eq 6). Passed as a table
  // from Python (top-hat, erf of a Gaussian photo-z, or from randoms), laid
  // out like redshift.clustering_zdist_table: rows 0 .. zdist_nbin - 1 hold
  // the bins, row zdist_nbin holds z. The z values are sample points (no
  // half-cell offset, unlike the galaxy n(z) files) and <phi_i|z> is their
  // piecewise-linear interpolant.
  int zdist_nbin;
  int zdist_nz;                          // number of z rows of the input
  double** zdist_table;                  // [zdist_nbin + 1][zdist_nz]
  double zdist_zall[2];                  // [RANGE_MIN, RANGE_MAX] of the table
  // support of <phi_i|z>: the zero nodes bracketing the nonzero values of
  // each column (zmin > 0), so a top-hat edge is kept exactly
  double zdist_z[2][MAX_SIZE_ARRAYS];    // [RANGE_MIN|RANGE_MAX][bin]
  // nominal z_lambda edges of each bin: the selection-bias zbar and the
  // physical scale cuts read these; a kernel table never overwrites them
  double zbin[2][MAX_SIZE_ARRAYS];       // [RANGE_MIN|RANGE_MAX][bin]

  // ---------------------------------------------------------------------------
  // TOMOGRAPHIC PAIRS
  // ---------------------------------------------------------------------------
  int cs_npowerspectra;                  // (cluster bin, source bin) pairs
  int cg_npowerspectra;                  // (cluster bin, lens bin) pairs
  int cg_lens_bin[MAX_SIZE_ARRAYS];      // lens bin paired with cluster bin
                                         // ni in w_cg (-1: none)
  int cc_npowerspectra;                  // cluster bins in w_cc (auto only;
                                         // each holds R(R+1)/2 richness
                                         // pairs, R = richness_nbin)
  // the three pair counts above are written by the pair maps of
  // redshift_spline_cluster.c (their single owner)

  // ---------------------------------------------------------------------------
  // NUISANCE PARAMETERS
  // ---------------------------------------------------------------------------
  // mass-observable relation (lighthouse order):
  //   mor[0] = ln lambda_0, mor[1] = A_lambda (slope in ln M),
  //   mor[2] = sigma_int,   mor[3] = B_lambda (slope in ln (1+z))
  double mor[MAX_SIZE_ARRAYS];
  // selection bias:
  //   CLUSTER_SELECTION_Y6: [0] = b_s1, [1] = b_s2, [2] = r_0 (comoving
  //                         Mpc/h), [3] = power of (1+zbar)/1.45 (0 in the
  //                         paper; lighthouse s3)
  //   CLUSTER_SELECTION_Y1: [0] = b_s0, [1] = b_s1 (mass slope),
  //                         [2] = b_s2 (power of (1+z)/1.45; Y1 eq 31)
  double selection[MAX_SIZE_ARRAYS];

  // ---------------------------------------------------------------------------
  // INTEGRATION LIMITS
  // ---------------------------------------------------------------------------
  double m[2];                           // mass range [RANGE_MIN, RANGE_MAX]
                                         // of the cluster integrals, Msun/h
} clusterparams;

extern clusterparams cluster;

void reset_cluster_struct(void);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
