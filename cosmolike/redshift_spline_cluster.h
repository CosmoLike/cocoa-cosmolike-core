#ifndef __COSMOLIKE_REDSHIFT_SPLINE_CLUSTER_H
#define __COSMOLIKE_REDSHIFT_SPLINE_CLUSTER_H
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// [SECTION] CLUSTER REDSHIFT DISTRIBUTIONS
// ============================================================================
//
// Inputs: cluster.zdist_table (selection kernels <phi_i|z_true>, set from
// Python), cluster.zbin[RANGE_MIN]/max, cluster.kernel_mode (structs_cluster.h).
// Index names: ni = cluster redshift bin, nl = richness bin, ns = source
// bin, ng = lens bin.

// ---------------------------------------------------------------------------
// Integration limits: the support of <phi_ni|z> in scale factor.
// ---------------------------------------------------------------------------
double amin_cluster(const int ni);
double amax_cluster(const int ni);

// ---------------------------------------------------------------------------
// Selection kernel <phi_ni|z> at true redshift z (0 outside the support),
// read from a uniform fine-z table (the nz_lens_photoz design).
// ---------------------------------------------------------------------------
double phi_cluster(const double z, const int ni);

// ---------------------------------------------------------------------------
// Normalized true-redshift distribution of clusters in bin ni and richness
// bin nl, per unit z (integrates to 1 over z):
//   CLUSTER_KERNEL_VOLUME:    n(z) = dV/dz <phi_ni|z> / int dz (same)
//                             (Y1 eq 15; nl is unused)
//   CLUSTER_KERNEL_ABUNDANCE: n(z) = dV/dz <phi_ni|z> n_nl(z) / int (same)
// dV/dz = chi^2/(H/H0) per steradian. Cosmology dependent (through dV/dz),
// so its cache keys include cosmology.random.
// ---------------------------------------------------------------------------
double nz_cluster(const double z, const int ni, const int nl);

// nominal midpoint of the z_lambda bin, (zbin_min + zbin_max)/2: the zbar
// of the selection bias (eq 23) and of the physical scale cuts
double zmid_cluster(const int ni);

// ---------------------------------------------------------------------------
// Lensing efficiency of the cluster distribution (for cluster
// magnification): g(a) = int_{a' < a} da' n(z(a')) dz/da'
//   f_K(chi(a) - chi(a')) / f_K(chi(a')), the g_lens convention.
// Nonzero at every a in front of the bin's far edge.
// ---------------------------------------------------------------------------
double g_cluster(const double a, const int ni, const int nl);

// ============================================================================
// [SECTION] TOMOGRAPHIC PAIR MAPS
// ============================================================================
//
// Rebuilt when cluster.random_pairs changes; warmed single-threaded by the
// interface before any threaded loop reads them.
//
// cluster lensing: every (cluster bin, source bin) pair, cluster-major
//   (lighthouse order); unwanted pairs are masked, not removed
int N_cs(const int ni, const int ns); // pair index, -1 if not a pair
int ZC_cs(const int n);               // cluster bin of pair n
int ZS_cs(const int n);               // source bin of pair n
// cluster-galaxy clustering: cluster bin ni with lens bin
//   cluster.cg_lens_bin[ni]
int N_cg(const int ni, const int ng);
int ZC_cg(const int n);
int ZG_cg(const int n);
// cluster-cluster clustering: auto z bin, all richness pairs nl1 <= nl2
int N_cc_richness(const int nl1, const int nl2);
int NL1_cc(const int n);
int NL2_cc(const int n);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
