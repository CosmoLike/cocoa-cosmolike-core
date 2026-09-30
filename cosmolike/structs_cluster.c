#include <stdlib.h>
#include "structs_cluster.h"

clusterparams cluster =
{
  .zdist_table = NULL
};

// ============================================================================
// [SECTION] RESET
// ============================================================================
//
// Defaults are the DES Y6 methods-paper model (arXiv 2503.13631): lognormal
// MOR with M_piv = 5e14 Msun/h and 1 + z_piv = 1.45 (eq 19; M_piv is not
// printed in the paper, it is the value of the lighthouse code the DES
// analyses ran), volume-only cluster kernel (what DES ran), scale-dependent
// selection bias, Y transform on, NLA-type IA in the 2-halo lensing term,
// C_c = -2 magnification. The mass range [1e12, 1e16] Msun/h brackets every
// halo that can reach lambda_obs >= 20.
void reset_cluster_struct(void)
{
  cluster.random_model = 0;
  cluster.random_zdist = 0;
  cluster.random_mor = 0;
  cluster.random_selection = 0;
  cluster.random_pairs = 0;

  cluster.mor_model = CLUSTER_MOR_LOGNORMAL;
  cluster.kernel_mode = CLUSTER_KERNEL_VOLUME;
  cluster.selection_model = CLUSTER_SELECTION_Y6;
  cluster.ytransform = 1;
  cluster.include_ia = 1;
  cluster.magnification = -2.0;
  cluster.mor_pivot_mass = 5.0e14;
  cluster.mor_pivot_1pz = 1.45;

  cluster.probe_N = 0;
  cluster.probe_cs = 0;
  cluster.probe_cc = 0;
  cluster.probe_cg = 0;

  cluster.richness_nbin = 0;

  cluster.zdist_nbin = 0;
  cluster.zdist_nz = 0;
  if (cluster.zdist_table != NULL) {
    free(cluster.zdist_table);
    cluster.zdist_table = NULL;
  }
  cluster.zdist_zmin_all = 0.0;
  cluster.zdist_zmax_all = 0.0;

  cluster.cs_npowerspectra = 0;
  cluster.cg_npowerspectra = 0;
  cluster.cc_npowerspectra = 0;

  for (int i=0; i<MAX_SIZE_ARRAYS; i++) {
    cluster.richness_min[i] = 0.0;
    cluster.richness_max[i] = 0.0;
    cluster.zdist_zmin[i] = 0.0;
    cluster.zdist_zmax[i] = 0.0;
    cluster.zbin_min[i] = 0.0;
    cluster.zbin_max[i] = 0.0;
    cluster.cg_lens_bin[i] = -1;
    cluster.mor[i] = 0.0;
    cluster.selection[i] = 0.0;
  }

  cluster.m_min = 1.0e12;
  cluster.m_max = 1.0e16;
}
