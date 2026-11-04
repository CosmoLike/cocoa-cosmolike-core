#include <math.h>
#include <stdlib.h>

#include "nonlimber_cluster_cov.h"
#include "nonlimber_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo_cluster.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/redshift_spline_cluster.h"
#include "cosmolike/structs.h"
#include "cosmolike/structs_cluster.h"
#include "log.c/src/log.h"

// -----------------------------------------------------------------------
// Selected comoving abundance of one cluster category at scale factor a:
// the probability <phi_i|z> that a cluster at true redshift z lands in
// cluster redshift bin i, times the abundance n_A(a) of richness bin A,
// with category = i*nrichness+A. Units (c/H0)^-3; zero outside the
// selection support and the cluster scale-factor tables.
// -----------------------------------------------------------------------
static double selected_density(const double a, const int category)
{
  const int nrichness = cluster.richness_nbin;
  return phi_cluster(1.0/a-1.0, category/nrichness)
         *ncl_richness(a, category%nrichness);
}


// -----------------------------------------------------------------------
// Copy a galaxy/source snapshot and insert the cluster catalogs after the
// lenses, as lens-type fields:
//   window[0] = q_c b_c,  q_c = f_K^2 n_c/nbar_c,
// window[1] (magnification) and window[2] (alignment) zero. q_c is the
// normalized window of survey_cluster.selected_windows: integral q_c dchi
// = 1 on the quadrature snapshot, whose number per steradian nbar_c the
// caller passes for both snapshots, so the exact and matched terms see
// the same windows. b_c is the selected bias of richness bin A.
// Returns a heap-owned snapshot; release it with free_radial_cov.
// -----------------------------------------------------------------------
static struct radial_cov* insert_clusters(
    const struct radial_cov* base, // galaxy/source snapshot
    const int ncount,              // cluster categories
    const double* number_per_sr    // [ncount] nbar_c of the quadrature
  )
{
  struct radial_cov* joint = malloc(sizeof(*joint));
  if (joint == NULL) {
    log_fatal("cluster_nonlimber_cov: snapshot allocation failed");
    exit(1);
  }
  const int nnode = base->nnode;
  joint->nnode = nnode;
  joint->nlens = base->nlens+ncount;
  joint->nsource = base->nsource;
  const int nfield = joint->nlens+joint->nsource;
  joint->geometry = (double**) malloc2d(4, nnode);
  joint->window = (double***) malloc3d(3, nfield, nnode);
  zero3d(joint->window, 3, nfield, nnode);

  // Geometry rows a, chi, f_K, dchi are shared. Galaxy fields keep their
  // positions; source s moves from base->nlens+s to joint->nlens+s.
  for (int row=0; row<4; row++) {
    for (int node=0; node<nnode; node++) {
      joint->geometry[row][node] = base->geometry[row][node];
    }
  }
  for (int role=0; role<3; role++) {
    for (int field=0; field<base->nlens+base->nsource; field++) {
      const int target = field < base->nlens ? field : field+ncount;
      for (int node=0; node<nnode; node++) {
        joint->window[role][target][node] = base->window[role][field][node];
      }
    }
  }

  // Cluster density windows, one category at a time (serial: the cluster
  // readers were warmed by the caller and are read on this thread only).
  for (int category=0; category<ncount; category++) {
    const int field = base->nlens+category;
    for (int node=0; node<nnode; node++) {
      const double a = joint->geometry[0][node];
      const double fk = joint->geometry[2][node];
      const double q = fk*fk*selected_density(a, category)
                       /number_per_sr[category];
      joint->window[0][field][node] = q
          *bcl_richness(a, category%cluster.richness_nbin);
    }
  }
  return joint;
}


// -----------------------------------------------------------------------
// Non-Limber correction of every pair of the joint cluster forecast.
//
// The selected clusters are density tracers like the lens galaxies, so
// they enter the separable linear projection of nonlimber_pairs_cov as
// further lens-type fields: an exact spherical-Bessel transfer
//   F_c(l,k) = integral dchi q_c b_c D j_l(k chi)
//              [- integral dchi q_c f D j_l''(k chi), with include_rsd],
// and the matched Limber term of the same windows. The galaxy fields get
// the same Kaiser term with their unbiased n(z) when include_rsd is 1.
// The correction exact - matched is added by the caller to the nonlinear
// Limber spectra below lmax (the FKEM split, Fang et al.
// arXiv:1911.11947), as the likelihood does for w_gg, w_cc and w_cg
// (cosmo2D.c C_cl_tomo; cosmo2D_cluster.c, section NON-LIMBER).
//
// Steps:
//   1. nbar_c = sum_nodes f_K^2 n_c dchi on the quadrature snapshot, the
//      normalization of survey_cluster.selected_windows;
//   2. the joint quadrature and ln(chi) snapshots (insert_clusters);
//   3. with include_rsd, the unbiased densities on the ln(chi) nodes:
//      n_l H/H0 (W_gal) for lens l, q_c for cluster c;
//   4. nonlimber_pairs_cov, then exact - matched in the rows that contain
//      a lens or cluster, zero in the source-source rows.
// Output rows use the i-major triangular order of the joint fields.
// -----------------------------------------------------------------------
void cluster_nonlimber_cov(
    const struct radial_cov* radial, // galaxy/source quadrature snapshot
    const double amin,               // far boundary of radial
    const int nwindow,               // efficiency samples used for radial
    const int include_rsd,           // Kaiser term switch
    const int lmax,                  // last corrected multipole
    const int nchi,                  // log-distance samples
    const double chi_min,            // near distance in c/H0
    double* const* correction        // joint triangular output rows
  )
{
  // The model checks of nonlimber_spectra_cov, plus the cluster setup:
  // initialized bins, no cluster magnification (the forecast's cluster
  // fields carry none) and a snapshot without the alignment window.
  const int intervals = nchi-1;
  if (lmax < 2
      || intervals < 64
      || (intervals & (intervals-1)) != 0
      || !isfinite(chi_min)
      || chi_min <= 0.0
      || (include_rsd != 0
          && include_rsd != 1)
      || cosmology.Omega_nu != 0.0
      || fabs(cosmology.Omega_m+cosmology.Omega_v-1.0) > 1.e-10) {
    log_fatal("cluster_nonlimber_cov needs flat massless cosmology, "
              "lmax >= 2, nchi=2^n+1 >= 65, chi_min > 0 and RSD 0/1");
    exit(1);
  }
  if (cluster.zdist_nbin < 1
      || cluster.richness_nbin < 1
      || cluster.magnification != 0.0) {
    log_fatal("cluster_nonlimber_cov: initialize the cluster bins, "
              "with cluster magnification 0");
    exit(1);
  }
  for (int source=0; source<radial->nsource; source++) {
    for (int node=0; node<radial->nnode; node++) {
      if (radial->window[2][radial->nlens+source][node] != 0.0) {
        log_fatal("cluster_nonlimber_cov: build the snapshot without IA");
        exit(1);
      }
    }
  }

  // --- 1. CATALOG NORMALIZATIONS ON THE QUADRATURE NODES ---

  // Warm the lazy cluster tables on this thread (ncl/bcl through
  // cluster_warmup, phi through its first read).
  cluster_warmup();
  (void) phi_cluster(1.0/radial->geometry[0][0]-1.0, 0);

  const int ncount = cluster.zdist_nbin*cluster.richness_nbin;
  double* number_per_sr = (double*) malloc1d(ncount);
  for (int category=0; category<ncount; category++) {
    double sum = 0.0;
    for (int node=0; node<radial->nnode; node++) {
      const double fk = radial->geometry[2][node];
      sum += selected_density(radial->geometry[0][node], category)
             *fk*fk*radial->geometry[3][node];
    }
    if (!(sum > 0.0)) {
      log_fatal("cluster_nonlimber_cov: cluster category %d is empty", category);
      exit(1);
    }
    number_per_sr[category] = sum;
  }

  // --- 2. JOINT SNAPSHOTS: LENSES, CLUSTERS, SOURCES ---

  struct radial_cov* joint = insert_clusters(radial, ncount, number_per_sr);
  struct radial_cov* logarithmic_base = radial_logchi_cov(
      amin, chi_min, nchi, nwindow, 0);
  struct radial_cov* logarithmic = insert_clusters(logarithmic_base, ncount,
      number_per_sr);
  free_radial_cov(logarithmic_base);

  // --- 3. UNBIASED DENSITIES FOR THE KAISER TERM ---

  double** rsd = NULL;
  if (include_rsd) {
    rsd = (double**) malloc2d(joint->nlens, nchi);
    for (int node=0; node<nchi; node++) {
      const double a = logarithmic->geometry[0][node];
      const double hubble = hoverh0v2(a, chi_all(a).dchida);
      const double fk = logarithmic->geometry[2][node];
      for (int field=0; field<joint->nlens; field++) {
        if (field < radial->nlens) {
          rsd[field][node] = W_gal(a, field, hubble);
        } else {
          const int category = field-radial->nlens;
          rsd[field][node] = fk*fk*selected_density(a, category)
                             /number_per_sr[category];
        }
      }
    }
  }

  // --- 4. EXACT AND MATCHED SPECTRA, AND THEIR DIFFERENCE ---

  const int nfield = joint->nlens+joint->nsource;
  const int npair = nfield*(nfield+1)/2;
  double*** terms = (double***) malloc3d(2, npair, lmax-1);
  nonlimber_pairs_cov(joint, logarithmic, (const double* const*) rsd, amin,
      lmax, nchi, chi_min, terms[0], terms[1]);

  // A pair holds a lens or cluster exactly when its first field does
  // (lens-type fields come first in the triangular order).
  int pair = 0;
  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      for (int index=0; index<lmax-1; index++) {
        correction[pair][index] = first < joint->nlens
            ? terms[0][pair][index]-terms[1][pair][index]
            : 0.0;
      }
      pair++;
    }
  }

  free(terms);
  free(rsd);
  free_radial_cov(logarithmic);
  free_radial_cov(joint);
  free(number_per_sr);
}
