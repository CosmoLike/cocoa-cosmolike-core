// ============================================================================
// redshift_spline_cluster.c
//
// Redshift distributions of the cluster sample and the tomographic pair
// maps of the cluster two-point blocks, for the 4x2pt + N analysis
// (DES Y6 methods paper, arXiv 2503.13631; Y1 choices: arXiv 2008.10757).
// The cluster analog of redshift_spline.c.
//
// Data flow:
//
//   Python selection kernels <phi_i|z>          (cluster.zdist_table)
//     -> uniform fine z grid, one per bin       (phi_cluster)
//     -> n(z) = dV/dz <phi_i|z> [n_A(z)] / norm on the same grid
//                                               (nz_cluster)
//     -> lensing efficiency g(a) of that n(z), for cluster magnification,
//        by the factored cumulative trapezoid   (g_cluster)
//     -> radial weights W_cluster, W_mag_cluster (radial_weights_cluster.c)
//     -> Limber integrals                       (cosmo2D_cluster.c)
//
// Units (the library's): z dimensionless, comoving distance chi in c/H0,
// dV/dz per steradian in (c/H0)^3, n(z) per unit z (integrates to 1).
//
// Caching idiom (the one of redshift_spline.c and halo.c): each table
// stores the uint64 keys it was built with (cosmology.random,
// Ntable.random, cluster.random_*) and refills when fdiff2 sees a
// different one. A refill is not thread-safe: the first call after any
// key change must run outside OpenMP regions. cluster_warmup (halo_cluster.c)
// does that for the kernel tables: one call of phi_cluster, nz_cluster
// and g_cluster fills every bin and every richness row, because each
// builder below fills all of them at once. The pair maps are warmed by
// the interface (any accessor call, see the pair-map banner).
//
// Nothing in this file writes the nominal bin edges cluster.zbin or the
// kernel support cluster.zdist_z (both [RANGE_MIN|RANGE_MAX][bin]): the
// tables copy them.
// ============================================================================

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.c/src/log.h"

#include "basics.h"
#include "cosmo3D.h"
#include "halo_cluster.h"
#include "redshift_spline_cluster.h"
#include "structs.h"
#include "structs_cluster.h"



// ============================================================================
// [SECTION] INTEGRATION LIMITS AND NOMINAL BIN CENTERS
// ============================================================================


// ---------------------------------------------------------------------------
// Lower scale-factor bound of line-of-sight integrals over cluster bin ni:
// the far edge of the support of <phi_ni|z>,
//
//   a_min = 1/(1 + zdist_zmax[ni]).
//
// The support (not the nominal z_lambda edges zbin_min/max) is the right
// range: a Gaussian photo-z kernel leaks beyond the nominal edges.
//
// Parameters:
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//
// Returns:
//   smallest scale factor where <phi_ni|z> can be nonzero
// ---------------------------------------------------------------------------
double amin_cluster(const int ni)
{
  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d (zdist_nbin = %d)", ni,
      cluster.zdist_nbin);
    exit(1);
  }
  return 1.0/(1.0 + cluster.zdist_z[RANGE_MAX][ni]);
}


// ---------------------------------------------------------------------------
// Upper scale-factor bound of line-of-sight integrals over cluster bin ni:
// the near edge of the support of <phi_ni|z>,
//
//   a_max = 1/(1 + zdist_zmin[ni]).
//
// The cluster density kernel vanishes in front of it; the magnification
// kernel does not (g_cluster is nonzero up to a = 1), so the Limber
// integrals of magnification terms extend beyond a_max.
//
// Parameters:
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//
// Returns:
//   largest scale factor where <phi_ni|z> can be nonzero
// ---------------------------------------------------------------------------
double amax_cluster(const int ni)
{
  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d (zdist_nbin = %d)", ni,
      cluster.zdist_nbin);
    exit(1);
  }
  return 1.0/(1.0 + cluster.zdist_z[RANGE_MIN][ni]);
}


// ---------------------------------------------------------------------------
// Nominal midpoint of the z_lambda bin ni,
//
//   zbar = (zbin_min[ni] + zbin_max[ni])/2,
//
// the zbar of the selection bias (eq 23 of 2503.13631) and of the physical
// scale cuts. Read from the nominal edges, never from the kernel support.
//
// Parameters:
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//
// Returns:
//   nominal bin midpoint (aborts if the nominal edges were never set)
// ---------------------------------------------------------------------------
double zmid_cluster(const int ni)
{
  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d (zdist_nbin = %d)", ni,
      cluster.zdist_nbin);
    exit(1);
  }
  if (!(cluster.zbin[RANGE_MAX][ni] > cluster.zbin[RANGE_MIN][ni])) {
    log_fatal("nominal z_lambda edges of cluster bin %d not set "
      "(zbin_min = %e, zbin_max = %e)", ni, cluster.zbin[RANGE_MIN][ni],
      cluster.zbin[RANGE_MAX][ni]);
    exit(1);
  }
  return 0.5*(cluster.zbin[RANGE_MIN][ni] + cluster.zbin[RANGE_MAX][ni]);
}



// ============================================================================
// [SECTION] SELECTION KERNEL <phi_i|z> ON A UNIFORM FINE z GRID
// ============================================================================
//
// <phi_i|z> is the probability that a cluster at true redshift z has its
// photometric redshift z_lambda in bin i (Y1 eq 6). Python passes it as a
// table (cluster.zdist_table, layout in structs_cluster.h): a top-hat, the
// erf edges of a Gaussian photo-z kernel, or a kernel measured from
// randoms. The z column holds sample points: table[i][j] is <phi_i|z_j>.
//
// Model: <phi_i|z> is the piecewise-linear interpolant of that table on
// the support [zdist_zmin[i], zdist_zmax[i]] and exactly 0 outside it.
// (The Python reference reproduces it with numpy.interp plus the cut.)
//
// Support convention (set by the interface): the zero nodes that bracket
// the nonzero values of the column, so the cut removes nothing of the
// interpolant. It matters for kernels whose edges are ramps: a top-hat
// tabulated with 0.5 on its edge nodes has the area of a sharp edge at
// that node only if both half-cells of the ramp are kept; a support
// starting at the first nonzero node (the lens n(z) loader's rule) drops
// dz/4 at each edge. A top-hat tabulated with 1 on its edge nodes and a
// support starting there instead gets a sharp edge at the node.
//
// The nz_lens_photoz design: the table is resampled once per input change
// onto a uniform fine grid per bin whose end nodes are the support edges;
// the hot path is one multiply for the index (no search) and a linear
// read.


// ---------------------------------------------------------------------------
// Reference step of the fine grid: its step is at most
// DZ_FINE_REFERENCE/Ntable.nz_fine_sampling_factor. The narrowest feature
// of a cluster kernel is the erf edge of a Gaussian photo-z,
// sigma_z = 0.006 (1 + z) for redMaPPer; at the baseline sampling factor
// (5) this bound is the 5e-4 that puts more than ten nodes per sigma_z,
// and every accuracy boost refines it.
// ---------------------------------------------------------------------------
static const double DZ_FINE_REFERENCE = 2.5e-3;

// ---------------------------------------------------------------------------
// Round-off guard, in units of one step, for the ceil() that counts
// steps: a support spanning a whole number of input steps keeps its fine
// nodes on the input nodes instead of gaining one extra, shifted step.
// ---------------------------------------------------------------------------
static const double ALIGNMENT_SLACK = 1.0e-6;


// ---------------------------------------------------------------------------
// The fine-grid tables of <phi_i|z>, built by selection_kernel_table (its
// header); zero at program start, so the first call builds. The grid
// geometry is shared with the n(z) tables of the next section.
// ---------------------------------------------------------------------------
static struct {
  uint64_t cache[2];              // [0] Ntable.random, [1] cluster.random_zdist
  int nbin;                       // cluster redshift bins of the allocation
  int n[MAX_SIZE_ARRAYS];         // fine z nodes of bin ni, both edges included
  double zmin[MAX_SIZE_ARRAYS];   // support of bin ni = first fine node
  double zmax[MAX_SIZE_ARRAYS];   // support of bin ni = last fine node
  double dz[MAX_SIZE_ARRAYS];     // fine z step of bin ni
  double inv_dz[MAX_SIZE_ARRAYS]; // 1/dz: the direct index is one multiply
  double** phi;                   // [nbin][max n] <phi_i|z_k> at the nodes
} phi_;


// ---------------------------------------------------------------------------
// Redshift of fine node k of bin ni. The last node is the support edge
// itself (not zmin + (n - 1) dz, which can differ from it by round-off).
// ---------------------------------------------------------------------------
static inline double fine_z_node(const int ni, const int k)
{
  if (k == phi_.n[ni] - 1) {
    return phi_.zmax[ni];
  }
  return phi_.zmin[ni] + k*phi_.dz[ni];
}


// ---------------------------------------------------------------------------
// Linear read, at z inside the support of bin ni, of a table sampled on
// the fine z grid of bin ni (phi_.phi[ni] or an n(z) row).
//
//   r = (z - zmin)/dz   fractional node index (one multiply, no search)
//   k = floor(r)        left node of the bracketing cell (clamped so the
//                       support edge z = zmax reads the last cell)
//   f(z) = f_k + (r - k) (f_{k+1} - f_k)
// ---------------------------------------------------------------------------
static inline double fine_z_read(const double* table, const int ni,
  const double z)
{
  const double r = (z - phi_.zmin[ni])*phi_.inv_dz[ni];

  int k = (int) r;
  if (k > phi_.n[ni] - 2) {
    k = phi_.n[ni] - 2;
  }
  if (k < 0) {
    k = 0;
  }

  const double t = r - (double) k;
  return table[k] + t*(table[k+1] - table[k]);
}


// ---------------------------------------------------------------------------
// Builds phi_: <phi_i|z> of every cluster bin on its own uniform fine z
// grid spanning exactly the support [zdist_zmin[i], zdist_zmax[i]].
//
// Fine step: dz_input/r, with dz_input the smallest step of the input z
// column and r the smallest integer that brings the step below
// DZ_FINE_REFERENCE/Ntable.nz_fine_sampling_factor. Because r is an
// integer, when the support edges sit on input nodes of a uniform table
// (they do under the support convention of the section banner) every
// input node is a fine node, and the linear read of the fine table
// reproduces the input interpolant exactly.
//
// Why a linear read and not the house cubic spline (the nz_lens_photoz
// read): a top-hat kernel, or any kernel with a sharp edge, has a jump.
// An interpolating cubic spline through a unit step overshoots by about
// 11% on both sides and rings with alternating sign, decaying only by
// 2 - sqrt(3) per node: <phi|z> would be negative just outside the bin
// and above 1 just inside, i.e. a negative cluster density. The linear
// read is monotone between nodes, so it never leaves the range of the
// input values; for a smooth erf edge its error is phi'' dz^2/8, small at
// the fine step above, and it integrates to second order (the kernel
// enters the observables only through integrals).
//
// Aborts: no table loaded, bin counts out of range, z column not strictly
// increasing, support not inside the table or not above z = 0 (g_cluster
// divides by chi(z) on the support), negative or non-finite kernel value.
//
// Cache invalidation: Ntable.random (nz_fine_sampling_factor) or
// cluster.random_zdist (the kernels, their z column and support).
// ---------------------------------------------------------------------------
static void selection_kernel_table(void)
{
  if (NULL == phi_.phi ||
      fdiff2(phi_.cache[0], Ntable.random) ||
      fdiff2(phi_.cache[1], cluster.random_zdist))
  {
    // --- 1. INPUT CHECKS ---
    const int nbin     = cluster.zdist_nbin;
    const int nz_input = cluster.zdist_nz;

    if (NULL == cluster.zdist_table) {
      log_fatal("cluster selection kernels <phi_i|z> not loaded");
      exit(1);
    }
    if (nbin < 1 || nbin > MAX_SIZE_ARRAYS) {
      log_fatal("invalid number of cluster redshift bins = %d", nbin);
      exit(1);
    }
    if (nz_input < 2) {
      log_fatal("cluster kernel table needs >= 2 z rows (got %d)", nz_input);
      exit(1);
    }
    if (Ntable.nz_fine_sampling_factor < 1) {
      log_fatal("invalid Ntable.nz_fine_sampling_factor = %d",
        Ntable.nz_fine_sampling_factor);
      exit(1);
    }

    const double* z_input = cluster.zdist_table[nbin]; // z column

    double dz_input = z_input[1] - z_input[0];
    for (int j=0; j<nz_input-1; j++) {
      const double step = z_input[j+1] - z_input[j];
      if (!(step > 0.0)) {
        log_fatal("cluster kernel z column not strictly increasing at "
          "row %d", j);
        exit(1);
      }
      dz_input = fmin(dz_input, step);
    }

    for (int ni=0; ni<nbin; ni++) {
      const double zmin = cluster.zdist_z[RANGE_MIN][ni];
      const double zmax = cluster.zdist_z[RANGE_MAX][ni];
      if (!(zmin > 0.0) || !(zmax > zmin)) {
        log_fatal("invalid support of cluster bin %d: [%e, %e] (need "
          "0 < zmin < zmax)", ni, zmin, zmax);
        exit(1);
      }
      if (zmin < z_input[0] || zmax > z_input[nz_input-1]) {
        log_fatal("support of cluster bin %d, [%e, %e], leaves the kernel "
          "table [%e, %e]", ni, zmin, zmax, z_input[0], z_input[nz_input-1]);
        exit(1);
      }
    }

    // --- 2. FINE STEP: AN INTEGER SUBDIVISION OF THE INPUT STEP ---
    const double dz_bound =
      DZ_FINE_REFERENCE/((double) Ntable.nz_fine_sampling_factor);

    int refinement = (int) ceil(dz_input/dz_bound - ALIGNMENT_SLACK);
    if (refinement < 1) {
      refinement = 1;
    }
    const double dz_target = dz_input/((double) refinement);

    // --- 3. ONE UNIFORM GRID PER BIN, ENDS ON THE SUPPORT EDGES ---
    int n_max = 0;
    for (int ni=0; ni<nbin; ni++) {
      const double width = cluster.zdist_z[RANGE_MAX][ni] - cluster.zdist_z[RANGE_MIN][ni];

      int n_steps = (int) ceil(width/dz_target - ALIGNMENT_SLACK);
      if (n_steps < 1) {
        n_steps = 1;
      }

      phi_.n[ni]      = n_steps + 1;
      phi_.zmin[ni]   = cluster.zdist_z[RANGE_MIN][ni];
      phi_.zmax[ni]   = cluster.zdist_z[RANGE_MAX][ni];
      phi_.dz[ni]     = width/((double) n_steps);
      phi_.inv_dz[ni] = 1.0/phi_.dz[ni];

      if (phi_.n[ni] > n_max) {
        n_max = phi_.n[ni];
      }
    }

    // --- 4. ALLOCATION ---
    if (phi_.phi != NULL) {
      free(phi_.phi);
    }
    phi_.nbin = nbin;
    phi_.phi  = (double**) malloc2d(nbin, n_max);

    // --- 5. LINEAR RESAMPLING OF THE INPUT ONTO THE FINE NODES ---
    // Fine and input nodes are both sorted, so the input cell holding
    // node k is found by walking one index forward (no search).
    for (int ni=0; ni<nbin; ni++) {
      const double* phi_input = cluster.zdist_table[ni];

      int j = 0; // input cell [z_j, z_{j+1}] holding the current node
      for (int k=0; k<phi_.n[ni]; k++) {
        const double z = fine_z_node(ni, k);

        while (j < nz_input - 2 && z_input[j+1] <= z) {
          j++;
        }

        const double t = (z - z_input[j])/(z_input[j+1] - z_input[j]);
        const double phi = phi_input[j] + t*(phi_input[j+1] - phi_input[j]);

        if (!isfinite(phi) || phi < 0.0) {
          log_fatal("cluster kernel <phi_%d|z = %e> = %e: must be finite "
            "and >= 0", ni, z, phi);
          exit(1);
        }
        phi_.phi[ni][k] = phi;
      }
    }

    // --- 6. CACHE TAGS: THE INPUTS THE TABLE NOW HOLDS ---
    phi_.cache[0] = Ntable.random;
    phi_.cache[1] = cluster.random_zdist;
  }
}


// ---------------------------------------------------------------------------
// Selection kernel <phi_ni|z> at true redshift z: the piecewise-linear
// interpolant of the Python table inside the support [zdist_zmin[ni],
// zdist_zmax[ni]] (edges included), 0 outside. Read linearly from the
// fine grid of selection_kernel_table (its header explains why linear).
//
// Parameters:
//   z  - true redshift
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//
// Returns:
//   <phi_ni|z>, dimensionless probability
// ---------------------------------------------------------------------------
double phi_cluster(const double z, const int ni)
{
  selection_kernel_table();

  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d (zdist_nbin = %d)", ni,
      cluster.zdist_nbin);
    exit(1);
  }

  if (z < phi_.zmin[ni] || z > phi_.zmax[ni]) {
    return 0.0;
  }
  return fine_z_read(phi_.phi[ni], ni, z);
}



// ============================================================================
// [SECTION] CLUSTER n(z) AND ITS LENSING EFFICIENCY g(a)
// ============================================================================
//
// Radial distribution of clusters of redshift bin i (and richness bin A),
// per unit z:
//
//   CLUSTER_KERNEL_VOLUME     n_i(z)  = dV/dz <phi_i|z>        / norm_i
//                                       (Y1 eq 15; what DES ran)
//   CLUSTER_KERNEL_ABUNDANCE  n_iA(z) = dV/dz <phi_i|z> n_A(z) / norm_iA
//                                       (the integrand of the counts, eq 16)
//
// with dV/dz = chi^2/(H/H0) per steradian and n_A(z) = ncl_richness
// (halo_cluster.c). Both are tabulated on the fine z grid of phi_, so the
// Limber integrals read them with one multiply and one linear read per
// node, never re-evaluating dV/dz. The volume kernel does not depend on
// the richness bin: its table has one row, shared by every nl.


// ---------------------------------------------------------------------------
// The n(z) and g(a) tables, built and refilled by cluster_kernel_tables
// (its header); zero at program start, so the first call builds.
// ---------------------------------------------------------------------------
static struct {
  uint64_t cache[5];              // [0] Ntable.random, [1] cosmology.random,
                                  //   [2] cluster.random_zdist,
                                  //   [3] cluster.random_model,
                                  //   [4] cluster.random_mor (abundance
                                  //       kernel only; 0 otherwise)
  int nbin;                       // cluster redshift bins of the allocation
  int nrows;                      // kernel rows per bin: 1 (volume) or
                                  //   cluster.richness_nbin (abundance)
  int n_a;                        // nodes of the a grid of g (every bin)
  double amin[MAX_SIZE_ARRAYS];   // a grid of bin ni: first node a(zmax),
  double da[MAX_SIZE_ARRAYS];     //   step, and 1/step; the last node is
  double inv_da[MAX_SIZE_ARRAYS]; //   a = 1 (today)
  double*** nz;                   // [nbin][nrows][max n] normalized n(z)
                                  //   at the fine z nodes of phi_
  double*** g;                    // [nbin][nrows][n_a] lensing efficiency
                                  //   at the a nodes
} kernel_;


// ---------------------------------------------------------------------------
// Number of kernel rows per cluster redshift bin under the current model:
// 1 for the volume kernel (no richness dependence), one per richness bin
// for the abundance kernel. Aborts on an unknown cluster.kernel_mode.
// ---------------------------------------------------------------------------
static int kernel_nrows(void)
{
  if (CLUSTER_KERNEL_VOLUME == cluster.kernel_mode) {
    return 1;
  }
  if (CLUSTER_KERNEL_ABUNDANCE == cluster.kernel_mode) {
    if (cluster.richness_nbin < 1 ||
        cluster.richness_nbin > MAX_SIZE_ARRAYS) {
      log_fatal("abundance kernel needs richness bins (richness_nbin = %d)",
        cluster.richness_nbin);
      exit(1);
    }
    return cluster.richness_nbin;
  }
  log_fatal("unknown cluster.kernel_mode = %d", cluster.kernel_mode);
  exit(1);
}


// ---------------------------------------------------------------------------
// Table row of richness bin nl: nl itself for the abundance kernel; row 0
// for the volume kernel, where every richness bin shares one kernel.
// Aborts on an invalid nl.
// ---------------------------------------------------------------------------
static int kernel_row(const int nl)
{
  if (CLUSTER_KERNEL_ABUNDANCE == cluster.kernel_mode) {
    if (nl < 0 || nl > cluster.richness_nbin - 1) {
      log_fatal("invalid richness bin nl = %d (richness_nbin = %d)", nl,
        cluster.richness_nbin);
      exit(1);
    }
    return nl;
  }

  // volume kernel: nl is not used, only checked when bins exist
  const int richness_bins_set = (cluster.richness_nbin > 0);
  if (nl < 0 || (richness_bins_set && nl > cluster.richness_nbin - 1)) {
    log_fatal("invalid richness bin nl = %d (richness_nbin = %d)", nl,
      cluster.richness_nbin);
    exit(1);
  }
  return 0;
}


// ---------------------------------------------------------------------------
// Fills kernel_: the normalized cluster n(z) on the fine z grid of every
// bin (nz_cluster) and its lensing efficiency on a fine a grid over
// [a(zmax), 1] (g_cluster), for every kernel row, in one pass.
//
//   dV/dz = chi^2 dchi/dz = chi^2/(H/H0)          per steradian, (c/H0)^3
//   u(z)  = dV/dz <phi_i|z>          [x n_A(z)]   unnormalized weight
//   norm  = int_{zmin}^{zmax} dz u(z)
//   n(z)  = u(z)/norm
//
//   g(a)  = int_{z(a)}^{zmax} dz' n(z') [1 - chi(a)/chi(z')]
//         = P(z(a)) - chi(a) Q(z(a))
//   P(z)  = int_z^{zmax} dz' n(z'),   Q(z) = int_z^{zmax} dz' n(z')/chi(z')
//
// This is g_lens of redshift_spline.c (the same integral written in z
// instead of a: n(z) dz = n(z(a')) da'/a'^2) with the factorization
// g = P - chi Q, valid for flat space (the library is flat:
// set_cosmological_parameters sets Omega_v = 1 - Omega_m, so f_K = chi;
// also why dV/dz uses chi^2).
//
// Numerics:
//   - dV/dz, n(z): exact at the fine z nodes of phi_; read linearly.
//   - norm: trapezoid over the fine z nodes, the exact integral of the
//     linear read of u, so the tabulated n(z) integrates to 1 up to
//     round-off. The fine nodes contain both support edges and, for the
//     interface's uniform tables, every input node: every kink of the
//     piecewise-linear <phi_i|z>. The trapezoid error is then only the
//     curvature of dV/dz inside a fine cell. A Gauss-Legendre rule over
//     the support would straddle those kinks and converge only
//     algebraically, and not monotonically in its size, for a sampled
//     erf kernel.
//   - P, Q: cumulative trapezoid over the same fine z nodes, from the far
//     edge zmax toward the observer; a top-hat edge sits on a node and is
//     integrated exactly (a grid uniform in a would cut through it).
//   - g at an a node inside the support: the running sums at the next
//     fine node plus the partial cell up to z(a), with the same trapezoid
//     and the linear read of n, so g is the exact continuation of the
//     fine sums; in front of the support P and Q are complete and
//     g = P(zmin) - chi(a) Q(zmin), with P(zmin) = 1.
//   - a grid: nz_fine_sampling_factor (N_a - 1) + 1 nodes per bin, from
//     the far edge a(zmax) (g = 0: no clusters behind it) to a = 1, read
//     linearly. g is continuous with a continuous slope
//     (dg/da = -Q dchi/da), so the linear read converges as da^2.
//   - Abundance kernel with norm <= 0 (no cluster predicted in that
//     richness bin anywhere in the bin, e.g. at an extreme MOR point): the
//     row is set to 0 with a warning, so a sampler rejects the point
//     instead of the process aborting. For the volume kernel it aborts:
//     the input kernel is empty.
//
// Serial by design: a refill evaluates a few thousand nodes, too little
// work to amortize an OpenMP region, and running serially lets its first
// call build the lazy tables it reads (ncl_richness in halo_cluster.c).
// Each table value is one serial sum: deterministic.
//
// Cache invalidation:
//   rebuild (sizes, allocations, a grids): Ntable.random,
//     cluster.random_zdist or cluster.random_model
//   refill: those three, cosmology.random, and cluster.random_mor for the
//     abundance kernel
// ---------------------------------------------------------------------------
static void cluster_kernel_tables(void)
{
  // n_A(z) depends on the MOR only through the abundance kernel: the
  // volume kernel ignores MOR changes (key slot held at 0)
  uint64_t mor_key = 0;
  if (CLUSTER_KERNEL_ABUNDANCE == cluster.kernel_mode) {
    mor_key = cluster.random_mor;
  }

  // --- 1. REBUILD: SIZES, ALLOCATIONS, a GRIDS ---
  int rebuilt = 0; // a rebuild leaves the new tables empty: refill below
  if (NULL == kernel_.nz ||
      fdiff2(kernel_.cache[0], Ntable.random) ||
      fdiff2(kernel_.cache[2], cluster.random_zdist) ||
      fdiff2(kernel_.cache[3], cluster.random_model))
  {
    selection_kernel_table(); // the fine z grids the tables live on

    if (Ntable.N_a < 2) {
      log_fatal("invalid Ntable.N_a = %d", Ntable.N_a);
      exit(1);
    }

    if (kernel_.nz != NULL) {
      free(kernel_.nz);
      free(kernel_.g);
    }

    kernel_.nbin  = phi_.nbin;
    kernel_.nrows = kernel_nrows();
    kernel_.n_a   = Ntable.nz_fine_sampling_factor*(Ntable.N_a - 1) + 1;

    int n_max = 0;
    for (int ni=0; ni<kernel_.nbin; ni++) {
      if (phi_.n[ni] > n_max) {
        n_max = phi_.n[ni];
      }
    }

    kernel_.nz = (double***) malloc3d(kernel_.nbin, kernel_.nrows, n_max);
    kernel_.g  = (double***) malloc3d(kernel_.nbin, kernel_.nrows,
                                      kernel_.n_a);

    // a grid of bin ni: from the far edge of its support to a = 1
    for (int ni=0; ni<kernel_.nbin; ni++) {
      const double n_steps = (double) kernel_.n_a - 1.0;

      kernel_.amin[ni]   = 1.0/(1.0 + phi_.zmax[ni]);
      kernel_.da[ni]     = (1.0 - kernel_.amin[ni])/n_steps;
      kernel_.inv_da[ni] = 1.0/kernel_.da[ni];
    }

    rebuilt = 1;
  }

  // --- REFILL: COSMOLOGY, KERNELS, MODEL OR MOR CHANGED ---
  if (rebuilt ||
      fdiff2(kernel_.cache[0], Ntable.random) ||
      fdiff2(kernel_.cache[1], cosmology.random) ||
      fdiff2(kernel_.cache[2], cluster.random_zdist) ||
      fdiff2(kernel_.cache[3], cluster.random_model) ||
      fdiff2(kernel_.cache[4], mor_key))
  {
    /* PHYSICAL DERIVATION & LOGIC FLOW
       2. geometry at the fine z nodes: chi_k, dV/dz_k = chi_k^2/(H/H0)_k
       3. unnormalized weights u_k = dV/dz_k <phi|z_k> [n_A(a_k)]
       4. norm = sum_k (z_{k+1} - z_k)(u_k + u_{k+1})/2; n_k = u_k/norm
       5. running sums from the far edge: P_k = int_{z_k}^{zmax} n dz,
          Q_k = int_{z_k}^{zmax} n/chi dz (same trapezoid)
       6. g(a_j) = P(z_j) - chi(a_j) Q(z_j) on the a grid, a_j in
          [a(zmax), 1] */

    const int nbin      = kernel_.nbin;
    const int nrows     = kernel_.nrows;
    const int abundance = (CLUSTER_KERNEL_ABUNDANCE == cluster.kernel_mode);

    int n_max = 0;
    for (int ni=0; ni<nbin; ni++) {
      if (phi_.n[ni] > n_max) {
        n_max = phi_.n[ni];
      }
    }

    // per-refill work arrays (freed at the end of the refill)
    double* chi_node     = (double*) malloc1d(n_max);       // chi(z_k)
    double* volume_node  = (double*) malloc1d(n_max);       // dV/dz <phi>
    double* cumulative_P = (double*) malloc1d(n_max);
    double* cumulative_Q = (double*) malloc1d(n_max);
    double* chi_a        = (double*) malloc1d(kernel_.n_a); // chi(a_j)

    // the abundance kernel reads ncl_richness: build its tables here,
    // serially, before any loop reads them
    if (abundance) {
      (void) ncl_richness(1.0/(1.0 + 0.5*(phi_.zmin[0] + phi_.zmax[0])), 0);
    }

    for (int ni=0; ni<nbin; ni++) {
      const int n_fine = phi_.n[ni];

      // --- 2. GEOMETRY AT THE FINE z NODES ---
      for (int k=0; k<n_fine; k++) {
        const double z = fine_z_node(ni, k);
        const double a = 1.0/(1.0 + z);

        const struct chis chidchi = chi_all(a);
        const double hoverh0 = hoverh0v2(a, chidchi.dchida);
        const double dV_dz   = chidchi.chi*chidchi.chi/hoverh0;

        chi_node[k]    = chidchi.chi;
        volume_node[k] = dV_dz*phi_.phi[ni][k];
      }

      // chi at the a nodes of this bin (shared by every row); the last
      // node is exactly a = 1, where chi = 0
      for (int j=0; j<kernel_.n_a; j++) {
        double a = kernel_.amin[ni] + j*kernel_.da[ni];
        if (j == kernel_.n_a - 1) {
          a = 1.0;
        }
        chi_a[j] = chi_all(a).chi;
      }

      for (int row=0; row<nrows; row++) {
        double* n_z = kernel_.nz[ni][row];

        // --- 3. UNNORMALIZED WEIGHTS u_k ---
        for (int k=0; k<n_fine; k++) {
          double weight = volume_node[k];
          if (abundance) {
            const double a = 1.0/(1.0 + fine_z_node(ni, k));
            weight *= ncl_richness(a, row);
          }
          n_z[k] = weight;
        }

        // --- 4. NORMALIZATION: TRAPEZOID OVER THE FINE NODES ---
        double norm = 0.0;
        for (int k=0; k<n_fine-1; k++) {
          const double dz = fine_z_node(ni, k+1) - fine_z_node(ni, k);
          norm += 0.5*dz*(n_z[k] + n_z[k+1]);
        }

        double inv_norm = 0.0;
        if (norm > 0.0) {
          inv_norm = 1.0/norm;
        }
        else if (abundance) {
          log_warn("cluster bin %d, richness bin %d: no clusters predicted "
            "(norm = %e); kernel set to 0", ni, row, norm);
        }
        else {
          log_fatal("cluster bin %d: dV/dz <phi|z> integrates to %e "
            "(empty selection kernel)", ni, norm);
          exit(1);
        }

        for (int k=0; k<n_fine; k++) {
          n_z[k] *= inv_norm;
        }

        // --- 5. RUNNING SUMS P, Q FROM THE FAR EDGE ---
        // P_k = int_{z_k}^{zmax} n dz,  Q_k = int_{z_k}^{zmax} n/chi dz
        cumulative_P[n_fine - 1] = 0.0;
        cumulative_Q[n_fine - 1] = 0.0;
        for (int k=n_fine-2; k>=0; k--) {
          const double dz = fine_z_node(ni, k+1) - fine_z_node(ni, k);
          cumulative_P[k] = cumulative_P[k+1]
                          + 0.5*dz*(n_z[k] + n_z[k+1]);
          cumulative_Q[k] = cumulative_Q[k+1]
                          + 0.5*dz*(n_z[k]/chi_node[k]
                                    + n_z[k+1]/chi_node[k+1]);
        }

        // --- 6. g = P - chi Q ON THE a GRID ---
        for (int j=0; j<kernel_.n_a; j++) {
          double a = kernel_.amin[ni] + j*kernel_.da[ni];
          if (j == kernel_.n_a - 1) {
            a = 1.0;
          }
          const double z = 1.0/a - 1.0;

          double g = 0.0;
          if (z >= phi_.zmax[ni]) {
            // at (or behind) the far edge: no clusters behind
            g = 0.0;
          }
          else if (z <= phi_.zmin[ni]) {
            // in front of the support: complete integrals
            g = cumulative_P[0] - chi_a[j]*cumulative_Q[0];
          }
          else {
            // inside the support: fine cell k holds z; add the partial
            // cell [z, z_{k+1}] to the running sums at node k + 1, with
            // the same trapezoid and the linear read n(z)
            const double r = (z - phi_.zmin[ni])*phi_.inv_dz[ni];
            int k = (int) r;
            if (k > n_fine - 2) {
              k = n_fine - 2;
            }
            const double t = r - (double) k;

            const double n_at_z  = n_z[k] + t*(n_z[k+1] - n_z[k]);
            const double partial = fine_z_node(ni, k+1) - z;

            const double P = cumulative_P[k+1]
                           + 0.5*partial*(n_at_z + n_z[k+1]);
            const double Q = cumulative_Q[k+1]
                           + 0.5*partial*(n_at_z/chi_a[j]
                                          + n_z[k+1]/chi_node[k+1]);
            g = P - chi_a[j]*Q;
          }
          kernel_.g[ni][row][j] = g;
        }
      }
    }

    free(chi_node);
    free(volume_node);
    free(cumulative_P);
    free(cumulative_Q);
    free(chi_a);

    // --- 7. CACHE TAGS: THE INPUTS THE TABLES NOW HOLD ---
    kernel_.cache[0] = Ntable.random;
    kernel_.cache[1] = cosmology.random;
    kernel_.cache[2] = cluster.random_zdist;
    kernel_.cache[3] = cluster.random_model;
    kernel_.cache[4] = mor_key;
  }
}


// ---------------------------------------------------------------------------
// Normalized true-redshift distribution of the clusters of redshift bin
// ni (and richness bin nl for the abundance kernel), per unit z:
//
//   CLUSTER_KERNEL_VOLUME:    n(z) = dV/dz <phi_ni|z>       / norm
//   CLUSTER_KERNEL_ABUNDANCE: n(z) = dV/dz <phi_ni|z> n_nl(z) / norm
//
// dV/dz = chi^2/(H/H0) per steradian. Tabulated by cluster_kernel_tables
// (its header) on the fine z grid of the selection kernel and read
// linearly: cheap enough for every Limber node.
//
// Parameters:
//   z  - true redshift
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//   nl - richness bin (0 .. cluster.richness_nbin - 1; unused by the
//        volume kernel)
//
// Returns:
//   n(z) per unit z; 0 outside the support [zdist_zmin[ni], zdist_zmax[ni]]
// ---------------------------------------------------------------------------
double nz_cluster(const double z, const int ni, const int nl)
{
  cluster_kernel_tables();

  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d (zdist_nbin = %d)", ni,
      cluster.zdist_nbin);
    exit(1);
  }
  const int row = kernel_row(nl);

  if (z < phi_.zmin[ni] || z > phi_.zmax[ni]) {
    return 0.0;
  }
  return fine_z_read(kernel_.nz[ni][row], ni, z);
}


// ---------------------------------------------------------------------------
// Lensing efficiency of the clusters of redshift bin ni (richness bin nl
// for the abundance kernel), the g_lens convention:
//
//   g(a) = int_{z(a)}^{zmax} dz' n(z') [1 - chi(a)/chi(z')],
//
// n = nz_cluster. Tabulated by cluster_kernel_tables (its header) on a
// uniform a grid over [a(zmax), 1] and read linearly: nonzero over the
// whole foreground of the bin, up to a = 1 (unlike g_lens, which is 0 in
// its last cell), and 0 behind the far edge of the support.
//
// Parameters:
//   a  - scale factor, 0 < a <= 1
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//   nl - richness bin (unused by the volume kernel)
//
// Returns:
//   g(a), dimensionless; 0 for a <= a(zmax)
// ---------------------------------------------------------------------------
double g_cluster(const double a, const int ni, const int nl)
{
  cluster_kernel_tables();

  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d (zdist_nbin = %d)", ni,
      cluster.zdist_nbin);
    exit(1);
  }
  const int row = kernel_row(nl);

  if (!(a > 0.0) || a > 1.0) {
    log_fatal("a = %e: need 0 < a <= 1", a);
    exit(1);
  }
  if (a <= kernel_.amin[ni]) {
    return 0.0;
  }

  // direct index on the uniform a grid, linear read
  const double r = (a - kernel_.amin[ni])*kernel_.inv_da[ni];
  int j = (int) r;
  if (j > kernel_.n_a - 2) {
    j = kernel_.n_a - 2;
  }
  const double t = r - (double) j;

  const double* g = kernel_.g[ni][row];
  return g[j] + t*(g[j+1] - g[j]);
}



// ============================================================================
// [SECTION] TOMOGRAPHIC PAIR MAPS
// ============================================================================
//
// Every cluster two-point block runs over a flat pair index n; these maps
// convert between n and the bins of the pair:
//
//   cluster lensing (cs):   every (cluster bin ni, source bin ns) pair,
//                           cluster-major: n = ni*shear_nbin + ns
//                           (lighthouse order; unwanted pairs are masked,
//                           not removed)
//   cluster-galaxy (cg):    cluster bin ni with lens bin
//                           cluster.cg_lens_bin[ni], for every ni whose
//                           entry is a valid lens bin (>= 0 and
//                           < redshift.clustering_nbin), in ni order
//   cluster-cluster (cc):   auto z bin; richness pairs nl1 <= nl2,
//                           row-major (0,0), (0,1), ..., (1,1), ...;
//                           N_cc_richness is symmetric in (nl1, nl2)
//
// The maps own the pair counts: their builder writes
//   cluster.cs_npowerspectra = zdist_nbin*shear_nbin
//   cluster.cg_npowerspectra = number of valid cg_lens_bin entries
//   cluster.cc_npowerspectra = zdist_nbin  (cluster bins with an auto w_cc)
// so the rule that defines a pair lives in one place. The interface sets
// the bin counts and cg_lens_bin, draws a new cluster.random_pairs, then
// calls any accessor below once, serially (the warm-up), and only then
// reads the counts. The richness pairs per z bin, R(R+1)/2 with
// R = richness_nbin, have no struct field.
//
// Cache invalidation: the maps rebuild when cluster.random_pairs or any
// bin count (cluster, source, lens, richness) differs from the ones they
// were built with. Zero at program start: empty maps until the first call
// that sees a nonzero bin count. The range checks use the counts of the
// maps themselves.


// ---------------------------------------------------------------------------
// The pair maps, built by cluster_pair_maps.
// ---------------------------------------------------------------------------
static struct {
  uint64_t cache;                      // cluster.random_pairs of the maps
  int nbin_cluster;                    // bin counts of the maps
  int nbin_source;
  int nbin_lens;
  int nbin_richness;
  int n_cs;                            // number of pairs of each block
  int n_cg;
  int n_cc;
  int cs_index[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS]; // [ni][ns] -> n
  int cs_cluster_bin[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS]; // n -> ni
  int cs_source_bin[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS];  // n -> ns
  int cg_index[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS]; // [ni][ng] -> n or -1
  int cg_cluster_bin[MAX_SIZE_ARRAYS];            // n -> ni
  int cg_lens_bin[MAX_SIZE_ARRAYS];               // n -> ng
  int cc_index[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS]; // [nl1][nl2] -> n
  int cc_richness1[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS]; // n -> nl1
  int cc_richness2[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS]; // n -> nl2 (>= nl1)
} pairs_;


// ---------------------------------------------------------------------------
// Builds pairs_ and writes the pair counts into cluster (section banner).
// Aborts on a bin count outside [0, MAX_SIZE_ARRAYS].
// ---------------------------------------------------------------------------
static void cluster_pair_maps(void)
{
  if (fdiff2(pairs_.cache, cluster.random_pairs) ||
      pairs_.nbin_cluster  != cluster.zdist_nbin ||
      pairs_.nbin_source   != redshift.shear_nbin ||
      pairs_.nbin_lens     != redshift.clustering_nbin ||
      pairs_.nbin_richness != cluster.richness_nbin)
  {
    const int nbin_cluster  = cluster.zdist_nbin;
    const int nbin_source   = redshift.shear_nbin;
    const int nbin_lens     = redshift.clustering_nbin;
    const int nbin_richness = cluster.richness_nbin;

    if (nbin_cluster  < 0 || nbin_cluster  > MAX_SIZE_ARRAYS ||
        nbin_source   < 0 || nbin_source   > MAX_SIZE_ARRAYS ||
        nbin_lens     < 0 || nbin_lens     > MAX_SIZE_ARRAYS ||
        nbin_richness < 0 || nbin_richness > MAX_SIZE_ARRAYS)
    {
      log_fatal("invalid bin counts (cluster, source, lens, richness) = "
        "(%d, %d, %d, %d)", nbin_cluster, nbin_source, nbin_lens,
        nbin_richness);
      exit(1);
    }

    // --- 1. CLUSTER LENSING: EVERY PAIR, CLUSTER-MAJOR ---
    int n = 0;
    for (int ni=0; ni<nbin_cluster; ni++) {
      for (int ns=0; ns<nbin_source; ns++) {
        pairs_.cs_index[ni][ns]  = n;
        pairs_.cs_cluster_bin[n] = ni;
        pairs_.cs_source_bin[n]  = ns;
        n++;
      }
    }
    pairs_.n_cs = n;

    // --- 2. CLUSTER-GALAXY: ONE LENS BIN PER CLUSTER BIN ---
    n = 0;
    for (int ni=0; ni<nbin_cluster; ni++) {
      for (int ng=0; ng<nbin_lens; ng++) {
        pairs_.cg_index[ni][ng] = -1;
      }

      const int ng = cluster.cg_lens_bin[ni];
      const int valid_lens_bin = (ng >= 0 && ng < nbin_lens);
      if (valid_lens_bin) {
        pairs_.cg_index[ni][ng]  = n;
        pairs_.cg_cluster_bin[n] = ni;
        pairs_.cg_lens_bin[n]    = ng;
        n++;
      }
    }
    pairs_.n_cg = n;

    // --- 3. CLUSTER-CLUSTER: RICHNESS PAIRS nl1 <= nl2, ROW-MAJOR ---
    n = 0;
    for (int nl1=0; nl1<nbin_richness; nl1++) {
      for (int nl2=nl1; nl2<nbin_richness; nl2++) {
        pairs_.cc_index[nl1][nl2] = n;
        pairs_.cc_index[nl2][nl1] = n;
        pairs_.cc_richness1[n]    = nl1;
        pairs_.cc_richness2[n]    = nl2;
        n++;
      }
    }
    pairs_.n_cc = n;

    // --- 4. THE COUNTS THE REST OF THE CODE READS ---
    cluster.cs_npowerspectra = pairs_.n_cs;
    cluster.cg_npowerspectra = pairs_.n_cg;
    cluster.cc_npowerspectra = nbin_cluster;

    // --- 5. CACHE TAGS ---
    pairs_.cache         = cluster.random_pairs;
    pairs_.nbin_cluster  = nbin_cluster;
    pairs_.nbin_source   = nbin_source;
    pairs_.nbin_lens     = nbin_lens;
    pairs_.nbin_richness = nbin_richness;
  }
}


// ---------------------------------------------------------------------------
// Cluster-lensing pair index of cluster bin ni and source bin ns
// (ni*shear_nbin + ns: every pair exists).
//
// Parameters:
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//   ns - source bin (0 .. redshift.shear_nbin - 1)
//
// Returns:
//   pair index (0 .. cluster.cs_npowerspectra - 1)
// ---------------------------------------------------------------------------
int N_cs(const int ni, const int ns)
{
  cluster_pair_maps();

  if (ni < 0 || ni > pairs_.nbin_cluster - 1 ||
      ns < 0 || ns > pairs_.nbin_source - 1)
  {
    log_fatal("invalid bin input (ni, ns) = (%d, %d) (max = (%d, %d))",
      ni, ns, pairs_.nbin_cluster, pairs_.nbin_source);
    exit(1);
  }
  return pairs_.cs_index[ni][ns];
}


// ---------------------------------------------------------------------------
// Cluster redshift bin of cluster-lensing pair n.
//
// Parameters:
//   n - pair index (0 .. cluster.cs_npowerspectra - 1)
//
// Returns:
//   cluster redshift bin ni of that pair
// ---------------------------------------------------------------------------
int ZC_cs(const int n)
{
  cluster_pair_maps();

  if (n < 0 || n > pairs_.n_cs - 1) {
    log_fatal("invalid pair input n = %d (cs pairs = %d)", n, pairs_.n_cs);
    exit(1);
  }
  return pairs_.cs_cluster_bin[n];
}


// ---------------------------------------------------------------------------
// Source bin of cluster-lensing pair n.
//
// Parameters:
//   n - pair index (0 .. cluster.cs_npowerspectra - 1)
//
// Returns:
//   source bin ns of that pair
// ---------------------------------------------------------------------------
int ZS_cs(const int n)
{
  cluster_pair_maps();

  if (n < 0 || n > pairs_.n_cs - 1) {
    log_fatal("invalid pair input n = %d (cs pairs = %d)", n, pairs_.n_cs);
    exit(1);
  }
  return pairs_.cs_source_bin[n];
}


// ---------------------------------------------------------------------------
// Cluster-galaxy pair index of cluster bin ni and lens bin ng.
//
// Parameters:
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//   ng - lens bin (0 .. redshift.clustering_nbin - 1)
//
// Returns:
//   pair index (0 .. cluster.cg_npowerspectra - 1), or -1 when ng is not
//   the lens bin cluster.cg_lens_bin[ni]
// ---------------------------------------------------------------------------
int N_cg(const int ni, const int ng)
{
  cluster_pair_maps();

  if (ni < 0 || ni > pairs_.nbin_cluster - 1 ||
      ng < 0 || ng > pairs_.nbin_lens - 1)
  {
    log_fatal("invalid bin input (ni, ng) = (%d, %d) (max = (%d, %d))",
      ni, ng, pairs_.nbin_cluster, pairs_.nbin_lens);
    exit(1);
  }
  return pairs_.cg_index[ni][ng];
}


// ---------------------------------------------------------------------------
// Cluster redshift bin of cluster-galaxy pair n.
//
// Parameters:
//   n - pair index (0 .. cluster.cg_npowerspectra - 1)
//
// Returns:
//   cluster redshift bin ni of that pair
// ---------------------------------------------------------------------------
int ZC_cg(const int n)
{
  cluster_pair_maps();

  if (n < 0 || n > pairs_.n_cg - 1) {
    log_fatal("invalid pair input n = %d (cg pairs = %d)", n, pairs_.n_cg);
    exit(1);
  }
  return pairs_.cg_cluster_bin[n];
}


// ---------------------------------------------------------------------------
// Lens bin of cluster-galaxy pair n.
//
// Parameters:
//   n - pair index (0 .. cluster.cg_npowerspectra - 1)
//
// Returns:
//   lens bin ng of that pair
// ---------------------------------------------------------------------------
int ZG_cg(const int n)
{
  cluster_pair_maps();

  if (n < 0 || n > pairs_.n_cg - 1) {
    log_fatal("invalid pair input n = %d (cg pairs = %d)", n, pairs_.n_cg);
    exit(1);
  }
  return pairs_.cg_lens_bin[n];
}


// ---------------------------------------------------------------------------
// Richness-pair index of (nl1, nl2) in the w_cc block of a cluster z bin.
// Symmetric (the N_shear convention): (nl1, nl2) and (nl2, nl1) give the
// index of the ordered pair (min, max).
//
// Parameters:
//   nl1, nl2 - richness bins (0 .. cluster.richness_nbin - 1)
//
// Returns:
//   pair index (0 .. R(R+1)/2 - 1, R = cluster.richness_nbin)
// ---------------------------------------------------------------------------
int N_cc_richness(const int nl1, const int nl2)
{
  cluster_pair_maps();

  if (nl1 < 0 || nl1 > pairs_.nbin_richness - 1 ||
      nl2 < 0 || nl2 > pairs_.nbin_richness - 1)
  {
    log_fatal("invalid bin input (nl1, nl2) = (%d, %d) (max = %d)",
      nl1, nl2, pairs_.nbin_richness);
    exit(1);
  }
  return pairs_.cc_index[nl1][nl2];
}


// ---------------------------------------------------------------------------
// First (smaller) richness bin of w_cc richness pair n.
//
// Parameters:
//   n - pair index (0 .. R(R+1)/2 - 1, R = cluster.richness_nbin)
//
// Returns:
//   richness bin nl1 of that pair
// ---------------------------------------------------------------------------
int NL1_cc(const int n)
{
  cluster_pair_maps();

  if (n < 0 || n > pairs_.n_cc - 1) {
    log_fatal("invalid pair input n = %d (cc richness pairs = %d)", n,
      pairs_.n_cc);
    exit(1);
  }
  return pairs_.cc_richness1[n];
}


// ---------------------------------------------------------------------------
// Second (larger or equal) richness bin of w_cc richness pair n.
//
// Parameters:
//   n - pair index (0 .. R(R+1)/2 - 1, R = cluster.richness_nbin)
//
// Returns:
//   richness bin nl2 of that pair
// ---------------------------------------------------------------------------
int NL2_cc(const int n)
{
  cluster_pair_maps();

  if (n < 0 || n > pairs_.n_cc - 1) {
    log_fatal("invalid pair input n = %d (cc richness pairs = %d)", n,
      pairs_.n_cc);
    exit(1);
  }
  return pairs_.cc_richness2[n];
}
