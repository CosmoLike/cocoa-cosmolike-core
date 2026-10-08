#include <math.h>
#include <stdlib.h>

#include "assembly_cov.h"
#include "gaussian_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"

// ---------------------------------------------------------------------------
// Assemble the Gaussian covariance of every measured pair of catalogs.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// A measured spectrum AB correlates with CD through two Wick contractions:
// C_AC*C_BD + C_AD*C_BC. The internal spectra include every catalog pair,
// even a pair excluded from the measured data vector. Independent catalog
// noise enters a crossed spectrum only when its two field IDs coincide.
// For observables r = AB and s = CD, gaussian_wick_cov returns
//
//   G_rs(ell) = [(C_AC+N_AC)(C_BD+N_BD) + (C_AD+N_AD)(C_BC+N_BC)]
//               / [(2 ell+1) fsky],       fsky = area_sr/(4 pi),
//
// diagonal in ell: in this approximation different multipoles do not
// couple. A measured row is linear in its spectrum,
// x_(r,i) = sum_ell K_(p_r,i)(ell) C_AB(ell), with K the bin operator of
// the probe p_r of observable r. The covariance is therefore the
// both-sides contraction (gaussian_project_cov)
//
//   Cov[(r,i),(s,j)] = sum_ell K_(p_r,i)(ell) G_rs(ell) K_(p_s,j)(ell),
//
// which is O_r G_rs O_s^T with a diagonal G_rs.
//
// Each bin operator averages the harmonic covariance into an angular bin
// or a Fourier band. In real space the white-noise tail extends beyond any
// finite ell_max: integrate signal and mixed noise here, then add the exact
// pure-noise pair-count expression on equal bins (gaussian_noise_pair_cov).
// A Fourier band has finite support and includes all three terms in its
// harmonic integral. For xi+ and xi- the B-mode Wick term is added with
// sign +1 for equal probes and -1 for xi+ with xi-.
//
// WORK OWNERSHIP AND THREADS
// A task owns an observable block (r <= s) and its transpose. One OpenMP
// team shares the flat task list; each worker reuses private scratch.
// Inside that active team the helpers find omp_in_parallel() true and
// start no nested team, so the calling worker computes its whole block.
// The C projection retains its SIMD arithmetic and ordered multipole sums,
// irrespective of the team size. With one observable the if clause leaves
// the region inactive (one thread); omp_in_parallel() is then false and
// the helpers distribute multipoles and bins themselves. No kernel,
// covariance formula or task loop lives in the interfaces.
//
// Inputs use the units and row shapes in assembly_cov.h. The output has
// observable first, bin second (row r*nbin + i), and is overwritten in
// full. Both triangles receive the same number, without averaging or
// repairing eigenvalues.
// ---------------------------------------------------------------------------
void gaussian_matrix_cov(
    const int nell,                       // consecutive integer multipoles
    const int nfield,                     // lens plus source catalogs
    const int nobs,                       // measured catalog pairs
    const int nbin,                       // angular or Fourier bins
    const int* rows,                      // flat [nobs,3] (probe,A,B)
    const double* const* spectra,         // [nfield*nfield,nell], signal
    const double* const* b_spectra,       // optional BB rows, NULL for E only
    const double* noise,                  // [nfield], white noise powers
    const double* const* kernels,         // [4*nbin,nell], or [nbin,nell]
    const int ell_min,                    // first input multipole
    const double area_sr,                 // common footprint area
    const double* pair_area,              // [nbin] sr^2, NULL for Fourier
    const int realspace,                  // exact real-space pure noise
    double* const* output                 // [nobs*nbin,nobs*nbin]
  )
{
  if (nell < 1
      || nfield < 1
      || nobs < 1
      || nbin < 1) {
    log_fatal("gaussian_matrix_cov needs positive axis lengths");
    exit(1);
  }

  // --- 1. ENUMERATE DISTINCT OBSERVABLE BLOCKS ---

  // Cov(AB,CD)=Cov(CD,AB), so only blocks with first <= second are
  // computed. The flat list holds (first, second) in tasks[2*task] and
  // tasks[2*task+1]. A flat triangle gives each worker similar numbers of
  // blocks; a loop over triangular rows would be unbalanced.
  const int ntask = nobs*(nobs+1)/2; // distinct observable pairs
  int* tasks = malloc(2*(size_t) ntask*sizeof(int)); // two row IDs per task
  int task = 0; // next unfilled task

  for (int first=0; first<nobs; first++) {
    for (int second=first; second<nobs; second++) {
      tasks[2*task] = first;
      tasks[2*task+1] = second;
      task++;
    }
  }

  // --- 2. PROJECT COMPLETE BLOCKS WITH PRIVATE SCRATCH ---

  // Every task reads the same spectra/operators but writes disjoint cells:
  // its block and that block's transpose. Keeping a complete ell sum on one
  // worker makes results independent of thread count. For a single
  // observable the if clause leaves this region inactive, and C
  // parallelizes its bins instead. Each worker's private scratch, reused by
  // all its tasks: harmonics[0] (alias harmonic) holds the E-mode, later
  // combined, G(ell), and harmonics[1] the B-mode G(ell); weighted holds
  // K_left*G of one block; block holds the projected nbin-by-nbin result.
  #pragma omp parallel if(nobs > 1)
  {
    double** harmonics = (double**) malloc2d(2, nell); // E and B Wick power
    double* harmonic = harmonics[0]; // combined angular covariance
    double** weighted = (double**) malloc2d(nbin, nell); // left times power
    double** block = (double**) malloc2d(nbin, nbin); // one projected block

    #pragma omp for schedule(static)
    for (int index=0; index<ntask; index++) {
      const int first = tasks[2*index]; // left observable
      const int second = tasks[2*index+1]; // right observable
      const int a = rows[3*first+1]; // first field of AB
      const int b = rows[3*first+2]; // second field of AB
      const int c = rows[3*second+1]; // first field of CD
      const int d = rows[3*second+2]; // second field of CD
      const int left_probe = realspace ? rows[3*first] : 0; // operator role
      const int right_probe = realspace ? rows[3*second] : 0; // role for CD
      const int fields[4] = {a, b, c, d}; // pure-noise catalog IDs
      const double noise_ab[2] = {noise[a], noise[b]}; // AB white powers
      const double cross_noise[4] = { // AC,BD,AD,BC in Wick order
        a == c ? noise[a] : 0.0,
        b == d ? noise[b] : 0.0,
        a == d ? noise[a] : 0.0,
        b == c ? noise[b] : 0.0
      };
      const double* cross[4] = { // contiguous ell rows, without copying
        spectra[a*nfield+c],
        spectra[b*nfield+d],
        spectra[a*nfield+d],
        spectra[b*nfield+c]
      };

      // G_rs(ell) of the E modes (or scalar fields) into harmonic, with
      // fsky = area_sr/(4 pi). The flag !realspace keeps the pure-noise
      // product NN only for Fourier bands; real space adds it below from
      // pair counts.
      gaussian_wick_cov(ell_min, nell, area_sr/(4.0*M_PI), cross,
          cross_noise, !realspace, harmonic);
      if (realspace
          && b_spectra != NULL
          && left_probe <= XI_MINUS_COV
          && right_probe <= XI_MINUS_COV) {
        // xi+ measures EE+BB; xi- measures EE-BB. With parity-symmetric
        // fields EB vanishes, so their Gaussian covariances add with the
        // product of these signs: G^EE + sign*G^BB, with sign = +1 for
        // equal probes and -1 for xi+ with xi-. Shape-noise BB*BB is
        // already included in the analytic pair term below: add only BB
        // signal and B-noise, the C^BB C^BB and mixed C^BB N terms (flag 0
        // omits NN). Shape noise is the same for E and B modes, so
        // cross_noise serves both calls.
        const double* cross_b[4] = {
          b_spectra[a*nfield+c],
          b_spectra[b*nfield+d],
          b_spectra[a*nfield+d],
          b_spectra[b*nfield+c]
        };
        gaussian_wick_cov(ell_min, nell, area_sr/(4.0*M_PI), cross_b,
            cross_noise, 0, harmonics[1]);
        const double sign = left_probe == right_probe ? 1.0 : -1.0;

        // scalar: for (int ell=0; ell<nell; ell++) {
        //           harmonic[ell] += sign*harmonics[1][ell];
        //         }
        // Here ell is a column index: column ell holds multipole
        // ell_min+ell. sign is +1 or -1, so sign*G^BB is exact and the
        // addition is the only rounding; fused or not, the scalar tail gives
        // the same double as the vector lanes. SIMD handles adjacent
        // multipoles, without changing either Wick contraction: lane 0 and
        // lane 1 are separate columns, never added to each other.

        // vsign = [sign, sign]: set1 copies the xi sign product into both
        // lanes.
        const simde__m128d vsign = simde_mm_set1_pd(sign);

        // Two adjacent columns per step while both exist (ell+1 < nell);
        // an odd nell leaves one column for the scalar remainder below.
        int ell = 0;
        for (; ell+1<nell; ell+=2) {
          // ve = [G^EE at column ell, at column ell+1]: loadu reads
          // harmonic[ell] and harmonic[ell+1] from ordinary storage.
          const simde__m128d ve = simde_mm_loadu_pd(harmonic+ell);

          // vb = [G^BB at column ell, at ell+1]: load harmonics[1][ell] and
          // harmonics[1][ell+1] into matching lanes (loadu).
          const simde__m128d vb = simde_mm_loadu_pd(harmonics[1]+ell);

          // vchange = sign*G^BB: mul applies the xi sign to each B
          // contribution independently; a product with +1 or -1 is exact.
          const simde__m128d vchange = simde_mm_mul_pd(vsign, vb);

          // vtotal = G^EE + sign*G^BB: add combines E and signed B without
          // mixing the two multipoles, one rounding per lane.
          const simde__m128d vtotal = simde_mm_add_pd(ve, vchange);

          // storeu writes lane 0 to harmonic[ell] and lane 1 to
          // harmonic[ell+1], the harmonic projection input; no vector
          // alignment is needed.
          simde_mm_storeu_pd(harmonic+ell, vtotal);
        }

        // scalar remainder: the last column of an odd nell.
        if (ell < nell) {
          harmonic[ell] += sign*harmonics[1][ell];
        }
      }

      // Both-sides contraction of this block (gaussian_project_cov):
      // block[i][j] = sum_ell K_left[i][ell] G(ell) K_right[j][ell], with K
      // the operator rows of each observable's probe (kernels+probe*nbin;
      // probe 0 for Fourier bands). weighted is this worker's K_left*G
      // scratch. Inside the team the helper runs on this worker alone and
      // sums each entry in increasing ell.
      gaussian_project_cov(nbin, nbin, nell, kernels+left_probe*nbin,
          kernels+right_probe*nbin, harmonic, weighted, block);

      if (realspace) {
        // Distinct angular bins do not share pure pair noise. Its diagonal
        // includes the entire white tail, rather than truncating at ell_max.
        // gaussian_noise_pair_cov returns the exact pair-count value, zero
        // unless the probes and catalogs match.
        for (int bin=0; bin<nbin; bin++) {
          block[bin][bin] += gaussian_noise_pair_cov(
              left_probe, right_probe, fields, noise_ab, pair_area[bin]);
        }
      }

      // Copy the block to rows first*nbin+left and columns
      // second*nbin+right, and mirror it into the transpose. A diagonal
      // observable block needs only its bin triangle: it is symmetric
      // mathematically, but block[left][right] and block[right][left] round
      // their products differently. Choose that triangle explicitly, so
      // reversed products never overwrite it. Only this task writes these
      // cells.
      for (int left=0; left<nbin; left++) {
        const int start = first == second ? left : 0; // distinct bin pair
        for (int right=start; right<nbin; right++) {
          const int i = first*nbin+left; // complete matrix row
          const int j = second*nbin+right; // complete matrix column
          output[i][j] = block[left][right];
          output[j][i] = block[left][right];
        }
      }
    }
    free(harmonics);
    free(weighted);
    free(block);
  }
  free(tasks);
}

// ---------------------------------------------------------------------------
// Add catalog windows to the angularly projected matter trispectrum.
//
// Observable r measures fields A,B and carries W_r=W_A*W_B in each radial
// shell. The connected covariance after the angular transforms is
//
//   C[(r,i),(s,j)] = sum_node W_r W_s T[p_r*nbin+i,p_s*nbin+j] measure.
//
// T is common to all catalog pairs with the same probe IDs and angular
// bins. Compute measure*T once for that angular block, then integrate all
// its catalog combinations using the existing SIMD projection. The measure
// contains dchi/(area*f_K^6). No abundance or bias approximation is added;
// the windows carry any linear bias. For one angular block this is
// W_p diag(measure*T) W_q^T: the same both-sides contraction as the
// Gaussian case, with radial nodes in place of multipoles and catalog
// windows in place of bin operators.
//
// Only blocks with left probe <= right probe, and left bin <= right bin
// for equal probes, are computed; their transposes fill the rest. The
// projected table is read only in those blocks, so it must be symmetric
// under exchanging its two transform indices, as T(k1,k2) is.
//
// Probe groups preserve input catalog order. Each task owns its block and
// transpose; round-robin static scheduling balances groups of unequal size
// over the OpenMP team. Inside the team the projection helper starts no
// nested team (omp_in_parallel), so every radial sum stays on one worker
// and retains the C kernel's order. With a single task the region is
// inactive and the helper parallelizes itself. Signed trispectra remain
// signed. Inputs are read-only and output is fully overwritten, with the
// row shapes and physical units in assembly_cov.h.
// ---------------------------------------------------------------------------
void connected_matrix_cov(
    const int nobs,                       // measured catalog pairs
    const int nbin,                       // bins for each observable
    const int nnode,                      // radial integration nodes
    const int* probes,                    // [nobs], statistic IDs 0..3
    const double* const* pair_window,     // [nobs,nnode], W_A*W_B
    const double* const* projected,       // [(4*nbin)^2,nnode], matter T
    const double* measure,                // [nnode], dchi/(area*f_K^6)
    double* const* output                 // [nobs*nbin,nobs*nbin]
  )
{
  if (nobs < 1
      || nbin < 1
      || nnode < 1) {
    log_fatal("connected_matrix_cov needs positive axis lengths");
    exit(1);
  }

  // --- 1. GROUP CATALOG PAIRS BY THEIR MEASUREMENT OPERATOR ---

  // The four probes have different spin kernels. Within a probe, all
  // catalogs share one transformed matter function. Store their row IDs
  // and borrow their radial windows; no numerical table is duplicated.
  // Slot probe*nobs+k holds the k-th observable of that probe, in input
  // order: groups[] has its row ID and windows[] its W_A*W_B row.
  int counts[4] = {0}; // number of catalog pairs in each probe
  int* groups = malloc(4*(size_t) nobs*sizeof(int)); // IDs within groups
  const double** windows = malloc(4*(size_t) nobs*sizeof(double*));
  int largest = 0; // maximum number of catalog pairs in one group

  for (int row=0; row<nobs; row++) {
    const int probe = probes[row]; // statistic assigned to this catalog pair
    if (probe < XI_PLUS_COV
        || probe > W_THETA_COV) {
      log_fatal("connected_matrix_cov: probe %d is outside 0..3", probe);
      exit(1);
    }
    const int slot = probe*nobs+counts[probe]; // next entry in its group
    groups[slot] = row;
    windows[slot] = pair_window[row];
    counts[probe]++;
    if (counts[probe] > largest) largest = counts[probe];
  }

  // --- 2. ENUMERATE DISTINCT PAIRS OF ANGULAR BINS AND PROBES ---

  // Equal probes need only the bin triangle. Different probes need every
  // bin pairing; swapping both observables supplies the transpose later.
  // At most 10 probe pairs times nbin*nbin bin pairs, four ints each,
  // bound the allocation. Probes without catalogs produce no tasks.
  int* tasks = malloc(40*(size_t) nbin*nbin*sizeof(int)); // p,q,i,j per task
  int ntask = 0; // filled entries of this upper-bound allocation
  for (int left=0; left<4; left++) {
    if (counts[left] == 0) continue;
    for (int right=left; right<4; right++) {
      if (counts[right] == 0) continue;
      for (int first=0; first<nbin; first++) {
        const int start = left == right ? first : 0; // first distinct bin
        for (int second=start; second<nbin; second++) {
          tasks[4*ntask] = left;
          tasks[4*ntask+1] = right;
          tasks[4*ntask+2] = first;
          tasks[4*ntask+3] = second;
          ntask++;
        }
      }
    }
  }

  // --- 3. INTEGRATE EACH ANGULAR BLOCK OVER ALL ITS CATALOG PAIRS ---

  // A worker receives whole blocks with private radial weights and matrix
  // scratch: weight holds measure*T at each node, weighted the left
  // windows times weight, and block one catalog block. One team serves all
  // blocks instead of one parallel region per small projection.
  // schedule(static, 1) deals tasks round-robin, task k to worker k modulo
  // the team size, spreading probe groups of very different catalog counts
  // over all workers. SIMD prepares two independent shells; the projection
  // still sums them in the established order and never divides a sum
  // among workers.
  #pragma omp parallel if(ntask > 1)
  {
    double* weight = malloc((size_t) nnode*sizeof(double)); // measure*T
    double** weighted = (double**) malloc2d(largest, nnode); // weighted W
    double** block = (double**) malloc2d(largest, largest); // catalog block

    #pragma omp for schedule(static, 1)
    for (int task=0; task<ntask; task++) {
      const int left = tasks[4*task]; // left probe
      const int right = tasks[4*task+1]; // right probe
      const int first = tasks[4*task+2]; // left angular bin
      const int second = tasks[4*task+3]; // right angular bin
      const int i = left*nbin+first; // combined left transform index
      const int j = right*nbin+second; // combined right transform index
      const double* matter = projected[i*(4*nbin)+j]; // T at radial nodes

      // scalar: for (int n=0; n<nnode; n++) {
      //           weight[n] = measure[n]*matter[n];
      //         }
      // weight[n] is the radial integrand shared by every catalog pair of
      // this angular block. SIMD puts shells n and n+1 in separate lanes
      // (positions) of one two-double register while both exist; an odd
      // nnode leaves one shell for the scalar remainder. These products
      // prepare the radial integrand, each rounded once as in the scalar
      // line; they do not add the shells or change their later summation
      // order.
      int node = 0; // first shell not yet weighted
      for (; node+1<nnode; node+=2) {
        // vm = [measure[node], measure[node+1]], the radial measures of the
        // two shells in low/high lanes. loadu accepts an ordinary double
        // address without requiring 16-byte alignment.
        const simde__m128d vm = simde_mm_loadu_pd(measure+node);

        // vt = [matter[node], matter[node+1]]: T of this angular block at
        // the same two shells, preserving lane correspondence (loadu).
        const simde__m128d vt = simde_mm_loadu_pd(matter+node);

        // Multiply within each lane, one rounding per product:
        // vw = [measure[n]*T[n], measure[n+1]*T[n+1]] with n = node.
        const simde__m128d vw = simde_mm_mul_pd(vm, vt);

        // Write lane 0 to weight[node] and lane 1 to weight[node+1], this
        // worker's integrand row. storeu has the same relaxed alignment
        // rule and preserves lane order.
        simde_mm_storeu_pd(weight+node, vw);
      }

      // scalar remainder: the last shell of an odd nnode.
      if (node < nnode) {
        weight[node] = measure[node]*matter[node];
      }

      // Both-sides contraction over catalogs (gaussian_project_cov):
      // block[a][b] = sum_node W_a weight W_b for left catalog pair a of
      // probe left and right pair b of probe right, that is
      // W_p diag(weight) W_q^T. Inside the team the helper runs on this
      // worker alone and sums the nodes in increasing order.
      gaussian_project_cov(counts[left], counts[right], nnode,
          windows+left*nobs, windows+right*nobs, weight, weighted, block);

      // On a diagonal angular block only one catalog triangle is distinct.
      // Other blocks retain all pairings. Mirror rather than averaging:
      // row groups[left*nobs+a]*nbin+first and column
      // groups[right*nobs+b]*nbin+second receive block[a][b], and so does
      // the transposed cell. Each cell belongs to exactly one task.
      const int diagonal = left == right
                           && first == second; // same statistic and bin
      for (int a=0; a<counts[left]; a++) {
        const int start = diagonal ? a : 0; // first distinct catalog partner
        for (int b=start; b<counts[right]; b++) {
          const int row = groups[left*nobs+a]*nbin+first; // matrix row
          const int col = groups[right*nobs+b]*nbin+second; // matrix column
          output[row][col] = block[a][b];
          output[col][row] = block[a][b];
        }
      }
    }
    free(weight);
    free(weighted);
    free(block);
  }
  free(tasks);
  free(groups);
  free(windows);
}
