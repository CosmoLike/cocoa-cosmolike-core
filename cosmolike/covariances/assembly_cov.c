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
// A measured spectrum AB correlates with CD through two Wick contractions:
// C_AC*C_BD + C_AD*C_BC. The internal spectra include every catalog pair,
// even a pair excluded from the measured data vector. Independent catalog
// noise enters a crossed spectrum only when its two field IDs coincide.
//
// Each bin operator averages the harmonic covariance into an angular bin
// or a Fourier band. In real space the white-noise tail extends beyond any
// finite ell_max: integrate signal and mixed noise here, then add the exact
// pure-noise pair-count expression. A Fourier band has finite support and
// includes all three terms in its harmonic integral.
//
// A task owns an observable block and its transpose. One OpenMP team shares
// the blocks; each worker reuses private scratch. The C projection retains
// its SIMD arithmetic and ordered multipole sums, irrespective of the team
// size. No kernel, covariance formula or task loop lives in the interfaces.
//
// Inputs use the units and row shapes in assembly_cov.h. The output has
// observable first, bin second, and is overwritten in full. Both triangles
// receive the same number, without averaging or repairing eigenvalues.
// ---------------------------------------------------------------------------
void gaussian_matrix_cov(
    const int nell,                       // consecutive integer multipoles
    const int nfield,                     // lens plus source catalogs
    const int nobs,                       // measured catalog pairs
    const int nbin,                       // angular or Fourier bins
    const int* rows,                      // flat [nobs,3] (probe,A,B)
    const double* const* spectra,         // [nfield*nfield,nell], signal
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

  // Cov(AB,CD)=Cov(CD,AB). A flat triangle gives each worker similar
  // numbers of blocks; a loop over triangular rows would be unbalanced.
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

  // Every task reads the same spectra/operators but writes disjoint cells.
  // Keeping a complete ell sum on one worker makes results independent of
  // thread count. For a single observable, C parallelizes its bins instead.
  #pragma omp parallel if(nobs > 1)
  {
    double* harmonic = malloc((size_t) nell*sizeof(double)); // Wick power
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

      gaussian_wick_cov(ell_min, nell, area_sr/(4.0*M_PI), cross,
          cross_noise, !realspace, harmonic);
      gaussian_project_cov(nbin, nbin, nell, kernels+left_probe*nbin,
          kernels+right_probe*nbin, harmonic, weighted, block);

      if (realspace) {
        // Distinct angular bins do not share pure pair noise. Its diagonal
        // includes the entire white tail, rather than truncating at ell_max.
        for (int bin=0; bin<nbin; bin++) {
          block[bin][bin] += gaussian_noise_pair_cov(
              left_probe, right_probe, fields, noise_ab, pair_area[bin]);
        }
      }

      // A diagonal observable block needs only its bin triangle. Choose
      // that triangle explicitly, so reversed products never overwrite it.
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
    free(harmonic);
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
// contains dchi/(area*f_K^6). No abundance or bias approximation is added.
//
// Probe groups preserve input catalog order. Each task owns its block and
// transpose; round-robin static scheduling balances groups of unequal size
// over the OpenMP team. Every radial sum retains the C kernel's order.
// Signed trispectra remain signed. Inputs are read-only and output is fully
// overwritten, with the row shapes and physical units in assembly_cov.h.
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
  // scratch. This avoids one parallel region per tiny Python projection.
  // SIMD prepares two independent shells; the projection still sums them
  // in the established order and never divides a sum among workers.
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

      // The scalar calculation is weight[n] = measure[n]*matter[n].
      // SIMD puts shells n and n+1 in separate lanes (positions) of one
      // two-double register. These products prepare the radial integrand;
      // they do not add the shells or change their later summation order.
      int node = 0; // first shell not yet weighted
      for (; node+1<nnode; node+=2) {
        // Load adjacent measures into low/high lanes. loadu accepts an
        // ordinary double address without requiring 16-byte alignment.
        const simde__m128d vm = simde_mm_loadu_pd(measure+node);

        // Load T at the same two shells, preserving lane correspondence.
        const simde__m128d vt = simde_mm_loadu_pd(matter+node);

        // Multiply within each lane: [measure[n]*T[n], measure[n+1]*T[n+1]].
        const simde__m128d vw = simde_mm_mul_pd(vm, vt);

        // Write both products into consecutive ordinary doubles. storeu
        // has the same relaxed alignment rule and preserves lane order.
        simde_mm_storeu_pd(weight+node, vw);
      }
      if (node < nnode) {
        weight[node] = measure[node]*matter[node];
      }

      gaussian_project_cov(counts[left], counts[right], nnode,
          windows+left*nobs, windows+right*nobs, weight, weighted, block);

      // On a diagonal angular block only one catalog triangle is distinct.
      // Other blocks retain all pairings. Mirror rather than averaging.
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
