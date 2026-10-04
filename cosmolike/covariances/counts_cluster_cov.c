#include <math.h>
#include <stdlib.h>

#include "counts_cluster_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"

// SIMD applies the same arithmetic to two doubles, called lanes. Each
// lane below describes one radial shell; it never mixes distinct shells.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Construct the mean count density and its long-mode response.
//
// Let n_i(chi) be the comoving abundance selected into observed bin i.
// It includes richness scatter, completeness and the redshift selection.
// A shell of thickness dchi subtends volume Omega_s f_K(chi)^2 dchi,
// so its expected count is
//
//   dN_i = Omega_s f_K^2 n_i dchi.
//
// Now place the shell in a coherent background overdensity delta_b.
// Write its selected abundance as n_i + B_i delta_b to first order.
// For a fixed selection, B_i is the mass integral of abundance times
// halo bias times selection probability. A selection that itself responds
// to environment needs that extra derivative in B_i as well. Therefore
// this routine accepts B_i explicitly instead of guessing it from a bias
// fitted to a different observable, such as cluster lensing.
//
// The two returned shell quantities are
//
//   S_i(chi)   = dN_i/dchi = Omega_s f_K^2 n_i,
//   Phi_i(chi) = Omega_s f_K^2 B_i.
//
// Mean counts follow from integral dchi S_i. Their SSC is
// integral dchi sigma_b^2 Phi_i Phi_j in the long-mode Limber model.
// sigma_b^2 has units LENGTH because its radial covariance contains a
// Dirac delta. Both S and Phi have units 1/LENGTH, so this covariance
// has the units of a squared count, as required.
//
// Use the SAME Phi with the two-point shell response for count-spectrum
// SSC. Its local term contains W_A W_B/f_K^2 times dP/d(delta_b).
// The f_K^2 here cancels that denominator in the local contribution.
// Omitting the denominator would leave two unwanted powers of distance
// and make the result depend on the chosen length unit. Keep any separate
// observed-mean correction too. Angular transforms add no radial factors.
//
// See Takada & Spergel (2014), arXiv:1307.4399, Sec. 4.1; and Schaan,
// Takada & Spergel (2014), arXiv:1406.3330, Eqs. 33 and 35. This function
// supplies responses only. The Poisson count term and the non-SSC
// count-spectrum correlation must be assembled separately. Counts are
// absolute numbers, so no observed-catalog-mean subtraction applies here.
//
// Input and ownership contract:
// All arrays are finite. density is nonnegative; its response may have
// either sign. One consistent length unit is used throughout. The caller
// owns both outputs and ensures that their rows do not overlap each other
// or any input. No allocation, global state, cache or cosmology mutation.
// ---------------------------------------------------------------------------
void counts_shell_cluster_cov(
    const int ncount,                       // observed count bins
    const int nnode,                        // common radial nodes
    const double area_sr,                  // angular survey area
    const double* distance,                // transverse distances
    const double* const* density,          // selected abundance
    const double* const* density_response, // abundance response
    double* const* shell,                   // count density output
    double* const* response                 // count response output
  )
{
  if (ncount < 1
      || nnode < 1
      || !isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    log_fatal("counts_shell_cluster_cov needs positive sizes and "
              "0 < area_sr <= 4 pi");
    exit(1);
  }

  // Volume conversion is undefined at a nonpositive distance. The
  // radial integration must sample shells away from the observer.
  for (int node=0; node<nnode; node++) {
    if (!isfinite(distance[node])
        || distance[node] <= 0.0) {
      log_fatal("counts_shell_cluster_cov: f_K[%d] = %g must be "
                "finite and positive", node, distance[node]);
      exit(1);
    }
  }

  // The same observed count bin may receive halos from several true
  // redshifts. Convert every abundance and response with its own shell
  // volume; no bin-midpoint or non-overlap cut replaces this calculation.
  // Collapse both indices so even one count bin can occupy eight workers.
  // Each worker writes a distinct pair of nodes. SIMD performs the same
  // conversion at the two distances, with no sum across lanes or threads.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int bin=0; bin<ncount; bin++) {
    for (int node=0; node<nnode; node+=2) {
      // These local pointers describe separate input and output rows.
      // restrict lets the compiler keep a read value in a register:
      // storing an output cannot change a later read from an input.
      const double* restrict number = density[bin];
      const double* restrict change = density_response[bin];
      double* restrict count_shell = shell[bin];
      double* restrict count_response = response[bin];

      if (node+1 < nnode) {
        // loadu puts f_K[node] in lane 0 and f_K[node+1] in lane 1.
        // It accepts an ordinary double array without special alignment.
        const v2d vdistance = simde_mm_loadu_pd(distance+node);

        // Square the two distances separately. Each shell's comoving
        // volume per steradian per dchi is its own f_K^2.
        const v2d vdistance2 = simde_mm_mul_pd(vdistance, vdistance);

        // Copy the footprint area into both lanes; it is common to shells.
        const v2d varea = simde_mm_set1_pd(area_sr);

        // Multiply each f_K^2 by the area, giving dV/dchi in both lanes.
        const v2d vvolume = simde_mm_mul_pd(varea, vdistance2);

        // Load n_i at the same two shells. loadu requires two valid
        // adjacent doubles but no vector-aligned address.
        const v2d vnumber = simde_mm_loadu_pd(number+node);

        // Load B_i in matching lane order, with the same alignment rule.
        const v2d vchange = simde_mm_loadu_pd(change+node);

        // Volume times abundance gives each shell's expected dN/dchi.
        const v2d vcount = simde_mm_mul_pd(vvolume, vnumber);

        // Volume times B_i gives each shell's response to delta_b.
        const v2d vresponse = simde_mm_mul_pd(vvolume, vchange);

        // Write the two count densities back to adjacent ordinary doubles.
        // storeu preserves lane order and needs no special alignment.
        simde_mm_storeu_pd(count_shell+node, vcount);

        // Write the two responses to their separate output row, likewise
        // allowing an ordinary address and preserving node/node+1 order.
        simde_mm_storeu_pd(count_response+node, vresponse);
      } else {
        // An odd final node has no partner. Use the identical sequence
        // of multiplications without reading beyond any array boundary.
        const double volume = area_sr*(distance[node]*distance[node]);
        count_shell[node] = volume*number[node];
        count_response[node] = volume*change[node];
      }
    }
  }
}
