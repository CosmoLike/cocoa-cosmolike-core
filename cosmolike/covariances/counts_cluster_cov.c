#include <math.h>
#include <stdlib.h>

#include "counts_cluster_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"

// SIMD (single instruction, multiple data) applies one arithmetic
// instruction to several numbers at once. A v2d, SIMDe's simde__m128d,
// holds two doubles; each position is called a lane, lane 0 the low and
// lane 1 the high double. Below, lane 0 holds radial shell node and lane
// 1 shell node+1. No instruction combines the two lanes, so each shell
// is converted exactly as in the scalar code.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Construct the mean count density and its long-mode response.
//
// Let n_i(chi) be the comoving abundance selected into observed bin i.
// It includes richness scatter, completeness and the redshift selection.
// At comoving distance chi, one radian on the sky spans the transverse
// comoving length f_K(chi), which equals chi only in a flat universe.
// A shell of thickness dchi over the solid angle Omega_s therefore has
// comoving volume dV = Omega_s f_K(chi)^2 dchi, and its expected count is
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
// integral dchi sigma_b^2 Phi_i Phi_j in the long-mode Limber model,
// which replaces the background correlation between two shells by
//
//   <delta_b(chi) delta_b(chi')> = delta_D(chi-chi') sigma_b^2(chi).
//
// The Dirac delta has units 1/length, so sigma_b^2 has units of length;
// it is not the dimensionless variance of a finite shell. Both S and Phi
// have units 1/length, so this covariance has the units of a squared
// count, as required.
//
// Counts are absolute numbers, not density contrasts divided by an
// observed catalog mean. Phi_i therefore contains no observed-mean
// subtraction.
//
// Distinct observed bins are exclusive: each halo receives at most one
// observed label, drawn independently of other halos given its mass and
// redshift. Splitting a Poisson population by independent labels gives
// independent Poisson populations, so the shot-noise covariance is
// diagonal, delta_ij N_i, even when two bins overlap in true mass or true
// redshift. Their SSC term is still nonzero: both bins respond to the
// same delta_b at every shell where both have halos.
//
// Use this same Phi with the two-point shell response for count-spectrum
// SSC. Its local term contains W_A W_B/f_K^2 times D = dP/d(delta_b).
// The f_K^2 here cancels that denominator in the local contribution:
//
//   Phi_i W_A W_B D/f_K^2 = Omega_s B_i W_A W_B D.
//
// Omitting the denominator would leave two unwanted powers of distance
// and make the result depend on the chosen length unit. The two-point
// response keeps its own observed-mean term, -(U_A+U_B) C_AB; only the
// count response lacks one. Angular transforms add no radial factors.
//
// References. Takada & Spergel (2014), arXiv:1307.4399, Sec. 4.1, derive
// the light-cone count covariance and its cross-covariance with lensing.
// Schaan, Takada & Spergel (2014), arXiv:1406.3330, Eq. 33 is the count
// covariance, a diagonal Poisson term plus
//
//   Omega_s^2 integral dchi n_i n_j b_i b_j chi^4 dsigma^2(chi),
//
// that is integral dchi sigma_b^2 Phi_i Phi_j with B = b n. The last term
// of their Eq. 35, Omega_s integral dchi q^2 (sum_i n_i b_i) D dsigma^2,
// is the count-lensing SSC: q^2 appears with no power of distance, the
// cancellation above. Their dsigma^2 is the sigma_b^2 used here. Both
// papers take a flat universe, chi = f_K.
//
// This function supplies responses only. The Poisson count term and the
// non-SSC count-spectrum correlation (the first line of Eq. 35) must be
// assembled separately.
//
// Parameters:
//   ncount           - number of observed count bins i, >= 1
//   nnode            - number of common radial nodes chi_j, >= 1
//   area_sr          - survey solid angle Omega_s in steradians,
//                      0 < area_sr <= 4 pi
//   distance         - [nnode] f_K(chi_j), positive, in length unit L
//   density          - [ncount][nnode] n_i(chi_j) in L^-3, nonnegative
//   density_response - [ncount][nnode] B_i(chi_j) in L^-3, either sign
//
// Outputs (every entry is written):
//   shell            - [ncount][nnode] S_i(chi_j) = dN_i/dchi, in L^-1
//   response         - [ncount][nnode] Phi_i(chi_j), in L^-1 per unit
//                      delta_b
//
// Input and ownership contract:
// All arrays are finite. One consistent length unit is used throughout.
// The caller owns both outputs and ensures that their rows do not overlap
// each other or any input. No allocation, global state, cache or
// cosmology mutation.
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

  // Every radial node must be a shell beyond the observer, where f_K > 0.
  // A zero or negative value means a node at the observer or a corrupted
  // distance array, so the function stops instead of converting it. The
  // two-point response paired with Phi divides by f_K^2 at the same nodes.
  for (int node=0; node<nnode; node++) {
    if (!isfinite(distance[node])
        || distance[node] <= 0.0) {
      log_fatal("counts_shell_cluster_cov: f_K[%d] = %g must be "
                "finite and positive", node, distance[node]);
      exit(1);
    }
  }

  // One iteration converts the selected abundance n_i and its response
  // B_i of one observed bin at two adjacent radial nodes into dN_i/dchi
  // and Phi_i. An observed bin receives halos from every true redshift
  // its selection allows, so each node uses its own shell volume
  // Omega_s f_K^2; no bin-midpoint or non-overlap cut replaces this.
  //
  // node advances by two: SIMD lane 0 holds shell node and lane 1 shell
  // node+1. The lanes are never added; each fills its own output entry.
  // When nnode is odd, the final iteration has one node left and takes
  // the scalar branch. Collapse both indices so even one count bin can
  // occupy eight workers. Each (bin, node pair) writes distinct entries
  // and nothing is summed, so the result does not depend on the threads.
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

      // scalar: for each of the two shells j = node and j = node+1,
      //   volume = area_sr*(distance[j]*distance[j]);
      //   count_shell[j] = volume*number[j];
      //   count_response[j] = volume*change[j];
      // The shell volume converts number per comoving volume into dN/dchi;
      // the same factor converts its density response into dN/dchi/delta_b.
      // The SIMD block evaluates these three lines at both shells together,
      // with the same multiplications in the same order. Each product is
      // rounded once, so every lane equals its scalar result bitwise.
      if (node+1 < nnode) {
        // scalar: volume = area_sr*(distance[j]*distance[j])

        // loadu puts f_K[node] = distance[node] in lane 0 and f_K[node+1]
        // = distance[node+1] in lane 1. It accepts an ordinary double
        // array without special alignment.
        const v2d vdistance = simde_mm_loadu_pd(distance+node);

        // mul_pd multiplies lane by lane: f_K[node]^2 in lane 0 and
        // f_K[node+1]^2 in lane 1. Each shell's comoving volume per
        // steradian per dchi is its own f_K^2.
        const v2d vdistance2 = simde_mm_mul_pd(vdistance, vdistance);

        // set1 copies the footprint area Omega_s = area_sr into both
        // lanes; it is common to the two shells.
        const v2d varea = simde_mm_set1_pd(area_sr);

        // mul_pd: Omega_s f_K^2 in each lane, the comoving volume per unit
        // distance dV/dchi of shell node (lane 0) and node+1 (lane 1).
        const v2d vvolume = simde_mm_mul_pd(varea, vdistance2);

        // scalar: count_shell[j] = volume*number[j];
        //         count_response[j] = volume*change[j]

        // loadu puts number[node] = n_i(chi_node) in lane 0 and
        // number[node+1] in lane 1. It requires two valid adjacent
        // doubles but no vector-aligned address.
        const v2d vnumber = simde_mm_loadu_pd(number+node);

        // loadu puts change[node] = B_i(chi_node) = dn_i/d(delta_b) in
        // lane 0 and change[node+1] in lane 1, with the same rule.
        const v2d vchange = simde_mm_loadu_pd(change+node);

        // mul_pd: dV/dchi times n_i in each lane, the shell's expected
        // count per unit distance S_i = dN_i/dchi.
        const v2d vcount = simde_mm_mul_pd(vvolume, vnumber);

        // mul_pd: dV/dchi times B_i in each lane, the shell's count
        // response Phi_i to delta_b, with no observed-mean subtraction.
        const v2d vresponse = simde_mm_mul_pd(vvolume, vchange);

        // storeu writes lane 0 to count_shell[node] and lane 1 to
        // count_shell[node+1]. It needs no special alignment, but both
        // elements must exist, which node+1 < nnode guarantees.
        simde_mm_storeu_pd(count_shell+node, vcount);

        // storeu writes the two responses to count_response[node] (lane
        // 0) and count_response[node+1] (lane 1), under the same rule.
        simde_mm_storeu_pd(count_response+node, vresponse);
      } else {
        // An odd final node (node = nnode-1) has no partner. Use the
        // identical sequence of multiplications without reading beyond
        // any array boundary.
        const double volume = area_sr*(distance[node]*distance[node]);
        count_shell[node] = volume*number[node];
        count_response[node] = volume*change[node];
      }
    }
  }
}
