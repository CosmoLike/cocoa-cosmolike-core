#include <math.h>
#include <stdlib.h>

#include "ssc_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// v2d stores two doubles in positions called lanes. SIMD operations apply
// the same calculation to both lanes, here two independent radial samples.
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Average the long-wavelength density field over the angular survey mask.
//
// A density perturbation larger than the footprint changes the clustering
// measured inside it. Its survey average at distance chi is
//
//   delta_b(chi) = integral dOmega W(n) delta(chi*n) / Omega_W.
//
// Write the mask in spherical harmonics. Its raw power spectrum is
// C_L^W = sum_M |w_LM|^2/(2L+1), including the monopole
// C_0^W = Omega_W^2/(4 pi). Angular averaging then gives the background
// covariance sum_L (2L+1) C_L^W C_L(chi,chi') / Omega_W^2.
//
// In the long-mode Limber approximation the radial covariance is local:
//
//   <delta_b(chi) delta_b(chi')> = delta_D(chi-chi') sigma_b^2(chi),
//   sigma_b^2(chi) = sum_L (2L+1) C_L^W P_lin((L+1/2)/f_K,a)
//                   / [Omega_W^2 f_K^2].
//
// The Dirac delta has units 1/length, so sigma_b^2 has units of LENGTH.
// It is not the dimensionless variance of a finite radial bin. The later
// SSC integral is sum_p dchi_p sigma_b^2(chi_p) Phi_i,p Phi_j,p.
// With Phi in 1/length this gives a dimensionless angular covariance.
//
// This is the spherical-mask version of Takada & Hu (2013), Appendix A,
// Eq. 54, arXiv:1302.6994v3. The mask harmonic normalization also follows
// Barreira, Krause & Schmidt (2018), Eqs. 60-62, arXiv:1711.07467.
// Applying Limber to the long modes is an APPROXIMATION even at L=0.
// This routine is not their full curved-sky, non-Limber SSC prediction.
// Keep a general radial covariance kernel available in the survey driver.
//
// Parameters and ownership:
//   nnode, nmask - positive sizes; mask nodes are all integers from zero
//   area_sr - integral of W, in steradians (not square degrees)
//   mask_cl - nonnegative raw C_L^W, NOT C_L/C_0 or a unit-variance mask
//   distance - positive f_K at each radial sample, in one common unit
//   power - finite nonnegative linear P, in that same unit cubed
//   sigma2 - caller-owned output, disjoint from the read-only inputs
// No allocation or cache. Row pointers allow padded, noncontiguous rows.
// Workers own radial pairs; each vector lane sums L in increasing order.
// No angular sum is shared between threads or reduced across vector lanes.
// ---------------------------------------------------------------------------
void ssc_mask_variance_cov(
    const int nnode,                    // number of radial nodes
    const int nmask,                    // number of mask multipoles
    const double area_sr,              // integral of the angular mask
    const double* mask_cl,             // raw mask power spectrum
    const double* distance,            // transverse comoving distances
    const double* const* power,        // linear P per radial/mask node
    double* sigma2                     // background strength per node
  )
{
  if (nnode < 1
      || nmask < 1
      || !isfinite(area_sr)
      || area_sr <= 0.0) {
    log_fatal("ssc_mask_variance_cov needs positive sizes and mask area");
    exit(1);
  }
  // The monopole checks the mask convention before any background sum.
  const double monopole = area_sr*area_sr/(4.0*M_PI);
  if (!isfinite(mask_cl[0])
      || fabs(mask_cl[0]/monopole-1.0) > 1.e-8) {
    log_fatal("ssc_mask_variance_cov needs raw C_0 = area^2/(4 pi); "
              "got %g, expected %g", mask_cl[0], monopole);
    exit(1);
  }
  // A mask power cannot be negative; every radial distance is positive.
  for (int ell=0; ell<nmask; ell++) {
    if (!isfinite(mask_cl[ell])
        || mask_cl[ell] < 0.0) {
      log_fatal("ssc_mask_variance_cov: invalid mask C_%d = %g",
                ell, mask_cl[ell]);
      exit(1);
    }
  }

  for (int node=0; node<nnode; node++) {
    if (!isfinite(distance[node])
        || distance[node] <= 0.0) {
      log_fatal("ssc_mask_variance_cov: need positive finite f_K[%d]",
                node);
      exit(1);
    }
  }

  // Area normalization is common to every radial sample. Distance is not:
  // each lane receives its own f_K^-2 only after the angular sum is done.
  const double inv_area2 = 1.0/(area_sr*area_sr);

  // Averaging a long-wavelength fluctuation over the footprint weights
  // its angular modes by the mask power. At each distance, the Limber
  // approximation converts mode L into k=(L+1/2)/f_K, so its contribution
  // uses that distance's linear P(k,a) and the geometric factor 1/f_K^2.
  // Sum those contributions to obtain sigma_b^2 at each distance. A worker
  // owns two distances; SIMD shares their mask weights but keeps one
  // complete angular sum per lane. A later stage integrates over distance.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node+=2) {
    // Each lane owns a complete sum over mask multipoles at one distance.
    // For an odd node count, repeat the last valid input in lane 1.
    const int next = node+1 < nnode ? node+1 : node;
    const double* restrict power0 = power[node];
    const double* restrict power1 = power[next];

    // Start both distance-specific sums at zero: vsum = [0,0].
    v2d vsum = simde_mm_setzero_pd();

    // Add one mask multipole to both radial sums. SIMD shares its angular
    // weight while keeping the two distance-dependent power values separate.
    for (int ell=0; ell<nmask; ell++) {
      // set_pd takes lane 1 first: lane 0 receives P at node, lane 1
      // P at next, both evaluated at this mask multipole.
      const v2d vpower = simde_mm_set_pd(power1[ell], power0[ell]);

      // This mask/mode-count weight is shared by the two distances.
      const double weight = (2.0*ell+1.0)*mask_cl[ell]*inv_area2;

      // Copy the common weight into both lanes, [weight,weight].
      const v2d vweight = simde_mm_set1_pd(weight);

      // Add P*weight to each distance's own sum. fmadd uses one rounding
      // for multiply-plus-add on native FMA hardware; lanes stay separate.
      vsum = simde_mm_fmadd_pd(vpower, vweight, vsum);
    }

    double result[2];

    // Store the two sums into result[0/1] for node/next. storeu permits
    // this ordinary two-double array without special vector alignment.
    simde_mm_storeu_pd(result, vsum);

    // Complete each sum with its own f_K^-2. Discard the duplicate second
    // lane at an odd endpoint, so each physical node is written once.
    sigma2[node] = result[0]/(distance[node]*distance[node]);
    if (node+1 < nnode) {
      sigma2[next] = result[1]/(distance[next]*distance[next]);
    }
  }
}


// ---------------------------------------------------------------------------
// Turn a three-dimensional power response into an angular shell response.
//
// Let D(k,chi) = dP(k,chi)/d(delta_b) at fixed global wavenumber. Replacing
// P by P + D delta_b in the Limber spectrum gives the first term below:
//
//   Phi_AB(ell,chi) = W_A W_B D((ell+1/2)/f_K,chi)/f_K^2
//                     - [U_A(chi)+U_B(chi)] C_AB(ell).
//
// U_A describes how the SURVEY MEAN used to normalize catalog A changes:
// mean_A = mean_A,0 [1 + integral dchi U_A(chi) delta_b(chi)]. Dividing
// the two observed fields by their perturbed means multiplies C_AB by
// 1 - integral dchi (U_A+U_B) delta_b. This derives the minus sign and
// explains why the complete C_AB appears, not just its local integrand.
// Set U=0 for a field whose estimator does not divide by a catalog mean.
// For a narrow galaxy slice without magnification U=b n_chi, recovering
// the familiar subtraction of one bias times P per galaxy leg.
//
// The local-mean correction is discussed by Takada & Hu (2013), Eq. 23.
// Its radial form here follows by differentiating the projected estimator;
// see the external study's derivation, report 10, Section 3.5. U for a
// general counts/RSD/magnification estimator must be supplied explicitly.
// It is not automatically the short-mode, ell-dependent window W_A.
//
// Parameters:
//   pair_window - product W_A W_B including the chosen spin convention
//   mean_window - U_A+U_B, in inverse length; zero for global-mean spectra
//   power_response - dimensional D, not its dimensionless ratio D/P
//   signal - full C_AB of the same model used for the Gaussian spectra
//   distance - positive f_K; response - output Phi, in inverse length
// nrow counts independent pair/multipole combinations; nnode is common.
// No dchi weight, mask variance, noise or halo-model choice is inserted.
//
// All arrays are caller-owned and finite. Writable rows are disjoint
// from each other and from inputs. There is no allocation, global state
// or cache. Each worker owns one row. Two lanes process adjacent nodes;
// the scalar tail uses the same fused subtract for an odd node count.
// ---------------------------------------------------------------------------
void ssc_shell_response_cov(
    const int nrow,                     // number of pair/multipole rows
    const int nnode,                    // radial node count
    const double* distance,            // transverse comoving distances
    const double* signal,              // full angular spectra
    const double* const* pair_window,  // short-mode window products
    const double* const* mean_window,  // catalog-mean response windows
    const double* const* power_response, // dimensional matter response
    double* const* response             // output angular shell response
  )
{
  if (nrow < 1
      || nnode < 1) {
    log_fatal("ssc_shell_response_cov needs positive row and node counts");
    exit(1);
  }
  for (int node=0; node<nnode; node++) {
    if (!isfinite(distance[node])
        || distance[node] <= 0.0) {
      log_fatal("ssc_shell_response_cov: need positive finite f_K[%d]",
                node);
      exit(1);
    }
  }

  // A background fluctuation changes both the matter power in a shell and
  // the catalog means used to normalize the observed fields. The first
  // effect adds W_A*W_B*D/f_K^2. Dividing by the perturbed means reduces
  // the signal by (U_A+U_B)*C_AB to first order, explaining the subtraction.
  // Each worker obtains this net response for one spectrum at every shell.
  // SIMD evaluates two shells together, one per lane. Their responses stay
  // separate because the subsequent SSC integral supplies the radial weights.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<nrow; row++) {
    const double* restrict pair = pair_window[row];
    const double* restrict mean = mean_window[row];
    const double* restrict dp = power_response[row];
    double* restrict phi = response[row];

    // The same complete C_AB multiplies both local mean responses.
    // set1_pd copies that spectrum into lanes 0 and 1.
    const v2d vsignal = simde_mm_set1_pd(signal[row]);

    int node = 0;

    // Compute Phi = (W_A W_B)*D/f_K^2 - (U_A+U_B)*C_AB at two radial nodes
    // per iteration, then store both values. SIMD lanes own node/node+1;
    // both must exist for each two-value load/store, and are never summed.
    for (; node+1<nnode; node+=2) {
      // Load distances at node and node+1 into lanes 0 and 1.
      // loadu allows ordinary double-array storage without vector alignment.
      const v2d vf = simde_mm_loadu_pd(distance+node);

      // Load W_A W_B at those same nodes in the same lane order.
      // loadu requires two valid doubles, but no vector-aligned address.
      const v2d vpair = simde_mm_loadu_pd(pair+node);

      // Load D=dP/d(delta_b) for node and node+1. loadu has no extra
      // vector-alignment requirement for this response array.
      const v2d vdp = simde_mm_loadu_pd(dp+node);

      // Load U_A+U_B at both nodes. loadu accepts the ordinary mean-window
      // array address and keeps node in lane 0, node+1 in lane 1.
      const v2d vmean = simde_mm_loadu_pd(mean+node);

      // Square each distance independently to form its own f_K^2.
      const v2d vdistance2 = simde_mm_mul_pd(vf, vf);

      // Multiply the two-field window by D at the matching radial node.
      const v2d vproduct = simde_mm_mul_pd(vpair, vdp);

      // Divide each product by its own distance squared: the local
      // projected power response before correcting the catalog means.
      const v2d vlocal = simde_mm_div_pd(vproduct, vdistance2);

      // fnmadd computes -(mean*signal)+local in each lane. The subtraction
      // is fused with the product into one native-FMA rounding.
      const v2d vphi = simde_mm_fnmadd_pd(vmean, vsignal, vlocal);

      // Write Phi to phi[node] and phi[node+1]. storeu accepts their
      // ordinary array address without requiring vector alignment.
      simde_mm_storeu_pd(phi+node, vphi);
    }

    // If one node remains, use the same scalar formula and fused rounding
    // without reading or writing a nonexistent second node.
    if (node < nnode) {
      const double local = pair[node]*dp[node]
                           /(distance[node]*distance[node]);
      phi[node] = fma(-mean[node], signal[row], local);
    }
  }
}
