#include <math.h>
#include <stdlib.h>

#include "ssc_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d; // two independent radial samples

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
  const double monopole = area_sr*area_sr/(4.0*M_PI);
  if (!isfinite(mask_cl[0])
      || fabs(mask_cl[0]/monopole-1.0) > 1.e-8) {
    log_fatal("ssc_mask_variance_cov needs raw C_0 = area^2/(4 pi); "
              "got %g, expected %g", mask_cl[0], monopole);
    exit(1);
  }
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
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node+=2) {
    const int next = node+1 < nnode ? node+1 : node;
    const double* restrict power0 = power[node];
    const double* restrict power1 = power[next];
    v2d vsum = simde_mm_setzero_pd();
    for (int ell=0; ell<nmask; ell++) {
      const v2d vpower = simde_mm_set_pd(power1[ell], power0[ell]);
      const double weight = (2.0*ell+1.0)*mask_cl[ell]*inv_area2;
      const v2d vweight = simde_mm_set1_pd(weight);
      vsum = simde_mm_fmadd_pd(vpower, vweight, vsum);
    }
    double result[2];
    simde_mm_storeu_pd(result, vsum);
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

  #pragma omp parallel for schedule(static)
  for (int row=0; row<nrow; row++) {
    const double* restrict pair = pair_window[row];
    const double* restrict mean = mean_window[row];
    const double* restrict dp = power_response[row];
    double* restrict phi = response[row];
    const v2d vsignal = simde_mm_set1_pd(signal[row]);
    int node = 0;
    for (; node+1<nnode; node+=2) {
      const v2d vf = simde_mm_loadu_pd(distance+node);
      const v2d vpair = simde_mm_loadu_pd(pair+node);
      const v2d vdp = simde_mm_loadu_pd(dp+node);
      const v2d vmean = simde_mm_loadu_pd(mean+node);
      const v2d vdistance2 = simde_mm_mul_pd(vf, vf);
      const v2d vproduct = simde_mm_mul_pd(vpair, vdp);
      const v2d vlocal = simde_mm_div_pd(vproduct, vdistance2);
      const v2d vphi = simde_mm_fnmadd_pd(vmean, vsignal, vlocal);
      simde_mm_storeu_pd(phi+node, vphi);
    }
    if (node < nnode) {
      const double local = pair[node]*dp[node]
                           /(distance[node]*distance[node]);
      phi[node] = fma(-mean[node], signal[row], local);
    }
  }
}
