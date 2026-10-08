#include <math.h>
#include <stdlib.h>

#include "ssc_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one arithmetic
// instruction to several numbers at once; each number occupies a position
// called a lane. v2d stores two doubles, lane 0 and lane 1. SIMDe is a
// header-only portability library: each simde_mm_* call compiles to the
// matching native instruction (SSE2/FMA on x86, NEON on ARM processors
// such as Apple Silicon), or to plain C where no such instruction exists.
// In both functions below the two lanes hold two independent radial
// samples; no instruction combines numbers from different lanes.
//
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// The two fused calls used here, fmadd (a*b+c) and fnmadd (c-a*b), are
// single-rounding instructions on ARM64 NEON and on x86 with FMA enabled.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Average the long-wavelength density field over the angular survey mask.
//
// A density perturbation larger than the footprint changes the clustering
// measured inside it. Its survey average at distance chi is
//
//   delta_b(chi) = integral dOmega W(n) delta(chi*n) / Omega_W,
//
// where W(n) is the mask (one inside the footprint and zero outside, or a
// weight) and Omega_W = integral dOmega W is its area in steradians.
//
// Write the mask in spherical harmonics, W(n) = sum_LM w_LM Y_LM(n). Its
// raw power spectrum is C_L^W = sum_M |w_LM|^2/(2L+1). The monopole is
// fixed by the area: Y_00 = 1/sqrt(4 pi), so w_00 = Omega_W/sqrt(4 pi)
// and C_0^W = Omega_W^2/(4 pi). Expanding delta on two shells the same way,
// the orthogonality of the Y_LM gives the background covariance
//
//   <delta_b(chi) delta_b(chi')> = sum_L (2L+1) C_L^W C_L(chi,chi')
//                                  / Omega_W^2,
//   C_L(chi,chi') = (2/pi) integral k^2 dk P_lin(k) j_L(k chi) j_L(k chi').
//
// The Limber approximation uses that the spherical Bessel functions j_L
// oscillate much faster in k than P_lin varies:
//
//   integral k^2 dk P(k) j_L(k chi) j_L(k chi')
//       ~ (pi/2) P((L+1/2)/chi) delta_D(chi-chi')/chi^2.
//
// Applied to the long modes, the radial covariance becomes local:
//
//   <delta_b(chi) delta_b(chi')> = delta_D(chi-chi') sigma_b^2(chi),
//   sigma_b^2(chi) = sum_L (2L+1) C_L^W P_lin((L+1/2)/f_K,a)
//                   / [Omega_W^2 f_K^2].
//
// The Dirac delta has units 1/length, so sigma_b^2 has units of length.
// It is not the dimensionless variance of a finite radial bin. The later
// SSC integral is sum_p dchi_p sigma_b^2(chi_p) Phi_i,p Phi_j,p.
// With Phi in 1/length this gives a dimensionless angular covariance.
//
// This is the spherical-mask version of Takada & Hu (2013), Appendix A,
// Eq. 54, arXiv:1302.6994v3. The mask harmonic normalization also follows
// Barreira, Krause & Schmidt (2018), Eqs. 60-62, arXiv:1711.07467.
// Applying Limber to the long modes is an approximation even at L=0.
// This routine is not their full curved-sky, non-Limber SSC prediction.
// Keep a general radial covariance kernel available in the survey driver.
//
// Map to the code: weight = (2L+1) mask_cl[L]/area_sr^2 is the same at
// every distance; power[p][L] is P_lin((L+1/2)/f_K, a) at distance p, with
// a the scale factor on the light cone there; the sum runs over the
// supplied band L = 0..nmask-1; the division by f_K^2 comes last.
//
// Parameters and ownership:
//   nnode, nmask - positive sizes; mask nodes are all integers from zero
//   area_sr - integral of W, in steradians (not square degrees)
//   mask_cl - nonnegative raw C_L^W, not C_L/C_0 or a unit-variance mask
//   distance - positive f_K at each radial sample, in one common unit
//   power - finite nonnegative linear P, in that same unit cubed
//   sigma2 - caller-owned output, disjoint from the read-only inputs
// No allocation or cache. Row pointers allow padded, noncontiguous rows.
// Workers own pairs of radial nodes; each vector lane sums L in increasing
// order. No angular sum is shared between threads or reduced across vector
// lanes.
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
  // A raw spectrum has C_0 = area^2/(4 pi). A file normalized to C_0 = 1,
  // or to a unit zero-lag correlation, would rescale sigma_b^2 without any
  // visible sign, so it stops here. The relative tolerance 1e-8 accepts
  // rounding in a stored spectrum, not a different normalization.
  const double monopole = area_sr*area_sr/(4.0*M_PI);
  if (!isfinite(mask_cl[0])
      || fabs(mask_cl[0]/monopole-1.0) > 1.e-8) {
    log_fatal("ssc_mask_variance_cov needs raw C_0 = area^2/(4 pi); "
              "got %g, expected %g", mask_cl[0], monopole);
    exit(1);
  }
  // A mask power cannot be negative: each C_L^W is a sum of squared
  // magnitudes |w_LM|^2. Every radial distance must be positive because
  // the result is divided by f_K^2.
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
  // One iteration of this loop handles nodes node and node+1; the static
  // schedule gives each thread a contiguous block of such pairs, and no
  // thread writes another thread's outputs.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node+=2) {
    // Each lane owns a complete sum over mask multipoles at one distance.
    // For an odd node count, repeat the last valid input in lane 1.
    const int next = node+1 < nnode ? node+1 : node;

    // P_lin rows of the two distances, indexed by mask multipole L.
    // restrict promises the compiler that no other pointer modifies these
    // rows while they are read. Both may point to the same row at an odd
    // endpoint; that is allowed because neither pointer writes.
    const double* restrict power0 = power[node];
    const double* restrict power1 = power[next];

    // scalar: for each distance j = node (lane 0) and j = next (lane 1),
    //   sum = 0;
    //   for (int ell=0; ell<nmask; ell++) {
    //     weight = (2*ell+1)*mask_cl[ell]*inv_area2;
    //     sum = fma(power[j][ell], weight, sum);
    //   }
    //   sigma2[j] = sum/(distance[j]*distance[j]);
    // Each mask mode contributes background power at that distance.
    // SIMD carries two independent sums; it does not add their distances.

    // vsum = [0, 0]: both distance-specific sums start at zero (setzero
    // sets every lane to 0.0).
    v2d vsum = simde_mm_setzero_pd();

    // Add one mask multipole per iteration to both radial sums, in
    // increasing L. The angular weight (2L+1) C_L^W/Omega_W^2 is common to
    // both distances; only the power differs, because the same L probes
    // the wavenumber k = (L+1/2)/f_K, which is different at each distance.
    for (int ell=0; ell<nmask; ell++) {
      // vpower = [power0[ell], power1[ell]] = [P at node, P at next], both
      // at this mask multipole. set_pd takes lane 1 first, lane 0 last.
      const v2d vpower = simde_mm_set_pd(power1[ell], power0[ell]);

      // weight = (2L+1) C_L^W/Omega_W^2, the mask/mode-count weight
      // shared by the two distances, computed once as an ordinary double.
      const double weight = (2.0*ell+1.0)*mask_cl[ell]*inv_area2;

      // vweight = [weight, weight]: set1 copies the common weight into
      // both lanes.
      const v2d vweight = simde_mm_set1_pd(weight);

      // scalar: sum = fma(power[j][ell], weight, sum) in each lane.
      // fmadd computes vpower*vweight + vsum with one rounding per lane on
      // native FMA hardware (NEON fmla on ARM64); lanes stay separate.
      vsum = simde_mm_fmadd_pd(vpower, vweight, vsum);
    }

    double result[2];

    // result[0..1] = [sum at node, sum at next]: storeu writes lane 0 to
    // result[0] and lane 1 to result[1]. This ordinary two-double array
    // needs no special vector alignment.
    simde_mm_storeu_pd(result, vsum);

    // Complete each sum with its own f_K^-2, sigma2[j] = sum_j/f_K[j]^2, in
    // scalar code. Discard the duplicate second lane at an odd endpoint, so
    // each physical node is written once.
    sigma2[node] = result[0]/(distance[node]*distance[node]);
    if (node+1 < nnode) {
      sigma2[next] = result[1]/(distance[next]*distance[next]);
    }
  }
}


// ---------------------------------------------------------------------------
// Turn a three-dimensional power response into an angular shell response.
//
// Let D(k,chi) = dP(k,chi)/d(delta_b) at fixed global wavenumber. In the
// Limber spectrum C_AB(ell) = integral dchi W_A W_B P((ell+1/2)/f_K)/f_K^2,
// replacing P by P + D delta_b gives the first term below:
//
//   Phi_AB(ell,chi) = W_A W_B D((ell+1/2)/f_K,chi)/f_K^2
//                     - [U_A(chi)+U_B(chi)] C_AB(ell).
//
// Phi is the functional derivative dC_AB(ell)/d delta_b(chi): to first
// order a background fluctuation changes the measured spectrum by
// delta C_AB(ell) = integral dchi Phi_AB(ell,chi) delta_b(chi).
// Units: W_A W_B is 1/length^2, D is length^3 and f_K^2 is length^2; U is
// 1/length and C_AB is dimensionless. Both terms are 1/length.
//
// U_A describes how the survey mean used to normalize catalog A changes:
// mean_A = mean_A,0 [1 + integral dchi U_A(chi) delta_b(chi)]. Dividing
// the two observed fields by their perturbed means multiplies C_AB by
// 1 - integral dchi (U_A+U_B) delta_b. This derives the minus sign and
// explains why the complete C_AB appears, not just its local integrand.
// Set U=0 for a field whose estimator does not divide by a catalog mean,
// such as cosmic shear.
//
// Example: a narrow galaxy slice of width Delta chi, linear bias b and no
// magnification has U = W = b n_chi with n_chi = 1/Delta chi inside the
// slice, and C_AA = b^2 P/(f_K^2 Delta chi). The mean term 2 U C_AA then
// equals W_A W_A (2 b P)/f_K^2, so Phi = W_A W_A (D - 2 b P)/f_K^2: the
// familiar subtraction of one bias times P per galaxy leg.
//
// The local-mean correction is Takada & Hu (2013), Eq. 23: for a density
// measured against the survey mean, dP_W/d delta_b = dP/d delta_b - 2P.
// Its radial form here follows by differentiating the projected estimator;
// see the external study's derivation, report 10, Section 3.5. U for a
// general counts/RSD/magnification estimator must be supplied explicitly.
// It is not automatically the short-mode, ell-dependent window W_A.
//
// Parameters:
//   pair_window - product W_A W_B including the chosen spin convention
//   mean_window - U_A+U_B, in inverse length; zero for global-mean spectra
//   power_response - dimensional D, not its dimensionless ratio D/P
//   signal - full C_AB(ell) of the mean model whose survey-mean
//            normalization the estimator adopts. The shared survey
//            assembly deliberately supplies zero-IA Limber spectra here,
//            even when its Gaussian signal adds non-Limber or IA terms;
//            the covariance README states that approximation.
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
  // One iteration of this loop is one row, a field pair at one multipole;
  // the static schedule gives each thread a block of whole rows.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<nrow; row++) {
    // This spectrum's rows of W_A W_B, U_A+U_B and D, and its output row.
    // restrict promises the compiler that the output row does not overlap
    // the input rows (the ownership contract above), so values already
    // loaded need not be reloaded after each store to phi.
    const double* restrict pair = pair_window[row];
    const double* restrict mean = mean_window[row];
    const double* restrict dp = power_response[row];
    double* restrict phi = response[row];

    // scalar: at each shell j of this spectrum,
    //   local = pair[j]*dp[j]/(distance[j]*distance[j]);
    //   phi[j] = fma(-mean[j], signal[row], local);
    // The first line projects the matter response. The second subtracts
    // the change caused by normalizing galaxy counts by their survey mean,
    // local - mean*signal with one rounding. Two SIMD lanes apply these
    // lines to adjacent shells j = node (lane 0) and node+1 (lane 1).

    // vsignal = [signal[row], signal[row]]: the same complete C_AB
    // multiplies both local mean responses. set1_pd copies that spectrum
    // into lanes 0 and 1.
    const v2d vsignal = simde_mm_set1_pd(signal[row]);

    int node = 0;

    // Compute Phi = (W_A W_B)*D/f_K^2 - (U_A+U_B)*C_AB at two radial nodes
    // per iteration, then store both values. SIMD lanes own node/node+1;
    // both must exist for each two-value load/store, and are never summed.
    // The bound node+1 < nnode guarantees both; an odd count leaves one
    // final node for the scalar tail after the loop.
    for (; node+1<nnode; node+=2) {
      // vf = distance[node..node+1], node in lane 0 and node+1 in lane 1.
      // loadu reads two consecutive doubles from ordinary array storage
      // without vector alignment.
      const v2d vf = simde_mm_loadu_pd(distance+node);

      // vpair = pair[node..node+1] = W_A W_B at the same nodes, in the
      // same lane order. loadu requires two valid doubles, but no
      // vector-aligned address.
      const v2d vpair = simde_mm_loadu_pd(pair+node);

      // vdp = dp[node..node+1] = D=dP/d(delta_b) at node and node+1. loadu
      // has no extra vector-alignment requirement for this response array.
      const v2d vdp = simde_mm_loadu_pd(dp+node);

      // vmean = mean[node..node+1] = U_A+U_B at both nodes. loadu accepts
      // the ordinary mean-window array address and keeps node in lane 0,
      // node+1 in lane 1.
      const v2d vmean = simde_mm_loadu_pd(mean+node);

      // scalar: local = pair[j]*dp[j]/(distance[j]*distance[j])

      // f_K^2 = distance*distance, each lane squaring its own distance.
      const v2d vdistance2 = simde_mm_mul_pd(vf, vf);

      // W_A W_B * D: the two-field window times D at the matching node.
      const v2d vproduct = simde_mm_mul_pd(vpair, vdp);

      // local = W_A W_B D/f_K^2, each product divided by its own distance
      // squared: the local projected power response before correcting the
      // catalog means.
      const v2d vlocal = simde_mm_div_pd(vproduct, vdistance2);

      // scalar: phi[j] = fma(-mean[j], signal[row], local)
      // fnmadd computes -(vmean*vsignal)+vlocal = local - (U_A+U_B) C_AB in
      // each lane. The subtraction is fused with the product into one
      // native-FMA rounding (NEON fmls on ARM64, one FMA instruction on
      // x86 with FMA).
      const v2d vphi = simde_mm_fnmadd_pd(vmean, vsignal, vlocal);

      // phi[node..node+1] = vphi: storeu writes lane 0 to phi[node] and
      // lane 1 to phi[node+1], both valid nodes of this row, without
      // requiring vector alignment.
      simde_mm_storeu_pd(phi+node, vphi);
    }

    // If one node remains, use the same scalar formula. C's fma() rounds
    // once, like the fused vector lanes on FMA hardware, and no
    // nonexistent second node is read or written.
    if (node < nnode) {
      const double local = pair[node]*dp[node]
                           /(distance[node]*distance[node]);
      phi[node] = fma(-mean[node], signal[row], local);
    }
  }
}
