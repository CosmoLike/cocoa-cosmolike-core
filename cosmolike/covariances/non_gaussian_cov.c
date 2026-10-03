#include <math.h>
#include <stdlib.h>

#include "non_gaussian_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d; // two independent k or K,Q samples

// ---------------------------------------------------------------------------
// Assemble a dimensional halo-model response with explicit model choices.
//
// The two-halo power is P_2h = I11^2 P_lin; the one-halo power is I02.
// A background density mode changes their amplitudes. In the adopted
// halo approximation the one-halo derivative is I12, while the two-halo
// term contains growth and a dilation (rescaling of wavenumber):
//
//   P_halo = P_2h + I02,
//   D_halo = (growth_coefficient - dilation_coefficient * slope) P_2h
//            + I12.
//
// The supplied slope is dlnP_X/dlnk, WITHOUT the k^3 factor. Published
// isotropic halo response: growth=47/21, dilation=1/3, P_X=P_2h.
// This is Takada & Hu (2013), arXiv:1302.6994v3, corrected Eq. 44:
// 68/21 - (1/3) dln(k^3 P_2h)/dlnk = 47/21 - (1/3) dlnP_2h/dlnk.
//
// A planar squeezed-tree construction instead uses growth=17/7,
// dilation=1/2 and P_X=P_lin (study report 10, Section 3.1). Its linear
// limit is R1+RK/6 with tree-level responses, as in Barreira, Krause &
// Schmidt (2018), arXiv:1711.07467, Sections 2 and 4.1. This is not a
// calibrated nonlinear tidal response. These choices are deliberately
// explicit inputs; the code does not label a study-derived form a paper
// erratum or silently replace the published two-halo slope.
//
// If fractional=1, preserve the halo model's FRACTIONAL response but use
// a supplied target power (for example Halofit): D=(D_halo/P_halo) P_target.
// If fractional=0, return D_halo. This replacement is a modeling choice,
// not an identity of perturbation theory. Galaxy survey-mean subtraction
// belongs later in ssc_shell_response_cov, not in this matter response.
//
// Inputs are six rows: P_lin, P_target, I11, I02, I12, slope. All must
// refer to the same density field and (k,a). Power and D have length^3;
// I11, slope and the coefficients are dimensionless. P_halo must be
// positive. output contains P_halo and D; it must not overlap inputs.
// No table lookup, allocation, cache, noise or survey window is added.
// Workers own pairs of points; an odd final lane is read but not stored.
// ---------------------------------------------------------------------------
void halo_response_cov(
    const int npoint,                // independent k,a points
    const double growth_coefficient, // constant response coefficient
    const double dilation_coefficient, // logarithmic-slope coefficient
    const int fractional,           // rescale fractional halo response
    const double* const* inputs,    // six physical input rows
    double* const* output            // halo power and dimensional response
  )
{
  if (npoint < 1
      || !isfinite(growth_coefficient)
      || !isfinite(dilation_coefficient)
      || (fractional != 0
          && fractional != 1)) {
    log_fatal("halo_response_cov needs points, finite coefficients, "
              "and fractional = 0 or 1");
    exit(1);
  }
  for (int point=0; point<npoint; point++) {
    const double p2h = inputs[2][point]*inputs[2][point]*inputs[0][point];
    const double phalo = p2h+inputs[3][point];
    if (!isfinite(phalo)
        || phalo <= 0.0) {
      log_fatal("halo_response_cov needs positive halo power at point %d",
                point);
      exit(1);
    }
  }

  const v2d vgrowth = simde_mm_set1_pd(growth_coefficient);
  const v2d vdilation = simde_mm_set1_pd(dilation_coefficient);
  #pragma omp parallel for schedule(static)
  for (int point=0; point<npoint; point+=2) {
    const int next = point+1 < npoint ? point+1 : point;
    v2d values[6];
    for (int role=0; role<6; role++) {
      values[role] = simde_mm_set_pd(inputs[role][next], inputs[role][point]);
    }
    const v2d vi11_squared = simde_mm_mul_pd(values[2], values[2]);
    const v2d vp2h = simde_mm_mul_pd(vi11_squared, values[0]);
    const v2d vhalo = simde_mm_add_pd(vp2h, values[3]);
    const v2d vcoefficient = simde_mm_fnmadd_pd(vdilation, values[5], vgrowth);
    v2d vresponse = simde_mm_fmadd_pd(vcoefficient, vp2h, values[4]);
    if (fractional) {
      vresponse = simde_mm_mul_pd(simde_mm_div_pd(vresponse, vhalo),
                                 values[1]);
    }
    double result[2][2];
    simde_mm_storeu_pd(result[0], vhalo);
    simde_mm_storeu_pd(result[1], vresponse);
    for (int role=0; role<2; role++) {
      output[role][point] = result[role][0];
      if (point+1 < npoint) {
        output[role][next] = result[role][1];
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Assemble the five connected halo contributions at each (K,Q,a).
//
// Four density legs can occupy one, two, three or four halos. Two halos
// have two different partitions: one leg plus three, or two plus two.
// For the covariance parallelogram (k,-k,q,-q), angular averaging gives
//
//   T_1h  = I04(K,K,Q,Q),
//   T_13  = 2 [P_K I11(K) I13(K,Q,Q) + P_Q I11(Q) I13(K,K,Q)],
//   T_22  = 2 I12(K,Q)^2 AvgP,
//   T_3h  = 4 I12(K,Q) I11(K) I11(Q) AvgB,
//   T_4h  = [I11(K) I11(Q)]^2 AvgT.
//
// In T_13 the isolated leg can be k, -k, q or -q: FOUR choices. The
// first two have the same magnitudes, as do the last two. Thus each
// displayed term has coefficient two, not one. This follows directly
// from the four set partitions in Takada & Hu (2013), Eqs. 28-29,
// arXiv:1302.6994; the study's old CosmoCov implementation has only one.
// The two nonzero exchange channels give T_22's factor two; four choices
// of the paired halo give T_3h's factor four. tree_averages_cov supplies
// the linear power, bispectrum and trispectrum averages with the exact
// zero-internal-momentum SSC channels already excluded.
//
// The moments follow halo_cov.h: rows I02, I12, I13(K,Q,Q), I13(K,K,Q),
// I04. I02 is not used here; keeping the same row layout avoids a second
// moment-table convention. pk and i11 have K and Q rows. tree has AvgP,
// AvgB, AvgT rows. All inputs are finite and use the same density field,
// redshift, wavenumbers and length unit. Every output has dimension L^9.
//
// terms stores these five contributions separately. The caller can sum
// them and apply the projection integral dchi W_A W_B W_C W_D T
// / (area*f_K^6). No field windows, spin factors, shot-noise trispectrum
// or survey geometry enter this routine. A tree/halo approximation need
// not be a positive-semidefinite covariance at every node; never clip its
// eigenvalues or replace a negative contribution with zero.
//
// All arrays are caller-owned. Output rows are disjoint from one another
// and all inputs. No cache, allocation or cross-thread sum is used.
// Two SIMD lanes process different pairs; the final odd lane is discarded.
// ---------------------------------------------------------------------------
void halo_trispectrum_cov(
    const int npoint,                // independent K,Q,a combinations
    const double* const* pk,        // P_lin(K), P_lin(Q)
    const double* const* i11,       // I11(K), I11(Q)
    const double* const* moments,   // five pair moments
    const double* const* tree,      // three planar tree averages
    double* const* terms             // five halo contributions
  )
{
  if (npoint < 1) {
    log_fatal("halo_trispectrum_cov needs at least one point");
    exit(1);
  }
  const v2d vtwo = simde_mm_set1_pd(2.0);
  const v2d vfour = simde_mm_set1_pd(4.0);
  #pragma omp parallel for schedule(static)
  for (int point=0; point<npoint; point+=2) {
    const int next = point+1 < npoint ? point+1 : point;
    v2d vp[2];
    v2d vi[2];
    v2d vm[5];
    v2d vt[3];
    for (int role=0; role<2; role++) {
      vp[role] = simde_mm_set_pd(pk[role][next], pk[role][point]);
      vi[role] = simde_mm_set_pd(i11[role][next], i11[role][point]);
    }
    for (int role=0; role<5; role++) {
      vm[role] = simde_mm_set_pd(moments[role][next], moments[role][point]);
    }
    for (int role=0; role<3; role++) {
      vt[role] = simde_mm_set_pd(tree[role][next], tree[role][point]);
    }

    const v2d vi_product = simde_mm_mul_pd(vi[0], vi[1]);
    const v2d vleft13 = simde_mm_mul_pd(simde_mm_mul_pd(vp[0], vi[0]), vm[2]);
    const v2d vright13 = simde_mm_mul_pd(simde_mm_mul_pd(vp[1], vi[1]), vm[3]);
    v2d vterms[5];
    vterms[0] = vm[4];
    vterms[1] = simde_mm_mul_pd(vtwo, simde_mm_add_pd(vleft13, vright13));
    vterms[2] = simde_mm_mul_pd(simde_mm_mul_pd(vtwo,
        simde_mm_mul_pd(vm[1], vm[1])), vt[0]);
    vterms[3] = simde_mm_mul_pd(simde_mm_mul_pd(vfour,
        simde_mm_mul_pd(vm[1], vi_product)), vt[1]);
    vterms[4] = simde_mm_mul_pd(simde_mm_mul_pd(vi_product, vi_product), vt[2]);
    for (int role=0; role<5; role++) {
      double result[2];
      simde_mm_storeu_pd(result, vterms[role]);
      terms[role][point] = result[0];
      if (point+1 < npoint) {
        terms[role][next] = result[1];
      }
    }
  }
}
