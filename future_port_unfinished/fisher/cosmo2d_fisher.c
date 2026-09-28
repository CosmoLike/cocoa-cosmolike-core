#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>

#include "cosmolike/basics.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/IA.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "cosmo2d_fisher.h"

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// EXPERIMENTAL: analytic d/dX of the Limber C_ss and of xi_pm (NLA).
//
// Why a separate file: the derivative needs its own copy of the ss
// quadrature and kernel precompute (the per-parameter brackets of the
// integrand live inside the vectorized fill loop), and cosmo2D.c stays
// untouched. The pieces repeated from cosmo2D.c / cosmo2D_scuts.c are the
// Gauss-Legendre node setup of C_ss_tomo_limber_work, g_tomo's fine-grid
// lensing-efficiency integral, and the bin-averaged Legendre kernels of
// xi_pm_tomo. Derivation and verification: fisher_derivatives.pdf.
//
// Consistency rule: every VALUE (chi, dchi/da, W_kappa, W_source, the NLA
// amplitude, P_NL) comes from the same cosmolike functions the likelihood
// uses, so C here is the likelihood's C_ss; only DERIVATIVES come from
// splines of the tables (cubic in z for dchi/dX and dlnG/dX, bicubic in
// (log10 k, z) for the slope dlnP_NL/dlnk).
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// The natural cubic splines of dchi/dX(z) and dlnG/dX(z) use the table
// nodes up to this redshift: every source distribution ends well below it,
// and cutting the fit here keeps the wide gap before the recombination
// block of the chi table (z ~ 50 -> 1070) out of the spline.
#define FISHER_ZSPLINE_MAX 10.0

// Nodes kept beyond FISHER_ZSPLINE_MAX, so the natural end condition of
// the spline sits away from the integration range.
#define FISHER_ZSPLINE_PAD 4

// ---------------------------------------------------------------------------
// Response tables handed over from Python (one slot per parameter X):
//   dchi[ip][j]     dchi/dX (Mpc/h) at z = cosmology.chi[0][j]
//   dlnG[ip][j]     dlnG/dX at z = cosmology.G[0][j]
//   dlnP[ip][i][j]  dlnP_NL/dX at log10k = cosmology.lnP[i][lnP_nz] and
//                   z = cosmology.lnP[lnP_nk][j] (the lnP table layout)
//   dlnOm[ip]       explicit dlnOmega_m/dX
// ---------------------------------------------------------------------------
static struct {
  int loaded[FISHER_NPARAM_MAX];
  double dlnOm[FISHER_NPARAM_MAX];
  int nz_chi;
  int nz_G;
  int nk;
  int nz_P;
  double** dchi;
  double** dlnG;
  double*** dlnP;
} fr = {0};

// ---------------------------------------------------------------------------
// Load one parameter slot. The storage is sized once for all
// FISHER_NPARAM_MAX slots and resized (clearing every slot) only when the
// table sizes change.
// ---------------------------------------------------------------------------
void set_fisher_response(
    const int ip,             // parameter slot (0 .. FISHER_NPARAM_MAX-1)
    const double dlnOm_dX,    // explicit dlnOmega_m/dX
    const double* dchi_dX,    // dchi/dX (Mpc/h) on the cosmology.chi z grid
    const int nz_chi,         // number of chi nodes
    const double* dlnG_dX,    // dlnG/dX on the cosmology.G z grid
    const int nz_G,           // number of growth nodes
    const double* dlnPNL_dX,  // dlnP_NL/dX, element (ik, jz) at ik*nz_P + jz
    const int nk,             // number of log10k nodes
    const int nz_P            // number of P-table z nodes
  )
{
  if (ip < 0 || ip > FISHER_NPARAM_MAX - 1) {
    log_fatal("parameter slot ip = %d outside [0, %d]", ip,
              FISHER_NPARAM_MAX - 1);
    exit(1);
  }
  if (nz_chi != cosmology.chi_nz || nz_G != cosmology.G_nz ||
      nk != cosmology.lnP_nk || nz_P != cosmology.lnP_nz) {
    log_fatal("response tables do not match the set_cosmology tables");
    exit(1);
  }
  if (NULL == fr.dchi || nz_chi != fr.nz_chi || nz_G != fr.nz_G ||
      nk != fr.nk || nz_P != fr.nz_P) {
    if (fr.dchi != NULL) free(fr.dchi);
    if (fr.dlnG != NULL) free(fr.dlnG);
    if (fr.dlnP != NULL) free(fr.dlnP);
    fr.dchi = (double**) malloc2d(FISHER_NPARAM_MAX, nz_chi);
    fr.dlnG = (double**) malloc2d(FISHER_NPARAM_MAX, nz_G);
    fr.dlnP = (double***) malloc3d(FISHER_NPARAM_MAX, nk, nz_P);
    fr.nz_chi = nz_chi;
    fr.nz_G   = nz_G;
    fr.nk     = nk;
    fr.nz_P   = nz_P;
    for (int m=0; m<FISHER_NPARAM_MAX; m++) {
      fr.loaded[m] = 0;
    }
  }
  for (int j=0; j<nz_chi; j++) {
    fr.dchi[ip][j] = dchi_dX[j];
  }
  for (int j=0; j<nz_G; j++) {
    fr.dlnG[ip][j] = dlnG_dX[j];
  }
  for (int i=0; i<nk; i++) {
    for (int j=0; j<nz_P; j++) {
      fr.dlnP[ip][i][j] = dlnPNL_dX[i*nz_P + j];
    }
  }
  fr.dlnOm[ip]  = dlnOm_dX;
  fr.loaded[ip] = 1;
}

void reset_fisher_response(void)
{
  for (int m=0; m<FISHER_NPARAM_MAX; m++) {
    fr.loaded[m] = 0;
  }
}

int fisher_nparam(void)
{
  int np = 0;
  while (np < FISHER_NPARAM_MAX && 1 == fr.loaded[np]) {
    np++;
  }
  for (int m=np; m<FISHER_NPARAM_MAX; m++) {
    if (1 == fr.loaded[m]) {
      log_fatal("parameter slots must be loaded contiguously from 0 "
                "(slot %d loaded, slot %d empty)", m, np);
      exit(1);
    }
  }
  return np;
}

// ---------------------------------------------------------------------------
// dlnP_NL/dX at (log10 k [h/Mpc], z): bilinear on the lnP grid with the
// same bracket clamping as p_nonlin, so beyond the table edges it
// extrapolates linearly exactly as p_nonlin extrapolates lnP — the
// derivative of p_nonlin's extrapolation is the extrapolation of the
// derivative.
// ---------------------------------------------------------------------------
static inline double fisher_dlnP(
    const int ip,         // parameter slot
    const double log10k,  // log10(k) with k in h/Mpc
    const double z        // redshift
  )
{
  const int nk = cosmology.lnP_nk;
  const int nz = cosmology.lnP_nz;
  int i = 0;
  {
    int ilo = 0;
    int ihi = nk - 1;
    while (ihi > ilo + 1) {
      const int ll = (ihi + ilo)/2;
      if (cosmology.lnP[ll][nz] > log10k) ihi = ll; else ilo = ll;
    }
    i = ilo;
  }
  int j = 0;
  {
    int ilo = 0;
    int ihi = nz - 1;
    while (ihi > ilo + 1) {
      const int ll = (ihi + ilo)/2;
      if (cosmology.lnP[nk][ll] > z) ihi = ll; else ilo = ll;
    }
    j = ilo;
  }
  const double dx = (log10k - cosmology.lnP[i][nz])/
                    (cosmology.lnP[i+1][nz] - cosmology.lnP[i][nz]);
  const double dy = (z - cosmology.lnP[nk][j])/
                    (cosmology.lnP[nk][j+1] - cosmology.lnP[nk][j]);
  double** R = fr.dlnP[ip];
  return (1-dx)*(1-dy)*R[i][j]   + (1-dx)*dy*R[i][j+1]
       + dx*(1-dy)*R[i+1][j]     + dx*dy*R[i+1][j+1];
}

// ---------------------------------------------------------------------------
// Everything the derivative needs beyond the per-call multipoles, built
// once per public call (single-threaded) and read-only afterwards; GSL
// splines are evaluated with NULL accelerators, which keeps the parallel
// reads thread-safe.
//
//   chiX[ip], lnGX[ip]  cubic splines of dchi/dX (Mpc/h) and dlnG/dX in z
//   gam0[ip]            dlnG/dX at z = 0 (cosmolike's D is normalized at
//                       z = 0, so dlnD/dX(z) = dlnG/dX(z) - gam0)
//   lnP                 bicubic spline of lnP_NL(log10 k, z); only its
//                       log10 k derivative is used (the slope n_eff)
//   gX[ip*nbin + b][N_a] d g_tomo/dX on g_tomo's own coarse a grid
//   a[npts], wt[npts]   the ss Gauss-Legendre nodes and weights
// ---------------------------------------------------------------------------
typedef struct {
  int np;
  gsl_spline* chiX[FISHER_NPARAM_MAX];
  gsl_spline* lnGX[FISHER_NPARAM_MAX];
  double gam0[FISHER_NPARAM_MAX];
  gsl_spline2d* lnP;
  double lkmin, lkmax, zPmin, zPmax;
  double ga_min, ga_max, ga_dx;
  double** gX;
  int npts;
  double* a;
  double* wt;
} fisher_state;

// number of leading table nodes the z splines use (FISHER_ZSPLINE_MAX
// plus the FISHER_ZSPLINE_PAD guard nodes)
static int fisher_zspline_nodes(
    const double* z,  // increasing redshift nodes
    const int n       // number of nodes
  )
{
  int m = 0;
  while (m < n && z[m] <= FISHER_ZSPLINE_MAX) {
    m++;
  }
  m += FISHER_ZSPLINE_PAD;
  return (m > n) ? n : m;
}

static void fisher_state_init(
    fisher_state* fs  // output: splines, lensing-efficiency responses, nodes
  )
{
  if (nuisance.IA_MODEL == IA_MODEL_TATT) {
    log_fatal("cosmo2d_fisher supports NLA only: the TATT one-loop kernels "
              "would need FAST-PT FFTs of the response");
    exit(1);
  }
  fs->np = fisher_nparam();
  if (0 == fs->np) {
    log_fatal("no parameter response loaded (call set_fisher_response)");
    exit(1);
  }
  if (fr.nz_chi != cosmology.chi_nz || fr.nz_G != cosmology.G_nz ||
      fr.nk != cosmology.lnP_nk || fr.nz_P != cosmology.lnP_nz) {
    log_fatal("response tables no longer match the set_cosmology tables");
    exit(1);
  }
  // -------------------------------------------------------------------------
  // cubic splines in z of the geometry and growth responses
  // -------------------------------------------------------------------------
  const int nchi = fisher_zspline_nodes(cosmology.chi[0], cosmology.chi_nz);
  const int nG = fisher_zspline_nodes(cosmology.G[0], cosmology.G_nz);
  for (int ip=0; ip<fs->np; ip++) {
    fs->chiX[ip] = gsl_spline_alloc(gsl_interp_cspline, nchi);
    gsl_spline_init(fs->chiX[ip], cosmology.chi[0], fr.dchi[ip], nchi);
    fs->lnGX[ip] = gsl_spline_alloc(gsl_interp_cspline, nG);
    gsl_spline_init(fs->lnGX[ip], cosmology.G[0], fr.dlnG[ip], nG);
    fs->gam0[ip] = gsl_spline_eval(fs->lnGX[ip], 0.0, NULL);
  }
  // -------------------------------------------------------------------------
  // bicubic spline of lnP_NL(log10 k, z) for the slope dlnP/dlnk
  // -------------------------------------------------------------------------
  {
    const int nk = cosmology.lnP_nk;
    const int nz = cosmology.lnP_nz;
    double* xa = (double*) malloc1d(nk);
    double* ya = (double*) malloc1d(nz);
    double* za = (double*) malloc1d(nk*nz);
    for (int i=0; i<nk; i++) {
      xa[i] = cosmology.lnP[i][nz];
    }
    for (int j=0; j<nz; j++) {
      ya[j] = cosmology.lnP[nk][j];
    }
    for (int j=0; j<nz; j++) {
      for (int i=0; i<nk; i++) {
        za[j*nk + i] = cosmology.lnP[i][j]; // GSL layout: x runs fastest
      }
    }
    fs->lnP = gsl_spline2d_alloc(gsl_interp2d_bicubic, nk, nz);
    gsl_spline2d_init(fs->lnP, xa, ya, za, nk, nz);
    fs->lkmin = xa[0];
    fs->lkmax = xa[nk-1];
    fs->zPmin = ya[0];
    fs->zPmax = ya[nz-1];
    free(xa);
    free(ya);
    free(za);
  }
  // -------------------------------------------------------------------------
  // Lensing-efficiency response on g_tomo's own grids. With
  //   g(a) = P(a) - chi(a) Q(a),  Q(a) = int_{amin}^{a} n/(chi a'^2) da',
  // only Q depends on the cosmology besides chi, and
  //   dg/dX = -chi_X Q + chi R,  R(a) = int_{amin}^{a} n chi_X/(chi^2 a'^2) da'
  // — the same factored cumulative trapezoid g_tomo runs, one more integral.
  // The coarse table and its linear interpolation mirror g_tomo, so gX is
  // the derivative of the g the likelihood uses.
  // -------------------------------------------------------------------------
  {
    const int nbin = redshift.shear_nbin;
    const int x  = 60*(1 + abs(Ntable.high_def_integration));
    const int Na = x*(Ntable.N_a - 1) + 1;
    const double amin = 1.0/(redshift.shear_zdist_zmax_all + 1.0);
    const double amax = 0.999999;
    const double da = (amax - amin)/((double) Na - 1.0);
    fs->ga_min = amin;
    fs->ga_max = amax;
    fs->ga_dx  = (amax - amin)/((double) Ntable.N_a - 1.0);
    fs->gX = (double**) malloc2d(fs->np*nbin, Ntable.N_a);
    double*** Qint = (double***) malloc3d(fs->np + 1, nbin, Na);
    (void) nz_source_photoz(0.0, 0); // init static variables
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b=0; b<nbin; b++) {
      for (int i=0; i<Na; i++) {
        const double a = amin + i*da;
        const double z = 1.0/a - 1.0;
        const double c = chi(a);
        const double Pint = nz_source_photoz(z, b)/(a*a);
        Qint[0][b][i] = Pint/c;
        for (int ip=0; ip<fs->np; ip++) {
          const double chiX =
            gsl_spline_eval(fs->chiX[ip], z, NULL)/cosmology.coverH0;
          Qint[1 + ip][b][i] = Pint*chiX/(c*c);
        }
      }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ip=0; ip<fs->np; ip++) {
      for (int b=0; b<nbin; b++) {
        double* restrict gx = fs->gX[ip*nbin + b];
        const double* restrict qi = Qint[0][b];
        const double* restrict ri = Qint[1 + ip][b];
        double Q = 0.0;
        double R = 0.0;
        gx[0] = 0.0;
        for (int i=1; i<Na; i++) {
          Q += 0.5*da*(qi[i-1] + qi[i]);
          R += 0.5*da*(ri[i-1] + ri[i]);
          if (i % x == 0) {
            const double a = amin + i*da;
            const double chiX = gsl_spline_eval(fs->chiX[ip], 1.0/a - 1.0,
                                                NULL)/cosmology.coverH0;
            gx[i/x] = -chiX*Q + chi(a)*R;
          }
        }
      }
    }
    free(Qint);
  }
  // -------------------------------------------------------------------------
  // the ss Gauss-Legendre rule and limits of C_ss_tomo_limber_work
  // -------------------------------------------------------------------------
  {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 :
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    gsl_integration_glfixed_table* w = malloc_gslint_glfixed(szint);
    const double amin = 1./(redshift.shear_zdist_zmax_all+1.);
    const double amax = 1./(1.+fmax(redshift.shear_zdist_zmin_all,1e-6));
    fs->npts = (int) w->n;
    fs->a  = (double*) malloc1d(fs->npts);
    fs->wt = (double*) malloc1d(fs->npts);
    for (int p=0; p<fs->npts; p++) {
      gsl_integration_glfixed_point(amin, amax, p, &fs->a[p], &fs->wt[p], w);
    }
    gsl_integration_glfixed_table_free(w);
  }
}

static void fisher_state_free(
    fisher_state* fs  // state built by fisher_state_init
  )
{
  for (int ip=0; ip<fs->np; ip++) {
    gsl_spline_free(fs->chiX[ip]);
    gsl_spline_free(fs->lnGX[ip]);
  }
  gsl_spline2d_free(fs->lnP);
  free(fs->gX);
  free(fs->a);
  free(fs->wt);
}

// d g_tomo(a, b)/dX: the coarse-table interpolation g_tomo itself uses
static inline double fisher_dg(
    const fisher_state* fs,  // prepared state
    const int ip,            // parameter slot
    const int b,             // source bin
    const double a           // scale factor
  )
{
  if (a <= fs->ga_min || a > 1.0 - fs->ga_dx) {
    return 0.0;
  }
  return interpol1d(fs->gX[ip*redshift.shear_nbin + b], Ntable.N_a,
                    fs->ga_min, fs->ga_max, fs->ga_dx, a);
}

// ---------------------------------------------------------------------------
// Core: C_ss and dC_ss/dX for every loaded parameter (NLA, E-mode).
//
// Differentiating the Limber integral at fixed scale factor (the limits
// are fixed by the source n(z) support in redshift), with the NLA kernel
// factorized per bin, q_b = W_kappa,b - W_source,b C1:
//
//   C(l)    = Pi(l) sum_p wt_p w_p P_p q_i q_j
//   dC/dX   = Pi(l) sum_p wt_p w_p P_p [ Lam_X q_i q_j
//                                        + dq_i q_j + q_i dq_j ]
//
// with the measure w = (dchi/da)/chi^2, Pi(l) = (l-1)l(l+1)(l+2)/(l+1/2)^4,
// chi_z = dchi/dz, and per node
//
//   Lam_X  = dln w/dX + dlnP/dX
//          = chi_X'/chi_z - 2 chi_X/chi       (measure)
//            - n_eff chi_X/chi + R_X(k, z)    (P_NL at k = (l+1/2)/chi)
//   dW_kappa = W_kappa (eps_X + chi_X/chi) + 1.5 Omega_m (chi/a) dg/dX
//   dW_source = -W_source chi_X'/chi_z       (W_source ~ H/H0 = 1/chi_z)
//   dC1   = C1 (eps_X - dlnD/dX)             (C1 ~ Omega_m/D)
//
// where chi_X' is the cubic-spline derivative of the dchi/dX input,
// n_eff = dlnP_NL/dlnk the bicubic-spline slope, R_X the dlnP_NL/dX input
// and eps_X = dlnOmega_m/dX. Per node the brackets are precomputed; the
// (l, pair) fill is then a branch-free SIMD reduction, as in
// C_ss_tomo_limber_work, computing C and all derivatives in one pass.
//
// Memory layout:
//   Q[shear_nbin][npts]           NLA kernels q_b
//   DQ[np][shear_nbin][npts]      dq_b/dX
//   AMP[nell][npts]               wt * (dchi/da)/chi^2 * P_NL(k_l, a)
//   LAM[np][nell][npts]           Lam_X
// ---------------------------------------------------------------------------
static void dC_ss_dX_tomo_limber_work(
    const fisher_state* fs,  // prepared splines, responses and nodes
    const double* lx,        // multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of tomo shear power spectra
    double*** table          // output [1 + np][NSIZE][nell]
  )
{
  const int np   = fs->np;
  const int npts = fs->npts;
  const int nbin = redshift.shear_nbin;
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const double a = fs->a[0];
    struct chis chidchi = chi_all(a);
    const double hoh0 = hoverh0v2(a, chidchi.dchida);
    const double gf = growfac(a);
    (void) W_kappa(a, chidchi.chi, 0);
    (void) W_source(a, 0, hoh0);
    (void) IA_A1_Z1(a, gf, 0);
    (void) Pdelta((lx[0] + 0.5)/chidchi.chi, a);
    (void) Z1(0);
    (void) Z2(0);
  }

  double** Q     = (double**)  malloc2d(nbin, npts);
  double*** DQ   = (double***) malloc3d(np, nbin, npts);
  double** DLNW  = (double**)  malloc2d(np, npts); // dln w/dX
  double** CHIX  = (double**)  malloc2d(np, npts); // chi_X/chi
  double* AMPa   = (double*)   malloc1d(npts);     // wt (dchi/da)/chi^2
  double* CHI    = (double*)   malloc1d(npts);
  double** AMP   = (double**)  malloc2d(nell, npts);
  double*** LAM  = (double***) malloc3d(np, nell, npts);

  // -----------------------------------------------------------------------
  // Precompute per quadrature node: kernels and their responses
  // -----------------------------------------------------------------------
  #pragma omp parallel for schedule(static)
  for (int p=0; p<npts; p++) {
    const double a = fs->a[p];
    const double z = 1.0/a - 1.0;
    struct chis chidchi = chi_all(a);
    const double fK     = chidchi.chi;
    const double dchida = chidchi.dchida;
    const double hoh0   = hoverh0v2(a, dchida);
    const double gf     = growfac(a);
    const double chiz   = dchida*a*a;              // dchi/dz (c/H0 units)
    CHI[p]  = fK;
    AMPa[p] = fs->wt[p]*dchida/(fK*fK);
    double WK[nbin];
    double WS[nbin];
    double C1[nbin];
    for (int b=0; b<nbin; b++) {
      WK[b] = W_kappa(a, fK, b);
      WS[b] = W_source(a, b, hoh0);
      C1[b] = IA_A1_Z1(a, gf, b);
      Q[b][p] = WK[b] - WS[b]*C1[b];
    }
    for (int ip=0; ip<np; ip++) {
      const double eps  = fr.dlnOm[ip];
      const double chiX = gsl_spline_eval(fs->chiX[ip], z, NULL)/
                          cosmology.coverH0;
      const double chiXz = gsl_spline_eval_deriv(fs->chiX[ip], z, NULL)/
                           cosmology.coverH0;
      const double dlnchiz = chiXz/chiz;          // dln(dchi/dz)/dX
      const double dlnD = gsl_spline_eval(fs->lnGX[ip], z, NULL) - fs->gam0[ip];
      CHIX[ip][p] = chiX/fK;
      DLNW[ip][p] = dlnchiz - 2.0*chiX/fK;
      for (int b=0; b<nbin; b++) {
        const double dWK = WK[b]*(eps + chiX/fK)
                         + 1.5*cosmology.Omega_m*(fK/a)*fisher_dg(fs, ip, b, a);
        const double dWS = -WS[b]*dlnchiz;
        const double dC1 = C1[b]*(eps - dlnD);
        DQ[ip][b][p] = dWK - dWS*C1[b] - WS[b]*dC1;
      }
    }
  }
  // -----------------------------------------------------------------------
  // Precompute per (ell, node): P_NL at k = (l + 1/2)/chi, its slope n_eff
  // (bicubic spline; queries outside the table are clamped to its edge,
  // where the slope continues as p_nonlin's linear extrapolation does),
  // and the full bracket Lam_X for every parameter
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<nell; i++) {
    for (int p=0; p<npts; p++) {
      const double a  = fs->a[p];
      const double z  = 1.0/a - 1.0;
      const double k  = (lx[i] + 0.5)/CHI[p];
      const double lk = log10(k/cosmology.coverH0);  // log10(k [h/Mpc])
      AMP[i][p] = AMPa[p]*Pdelta(k, a);
      const double lkc = fmin(fmax(lk, fs->lkmin), fs->lkmax);
      const double zc  = fmin(fmax(z, fs->zPmin), fs->zPmax);
      const double neff =
        gsl_spline2d_eval_deriv_x(fs->lnP, lkc, zc, NULL, NULL)/M_LN10;
      for (int ip=0; ip<np; ip++) {
        LAM[ip][i][p] = DLNW[ip][p] - neff*CHIX[ip][p] + fisher_dlnP(ip, lk, z);
      }
    }
  }
  // -----------------------------------------------------------------------
  // Main fill loop: C and every dC/dX as SIMD reductions over the nodes.
  // Ell prefactor: l*(l-1)*(l+1)*(l+2)/(l+0.5)^4 (1812.05995 eqs 74-79)
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i=0; i<nell; i++) {
    for (int k=0; k<NSIZE; k++) {
      const int Z1NZ = Z1(k);
      const int Z2NZ = Z2(k);
      // Local restrict pointers: pointer-to-pointer rows inside a
      // collapse(2) region would otherwise block the SIMD reductions
      const double* restrict amp = AMP[i];
      const double* restrict q1  = Q[Z1NZ];
      const double* restrict q2  = Q[Z2NZ];
      const double l = lx[i];
      const double ell = l + 0.5;
      const double ell_pf = l*(l-1.)*(l+1.)*(l+2.)/(ell*ell*ell*ell);
      double sC = 0.0;
      #pragma omp simd reduction(+:sC)
      for (int p=0; p<npts; p++) {
        sC += amp[p]*q1[p]*q2[p];
      }
      table[0][k][i] = sC*ell_pf;
      for (int ip=0; ip<np; ip++) {
        const double* restrict lam = LAM[ip][i];
        const double* restrict dq1 = DQ[ip][Z1NZ];
        const double* restrict dq2 = DQ[ip][Z2NZ];
        double sD = 0.0;
        #pragma omp simd reduction(+:sD)
        for (int p=0; p<npts; p++) {
          sD += amp[p]*(lam[p]*q1[p]*q2[p] + dq1[p]*q2[p] + q1[p]*dq2[p]);
        }
        table[1 + ip][k][i] = sD*ell_pf;
      }
    }
  }
  free(Q);
  free(DQ);
  free(DLNW);
  free(CHIX);
  free(AMPa);
  free(CHI);
  free(AMP);
  free(LAM);
}

// ---------------------------------------------------------------------------
// Batch C_ss and dC_ss/dX at arbitrary multipoles (one state build, one
// work call).
// ---------------------------------------------------------------------------
void dC_ss_dX_tomo_limber_nointerp_ells(
    const double* ells,  // multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of tomo shear power spectra
    double*** out        // output [1 + nparam][NSIZE][nell]
  )
{
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell); exit(1);
  }
  fisher_state fs;
  fisher_state_init(&fs);
  dC_ss_dX_tomo_limber_work(&fs, ells, nell, NSIZE, out);
  fisher_state_free(&fs);
}

// ---------------------------------------------------------------------------
// xi_pm and dxi_pm/dX for every loaded parameter.
//
//   xi_pm(theta)    = sum_l Glpm(theta, l) C_EE(l)
//   dxi_pm/dX       = sum_l Glpm(theta, l) dC_EE/dX(l)
//
// (NLA: C_BB = 0). The Legendre kernels do not depend on the cosmology, so
// the map from dC to dxi is linear and exact. Same pipeline as xi_pm_tomo:
//   1. exact multipoles l = 1 .. LMIN_tab-1 (one work call);
//   2. the log-ell table of C_ss_tomo_limber (same limits and N_ell), one
//      work call for C and every dC/dX together, gathered onto every
//      integer multipole l = LMIN_tab .. LMAX-1 by limber_fill_interp;
//   3. SIMD Legendre sums against the bin-averaged Glpm kernels.
//
// Static state, rebuilt when Ntable changes:
//   Glpm[2][Ntheta][LMAX] - bin-averaged Legendre kernels (Gl+ and Gl-)
//   ln_ell[LMAX]          - log(l) at every integer multipole
// ---------------------------------------------------------------------------
void dxi_pm_dX_tomo(
    double**** out   // output [1 + nparam][2][NSIZE][Ntheta]
  )
{
  static double*** Glpm = NULL; // Glpm[0] = Gl+, Glpm[1] = Gl-
  static double* ln_ell = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  const int lmin = 1;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  if (NULL == Glpm || fdiff2(cache[0], Ntable.random))
  {
    if (Glpm != NULL) free(Glpm);
    Glpm = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX);

    double*** P = (double***) malloc3d(4, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    double** dPmin = P[2]; double** dPmax = P[3];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i, 0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r   = set_bin_average(i, l);
        Pmin[i][l]  = r.Pmin;
        Pmax[i][l]  = r.Pmax;
        dPmin[i][l] = r.dPmin;
        dPmax[i][l] = r.dPmax;
      }
    }
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Glpm[0][i][l] = 0.0;
        Glpm[1][i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        Glpm[0][i][l] = (2.*l+1)/(2.*M_PI*l*l*(l+1)*(l+1))*(
          -l*(l-1.)/2*(l+2./(2*l+1)) * (Pmin[i][l-1]-Pmax[i][l-1])
          -l*(l-1.)*(2.-l)/2 * (xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          +l*(l-1.)/(2.*l+1) * (Pmin[i][l+1]-Pmax[i][l+1])
          +(4-l)*(dPmin[i][l]-dPmax[i][l])
          +(l+2)*(xmin[i]*dPmin[i][l-1] - xmax[i]*dPmax[i][l-1] - Pmin[i][l-1] + Pmax[i][l-1])
          +2*(l-1)*(xmin[i]*dPmin[i][l] - xmax[i]*dPmax[i][l] - Pmin[i][l] + Pmax[i][l])
          -2*(l+2)*(dPmin[i][l-1]-dPmax[i][l-1])
        )/(xmin[i]-xmax[i]);

        Glpm[1][i][l] = (2.*l+1)/(2.*M_PI*l*l*(l+1)*(l+1))*(
          -l*(l-1.)/2*(l+2./(2*l+1)) * (Pmin[i][l-1]-Pmax[i][l-1])
          -l*(l-1.)*(2.-l)/2 * (xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          +l*(l-1.)/(2.*l+1)* (Pmin[i][l+1]-Pmax[i][l+1])
          +(4-l)*(dPmin[i][l]-dPmax[i][l])
          +(l+2)*(xmin[i]*dPmin[i][l-1] - xmax[i]*dPmax[i][l-1] - Pmin[i][l-1] + Pmax[i][l-1])
          -2*(l-1)*(xmin[i]*dPmin[i][l] - xmax[i]*dPmax[i][l] - Pmin[i][l] + Pmax[i][l])
          +2*(l+2)*(dPmin[i][l-1]-dPmax[i][l-1])
          )/(xmin[i]-xmax[i]);
      }
    }
    free(P);

    if (ln_ell != NULL) free(ln_ell);
    ln_ell = (double*) malloc1d(Ntable.LMAX);
    ln_ell[0] = 0.0; // unused (the sums start at lmin = 1)
    for (int l=1; l<Ntable.LMAX; l++) {
      ln_ell[l] = log((double) l);
    }
    cache[0] = Ntable.random;
  }

  fisher_state fs;
  fisher_state_init(&fs);
  const int nm = 1 + fs.np; // C plus one derivative per parameter

  double*** Cl = (double***) malloc3d(nm, NSIZE, Ntable.LMAX);
  zero3d(Cl, nm, NSIZE, Ntable.LMAX);
  // -----------------------------------------------------------------------
  // 1. exact multipoles lmin .. LMIN_tab-1
  // -----------------------------------------------------------------------
  const int nlow = limits.LMIN_tab - lmin;
  if (nlow > 0) {
    double* lx = (double*) malloc1d(nlow);
    for (int i=0; i<nlow; i++) {
      lx[i] = (double) (lmin + i);
    }
    double*** tab = (double***) malloc3d(nm, NSIZE, nlow);
    dC_ss_dX_tomo_limber_work(&fs, lx, nlow, NSIZE, tab);
    for (int m=0; m<nm; m++) {
      for (int nz=0; nz<NSIZE; nz++) {
        for (int i=0; i<nlow; i++) {
          Cl[m][nz][lmin + i] = tab[m][nz][i];
        }
      }
    }
    free(tab);
    free(lx);
  }
  // -----------------------------------------------------------------------
  // 2. log-ell table (the C_ss_tomo_limber grid) gathered to every integer
  //    multipole LMIN_tab .. LMAX-1
  // -----------------------------------------------------------------------
  {
    const int nell = Ntable.N_ell;
    const double lim0 = log(fmax(limits.LMIN_tab - 1., 1.0));
    const double lim1 = log(Ntable.LMAX + 1.);
    const double dlx  = (lim1 - lim0)/((double) nell - 1.);
    double* lx = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim0 + i*dlx);
    }
    double*** tab = (double***) malloc3d(nm, NSIZE, nell);
    dC_ss_dX_tomo_limber_work(&fs, lx, nell, NSIZE, tab);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m=0; m<nm; m++) {
      for (int nz=0; nz<NSIZE; nz++) {
        const double* tab1[1] = {tab[m][nz]};
        double* dst[1] = {Cl[m][nz]};
        limber_fill_interp(1, tab1, dst, limits.LMIN_tab, Ntable.LMAX,
                           ln_ell, lim0, 1.0/dlx, nell);
      }
    }
    free(tab);
    free(lx);
  }
  fisher_state_free(&fs);
  // -----------------------------------------------------------------------
  // 3. Legendre sums (NLA: C_BB = 0, so xi+ and xi- read the same row)
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(3) schedule(static)
  for (int m=0; m<nm; m++) {
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const double* restrict c  = Cl[m][nz];
        const double* restrict gp = Glpm[0][i];
        const double* restrict gm = Glpm[1][i];
        double sum0 = 0.0;
        double sum1 = 0.0;
        #pragma omp simd reduction(+:sum0, sum1)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum0 += gp[l]*c[l];
          sum1 += gm[l]*c[l];
        }
        out[m][0][nz][i] = sum0;
        out[m][1][nz][i] = sum1;
      }
    }
  }
  free(Cl);
}
