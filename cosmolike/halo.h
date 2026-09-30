#ifndef __COSMOLIKE_HALO_H
#define __COSMOLIKE_HALO_H
#ifdef __cplusplus
extern "C" {
#endif

// HALO BIAS OPTIONS ---------------------------
#define HALO_BIAS_TINKER_2010 0

// HMF OPTIONS ---------------------------------
#define HMF_TINKER_2010 0

// CONCENTRATION OPTIONS -----------------------
#define CONCENTRATION_BHATTACHARYA_2013 0

// HALO PROFILE OPTIONS OPTIONS -----------------------
#define HALO_PROFILE_NFW 0

// DENSITY FIELD OF THE PEAK HEIGHT (like.halo_model[4]) ---------------
// The field whose linear variance sigma^2(M) sets nu and whose mean
// density sets the Lagrangian radius and the rho/M of dn/dlnM:
//   HALO_FIELD_MATTER = total matter (P_lin, rho_crit Omega_m)
//   HALO_FIELD_CB     = cold dark matter + baryons (P_cb,
//                       rho_crit (Omega_m - Omega_nu)): halos do not
//                       collect free-streaming massive neutrinos
//                       (DES Y1 clusters, 2010.01138)
// Everything else (r_Delta, M/rho_m of the matter window, the lensing
// kernels, the 2-halo spectrum) stays total matter under either.
#define HALO_FIELD_MATTER 0
#define HALO_FIELD_CB 1

double hb1nu(const double nu, const double a);

double fnu(const double nu, const double a);

double conc(const double m, const double growfac_a);

double bias_norm(const double a);

double dlognudlogm(const double M);

double u_nfw_c(const double c, const double k, const double m, const double a);

double u_KS(double c, double k, const double rv);

double ngal(const int ni, const double a);

double bgal(const int ni, const double a);

double p_mm(const double k, const double a);

double p_gm(const double k, const double a, const int ni);

double p_gg(const double k, const double a, const int ni, const int nj);

double p_my(const double k, const double a);

double p_yy(const double k, const double a);

void set_HOD(const int ni);

// halo-model intrinsic alignment (Fortuna et al. 2021), one IA
// population over the source redshift range; k in (c/H0)^-1
double ia_f_red_central(const double a);

double ia_p1h_dI(const double k, const double a);

double ia_p1h_II(const double k, const double a);

double ia_window_2h(const double k);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
