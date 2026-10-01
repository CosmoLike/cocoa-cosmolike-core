#include <stdio.h>
#include "basics.h"
#include "structs.h"
#include "halo.h"

#include "log.c/src/log.h"

FPT FPTIA;
FPT FPTbias;
cosmopara cosmology;
tomopara tomo;
redshiftparams redshift;

nuisanceparams nuisance;
likepara like;
sur survey;

pdeltapara pdeltaparams =
{
  .runmode = "Halofit"
};

CMBparams cmb =
{
  .random = 0.0,
  .fwhm = 0.0,
  .healpixwin_ncls = 0,
  .healpixwin = NULL,
  .lk_wxk = {0, 0},
  .alpha_Hartlap_cov_kkkk = 1.0,
  .nbp_kk = 0,
  .lbp_kk = {0, 0},
  .binning_matrix_kk = NULL,
  .theory_offset_kk = NULL
};

lim limits = 
{
  .a_min = 1.0/(1.0 + 40.0),    // a_min (z = 40, needed for CMB lensing)
  .k_cH0 = {2.e-2, 3.e+6},     // k range in units of H0/c
  .LMIN_tab = 20,               // LMIN_tab
  .LMAX_NOLIMBER = 150,         // LMAX_NOLIMBER
  .halo_m = {1.0e+6, 1.0e+17},  // halo.c mass range (M_sun/h)
  .halo_uks_c = {0.05, 100.0}   // halo.c u_KS concentration range (queries
                                //   outside are clamped to it)
};

Ntab Ntable;

//  ----------------------------------------------------------------------------------
//  ----------------------------------------------------------------------------------
//  ----------------------------------------------------------------------------------
//  RESET STRUCT 
//  ----------------------------------------------------------------------------------
//  ----------------------------------------------------------------------------------
//  ----------------------------------------------------------------------------------

void reset_like_struct(void)
{
  like.bias = 0;
  like.Ncl = 0;
  like.Ncos = 0;
  like.Ndata = 0;
  like.lmin = 0;
  like.lmax = 0;
  if((like.ell != NULL) == 1)
  {
    free(like.ell);
    like.ell = NULL;
  }
  like.cosmax = 0;
  like.Rmin_bias = 0;
  like.Rmin_shear = 0;
  like.lmax_shear = 0;
  
  like.bias = 0;
  for (int i=0; i<NPROBES; i++) {
    like.probe[i] = 0;
  }
  like.adopt_limber[LIMBER_GG] = 0;
  like.adopt_limber[LIMBER_GS] = 1;
  // halo.c model choices (halo.h macros, all 0): HMF_TINKER_2010,
  // HALO_BIAS_TINKER_2010, CONCENTRATION_BHATTACHARYA_2013,
  // HALO_PROFILE_NFW, HALO_FIELD_MATTER - set explicitly so the
  // defaults are deliberate
  like.halo_model[0] = 0;
  like.halo_model[1] = 0;
  like.halo_model[2] = 0;
  like.halo_model[3] = 0;
  like.halo_model[4] = 0;
}

void reset_cosmology_struct(void)
{
  cosmology.coverH0 = 2997.92458;
  cosmology.rho_crit = 7.4775e+21;
  cosmology.MGSigma = 0.0;
  cosmology.MGmu = 0.0;
  cosmology.Omega_b = 0.0;
  cosmology.Omega_m = 0.0;
  cosmology.Omega_v = 0.0;
  cosmology.h0 = 0.0;
  cosmology.Omega_nu = 0.0;
  cosmology.random = 0.0;
  cosmology.sigma_8 = 0.0;
  cosmology.lnP_nk = 0;
  cosmology.lnP_nz = 0;
  cosmology.lnP = NULL;
  cosmology.lnPL_nk = 0;
  cosmology.lnPL_nz = 0;
  cosmology.lnPL = NULL;
  // the P_cb table belongs to the lnPL table it was installed with
  // (structs.h), so it goes with it; free(NULL) is a no-op on the
  // first call (the global struct starts zeroed)
  free(cosmology.lnPL_cb);
  cosmology.lnPL_cb = NULL;
  cosmology.chi_nz = 0;
  cosmology.chi = NULL;
  // the bucket index belongs to the chi table (structs.h)
  free(cosmology.chi_bucket);
  cosmology.chi_bucket = NULL;
  cosmology.chi_nbucket = 0;
  cosmology.G_nz = 0;
  cosmology.G = NULL;
}

void reset_tomo_struct(void)
{
  tomo.shear_Npowerspectra = 0;
  tomo.clustering_Npowerspectra = 0;
  tomo.ggl_Npowerspectra = 0;
  if (tomo.ggl_exclude != NULL) {
    free(tomo.ggl_exclude);
    tomo.ggl_exclude = NULL;
  }
  tomo.N_ggl_exclude = 0;
  tomo.random_ggl = 0;
}

void reset_redshift_struct(void)
{
  redshift.random_shear = 0.0;
  redshift.random_clustering = 0.0;

  redshift.shear_nbin = 0;
  redshift.shear_photoz = 0;
  if (redshift.shear_zdist_table != NULL) {
    free(redshift.shear_zdist_table);
    redshift.shear_zdist_table = NULL;
  }
  redshift.shear_nzbins = 0;
  redshift.shear_zdist_zall[RANGE_MIN] = 0.0;
  redshift.shear_zdist_zall[RANGE_MAX] = 0.0;

  redshift.clustering_nbin = 0;
  redshift.clustering_nzbins = 0;
  if (redshift.clustering_zdist_table != NULL) {
    free(redshift.clustering_zdist_table);
    redshift.clustering_zdist_table = NULL;
  }
  redshift.clustering_photoz = 0;
  redshift.clustering_zdist_zall[RANGE_MIN] = 0.0;
  redshift.clustering_zdist_zall[RANGE_MAX] = 0.0;

  for (int i=0; i<MAX_SIZE_ARRAYS; i++) {
    redshift.shear_zdist_z[RANGE_MIN][i] = 0.0;
    redshift.shear_zdist_z[RANGE_MAX][i] = 0.0; 
    redshift.clustering_zdist_z[RANGE_MIN][i] = 0.0;
    redshift.clustering_zdist_z[RANGE_MAX][i] = 0.0;
    redshift.clustering_zdist_z[ZDIST_MEAN][i] = 0.0;
  }
}

void reset_survey_struct(void)
{
  survey.area = 0.0;
  survey.n_gal = 0.0;
  survey.sigma_e = 0.0;
  survey.area_conversion_factor =
    60.0 * 60.0 * 2.90888208665721580e-4 * 2.90888208665721580e-4;
  survey.n_gal_conversion_factor =
    1.0 / 2.90888208665721580e-4 / 2.90888208665721580e-4;
  survey.n_lens = 0.0;
  survey.m_lim = 0.0;
  sprintf(survey.name, "%s", "");
}

void reset_pdeltaparams_struct(void)
{
  sprintf(pdeltaparams.runmode, "%s", "Halofit");
}

void reset_nuisance_struct(void)
{
  nuisance.random_ia = 0.0;
  nuisance.random_ia_halo = 0.0;
  nuisance.random_photoz_shear = 0.0;
  nuisance.random_photoz_clustering = 0.0;
  for (int i=0; i<MAX_SIZE_ARRAYS; i++) {
    nuisance.shear_calibration_m[i] = 0.0;
    nuisance.gc[i] = 0.0;
    nuisance.gas[i] = 0.0;
    nuisance.ia_halo[i] = 0.0;
    nuisance.ia_red[i] = 0.0;
    nuisance.ia_hod[i] = 0.0;
    for (int j=0; j<MAX_SIZE_ARRAYS; j++) {
      nuisance.ia[i][j] = 0.0;
      nuisance.ia[i][j] = 0.0;
      nuisance.ia[i][j] = 0.0;
      nuisance.gb[i][j] = 0.0;
      nuisance.hod[i][j] = 0.0;
      for (int k=0; k<MAX_SIZE_ARRAYS; k++) {
        if (j==1) {
          nuisance.photoz[i][j][k] = 1.0;
        } // photo-z stretch params
        else {
          nuisance.photoz[i][j][k] = 0.0;
        }
      }
    }
  }
  nuisance.oneplusz0_ia = 0.0;
  nuisance.c1rhocrit_ia = 0.01389;
  nuisance.IA = 0;
  nuisance.IA_MODEL = 0;
  nuisance.IA_code = 0; // 0 = CFASTPT; 1 = PyFASTPT
}

void reset_cmb_struct(void)
{
  cmb.random = 0.0;
  cmb.fwhm = 0.0;
  cmb.healpixwin_ncls = 0;
  if (cmb.healpixwin != NULL) {
    free(cmb.healpixwin);
    cmb.healpixwin = NULL;
  }
  cmb.alpha_Hartlap_cov_kkkk = 1.0;
  cmb.nbp_kk = 0;
  cmb.lbp_kk[RANGE_MIN] = 0;
  cmb.lbp_kk[RANGE_MAX] = 0;
  if (cmb.theory_offset_kk != NULL) {
    free(cmb.theory_offset_kk);
    cmb.theory_offset_kk = NULL;
  }
  if (cmb.binning_matrix_kk != NULL) {
    free(cmb.binning_matrix_kk);
    cmb.binning_matrix_kk = NULL;
  }
}

void reset_Ntable_struct(void)
{ // Multiples of 8 (cache line; OpenMP Core usually set to 4/8; AVX2)
  Ntable.LMAX     = 100000;
  Ntable.random   = 0.0;
  Ntable.N_a      = 256;   // N_a       
  Ntable.N_k_lin  = 512;   // N_k_lin
  Ntable.N_k_nlin = 512;   // N_k_nlin
  Ntable.N_ell[NODES_DENSE]    = 512;   // N_ell
  Ntable.N_ell[NODES_COARSE] = 192; // ss/gs table coarse grid; 0 = exact N_ell      
  Ntable.Ntheta   = 256;   // N_theta (not used by cosmo2d) 
  Ntable.N_M[NODES_DENSE]      = 1024;  // N_M, M = mass (Halo Model)
  Ntable.N_M[NODES_COARSE] = 192; // coarse sigma^2(M) nodes (upsampled to N_M)
  Ntable.halo_uks_n[UKS_N_LNC] = 40;       // u_KS coarse ln c nodes (upsampled; halo.c)
  Ntable.halo_uks_n[UKS_N_LNZ] = 64;       // u_KS coarse ln z nodes (upsampled; halo.c)
  Ntable.halo_nfw_n = 131072; // u_nfw_c exact dense ln t nodes (halo.c)
  Ntable.halo_spline_pad = 6; // halo.c coarse spline padding nodes
  Ntable.halo_uks_m[UKS_M_LNC2D] = 12;     // u_KS dense refinement: ln c (2D)
  Ntable.halo_uks_m[UKS_M_W] = 32;         // u_KS dense refinement: w
  Ntable.halo_uks_m[UKS_M_LNZ] = 16;       // u_KS dense refinement: ln z
  Ntable.halo_uks_m[UKS_M_LNY] = 115;      // u_KS dense refinement: ln y
  Ntable.halo_uks_m[UKS_M_LNC1D] = 70;     // u_KS dense refinement: ln c (1D)
  Ntable.halo_hmf_n[NODES_COARSE][HMF_TINKER_2010] = 128;  // tinker_alpha exact aa
  Ntable.halo_hmf_n[NODES_DENSE][HMF_TINKER_2010] = 4096; // tinker_alpha dense aa
  Ntable.halo_nm = 64;       // spectra mass nodes at hdi 0: chi2 ladder in
                             // the skill file (floor: 64)
  Ntable.halo_nk_step = 4;   // p_gm/p_gg coarse ln k step at hdi 0
  Ntable.halo_na_lens = 51;  // p_gm/p_gg a nodes per lens bin
  Ntable.halo_ia_lmax = 6;   // halo IA multipoles: F21's l <= 6
  Ntable.halo_ia_na = 51;    // halo IA a nodes over the source range
  Ntable.NL_Nchi  = 512;   // Cosmo2D - NL = NonLimber (NL_Nchi)
  Ntable.high_def_integration = 0;
  Ntable.FPTboost=0;
  Ntable.dCX_dlnk_nlnk[NODES_DENSE] = 256;
  Ntable.dCX_dlnk_k[RANGE_MIN] = 1.e-5;
  Ntable.dCX_dlnk_k[RANGE_MAX] = 1.e2;
  Ntable.dCX_dlnk_nlnk[NODES_COARSE] = 128; // half the dCX grid: measured
  // response error <= what the retired fixed quadrature imposed
  // (max |dRF| 5.9e-3, medians ~1e-6) at 2x the refill; 0 = exact 
  Ntable.nz_fine_sampling_factor = 5; // nz fine-sampling (to ensure uniform points)
  Ntable.photoz_interpolation_type = 0; // 0: cspline, 1: linear, 2+: steffen
  Ntable.photoz_zmid_convention = 0;    // 0: z column = Z_LOW (left edges); 1: Z_MID (points)
  // C-FAST-PT convolution grid / output grid. 0.5 is converged: the
  // 2026-09-25 lsst_y1 scan measured delta^T C^-1 delta <= 1e-9 vs the
  // single-grid path down to 0.27, and 1.0 recovers that path exactly
  Ntable.FPT_internal_accuracy_boost = 0.5;
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
