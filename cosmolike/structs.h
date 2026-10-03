#include <stdint.h>
#include <gsl/gsl_interp2d.h>

#ifndef __COSMOLIKE_STRUCTS_H
#define __COSMOLIKE_STRUCTS_H
#ifdef __cplusplus
extern "C" {
#endif

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// Maximum number of piecewise-uniform segments tracked for grid metadata
#define MAX_GRID_SEGMENTS 10
#endif

#define CHAR_MAX_SIZE 1024
#define MAX_SIZE_ARRAYS 20

// Slots of every [min, max] range array (limits.halo_m[], limits.k_cH0[],
// Ntable.vt[], redshift.shear_zdist_z[][], cmb.lk_wxk[], ...)
#define RANGE_MIN 0
#define RANGE_MAX 1
// third slot of redshift.clustering_zdist_z[][]: the fiducial mean
#define ZDIST_MEAN 2

// Slots of every two-entry node-count array of Ntable whose table is
// computed exactly on coarse nodes and spline-upsampled to dense ones
// (Ntable.N_ell[], N_M[], dCX_dlnk_nlnk[], halo_hmf_n[][])
#define NODES_DENSE 0    // the nodes the table is read on
#define NODES_COARSE 1   // the exact-quadrature nodes; 0 = exact on
                         // every dense node

typedef struct 
{
  double a_min;
  double a_min_hm;
  double k_cH0[2];      // k range [RANGE_MIN, RANGE_MAX] in units of H0/c
  
  int LMIN_tab;
  int LMAX_NOLIMBER;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // HALO MODEL
  // ---------------------------------------------------
  // --------------------------------------------------- 
  double halo_m[2];       // halo.c mass range [RANGE_MIN, RANGE_MAX]
                          // (M_sun/h)
  double halo_uks_c[2];   // u_KS concentration range [RANGE_MIN,
                          // RANGE_MAX]; queries outside are clamped to it
                          // (u_KS: future_port_unfinished/halo_tsz.c, not compiled)
} lim;

// Slots of Ntable.halo_uks_n[]: coarse node counts of the u_KS tables
// (both scaled by init_accuracy_boost; u_KS is in
// future_port_unfinished/halo_tsz.c, not compiled)
#define UKS_N_LNC 0     // ln c axis
#define UKS_N_LNZ 1     // ln z axis
#define NUKS_N 2
// Slots of Ntable.halo_uks_m[]: dense refinement factors of the u_KS
// coarse -> dense splines (future_port_unfinished/halo_tsz.c)
#define UKS_M_LNC2D 0   // ln c axis of the 2D table
#define UKS_M_W 1       // w axis
#define UKS_M_LNZ 2     // ln z axis
#define UKS_M_LNY 3     // ln y axis
#define UKS_M_LNC1D 4   // ln c axis of the 1D table
#define NUKS_M 5

typedef struct 
{
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CACHE VARIABLES
  // ---------------------------------------------------
  // ---------------------------------------------------
  uint64_t random; 
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CONTROL NUM POINTS ON COSMOLIKE TABLES
  // ---------------------------------------------------
  // ---------------------------------------------------
  int LMAX;            // Cosmo2d: lmax used to compute 2D correlation function
  int N_a;          
  int N_k_lin;
  int N_k_nlin;
  int N_ell[2];     // [NODES_DENSE] ell nodes of the C_l tables;
                    // [NODES_COARSE] coarse exact-quadrature ell nodes of
                    // the C_ss, C_gs, C_gk and C_ks tables, cubic-spline
                    // upsampled to the dense ones; 0 = exact (C_gg always
                    // exact: BAO wiggles)
  int Ntheta;
  int N_M[2];       // [NODES_DENSE] ln M nodes of the sigma^2(M) halo-
                    // model table; [NODES_COARSE] its coarse exact nodes,
                    // cubic-spline upsampled to the dense ones in
                    // ln sigma^2; 0 = exact
  int NL_Nell_block;   // Cosmo2D - NL = NonLimber
  int NL_Nchi;         // Cosmo2D - NL = NonLimber
  double NL_Nchi_boost; // NL_Nchi multiplier on top of the accuracy boost
                        // (init_nonlimber_accuracy_boost; 1 = none)
  // ---------------------------------------------------
  // ---------------------------------------------------
  // THETA RANGE ON REAL SPACE CORRELATION FUNCTIONS
  // ---------------------------------------------------
  // ---------------------------------------------------
  double vt[2];     // theta range [RANGE_MIN, RANGE_MAX] (radians)
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CONTROL NUM POINTS EVALUATED ON COSMOLIKE INTEGRALS
  // ---------------------------------------------------
  // ---------------------------------------------------
  int high_def_integration;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CONTROL NUM POINTS EVALUATED ON LIMBER DERIVATIVES
  // ---------------------------------------------------
  // ---------------------------------------------------
  int dCX_dlnk_nlnk[2]; // [NODES_DENSE] ln k nodes of the dC_X/dlnk
                        // scale-cut tables; [NODES_COARSE] their coarse
                        // exact ln k nodes (the ell axis pairs with
                        // N_ell[NODES_COARSE]; bicubic upsampled); 0 =
                        // exact (ln k carries the BAO wiggles)
  double dCX_dlnk_k[2];  // k range [RANGE_MIN, RANGE_MAX] of the dC_X/dlnk tables
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CONTROL NUM POINTS EVALUATED BY FASPT
  // ---------------------------------------------------
  // ---------------------------------------------------
  int FPTboost;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // HALO MODEL
  // ---------------------------------------------------
  // ---------------------------------------------------  
  int halo_uks_n[NUKS_N];  // u_KS coarse nodes (not compiled; boosted); slots
                           // UKS_N_* (above Ntab)
  int halo_nfw_n;   // u_nfw_c dense ln t nodes (halo.c; boosted)
  int halo_spline_pad; // exact coarse nodes beyond each end of every
                       // halo.c coarse -> dense spline (tinker_alpha,
                       // coarse ln k of p_gm/p_gg; the u_KS axes)
  int halo_uks_m[NUKS_M];  // u_KS dense refinement factors; slots UKS_M_*
  // mass-function table sizes, one entry per like.halo_model[0] option
  // (HMF_TINKER_2010: the tinker_alpha normalization table)
  int halo_hmf_n[2][MAX_SIZE_ARRAYS]; // [NODES_COARSE] exact aa nodes on
                                      // [0.25, 1]; [NODES_DENSE] dense aa
                                      // lookup nodes
  int halo_nm;      // spectra GL mass nodes at high_def_integration 0
                    // (doubled per rung; not boosted)
  int halo_nk_step; // p_gm/p_gg coarse ln k step at high_def_integration
                    // 0 (halved per rung, down to exact; not boosted)
  int halo_na_lens; // p_gm/p_gg a nodes per lens bin (boosted)
  int halo_ia_lmax; // halo-model IA: highest multipole (2, 4 or 6)
  int halo_ia_na;   // halo-model IA tables: a nodes over the source range
                    // (boosted)
  // ---------------------------------------------------
  // ---------------------------------------------------
  // n(z) fin-sampling
  // ---------------------------------------------------
  // --------------------------------------------------- 
  int nz_fine_sampling_factor;
  int photoz_interpolation_type; // 0: cspline, 1: linear, 2+: steffen (see basics.c: malloc_gsl_interp)
  int photoz_zmid_convention;    // 0: n(z) z column = Z_LOW (left bin edges); 1: Z_MID (sample points)
  double FPT_internal_accuracy_boost; // C-FAST-PT convolution grid / output grid (pt_cfastpt.c)
} Ntab;

typedef struct
{
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CACHE VARIABLES
  // ---------------------------------------------------
  // ---------------------------------------------------
  uint64_t random;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // COSMO PARAMETERS
  // ---------------------------------------------------
  // ---------------------------------------------------
  double Omega_b;  // baryon density paramter
  double Omega_m;  // matter density parameter
  double Omega_v;  // cosmogical constant parameter
  double h0;       // Hubble constant
  double Omega_nu; // massive neutrinos today, omega_nu h^2/h^2; part of
                   // Omega_m = Omega_cdm + Omega_b + Omega_nu
  double coverH0;  // units for comoving distances - speeds up code
  double rho_crit; // = 3 H_0^2/(8 pi G), critical comoving density
  double MGSigma;
  double MGmu;
  double sigma_8;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // MATTER POWER SPECTRUM
  // size = (lnP_nk+1,lnP_nz+1)
  // z = lnP[lnP_nk,j<lnP_nz]
  // k = lnP[i<lnP_nk,lnP_nz]
  // ---------------------------------------------------
  // ---------------------------------------------------
  int lnP_nk;
  int lnP_nz;
  double** lnP;
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
  // Direct-index lookup metadata
  // log10k axis is required to be a single uniform segment.
  // z axis may be piecewise-uniform with up to MAX_GRID_SEGMENTS segments.
  double  lnP_log10k_min;
  double  lnP_log10k_inv_dx;
  int     lnP_z_nseg;
  int     lnP_z_seg_start[MAX_GRID_SEGMENTS];
  int     lnP_z_seg_len[MAX_GRID_SEGMENTS];
  double  lnP_z_seg_xmin[MAX_GRID_SEGMENTS];
  double  lnP_z_seg_inv_dx[MAX_GRID_SEGMENTS];
#endif
  // ---------------------------------------------------
  // ---------------------------------------------------
  // LINEAR MATTER POWER SPECTRUM
  // size = (lnP_nk+1,lnP_nz+1)
  // z = lnPL[lnP_nk,j<lnP_nz]
  // k = lnPL[i<lnP_nk,lnP_nz]
  // ---------------------------------------------------
  // ---------------------------------------------------
  int lnPL_nk;
  int lnPL_nz;
  double** lnPL;
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM 
  // Direct-index lookup metadata.
  // log10k axis is required to be a single uniform segment.
  // z axis may be piecewise-uniform with up to MAX_GRID_SEGMENTS segments.
  double  lnPL_log10k_min;
  double  lnPL_log10k_inv_dx;
  int     lnPL_z_nseg;
  int     lnPL_z_seg_start [MAX_GRID_SEGMENTS];
  int     lnPL_z_seg_len   [MAX_GRID_SEGMENTS];
  double  lnPL_z_seg_xmin  [MAX_GRID_SEGMENTS];
  double  lnPL_z_seg_inv_dx[MAX_GRID_SEGMENTS];
#endif
  // ---------------------------------------------------
  // ---------------------------------------------------
  // LINEAR CDM + BARYON POWER SPECTRUM P_cb (the matter
  // without the massive neutrinos; read when
  // like.halo_model[4] = HALO_FIELD_CB)
  // size = (lnPL_nk, lnPL_nz), values only:
  // lnPL_cb[i][j] = ln P_cb at (log10k_i, z_j) of lnPL,
  // whose axes and direct-index metadata p_lin_cb reads.
  // NULL = not installed. A new lnPL table drops it
  // (set_linear_power_spectrum): the two are installed
  // as a pair.
  // ---------------------------------------------------
  // ---------------------------------------------------
  double** lnPL_cb;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // DISTANCE chi(a)
  // ---------------------------------------------------
  // ---------------------------------------------------
  // z   = chi[0,j<chi_nz]
  // chi = chi[1,j<chi_nz]
  int chi_nz;
  double** chi;
  // Bucket index of the chi column for a_chi (built by
  // set_chi_bucket_index, which set_distances calls): the range
  // [chi[1][0], chi[1][chi_nz-1]] cut into chi_nbucket equal buckets;
  // chi_bucket[b] is the bracket of bucket b's lower edge.
  int     chi_nbucket;
  int*    chi_bucket;
  double  chi_bucket_min;
  double  chi_bucket_inv_dx;
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
  // Direct-index lookup metadata for the z axis (chi[0]).
  // z axis may be piecewise-uniform with up to MAX_GRID_SEGMENTS segments.
  int     chi_z_nseg;
  int     chi_z_seg_start [MAX_GRID_SEGMENTS];
  int     chi_z_seg_len   [MAX_GRID_SEGMENTS];
  double  chi_z_seg_xmin  [MAX_GRID_SEGMENTS];
  double  chi_z_seg_inv_dx[MAX_GRID_SEGMENTS];
#endif
  // ---------------------------------------------------
  // ---------------------------------------------------
  // GROWTH FACTOR
  // ---------------------------------------------------
  // ---------------------------------------------------
  // z = G[0,j<chi_nz]
  // G = G[1,j<chi_nz]
  int G_nz;
  double** G;
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM 
  // Direct-index lookup metadata for the z axis (G[0]).
  // z axis may be piecewise-uniform with up to MAX_GRID_SEGMENTS segments.
  // Used by f_growth, growfac, norm_growfac, norm_growfac_all.
  int     G_z_nseg;
  int     G_z_seg_start [MAX_GRID_SEGMENTS];
  int     G_z_seg_len   [MAX_GRID_SEGMENTS];
  double  G_z_seg_xmin  [MAX_GRID_SEGMENTS];
  double  G_z_seg_inv_dx[MAX_GRID_SEGMENTS];
#endif
} cosmopara;

typedef struct
{ // parameters for power spectrum passed to FASTPT
  int N;      // output grid points (what the likelihood interpolates)
  int N_int;  // internal (convolution) grid points; == N: single grid
  double krange[2]; // k range [RANGE_MIN, RANGE_MAX] (units of H0/c)
  double k_cutoff;
  double sigma4;
  double** tab;     // output tables
  double** tab_int; // internal work tables; aliases tab when N_int == N
} FPT;

typedef struct
{
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CACHE VARIABLES
  // ---------------------------------------------------
  // ---------------------------------------------------
  uint64_t random_photoz_shear;
  uint64_t random_photoz_clustering;
  uint64_t random_ia;
  uint64_t random_galaxy_bias;
  uint64_t random_gas;      // gas parameters (u_KS, not compiled)
  uint64_t random_ia_halo;  // halo-model IA parameters (below)
  // ---------------------------------------------------
  // ---------------------------------------------------
  // INTRINSIC ALIGMENT --------------------------------
  // ---------------------------------------------------
  // --------------------------------------------------- 
  // ia[0][0] = A_ia          if(IA_NLA_LF || IA_REDSHIFT_EVOLUTION)
  // ia[0][1] = eta_ia        if(IA_NLA_LF || IA_REDSHIFT_EVOLUTION)
  // ia[0][2] = eta_ia_highz  if(IA_NLA_LF, Joachimi2012)
  // ia[0][3] = beta_ia       if(IA_NLA_LF, Joachimi2012)
  // ia[0][4] = LF_alpha      if(IA_NLA_LF, Joachimi2012)
  // ia[0][5] = LF_P          if(IA_NLA_LF, Joachimi2012)
  // ia[0][6] = LF_Q          if(IA_NLA_LF, Joachimi2012)
  // ia[0][7] = LF_red_alpha  if(IA_NLA_LF, Joachimi2012)
  // ia[0][8] = LF_red_P      if(IA_NLA_LF, Joachimi2012)
  // ia[0][9] = LF_red_Q      if(IA_NLA_LF, Joachimi2012)
  // ------------------
  // ia[1][0] = A2_ia        if IA_REDSHIFT_EVOLUTION
  // ia[1][1] = eta_ia_tt    if IA_REDSHIFT_EVOLUTION
  // ------------------
  // ia[2][MAX_SIZE_ARRAYS] = b_ta_z[MAX_SIZE_ARRAYS]
  int IA;
  int IA_MODEL;
  int IA_code; // 0 = CFASTPT; 1 = PyFASTPT; 
  double ia[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS];
  double oneplusz0_ia;
  double c1rhocrit_ia;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // PHOTOZ --------------------------------------------
  // ---------------------------------------------------
  // ---------------------------------------------------
  // 1st index: photoz[0][:][:] = SHEAR; photoz[1][:][:] = CLUSTERING
  // 2nd index: photoz[:][0][:] = bias; photoz[:][1][:] = stretch
  double photoz[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS]; 
  // ---------------------------------------------------
  // ---------------------------------------------------
  // SHEAR CALIBRATION ---------------------------------
  // ---------------------------------------------------
  // ---------------------------------------------------
  double shear_calibration_m[MAX_SIZE_ARRAYS];
  // ---------------------------------------------------
  // ---------------------------------------------------
  // GALAXY BIAS ---------------------------------------
  // ---------------------------------------------------
  // ---------------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias in clustering bin i 
  //            b[4][i]: amplitude of magnification bias in clustering bin i 
  //            b[5][i]: nonlocal bK galaxy bias in clustering bin i
  double gb[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS]; // galaxy bias
  // HOD[i] contains HOD parameters of galaxies in clustering bin i
  // 5 parameter model of Zehavi et al. 2011 + modification of concentration
  double hod[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS]; 
  double gc[MAX_SIZE_ARRAYS];  // galaxy concentration parameter
  // ---------------------------------------------------
  // ---------------------------------------------------
  // GAS -----------------------------------------------
  // ---------------------------------------------------
  // ---------------------------------------------------
  //gas[0] = gas_Gamma_KS; // Gamma in K-S profile
  //gas[1] = gas_beta;     // beta: mass scaling index in bound gas fraction
  //gas[2] = gas_lgM0;     // critical halo mass, below which gas ejection is significant
  //gas[3] = gas_eps1;
  //gas[4] = gas_eps2;
  //gas[5] = gas_alpha;
  //gas[6] = gas_A_star;
  //gas[7] = gas_lgM_star;
  //gas[8] = gas_sigma_star;
  //gas[9] = gas_lgT_w;
  //gas[10] = gas_f_H;
  double gas[MAX_SIZE_ARRAYS]; // Compton-Y related variables (read only by
                               // future_port_unfinished/halo_tsz.c, not compiled)
  // ---------------------------------------------------
  // HALO-MODEL INTRINSIC ALIGNMENT (Fortuna et al. 2021; halo.c)
  // ---------------------------------------------------
  // ia_halo[0] = a_1h (satellite radial alignment amplitude)
  // ia_halo[1] = eta_1h (a_1h (1+z)^eta_1h / (1+z_pivot)^eta_1h)
  // ia_halo[2] = z_pivot
  double ia_halo[MAX_SIZE_ARRAYS];
  // red fractions of centrals and satellites (sigmoids in log10 M):
  // [0] lg M_c,cen [1] width_cen [2] lg M_c,sat [3] width_sat
  double ia_red[MAX_SIZE_ARRAYS];
  // HOD of the IA (source) population, {lg M_min, sigma_lgM, lg M_1,
  // lg M_0, alpha, f_c} (the Zheng07 form of HOD_nc, HOD_ns)
  double ia_hod[MAX_SIZE_ARRAYS];
} nuisanceparams;

// Slots of like.probe[]: 1 = the probe is part of the data vector. The
// probe strings of init_probes (generic_interface.cpp) and
// init_probes_cluster (generic_interface_cluster.cpp) set them; the
// cluster probes are cluster.probe[] (structs_cluster.h).
#define PROBE_SS 0           // cosmic shear (xi+-, C_ss)
#define PROBE_GS 1           // galaxy-galaxy lensing (gamma_t, C_gs)
#define PROBE_GG 2           // galaxy clustering (w, C_gg)
#define PROBE_GK 3           // galaxy x CMB lensing
#define PROBE_KK 4           // CMB lensing
#define PROBE_KS 5           // CMB lensing x shear
#define PROBE_GY 6           // galaxy x tSZ
#define PROBE_SY 7           // shear x tSZ
#define PROBE_KY 8           // CMB lensing x tSZ
#define PROBE_YY 9           // tSZ
#define NPROBES 10

// Slots of like.adopt_limber[]
#define LIMBER_GG 0          // galaxy clustering
#define LIMBER_GS 1          // galaxy-galaxy lensing
#define NLIMBER 2

typedef struct
{
  int Ncl;
  int Ncos;
  int Ndata;
  int lrange[2];         // multipole range [RANGE_MIN, RANGE_MAX] of the
                         // Fourier bands (like.ell: Ncl log-spaced centers)
  double* ell;
  double cosmax;
  double Rmin_bias;
  double Rmin_shear;
  int lmax_shear;
  int bias;
  int probe[NPROBES];         // 1 = the probe is in the data vector;
                              // slots PROBE_* (above likepara)
  int adopt_limber[NLIMBER];  // 1 = Limber at every multipole, 0 = the
                              // non-Limber low-l path; slots LIMBER_*
  int use_ggl_efficiency_zoverlap;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // HALO MODEL CHOICES
  // ---------------------------------------------------
  // ---------------------------------------------------
  int galaxy_bias_model[MAX_SIZE_ARRAYS]; // [0] = b1, 
                                          // [1] = b2, 
                                          // [2] = bs2, 
                                          // [3] = b3, 
                                          // [4] = bmag 
  int halo_model[MAX_SIZE_ARRAYS]; // [0] = HMF,
                                   // [1] = BIAS,
                                   // [2] = CONCENTRATION
                                   // [3] = HALO PROFILE
                                   // [4] = DENSITY FIELD of sigma(M)
                                   //       and of the mass function
                                   //       (halo.h: HALO_FIELD_*)
} likepara;

typedef struct
{
  double area;                    // survey_area in deg^2.
  double n_gal;                   // galaxy density per arcmin^2
  double sigma_e;                 // rms inrinsic ellipticity noise
  double area_conversion_factor;  // factor from deg^2 to radian^2:
  double n_gal_conversion_factor; // factor from n_gal/arcmin^2 to n_gal/radian^2:
  double n_lens;                  // lens galaxy density per arcmin^2
  double m_lim;
  char name[CHAR_MAX_SIZE];
} sur;

typedef struct
{
  char runmode[CHAR_MAX_SIZE];
} pdeltapara;

typedef struct
{
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CACHE VARIABLES
  // ---------------------------------------------------
  // ---------------------------------------------------
  uint64_t random;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // To be applied on w_xk (real space cross-corr w/ cmb lensing)
  // Why? cross-correlations are measured by projecting the CMB lensing
  // potential into healpix map, then smoothed by a Gaussian beam  
  // then cross-correlating with galaxy catalog using TreeCorr.
  // ---------------------------------------------------
  // ---------------------------------------------------
  double fwhm;     // beam fwhm in rad (smoothed by a Gaussian beam)
  int healpixwin_ncls; // Precomputed HealPix window function
  double* healpixwin;
  int lk_wxk[2];   // multipole range [RANGE_MIN, RANGE_MAX] of the w_xk sums
  // ---------------------------------------------------
  // ---------------------------------------------------
  // auto-correlation kk bandpower
  // ---------------------------------------------------
  // ---------------------------------------------------
  int nbp_kk;
  int lbp_kk[2];   // multipole range [RANGE_MIN, RANGE_MAX] of the kk bands
  double alpha_Hartlap_cov_kkkk;
  double* theory_offset_kk;
  double** binning_matrix_kk;
} CMBparams;

typedef struct 
{
  int shear_Npowerspectra;       // num shear-shear tomo power combinations
  int ggl_Npowerspectra;         // num galaxy-galaxy lensing tomo combinations
  int clustering_Npowerspectra;  // num galaxy-galaxy clustering tomo combinations
  int* ggl_exclude;              // l-s pairs that are excluded in ggl
  int N_ggl_exclude;             // number of l-s ggl pairs excluded
  uint64_t random_ggl;           // new value whenever the ggl pair list can
                                 // change (bins, ggl_exclude): the pair maps
                                 // test_zoverlap/ZL/ZS/N_ggl rebuild on it
} tomopara;

typedef struct
{
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CACHE VARIABLES
  // ---------------------------------------------------
  // ---------------------------------------------------
  uint64_t random_shear;
  uint64_t random_clustering;
  // ---------------------------------------------------
  // ---------------------------------------------------
  // SOURCE n(Z)
  // ---------------------------------------------------
  // ---------------------------------------------------
  int shear_nbin;         // number of source tomography bins
  int shear_photoz;
  int shear_nzbins;
  double** shear_zdist_table;
  double shear_zdist_zall[2];  // [RANGE_MIN, RANGE_MAX] of the whole table
  double shear_zdist_z[2][MAX_SIZE_ARRAYS]; // [RANGE_MIN|RANGE_MAX][bin]:
                                            // each bin's n(z) support
  // ---------------------------------------------------
  // ---------------------------------------------------
  // CLUSTERING n(z)
  // ---------------------------------------------------
  // ---------------------------------------------------
  int clustering_nbin;    // number of lens galaxy bins
  int clustering_photoz;
  int clustering_nzbins;
  double** clustering_zdist_table;
  double clustering_zdist_zall[2];  // [RANGE_MIN, RANGE_MAX] of the table
  double clustering_zdist_z[3][MAX_SIZE_ARRAYS]; // [RANGE_MIN|RANGE_MAX][bin]:
                                                 // each bin's n(z) support;
                                                 // [ZDIST_MEAN][bin]: its
                                                 // fiducial mean (zmean())
} redshiftparams;

// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
//  EXTERN STRUCTS 
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------

extern likepara like;

extern cosmopara cosmology;

extern tomopara tomo;

extern redshiftparams redshift;

extern sur survey;

extern pdeltapara pdeltaparams;

extern nuisanceparams nuisance;

extern CMBparams cmb;

extern lim limits;

extern Ntab Ntable;

extern FPT FPTbias;

extern FPT FPTIA;

// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
//  RESET STRUCTS 
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------

void reset_cmb_struct(void);
void reset_like_struct(void);
void reset_bary_struct(void);
void reset_nuisance_struct(void);
void reset_redshift_struct(void);
void reset_cosmology_struct(void);
void reset_tomo_struct(void);
void reset_Ntable_struct(void);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
