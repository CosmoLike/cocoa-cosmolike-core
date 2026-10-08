#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

#include "basics.h"
#include "IA.h"
#include "cosmo3D.h"
#include "structs.h"

#include "log.c/src/log.h"

// Layout of the IA nuisance array nuisance.ia[row][column] for each
// redshift model nuisance.IA (IA.h).
//
// IA_NLA_LF: the luminosity- and redshift-dependent NLA amplitude of
// Joachimi et al. 2011 (arXiv:1008.3491), averaged over the red-galaxy
// luminosity function and multiplied by the red fraction f_red, with the
// high-redshift slope and the six luminosity-function nuisance
// parameters of Krause, Eifler & Blazek 2016 (arXiv:1506.08730, Secs.
// 2.1, 4.1 and 5.1); see A_IA_Joachimi below.
//   ia[0][0] = A_ia          (also IA_REDSHIFT_EVOLUTION)
//   ia[0][1] = eta_ia        (also IA_REDSHIFT_EVOLUTION)
//   ia[0][2] = eta_ia_highz
//   ia[0][3] = beta_ia
//   ia[0][4] = LF_alpha
//   ia[0][5] = LF_P
//   ia[0][6] = LF_Q
//   ia[0][7] = LF_red_alpha
//   ia[0][8] = LF_red_P
//   ia[0][9] = LF_red_Q
// IA_REDSHIFT_EVOLUTION: one power law in (1+z)/(1+z0), shared by all
// source bins (1+z0 = nuisance.oneplusz0_ia)
//   ia[0][0] = A1 (A_ia), ia[0][1] = eta1 (eta_ia)
//   ia[1][0] = A2,        ia[1][1] = eta2   (tidal torquing)
//   ia[2][0] = b_TA                          (density weighting)
// IA_REDSHIFT_BINNING: one value per source bin n
//   ia[0][n] = A1, ia[1][n] = A2, ia[2][n] = b_TA
// A2 and b_TA multiply only the TATT one-loop terms.

static double LF_coefficients[2][5] =
 { 
    {-1.,0.,0.,0.,0.},
    {-1.,0.,0.,0.,0.}
  }; //{ Phistat, alpha, Mstar, Q, P} [0][] = red, [1][] = all

static double LF_coefficients_GAMA[2][5] =
  {
    {1.11e-3,-0.57,-20.34,1.8,-1.2},
    {0.94e-3,-1.23,-20.70,0.7,1.8}
  }; //from GAMA survey http://arxiv.org/pdf/1111.0166v2.pdf

static double LF_coefficients_DEEP2[2][5] =
  {
    {1.11e-3,-0.57,-20.34,1.20,-1.15},
    {0.94e-3,-1.23,-20.70,1.23,-.3}
  }; //P,Q from DEEP2 (B-band), otherwise GAMA

// ---------------------------------------------------------------------------
// k+e corrections for early types, restframe r band at
// z = 0.,0.1,..,3.0; interpolated from
// http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BAS/122/399
// ---------------------------------------------------------------------------
static double KE[31] = 
  {
    0. ,   -0.018,  0.013,  0.043,  0.091,  0.236,  0.449,
    0.667,  0.827,  0.907,  0.916,  0.91 ,  0.932,  0.901,
    0.835,  0.735,  0.594,  0.427,  0.179, -0.025, -0.226,
   -0.423, -0.591, -0.754, -0.913, -1.061, -1.181, -1.301,
   -1.421, -1.541, -1.661
  };

void set_LF_GAMA(void)
{
  for (int i = 0; i <5; i++)
  {
    LF_coefficients[0][i] = LF_coefficients_GAMA[0][i];
    LF_coefficients[1][i] = LF_coefficients_GAMA[1][i];
  }
}

void set_LF_DEEP2(void)
{
  for (int i = 0; i <5; i++)
  {
    LF_coefficients[0][i] = LF_coefficients_DEEP2[0][i];
    LF_coefficients[1][i] = LF_coefficients_DEEP2[1][i];
  }
}

double M_abs(const double mag, const double a)
{ //in h = 1 units, incl. Poggianti 1997 k+e-corrections
  static double* table;

  if (table == 0)
  {
    // read in + tabulate k+e corrections for early types, restframe r band
    // interpolated from 
    // http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BAS/122/399
    const int size = 31;
    
    table = (double*) malloc(sizeof(double)*size);
    
    for (int i = 0; i<size; i++) table[i] = KE[i];
  }

  // the k+e table ends at z = 3, and beyond it there is no meaningful IA
  // model either: clamp z to 2.99
  const double z = 1./a - 1.0 >= 3.0 ? 2.99 : 1./a - 1.0;
  const double ke = interpol1d(table, 31, 0., 3.0, 0.1, z);
  struct chis chidchi = chi_all(a);
  const double fK = f_K(chidchi.chi);

  return mag - 5.0*log10(fK/a*cosmology.coverH0) - 25.0 - ke;
}

double f_red_LF(const double mag, const double a)
{
  if (LF_coefficients[0][0]< 0)
  {
    log_fatal("Missing Luminosity function");
    exit(1);
  }

  const double LF_alpha = nuisance.ia[0][4];
  const double LF_P = nuisance.ia[0][5]; 
  const double LF_Q = nuisance.ia[0][6];
  const double LF_red_alpha = nuisance.ia[0][7];
  const double LF_red_P = nuisance.ia[0][8];   
  const double LF_red_Q = nuisance.ia[0][9]; 

  // r-band LF parameters, Tab. 5 in http://arxiv.org/pdf/1111.0166v2.pdf

  //red galaxies
  const double alpha_red = LF_coefficients[0][1] + LF_red_alpha;
  const double Mstar_red = LF_coefficients[0][2]; //in h = 1 units
  const double Q_red = LF_coefficients[0][3] + LF_red_Q;
  const double P_red = LF_coefficients[0][4] + LF_red_P;
  const double Phistar_red = LF_coefficients[0][0];

  double LF_red[3];
  LF_red[0] = Phistar_red*pow(10.0,0.4*P_red*(1./a-1));
  LF_red[1] = Mstar_red-Q_red*(1./a-1. -0.1);
  LF_red[2] = alpha_red;

  //all galaxies
  const double alpha = LF_coefficients[1][1] + LF_alpha;
  const double Mstar = LF_coefficients[1][2];
  const double Q = LF_coefficients[1][3] + LF_Q;
  const double P = LF_coefficients[1][4] + LF_P;
  const double Phistar = LF_coefficients[1][0];

  double LF_all[3];
  LF_all[0] = Phistar*pow(10.0,0.4*P*(1./a-1));
  LF_all[1] = Mstar -Q*(1./a-1. - 0.1);
  LF_all[2] = alpha;

  const double Mlim = M_abs(mag,a); //also in h = 1 units

  return LF_red[0]/LF_all[0]*
    gsl_sf_gamma_inc(LF_red[2]+1, pow(10.0,-0.4*(Mlim-LF_red[1])))/
    gsl_sf_gamma_inc(LF_all[2]+1, pow(10.0,-0.4*(Mlim-LF_all[1])));
}

double A_LF(double mag, double a)
{ // averaged (L/L_0)^beta over red galaxy LF
  if (LF_coefficients[0][0]< 0)
  {
    log_fatal("Missing Luminosity function");
    exit(1);
  }

  const double beta_ia = nuisance.ia[0][3];
  const double LF_red_alpha = nuisance.ia[0][7];
  const double LF_red_Q = nuisance.ia[0][9]; 

  // r-band LF parameters, Tab. 5 in http://arxiv.org/pdf/1111.0166v2.pdf

  //red galaxies
  const double alpha_red = LF_coefficients[0][1] + LF_red_alpha;
  const double Mstar_red = LF_coefficients[0][2]; //in h = 1 units
  const double Q_red = LF_coefficients[0][3] + LF_red_Q;

  double LF_red[3];
  LF_red[1] = Mstar_red-Q_red*(1./a-1. -.1);
  LF_red[2]= alpha_red;

  const double Mlim = M_abs(mag, a);
  const double Lstar = pow(10.0, -0.4*LF_red[1]);
  const double x = pow(10.0, -0.4*(Mlim - LF_red[1])); //Llim/Lstar
  const double L0 =pow(10.0, -0.4*(-22.)); //all in h = 1 units

  return pow(Lstar/L0, beta_ia)* gsl_sf_gamma_inc(LF_red[2] + beta_ia + 1, x)/
    gsl_sf_gamma_inc(LF_red[2] + 1, x);
}

// ---------------------------------------------------------------------------
// Return 1 if the all-galaxy and red-galaxy LF parameters (with their
// nuisance shifts) are unphysical for the source sample, 0 otherwise
// (the rejection of Krause, Eifler & Blazek 2016, Sec. 5.1). The scan
// starts at a = 1/(1 + zmax) + 0.005, zmax =
// redshift.shear_zdist_zall[RANGE_MAX]. It fails when, at that starting
// a, the survey limit survey.m_lim maps to an absolute magnitude brighter
// than M*(z) of either LF, or when f_red > 1 at any a from there to
// a = 1 in steps of 0.01.
// ---------------------------------------------------------------------------
int check_LF(void)
{
  const double LF_Q = nuisance.ia[0][6];
  const double LF_red_Q = nuisance.ia[0][9]; 

  double a = 1./(1. + redshift.shear_zdist_zall[RANGE_MAX]) + 0.005;
  
  const double MABS = M_abs(survey.m_lim, a);

  const double x1 = 
    LF_coefficients[1][2] - (LF_coefficients[1][3] + LF_Q)*(1./a-1. - 0.1);

  const double x2 = 
    LF_coefficients[0][2] - (LF_coefficients[0][3] + LF_red_Q)*(1./a-1. - 0.1);

  while (a < 1.)
  {
    if (MABS < x1 || MABS < x2)
      return 1;

    if (f_red_LF(survey.m_lim,a) > 1.0)
      return 1;

    a += 0.01;
  }

  return 0;
}

double A_IA_Joachimi(const double a)
{
  const double highz = 0.75;
  const double z = 1.0/a - 1.0;

  const double A_ia = nuisance.ia[0][0];
  const double eta_ia = nuisance.ia[0][1];
  const double eta_ia_highz = nuisance.ia[0][2];

  // A_0* < (L/L_0)^beta > *f_red
  const double A_red = A_ia*A_LF(survey.m_lim, a)*f_red_LF(survey.m_lim, a);

  if (a < 1./(1.+ highz))
  { // z > highz, factor in uncertainty in extrapolation of redshift scaling
    return A_red*pow((1.0 + z)/nuisance.oneplusz0_ia, eta_ia)*
      pow((1.0 + z)/(1.0 + highz), eta_ia_highz);
  }
  else
  { //standard redshift scaling
    return A_red*pow((1.0 + z)/nuisance.oneplusz0_ia, eta_ia);
  }
}

// ---------------------------------------------------------------------------
// Linear (tidal alignment, NLA) IA amplitude of source bins n1 and n2 at
// scale factor a:
//
//   res[i] = A1(z, n_i) * Omega_m * c1rhocrit_ia / D(a),   i = 0, 1
//
// with D(a) = growfac_a, the linear growth factor normalized to D(1) = 1,
// and c1rhocrit_ia the dimensionless product Cbar1 rho_crit of the
// SuperCOSMOS normalization (Bridle & King 2007). A1 by nuisance.IA, with
// the parameters listed at the top of this file:
//
//   NO_IA                   0
//   IA_NLA_LF               A_IA_Joachimi(a), the same for both bins
//   IA_REDSHIFT_BINNING     ia[0][n_i]
//   IA_REDSHIFT_EVOLUTION   ia[0][0] * ((1+z)/(1+z0))^ia[0][1]
//
// The 1/D(a) keeps the alignment at its formation value: the shapes
// respond to the tidal field at formation, while the density field they
// are correlated with keeps growing as D(a).
//
// The returned amplitude is positive for A1 > 0. Blazek et al. 2019
// (arXiv:1708.09247) write the coefficient of the tidal field as
// C1 = -A1 Cbar1 rho_crit Omega_m / D: galaxies align radially towards
// overdensities, opposite to the tangential lensing shear. The callers
// apply that minus sign: they subtract W_source * IA_A1_Z1 from the
// lensing kernel (cosmo2D.c, cosmo2D_cluster.c, covariances/). The
// COCOA warnings in the body compare with C1_TA of the original CosmoLike
// TATT code (cosmo2D_fullsky_TATT.c, not part of this library), which
// carried the sign itself.
//
// Parameters:
//   a         - scale factor, a > 0
//   growfac_a - D(a), normalized to D(1) = 1
//   n1, n2    - source tomographic bin indices
//   res       - output: res[0] for bin n1, res[1] for bin n2
// ---------------------------------------------------------------------------
void IA_A1_Z1Z2(
    const double a, 
    const double growfac_a, 
    const int n1, 
    const int n2, 
    double res[2]
  )
{ 
  // COCOA (WARNING): THERE IS MINUS SIGN DIFFERENCE COMPARED TO C1_TA 
  // COCOA (WARNING): IN ORIGINAL COSMOLIKE (SEE: cosmo2D_fullsky_TATT.c)

  if (!(a>0)) 
  {
    log_fatal("a>0 not true");
    exit(1);
  }

  double A_Z1 = 0.0;
  double A_Z2 = 0.0;
  
  switch(nuisance.IA)
  {
    case NO_IA:
    {
      A_Z1 = 0.0;
      A_Z2 = 0.0;
      break;
    }
    case IA_NLA_LF:
    {
      A_Z1 = A_IA_Joachimi(a);
      A_Z2 = A_Z1;
      break;
    }
    case IA_REDSHIFT_BINNING:
    { 
      A_Z1 = nuisance.ia[0][n1];
      A_Z2 = nuisance.ia[0][n2];
      break;
    }
    case IA_REDSHIFT_EVOLUTION:
    {
      const double A_IA = nuisance.ia[0][0];
      const double eta  = nuisance.ia[0][1];
      A_Z1 = A_IA*pow((1.0/a)/nuisance.oneplusz0_ia, eta);
      A_Z2 = A_Z1;
      break;
    }
    default:
    {
      log_fatal("nuisance.IA = %d not supported", nuisance.IA);
      exit(1);
    }
  }
  
  const double x = cosmology.Omega_m*nuisance.c1rhocrit_ia/growfac_a;
  res[0] = A_Z1 * x;
  res[1] = A_Z2 * x;
}

double IA_A1_Z1(const double a, const double growfac_a, const int n1)
{
  double res[2];
  IA_A1_Z1Z2(a, growfac_a, n1, n1, res);
  return res[0];
}

// ---------------------------------------------------------------------------
// Tidal-torquing (TATT) IA amplitude of source bins n1 and n2 at scale
// factor a:
//
//   res[i] = A2(z, n_i) * Omega_m * c1rhocrit_ia / D(a)^2,   i = 0, 1
//
// with one 1/D per power of the tidal field in the quadratic term. A2 by
// nuisance.IA: 0 (NO_IA), ia[1][n_i] (IA_REDSHIFT_BINNING),
// ia[1][0] * ((1+z)/(1+z0))^ia[1][1] (IA_REDSHIFT_EVOLUTION); IA_NLA_LF
// has no torquing amplitude and stops with an error.
//
// Blazek et al. 2019 (arXiv:1708.09247) write the coefficient of the
// quadratic tidal field as C2 = 5 A2 Cbar1 rho_crit Omega_m^2
// / (Omega_m,fid D^2), positive for A2 > 0 (no minus sign, unlike C1).
// This amplitude leaves out the 5, which the callers apply (5 C2 in the
// terms linear in C2, 25 in the C2^2 term; cosmo2D.c,
// covariances/ia_cov.c), and carries one power of Omega_m, the scaling
// the DES Y1 cosmic-shear analysis adopted for C1 and C2 alike (Troxel et
// al. 2018, as noted by Blazek et al. 2019). The COCOA warning in the
// body compares with C2_TT of the original CosmoLike TATT code, which
// included the 5.
//
// Parameters: as IA_A1_Z1Z2.
// ---------------------------------------------------------------------------
void IA_A2_Z1Z2(
    const double a, 
    const double growfac_a, 
    const int n1, 
    const int n2, 
    double res[2]
  )
{ // COCOA (WARNING): THERE IS factor x5 DIFFERENCE COMPARED TO C2_TT 
  // COCOA (WARNING): IN ORIGINAL COSMOLIKE (SEE: cosmo2D_fullsky_TATT.c)
  if (!(a>0)) 
  {
    log_fatal("a>0 not true");
    exit(1);
  }
  
  double A2_Z1 = 0.0;
  double A2_Z2 = 0.0;

  switch(nuisance.IA)
  {
    case NO_IA:
    {
      A2_Z1 = 0.0;
      A2_Z2 = 0.0;
      break;
    }
    case IA_NLA_LF:
    {
      log_fatal("IA_NLA_LF TT not supported");
      exit(1);      
    }
    case IA_REDSHIFT_BINNING:
    { 
      A2_Z1 = nuisance.ia[1][n1];
      A2_Z2 = nuisance.ia[1][n2];
      break;
    }
    case IA_REDSHIFT_EVOLUTION:
    {
      const double A_IA = nuisance.ia[1][0];
      const double eta  = nuisance.ia[1][1];
      A2_Z1 = A_IA*pow(1.0/(a*nuisance.oneplusz0_ia), eta);
      A2_Z2 = A2_Z1;
      break;
    } 

    default:
    {
      log_fatal("nuissance.IA = %d not supported", nuisance.IA);
      exit(1);  
    }
  }

  const double x = cosmology.Omega_m*nuisance.c1rhocrit_ia/(growfac_a*growfac_a);
  res[0] = A2_Z1 * x;
  res[1] = A2_Z2 * x;
}

double IA_A2_Z1(const double a, const double growfac_a, const int n1)
{
  double res[2];
  IA_A2_Z1Z2(a, growfac_a, n1, n1, res);
  return res[0];
}

// ---------------------------------------------------------------------------
// Density-weighting coefficient b_TA of source bins n1 and n2 (TATT). The
// tidal-alignment term weighted by the local density, C1delta (delta s),
// has C1delta = b_TA C1 (Blazek et al. 2019, arXiv:1708.09247; b_TA = 1
// is pure density weighting of sources with linear bias 1). b_TA by
// nuisance.IA: 1 (NO_IA, where C1 = 0 anyway), ia[2][n_i]
// (IA_REDSHIFT_BINNING), ia[2][0] for every bin and redshift
// (IA_REDSHIFT_EVOLUTION); IA_NLA_LF stops with an error. a and
// growfac_a are unused.
// ---------------------------------------------------------------------------
void IA_BTA_Z1Z2(
    const double a __attribute__((unused)), 
    const double growfac_a __attribute__((unused)), 
    const int n1, const int n2, double res[2]
  )
{
  double BTA_Z1 = 0.0;
  double BTA_Z2 = 0.0;

  switch(nuisance.IA)
  {
    case NO_IA:
    {
      BTA_Z1 = 1.0;
      BTA_Z2 = 1.0;
      break;
    }
    case IA_NLA_LF:
    {
      log_fatal("IA_NLA_LF BTA not supported");
      exit(1);      
    }
    case IA_REDSHIFT_BINNING:
    {
      BTA_Z1 = nuisance.ia[2][n1];
      BTA_Z2 = nuisance.ia[2][n2];
      break;
    }
    case IA_REDSHIFT_EVOLUTION:
    {
      BTA_Z1 = nuisance.ia[2][0];
      BTA_Z2 = BTA_Z1;
      break;
    }
    default:
    {
      log_fatal("nuisance.IA = %d not supported", nuisance.IA);
      exit(1);  
    }
  }

  res[0] = BTA_Z1;
  res[1] = BTA_Z2;

  return;
}

double IA_BTA_Z1(const double a, const double growfac_a, const int n1)
{
  double res[2];
  IA_BTA_Z1Z2(a, growfac_a, n1, n1, res);
  return res[0];
}
