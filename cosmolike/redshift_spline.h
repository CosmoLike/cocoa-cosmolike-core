#ifndef __COSMOLIKE_REDSHIFT_SPLINE_H
#define __COSMOLIKE_REDSHIFT_SPLINE_H
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// integration boundary routines
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double amin_source(int i);

double amax_source(int i);

double amax_source_IA(int i);

double amin_lens(int i);

double amax_lens(int i);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// redshift overlap tests, allowed tomography combinations
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// test whether the (l,zl) bin is in the linear clustering regime
int test_kmax(double l, int zl); 

// test whether source bin zs is behind lens bin zl
int test_zoverlap(int zl, int zs); 

int ZL(int Nbin); // find zlens bin of tomo combination (galaxy-galaxy lensing)

int ZS(int Nbin); // find zsource bin of tomo combination (galaxy-galaxy lensing)

int N_ggl(int zl, int zs); // find tomo bin number tomography combination

int Z1(int Nbin); // find z1 bin of tomo combination (cosmic shear)

int Z2(int Nbin); // find z2 bin of tomo combination (cosmic shear)

// find tomo bin number tomography combination (cosmic shear)
int N_shear(int z1, int z2); 

int ZCL1(int Nbin); // find z1 bin of tomo combination (galaxy clustering)

int ZCL2(int Nbin); // find z2 bin of tomo combination (galaxy clustering)

// find tomo bin number tomography combination (galaxy clustering)
int N_CL(int z1, int z2);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// shear routines (for redshift distributions, including photo-zs)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// returns n(ztrue | j), works only with binned distributions; 
// j =-1 -> no tomography; j>= 0 -> tomography bin j
double zdistr_photoz(double zz, const int nj); 

// mean true redshift of source galaxies in tomography bin ni
double zmean_source(int ni); 

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// clustering routines
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// returns n(ztrue | j), works only with binned distributions;
// j =-1 -> no tomography; j>= 0 -> tomography bin j
double pf_photoz(double zz, int j); 

// mean true redshift of (clustering/lens) galaxies within redshift bin ni
double zmean(int ni);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// lens efficiencies
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double g_cmb(double a); // lens efficiency for CMB lensing

// lens efficiency of source galaxies in tomography bin ni
double g_tomo(double a, const int ni);

// lens efficiency of source galaxies in tomography bin ni
double g2_tomo(double a, int ni);

// lens efficiency of *lens* galaxies in tomography bin ni
// used in magnification calculations
double g_lens(double a,  int ni);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
