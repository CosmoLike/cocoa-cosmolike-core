// ---------------------------------------------------------------------------
// PARKING LOT: unfinished Compton-y port from the original cosmolike.
//
// Nothing in cocoa calls the C_gy, C_ys, C_ky and C_yy family: no project
// enables the gy/sy/ky/yy probes and every python binding for them was
// commented out. The pieces live here, out of the build, until the port
// from the original cosmolike is finished; on revival they need the batch
// _work treatment the shear pipeline now uses. This file is NOT compiled.
//
// Contents: the cosmo2D.h declarations, the cosmo2D.c functions, and the
// cosmo2D_wrapper.cpp python bindings (kept commented, as they were).
// ---------------------------------------------------------------------------

// ------------------------------ cosmo2D.h ---------------------------------

double C_gy_tomo_limber(const double l, const int ni);
double C_ys_tomo_limber(const double l, const int ni);
double C_ky_limber(const double l);
double C_yy_limber(const double l);
double C_gy_tomo_limber_nointerp(const double l, const int ni, const int init);
double C_ys_tomo_limber_nointerp(const double l, const int ni, const int init);
double C_ky_limber_nointerp(const double l, const int init);
double C_yy_limber_nointerp(const double l, const int init);
double int_for_C_gy_tomo_limber(double a, void* params);
double int_for_C_ys_tomo_limber(double a, void* params);
double int_for_C_ky_limber(double a, void* params);
double int_for_C_yy_limber(double a, void *params);

// ------------------------------ cosmo2D.c ---------------------------------

double int_for_C_gy_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;
  const int nl = (int) ar[0];
  if (nl < 0 || nl > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", nl); exit(1);
  }
  const double l = ar[1];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi;
  const double k = ell/fK;
  const double z = 1./a - 1.;

  const double b1   = gb1(z, nl);
  const double bmag = gbmag(z, nl);

  const double WY = W_y(a);
  const double WGAL = W_gal(a, nl, hoverh0);
  const double WMAG = W_mag(a, fK, nl);

  const double ell_prefactor = l*(l + 1.)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)

  double res = WY;

  if (include_HOD_GX == 1)
  {
    if (include_RSD_GY == 1) {
      log_fatal("RSD not implemented with (HOD = TRUE)"); exit(1);
    }
    else { 
      log_fatal("(HOD = TRUE) not implemented"); exit(1);
    }
  }
  else
  {
    if (include_RSD_GY == 1)
    {
      log_fatal("RSD not implemented");
      exit(1);
    }
    else
      res *= WGAL*b1 + WMAG*ell_prefactor*bmag;

    const double PK = p_my(k, a);
    res *= PK;
  }
  return res*chidchi.dchida/(fK*fK);
}

double C_gy_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (w == NULL || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 : 
                         (1 == hdi) ? 128 : 
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[2] = {(double) ni, l};
  const double amin = amin_lens(ni);
  const double amax = 0.99999;

  double res = 0.0;
  if (init == 1)
    res = int_for_C_gy_tomo_limber(amin, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_gy_tomo_limber;
    res =  gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_gy_tomo_limber(double l, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[4], Ntable.random))
  {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2]   = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);

    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_ell);
  }

  if (fdiff2(cache[1], cosmology.random) || 
      fdiff2(cache[2], nuisance.random_photoz_clustering) ||
      fdiff2(cache[3], redshift.random_clustering) ||
      fdiff2(cache[4], Ntable.random) ||
      fdiff2(cache[5], nuisance.random_galaxy_bias))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_gy_tomo_limber_nointerp(exp(lim[0]), 0, 1);
    }    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k=0; k<redshift.clustering_nbin; k++) {
      for (int i=0; i<Ntable.N_ell; i++) {
        table[k][i]= C_gy_tomo_limber_nointerp(exp(lim[0] + i*lim[2]), k, 0);
      }
    }
    cache[1] = cosmology.random;
    cache[2] = nuisance.random_photoz_clustering;
    cache[3] = redshift.random_clustering;
    cache[4] = Ntable.random;
    cache[5] = nuisance.random_galaxy_bias;
  }

  const int q =  ni; 
  if (q < 0 || q > redshift.clustering_nbin - 1)
  {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  } 
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));

  return interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_ys_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }

  double* ar = (double*) params;
  const int ni = (int) ar[0];
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double l = ar[1];
  
  const double ell = l + 0.5;
  const double growfac_a = growfac(a);
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi;
  const double k  = ell/fK;
  
  const double PK = p_my(k, a);

  const double WK1 = W_kappa(a, fK, ni);
  const double WY  = W_y(a);

  const double tmp = (l - 1.0)*l*(l + 1.0)*(l + 2.0); // prefactor correction (1812.05995 eqs 74-79)
  const double ell_prefactor2 = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;

  const double A_Z1 = IA_A1_Z1(a, growfac_a, ni);
  const double WS1  = W_source(a, ni, hoverh0) * A_Z1;

  const double res = (WK1 - WS1)*WY;

  return res*PK*(chidchi.dchida/(fK*fK))*ell_prefactor2;
}

double C_ys_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.shear_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  } 

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 80 + 50 * abs(Ntable.high_def_integration);
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[2] = {(double) ni, l};
  const double amin = amin_source(ni);
  const double amax = 0.99999;

  double res = 0.0;
  if (init == 1)
    res = int_for_C_ys_tomo_limber(amin, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_ys_tomo_limber;
    res =  gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_ys_tomo_limber(double l, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[4], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_ell);

    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_ys_tomo_limber_nointerp(exp(lim[0]), 0, 1);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k=0; k<redshift.shear_nbin; k++) {
      for (int i=0; i<Ntable.N_ell; i++) {
        table[k][i] = C_ys_tomo_limber_nointerp(exp(lim[0] + i*lim[2]), k, 0);
      }
    } 
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  
  if (ni < 0 || ni > redshift.shear_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d (max %d)", ni, 
      redshift.shear_nbin);
    exit(1);
  }

  const int q =  ni; 
  if (q < 0 || q > redshift.shear_nbin - 1)
  {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  } 
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));

  return interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_ky_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }

  double *ar = (double*) params;
  const double l = ar[0];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double fK = chidchi.chi;
  const double k = ell/fK;
  
  const double PK = p_my(k, a);
  const double WK = W_k(a, fK);
  const double WY = W_y(a);

  const double ell_prefactor = l*(l + 1.0)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)

  return (WK*WY*PK*chidchi.dchida/(fK*fK))*ell_prefactor;
}

double C_ky_limber_nointerp(const double l, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (w == NULL || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 80 + 50 * abs(Ntable.high_def_integration);
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {l};
  const double amin = limits.a_min_hm;
  const double amax = 1.0 - 1.e-5;

  double res = 0.0;
  if (init == 1)
    res = int_for_C_ky_limber(amin, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_ky_limber;
    res =  gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_ky_limber(double l)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[1], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double*) malloc1d(Ntable.N_ell);

    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    (void) C_ky_limber_nointerp(exp(lim[0]), 1);  // init static vars
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_ell; i++) {
      table[i] = C_ky_limber_nointerp(exp(lim[0] + i*lim[2]), 0);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > l_max = %e. Extrapolation adopted", l, exp(lim[1]));

  return interpol1d(table, Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_yy_limber(double a, void *params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;
  const double l = ar[0];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double fK = chidchi.chi;
  const double k  = ell/fK;

  const double PK = p_yy(k, a);
  
  const double WY = W_y(a);

  return WY*WY*PK*chidchi.dchida/(fK*fK);
}

double C_yy_limber_nointerp(const double l, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 80 + 50 * abs(Ntable.high_def_integration);
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {l};
  const double amin = limits.a_min;
  const double amax = 1.0 - 1.e-5;

  double res = 0.0;
  if (init == 1) {
    res = int_for_C_yy_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_yy_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_yy_limber(double l)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[1], Ntable.random))
  {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1.0);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);

    if (table != NULL) free(table);
    table = (double*) malloc1d(Ntable.N_ell);
  }

  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_yy_limber_nointerp(exp(lim[0]), 1);
    }
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_ell; i++) {
      table[i] = C_yy_limber_nointerp(exp(lim[0] + i*lim[2]), 0);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  return interpol1d(table, Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ------------------------- cosmo2D_wrapper.cpp ----------------------------

/*
double C_gy_tomo_limber_cpp(const double l, const int ni)
{
  return C_gy_tomo_limber_nointerp(l, ni, 0, 0);
}

arma::Mat<double> C_gy_tomo_limber_cpp(const arma::Col<double> l)
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_gy_tomo_limber_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Mat<double> result(l.n_elem, redshift.clustering_nbin);
  for (int nz=0; nz<redshift.clustering_nbin; nz++) { // init static variables
    double tmp = C_gy_tomo_limber_nointerp(l(0), nz, 0, 1);
  }
  #pragma omp parallel for collapse(2)
  for (int nz=0; nz<redshift.clustering_nbin; nz++) {
    for (int i=0; i<static_cast<int>(l.n_elem); i++) {
      result(i, nz) = C_gy_tomo_limber_nointerp(l(i), nz, 0, 0);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_ys_tomo_limber_cpp(const double l, const int ni)
{
  return C_ys_tomo_limber_nointerp(l, ni, 0, 0);
}

arma::Mat<double> C_ys_tomo_limber_cpp(const arma::Col<double> l)
{
  if (l.n_elem == 0)
  {
    spdlog::critical("{}: l array size = {}", 
                     "C_ys_tomo_limber_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Mat<double> result(l.n_elem, redshift.shear_nbin);
  for (int nz=0; nz<redshift.shear_nbin; nz++) { // init static variables
    (void) C_ys_tomo_limber_nointerp(l(0), nz, 0, 1);
  }
  #pragma omp parallel for collapse(2)
  for (int nz=0; nz<redshift.shear_nbin; nz++) {
    for (int i=0; i<l.n_elem; i++) {
      result(i, nz) = C_ys_tomo_limber_nointerp(l(i), nz, 0, 0);
    }
  }
  return result;
}
*/

/*
double C_ky_limber_cpp(const double l)
{
  return C_ky_limber_nointerp(l, 0, 0);
}

arma::Col<double> C_ky_limber_nointerp_cpp(const arma::Col<double> l)
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_ky_limber_nointerp_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Col<double> result(l.n_elem);
  { // init static variables
    (void) C_ky_limber_nointerp(l(0), 0, 1);
  }
  #pragma omp parallel for
  for (int i=0; i<l.n_elem; i++)
    result(i) = C_ky_limber_nointerp(l(i), 0, 0);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_yy_limber_cpp(double l)
{
  return C_yy_limber_nointerp(l, 0, 0);
}

arma::Col<double> C_yy_limber_nointerp_cpp(const arma::Col<double> l)
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_yy_limber_nointerp_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Col<double> result(l.n_elem);
  { // init static variables
    (void) C_yy_limber_nointerp(l(0), 0, 1);
  }
  #pragma omp parallel for
  for (int i=0; i<l.n_elem; i++)
    result(i) = C_yy_limber_nointerp(l(i), 0, 0);
  return result;
}
*/

/*
double int_for_C_ky_limber_cpp(const double a, const double l)
{
  double ar[2] = {l, (double) 0.0}; 
  return int_for_C_ky_limber(a, (void*) ar);
}

arma::Mat<double> int_for_C_ky_limber_cpp(
    const arma::Col<double> a, 
    const arma::Col<double> l
  )
{
  if (!(l.n_elem > 0 && a.n_elem > 0)) {
    spdlog::critical("{}: l array size = {} and scale factor array size = {}", 
                      "int_for_C_ky_limber_cpp", 
                      l.n_elem,
                      a.n_elem);
    exit(1);
  }

  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-variable"
  { // init static variables
    double tmp = int_for_C_ky_limber_cpp(a(0), l(0));
  }
  #pragma GCC diagnostic pop

  arma::Mat<double> result(a.n_elem, l.n_elem);

  #pragma omp parallel for collapse(2)
  for (int i=0; i<l.n_elem; i++)
  {
    for (int j=0; j<a.n_elem; j++)
    {
      result(j, i) = int_for_C_ky_limber_cpp(a(j), l(i));
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_yy_limber_cpp(const double a, const double l)
{
  double ar[2] = {l, (double) 0.0}; 
  return int_for_C_yy_limber(a, (void*) ar);
}

arma::Mat<double> int_for_C_yy_limber_cpp(
    const arma::Col<double> a, 
    const arma::Col<double> l
  )
{
  if (!(l.n_elem > 0 && a.n_elem > 0)) {
    spdlog::critical("{}: l array size = {} and scale factor array size = {}", 
                      "int_for_C_yy_limber_cpp", 
                      l.n_elem,
                      a.n_elem);
    exit(1);
  }

  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-variable"
  { // init static variables
    double tmp = int_for_C_yy_limber_cpp(a(0), l(0));
  }
  #pragma GCC diagnostic pop

  arma::Mat<double> result(a.n_elem, l.n_elem);

  #pragma omp parallel for collapse(2)
  for (int i=0; i<l.n_elem; i++)
  {
    for (int j=0; j<a.n_elem; j++)
    {
      result(j, i) = int_for_C_yy_limber_cpp(a(j), l(i));
    }
  }

  return result;
}

*/
