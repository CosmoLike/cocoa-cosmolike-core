// The halo-model matter x electron-pressure (p_my) and pressure auto (p_yy)
// spectra of the thermal-SZ (Compton-y) side of the halo model
// (2005.00009, HMx; one-halo damping of 2009.01858, HMcode-2020). No
// likelihood uses them, so they are kept here, outside the compiled
// sources.
//
// They depend on halo.c: its static halo_warmup and mass-function helpers
// (fnu_params_at, hb1nu_params_at, bias_norm, conc) and the GAS PROFILES
// section (u_KS, frac_bnd, frac_ejc, u_y_ejc, nuisance.gas), which stays in
// halo.c. To use them again:
//   1. paste the two functions back into halo.c, after p_mm;
//   2. declare them in halo.h (double p_my(const double k, const double a);
//      double p_yy(const double k, const double a););
//   3. move the wrappers of halo_tsz_wrapper.cpp back into halo_wrapper.cpp
//      and declare them in halo_wrapper.hpp;
//   4. before using p_my/p_yy with massive neutrinos, read
//      .claude/skills/cosmolike-dev/references/fable_review_neutrino_halos.md:
//      their k_s damping is HMcode-2020's, whose fitted variable is the cold
//      sigma8(z).
// cosmo2d_tmp.c (this folder) holds unfinished C_l builders that call them.
//
// Their windows (in the halo-model notation of halo.c, HALO MODEL POWER
// SPECTRA banner):
//
//   pressure  W_y = W_p (1-halo); + u_y_ejc in I11_y       U (energy)
//
// One M/rho_m per matter leg, none on the pressure: W_p is already the
// volume integral of the pressure (~ M^(5/3) at k -> 0, 2005.00009
// Eq. 41); with an extra M/rho_m a halo's pressure would scale as M^(8/3).
// The ejected gas follows the linear field outside halos, so it enters
// the 2-halo term only, as a k-independent window (2005.00009 sec. 3.3).
//
// The y spectra damp their 1-halo term at low k, I02 -> I02 x/(1 + x),
// x = (k/k_s)^4, k_s = 0.05618 (sigma8 a)^-1.013 h/Mpc (2009.01858
// Eq. 17 and Table 2).

// ---------------------------------------------------------------------------
// P_my(k, a), the halo-model matter-pressure cross spectrum, from a table
// of ln P on Ntable.N_a x Ntable.N_k_nlin nodes uniform in (a, ln k), read
// bilinearly (interpol2d) and exponentiated (section banner):
//
//   P_my   = I02_my S(k, a) + I11_m I11_y P_lin        (2005.00009 Eqs. 1-2)
//   I02_my = int dlnM dn/dlnM (M/rho_m) u(k|M) W_y(M, k)
//   I11_m  = int dlnM dn/dlnM b(nu) (M/rho_m) u(k|M) + A(a) u(k|M_min)
//   I11_y  = int dlnM dn/dlnM b(nu) [W_y(M, k) + W_ejc(M)]
//            + A(a) [W_y(M_min, k) + W_ejc(M_min)]/(M_min/rho_m)
//   S      = x/(1 + x),  x = (k/k_s)^4                   (2009.01858 Eq. 17)
//
// dn/dlnM, (M/rho_m) u, b and A(a) = 1 - bias_norm(a) as in p_mm (its
// header). W_y = Y(a) B(M) u_KS(c, k, r_Delta) is the bound-gas pressure
// window (GAS PROFILES banner), Y(a) = (2 alpha/(3a)) mu_p/mu_e and
// B(M) = f_bnd(M) M^2/r_Delta; W_ejc = u_y_ejc(M) the ejected gas, in the
// 2-halo term only and k-independent. k_s(a) = 0.05618 (sigma8 a)^-1.013
// h/Mpc (2009.01858 Table 2) with sigma8 = sigma(M8) read from the sigma2
// table, M8 = (4 pi/3) rho_m (8 Mpc/h)^3.
//
// 1. Quadrature: the Gauss-Legendre rule of p_mm (its header,
// item 1) over [ln M_min, ln M_max], mapped once in the rebuild block.
//
// 2. Loop levels as in p_mm (its header, item 2), two kernels per node:
// nfw_um for the matter leg, u_KS for the pressure leg:
//
//   per refill, per node q       M, w; nu0 = delta_c/sigma(M);
//     (mass_node)                w (rho_m/M) dlnnu/dlnM; M/rho_m; r_Delta;
//                                B(M); W_ejc(M)
//   per refill                   mu_p/mu_e; sigma8; the M_min pieces of the
//                                HMx terms
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min);
//                                Y(a); k_s(a)
//   per (i, q) (a_node[i])       nu = nu0/D; c = conc(M, D); ln(1+c);
//                                m(c); r_s = r_Delta/c, ln r_s;
//                                w1h_q = dn (M/rho_m)/m(c) Y B,
//                                w2hm_q = dn b (M/rho_m)/m(c),
//                                w2hy_q = dn b Y B, with
//                                dn = w (rho_m/M) dlnnu/dlnM f(nu) nu;
//                                sum_ejc = sum dn b W_ejc
//   per (i, k), sum over q       um = nfw_um(c, k r_s, ln k + ln r_s, ln(1+c))
//                                uy = u_KS(c, k, r_Delta);
//                                I02 = sum w1h_q uy um;
//                                I11_m = sum w2hm_q um + HMx;
//                                I11_y = sum w2hy_q uy + sum_ejc + HMx; ln P
//
// The 1/m(c) of u = um/m(c) lives in w1h_q and w2hm_q. The rows read the
// NFW kernel directly, so like.halo_model[3] must be HALO_PROFILE_NFW; the
// gas needs cosmology.Omega_b > 0 (f_bnd) and a polytropic index
// nuisance.gas[0] > 1 (u_KS); anything else aborts.
//
// Thread safety: the single-threaded halo_warmup(a_min, k_min, 1, 0) call
// before the threaded loop builds every lazy table the rows read (its
// header), so inside the loop they are only read.
//
// Cache invalidation:
//   rebuild block (table, mass_node, a_node, GL nodes, both grids; every
//     allocation lives here, one block each from malloc2d/malloc3d):
//     Ntable.random
//   refill: cosmology.random, Ntable.random or nuisance.random_gas
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_my(k, a) in U = G (M_sun/h)^2/(c/H0) (GAS PROFILES banner); 0 outside
//   [limits.a_min, 0.9999999]; ln P continued with unit slope outside
//   [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_my(
    const double k,
    const double a
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];  // tags the table was built from
  static double** table = NULL;            // ln P on the (a, ln k) grid
  static double lim[2][3];           // [0] a grid: min, max, step;
                                     // [1] ln k grid: min, max, step
  static int n_nodes = 0;            // Gauss-Legendre nodes in ln M
  static double** mass_node = NULL;  // [8][n_nodes] per mass node q:
                                     // 0 M, 1 GL weight w, 2 nu0 = nu(D=1),
                                     // 3 w (rho_m/M) dlnnu/dlnM, 4 M/rho_m,
                                     // 5 r_Delta, 6 B = f_bnd M^2/r_Delta,
                                     // 7 W_ejc
  static double*** a_node = NULL;    // [N_a][7][n_nodes] per (a row, node):
                                     // 0 c, 1 ln(1+c), 2 r_s, 3 ln r_s,
                                     // 4 w1h, 5 w2hm, 6 w2hy

  // --- 1. NTABLE REBUILD: TABLE, NODE ARRAYS, GL RULE, GRIDS ---
  // the table, the per-node and per-(a, node) arrays (one block each from
  // malloc2d/malloc3d, so one free each), the GL nodes mapped onto
  // [ln M_min, ln M_max], both grids (header, item 1)
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
      free(mass_node);
      free(a_node);
    }

    table     = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    // mass-node ladder: the default already lands far inside the
    // code's chi2 error budget (measured ladder: the skill file's
    // halo.c numerics); high_def_integration steps toward the largest
    // GSL rule
    if (0 == abs(Ntable.high_def_integration)) {
      n_nodes = Ntable.halo_nm;
    }
    else if (1 == abs(Ntable.high_def_integration)) {
      n_nodes = 2*Ntable.halo_nm;
    }
    else if (2 == abs(Ntable.high_def_integration)) {
      n_nodes = 4*Ntable.halo_nm;
    }
    else {
      n_nodes = 1024;
    }
    mass_node = (double**) malloc2d(8, n_nodes);
    a_node    = (double***) malloc3d(Ntable.N_a, 7, n_nodes);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    const double lnMmin = log(limits.halo_m[RANGE_MIN]);
    const double lnMmax = log(limits.halo_m[RANGE_MAX]);
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(n_nodes);
    for (int q=0; q<n_nodes; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM,
                                    &mass_node[1][q], gl_table);
      mass_node[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // the uniform table axes: [0] a, [1] ln k (min, max, step)
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;  // a_max, just below a = 1 (today)
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_cH0[RANGE_MIN]);
    lim[1][1] = log(limits.k_cH0[RANGE_MAX]);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL GUARD AND SINGLE-THREADED WARM-UP ---
  // refill when the cosmology, Ntable or gas tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  {
    // the k columns read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    // total-matter halos only: the 2-halo term I11_m I11_y P_lin has no
    // cb form (HALO MODEL POWER SPECTRA banner)
    if (like.halo_model[4] != HALO_FIELD_MATTER) {
      log_fatal("p_my: like.halo_model[4] = %d not supported (the "
                "matter-pressure spectrum needs the total-matter halo "
                "field)", like.halo_model[4]);
      exit(1);
    }

    // the gas: f_bnd carries Omega_b/Omega_m, u_KS the exponents
    // Gamma/(Gamma - 1) and 1/(Gamma - 1)
    if (!(cosmology.Omega_b > 0)) {
      log_fatal("Compton-y spectra need cosmology.Omega_b > 0 "
                "(set_cosmological_parameters)");
      exit(1);
    }

    if (!(nuisance.gas[0] > 1)) {
      log_fatal("Compton-y spectra need a polytropic index gas[0] = %g > 1",
                nuisance.gas[0]);
      exit(1);
    }

    // warm-up (header, Thread safety): every lazy table the threaded
    // loop reads is built here, on one thread
    halo_warmup(lim[0][0], exp(lim[1][0]), 1, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (full derivation: the header)
       1. node q: M_q, w_q, nu0_q; B = f_bnd M^2/r_Delta; W_ejc
       2. row i (threaded): nu = nu0_q/D(a); c = conc(M_q, D); Y(a); k_s(a);
          w1h_q = dn (M/rho_m)/m(c) Y B; w2hm_q = dn b (M/rho_m)/m(c);
          w2hy_q = dn b Y B; sum_ejc = sum_q dn b W_ejc
       3. column j: I02 = sum_q w1h_q uy um; I11_m = sum_q w2hm_q um + HMx;
          I11_y = sum_q w2hy_q uy + sum_ejc + HMx
       4. table = ln(I02 S + I11_m I11_y P_lin),
          S = x/(1 + x), x = (k/k_s)^4 */

    // --- 3. PER MASS NODE ---
    // quantities that depend on M alone (header, item 2, first row)
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;  // Delta x mean matter density

    for (int q=0; q<n_nodes; q++) {
      const double m = mass_node[0][q];
      mass_node[2][q] = delta_c/sqrt(sigma2(m));  // nu0 = delta_c/sigma(M)
      // GL weight x the (rho_m/M) dlnnu/dlnM of the mass function
      mass_node[3][q] = mass_node[1][q]*(rho_m/m)*dlognudlogm(m);
      mass_node[4][q] = m/rho_m;  // the matter window amplitude
      // r_Delta = (3 M/(4 pi rho_Delta))^(1/3), the halo edge
      mass_node[5][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mass_node[6][q] = frac_bnd(m)*m*(m/mass_node[5][q]);  // B(M)
      mass_node[7][q] = u_y_ejc(m);  // W_ejc(M), the ejected gas
    }

    // --- 4. GAS AND M_MIN CONSTANTS ---
    // mu_p, mu_e of the ionized gas (GAS PROFILES banner), sigma8 from
    // the sigma2 table at M8 = (4 pi/3) rho_m (8 Mpc/h)^3, and the M_min
    // pieces of the HMx terms (header, item 2, second row)
    const double mu_p = 4.0/(3.0 + 5*nuisance.gas[10]);  // 4/(3 + 5 f_H)
    const double mu_e = 2.0/(1.0 + nuisance.gas[10]);    // 2/(1 + f_H)

    const double R8     = 8.0/cosmology.coverH0;  // 8 Mpc/h in c/H0 units
    const double sigma8 = sqrt(sigma2(4.0*M_PI/3.0*rho_m*R8*R8*R8));

    // the M_min pieces: M_min/rho_m, r_Delta, B(M_min), W_ejc(M_min)
    const double m_min      = limits.halo_m[RANGE_MIN];
    const double vol_min    = m_min/rho_m;
    const double rdelta_min = pow(3./(4.0*M_PI)*(m_min/rho_delta), 1./3.);
    const double B_min      = frac_bnd(m_min)*m_min*(m_min/rdelta_min);
    const double Wejc_min   = u_y_ejc(m_min);

    // --- 5. PER a ROW, THREADED ---
    // each row: D(a), the Tinker f and b parameters, A(a), c(M_min),
    // Y(a) of the bound-gas window, k_s(a) of the damping (header,
    // item 2, third row)
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      const double ai = lim[0][0] + i*lim[0][2];
      const double D  = growfac(ai);

      // the nu-independent halves of Tinker f(nu) and b(nu) at this a
      const fnu_params   f_params = fnu_params_at(ai);
      const hb1nu_params b_params = hb1nu_params_at(ai);

      // A(a) = 1 - bias_norm(a): the HMx share of matter below M_min
      const double A_hmx    = 1.0 - bias_norm(ai);
      const double conc_min = conc(m_min, D);  // its c(M_min, a)

      // Y(a) = (2 alpha/(3 a)) mu_p/mu_e, alpha = nuisance.gas[5] (header)
      const double Y_gas = (2.0*nuisance.gas[5]/(3.0*ai))*(mu_p/mu_e);
      // k_s(a) = 0.05618 (sigma8 a)^-1.013 h/Mpc in (c/H0)^-1 (header)
      const double k_s = 0.05618/pow(sigma8*ai, 1.013)*cosmology.coverH0;

      // per-node rows of this a row (restrict: distinct rows)
      const double* restrict rdelta_q = mass_node[5];  // r_Delta(M_q)
      double* restrict conc_q = a_node[i][0];  // c(M_q, a)
      double* restrict ln1c_q = a_node[i][1];  // ln(1 + c)
      double* restrict rs_q   = a_node[i][2];  // r_s = r_Delta/c
      double* restrict lnrs_q = a_node[i][3];  // ln r_s
      double* restrict w1h_q  = a_node[i][4];  // 1-halo weight
      double* restrict w2hm_q = a_node[i][5];  // 2-halo matter weight
      double* restrict w2hy_q = a_node[i][6];  // 2-halo pressure weight

      // per (a, node): c(M, a), r_s and their logs, and the weights with
      // the 1/m(c) of u = um/m(c) folded into the matter legs (header,
      // item 2, fourth row); dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
      double sum_ejc = 0.0;  // sum_q dn b W_ejc: the k-independent
                             // ejected-gas share of I11_y
      for (int q=0; q<n_nodes; q++) {
        const double nu   = mass_node[2][q]/D;  // nu = nu0/D(a)
        const double c    = conc(mass_node[0][q], D);
        const double dn   = mass_node[3][q]*fnu_core(nu, &f_params)*nu;
        const double bias = hb1nu_core(nu, &b_params);
        const double YB   = Y_gas*mass_node[6][q];  // W_y without u_KS
        conc_q[q] = c;
        ln1c_q[q] = log1p(c);
        rs_q[q]   = rdelta_q[q]/c;  // r_s = r_Delta/c
        lnrs_q[q] = log(rs_q[q]);
        const double mc = ln1c_q[q] - c/(1.0 + c);  // NFW norm m(c)
        w1h_q[q]  = dn*(mass_node[4][q]/mc)*YB;
        w2hm_q[q] = dn*bias*(mass_node[4][q]/mc);
        w2hy_q[q] = dn*bias*YB;
        sum_ejc += dn*bias*mass_node[7][q];
      }

      // per k column: I02 and the two I11 as sums of the two kernels
      // over the nodes, the HMx terms at M_min, the damping S, then ln P
      // (header, item 2, last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lnk = lim[1][0] + j*lim[1][2];
        const double kj  = exp(lnk);

        double sum_I02  = 0.0;  // 1-halo: sum_q w1h_q uy um
        double sum_I11m = 0.0;  // 2-halo matter leg: sum_q w2hm_q um
        double sum_I11y = 0.0;  // 2-halo pressure leg: sum_q w2hy_q uy

#ifdef HALO_NOT_USE_SIMD
        for (int q=0; q<n_nodes; q++) {
          const double um = nfw_um(conc_q[q], kj*rs_q[q],
                                   lnk + lnrs_q[q], ln1c_q[q]);
          const double uy = u_KS(conc_q[q], kj, rdelta_q[q]);
          sum_I02  += w1h_q[q]*uy*um;
          sum_I11m += w2hm_q[q]*um;
          sum_I11y += w2hy_q[q]*uy;
        }
#else
        // the scalar loop above, four nodes q, q+1, q+2, q+3 per step
        // (one per lane, as in p_mm): the NFW leg through nfw_um4 (nfw_um
        // on each lane, bitwise), the pressure leg u_KS lane by lane;
        // four-lane partial sums added in a fixed lane order
        // (simd_horizontal_sum), then a scalar tail (summation order as
        // in p_mm)

        // k and ln k of this column in all four lanes
        const v4d vk   = simde_mm256_set1_pd(kj);   // k
        const v4d vlnk = simde_mm256_set1_pd(lnk);  // ln k

        // the four-lane partial sums, from (0, 0, 0, 0)
        v4d vsum_I02  = simde_mm256_setzero_pd();  // sum_I02
        v4d vsum_I11m = simde_mm256_setzero_pd();  // sum_I11m
        v4d vsum_I11y = simde_mm256_setzero_pd();  // sum_I11y

        int q = 0;
        for (; q<=n_nodes-4; q+=4) {
          // the four arguments of nfw_um at nodes q..q+3; scalar:
          //   nfw_um(conc_q[q], kj*rs_q[q], lnk + lnrs_q[q], ln1c_q[q])

          // c, the concentrations of nodes q..q+3
          const v4d vconc = simde_mm256_loadu_pd(conc_q + q);

          // r_s of nodes q..q+3
          const v4d vrs = simde_mm256_loadu_pd(rs_q + q);

          // x = k r_s
          const v4d vkrs = simde_mm256_mul_pd(vk, vrs);

          // ln r_s of nodes q..q+3
          const v4d vlnrs = simde_mm256_loadu_pd(lnrs_q + q);

          // ln x = ln k + ln r_s
          const v4d vlnkrs = simde_mm256_add_pd(vlnk, vlnrs);

          // ln(1 + c) of nodes q..q+3
          const v4d vln1c = simde_mm256_loadu_pd(ln1c_q + q);

          // um = u m(c) at nodes q..q+3
          const v4d vum = nfw_um4(vconc, vkrs, vlnkrs, vln1c);

          // scalar: uy = u_KS(conc_q[q], kj, rdelta_q[q]), one node at a
          // time (u_KS is a scalar table read), into a plain double[4]
          double uy[4];
          for (int lane=0; lane<4; lane++) {
            uy[lane] = u_KS(conc_q[q + lane], kj, rdelta_q[q + lane]);
          }

          // uy at nodes q..q+3 into the four lanes
          const v4d vuy = simde_mm256_loadu_pd(uy);

          // the weights of nodes q..q+3
          const v4d vw1h  = simde_mm256_loadu_pd(w1h_q + q);   // 1-halo
          const v4d vw2hm = simde_mm256_loadu_pd(w2hm_q + q);  // 2-halo m
          const v4d vw2hy = simde_mm256_loadu_pd(w2hy_q + q);  // 2-halo y

          // scalar: sum_I02 += w1h_q[q]*uy*um, as (w1h uy) um + sum, fused

          // w1h uy
          const v4d vw1h_uy = simde_mm256_mul_pd(vw1h, vuy);

          // (w1h uy) um + sum_I02, lane by lane
          vsum_I02 = nfw_fmadd4(vw1h_uy, vum, vsum_I02);

          // scalar: sum_I11m += w2hm_q[q]*um, fused
          vsum_I11m = nfw_fmadd4(vw2hm, vum, vsum_I11m);

          // scalar: sum_I11y += w2hy_q[q]*uy, fused
          vsum_I11y = nfw_fmadd4(vw2hy, vuy, vsum_I11y);
        }

        // lane 0 + lane 1 + lane 2 + lane 3 of each partial sum
        sum_I02  = simd_horizontal_sum(vsum_I02);   // sum_q w1h_q uy um
        sum_I11m = simd_horizontal_sum(vsum_I11m);  // sum_q w2hm_q um
        sum_I11y = simd_horizontal_sum(vsum_I11y);  // sum_q w2hy_q uy

        // scalar tail: n_nodes not a multiple of four
        for (; q<n_nodes; q++) {
          const double um = nfw_um(conc_q[q], kj*rs_q[q],
                                   lnk + lnrs_q[q], ln1c_q[q]);
          const double uy = u_KS(conc_q[q], kj, rdelta_q[q]);
          sum_I02  += w1h_q[q]*uy*um;
          sum_I11m += w2hm_q[q]*um;
          sum_I11y += w2hy_q[q]*uy;
        }
#endif

        // the damping S = x/(1 + x), x = (k/k_s)^4 (header)
        const double x4  = (kj/k_s)*(kj/k_s)*(kj/k_s)*(kj/k_s);
        const double P1H = sum_I02*(x4/(x4 + 1.0));

        const double I11m = sum_I11m + A_hmx*u_c(conc_min, kj, m_min, ai);

        // W_y + W_ejc at M_min, the window of the HMx term of I11_y
        const double W_min =
            Y_gas*B_min*u_KS(conc_min, kj, rdelta_min) + Wejc_min;
        const double I11y = sum_I11y + sum_ejc + A_hmx*W_min/vol_min;

        table[i][j] = log(P1H + I11m*I11y*p_lin(kj, ai));
      }
    }

    // stamp the table with the tags it was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }

  // --- 6. TABLE READ ---
  // bilinear read of ln P; 0 outside the a range
  if ((a < lim[0][0]) || (a > lim[0][1])) {
    return 0.0;
  }

  return exp(interpol2d(table,
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2],
                        log(k)));
}


// ---------------------------------------------------------------------------
// P_yy(k, a), the halo-model pressure auto spectrum, from a table of ln P
// on Ntable.N_a x Ntable.N_k_nlin nodes uniform in (a, ln k), read
// bilinearly (interpol2d) and exponentiated (section banner):
//
//   P_yy   = I02_yy S(k, a) + I11_y^2 P_lin              (2005.00009 Eqs. 1-2)
//   I02_yy = int dlnM dn/dlnM W_y(M, k)^2
//   I11_y  = int dlnM dn/dlnM b(nu) [W_y(M, k) + W_ejc(M)]
//            + A(a) [W_y(M_min, k) + W_ejc(M_min)]/(M_min/rho_m)
//   S      = x/(1 + x),  x = (k/k_s)^4                   (2009.01858 Eq. 17)
//
// W_y = Y(a) B(M) u_KS, W_ejc, k_s(a), dn/dlnM, b and A(a) as in p_my (its
// header). No matter leg: the only kernel is u_KS, and no M/rho_m enters
// (the pressure window is the full volume integral, GAS PROFILES banner).
//
// 1. Quadrature: the Gauss-Legendre rule of p_mm (its header,
// item 1) over [ln M_min, ln M_max], mapped once in the rebuild block.
//
// 2. Loop levels as in p_my (its header, item 2) without the matter leg:
//
//   per refill, per node q       M, w; nu0; w (rho_m/M) dlnnu/dlnM; r_Delta;
//     (mass_node)                B(M); W_ejc(M)
//   per refill                   mu_p/mu_e; sigma8; the M_min pieces
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min);
//                                Y(a); k_s(a)
//   per (i, q) (a_node[i])       nu = nu0/D; c = conc(M, D);
//                                w1h_q = dn (Y B)^2, w2hy_q = dn b Y B;
//                                sum_ejc = sum dn b W_ejc
//   per (i, k), sum over q       uy = u_KS(c, k, r_Delta);
//                                I02 = sum w1h_q uy^2; I11_y = sum w2hy_q uy
//                                + sum_ejc + HMx; ln P
//
// The gas needs cosmology.Omega_b > 0 (f_bnd) and a polytropic index
// nuisance.gas[0] > 1 (u_KS); anything else aborts.
//
// Thread safety: the single-threaded halo_warmup(a_min, k_min, 1, 0) call
// before the threaded loop builds every lazy table the rows read (its
// header), so inside the loop they are only read.
//
// Cache invalidation:
//   rebuild block (table, mass_node, a_node, GL nodes, both grids; every
//     allocation lives here, one block each from malloc2d/malloc3d):
//     Ntable.random
//   refill: cosmology.random, Ntable.random or nuisance.random_gas
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_yy(k, a) in U^2 (c/H0)^-3, U = G (M_sun/h)^2/(c/H0) (GAS PROFILES
//   banner); 0 outside [limits.a_min, 0.9999999]; ln P continued with unit
//   slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_yy(
    const double k,
    const double a
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];  // tags the table was built from
  static double** table = NULL;            // ln P on the (a, ln k) grid
  static double lim[2][3];           // [0] a grid: min, max, step;
                                     // [1] ln k grid: min, max, step
  static int n_nodes = 0;            // Gauss-Legendre nodes in ln M
  static double** mass_node = NULL;  // [7][n_nodes] per mass node q:
                                     // 0 M, 1 GL weight w, 2 nu0 = nu(D=1),
                                     // 3 w (rho_m/M) dlnnu/dlnM, 4 r_Delta,
                                     // 5 B = f_bnd M^2/r_Delta, 6 W_ejc
  static double*** a_node = NULL;    // [N_a][3][n_nodes] per (a row, node):
                                     // 0 c, 1 w1h, 2 w2hy

  // --- 1. NTABLE REBUILD: TABLE, NODE ARRAYS, GL RULE, GRIDS ---
  // the table, the per-node and per-(a, node) arrays (one block each from
  // malloc2d/malloc3d, so one free each), the GL nodes mapped onto
  // [ln M_min, ln M_max], both grids (header, item 1)
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
      free(mass_node);
      free(a_node);
    }

    table     = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    // mass-node ladder: the default already lands far inside the
    // code's chi2 error budget (measured ladder: the skill file's
    // halo.c numerics); high_def_integration steps toward the largest
    // GSL rule
    if (0 == abs(Ntable.high_def_integration)) {
      n_nodes = Ntable.halo_nm;
    }
    else if (1 == abs(Ntable.high_def_integration)) {
      n_nodes = 2*Ntable.halo_nm;
    }
    else if (2 == abs(Ntable.high_def_integration)) {
      n_nodes = 4*Ntable.halo_nm;
    }
    else {
      n_nodes = 1024;
    }
    mass_node = (double**) malloc2d(7, n_nodes);
    a_node    = (double***) malloc3d(Ntable.N_a, 3, n_nodes);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    const double lnMmin = log(limits.halo_m[RANGE_MIN]);
    const double lnMmax = log(limits.halo_m[RANGE_MAX]);
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(n_nodes);
    for (int q=0; q<n_nodes; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM,
                                    &mass_node[1][q], gl_table);
      mass_node[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // the uniform table axes: [0] a, [1] ln k (min, max, step)
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;  // a_max, just below a = 1 (today)
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_cH0[RANGE_MIN]);
    lim[1][1] = log(limits.k_cH0[RANGE_MAX]);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL GUARD AND SINGLE-THREADED WARM-UP ---
  // refill when the cosmology, Ntable or gas tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  {
    // total-matter halos only: the 2-halo term I11_y^2 P_lin has no cb
    // form (HALO MODEL POWER SPECTRA banner)
    if (like.halo_model[4] != HALO_FIELD_MATTER) {
      log_fatal("p_yy: like.halo_model[4] = %d not supported (the "
                "pressure spectrum needs the total-matter halo field)",
                like.halo_model[4]);
      exit(1);
    }

    // the gas: f_bnd carries Omega_b/Omega_m, u_KS the exponents
    // Gamma/(Gamma - 1) and 1/(Gamma - 1)
    if (!(cosmology.Omega_b > 0)) {
      log_fatal("Compton-y spectra need cosmology.Omega_b > 0 "
                "(set_cosmological_parameters)");
      exit(1);
    }

    if (!(nuisance.gas[0] > 1)) {
      log_fatal("Compton-y spectra need a polytropic index gas[0] = %g > 1",
                nuisance.gas[0]);
      exit(1);
    }

    // warm-up (header, Thread safety): every lazy table the threaded
    // loop reads is built here, on one thread
    halo_warmup(lim[0][0], exp(lim[1][0]), 1, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (full derivation: the header)
       1. node q: M_q, w_q, nu0_q; B = f_bnd M^2/r_Delta; W_ejc
       2. row i (threaded): nu = nu0_q/D(a); c = conc(M_q, D); Y(a); k_s(a);
          w1h_q = dn (Y B)^2; w2hy_q = dn b Y B; sum_ejc = sum_q dn b W_ejc
       3. column j: I02 = sum_q w1h_q uy^2;
          I11_y = sum_q w2hy_q uy + sum_ejc + HMx
       4. table = ln(I02 S + I11_y^2 P_lin),
          S = x/(1 + x), x = (k/k_s)^4 */

    // --- 3. PER MASS NODE ---
    // quantities that depend on M alone (header, item 2, first row)
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;  // Delta x mean matter density

    for (int q=0; q<n_nodes; q++) {
      const double m = mass_node[0][q];
      mass_node[2][q] = delta_c/sqrt(sigma2(m));  // nu0 = delta_c/sigma(M)
      // GL weight x the (rho_m/M) dlnnu/dlnM of the mass function
      mass_node[3][q] = mass_node[1][q]*(rho_m/m)*dlognudlogm(m);
      // r_Delta = (3 M/(4 pi rho_Delta))^(1/3), the halo edge
      mass_node[4][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mass_node[5][q] = frac_bnd(m)*m*(m/mass_node[4][q]);  // B(M)
      mass_node[6][q] = u_y_ejc(m);  // W_ejc(M), the ejected gas
    }

    // --- 4. GAS AND M_MIN CONSTANTS ---
    // mu_p, mu_e of the ionized gas (GAS PROFILES banner), sigma8 from
    // the sigma2 table at M8 = (4 pi/3) rho_m (8 Mpc/h)^3, and the M_min
    // pieces of the HMx term (header, item 2, second row)
    const double mu_p = 4.0/(3.0 + 5*nuisance.gas[10]);  // 4/(3 + 5 f_H)
    const double mu_e = 2.0/(1.0 + nuisance.gas[10]);    // 2/(1 + f_H)

    const double R8     = 8.0/cosmology.coverH0;  // 8 Mpc/h in c/H0 units
    const double sigma8 = sqrt(sigma2(4.0*M_PI/3.0*rho_m*R8*R8*R8));

    // the M_min pieces: M_min/rho_m, r_Delta, B(M_min), W_ejc(M_min)
    const double m_min      = limits.halo_m[RANGE_MIN];
    const double vol_min    = m_min/rho_m;
    const double rdelta_min = pow(3./(4.0*M_PI)*(m_min/rho_delta), 1./3.);
    const double B_min      = frac_bnd(m_min)*m_min*(m_min/rdelta_min);
    const double Wejc_min   = u_y_ejc(m_min);

    // --- 5. PER a ROW, THREADED ---
    // each row: D(a), the Tinker f and b parameters, A(a), c(M_min),
    // Y(a) of the bound-gas window, k_s(a) of the damping (header,
    // item 2, third row)
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      const double ai = lim[0][0] + i*lim[0][2];
      const double D  = growfac(ai);

      // the nu-independent halves of Tinker f(nu) and b(nu) at this a
      const fnu_params   f_params = fnu_params_at(ai);
      const hb1nu_params b_params = hb1nu_params_at(ai);

      // A(a) = 1 - bias_norm(a): the HMx share of matter below M_min
      const double A_hmx    = 1.0 - bias_norm(ai);
      const double conc_min = conc(m_min, D);  // its c(M_min, a)

      // Y(a) = (2 alpha/(3 a)) mu_p/mu_e, alpha = nuisance.gas[5] (header)
      const double Y_gas = (2.0*nuisance.gas[5]/(3.0*ai))*(mu_p/mu_e);
      // k_s(a) = 0.05618 (sigma8 a)^-1.013 h/Mpc in (c/H0)^-1 (header)
      const double k_s = 0.05618/pow(sigma8*ai, 1.013)*cosmology.coverH0;

      // per-node rows of this a row (restrict: distinct rows)
      const double* restrict rdelta_q = mass_node[4];  // r_Delta(M_q)
      double* restrict conc_q = a_node[i][0];  // c(M_q, a)
      double* restrict w1h_q  = a_node[i][1];  // 1-halo weight
      double* restrict w2hy_q = a_node[i][2];  // 2-halo pressure weight

      // per (a, node): c(M, a) and the weights (header, item 2, fourth
      // row); dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
      double sum_ejc = 0.0;  // sum_q dn b W_ejc: the k-independent
                             // ejected-gas share of I11_y
      for (int q=0; q<n_nodes; q++) {
        const double nu   = mass_node[2][q]/D;  // nu = nu0/D(a)
        const double dn   = mass_node[3][q]*fnu_core(nu, &f_params)*nu;
        const double bias = hb1nu_core(nu, &b_params);
        const double YB   = Y_gas*mass_node[5][q];  // W_y without u_KS
        conc_q[q] = conc(mass_node[0][q], D);
        w1h_q[q]  = dn*YB*YB;
        w2hy_q[q] = dn*bias*YB;
        sum_ejc += dn*bias*mass_node[6][q];
      }

      // per k column: I02 and I11_y as sums of the kernel over the
      // nodes, the HMx term at M_min, the damping S, then ln P (header,
      // item 2, last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lnk = lim[1][0] + j*lim[1][2];
        const double kj  = exp(lnk);

        double sum_I02  = 0.0;  // 1-halo: sum_q w1h_q uy^2
        double sum_I11y = 0.0;  // 2-halo: sum_q w2hy_q uy

        for (int q=0; q<n_nodes; q++) {
          const double uy = u_KS(conc_q[q], kj, rdelta_q[q]);
          sum_I02  += w1h_q[q]*uy*uy;
          sum_I11y += w2hy_q[q]*uy;
        }

        // the damping S = x/(1 + x), x = (k/k_s)^4 (header)
        const double x4  = (kj/k_s)*(kj/k_s)*(kj/k_s)*(kj/k_s);
        const double P1H = sum_I02*(x4/(x4 + 1.0));

        // W_y + W_ejc at M_min, the window of the HMx term of I11_y
        const double W_min =
            Y_gas*B_min*u_KS(conc_min, kj, rdelta_min) + Wejc_min;
        const double I11y = sum_I11y + sum_ejc + A_hmx*W_min/vol_min;

        table[i][j] = log(P1H + I11y*I11y*p_lin(kj, ai));
      }
    }

    // stamp the table with the tags it was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }

  // --- 6. TABLE READ ---
  // bilinear read of ln P; 0 outside the a range
  if ((a < lim[0][0]) || (a > lim[0][1])) {
    return 0.0;
  }

  return exp(interpol2d(table,
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2],
                        log(k)));
}
