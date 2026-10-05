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
//   1. paste the two functions back into halo.c, in the HALO MODEL POWER
//      SPECTRA section (p_mm, their template, is in halo_pmm.c here);
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

        // Evaluate the weighted sums above at four nodes per step
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


// ============================================================================
// The GAS PROFILES section of halo.c, kept here with the spectra it serves.
// To restore: paste it back into halo.c before HALO MODEL ROUTINES, declare
// u_KS, frac_bnd, frac_ejc and u_y_ejc in halo.h, restore the gas branch of
// halo_warmup (one u_KS read at M_min, r_v = r_Delta(M_min)), the u_KS and
// set_nuisance_gas wrappers and bindings (halo_tsz_wrapper.cpp) and the tests
// (halo_tsz_tests.py). The struct fields it reads (nuisance.gas,
// nuisance.random_gas, limits.halo_uks_c, Ntable.halo_uks_n/m) stay in
// structs.h.
// ============================================================================
// ============================================================================
// [SECTION] GAS PROFILES
// ============================================================================
//
// The electron-pressure (thermal SZ) side of the halo model, after the
// HMx model of Mead et al. 2020 (2005.00009 secs. 3.2-3.3). The baryons
// that belong to a halo of mass M split into three parts:
//
//   f_bnd(M) = gas bound inside r_Delta, in hydrostatic
//              equilibrium, Komatsu-Seljak profile        -> frac_bnd
//   f_*(M)   = stars (a local of frac_ejc)
//   f_ejc(M) = gas ejected beyond r_Delta,
//              Omega_b/Omega_m - f_bnd - f_*               -> frac_ejc
//
// The bound gas enters both halo terms through its pressure window
// W_p; the ejected gas is a smooth, warm component that enters
// the 2-halo term only (u_y_ejc).
//
// Both windows are volume integrals of the electron pressure, i.e.
// energies, in units of U = G (M_sun/h)^2/(c/H0). No factor
// sigma_T/(m_e c^2) is applied: the "y" functions below return pressure
// windows, not Compton-y.
//
// The gas parameters, nuisance.gas[0..10] (structs.h):
//
//   [0]  = Gamma       polytropic index of the bound gas, > 1
//   [1]  = beta        mass slope of f_bnd
//   [2]  = log10 M_0   mass at which halos keep half their gas bound
//   [3]  = eps1        (not read in this file)
//   [4]  = eps2        (not read in this file)
//   [5]  = alpha       bound-gas temperature in units of T_v
//   [6]  = A_*         peak stellar fraction
//   [7]  = log10 M_*   mass of that peak
//   [8]  = sigma_*     width of the stellar peak in log10 M
//   [9]  = log10 T_w   temperature of the ejected gas in K
//   [10] = f_H         hydrogen mass fraction
//
// Electron-pressure window of the bound gas: the Fourier-weighted volume
// integral of the pressure (2005.00009 Eq. 4 with the pressure profile),
//
//   W_p(M, k) = int_0^{r_Delta} 4 pi r^2 [sin(kr)/(kr)] P_e(r) dr
//
// Derivation, chaining 2005.00009 Eqs. 40, 38, 39 and 13:
//
//   P_e = n_e k_B T_g,  n_e = rho_bnd/(m_p mu_e),  T_g = T_v theta
//     ->  W_p = [k_B T_v/(m_p mu_e)] f_bnd M u_KS
//   (3/2) k_B T_v = alpha G M m_p mu_p/(a r_v)
//     ->  W_p = (2 alpha/(3a)) (mu_p/mu_e) f_bnd (G M^2/r_v) u_KS
//
// The value comes back with G left out, i.e. in units of
// U = G (M_sun/h)^2/(c/H0): an energy (pressure times volume). The a
// turns the comoving r_v into the physical radius. At k -> 0,
// W_p ~ f_bnd M^(5/3) (2005.00009 Eq. 41): gas mass times a virial
// temperature ~ M/r_v ~ M^(2/3).
//
// Mean particle masses of a fully ionized hydrogen-helium gas with
// hydrogen mass fraction f_H (2005.00009, footnote to Eq. 40): per
// proton mass there are 2 f_H + 3(1 - f_H)/4 particles and
// f_H + (1 - f_H)/2 electrons, hence
//
//   mu_p = 4/(3 + 5 f_H),   mu_e = 2/(1 + f_H)
//
// r_v is r_Delta of this file (Delta = 200 times the mean density);
// 2005.00009 uses the virial radius (its Eq. 22), and the free alpha
// absorbs the difference, so its fitted alpha does not carry over.
//
// The pressure spectra (p_my, p_yy; future_port_unfinished/halo_tsz.c)
// evaluate W_p as Y(a) B(M) u_KS(c, k, r_Delta), with
// Y = (2 alpha/(3a)) mu_p/mu_e and B = f_bnd M^2/r_Delta.
//
// Masses in M_sun/h.
// ============================================================================


// ---------------------------------------------------------------------------
// Komatsu-Seljak profile shape and the integrand of the u_KS contour
// integrals (u_KS header) at complex radius x = r/r_s:
//
//   theta(x) = ln(1 + x)/x,   g(x) = x theta(x)^p,   p = Gamma/(Gamma - 1).
//
// theta is the gas temperature in units of the central one, T_g/T_v:
// for a polytrope (P proportional to rho^Gamma) in hydrostatic
// equilibrium, T_g is a linear function of the potential, and the NFW
// potential is proportional to ln(1 + x)/x (2005.00009 sec. 3.2, the
// rho_bnd equation, after Komatsu & Seljak 2001). rho_bnd = theta^q and
// P_e = theta^p (in central units) follow from P ~ rho^Gamma and
// P ~ rho T. On the real axis x >= 0, theta falls from 1 at the
// centre to ln(1 + c)/c at the edge.
//
// Why complex x: u_KS traces the Fourier integral of theta^p off the
// real axis onto the rays x = i tau and x = c + i tau (tau >= 0), where
// e^{iyx} stops oscillating. Both rays lie in Re x >= 0, so the branch
// cut of ln(1 + x), the real axis left of x = -1, is never approached,
// and theta is analytic on and between the rays (needed by the Cauchy
// argument of the u_KS header).
// ---------------------------------------------------------------------------
static inline double complex ks_ctheta(
    const double complex x  // complex radius r/r_s, Re x > -1
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. theta(x) = ln(1 + x)/x, a 0/0 at x = 0 (theta(0) = 1); the
        u_KS rays start at x = 0 and x = c, and the ln tau grid of P
        reaches far below |x| = 1, so tiny |x| is a normal input
     2. |x| < X_TAYLOR: the Taylor series ln(1 + x)/x = 1 - x/2 +
        x^2/3 - x^3/4 + x^4/5 - ... takes over (the dropped term,
        |x|^5/6, is ~1e-21 there)
     3. else ln1p = ln(1 + x), built from its two parts:
        Re ln(1 + x) = ln|1 + x| = (1/2) log1p(2 Re x + |x|^2)
        Im ln(1 + x) = arg(1 + x) = atan2(Im x, 1 + Re x)  in (-pi, pi]
        log1p of the small quantity |1 + x|^2 - 1 = 2 Re x + |x|^2 keeps
        the real part accurate for |x| << 1 (on the ray x = i tau it is
        log1p(tau^2)/2, where ln(1 + tau^2) would lose digits) */

  // below this |x| the series replaces the 0/0 form ln(1 + x)/x
  const double X_TAYLOR = 1e-4;

  if (cabs(x) < X_TAYLOR) {
    return 1.0 - x/2.0 + x*x/3.0 - x*x*x/4.0 + x*x*x*x/5.0;
  }

  const double x_re = creal(x);
  const double x_im = cimag(x);

  const double complex ln1p = 0.5*log1p(2.0*x_re + x_re*x_re + x_im*x_im) +
                              I*atan2(x_im, 1.0 + x_re);
  return ln1p/x;
}


static inline double complex ks_cg(
    const double complex x, // complex radius r/r_s
    const double p          // Gamma/(Gamma - 1)
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. g(x) = x theta(x)^p is the integrand of the pressure transform
        F (u_KS header): x^2 theta^p sin(yx)/(yx) = g(x) sin(yx)/y
     2. p is not an integer, so theta^p means exp(p ln theta) and needs
        one branch of ln theta on the whole closed region the u_KS
        contour encloses (Re x >= 0, Im x >= 0). The principal branch
        serves, because Re theta > 0 there:
          Re theta = [ln|1 + x| Re x + arg(1 + x) Im x]/|x|^2,
        and both terms are >= 0 in that quadrant (|1 + x| >= 1 and
        0 <= arg(1 + x) < pi/2), vanishing together only at x = 0,
        where theta = 1. So |arg theta| < pi/2, theta never crosses the
        cut of clog (the negative real axis), and g is one analytic
        function on and between the rays x = i tau and x = c + i tau */
  return x*cexp(p*clog(ks_ctheta(x)));
}


// ---------------------------------------------------------------------------
// Natural cubic spline through the n_coarse coarse values y_coarse
// (uniform nodes, spacing h_coarse), evaluated on the dense grid that
// splits every interval into m: (n_coarse - 1) m + 1 nodes sharing both
// ends with the coarse grid. On interval j, for 0 <= t <= h_coarse,
//
//   S(x_j + t) = y_j + t (b + t (c_j + t d)),   c_j = S''(x_j)/2,
//   b = (y_{j+1} - y_j)/h_coarse - h_coarse (c_{j+1} + 2 c_j)/3,
//   d = (c_{j+1} - c_j)/(3 h_coarse),
//
// with c = 0 at both ends (natural). In the code c_coef[j] = c_j and
// y_dense[j m + r] = S(x_j + r h_coarse/m); r = 0 returns y_j exactly.
// u_KS upsamples its three 1D tables (ln P, ln F0, ln g) with it: the
// coarse values are exact but costly, the dense grid is what the
// lookup reads linearly, and the natural end condition is harmless
// because the used range stops PAD coarse nodes short of both ends.
// ---------------------------------------------------------------------------
static void ks_upsample1d(
    const double* y_coarse, // coarse values
    const int n_coarse,     // coarse nodes
    const double h_coarse,  // coarse spacing
    double* c_coef,         // workspace [n_coarse]: spline c coefficients
    double* y_dense,        // output [(n_coarse - 1) m + 1]
    const int m             // refinement factor
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. c_j = S''(x_j)/2 at the coarse nodes (natural: c = 0 at ends)
     2. interval j: b and d of the header, then
        S(x_j + t) = y_j + t (b + t (c_j + t d)) at t = r h_coarse/m */

  spline_coeffs_uniform(y_coarse, n_coarse, h_coarse, c_coef);

  for (int j=0; j<n_coarse-1; j++) {
    // linear (b) and cubic (d) coefficients of interval j
    const double b = (y_coarse[j+1] - y_coarse[j])/h_coarse -
                     h_coarse*(c_coef[j+1] + 2.0*c_coef[j])/3.0;
    const double d = (c_coef[j+1] - c_coef[j])/(3.0*h_coarse);

    // S at the m dense nodes t = r h_coarse/m of this interval
    for (int r=0; r<m; r++) {
      const double t = h_coarse*((double) r)/((double) m);
      y_dense[j*m + r] = y_coarse[j] + t*(b + t*(c_coef[j] + t*d));
    }
  }

  // last coarse node: no interval starts there
  y_dense[(n_coarse - 1)*m] = y_coarse[n_coarse - 1];
}


// ---------------------------------------------------------------------------
// Shape factor of the bound-gas pressure window:
//
//   u_KS(c, k, r_v) = F(c, y)/F0(c),   y = k r_v/c = k r_s,
//
//   F0(c)    = int_0^c x^2 theta(x)^q dx             (bound-gas mass)
//   F(c, y)  = int_0^c x sin(y x)/y theta(x)^p dx    (pressure transform)
//   theta(x) = ln(1 + x)/x,   p = Gamma/(Gamma - 1),   q = 1/(Gamma - 1),
//
// with x = r/r_s the radius in units of the NFW scale radius. theta^q
// is the Komatsu-Seljak ("KS") density profile of gas in hydrostatic
// equilibrium inside an NFW halo, theta^p its pressure profile and
// Gamma = nuisance.gas[0] its polytropic index (2005.00009 sec. 3.2,
// the rho_bnd equation; Komatsu & Seljak 2001).
//
// 0. From the pressure window to u_KS. The window is the Fourier-
// weighted volume integral of the electron pressure (GAS PROFILES
// banner),
//
//   W_p(M, k) = int_0^{r_v} 4 pi r^2 [sin(kr)/(kr)] P_e(r) dr,
//
// with P_e = n_e k_B T_g, n_e = rho_bnd/(m_p mu_e), rho_bnd = rho_0
// theta^q and T_g = T_v theta (the KS solution: the temperature tracks
// theta, the density its power 1/(Gamma - 1), ks_ctheta). Together
// P_e = [rho_0 k_B T_v/(m_p mu_e)] theta^{q+1}, and q + 1 = p. rho_0
// is fixed by the bound-gas mass,
//
//   f_bnd M = int_0^{r_v} 4 pi r^2 rho_bnd dr = 4 pi rho_0 r_s^3 F0(c),
//
// after r = r_s x, r_v = r_s c. The same substitution in W_p, with
// kr = y x and x^2 sin(yx)/(yx) = x sin(yx)/y, gives 4 pi rho_0 r_s^3
// F(c, y) times the pressure prefactor, so
//
//   W_p(M, k) = [k_B T_v f_bnd M/(m_p mu_e)] F/F0
//             = [k_B T_v f_bnd M/(m_p mu_e)] u_KS:
//
// the gas mass, times k_B T_v per unit electron mass, times the shape
// factor; rho_0 and r_s^3 cancel between F and F0. Unlike the matter
// u(k|M), u_KS does not tend to 1 at k -> 0: u_KS(c, 0) = <T_g>/T_v
// < 1, the mass-weighted gas temperature in units of the central one
// (theta(0) = 1), and |u_KS(c, k)| <= u_KS(c, 0).
//
// Two phases appear below: y = k r_s, the argument of the integral, and
// z = y c = k r_v, the phase at the outer edge x = c. The profile is
// cut off sharply at r_v, and a sharp edge in real space rings in
// Fourier space: the sin(y x) makes u oscillate like cos z under a
// slowly falling envelope out to very large z, far too many zero
// crossings for a table of u itself. So u is tabulated directly only
// below the switch phase ZSW (item 3); above, the oscillation is taken
// out of the integral analytically and put back exactly at lookup
// (items 1 and 2).
//
// 1. The contour formula (z >= ZSW). Write the sine as the imaginary
// part of a complex exponential: with g(x) = x theta(x)^p (ks_cg),
//
//   F = Im J/y,    J(c, y) = int_0^c g(x) e^{i y x} dx.
//
// g is analytic on the closed quadrant Re x >= 0, Im x >= 0 (the branch
// cut of ln(1 + x) runs along x < -1, and theta^p keeps its principal
// branch there, ks_cg). Cauchy's theorem on the rectangle
//
//       i T -------------- c + i T    top edge: |e^{iyx}| = e^{-yT} -> 0
//        ^                   ^        as T -> inf, so it drops out
//        |                   |
//        0 ---------------> c        the wanted path
//
// says the path 0 -> c equals the ray 0 -> i inf minus the ray
// c -> c + i inf. On both rays e^{iyx} is the real, decaying
// e^{-y tau}: nothing oscillates, and the whole phase sits in the
// single factor e^{iyc} = e^{iz} of the second ray:
//
//   ray 0:  x = i tau,     dx = i dtau:
//           i int_0^inf g(i tau) e^{-y tau} dtau           = i I0(y)
//   ray c:  x = c + i tau, dx = i dtau:
//           i e^{iz} int_0^inf g(c + i tau) e^{-y tau} dtau.
//
// On the second ray tau = c t (t is the height in units of c, so the
// same t window serves every c), c = z/y and g(c) is pulled out:
//
//   J = i I0(y) - e^{iz} (i g(c)/y) Q(c, z),
//
//   I0(y)   = int_0^inf g(i tau) e^{-y tau} dtau,     P(y) = Re I0(y),
//   Q(c, z) = z int_0^inf [g(c + i c t)/g(c)] e^{-z t} dt.
//
// Taking the imaginary part, Im[i I0] = Re I0 = P and Im[i e^{iz} Q] =
// Re[e^{iz} Q] = cos z Re Q - sin z Im Q, and dividing by y F0,
//
//   u = [P(y) - (g(c)/y) (cos z Re Q - sin z Im Q)]/(y F0(c)).
//
// What each piece means:
//
// - P comes from the ray anchored at the centre x = 0. P/y is the
//   transform of the untruncated profile (send c -> inf: the second
//   ray drops out, g(c) -> 0) - the smooth part of u.
//
// - Q comes from the ray anchored at the edge x = c: the correction
//   for cutting the profile at r_v, which carries the ringing.
//
// - Q -> 1 at large z. Its weight z e^{-zt} has unit integral and
//   averages the ratio g(c + i c t)/g(c) over t up to ~1/z, where the
//   ratio is near 1. The ringing then tends to -g(c) cos z/(y^2 F0):
//   the pressure at the edge times the phase at the edge.
//
// - At large y, g(i tau) = i tau + p tau^2/2 + ... (theta = 1 - x/2 +
//   ... at small x, ks_ctheta), so P -> p/y^3: the P term of u falls
//   like 1/y^4, against the 1/y^2 of the Q term.
//
// - P, Q, g and F0 are smooth and tabulated; cos z and sin z are exact
//   at lookup. Above ZHI, the top of the ln z axis, Q is held at its
//   top value, which is 1 to O(1/ZHI).
//
// Why not use the contour formula down to z = 0: as z -> 0 the weight
// z e^{-zt} spreads out to t ~ 1/z, the two ray integrals grow and
// nearly cancel in u, and neither fits a fixed window in ln t. The
// direct table of item 3 covers that end; ZSW is the phase where the
// two meet.
//
// 2. Q and P by the trapezoid rule in s = ln t (t = e^s, dt = t ds, and
// tau = t/y in P):
//
//   Q(c, z) = z int [g(c + i c t)/g(c)] t e^{-z t} ds,
//   P(y)    = (1/y) int Re g(i t/y) t e^{-t} ds.
//
// Each integrand is one smooth bump (a power of e^s toward s -> -inf,
// like exp(-e^s) toward +inf), on which the trapezoid rule converges
// exponentially in the step h (the classic result for integrands
// analytic in a strip around the real s axis); the end weights need
// no halving because the integrand vanishes at both ends of the
// window. In P the cut-off e^{-t} is the same for every y, so one s
// window serves all y; with the ln y spacing hy = h/rP, rP an integer,
// every t_k/y_j is a node of one grid in ln tau, so g(i tau) is
// evaluated once per node rather than once per (k, j) pair. In Q the
// cut-off e^{-zt} moves with z, but z >= ZSW keeps it inside the same
// window.
//
// 3. Small z (z < ZSW): u tabulated directly on (ln c, w = z^2).
// With x = c s (the c^3 of both integrals cancels),
//
//   u(c, z) = int_0^1 s sin(z s)/z theta(c s)^p ds
//             / int_0^1 s^2 theta(c s)^q ds,
//
// two Gauss-Legendre integrals on [0, 1]. sin(z s)/z = s - z^2 s^3/6 +
// ... is even in z, so u is an analytic function of w = z^2 and a
// straight line in w near z = 0 (u0 - a w + ...); tabulating in w
// rather than z gives the spline a smooth function through z = 0. At
// the padding nodes w < 0, z = i kappa with kappa = sqrt(-w), and
// sin(z s)/z = sinh(kappa s)/kappa: the same analytic function,
// continued to negative w.
//
// 4. Tables. Each smooth ingredient is computed exactly on a coarse
// uniform grid, upsampled by a natural cubic spline onto a dense grid
// sharing its ends, and read from the dense grid by linear
// interpolation (exact values are expensive, the spline makes them
// dense, the linear read is a direct index):
//
//   quantity        axes          coarse               dense
//   u (item 3)      ln c, w       u_coarse[i][j]       u_dense
//   Re Q, Im Q      ln c, ln z    Q_coarse[0|1][i][j]  Q_dense[0|1]
//   ln P            ln y          lnP_coarse[j]        lnP_dense
//   ln F0, ln g     ln c (1D)     lnF0g_coarse[0|1]    lnF0g_dense[0|1]
//
// Why these axes: u depends on (c, z) only (item 3); Q on (c, z) by
// its definition; P on y alone, which is what makes it a 1D table; g
// and F0 on c alone. Logs of c, z and y because each spans decades;
// ln P, ln F0 and ln g because the logs of these positive, power-law-
// like quantities (P -> p/y^3) are gentler curves for the spline than
// the quantities themselves.
//
// Used ranges: ln c in [ln limits.halo_uks_c[RANGE_MIN], ln limits.halo_uks_c[RANGE_MAX]]
// (a query outside is clamped to the edge), w in [0, ZSW^2], ln z in
// [ln ZSW, ln ZHI], ln y in [ln(ZSW/cmax), ln(ZHI/cmin)] (y = z/c at
// the corners of the (c, z) range). The coarse node counts of ln c and
// ln z are Ntable.halo_uks_n[UKS_N_LNC] and Ntable.halo_uks_n[UKS_N_LNZ] (scaled by
// init_accuracy_boost); the others follow. Each used end gets PAD
// extra coarse nodes (a natural spline sets S'' = 0 at its ends; the
// lookups clamp to the used range, so the padding is never read),
// except the top of ln z, where Q is flat to O(1/ZHI). Dense counts are
// (coarse - 1) m + 1, m the refinement factor of that axis.
//
// Accuracy: u to 5e-6 of its local envelope, and to 2e-5 relative where
// |u| > 1e-2, at init_accuracy_boost = 1.
//
// Cache invalidation:
//   Ntable.random rebuilds the allocation, axes, nodes and weights;
//   nuisance.random_gas (Gamma) or Ntable.random refills the tables.
//
// Parameters:
//   c  - concentration r_Delta/r_s
//   k  - wavenumber in (c/H0)^-1
//   rv - halo radius r_Delta in c/H0 (comoving)
//
// Returns:
//   u_KS, dimensionless, with u_KS(c, 0) < 1
// ---------------------------------------------------------------------------
double u_KS(
    double c,        // concentration r_Delta/r_s
    double k,        // wavenumber in (c/H0)^-1
    const double rv  // halo radius r_Delta in c/H0 (comoving)
  )
{
  // --- 1. CONFIGURATION ---
  // ZSW is the switch phase z = k r_v: below it the direct table of
  // u(ln c, w = z^2) (header item 3), at and above it the contour
  // formula (item 1). ZHI is the top of the ln z axis of Q, above
  // which Q is held at its top value. PAD is the number of coarse
  // padding nodes beyond each used end; MC..M1 are the refinement
  // factors m of header item 4 for the five axes ln c, w, ln z, ln y,
  // ln c (1D).
  const double ZSW = 3.0;    // z = k r_v below: table of u(ln c, z^2)
  const double ZHI = 2.5e5;  // top of the ln z axis of Q
  const int PAD = Ntable.halo_spline_pad; // coarse padding beyond ends
  const int MC  = Ntable.halo_uks_m[UKS_M_LNC2D];     // dense refinement factors
  const int MW  = Ntable.halo_uks_m[UKS_M_W];
  const int MZ  = Ntable.halo_uks_m[UKS_M_LNZ];
  const int MY  = Ntable.halo_uks_m[UKS_M_LNY];
  const int M1  = Ntable.halo_uks_m[UKS_M_LNC1D];

  // --- 2. STATIC STATE ---
  // Built on the first call (u_dense == NULL). Suffix "p" = padded
  // coarse count, "d" = dense count; lim[a] = {first node, last node,
  // spacing} of dense axis a (coarse spacing = lim[a][2] times m).
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static int ncp, nwp, nzp, nyp, n1p;  // padded coarse sizes
  static int ncd, nwd, nzd, nyd, n1d;  // dense sizes (shared endpoints)
  static int ngl;   // Gauss-Legendre nodes s_q on [0, 1]
  static int nt;    // trapezoid nodes t_k in s = ln t
  static int rP;    // hs/hy: trapezoid steps per ln y step
  static int ntau;  // nodes tau_m of the shared ln tau grid
  static double** u_dense = NULL;      // [ncd][nwd] u(ln c, w)
  static double*** Q_dense = NULL;     // [2][ncd][nzd] Re Q, Im Q
                                       // (ln c, ln z)
  static double* lnP_dense = NULL;     // [nyd] ln P(ln y)
  static double** lnF0g_dense = NULL;  // [2][n1d] ln F0, ln g (ln c)
  static double lim[5][3];             // dense axes (padded extents):
                                       // ln c, w, ln z, ln y, ln c (1D)
  static double** u_coarse = NULL;     // coarse exact values, same layouts
  static double*** Q_coarse = NULL;
  static double* lnP_coarse = NULL;
  static double** lnF0g_coarse = NULL;
  static double** spline_ws = NULL;    // [3][max(nyp, n1p)] workspaces
  static double** gl = NULL;           // [2][ngl] GL nodes s_q, weights w_q
  static double** sin_kern = NULL;     // [ngl][nwp] s_q sin(z_j s_q)/z_j
                                       // (sinh at w < 0)
  static double** Qwgt = NULL;         // [nt][nzp] z_j h t_k e^{-z_j t_k}
  static double** trap = NULL;         // [2][nt] t_k = e^{s_k},
                                       // h t_k e^{-t_k}
  static double** tau_g = NULL;        // [2][ntau] tau_m, Re g(i tau_m)

  // --- 3. NTABLE REBUILD ---
  // Sizes, dense axes, allocation, and the Gamma-independent quadrature
  // nodes and weights.
  if (NULL == u_dense || fdiff2(cache[1], Ntable.random)) {
    if (u_dense != NULL) {
      free(u_dense);
      free(Q_dense);
      free(lnP_dense);
      free(lnF0g_dense);
      free(u_coarse);
      free(Q_coarse);
      free(lnP_coarse);
      free(lnF0g_coarse);
      free(spline_ws);
      free(gl);
      free(sin_kern);
      free(trap);
      free(Qwgt);
      free(tau_g);
    }

    // Used ranges of ln c and ln y (header, item 4): y = z/c is smallest
    // at z = ZSW, c = cmax and largest at z = ZHI, c = cmin.
    const double lnc0 = log(limits.halo_uks_c[RANGE_MIN]);
    const double lnc1 = log(limits.halo_uks_c[RANGE_MAX]);
    const double lny0 = log(ZSW/limits.halo_uks_c[RANGE_MAX]);
    const double lny1 = log(ZHI/limits.halo_uks_c[RANGE_MIN]);

    // Coarse spacings hc (ln c) and hz (ln z) from the two knobs. hs is
    // the trapezoid step h of header item 2, halved for high-def
    // integration, and hy = hs/rP with rP an integer puts every t_k/y_j
    // on one ln tau grid (rP is the smallest integer that makes the
    // ln y axis at least as fine as the ln z axis). NW, NY and N1 are
    // the coarse counts of w, ln y and the 1D ln c axis; NY is whatever
    // the spacing hy needs to cover the used ln y range.
    const int NC = Ntable.halo_uks_n[UKS_N_LNC];
    const int NZ = Ntable.halo_uks_n[UKS_N_LNZ];
    const double hc = (lnc1 - lnc0)/((double) NC - 1.0);
    const double hz = (log(ZHI) - log(ZSW))/((double) NZ - 1.0);
    const int hdi = abs(Ntable.high_def_integration);  // accuracy knob

    double hs = 0.2;  // trapezoid step
    if (hdi >= 2) {
      hs = 0.1;
    }
    rP = (int) ceil(hs/hz);
    const double hy = hs/rP;

    const int NW = (int) ceil(6.0*NZ/64.0);  // w axis scales with ln z
    const int NY = (int) ceil((lny1 - lny0)/hy) + 1;
    const int N1 = (int) ceil(1.5*NC);       // denser 1D ln c axis
    const double hw = ZSW*ZSW/((double) NW - 1.0);
    const double h1 = (lnc1 - lnc0)/((double) N1 - 1.0);

    // Padded coarse counts (PAD nodes beyond each used end) and dense
    // counts sharing their endpoints: m dense steps per coarse interval.
    ncp = NC + 2*PAD;
    nwp = NW + 2*PAD;
    nzp = NZ + PAD;  // ln z is padded below only
    nyp = NY + 2*PAD;
    n1p = N1 + 2*PAD;
    ncd = (ncp - 1)*MC + 1;
    nwd = (nwp - 1)*MW + 1;
    nzd = (nzp - 1)*MZ + 1;
    nyd = (nyp - 1)*MY + 1;
    n1d = (n1p - 1)*M1 + 1;

    // Quadrature sizes from hdi: ngl Gauss-Legendre nodes for the [0, 1]
    // integrals of u and F0 (header, item 3); nt trapezoid nodes at the
    // step hs on the window [smin, smax] in s = ln t for Q and P (item
    // 2); ntau nodes of the shared ln tau grid of P. The window holds
    // the whole bump of both integrands: below smin the integrand is a
    // vanishing power of e^s, above smax the cut-off exp(-e^s) has
    // killed it (for Q the cut-off exp(-z e^s) is even earlier, z >=
    // ZSW); the trapezoid sums then need no end corrections.
    switch (hdi) {  // predefined GSL table sizes
      case 0:
        ngl = 96;
        break;
      case 1:
        ngl = 128;
        break;
      case 2:
        ngl = 256;
        break;
      case 3:
        ngl = 512;
        break;
      default:
        ngl = 1024;
        break;
    }

    double smin = -32.0;  // s = ln tau window
    if (hdi != 0) {
      smin = -40.0;
    }
    const double smax = 4.0;
    nt = (int) lround((smax - smin)/hs) + 1;
    ntau = (nt - 1)*rP + nyp;

    // Allocation; spline_ws holds one workspace per 1D upsampling job,
    // sized for the longer of the ln y and 1D ln c axes.
    int n_work = n1p;
    if (nyp > n1p) {
      n_work = nyp;
    }
    u_dense      = (double**) malloc2d(ncd, nwd);
    Q_dense      = (double***) malloc3d(2, ncd, nzd);
    lnP_dense    = (double*) malloc1d(nyd);
    lnF0g_dense  = (double**) malloc2d(2, n1d);
    u_coarse     = (double**) malloc2d(ncp, nwp);
    Q_coarse     = (double***) malloc3d(2, ncp, nzp);
    lnP_coarse   = (double*) malloc1d(nyp);
    lnF0g_coarse = (double**) malloc2d(2, n1p);
    spline_ws    = (double**) malloc2d(3, n_work);
    gl           = (double**) malloc2d(2, ngl);
    sin_kern     = (double**) malloc2d(ngl, nwp);
    trap         = (double**) malloc2d(2, nt);
    Qwgt         = (double**) malloc2d(nt, nzp);
    tau_g        = (double**) malloc2d(2, ntau);

    // Dense axes: first node = used start minus PAD coarse spacings,
    // spacing = coarse spacing/m. The w axis starts below w = 0
    // (header, item 3).
    lim[0][0] = lnc0 - PAD*hc;      // axis 0: ln c
    lim[0][2] = hc/MC;
    lim[1][0] = -PAD*hw;            // axis 1: w
    lim[1][2] = hw/MW;
    lim[2][0] = log(ZSW) - PAD*hz;  // axis 2: ln z
    lim[2][2] = hz/MZ;
    lim[3][0] = lny0 - PAD*hy;      // axis 3: ln y
    lim[3][2] = hy/MY;
    lim[4][0] = lnc0 - PAD*h1;      // axis 4: ln c (1D)
    lim[4][2] = h1/M1;

    // last node = first + (count - 1) spacings
    lim[0][1] = lim[0][0] + (ncd - 1)*lim[0][2];
    lim[1][1] = lim[1][0] + (nwd - 1)*lim[1][2];
    lim[2][1] = lim[2][0] + (nzd - 1)*lim[2][2];
    lim[3][1] = lim[3][0] + (nyd - 1)*lim[3][2];
    lim[4][1] = lim[4][0] + (n1d - 1)*lim[4][2];

    // Gauss-Legendre nodes gl[0][q] = s_q and weights gl[1][q] = w_q on
    // [0, 1] (header, item 3).
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(ngl);
    for (int q=0; q<ngl; q++) {
      gsl_integration_glfixed_point(0.0, 1.0, q, &gl[0][q], &gl[1][q],
                                    gl_table);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // sin_kern[q][j] = s_q sin(z_j s_q)/z_j, z_j = sqrt(w_j): the
    // z-dependent factor of the numerator integrand of header item 3 at
    // GL node q and w node w_j = (j - PAD) hw. At w < 0 it is
    // s sinh(kappa s)/kappa, kappa = sqrt(-w), and s^2 at w = 0: one
    // analytic function of w (s sin(zs)/z = s^2 - w s^4/6 + ..., the
    // same series on either side of w = 0). Gamma enters only through
    // theta, so this kernel is built once and reused by every refill.
    for (int j=0; j<nwp; j++) {
      const double w = -PAD*hw + j*hw;
      for (int q=0; q<ngl; q++) {
        if (w > 0) {
          const double z = sqrt(w);
          sin_kern[q][j] = gl[0][q]*sin(z*gl[0][q])/z;
        }
        else if (w < 0) {
          const double kappa = sqrt(-w);
          sin_kern[q][j] = gl[0][q]*sinh(kappa*gl[0][q])/kappa;
        }
        else {
          sin_kern[q][j] = gl[0][q]*gl[0][q];
        }
      }
    }

    // Trapezoid nodes of header item 2: trap[0][k] = t_k = e^{s_k},
    // s_k = smin + k hs, and trap[1][k] = h t_k e^{-t_k}, the weight of
    // the P sum. End weights are not halved: the integrand vanishes at
    // both ends.
    for (int k=0; k<nt; k++) {
      trap[0][k] = exp(smin + k*hs);
      trap[1][k] = hs*trap[0][k]*exp(-trap[0][k]);
    }

    // Qwgt[k][j] = z_j h t_k e^{-z_j t_k}, the weight of the Q sum
    // (header, item 2; leading z included) at node j of the padded ln z
    // axis: h t_k from dt = t ds, z_j e^{-z_j t_k} the unit-integral
    // weight that averages the g ratio over t up to ~1/z_j.
    for (int j=0; j<nzp; j++) {
      const double z = exp(lim[2][0] + j*hz);
      for (int k=0; k<nt; k++) {
        Qwgt[k][j] = z*hs*trap[0][k]*exp(-z*trap[0][k]);
      }
    }

    // Shared ln tau grid of P (header, item 2). With s_k = smin + k hs =
    // smin + k rP hy and ln y_j = lim[3][0] + j hy,
    //
    //   ln(t_k/y_j) = (smin - lim[3][0]) + (k rP - j) hy,
    //
    // so tau = t_k/y_j is node m = k rP - j + (nyp - 1) of the grid
    // tau_g[0][m] = tau_m (m = 0 at (k, j) = (0, nyp - 1), m = ntau - 1
    // at (nt - 1, 0)). The refill fills tau_g[1][m] = Re g(i tau_m).
    for (int m=0; m<ntau; m++) {
      tau_g[0][m] = exp(smin - lim[3][0] + (m - (nyp - 1))*hy);
    }
  }

  // --- 4. GAMMA REFILL ---
  // Everything Gamma enters, rebuilt from the nodes and weights above.
  if (fdiff2(cache[0], nuisance.random_gas) || fdiff2(cache[1], Ntable.random))
  {
    // The KS exponents p (pressure) and q (density) of the header; the
    // coarse spacings come back from the dense ones.
    const double p  = nuisance.gas[0]/(nuisance.gas[0] - 1.0);
    const double q  = 1.0/(nuisance.gas[0] - 1.0);
    const double hc = lim[0][2]*MC;
    const double hw = lim[1][2]*MW;
    const double hz = lim[2][2]*MZ;
    const double hy = lim[3][2]*MY;
    const double h1 = lim[4][2]*M1;

    /* PHYSICAL DERIVATION & LOGIC FLOW (one c node per iteration)
       c = c_i = exp(lim[0][0] + i hc), a node of the padded ln c axis;
       the two (ln c, .) tables u and Q share this loop because both
       need the profile at this c and nothing else couples their axes
       1. GL pass at x = c s_q (nodes s_q = gl[0][q], weights w_q =
          gl[1][q]), theta = log1p(x)/x:
          f0      = sum_q w_q s_q^2 theta^q      (denominator of item 3,
                    i.e. F0(c)/c^3; the c^3 cancels in u)
          wthp[q] = w_q theta^p                  (numerator, z-free part)
       2. u(c_i, w_j) = sum_q wthp[q] sin_kern[q][j]/f0, all w at once:
          sin_kern[q][j] = s_q sin(z_j s_q)/z_j completes the numerator
          integrand s theta(cs)^p sin(zs)/z of item 3 at w_j = z_j^2
       3. g_re[k] + i g_im[k] = g(c + i c t_k)/g(c) at the trapezoid
          nodes t_k = trap[0][k], with g_at_c = g(c) = c theta(c)^p:
          ks_cg on the ray x = c + i c t of item 1, normalised by its
          value at the foot of the ray (so the ratio -> 1 as t -> 0)
       4. Q(c_i, z_j) = sum_k (g_re[k] + i g_im[k]) Qwgt[k][j], the
          trapezoid sum of item 2, Qwgt[k][j] = z_j h t_k e^{-z_j t_k} */
    #pragma omp parallel for schedule(static)
    for (int i=0; i<ncp; i++) {
      const double c = exp(lim[0][0] + i*hc);

      // GL pass: mass norm f0 and the z-free pressure weights wthp[q]
      double wthp[ngl];
      double f0 = 0.0;
      for (int k=0; k<ngl; k++) {
        const double x = c*gl[0][k];
        const double theta = log1p(x)/x;
        f0 += gl[1][k]*gl[0][k]*gl[0][k]*pow(theta, q);
        wthp[k] = gl[1][k]*pow(theta, p);
      }

      // u(c_i, w_j) = sum_q wthp[q] sin_kern[q][j]/f0, all w nodes at
      // once (four per SIMDe step, then a scalar tail;
      // the plain loop is retained in the historical scalar reference)
      double* restrict u_row = u_coarse[i];
      for (int j=0; j<nwp; j++) {
        u_row[j] = 0.0;
      }
      for (int k=0; k<ngl; k++) {
        const double wthp_k = wthp[k];
        const double* restrict kern_k = sin_kern[k];
        // scalar: u_row[j] += wthp_k*kern_k[j], four w nodes j, j+1,
        // j+2, j+3 per step (one per lane of a v4d)

        // the weight wthp_k of this GL node in all four lanes (set1
        // copies one scalar into every lane)
        const v4d vwthp = simde_mm256_set1_pd(wthp_k);
        int j = 0;
        for (; j <= nwp - 4; j += 4) {
          // sin_kern[k][j..j+3] (loadu reads four consecutive doubles
          // from memory into the lanes; u = any address, aligned or not)
          const v4d vkern = simde_mm256_loadu_pd(kern_k + j);

          // wthp_k sin_kern[k][j..j+3], lane by lane
          const v4d vprod = simde_mm256_mul_pd(vwthp, vkern);

          // the running sums u_row[j..j+3]
          const v4d vu_old = simde_mm256_loadu_pd(u_row + j);

          // u_row + wthp_k sin_kern
          const v4d vu_new = simde_mm256_add_pd(vu_old, vprod);

          // back to u_row[j..j+3] (storeu writes the four lanes to memory)
          simde_mm256_storeu_pd(u_row + j, vu_new);
        }
        for (; j < nwp; j++) {
          u_row[j] += wthp_k*kern_k[j];
        }
      }
      for (int j=0; j<nwp; j++) {
        u_row[j] /= f0;
      }

      // g ratio at the trapezoid nodes (derivation step 3)
      double g_re[nt];
      double g_im[nt];
      const double g_at_c = c*pow(log1p(c)/c, p);
      for (int k=0; k<nt; k++) {
        const double complex g_ratio = ks_cg(c + I*c*trap[0][k], p)/g_at_c;
        g_re[k] = creal(g_ratio);
        g_im[k] = cimag(g_ratio);
      }

      // Q(c_i, z_j) = sum_k (g_re[k] + i g_im[k]) Qwgt[k][j], all z
      // nodes at once; Re Q goes to Q_coarse[0][i], Im Q to
      // Q_coarse[1][i], as for u above
      double* restrict Qre_row = Q_coarse[0][i];
      double* restrict Qim_row = Q_coarse[1][i];
      for (int j=0; j<nzp; j++) {
        Qre_row[j] = 0.0;
        Qim_row[j] = 0.0;
      }
      for (int k=0; k<nt; k++) {
        const double g_re_k = g_re[k];
        const double g_im_k = g_im[k];
        const double* restrict Qwgt_k = Qwgt[k];
        // scalar: Qre_row[j] += Qwgt_k[j]*g_re_k and
        //         Qim_row[j] += Qwgt_k[j]*g_im_k, four z nodes j..j+3
        // per step (set1, loadu, mul, add, storeu as in the u sum above)

        // Re g_k and Im g_k of this trapezoid node in all four lanes
        const v4d vg_re = simde_mm256_set1_pd(g_re_k);  // Re g_k
        const v4d vg_im = simde_mm256_set1_pd(g_im_k);  // Im g_k
        int j = 0;
        for (; j <= nzp - 4; j += 4) {
          // Qwgt[k][j..j+3]
          const v4d vQwgt = simde_mm256_loadu_pd(Qwgt_k + j);

          // the running sums Re Q at z nodes j..j+3
          const v4d vQre_old = simde_mm256_loadu_pd(Qre_row + j);

          // Qwgt Re g_k
          const v4d vQre_add = simde_mm256_mul_pd(vQwgt, vg_re);

          // Re Q + Qwgt Re g_k
          const v4d vQre_new = simde_mm256_add_pd(vQre_old, vQre_add);

          // back to Qre_row[j..j+3]
          simde_mm256_storeu_pd(Qre_row + j, vQre_new);

          // the running sums Im Q at z nodes j..j+3
          const v4d vQim_old = simde_mm256_loadu_pd(Qim_row + j);

          // Qwgt Im g_k
          const v4d vQim_add = simde_mm256_mul_pd(vQwgt, vg_im);

          // Im Q + Qwgt Im g_k
          const v4d vQim_new = simde_mm256_add_pd(vQim_old, vQim_add);

          // back to Qim_row[j..j+3]
          simde_mm256_storeu_pd(Qim_row + j, vQim_new);
        }
        for (; j < nzp; j++) {
          Qre_row[j] += Qwgt_k[j]*g_re_k;
          Qim_row[j] += Qwgt_k[j]*g_im_k;
        }
      }
    }

    /* PHYSICAL DERIVATION & LOGIC FLOW (header, item 2)
       1. tau_g[1][m] = Re g(i tau_m), once per shared ln tau node: the
          integrand of I0 on the ray x = i tau of item 1. Only the real
          part is needed, since P = Re I0 and e^{-y tau} is real
       2. P(y_j) = (1/y_j) sum_k trap[1][k] tau_g[1][k rP - j + nyp - 1]
          (g_row = tau_g[1] + nyp - 1 - j, read at g_row[k rP]): the
          trapezoid sum of item 2 with trap[1][k] = h t_k e^{-t_k}, the
          node tau = t_k/y_j found at its index on the shared grid, and
          the 1/y_j from dtau = dt/y
       3. stored as ln P: P -> p/y^3 at large y, so ln P is close to a
          straight line in ln y */
    #pragma omp parallel for schedule(static)
    for (int m=0; m<ntau; m++) {
      tau_g[1][m] = creal(ks_cg(I*tau_g[0][m], p));
    }
    #pragma omp parallel for schedule(static)
    for (int j=0; j<nyp; j++) {
      const double y = exp(lim[3][0] + j*hy);
      const double* restrict g_row = tau_g[1] + (nyp - 1 - j);
      const double* restrict wgt = trap[1];
      double sum = 0.0;
      for (int k=0; k<nt; k++) {
        sum += wgt[k]*g_row[k*rP];
      }
      lnP_coarse[j] = log(sum/y);
    }

    /* PHYSICAL DERIVATION & LOGIC FLOW (1D ln c axis)
       The contour formula divides by y F0(c) and multiplies by g(c),
       the two c-only ingredients of item 1; they get their own, finer
       ln c axis (lim[4]) because the lookup reads them as exponentials
       of their logs and their relative error goes straight into u
       1. F0 = c^3 sum_q w_q s_q^2 theta(c s_q)^q  (the c^3 from x = c s;
          unlike f0 of the (ln c, w) loop, the full mass integral)
       2. ln g = ln c + p ln theta(c), the integrand of F at the edge */
    for (int j=0; j<n1p; j++) {
      const double c = exp(lim[4][0] + j*h1);
      double f0 = 0.0;
      for (int k=0; k<ngl; k++) {
        const double x = c*gl[0][k];
        f0 += gl[1][k]*gl[0][k]*gl[0][k]*pow(log1p(x)/x, q);
      }
      lnF0g_coarse[0][j] = log(c*c*c*f0);
      lnF0g_coarse[1][j] = log(c) + p*log(log1p(c)/c);
    }

    // Upsampling (header, item 4): six independent jobs, each a natural
    // cubic spline from a padded coarse grid to the dense grid sharing
    // its ends. spline2d_upsample_uniform (basics.c) splines along each
    // axis in turn; ks_upsample1d is the 1D form.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int job=0; job<6; job++) {
      switch (job) {
        case 0:
          spline2d_upsample_uniform(Q_coarse[0], ncp, nzp, hc, hz,
                                    Q_dense[0], ncd, nzd);
          break;
        case 1:
          spline2d_upsample_uniform(Q_coarse[1], ncp, nzp, hc, hz,
                                    Q_dense[1], ncd, nzd);
          break;
        case 2:
          spline2d_upsample_uniform(u_coarse, ncp, nwp, hc, hw,
                                    u_dense, ncd, nwd);
          break;
        case 3:
          ks_upsample1d(lnP_coarse, nyp, hy, spline_ws[0], lnP_dense, MY);
          break;
        case 4:
          ks_upsample1d(lnF0g_coarse[0], n1p, h1, spline_ws[1],
                        lnF0g_dense[0], M1);
          break;
        default:
          ks_upsample1d(lnF0g_coarse[1], n1p, h1, spline_ws[2],
                        lnF0g_dense[1], M1);
          break;
      }
    }

    cache[0] = nuisance.random_gas;
    cache[1] = Ntable.random;
  }

  // --- 5. LOOKUP ---
  // c is clamped to the tabulated range; z = k r_v is formed from the
  // arguments as given, so a clamped query returns u at the edge
  // concentration and the true phase. (interpol2d returns 0 outside
  // its first axis, so the clamp on ln c is what keeps every read
  // inside the padded table.)
  const double c_clamped = fmin(fmax(c, limits.halo_uks_c[RANGE_MIN]),
                                limits.halo_uks_c[RANGE_MAX]);
  const double lnc = log(c_clamped);
  const double z   = k*rv;  // the phase z = k r_v

  // z < ZSW: one bilinear read of u(ln c, w) at w = z^2 (header, item 3)
  if (z < ZSW) {
    return interpol2d(u_dense, ncd, lim[0][0], lim[0][1], lim[0][2], lnc,
                      nwd, lim[1][0], lim[1][1], lim[1][2], z*z);
  }

  // z >= ZSW: the contour formula of header item 1 at y = z/c (the
  // phase at the edge, z, over the concentration gives k r_s). lnz
  // clamps z to ZHI (Q held at its ZHI value, 1 to O(1/ZHI)); lny
  // clamps ln y to its axis, which acts only above ZHI, where the P
  // term (order 1/y^4) is negligible against the Q term (order 1/y^2).
  // P, g and F0 come back from their logs.
  const double y   = z/c_clamped;
  const double lnz = log(fmin(z, ZHI));
  const double lny = fmin(fmax(log(y), log(ZSW/limits.halo_uks_c[RANGE_MAX])),
                          log(ZHI/limits.halo_uks_c[RANGE_MIN]));

  const double P = exp(interpol1d(lnP_dense, nyd, lim[3][0], lim[3][1],
                                  lim[3][2], lny));
  const double Q_re = interpol2d(Q_dense[0], ncd, lim[0][0], lim[0][1],
                                 lim[0][2], lnc, nzd, lim[2][0], lim[2][1],
                                 lim[2][2], lnz);
  const double Q_im = interpol2d(Q_dense[1], ncd, lim[0][0], lim[0][1],
                                 lim[0][2], lnc, nzd, lim[2][0], lim[2][1],
                                 lim[2][2], lnz);
  const double g  = exp(interpol1d(lnF0g_dense[1], n1d, lim[4][0],
                                   lim[4][1], lim[4][2], lnc));
  const double F0 = exp(interpol1d(lnF0g_dense[0], n1d, lim[4][0],
                                   lim[4][1], lim[4][2], lnc));

  // u = [P - (g/y)(cos z Re Q - sin z Im Q)]/(y F0), header item 1:
  // Im J/y over F0, with Im J = P - (g/y) Re[e^{iz} Q]. P is the
  // untruncated (centre-ray) transform, the Q term the edge-ray
  // correction, and the oscillation enters only through the exact
  // cos z and sin z: the tables hold nothing that rings.
  return (P - g/y*(cos(z)*Q_re - sin(z)*Q_im))/(y*F0);
}


// ---------------------------------------------------------------------------
// Fraction of the halo mass in bound gas (2005.00009 Eq. 25, from
// 1510.06034 Eq. 2.19):
//
//   f_bnd(M) = (Omega_b/Omega_m) / [1 + (M_0/M)^beta]
//
// Massive halos keep their cosmic share of baryons as hot bound gas
// (M >> M_0: f_bnd -> Omega_b/Omega_m); feedback empties light halos
// (M << M_0: f_bnd ~ (Omega_b/Omega_m)(M/M_0)^beta -> 0). A halo of
// mass M_0 keeps half; beta sets how sharp the transition is (HMx
// defaults, 2005.00009 sec. 3.2: M_0 = 1e14 M_sun, beta = 0.6).
//
// Parameters:
//   M - halo mass in M_sun/h (M_0 = 10^nuisance.gas[2] M_sun/h,
//       beta = nuisance.gas[1])
//
// Returns:
//   f_bnd in [0, Omega_b/Omega_m]
// ---------------------------------------------------------------------------
double frac_bnd(
    double M  // halo mass in M_sun/h
  )
{
  const double M0   = pow(10.0, nuisance.gas[2]);  // half-bound mass
  const double beta = nuisance.gas[1];             // mass slope

  // f_bnd = (Omega_b/Omega_m)/[1 + (M_0/M)^beta]
  const double suppression = pow(M0/M, beta);
  return cosmology.Omega_b/(cosmology.Omega_m*(1.0 + suppression));
}


// ---------------------------------------------------------------------------
// Fraction of the halo mass in ejected gas (2005.00009 Eq. 26):
//
//   f_ejc(M) = Omega_b/Omega_m - f_bnd(M) - f_*(M)
//
// The baryons of the halo's initial overdensity that are neither bound
// gas nor stars have been pushed beyond r_Delta by feedback. The
// stellar fraction (2005.00009 Eq. 27, from 1401.2997 sec. 3.3)
//
//   f_*(M) = A_* exp[-log10^2(M/M_*)/(2 sigma_*^2)]
//
// peaks at M_* with height A_* and width sigma_* in dex. Above M_* it
// is floored at A_*/3, the high-mass saturation of the stellar-to-halo
// mass relation (2005.00009 sec. 3.2):
//
//   f_*(M > M_*) = max(f_*(M), A_*/3)
//
// Clip: f_ejc is floored at 0 where f_bnd + f_* would exceed
// Omega_b/Omega_m (the heaviest halos). 2005.00009 (footnote in
// sec. 3.2) takes the excess out of the stars instead; the gas to eject
// is zero either way, and f_* is used nowhere else.
//
// Parameters:
//   M - halo mass in M_sun/h (A_* = nuisance.gas[6],
//       log10 M_* = nuisance.gas[7], sigma_* = nuisance.gas[8])
//
// Returns:
//   f_ejc in [0, Omega_b/Omega_m]
// ---------------------------------------------------------------------------
double frac_ejc(
    double M  // halo mass in M_sun/h
  )
{
  // stellar fraction: Gaussian peak in log10 M, delta in units of
  // sigma_*
  const double log10M = log10(M);
  const double delta  = (log10M - nuisance.gas[7])/nuisance.gas[8];
  const double f_star_gauss = nuisance.gas[6]*exp(-0.5*delta*delta);

  // above M_*, f_* saturates at the floor A_*/3 (header)
  const double f_star_floor = nuisance.gas[6]/3.0;
  double frac_star = f_star_gauss;
  if ((log10M > nuisance.gas[7]) && (f_star_gauss < f_star_floor)) {
    frac_star = f_star_floor;
  }

  // clip at 0: no gas to eject once f_bnd + f_* fill the baryon budget
  return fmax(0.0,
              cosmology.Omega_b/cosmology.Omega_m - frac_bnd(M) - frac_star);
}


// ---------------------------------------------------------------------------
// Electron-pressure window of the ejected gas: its electron count times
// k_B T_w, at the warm temperature T_w,
//
//   W_ejc(M) = N_e k_B T_w,   N_e = f_ejc M/(mu_e m_p)
//
// The ejected gas follows the linear density field outside halos, so it
// has no 1-halo term and a k-independent (point-like) window in the
// 2-halo term (2005.00009 sec. 3.3 and Eq. 36).
//
// Unit chain, landing on the units of W_p so the two windows add:
//
//   num_p = M_sun/m_p = 1.1892e57           (M_sun = 1.989e30 kg)
//   num_p m[M_sun/h]  = h N_p
//   k_B T_w [eV]      = 8.6173e-5 T_w[K]
//   1 eV              = 5.616e-44 h U,   U = G (M_sun/h)^2/(c/H0)
//     ->  E_w = 8.6173e-5 T_w 5.616e-44 = (k_B T_w in U)/h
//     ->  num_p m f_ejc E_w/mu_e = N_e k_B T_w in U   (h cancels)
//
// W_ejc >= 0, since frac_ejc clips f_ejc at 0.
//
// Parameters:
//   m - halo mass in M_sun/h (T_w = 10^nuisance.gas[9] K,
//       f_H = nuisance.gas[10])
//
// Returns:
//   W_ejc(M) in U = G (M_sun/h)^2/(c/H0)
// ---------------------------------------------------------------------------
double u_y_ejc(
    double m  // halo mass in M_sun/h
  )
{
  // the unit chain of the header, one named factor per step
  const double num_p    = 1.1892e57;  // M_sun/m_p: protons per M_sun
  const double kB_eV_K  = 8.6173e-5;  // k_B in eV per K
  const double eV_to_hU = 5.616e-44;  // 1 eV in h U

  // k_B T_w: K -> eV -> U
  const double E_w = pow(10, nuisance.gas[9])*kB_eV_K*eV_to_hU;

  // mu_e = 2/(1 + f_H): proton masses per electron of the ionized gas
  const double mu_e = 2./(1. + nuisance.gas[10]);

  // W_ejc = N_e k_B T_w
  return (num_p * m * frac_ejc(m) / mu_e) * E_w;
}



