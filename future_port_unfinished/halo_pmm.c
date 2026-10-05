// The halo-model matter power spectrum p_mm (P_mm = I02 + I11^2 P_lin with
// the HMx correction for the halos below M_min, 2005.00009). No likelihood
// or covariance uses it, so it is kept here, outside the compiled sources.
// The quadrature and loop structure it documents stay in halo.c (POWER
// SPECTRA banner, items 1 and 2) as the template of p_gm, p_gg and the
// halo-model IA.
//
// It depends on halo.c statics (halo_warmup with hod = 0, fnu_params_at,
// hb1nu_params_at, nfw_um/nfw_um4) and on u_c, bias_norm, conc, sigma2. To
// use it again:
//   1. paste it back into halo.c, in the HALO MODEL POWER SPECTRA section;
//   2. declare it in halo.h (double p_mm(const double k, const double a););
//   3. move the wrappers of halo_pmm_wrapper.cpp back into halo_wrapper.cpp
//      and declare them in halo_wrapper.hpp; restore the bindings and tests
//      listed there;
//   4. with massive neutrinos, read
//      .claude/skills/cosmolike-dev/references/fable_review_neutrino_halos.md
//      (HMcode-2020's split: cold nu, one-halo times (1 - f_nu)^2, two-halo
//      with the total linear P).
// The CosmoCov rewrite (cosmolike/covariance) builds its own P_hm next to the
// response D_hm instead of calling this function.

// ---------------------------------------------------------------------------
// P_mm(k, a), the halo-model matter power spectrum, from a table of ln P
// on Ntable.N_a x Ntable.N_k_nlin nodes uniform in (a, ln k), read
// bilinearly (interpol2d) and exponentiated:
//
//   P_mm = I02 + I11^2 P_lin                          (2005.00009 Eqs. 1-2)
//   I02  = int dlnM dn/dlnM (M/rho_m)^2 u(k|M)^2
//   I11  = int dlnM dn/dlnM b(nu) (M/rho_m) u(k|M) + A(a) u(k|M_min)
//
// dn/dlnM = (rho_m/M) nu f(nu) dlnnu/dlnM is the mass function, (M/rho_m) u
// the matter window, b the Tinker bias, A(a) = 1 - bias_norm(a) the HMx
// share of matter below M_min put back as halos of mass M_min (section
// banner).
//
// 1. Quadrature: the n-point Gauss-Legendre rule in ln M (exact for
// polynomials of degree 2n - 1; n follows an
// Ntable.high_def_integration ladder); nodes M_q and weights w_q on
// [ln M_min, ln M_max] are mapped once in the rebuild block
// (gsl_integration_glfixed_point). High k converges slowest: the
// profile's ringing is sampled in ln M.
//
// 2. Loop levels: each factor is computed at the outermost level it
// depends on, so the innermost loop is the NFW kernel alone (nfw_um:
// three table reads and two sines; no pow, exp or log):
//
//   per refill, per node q       M_q, w_q; nu0_q = delta_c/sigma(M_q);
//     (mass_node)                w_q (rho_m/M_q) dlnnu/dlnM; M_q/rho_m;
//                                r_Delta(M_q)
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min)
//   per (i, q) (a_node[i])       nu = nu0/D; c = conc(M, D); ln(1+c);
//                                m(c) = ln(1+c) - c/(1+c); r_s = r_Delta/c,
//                                ln r_s; w1h_q = dn (M/rho_m)^2/m(c)^2;
//                                w2h_q = dn b(nu) (M/rho_m)/m(c), with
//                                dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
//   per (i, k), sum over q       x = k r_s, ln x = ln k + ln r_s,
//                                um = u m(c) = nfw_um(c, x, ln x, ln(1+c));
//                                I02 = sum w1h_q um^2,
//                                I11 = sum w2h_q um + A u_c(k|M_min); ln P
//
// The 1/m(c) of u = um/m(c) lives in w1h_q and w2h_q. The rows read the
// NFW kernel directly, so like.halo_model[3] must be HALO_PROFILE_NFW (the
// only option of u_c); anything else aborts.
//
// Thread safety: the single-threaded halo_warmup call before the
// threaded loop builds every lazy table the rows read (its header).
//
// Cache invalidation:
//   rebuild block (table, mass_node, a_node, GL nodes, both grids; every
//     allocation lives here, one block each from malloc2d/malloc3d):
//     Ntable.random
//   refill: cosmology.random or Ntable.random
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_mm(k, a) in (c/H0)^3; 0 outside [limits.a_min, 0.9999999]; ln P
//   continued with unit slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_mm(
    const double k,
    const double a
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];  // tags the table was built from
  static double** table = NULL;            // ln P on the (a, ln k) grid
  static double lim[2][3];           // [0] a grid: min, max, step;
                                     // [1] ln k grid: min, max, step
  static int n_nodes = 0;            // Gauss-Legendre nodes in ln M
  static double** mass_node = NULL;  // [6][n_nodes] per mass node q:
                                     // 0 M, 1 GL weight w, 2 nu0 = nu(D=1),
                                     // 3 w (rho_m/M) dlnnu/dlnM,
                                     // 4 M/rho_m, 5 r_Delta
  static double*** a_node = NULL;    // [N_a][6][n_nodes] per (a row, node):
                                     // 0 c, 1 ln(1+c), 2 r_s, 3 ln r_s,
                                     // 4 w1h (1-halo), 5 w2h (2-halo)

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
    mass_node = (double**) malloc2d(6, n_nodes);
    a_node    = (double***) malloc3d(Ntable.N_a, 6, n_nodes);

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
  // refill when the cosmology or Ntable tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // the k columns read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    // total-matter halos only: the 2-halo term I11^2 P_lin has no cb
    // form (HALO MODEL POWER SPECTRA banner)
    if (like.halo_model[4] != HALO_FIELD_MATTER) {
      log_fatal("p_mm: like.halo_model[4] = %d not supported (the matter "
                "spectrum needs the total-matter halo field)",
                like.halo_model[4]);
      exit(1);
    }

    // warm-up (header, Thread safety): every lazy table the threaded
    // loop reads is built here, on one thread
    halo_warmup(lim[0][0], exp(lim[1][0]), 0, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (full derivation: the header)
       1. node q: M_q, GL weight w_q; nu0_q = delta_c/sigma(M_q)
       2. row i (threaded): nu = nu0_q/D(a); c = conc(M_q, D);
          w1h_q = dn (M/rho_m)^2/m(c)^2; w2h_q = dn b(nu) (M/rho_m)/m(c);
          dn = w_q (rho_m/M) dlnnu/dlnM f(nu) nu
       3. column j: um = nfw_um(c, k r_s); I02 = sum_q w1h_q um^2;
          I11 = sum_q w2h_q um + A u_c(k|M_min)
       4. table = ln(I02 + I11^2 P_lin) */

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
    }

    // --- 4. PER a ROW, THREADED ---
    // each row: D(a), the Tinker f and b parameters (*_params_at, the
    // nu-independent halves), A(a), c(M_min) (header, item 2, second row)
    const double m_min = limits.halo_m[RANGE_MIN];  // M_min of A u_c(k|M_min)

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

      // per-node rows of this a row (restrict: distinct rows)
      double* restrict conc_q = a_node[i][0];  // c(M_q, a)
      double* restrict ln1c_q = a_node[i][1];  // ln(1 + c)
      double* restrict rs_q   = a_node[i][2];  // r_s = r_Delta/c
      double* restrict lnrs_q = a_node[i][3];  // ln r_s
      double* restrict w1h_q  = a_node[i][4];  // 1-halo weight
      double* restrict w2h_q  = a_node[i][5];  // 2-halo weight

      // per (a, node): c(M, a), r_s and their logs, and the weights
      // w1h_q, w2h_q with the 1/m(c) of u = um/m(c) folded in (header,
      // item 2, third row); dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
      for (int q=0; q<n_nodes; q++) {
        const double nu   = mass_node[2][q]/D;  // nu = nu0/D(a)
        const double c    = conc(mass_node[0][q], D);
        const double ln1c = log1p(c);
        const double mc   = ln1c - c/(1.0 + c);  // NFW norm m(c)
        const double dn   = mass_node[3][q]*fnu_core(nu, &f_params)*nu;
        conc_q[q] = c;
        ln1c_q[q] = ln1c;
        rs_q[q]   = mass_node[5][q]/c;  // r_s = r_Delta/c
        lnrs_q[q] = log(rs_q[q]);
        w1h_q[q]  = dn*(mass_node[4][q]/mc)*(mass_node[4][q]/mc);
        w2h_q[q]  = dn*hb1nu_core(nu, &b_params)*(mass_node[4][q]/mc);
      }

      // per k column: I02 and I11 as sums of the NFW kernel over the
      // nodes, the HMx term A u_c(k|M_min), then ln P (header, item 2,
      // last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lnk = lim[1][0] + j*lim[1][2];
        const double kj  = exp(lnk);

        double sum_I02 = 0.0;  // 1-halo: sum_q w1h_q um^2
        double sum_I11 = 0.0;  // 2-halo: sum_q w2h_q um

        // Evaluate the weighted sums above at four nodes per step:
        // each lane of a v4d holds one node (nfw_um4 = nfw_um on each
        // lane, bitwise). The four lanes accumulate four partial sums,
        // added in a fixed lane order at the end (simd_horizontal_sum),
        // then a scalar tail takes the leftover nodes. The summation
        // order differs from the scalar loop's (last digits of the sums)
        // but never depends on the thread count.

        // k and ln k of this column in all four lanes (set1 copies one
        // scalar into every lane)
        const v4d vk   = simde_mm256_set1_pd(kj);   // k
        const v4d vlnk = simde_mm256_set1_pd(lnk);  // ln k

        // the four-lane partial sums, from (0, 0, 0, 0)
        v4d vsum_I02 = simde_mm256_setzero_pd();  // sum_I02
        v4d vsum_I11 = simde_mm256_setzero_pd();  // sum_I11

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

          // the weights of nodes q..q+3
          const v4d vw1h = simde_mm256_loadu_pd(w1h_q + q);  // 1-halo
          const v4d vw2h = simde_mm256_loadu_pd(w2h_q + q);  // 2-halo

          // scalar: sum_I02 += w1h_q[q]*um*um, as (w1h um) um + sum, fused

          // w1h um
          const v4d vw1h_um = simde_mm256_mul_pd(vw1h, vum);

          // (w1h um) um + sum_I02, lane by lane
          vsum_I02 = nfw_fmadd4(vw1h_um, vum, vsum_I02);

          // scalar: sum_I11 += w2h_q[q]*um, fused
          vsum_I11 = nfw_fmadd4(vw2h, vum, vsum_I11);
        }

        // lane 0 + lane 1 + lane 2 + lane 3 of each partial sum
        sum_I02 = simd_horizontal_sum(vsum_I02);  // sum_q w1h_q um^2
        sum_I11 = simd_horizontal_sum(vsum_I11);  // sum_q w2h_q um

        // scalar tail: n_nodes not a multiple of four
        for (; q<n_nodes; q++) {
          const double um = nfw_um(conc_q[q], kj*rs_q[q],
                                   lnk + lnrs_q[q], ln1c_q[q]);
          sum_I02 += w1h_q[q]*um*um;
          sum_I11 += w2h_q[q]*um;
        }

        const double I11 = sum_I11 + A_hmx*u_c(conc_min, kj, m_min, ai);
        table[i][j] = log(sum_I02 + I11*I11*p_lin(kj, ai));
      }
    }

    // stamp the table with the tags it was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  // --- 5. TABLE READ ---
  // bilinear read of ln P; 0 outside the a range
  if ((a < lim[0][0]) || (a > lim[0][1])) {
    return 0.0;
  }

  return exp(interpol2d(table,
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2],
                        log(k)));
}


// u_c: the profile dispatcher p_mm reads for the HMx term A(a) u(k|M_min)
// (NFW is its only option); no compiled caller is left, so it is kept here.
// Restore with p_mm (declaration: double u_c(const double c, const double k,
// const double m, const double a);).
// ---------------------------------------------------------------------------
// Normalized Fourier transform u(k|M) of the halo matter profile, with
// the profile chosen by like.halo_model[3]: HALO_PROFILE_NFW (u_nfw_c)
// is the only option, other values abort.
//
// Parameters:
//   c - concentration r_Delta/r_s
//   k - wavenumber in (c/H0)^-1
//   m - halo mass in M_sun/h
//   a - scale factor
//
// Returns:
//   u(k|M), dimensionless; 1 at k -> 0
// ---------------------------------------------------------------------------
double u_c(
    const double c, // concentration r_Delta/r_s
    const double k, // wavenumber in (c/H0)^-1
    const double m, // halo mass in M_sun/h
    const double a  // scale factor (passed to the selected profile)
  )
{
  double ans;

  switch (like.halo_model[3])
  {
    case HALO_PROFILE_NFW:
    {
      ans = u_nfw_c(c, k, m, a);
      break;
    }
    default:
    {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }
  }

  return ans;
}
