// Python-facing wrappers of p_my and p_yy (future_port_unfinished/halo_tsz.c).
// Kept here, outside the compiled sources, with the functions they call; see
// the header of halo_tsz.c for how to restore both.

// ---------------------------------------------------------------------------
// Matter-Compton-y cross power spectrum at one (k, a). The 1-halo term
// carries the low-k damping of 2009.01858 Eq. 17 (Table 2 scale).
//
// Calls halo.c p_my (table rebuilt when cosmology.random, Ntable.random
// or nuisance.random_gas changes); the gas parameters must be set first
// (set_nuisance_gas_cpp).
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor inside [limits.a_min, 0.9999999]
//
// Returns:
//   P_my(k, a) in U = G (M_sun/h)^2/(c/H0) (halo.c GAS PROFILES banner)
// ---------------------------------------------------------------------------
double p_my_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("p_my_cpp", k);
  return p_my(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Matter-Compton-y cross power spectrum at many k, one a (serial loop
// over the scalar call).
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0 aborts
//   a - scale factor inside [limits.a_min, 0.9999999]
//
// Returns:
//   arma::Col of P_my(k(i), a), same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> p_my_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a               // scale factor
  )
{
  check_wavenumbers("p_my_cpp", k);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = p_my(k(i), a);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Compton-y auto power spectrum at one (k, a), with the same low-k
// damping of the 1-halo term as p_my.
//
// Calls halo.c p_yy (table rebuilt on the same keys as p_my).
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor inside [limits.a_min, 0.9999999]
//
// Returns:
//   P_yy(k, a) in U^2 (c/H0)^-3, U = G (M_sun/h)^2/(c/H0) (halo.c GAS
//   PROFILES banner)
// ---------------------------------------------------------------------------
double p_yy_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("p_yy_cpp", k);
  return p_yy(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Compton-y auto power spectrum at many k, one a (serial loop over the
// scalar call).
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0 aborts
//   a - scale factor inside [limits.a_min, 0.9999999]
//
// Returns:
//   arma::Col of P_yy(k(i), a), same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> p_yy_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a               // scale factor
  )
{
  check_wavenumbers("p_yy_cpp", k);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = p_yy(k(i), a);
  }
  return res;
}

// pybind11 bindings, identical in every project's interface/interface.cpp
// (paste back after the p_mm bindings to restore):
#if 0
  m.def("p_my",
      py::overload_cast<const double, const double>(
        &cosmolike_interface::p_my_cpp
      ),
      "Halo-model matter-Compton y power spectrum at one (k, a); k in "
      "(c/H0)^-1",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert()
    );

  m.def("p_my",
      py::overload_cast<const arma::Col<double>, const double>(
        &cosmolike_interface::p_my_cpp
      ),
      "Halo-model matter-Compton y power spectrum at many k, one a "
      "(vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::return_value_policy::move
    );

  m.def("p_yy",
      py::overload_cast<const double, const double>(
        &cosmolike_interface::p_yy_cpp
      ),
      "Halo-model Compton y power spectrum at one (k, a); k in "
      "(c/H0)^-1",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert()
    );

  m.def("p_yy",
      py::overload_cast<const arma::Col<double>, const double>(
        &cosmolike_interface::p_yy_cpp
      ),
      "Halo-model Compton y power spectrum at many k, one a (vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::return_value_policy::move
    );
#endif

// u_KS_cpp and set_nuisance_gas_cpp (the gas profile and its parameter
// setter), kept with the gas section of halo_tsz.c; restore them into
// halo_wrapper.cpp with their halo_wrapper.hpp declarations:
//   double u_KS_cpp(const double c, const double k, const double rv);
//   void set_nuisance_gas_cpp(const arma::Col<double> gas);

// ---------------------------------------------------------------------------
// Fourier transform of the Komatsu-Seljak bound-gas pressure profile
// (polytropic index Gamma = nuisance.gas[0]), divided by the integral of
// the matching gas density profile:
//
//   u_KS = int_0^c x sin(y x)/y theta(x)^(Gamma/(Gamma-1)) dx
//          / int_0^c x^2 theta(x)^(1/(Gamma-1)) dx
//
//   theta(x) = ln(1 + x)/x,  x = r/r_s,  y = k r_s = k rv/c
//
// Calls halo.c u_KS: cached tables in ln c, the phase z = k rv and ln y
// (Ntable.halo_uks_n[UKS_N_LNC] and halo_uks_n[UKS_N_LNZ] coarse nodes, cubic-upsampled;
// the oscillation in z is carried by an exact cos z, sin z at lookup,
// see the u_KS header), refilled when nuisance.random_gas changes
// (set_nuisance_gas_cpp bumps it).
//
// Parameters:
//   c  - concentration
//   k  - wavenumber in (c/H0)^-1
//   rv - halo radius r_Delta in c/H0
//
// Returns:
//   u_KS(k), dimensionless
// ---------------------------------------------------------------------------
double u_KS_cpp(
    const double c,   // concentration
    const double k,   // wavenumber in (c/H0)^-1
    const double rv   // halo radius in c/H0
  )
{
  return u_KS(c, k, rv);
}


// ---------------------------------------------------------------------------
// Set the gas (Compton-y) parameters nuisance.gas[0..n-1] (layout in
// structs.h):
//
//   gas(0)  = Gamma     (polytropic index of the Komatsu-Seljak profile)
//   gas(1)  = beta      (mass slope of the bound-gas fraction)
//   gas(2)  = lg M_0    (mass below which gas is ejected, M_sun/h)
//   gas(3)  = eps1
//   gas(4)  = eps2
//   gas(5)  = alpha     (bound-gas temperature / virial temperature)
//   gas(6)  = A_star    (peak stellar fraction)
//   gas(7)  = lg M_star (mass of that peak, M_sun/h)
//   gas(8)  = sigma_star (width of the stellar-fraction peak in lg M)
//   gas(9)  = lg T_w    (temperature of the ejected gas, K)
//   gas(10) = f_H       (hydrogen mass fraction)
//
// Cache invalidation:
// draws a new nuisance.random_gas when any value changed (fdiff);
// unchanged input leaves the key alone.
//
// Parameters:
//   gas - the first n gas parameters (1 <= n <= MAX_SIZE_ARRAYS); an
//         empty or oversized array, or a NaN entry, aborts
//         (spdlog::critical + exit)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_gas_cpp(
    const arma::Col<double> gas   // gas parameters, structs.h layout
  )
{
  const int n = static_cast<int>(gas.n_elem);
  if (n < 1 || n > MAX_SIZE_ARRAYS) {
    spdlog::critical("{}: gas array size = {} (allowed 1 to {})",
                     "set_nuisance_gas_cpp", n, MAX_SIZE_ARRAYS);
    exit(1);
  }
  int cache_update = 0;
  for (int j=0; j<n; j++) {
    if (std::isnan(gas(j))) {
      spdlog::critical("{}: NaN found on index {}",
                       "set_nuisance_gas_cpp", j);
      exit(1);
    }
    if (fdiff(nuisance.gas[j], gas(j))) {
      cache_update = 1;
      nuisance.gas[j] = gas(j);
    }
  }
  if (1 == cache_update) {
    nuisance.random_gas = RandomNumber::get_instance().get();
  }
}

// pybind11 bindings of u_KS and set_nuisance_gas, identical in every project's
// interface/interface.cpp (u_KS before the ngal binding, set_nuisance_gas
// before set_nuisance_ia_halo):
#if 0
  m.def("u_KS",
      &cosmolike_interface::u_KS_cpp,
      "Fourier transform of the Komatsu-Seljak gas pressure profile "
      "(cached table); k in (c/H0)^-1, rv in c/H0",
      py::arg("c").none(false),
      py::arg("k").none(false),
      py::arg("rv").none(false)
    );

  m.def("set_nuisance_gas",
      &cosmolike_interface::set_nuisance_gas_cpp,
      "Set the gas (Compton-y) parameters nuisance.gas[0..n-1]",
      py::arg("gas").none(false)
    );

#endif
