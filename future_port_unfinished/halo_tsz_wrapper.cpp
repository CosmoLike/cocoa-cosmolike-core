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
