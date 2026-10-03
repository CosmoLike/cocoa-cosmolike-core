// Python-facing wrappers of p_mm (future_port_unfinished/halo_pmm.c), kept
// with the function they call; see the header of halo_pmm.c for how to
// restore both.

// ---------------------------------------------------------------------------
// Matter-matter power spectrum at one (k, a).
//
// Calls halo.c p_mm (table rebuilt when cosmology.random or
// Ntable.random changes).
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor inside [limits.a_min, 0.9999999]
//
// Returns:
//   P_mm(k, a) in (c/H0)^3
// ---------------------------------------------------------------------------
double p_mm_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("p_mm_cpp", k);
  return p_mm(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Matter-matter power spectrum at many k, one a: the scalar call in a
// serial loop (after the first call builds the table, each entry is one
// bilinear lookup).
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0 aborts
//   a - scale factor inside [limits.a_min, 0.9999999]
//
// Returns:
//   arma::Col of P_mm(k(i), a) in (c/H0)^3, same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> p_mm_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a               // scale factor
  )
{
  check_wavenumbers("p_mm_cpp", k);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = p_mm(k(i), a);
  }
  return res;
}

// pybind11 bindings, identical in every project's interface/interface.cpp
// (paste back before the p_gm bindings to restore):
#if 0
  m.def("p_mm",
      py::overload_cast<const double, const double>(
        &cosmolike_interface::p_mm_cpp
      ),
      "Halo-model matter power spectrum at one (k, a); k in (c/H0)^-1, "
      "P in (c/H0)^3",
      py::arg("k").none(false).noconvert(),
      py::arg("a").none(false).noconvert()
    );

  m.def("p_mm",
      py::overload_cast<const arma::Col<double>, const double>(
        &cosmolike_interface::p_mm_cpp
      ),
      "Halo-model matter power spectrum at many k, one a (vectorized)",
      py::arg("k").none(false),
      py::arg("a").none(false),
      py::return_value_policy::move
    );

#endif
