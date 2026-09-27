#include <carma.h>
#include <armadillo>
#include <map>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_COSMO2D_SCUTS_WRAPPER_HPP
#define __COSMOLIKE_COSMO2D_SCUTS_WRAPPER_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(const double k);

py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(const arma::Col<double> k);

// -----------------------------------------------------------------------------

arma::Mat<double> dlnw_ks_dlnk_tomo_cpp(const double k);

py::array_t<double,py::array::f_style> dlnw_ks_dlnk_tomo_cpp(
    const arma::Col<double> k
  );

// -----------------------------------------------------------------------------

py::tuple RF_xi_tomo_limber_cpp(
    const double k,
    const int nt,
    const int ni,
    const int nj
  );

py::tuple RF_xi_tomo_limber_cpp(const arma::Col<double> k);

// -----------------------------------------------------------------------------

double RF_w_ks_tomo_cpp(
    const double k,
    const int nt,
    const int ni
  );

py::array_t<double,py::array::f_style> RF_w_ks_tomo_cpp(
    const arma::Col<double> k
  );

// -----------------------------------------------------------------------------

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni,
    const int nj
  );

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,
    const arma::Col<double> l
  );

// -----------------------------------------------------------------------------

double dlnC_ks_dlnk_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni
  );

py::array_t<double,py::array::f_style> dlnC_ks_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,
    const arma::Col<double> l
  );

// -----------------------------------------------------------------------------

py::tuple RF_C_ss_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni,
    const int nj
  );

py::tuple RF_C_ss_tomo_limber_cpp(const arma::Col<double> k,
                                  const arma::Col<double> l);

// -----------------------------------------------------------------------------

double RF_C_ks_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni
  );

py::array_t<double,py::array::f_style> RF_C_ks_tomo_limber_cpp(
    const arma::Col<double> k,
    const arma::Col<double> l
  );

// -----------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
