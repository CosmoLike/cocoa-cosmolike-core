#include <carma.h>
#include <armadillo>
#include <map>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_COSMO2D_WRAPPER_HPP
#define __COSMOLIKE_COSMO2D_WRAPPER_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

inline py::array_t<double,py::array::f_style> to_np4d(
    const arma::field<arma::Cube<double>>& f
  )
{
  if (0 == f.n_elem) {
    return py::array_t<double,py::array::f_style>(std::vector<int>{0,0,0,0});
  }
  const auto& c0 = f(0);
  const int n1 = static_cast<int>(c0.n_rows);
  const int n2 = static_cast<int>(c0.n_cols);
  const int n3 = static_cast<int>(c0.n_slices);
  const int n4 = static_cast<int>(f.n_elem);

  for (int k=1; k<n4; ++k) { // we need all cubes to have the same shape
    const auto& ck = f(k);
    const int m1 = static_cast<int>(ck.n_rows);
    const int m2 = static_cast<int>(ck.n_cols);
    const int m3 = static_cast<int>(ck.n_slices);
    if (m1 != n1 || m2 != n2 || m3 != n3) {
      spdlog::critical("{}: incompatible array structure", "to_np4d"); exit(1);
    }
  }

  // first: do a list of cubes
  py::list t; 
  for (int k=0; k<n4; ++k) t.append(carma::cube_to_arr(f(k)));

  // second: stack the list of cubes into 4d np tensor
  py::module_ np = py::module_::import("numpy");
  py::array tmp = np.attr("stack")(t, py::arg("axis") = 0).cast<py::array>();

  // last: ensure Fortran-contiguous output to match the return type (f_style).
  py::array fA = np.attr("asfortranarray")(tmp).cast<py::array>();
  return fA.cast<py::array_t<double,py::array::f_style>>();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Col<double> get_binning_real_space();

arma::Col<double> get_binning_fourier_space();

pybind11::tuple xi_pm_tomo_cpp();

arma::Cube<double> w_gammat_tomo_cpp();

arma::Cube<double> w_gg_tomo_cpp();

/*
arma::Col<double> w_gg_tomo_cpp();

arma::Col<double> w_gk_tomo_cpp();

arma::Col<double> w_ks_tomo_cpp();
*/

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple C_ss_tomo_limber_cpp(
    const double l, 
    const int ni, 
    const int nj
  );

py::tuple C_ss_tomo_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Mat<double> gs_bins();

double C_gs_tomo_limber_cpp(
    const double l, 
    const int ni, 
    const int nj
  );

arma::Cube<double> C_gs_tomo_limber_cpp(
    const arma::Col<double> l
  );


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Cube<double> C_gg_tomo_limber_cpp(const arma::Col<double> l);

arma::Cube<double> C_gg_tomo_cpp(const arma::Col<double> l);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gk_tomo_limber_cpp(
    const double l, 
    const int ni
  );

arma::Mat<double> C_gk_tomo_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_ks_tomo_limber_cpp(
    const double l, 
    const int ni
  );

arma::Mat<double> C_ks_tomo_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_kk_limber_cpp(
    const double l
  );

arma::Col<double> C_kk_limber_cpp(
    const arma::Col<double> l
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
