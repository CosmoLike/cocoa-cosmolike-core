#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <variant>
#include <cmath> 

// SPDLOG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/IA.h"
#include "cosmolike/cosmo2D_wrapper.hpp"
#include "cosmolike/cosmo2D_scuts.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using cube = arma::Cube<double>;

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// Stack a field of equally shaped matrices into a (n, rows, cols) numpy
// array: the 3d analogue of cosmo2D_wrapper.hpp's to_np4d, used by the
// ks diagnostics, whose single component (no EE/BB or xi+/xi- split, one
// source bin instead of a pair) drops one dimension from the ss outputs.
//
// Parameters:
//   f - field of equally shaped (rows, cols) matrices; a shape mismatch
//       aborts (spdlog::critical + exit)
//
// Returns:
//   Fortran-ordered numpy array of shape (n_elem, rows, cols); an empty
//   field gives a (0, 0, 0) array
// ---------------------------------------------------------------------------
static py::array_t<double,py::array::f_style> to_np3d(
    const arma::field<arma::Mat<double>>& f
  )
{
  if (0 == f.n_elem) {
    return py::array_t<double,py::array::f_style>(std::vector<int>{0,0,0});
  }
  const auto& m0 = f(0);
  const int n1 = static_cast<int>(m0.n_rows);
  const int n2 = static_cast<int>(m0.n_cols);
  const int n3 = static_cast<int>(f.n_elem);

  for (int k=1; k<n3; ++k) { // we need all matrices to have the same shape
    const auto& mk = f(k);
    const int m1 = static_cast<int>(mk.n_rows);
    const int m2 = static_cast<int>(mk.n_cols);
    if (m1 != n1 || m2 != n2) {
      spdlog::critical("{}: incompatible array structure", "to_np3d"); exit(1);
    }
  }

  // first: do a list of matrices
  py::list t;
  for (int k=0; k<n3; ++k) t.append(carma::mat_to_arr(f(k)));

  // second: stack the list of matrices into 3d np tensor
  py::module_ np = py::module_::import("numpy");
  py::array tmp = np.attr("stack")(t, py::arg("axis") = 0).cast<py::array>();

  // last: ensure Fortran-contiguous output to match the return type (f_style).
  py::array fA = np.attr("asfortranarray")(tmp).cast<py::array>();
  return fA.cast<py::array_t<double,py::array::f_style>>();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// REAL SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared engine of the two dlnxi_dlnk_pm_tomo_limber_cpp overloads: one
// dlnxi_dlnk_pm_tomo_nointerp call at wavenumber k (which computes every
// tomographic pair and angular bin at once), scattered into the
// (theta, ni, nj) cubes with both bin orderings filled (xi is symmetric).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   XP - output xi+ cube (Ntheta, shear_nbin, shear_nbin)
//   XM - output xi- cube (Ntheta, shear_nbin, shear_nbin)
//
// Returns:
//   nothing; the result is written into XP and XM
// ---------------------------------------------------------------------------
static void dlnxi_dlnk_cubes(
    const double k,           // wavenumber in (Mpc/h)^-1
    arma::Cube<double>& XP,   // output xi+ (Ntheta, shear_nbin, shear_nbin)
    arma::Cube<double>& XM    // output xi- (Ntheta, shear_nbin, shear_nbin)
  )
{
  const int NSIZE = tomo.shear_Npowerspectra;
  double** tmp = dlnxi_dlnk_pm_tomo_nointerp(k);
  for (int nz=0; nz<NSIZE; nz++) {
    const int z1 = Z1(nz);
    const int z2 = Z2(nz);
    for (int i=0; i<Ntable.Ntheta; i++) {
      const int q = nz * Ntable.Ntheta + i;
      XP(i,z1,z2) = tmp[0][q];
      XP(i,z2,z1) = tmp[0][q];
      XM(i,z1,z2) = tmp[1][q];
      XM(i,z2,z1) = tmp[1][q];
    }
  }
  free(tmp);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnxi_pm/dlnk at one wavenumber, every tomographic pair and angular bin
// (one wavenumber is already a full batch: dlnxi_dlnk_cubes computes all
// pairs and bins in one call).
//
// Parameters:
//   k - wavenumber in (Mpc/h)^-1
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (Ntheta, shear_nbin, shear_nbin): rows = angular bin, both bin
//   orderings filled (xi is symmetric)
// ---------------------------------------------------------------------------
py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(
    const double k    // wavenumber in (Mpc/h)^-1
  )
{
  arma::Cube<double> dlnxp_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  arma::Cube<double> dlnxm_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  dlnxi_dlnk_cubes(k, dlnxp_dlnk, dlnxm_dlnk);
  return py::make_tuple(carma::cube_to_arr(dlnxp_dlnk),
                        carma::cube_to_arr(dlnxm_dlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnxi_pm/dlnk at many wavenumbers, stacked by to_np4d from one
// dlnxi_dlnk_cubes call per k (serial loop; each call is already a full
// batch over pairs and bins).
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (nk, Ntheta, shear_nbin, shear_nbin): leading axis = wavenumber,
//   both bin orderings filled (xi is symmetric)
// ---------------------------------------------------------------------------
py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(
    const arma::Col<double> k    // wavenumbers in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}",
                     "dlnxi_dlnk_pm_tomo_limber_cpp", nk);
    exit(1);
  }
  arma::field<arma::Cube<double>> dlnxp_dlnk(nk);
  arma::field<arma::Cube<double>> dlnxm_dlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> tdlnxp_dlnk(Ntable.Ntheta,
                                   redshift.shear_nbin,
                                   redshift.shear_nbin,
                                   arma::fill::zeros);
    arma::Cube<double> tdlnxm_dlnk(Ntable.Ntheta,
                                   redshift.shear_nbin,
                                   redshift.shear_nbin,
                                   arma::fill::zeros);
    dlnxi_dlnk_cubes(k(m), tdlnxp_dlnk, tdlnxm_dlnk);
    dlnxp_dlnk(m) = tdlnxp_dlnk;
    dlnxm_dlnk(m) = tdlnxm_dlnk;
  }
  return py::make_tuple(to_np4d(dlnxp_dlnk), to_np4d(dlnxm_dlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared engine of the two dlnw_ks_dlnk_tomo_cpp overloads: one
// dlnw_ks_dlnk_tomo_nointerp call at wavenumber k (which computes every
// source bin and angular bin at once), scattered into the (theta, ni)
// matrix (one source bin per column: the CMB is a single lens plane).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   WK - output matrix (Ntheta, shear_nbin)
//
// Returns:
//   nothing; the result is written into WK
// ---------------------------------------------------------------------------
static void dlnw_ks_dlnk_mat(
    const double k,        // wavenumber in (Mpc/h)^-1
    arma::Mat<double>& WK  // output (Ntheta, shear_nbin)
  )
{
  const int NSIZE = redshift.shear_nbin;
  double* tmp = dlnw_ks_dlnk_tomo_nointerp(k);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      WK(i, nz) = tmp[nz * Ntable.Ntheta + i];
    }
  }
  free(tmp);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnw_ks/dlnk at one wavenumber, every source bin and angular bin (one
// wavenumber is already a full batch: dlnw_ks_dlnk_mat computes all bins
// in one call).
//
// Parameters:
//   k - wavenumber in (Mpc/h)^-1
//
// Returns:
//   arma::Mat (Ntheta, shear_nbin): rows = angular bin, columns =
//   source bin
// ---------------------------------------------------------------------------
arma::Mat<double> dlnw_ks_dlnk_tomo_cpp(
    const double k    // wavenumber in (Mpc/h)^-1
  )
{
  arma::Mat<double> dlnwks_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  dlnw_ks_dlnk_mat(k, dlnwks_dlnk);
  return dlnwks_dlnk;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnw_ks/dlnk at many wavenumbers, stacked by to_np3d from one
// dlnw_ks_dlnk_mat call per k (serial loop; each call is already a full
// batch over bins).
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   numpy array of shape (nk, Ntheta, shear_nbin): leading axis =
//   wavenumber, then angular bin, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> dlnw_ks_dlnk_tomo_cpp(
    const arma::Col<double> k    // wavenumbers in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}",
                     "dlnw_ks_dlnk_tomo_cpp", nk);
    exit(1);
  }
  arma::field<arma::Mat<double>> dlnwks_dlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> tdlnwks_dlnk(Ntable.Ntheta,
                                   redshift.shear_nbin,
                                   arma::fill::zeros);
    dlnw_ks_dlnk_mat(k(m), tdlnwks_dlnk);
    dlnwks_dlnk(m) = tdlnwks_dlnk;
  }
  return to_np3d(dlnwks_dlnk);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// FOURIER SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two dlnC_ss_dlnk_tomo_limber_cpp overloads:
// dlnC = dC/C computed exactly on the (ln k, ell) grid by the normalized
// mode of dC_ss_dlnk_tomo_limber_work, indexed [2][NSIZE][nk][nl] with
// row nz the pair (Z1(nz), Z2(nz)). The caller owns (and frees) the
// returned array.
//
// Parameters:
//   lnkx  - ln k grid values (length nk), k in (Mpc/h)^-1
//   nk    - number of k values
//   lx    - multipole values (length nl)
//   nl    - number of multipole values
//   NSIZE - number of tomo shear power spectra
//
// Returns:
//   malloc4d array [2][NSIZE][nk][nl] ([0] = EE, [1] = BB)
// ---------------------------------------------------------------------------
static double**** dlnC_ss_dlnk_grid(
    const double* lnkx, // ln k grid values (length nk), k in (Mpc/h)^-1
    const int nk,       // number of k values
    const double* lx,   // multipole values (length nl)
    const int nl,       // number of multipole values
    const int NSIZE     // number of tomo shear power spectra
  )
{
  double**** dlnC = (double****) malloc4d(2, NSIZE, nk, nl);
  dC_ss_dlnk_tomo_limber_work(lnkx, nk, lx, nl, NSIZE, 1, dlnC);
  return dlnC;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ss/dlnk at one (k, l) for the (ni, nj) source pair in either
// ordering; 0 when the Limber node chi = (l + 1/2)/k falls outside the
// source support.
//
// Point diagnostic: runs the full batch of dlnC_ss_dlnk_grid at a
// single (k, l) and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (k, l, ni, nj) should call the array
// overload once and index the returned arrays instead.
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1; k <= 0 aborts (spdlog::critical +
//        exit)
//   l  - multipole
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (EE, BB) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "dlnC_ss_dlnk_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "dlnC_ss_dlnk_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  const double lnkx = std::log(k);
  const double lx = l;
  double**** dlnC = dlnC_ss_dlnk_grid(&lnkx, 1, &lx, 1, NSIZE);
  const int q = N_shear(ni, nj); // symmetric: any (ni, nj) ordering works
  const double CEE = dlnC[0][q][0][0];
  const double CBB = dlnC[1][q][0][0];
  free(dlnC);
  return py::make_tuple(CEE, CBB);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ss/dlnk on a (k, l) grid, scattered from one dlnC_ss_dlnk_grid
// call; only the enumerated Z1 <= Z2 ordering is filled, the reversed
// entries stay zero.
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array or any k(m) <= 0
//       aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   (EE, BB) tuple of numpy arrays of shape
//   (nk, nl, shear_nbin, shear_nbin): leading axis = wavenumber, then
//   multipole, then the source bin pair
// ---------------------------------------------------------------------------
py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,   // wavenumbers in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "dC_ss_dlnk_tomo_limber_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dC_ss_dlnk_tomo_limber_cpp", nk);
    exit(1);
  } 
  const int NSIZE = tomo.shear_Npowerspectra;
  double* lnkx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "dlnC_ss_dlnk_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double**** dlnC = dlnC_ss_dlnk_grid(lnkx, nk, lx, nl, NSIZE);
  arma::field<arma::Cube<double>> dlnCEEdlnk(nk);
  arma::field<arma::Cube<double>> dlnCBBdlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> EE(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> BB(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        EE(i, Z1(nz), Z2(nz)) = dlnC[0][nz][m][i];
        BB(i, Z1(nz), Z2(nz)) = dlnC[1][nz][m][i];
      }
    }
    dlnCEEdlnk(m) = EE;
    dlnCBBdlnk(m) = BB;
  }
  free(dlnC);
  free(lnkx);
  free(lx);
  return py::make_tuple(to_np4d(dlnCEEdlnk), to_np4d(dlnCBBdlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two dlnC_ks_dlnk_tomo_limber_cpp overloads:
// dlnC = dC/C computed exactly on the (ln k, ell) grid by the normalized
// mode of dC_ks_dlnk_tomo_limber_work, indexed [NSIZE][nk][nl] with row
// nz the source bin (one component per bin: the CMB is a single lens
// plane). The caller owns (and frees) the returned array.
//
// Parameters:
//   lnkx  - ln k grid values (length nk), k in (Mpc/h)^-1
//   nk    - number of k values
//   lx    - multipole values (length nl)
//   nl    - number of multipole values
//   NSIZE - number of source tomographic bins (= shear_nbin)
//
// Returns:
//   malloc3d array [NSIZE][nk][nl]
// ---------------------------------------------------------------------------
static double*** dlnC_ks_dlnk_grid(
    const double* lnkx, // ln k grid values (length nk), k in (Mpc/h)^-1
    const int nk,       // number of k values
    const double* lx,   // multipole values (length nl)
    const int nl,       // number of multipole values
    const int NSIZE     // number of source tomographic bins (= shear_nbin)
  )
{
  double*** dlnC = (double***) malloc3d(NSIZE, nk, nl);
  dC_ks_dlnk_tomo_limber_work(lnkx, nk, lx, nl, NSIZE, 1, dlnC);
  return dlnC;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ks/dlnk at one (k, l) for source bin ni; 0 when the Limber node
// chi = (l + 1/2)/k falls outside the bin's source support.
//
// Point diagnostic: runs the full batch of dlnC_ks_dlnk_grid at a
// single (k, l) and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (k, l, ni) should call the array
// overload once and index the returned array instead.
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1; k <= 0 aborts (spdlog::critical +
//        exit)
//   l  - multipole
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   dlnC_ks/dlnk as a double
// ---------------------------------------------------------------------------
double dlnC_ks_dlnk_tomo_limber_cpp(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "dlnC_ks_dlnk_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "dlnC_ks_dlnk_tomo_limber_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  const double lnkx = std::log(k);
  const double lx = l;
  double*** dlnC = dlnC_ks_dlnk_grid(&lnkx, 1, &lx, 1, NSIZE);
  const double CKS = dlnC[ni][0][0];
  free(dlnC);
  return CKS;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnC_ks/dlnk on a (k, l) grid, scattered from one dlnC_ks_dlnk_grid
// call (one column per source bin).
//
// Parameters:
//   k - wavenumbers in (Mpc/h)^-1; an empty array or any k(m) <= 0
//       aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   numpy array of shape (nk, nl, shear_nbin): leading axis =
//   wavenumber, then multipole, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> dlnC_ks_dlnk_tomo_limber_cpp(
    const arma::Col<double> k,   // wavenumbers in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "dlnC_ks_dlnk_tomo_limber_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dlnC_ks_dlnk_tomo_limber_cpp", nk);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double* lnkx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "dlnC_ks_dlnk_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double*** dlnC = dlnC_ks_dlnk_grid(lnkx, nk, lx, nl, NSIZE);
  arma::field<arma::Mat<double>> dlnCKSdlnk(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> KS(l.n_elem, redshift.shear_nbin, arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        KS(i, nz) = dlnC[nz][m][i];
      }
    }
    dlnCKSdlnk(m) = KS;
  }
  free(dlnC);
  free(lnkx);
  free(lx);
  return to_np3d(dlnCKSdlnk);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RESPONSE FUNCTION (2011.06469 eq 17) - REAL SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_xi_tomo_limber_cpp overloads:
// RF computed by RF_xi_tomo_limber_work on the ln kmax grid, indexed
// [2][NSIZE][nk][Ntheta] with row nz the pair (Z1(nz), Z2(nz)). The
// caller owns (and frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   NSIZE   - number of tomo shear power spectra
//
// Returns:
//   malloc4d array [2][NSIZE][nk][Ntheta] ([0] = xi+, [1] = xi-)
// ---------------------------------------------------------------------------
static double**** RF_xi_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const int NSIZE        // number of tomo shear power spectra
  )
{
  double**** RF = (double****) malloc4d(2, NSIZE, nk, Ntable.Ntheta);
  RF_xi_tomo_limber_work(lnkmaxx, nk, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Real-space response function RF(kmax, theta_nt) (2011.06469 eq 17)
// for the (ni, nj) source pair in either ordering.
//
// Point diagnostic: runs the full batch of RF_xi_grid at a single kmax
// and reads one entry, so it pays the whole-tomography batch cost per
// call. Loops over (kmax, nt, ni, nj) should call the array overload
// once and index the returned arrays instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   nt - angular bin index; outside [0, Ntheta) aborts
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (RF for xi+, RF for xi-) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple RF_xi_tomo_limber_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_xi_tomo_limber_cpp", k);
    exit(1);
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    spdlog::critical("{}: invalid angular bin input nt = {}",
                     "RF_xi_tomo_limber_cpp", nt);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "RF_xi_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  const double lnkmax = std::log(k);
  double**** RF = RF_xi_grid(&lnkmax, 1, NSIZE);
  const int q = N_shear(ni, nj); // symmetric: any (ni, nj) ordering works
  const double RFXIP = RF[0][q][0][nt];
  const double RFXIM = RF[1][q][0][nt];
  free(RF);
  return py::make_tuple(RFXIP, RFXIM);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Real-space response function RF(kmax, theta) (2011.06469 eq 17) on a
// kmax grid, scattered from one RF_xi_grid call with both bin orderings
// filled (xi is symmetric).
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (nk, Ntheta, shear_nbin, shear_nbin): leading axis = kmax, then
//   angular bin, then the source bin pair
// ---------------------------------------------------------------------------
py::tuple RF_xi_tomo_limber_cpp(
    const arma::Col<double> k    // cutoff wavenumbers kmax in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_xi_tomo_limber_cpp", nk);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_xi_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double**** RF = RF_xi_grid(lnkmaxx, nk, NSIZE);
  arma::field<arma::Cube<double>> RFXIP(nk);
  arma::field<arma::Cube<double>> RFXIM(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> XP(Ntable.Ntheta,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> XM(Ntable.Ntheta,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      for (int i=0; i<Ntable.Ntheta; i++) {
        XP(i,z1,z2) = RF[0][nz][m][i];
        XP(i,z2,z1) = XP(i,z1,z2);
        XM(i,z1,z2) = RF[1][nz][m][i];
        XM(i,z2,z1) = XM(i,z1,z2);
      }
    }
    RFXIP(m) = XP;
    RFXIM(m) = XM;
  }
  free(RF);
  free(lnkmaxx);
  return py::make_tuple(to_np4d(RFXIP), to_np4d(RFXIM));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_w_ks_tomo_cpp overloads: RF computed
// by RF_w_ks_tomo_limber_work on the ln kmax grid, indexed
// [NSIZE][nk][Ntheta] with row nz the source bin. The caller owns (and
// frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//
// Returns:
//   malloc3d array [NSIZE][nk][Ntheta]
// ---------------------------------------------------------------------------
static double*** RF_w_ks_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const int NSIZE        // number of source tomographic bins (= shear_nbin)
  )
{
  double*** RF = (double***) malloc3d(NSIZE, nk, Ntable.Ntheta);
  RF_w_ks_tomo_limber_work(lnkmaxx, nk, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear response function RF(kmax, theta_nt) (2011.06469
// eq 17) for source bin ni.
//
// Point diagnostic: runs the full batch of RF_w_ks_grid at a single
// kmax and reads one entry, so it pays the whole-tomography batch cost
// per call. Loops over (kmax, nt, ni) should call the array overload
// once and index the returned array instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   nt - angular bin index; outside [0, Ntheta) aborts
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   RF as a double
// ---------------------------------------------------------------------------
double RF_w_ks_tomo_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni      // source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_w_ks_tomo_cpp", k);
    exit(1);
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    spdlog::critical("{}: invalid angular bin input nt = {}",
                     "RF_w_ks_tomo_cpp", nt);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "RF_w_ks_tomo_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  const double lnkmax = std::log(k);
  double*** RF = RF_w_ks_grid(&lnkmax, 1, NSIZE);
  const double RFWKS = RF[ni][0][nt];
  free(RF);
  return RFWKS;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear response function RF(kmax, theta) (2011.06469
// eq 17) on a kmax grid, scattered from one RF_w_ks_grid call (one
// column per source bin).
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//
// Returns:
//   numpy array of shape (nk, Ntheta, shear_nbin): leading axis =
//   kmax, then angular bin, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> RF_w_ks_tomo_cpp(
    const arma::Col<double> k    // cutoff wavenumbers kmax in (Mpc/h)^-1
  )
{
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_w_ks_tomo_cpp", nk);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_w_ks_tomo_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double*** RF = RF_w_ks_grid(lnkmaxx, nk, NSIZE);
  arma::field<arma::Mat<double>> RFWKS(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> WK(Ntable.Ntheta, redshift.shear_nbin,
                         arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        WK(i, nz) = RF[nz][m][i];
      }
    }
    RFWKS(m) = WK;
  }
  free(RF);
  free(lnkmaxx);
  return to_np3d(RFWKS);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// RESPONSE FUNCTION (2011.06469 eq 17) - FOURIER
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_C_ss_tomo_limber_cpp overloads:
// RF computed by RF_C_ss_tomo_limber_work on the (ln kmax, ell) grid,
// indexed [2][NSIZE][nk][nl] with row nz the pair (Z1(nz), Z2(nz)).
// The caller owns (and frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of tomo shear power spectra
//
// Returns:
//   malloc4d array [2][NSIZE][nk][nl] ([0] = EE, [1] = BB)
// ---------------------------------------------------------------------------
static double**** RF_C_ss_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE        // number of tomo shear power spectra
  )
{
  double**** RF = (double****) malloc4d(2, NSIZE, nk, nl);
  RF_C_ss_tomo_limber_work(lnkmaxx, nk, lx, nl, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fourier-space response function RF(kmax, l) (2011.06469 eq 17) for
// the (ni, nj) source pair in either ordering.
//
// Point diagnostic: runs the full batch of RF_C_ss_grid at a single
// (kmax, l) and reads one entry, so it pays the whole-tomography batch
// cost per call. Loops over (kmax, l, ni, nj) should call the array
// overload once and index the returned arrays instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   l  - multipole
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (RF for EE, RF for BB) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple RF_C_ss_tomo_limber_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_C_ss_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "RF_C_ss_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  const double lnkmax = std::log(k);
  const double lx = l;
  double**** RF = RF_C_ss_grid(&lnkmax, 1, &lx, 1, NSIZE);
  const int q = N_shear(ni, nj); // symmetric: any (ni, nj) ordering works
  const double RFEE = RF[0][q][0][0];
  const double RFBB = RF[1][q][0][0];
  free(RF);
  return py::make_tuple(RFEE, RFBB);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fourier-space response function RF(kmax, l) (2011.06469 eq 17) on a
// (kmax, l) grid, scattered from one RF_C_ss_grid call; only the
// enumerated Z1 <= Z2 ordering is filled, the reversed entries stay
// zero.
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   (EE, BB) tuple of numpy arrays of shape
//   (nk, nl, shear_nbin, shear_nbin): leading axis = kmax, then
//   multipole, then the source bin pair
// ---------------------------------------------------------------------------
py::tuple RF_C_ss_tomo_limber_cpp(
    const arma::Col<double> k,   // cutoff wavenumbers kmax in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "dC_ss_dlnk_tomo_limber_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dC_ss_dlnk_tomo_limber_cpp", nk);
    exit(1);
  } 
  const int NSIZE = tomo.shear_Npowerspectra;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_C_ss_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double**** RF = RF_C_ss_grid(lnkmaxx, nk, lx, nl, NSIZE);
  arma::field<arma::Cube<double>> RFEE(nk);
  arma::field<arma::Cube<double>> RFBB(nk);
  for (int m=0; m<nk; m++) {
    arma::Cube<double> EE(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> BB(l.n_elem,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        EE(i, Z1(nz), Z2(nz)) = RF[0][nz][m][i];
        BB(i, Z1(nz), Z2(nz)) = RF[1][nz][m][i];
      }
    }
    RFEE(m) = EE;
    RFBB(m) = BB;
  }
  free(RF);
  free(lnkmaxx);
  free(lx);
  return py::make_tuple(to_np4d(RFEE), to_np4d(RFBB));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two RF_C_ks_tomo_limber_cpp overloads:
// RF computed by RF_C_ks_tomo_limber_work on the (ln kmax, ell) grid,
// indexed [NSIZE][nk][nl] with row nz the source bin. The caller owns
// (and frees) the returned array.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nk), k in (Mpc/h)^-1
//   nk      - number of kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//
// Returns:
//   malloc3d array [NSIZE][nk][nl]
// ---------------------------------------------------------------------------
static double*** RF_C_ks_grid(
    const double* lnkmaxx, // ln kmax values (length nk), k in (Mpc/h)^-1
    const int nk,          // number of kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE        // number of source tomographic bins (= shear_nbin)
  )
{
  double*** RF = (double***) malloc3d(NSIZE, nk, nl);
  RF_C_ks_tomo_limber_work(lnkmaxx, nk, lx, nl, NSIZE, RF);
  return RF;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear fourier-space response function RF(kmax, l)
// (2011.06469 eq 17) for source bin ni.
//
// Point diagnostic: runs the full batch of RF_C_ks_grid at a single
// (kmax, l) and reads one entry, so it pays the whole-tomography batch
// cost per call. Loops over (kmax, l, ni) should call the array
// overload once and index the returned array instead.
//
// Parameters:
//   k  - cutoff wavenumber kmax in (Mpc/h)^-1; k <= 0 aborts
//        (spdlog::critical + exit)
//   l  - multipole
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   RF as a double
// ---------------------------------------------------------------------------
double RF_C_ks_tomo_limber_cpp(
    const double k,   // cutoff wavenumber kmax in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive",
                     "RF_C_ks_tomo_limber_cpp", k);
    exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "RF_C_ks_tomo_limber_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  const double lnkmax = std::log(k);
  const double lx = l;
  double*** RF = RF_C_ks_grid(&lnkmax, 1, &lx, 1, NSIZE);
  const double RFKS = RF[ni][0][0];
  free(RF);
  return RFKS;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear fourier-space response function RF(kmax, l)
// (2011.06469 eq 17) on a (kmax, l) grid, scattered from one
// RF_C_ks_grid call (one column per source bin).
//
// Parameters:
//   k - cutoff wavenumbers kmax in (Mpc/h)^-1; an empty array or any
//       k(m) <= 0 aborts (spdlog::critical + exit)
//   l - multipoles; an empty array aborts
//
// Returns:
//   numpy array of shape (nk, nl, shear_nbin): leading axis = kmax,
//   then multipole, then source bin
// ---------------------------------------------------------------------------
py::array_t<double,py::array::f_style> RF_C_ks_tomo_limber_cpp(
    const arma::Col<double> k,   // cutoff wavenumbers kmax in (Mpc/h)^-1
    const arma::Col<double> l    // multipoles
  )
{
  const int nl = static_cast<int>(l.n_elem);
  const int nk = static_cast<int>(k.n_elem);
  if (!(nl > 0)) {
    spdlog::critical("{}: l array size = {}", "RF_C_ks_tomo_limber_cpp", nl);
    exit(1);
  }
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "RF_C_ks_tomo_limber_cpp", nk);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double* lnkmaxx = (double*) malloc1d(nk);
  for (int m=0; m<nk; m++) {
    if (!(k(m) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive",
                       "RF_C_ks_tomo_limber_cpp", m, k(m));
      exit(1);
    }
    lnkmaxx[m] = std::log(k(m));
  }
  double* lx = (double*) malloc1d(nl);
  for (int i=0; i<nl; i++) {
    lx[i] = l(i);
  }
  double*** RF = RF_C_ks_grid(lnkmaxx, nk, lx, nl, NSIZE);
  arma::field<arma::Mat<double>> RFKS(nk);
  for (int m=0; m<nk; m++) {
    arma::Mat<double> KS(l.n_elem, redshift.shear_nbin, arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nl; i++) {
        KS(i, nz) = RF[nz][m][i];
      }
    }
    RFKS(m) = KS;
  }
  free(RF);
  free(lnkmaxx);
  free(lx);
  return to_np3d(RFKS);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
