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
#include "cosmolike/cosmo2D_scuts.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using cube = arma::Cube<double>;

namespace cosmolike_interface
{

// 1 if any lens bin carries a nonzero second-order galaxy bias
static int has_b2_galaxies()
{
  int res = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) 
    if (nuisance.gb[1][i])
      res = 1;
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// area-weighted bin-center angles (arcmin) of the Ntheta angular bins
arma::Col<double> get_binning_real_space()
{  
  arma::Col<double> result(Ntable.Ntheta, arma::fill::none);
  const double logdt=(std::log(Ntable.vtmax)-std::log(Ntable.vtmin))/Ntable.Ntheta;
  for (int i = 0; i < Ntable.Ntheta; i++) {  
    const double thetamin = std::exp(log(Ntable.vtmin) + (i + 0.0) * logdt);
    const double thetamax = std::exp(log(Ntable.vtmin) + (i + 1.0) * logdt);
    const double theta = (2./ 3.) * (std::pow(thetamax,3) - std::pow(thetamin,3)) /
                                    (thetamax*thetamax    - thetamin*thetamin);
    result(i) = theta / 2.90888208665721580e-4; 
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// log-spaced bin-center multipoles of the Ncl fourier-space bins
arma::Col<double> get_binning_fourier_space()
{  
  arma::Col<double> result(like.Ncl, arma::fill::none);
  const double logdl = (std::log(like.lmax) - std::log(like.lmin))/like.Ncl;
  for (int i = 0; i < like.Ncl; i++) {  
    result(i) = std::exp(std::log(like.lmin) + (i + 0.5)*logdl);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// cosmic shear xi+ and xi- at every angular and tomographic bin (both
// bin orderings filled: xi is symmetric)
py::tuple xi_pm_tomo_cpp()
{ 
  arma::Cube<double> xp(Ntable.Ntheta,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  arma::Cube<double> xm(Ntable.Ntheta,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {    
    for (int i=0; i<Ntable.Ntheta; i++) {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      xp(i,z1,z2) = xi_pm_tomo(1, i, z1, z2, 1);
      xm(i,z1,z2) = xi_pm_tomo(-1, i, z1, z2, 1);
      xp(i,z2,z1) = xp(i,z1,z2);
      xm(i,z2,z1) = xm(i,z1,z2);
    }
  }
  return py::make_tuple(carma::cube_to_arr(xp), carma::cube_to_arr(xm));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy-galaxy lensing gamma_t at every angular bin and ggl pair
arma::Cube<double> w_gammat_tomo_cpp()
{  
  arma::Cube<double> result(Ntable.Ntheta,
                            redshift.clustering_nbin, 
                            redshift.shear_nbin,
                            arma::fill::zeros);
  for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      result(i,ZL(nz),ZS(nz)) = w_gammat_tomo(i, ZL(nz), ZS(nz), 1);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy clustering w(theta) at every angular bin (auto pairs only)
arma::Cube<double> w_gg_tomo_cpp()
{
  arma::Cube<double> result(Ntable.Ntheta,
                            redshift.clustering_nbin,
                            redshift.clustering_nbin,
                            arma::fill::zeros);
  for (int nz=0; nz<tomo.clustering_Npowerspectra; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      result(i, nz, nz) = w_gg_tomo(i, nz, nz, 0);
    }
  }
  return result;
}

/*

arma::Col<double> w_gk_tomo_cpp()
{
  arma::Col<double> result(Ntable.Ntheta*redshift.clustering_nbin,arma::fill::none);
  for (int nz=0; nz<redshift.clustering_nbin; nz++)
    for (int i=0; i<Ntable.Ntheta; i++)
      result(Ntable.Ntheta*nz+i) = w_gk_tomo(i, nz, 1);
  return result;
}

arma::Col<double> w_ks_tomo_cpp()
{
  arma::Col<double> result(Ntable.Ntheta*redshift.shear_nbin,arma::fill::none);
  for (int nz=0; nz<redshift.clustering_nbin; nz++)
    for (int i=0; i<Ntable.Ntheta; i++)
      result(Ntable.Ntheta*nz+i) = w_ks_tomo(i, nz, 1);
  return result;
}
*/

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two C_ss_tomo_limber_cpp overloads: a single
// C_ss_tomo_limber_nointerp_ells call fills every enumerated tomographic
// pair at every multipole (row nz of the work arrays is the pair
// (Z1(nz), Z2(nz)) with Z1 <= Z2), and the values are scattered into the
// (ell, ni, nj) cubes; the reversed (nj, ni) entries stay zero.
// ---------------------------------------------------------------------------
static void C_ss_tomo_limber_cubes(
    const arma::Col<double>& l, // multipole values
    arma::Cube<double>& EE,     // output (nell, shear_nbin, shear_nbin)
    arma::Cube<double>& BB      // output (nell, shear_nbin, shear_nbin)
  )
{
  const int nell = (int) l.n_elem;
  const int NSIZE = tomo.shear_Npowerspectra;
  double** tmp_EE = (double**) malloc2d(NSIZE, nell);
  double** tmp_BB = (double**) malloc2d(NSIZE, nell);
  C_ss_tomo_limber_nointerp_ells(l.memptr(), nell, NSIZE, tmp_EE, tmp_BB);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      EE(i, Z1(nz), Z2(nz)) = tmp_EE[nz][i];
      BB(i, Z1(nz), Z2(nz)) = tmp_BB[nz][i];
    }
  }
  free(tmp_EE);
  free(tmp_BB);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple C_ss_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (!(l.n_elem > 0)) {
    spdlog::critical("{}: l array size = {}", "C_ss_tomo_limber_cpp", l.n_elem);
    exit(1);
  }
  arma::Cube<double> EE(l.n_elem,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  arma::Cube<double> BB(l.n_elem,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  C_ss_tomo_limber_cubes(l, EE, BB);
  return py::make_tuple(carma::cube_to_arr(EE), carma::cube_to_arr(BB));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple C_ss_tomo_limber_cpp(
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{ // point diagnostic: runs the full batch of C_ss_tomo_limber_cubes at a
  // single multipole and reads one entry, so it pays the whole-tomography
  // batch cost per call. Loops over (l, ni, nj) should call the array
  // overload once and index the returned cubes instead.
  if (ni < 0 || ni > redshift.shear_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "C_ss_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  arma::Col<double> ell(1);
  ell(0) = l;
  arma::Cube<double> EE(1,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  arma::Cube<double> BB(1,
                        redshift.shear_nbin,
                        redshift.shear_nbin,
                        arma::fill::zeros);
  C_ss_tomo_limber_cubes(ell, EE, BB);
  // C_ss is symmetric in (ni, nj) and the cubes fill only the
  // enumerated Z1 <= Z2 ordering, so read the ordered entry
  const int zmin = (ni < nj) ? ni : nj;
  const int zmax = (ni < nj) ? nj : ni;
  return py::make_tuple(EE(0, zmin, zmax), BB(0, zmin, zmax));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// the (lens, source) bin indices of every enumerated ggl pair
arma::Mat<double> gs_bins()
{
  arma::Mat<double> result(tomo.ggl_Npowerspectra, 2);
  for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
    result(nz,0) = ZL(nz);
    result(nz,1) = ZS(nz);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Cube<double> C_gs_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (!(l.n_elem > 0)) {
    spdlog::critical("{}: l array size = {}",
                     "C_gs_tomo_limber_cpp",
                     l.n_elem);
    exit(1);
  }
  arma::Cube<double> result(l.n_elem,
                            redshift.clustering_nbin,
                            redshift.shear_nbin,
                            arma::fill::zeros);
  // batched computation: a single C_gs_tomo_limber_nointerp_ells call
  // fills every enumerated ggl pair at every multipole (row nz of the
  // work array is the pair (ZL(nz), ZS(nz))); pairs outside the
  // enumeration stay zero, matching the data-vector convention
  const int nell = (int) l.n_elem;
  const int NSIZE = tomo.ggl_Npowerspectra;
  double** tmp = (double**) malloc2d(NSIZE, nell);
  C_gs_tomo_limber_nointerp_ells(l.memptr(), nell, NSIZE, tmp);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      result(i, ZL(nz), ZS(nz)) = tmp[nz][i];
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gs_tomo_limber_cpp(
    const double l,   // multipole
    const int ni,     // lens redshift bin
    const int nj      // source redshift bin
  )
{ // point diagnostic: runs the full batch of the array overload above at
  // a single multipole and reads one entry, so it pays the
  // whole-tomography batch cost per call. Loops over (l, ni, nj) should
  // call the array overload once and index the returned cube instead.
  // A pair outside the enumerated ggl list returns 0.
  if (ni < 0 || ni > redshift.clustering_nbin - 1 ||
      nj < 0 || nj > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input (ni, nj) = ({}, {})",
                     "C_gs_tomo_limber_cpp", ni, nj);
    exit(1);
  }
  arma::Col<double> ell(1);
  ell(0) = l;
  const arma::Cube<double> res = C_gs_tomo_limber_cpp(ell);
  return res(0, ni, nj);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy-clustering C_l (limber) at one multipole and one auto pair
double C_gg_tomo_limber_cpp(
    const double l,   // multipole
    const int nz      // lens redshift bin (auto pair nz-nz)
  )
{
  return C_gg_tomo_limber_nointerp(l, nz, nz, 0);
}

// ---------------------------------------------------------------------------

// galaxy-clustering C_l (limber) at every auto pair and many multipoles
arma::Cube<double> C_gg_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_gg_tomo_limber_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Cube<double> result(l.n_elem, 
                            redshift.clustering_nbin,
                            redshift.clustering_nbin,
                            arma::fill::zeros);
  for (int nz=0; nz<redshift.clustering_nbin; nz++) { // init static vars
    (void) C_gg_tomo_limber_nointerp(l(0), 0, 0, 1);
  }
  #pragma omp parallel for collapse(2)
  for (int nz=0; nz<redshift.clustering_nbin; nz++) {
    for (int i=0; i<static_cast<int>(l.n_elem); i++) {
      result(i, nz, nz) = C_gg_tomo_limber_nointerp(l(i), nz, nz, 0);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy-clustering C_l with the non-limber low multipoles the
// likelihood uses (limber above limits.LMAX_NOLIMBER)
arma::Cube<double> C_gg_tomo_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_gg_tomo_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Cube<double> result = C_gg_tomo_limber_cpp(l);
  arma::uvec idxs = arma::find(l < limits.LMAX_NOLIMBER);
  if (idxs.n_elem > 0) {
    const double tolerance = 0.01;

    double** Cl = (double**) malloc2d(redshift.clustering_nbin, 
                                      limits.LMAX_NOLIMBER + 1);

    C_cl_tomo((double* const* const) Cl, tolerance);

    for (int nz = 0; nz < redshift.clustering_nbin; nz++) {
      for (int i = 0; i < static_cast<int>(idxs.n_elem); i++) {
        result(idxs(i), nz, nz) = Cl[nz][static_cast<int>(l(idxs(i)) + 1e-13)];
      }
    }

    free((void*) Cl);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy x CMB-lensing C_l (limber) at one multipole and lens bin
double C_gk_tomo_limber_cpp(
    const double l,   // multipole
    const int ni      // lens redshift bin
  )
{
  return C_gk_tomo_limber_nointerp(l, ni, 0);
}

// galaxy x CMB-lensing C_l (limber) at every lens bin and many multipoles
arma::Mat<double> C_gk_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_gk_tomo_limber_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Mat<double> result(l.n_elem, redshift.clustering_nbin);
  for (int nz=0; nz<redshift.clustering_nbin; nz++) { // init static vars
    (void) C_gk_tomo_limber_nointerp(l(0), nz, 1);
  }
  #pragma omp parallel for collapse(2)
  for (int nz=0; nz<redshift.clustering_nbin; nz++) {
    for (int i=0; i<static_cast<int>(l.n_elem); i++) {
      result(i, nz) = C_gk_tomo_limber_nointerp(l(i), nz, 0);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// CMB-lensing x shear C_l (limber) at one multipole and source bin
double C_ks_tomo_limber_cpp(
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  return C_ks_tomo_limber_nointerp(l, ni, 0);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// CMB-lensing x shear C_l (limber) at every source bin and many multipoles
arma::Mat<double> C_ks_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_ks_tomo_limber_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Mat<double> result(l.n_elem, redshift.shear_nbin);
  for (int nz=0; nz<redshift.shear_nbin; nz++) { // init static vars
    (void) C_ks_tomo_limber_nointerp(l(0), nz, 1);
  }
  #pragma omp parallel for collapse(2)
  for (int nz=0; nz<redshift.shear_nbin; nz++) {
    for (int i=0; i<static_cast<int>(l.n_elem); i++) {
      result(i, nz) = C_ks_tomo_limber_nointerp(l(i), nz, 0);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// CMB-lensing auto C_l (limber) at one multipole
double C_kk_limber_cpp(
    const double l    // multipole
  )
{
  return C_kk_limber_nointerp(l, 0);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// CMB-lensing auto C_l (limber) at many multipoles
arma::Col<double> C_kk_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (l.n_elem == 0) {
    spdlog::critical("{}: l array size = {}", 
                     "C_kk_limber_cpp", 
                     l.n_elem);
    exit(1);
  }
  arma::Col<double> result(l.n_elem);
  (void) C_kk_limber_nointerp(l(0), 1);  // init static vars
  #pragma omp parallel for 
  for (int i=0; i<static_cast<int>(l.n_elem); i++) {
    result(i) = C_kk_limber_nointerp(l(i), 0);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
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
