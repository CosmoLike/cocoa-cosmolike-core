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

// ---------------------------------------------------------------------------
// Pybind-facing batch evaluators of the 2D projections: the Python
// side asks for whole data products, not per-point C calls.
//
//   Python -> *_cpp overload (scalar diagnostic or array batch)
//     -> *_nointerp_ells / w_*_tomo batch engines (cosmo2D.c)
//     -> numpy arrays (ell-or-theta, bin_i, bin_j), carma-converted
//
// Layout conventions: rows = angular bin or multipole; the trailing
// axes are tomographic bins, and only the enumerated pairs are filled
// (Z1 <= Z2 for ss, (ZL, ZS) for gs, the diagonal for gg) - all other
// entries stay zero. The scalar overloads are point diagnostics and
// pay the full batch cost per call (see each header).
// ---------------------------------------------------------------------------
namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Area-weighted bin-center angles (arcmin) of the Ntheta angular bins.
// Bin edges are log-spaced between Ntable.vt[RANGE_MIN] and Ntable.vt[RANGE_MAX]
// (radians); each center is the area-weighted mean angle over its
// annulus,
//
//   theta_i = (2/3) (tmax^3 - tmin^3) / (tmax^2 - tmin^2),
//
// divided by 2.90888208665721580e-4 = pi/10800 (one arcmin in radians)
// to convert radians -> arcmin. Same edges as the bin-averaged
// real-space kernels in cosmo2D.c.
//
// Parameters:
//   none (reads Ntable.Ntheta, Ntable.vt[RANGE_MIN], Ntable.vt[RANGE_MAX])
//
// Returns:
//   arma::Col of length Ntable.Ntheta: the bin-center angles in arcmin
// ---------------------------------------------------------------------------
arma::Col<double> get_binning_real_space()
{  
  arma::Col<double> result(Ntable.Ntheta, arma::fill::none);
  const double logdt=(std::log(Ntable.vt[RANGE_MAX])-std::log(Ntable.vt[RANGE_MIN]))/Ntable.Ntheta;
  for (int i = 0; i < Ntable.Ntheta; i++) {  
    const double thetamin = std::exp(log(Ntable.vt[RANGE_MIN]) + (i + 0.0) * logdt);
    const double thetamax = std::exp(log(Ntable.vt[RANGE_MIN]) + (i + 1.0) * logdt);
    const double theta = (2./ 3.) * (std::pow(thetamax,3) - std::pow(thetamin,3)) /
                                    (thetamax*thetamax    - thetamin*thetamin);
    result(i) = theta / 2.90888208665721580e-4; 
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Log-spaced bin-center multipoles of the Ncl fourier-space bins:
//
//   ell_i = exp(ln(lmin) + (i + 1/2) dlnl), dlnl = ln(lmax/lmin)/Ncl,
//   [lmin, lmax] = like.lrange[RANGE_MIN, RANGE_MAX],
//
// the log-space midpoint (geometric center) of each bin.
//
// Parameters:
//   none (reads like.Ncl, like.lrange[RANGE_MIN], like.lrange[RANGE_MAX])
//
// Returns:
//   arma::Col of length like.Ncl: the bin-center multipoles
// ---------------------------------------------------------------------------
arma::Col<double> get_binning_fourier_space()
{  
  arma::Col<double> result(like.Ncl, arma::fill::none);
  const double logdl = (std::log(like.lrange[RANGE_MAX]) -
                        std::log(like.lrange[RANGE_MIN]))/like.Ncl;
  for (int i = 0; i < like.Ncl; i++) {  
    result(i) = std::exp(std::log(like.lrange[RANGE_MIN]) + (i + 0.5)*logdl);
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cosmic shear xi+ and xi- at every angular and tomographic bin, with
// both bin orderings filled (xi is symmetric in ni <-> nj).
//
// Engine: xi_pm_tomo(pm, nt, z1, z2, limber = 1) (full Limber, the only
// supported option) over the enumerated pairs (Z1(nz), Z2(nz)); the
// engine tests pm > 0, so the -1 below selects xi-. Serial loop: the
// first engine call computes and caches the whole (pair, theta) table.
//
// Parameters:
//   none (reads Ntable.Ntheta, redshift.shear_nbin,
//   tomo.shear_Npowerspectra)
//
// Returns:
//   (xi+, xi-) tuple of numpy arrays of shape
//   (Ntheta, shear_nbin, shear_nbin): rows = angular bin, the two
//   trailing axes = source bin pair
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Galaxy-galaxy lensing gamma_t at every angular bin and ggl pair.
//
// Engine: w_gammat_tomo with the limber flag = like.adopt_limber[LIMBER_GS]
// (1 = full Limber; 0 = non-Limber FFTLog + Limber hybrid below
// limits.LMAX_NOLIMBER), so the wrapper follows the likelihood's gs
// Limber choice. Serial loop: the first engine call computes and caches
// the whole (pair, theta) table.
//
// Parameters:
//   none (reads Ntable.Ntheta, tomo.ggl_Npowerspectra, redshift bin
//   counts, like.adopt_limber[LIMBER_GS])
//
// Returns:
//   arma::Cube (Ntheta, clustering_nbin, shear_nbin): rows = angular
//   bin, entry (i, ZL(nz), ZS(nz)) filled for the enumerated ggl pairs
//   only, everything else stays zero
// ---------------------------------------------------------------------------
arma::Cube<double> w_gammat_tomo_cpp()
{  
  arma::Cube<double> result(Ntable.Ntheta,
                            redshift.clustering_nbin, 
                            redshift.shear_nbin,
                            arma::fill::zeros);
  for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      result(i,ZL(nz),ZS(nz)) = w_gammat_tomo(i, ZL(nz), ZS(nz), 
                                               like.adopt_limber[LIMBER_GS]);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy clustering w(theta) at every angular bin (auto pairs only).
//
// Engine: w_gg_tomo with the limber flag = like.adopt_limber[LIMBER_GG]
// (1 = full Limber; 0 = non-Limber FFTLog + Limber hybrid below
// limits.LMAX_NOLIMBER), so the wrapper follows the likelihood's gg
// Limber choice. Serial loop over the auto enumeration
// (clustering_Npowerspectra = clustering_nbin).
//
// Parameters:
//   none (reads Ntable.Ntheta, tomo.clustering_Npowerspectra,
//   redshift.clustering_nbin, like.adopt_limber[LIMBER_GG])
//
// Returns:
//   arma::Cube (Ntheta, clustering_nbin, clustering_nbin): rows =
//   angular bin, only the diagonal (nz, nz) entries filled, cross
//   entries stay zero
// ---------------------------------------------------------------------------
arma::Cube<double> w_gg_tomo_cpp()
{
  arma::Cube<double> result(Ntable.Ntheta,
                            redshift.clustering_nbin,
                            redshift.clustering_nbin,
                            arma::fill::zeros);
  for (int nz=0; nz<tomo.clustering_Npowerspectra; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      result(i, nz, nz) = w_gg_tomo(i, nz, nz, like.adopt_limber[LIMBER_GG]);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB lensing x shear w_ks at every angular bin and source bin (the CMB
// is a single lens plane, so one column per source bin).
//
// Engine: w_ks_tomo(nt, nz, limber = 1) (full Limber, the only
// supported option) over all source bins. Serial loop: the first engine
// call computes and caches the whole (bin, theta) table.
//
// Parameters:
//   none (reads Ntable.Ntheta, redshift.shear_nbin)
//
// Returns:
//   arma::Mat (Ntheta, shear_nbin): rows = angular bin, columns =
//   source bin
// ---------------------------------------------------------------------------
arma::Mat<double> w_ks_tomo_cpp()
{
  arma::Mat<double> result(Ntable.Ntheta,
                           redshift.shear_nbin,
                           arma::fill::zeros);
  for (int nz=0; nz<redshift.shear_nbin; nz++) {
    for (int i=0; i<Ntable.Ntheta; i++) {
      result(i, nz) = w_ks_tomo(i, nz, 1);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared batch engine of the two C_ss_tomo_limber_cpp overloads: a single
// C_ss_tomo_limber_nointerp_ells call fills every enumerated tomographic
// pair at every multipole (row nz of the work arrays is the pair
// (Z1(nz), Z2(nz)) with Z1 <= Z2), and the values are scattered into the
// (ell, ni, nj) cubes; the reversed (nj, ni) entries stay zero.
//
// Parameters:
//   l  - multipole values
//   EE - output EE cube (nell, shear_nbin, shear_nbin)
//   BB - output BB cube (nell, shear_nbin, shear_nbin)
//
// Returns:
//   nothing; the result is written into EE and BB
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

// ---------------------------------------------------------------------------
// Shear-shear Limber C_l at many multipoles, filled by
// C_ss_tomo_limber_cubes (one batched C_ss_tomo_limber_nointerp_ells
// call); only the enumerated Z1 <= Z2 ordering is filled, the reversed
// entries stay zero.
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   (EE, BB) tuple of numpy arrays of shape
//   (nell, shear_nbin, shear_nbin): rows = multipole, the two trailing
//   axes = source bin pair
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

// ---------------------------------------------------------------------------
// Shear-shear Limber C_l at one multipole for the (ni, nj) source pair
// in either ordering.
//
// Point diagnostic: runs the full batch of C_ss_tomo_limber_cubes at a
// single multipole and reads one entry, so it pays the whole-tomography
// batch cost per call. Loops over (l, ni, nj) should call the array
// overload once and index the returned cubes instead.
//
// Parameters:
//   l  - multipole
//   ni - first source redshift bin; outside [0, shear_nbin) aborts
//        (spdlog::critical + exit)
//   nj - second source redshift bin; same validation as ni
//
// Returns:
//   (EE, BB) tuple of doubles
// ---------------------------------------------------------------------------
py::tuple C_ss_tomo_limber_cpp(
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
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

// ---------------------------------------------------------------------------
// The (lens, source) bin indices of every enumerated ggl pair, stored
// as doubles.
//
// Parameters:
//   none (reads tomo.ggl_Npowerspectra and the ZL/ZS enumeration)
//
// Returns:
//   arma::Mat (ggl_Npowerspectra, 2) with row nz = (ZL(nz), ZS(nz))
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Galaxy-galaxy lensing Limber C_l at many multipoles: a single batched
// C_gs_tomo_limber_nointerp_ells call fills every enumerated ggl pair
// at every multipole (row nz of the work array is the pair
// (ZL(nz), ZS(nz))); pairs outside the enumeration stay zero, matching
// the data-vector convention.
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   arma::Cube (nell, clustering_nbin, shear_nbin): rows = multipole,
//   entry (i, ZL(nz), ZS(nz)) filled for the enumerated ggl pairs only
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

// ---------------------------------------------------------------------------
// Galaxy-galaxy lensing Limber C_l at one multipole for the (lens ni,
// source nj) pair.
//
// Point diagnostic: runs the full batch of the array overload above at
// a single multipole and reads one entry, so it pays the
// whole-tomography batch cost per call. Loops over (l, ni, nj) should
// call the array overload once and index the returned cube instead.
//
// Parameters:
//   l  - multipole
//   ni - lens redshift bin; outside [0, clustering_nbin) aborts
//        (spdlog::critical + exit)
//   nj - source redshift bin; outside [0, shear_nbin) aborts
//
// Returns:
//   C_l^gs of the (ni, nj) pair; 0 for a pair outside the enumerated
//   ggl list
// ---------------------------------------------------------------------------
double C_gs_tomo_limber_cpp(
    const double l,   // multipole
    const int ni,     // lens redshift bin
    const int nj      // source redshift bin
  )
{
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

// ---------------------------------------------------------------------------
// Galaxy-clustering Limber C_l at every auto pair and many multipoles:
// a single batched C_gg_tomo_limber_nointerp_ells call fills every lens
// bin at every multipole (the likelihood's auto-only gg enumeration).
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   arma::Cube (nell, clustering_nbin, clustering_nbin): rows =
//   multipole, only the diagonal (nz, nz) entries filled, cross entries
//   stay zero
// ---------------------------------------------------------------------------
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
  const int nell = static_cast<int>(l.n_elem);
  const int NSIZE = redshift.clustering_nbin;
  double** tmp = (double**) malloc2d(NSIZE, nell);
  C_gg_tomo_limber_nointerp_ells(l.memptr(), nell, NSIZE, tmp);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      result(i, nz, nz) = tmp[nz][i];
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy-clustering Limber C_l at one multipole and one auto pair
// (nz, nz).
//
// Point diagnostic: runs the batch of the array overload above at a
// single multipole (every lens bin) and reads one entry, so it pays
// the whole-tomography batch cost per call. Loops over (l, nz) should
// call the array overload once and index the returned cube instead.
//
// Parameters:
//   l  - multipole
//   nz - lens redshift bin (auto pair nz-nz); outside
//        [0, clustering_nbin) aborts (spdlog::critical + exit)
//
// Returns:
//   C_l^gg of the (nz, nz) auto pair
// ---------------------------------------------------------------------------
double C_gg_tomo_limber_cpp(
    const double l,   // multipole
    const int nz      // lens redshift bin (auto pair nz-nz)
  )
{
  if (nz < 0 || nz > redshift.clustering_nbin - 1) {
    spdlog::critical("{}: invalid bin input nz = {}",
                     "C_gg_tomo_limber_cpp", nz);
    exit(1);
  }
  arma::Col<double> ell(1);
  ell(0) = l;
  const arma::Cube<double> res = C_gg_tomo_limber_cpp(ell);
  return res(0, nz, nz);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy-clustering C_l with the non-limber low multipoles the
// likelihood uses (limber above limits.LMAX_NOLIMBER): starts from
// C_gg_tomo_limber_cpp (same cube layout, diagonal only) and overwrites
// every requested l < LMAX_NOLIMBER with the non-Limber C_cl_tomo value
// (Limber-convergence tolerance 0.01) read at the integer multipole
// (int)(l + 1e-13) - the requested low multipoles are expected to be
// integer-valued.
//
// Parameters:
//   l - multipole values; an empty array aborts (spdlog::critical +
//       exit)
//
// Returns:
//   arma::Cube (nell, clustering_nbin, clustering_nbin): rows =
//   multipole, only the diagonal (nz, nz) entries filled
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Galaxy x CMB-lensing Limber C_l at one multipole and lens bin: a
// direct C_gk_tomo_limber_nointerp call.
//
// Point diagnostic: the engine runs the full batch of
// C_gk_tomo_limber_nointerp_ells at a single multipole and reads one
// entry, so it pays the whole-tomography batch cost per call. Loops
// over (l, ni) should call the array overload once and index the
// returned matrix instead.
//
// Parameters:
//   l  - multipole
//   ni - lens redshift bin; validated by the engine (log_fatal + exit
//        outside [0, clustering_nbin))
//
// Returns:
//   C_l^gk of lens bin ni
// ---------------------------------------------------------------------------
double C_gk_tomo_limber_cpp(
    const double l,   // multipole
    const int ni      // lens redshift bin
  )
{
  return C_gk_tomo_limber_nointerp(l, ni);
}

// ---------------------------------------------------------------------------
// Galaxy x CMB-lensing Limber C_l at every lens bin and many
// multipoles: a single batched C_gk_tomo_limber_nointerp_ells call
// fills every lens bin at every multipole (the CMB is a single source
// plane, so one spectrum per lens bin).
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   arma::Mat (nell, clustering_nbin): rows = multipole, columns =
//   lens bin
// ---------------------------------------------------------------------------
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
  const int nell = (int) l.n_elem;
  const int NSIZE = redshift.clustering_nbin;
  double** tmp = (double**) malloc2d(NSIZE, nell);
  C_gk_tomo_limber_nointerp_ells(l.memptr(), nell, NSIZE, tmp);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      result(i, nz) = tmp[nz][i];
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear Limber C_l at one multipole and source bin.
//
// Point diagnostic: runs the full batch of
// C_ks_tomo_limber_nointerp_ells at a single multipole and reads one
// entry, so it pays the whole-tomography batch cost per call. Loops
// over (l, ni) should call the array overload once and index the
// returned matrix instead.
//
// Parameters:
//   l  - multipole
//   ni - source redshift bin; outside [0, shear_nbin) aborts
//        (spdlog::critical + exit)
//
// Returns:
//   C_l^ks of source bin ni
// ---------------------------------------------------------------------------
double C_ks_tomo_limber_cpp(
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {}",
                     "C_ks_tomo_limber_cpp", ni);
    exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double** tmp = (double**) malloc2d(NSIZE, 1);
  double ell[1] = {l};
  C_ks_tomo_limber_nointerp_ells(ell, 1, NSIZE, tmp);
  const double res = tmp[ni][0];
  free(tmp);
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear Limber C_l at every source bin and many
// multipoles: a single batched C_ks_tomo_limber_nointerp_ells call
// fills every source bin at every multipole (the CMB is a single lens
// plane, so one spectrum per source bin).
//
// Parameters:
//   l - multipole values (need not be integers); an empty array aborts
//       (spdlog::critical + exit)
//
// Returns:
//   arma::Mat (nell, shear_nbin): rows = multipole, columns = source
//   bin
// ---------------------------------------------------------------------------
arma::Mat<double> C_ks_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  if (!(l.n_elem > 0)) {
    spdlog::critical("{}: l array size = {}",
                     "C_ks_tomo_limber_cpp",
                     l.n_elem);
    exit(1);
  }
  arma::Mat<double> result(l.n_elem, redshift.shear_nbin);
  const int nell = (int) l.n_elem;
  const int NSIZE = redshift.shear_nbin;
  double** tmp = (double**) malloc2d(NSIZE, nell);
  C_ks_tomo_limber_nointerp_ells(l.memptr(), nell, NSIZE, tmp);
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      result(i, nz) = tmp[nz][i];
    }
  }
  free(tmp);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing auto Limber C_l at one multipole: a direct
// C_kk_limber_nointerp(l, init = 0) call (init = 0 computes).
//
// Parameters:
//   l - multipole
//
// Returns:
//   C_l^kk at multipole l
// ---------------------------------------------------------------------------
double C_kk_limber_cpp(
    const double l    // multipole
  )
{
  return C_kk_limber_nointerp(l, 0);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing auto Limber C_l at many multipoles. The serial init = 1
// warmup call populates lazily initialized statics down the call chain
// so the OpenMP (omp parallel for) loop of init = 0 calls is race-free.
//
// Parameters:
//   l - multipole values; an empty array aborts (spdlog::critical +
//       exit)
//
// Returns:
//   arma::Col of length nell: entry i = C_l^kk at l(i)
// ---------------------------------------------------------------------------
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

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
