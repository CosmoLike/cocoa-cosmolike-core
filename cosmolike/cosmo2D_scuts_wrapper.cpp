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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// REAL SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(const double k)
{ 
  arma::Cube<double> dlnxp_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  arma::Cube<double> dlnxm_dlnk(Ntable.Ntheta,
                                redshift.shear_nbin,
                                redshift.shear_nbin,
                                arma::fill::zeros);
  
  const int NSIZE = tomo.shear_Npowerspectra;
  double** tmp = dlnxi_dlnk_pm_tomo_nointerp(k);
  for (int nz=0; nz<NSIZE; nz++) {    
    const int z1 = Z1(nz);
    const int z2 = Z2(nz);
    for (int i=0; i<Ntable.Ntheta; i++) {
      const int q = nz * Ntable.Ntheta + i;
      dlnxp_dlnk(i,z1,z2) = tmp[0][q];
      dlnxp_dlnk(i,z2,z1) = tmp[0][q];
      dlnxm_dlnk(i,z1,z2) = tmp[1][q];
      dlnxm_dlnk(i,z2,z1) = tmp[1][q];
    }
  }
  free(tmp);
  return py::make_tuple(carma::cube_to_arr(dlnxp_dlnk), 
                        carma::cube_to_arr(dlnxm_dlnk));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple dlnxi_dlnk_pm_tomo_limber_cpp(const arma::Col<double> k)
{ 
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dlnxi_dlnk_pm_tomo_cpp", nk);
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
    const int NSIZE = tomo.shear_Npowerspectra;
    double** tmp = dlnxi_dlnk_pm_tomo_nointerp(k(m));
    for (int nz=0; nz<NSIZE; nz++) {    
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        tdlnxp_dlnk(i,z1,z2) = tmp[0][q];
        tdlnxp_dlnk(i,z2,z1) = tmp[0][q];
        tdlnxm_dlnk(i,z1,z2) = tmp[1][q];
        tdlnxm_dlnk(i,z2,z1) = tmp[1][q];
      }
    }
    free(tmp);
    dlnxp_dlnk(m) = tdlnxp_dlnk;
    dlnxm_dlnk(m) = tdlnxm_dlnk;
  }
  return py::make_tuple(to_np4d(dlnxp_dlnk), to_np4d(dlnxm_dlnk));
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

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni,
    const int nj
  )
{ // point diagnostic: runs the full batch of dlnC_ss_dlnk_grid at a single
  // (k, l) and reads one entry, so it pays the whole-tomography batch cost
  // per call. Loops over (k, l, ni, nj) should call the array overload
  // once and index the returned arrays instead.
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

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(const arma::Col<double> k, 
                                       const arma::Col<double> l)
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
// RESPONSE FUNCTION (2011.06469 eq 17) - REAL SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple RF_xi_tomo_limber_cpp(
    const double k, 
    const int nt, 
    const int ni, 
    const int nj
  )
{
  const double RFXIP = RF_xi_tomo_limber_nointerp(k, 1, nt, ni, nj, 0);
  const double RFXIM = RF_xi_tomo_limber_nointerp(k, 0, nt, ni, nj, 0);
  return py::make_tuple(RFXIP, RFXIM);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

py::tuple RF_xi_tomo_limber_cpp(const arma::Col<double> k)
{ 
  const int nk = static_cast<int>(k.n_elem);
  if (!(nk > 0)) {
    spdlog::critical("{}: k array size = {}", "dlnxi_dlnk_pm_tomo_cpp", nk);
    exit(1);
  } 
  arma::field<arma::Cube<double>> RFXIP(nk); 
  arma::field<arma::Cube<double>> RFXIM(nk);  
  const int NSIZE = tomo.shear_Npowerspectra;
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
        XP(i,z1,z2) = RF_xi_tomo_limber_nointerp(k(m), 1, i, z1, z2, 0);
        XP(i,z2,z1) = XP(i,z1,z2);
        XM(i,z1,z2) = RF_xi_tomo_limber_nointerp(k(m), 0, i, z1, z2, 0);
        XM(i,z2,z1) = XM(i,z1,z2);
      }
    } 
    RFXIP(m) = XP;
    RFXIM(m) = XM;
  }
  return py::make_tuple(to_np4d(RFXIP), to_np4d(RFXIM));
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

py::tuple RF_C_ss_tomo_limber_cpp(
    const double k,
    const double l,
    const int ni,
    const int nj
  )
{ // point diagnostic: runs the full batch of RF_C_ss_grid at a single
  // (kmax, l) and reads one entry, so it pays the whole-tomography batch
  // cost per call. Loops over (kmax, l, ni, nj) should call the array
  // overload once and index the returned arrays instead.
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

py::tuple RF_C_ss_tomo_limber_cpp(const arma::Col<double> k, 
                                  const arma::Col<double> l)
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

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
