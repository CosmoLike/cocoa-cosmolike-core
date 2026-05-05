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

py::tuple dlnC_ss_dlnk_tomo_limber_cpp(
    const double k, 
    const double l, 
    const int ni, 
    const int nj
  )
{
  double CEE = 0;
  double CBB = 0;
  {
    const int EE = 1;
    const double dC = dC_ss_dlnk_tomo_limber_nointerp(k, l, ni, nj, EE);
    const double C = (fabs(dC) > 1e-30) ? 
                     C_ss_tomo_limber_nointerp(l, ni, nj, EE, 0) : 1;
    CEE = (fabs(C) > 1e-30) ? dC/C : 0.0; 
  }
  {
    const int EE = 0;
    const double dC = dC_ss_dlnk_tomo_limber_nointerp(k, l, ni, nj, EE);
    const double C = (fabs(dC) > 1e-30) ? 
                     C_ss_tomo_limber_nointerp(l, ni, nj, EE, 0) : 1;
    CBB = (fabs(C) > 1e-30) ? dC/C : 0.0; 
  }
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
  for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
    const double Z1NZ = Z1(nz);
    const double Z2NZ = Z2(nz);
    (void) dC_ss_dlnk_tomo_limber_nointerp(k(0), l(0), Z1NZ, Z2NZ, 1); // EE
    (void) C_ss_tomo_limber_nointerp(l(0), Z1NZ, Z2NZ, 1, 1);
    (void) dC_ss_dlnk_tomo_limber_nointerp(k(0), l(0), Z1NZ, Z2NZ, 0); // BB
    (void) C_ss_tomo_limber_nointerp(l(0), Z1NZ, Z2NZ, 1, 1);
  }
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
    #pragma omp parallel for collapse(2)
    for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
      for (int i=0; i<nl; i++) {
        const int ni = Z1(nz);
        const int nj = Z2(nz);
        const double lx = l(i);
        const double kx = k(m);
        {
          const double dC = dC_ss_dlnk_tomo_limber_nointerp(kx, lx, ni, nj, 1);
          const double C = (fabs(dC) > 1e-30) ? 
                           C_ss_tomo_limber_nointerp(lx, ni, nj, 1, 0) : 1;
          EE(i, ni, nj) = (fabs(C) > 1e-30) ? dC/C : 0.0; 
        }
        {
          const double dC = dC_ss_dlnk_tomo_limber_nointerp(kx, lx, ni, nj, 0);
          const double C = (fabs(dC) > 1e-30) ? 
                           C_ss_tomo_limber_nointerp(lx, ni, nj, 0, 0) : 1;
          BB(i, ni, nj) = (fabs(C) > 1e-30) ? dC/C : 0.0; 
        }
      }
    }
    dlnCEEdlnk(m) = EE;
    dlnCBBdlnk(m) = BB;
  }
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

py::tuple RF_C_ss_tomo_limber_cpp(
    const double k, 
    const double l, 
    const int ni, 
    const int nj
  )
{
  const double RFEE = RF_C_ss_tomo_limber_nointerp(k, l, ni, nj, 1, 0);
  const double RFBB = RF_C_ss_tomo_limber_nointerp(k, l, ni, nj, 0, 0);
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
  for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) { // init static vars
    const double Z1NZ = Z1(nz);
    const double Z2NZ = Z2(nz);
    (void) RF_C_ss_tomo_limber_nointerp(k(0), l(0), Z1NZ, Z2NZ, 1, 1);
    (void) RF_C_ss_tomo_limber_nointerp(k(0), l(0), Z1NZ, Z2NZ, 0, 1);
  }
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
    #pragma omp parallel for collapse(2)
    for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
      for (int i=0; i<nl; i++) {
        const int ni = Z1(nz);
        const int nj = Z2(nz);
        const double lx = l(i);
        const double kx = k(m);
        EE(i, ni, nj) = RF_C_ss_tomo_limber_nointerp(kx, lx, ni, nj, 1, 0);
        BB(i, ni, nj) = RF_C_ss_tomo_limber_nointerp(kx, lx, ni, nj, 0, 0);
      }
    }
    RFEE(m) = EE;
    RFBB(m) = BB;
  }
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
