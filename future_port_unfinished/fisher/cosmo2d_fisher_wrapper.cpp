#include <string>
#include <vector>
#include <cmath>

// SPDLOG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo2D_wrapper.hpp"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"
#include "cosmo2d_fisher.h"
#include "cosmo2d_fisher_wrapper.hpp"

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// The response tables must sit on the exact grids of the current
// set_cosmology call: the C side reads them node by node against
// cosmology.chi / cosmology.G / cosmology.lnP. This checks sizes and
// every grid node, the way set_non_linear_power_spectrum checks its input,
// then hands the tables to set_fisher_response. The P response arrives in
// the layout of lnP_nonlinear (a (z, k) table flattened in Fortran order,
// element (ik, jz) at ik*nz + jz), which is also what the C side expects.
// ---------------------------------------------------------------------------
void set_fisher_response_cpp(
    const int ip,                      // parameter slot
    const double dlnOm_dX,             // explicit dlnOmega_m/dX
    const arma::Col<double> z_chi,     // chi-table redshifts
    const arma::Col<double> dchi_dX,   // dchi/dX in Mpc/h
    const arma::Col<double> z_G,       // growth-table redshifts
    const arma::Col<double> dlnG_dX,   // dlnG/dX
    const arma::Col<double> log10k,    // log10(k [h/Mpc]) of the P table
    const arma::Col<double> z_P,       // redshifts of the P table
    const arma::Col<double> dlnPNL_dX  // dlnP_NL/dX (Fortran-flattened)
  )
{
  const char* fname = "set_fisher_response_cpp";
  if (NULL == cosmology.chi || NULL == cosmology.G || NULL == cosmology.lnP) {
    spdlog::critical("{}: call set_cosmology before loading a response", fname);
    exit(1);
  }
  const int nchi = cosmology.chi_nz;
  const int nG   = cosmology.G_nz;
  const int nk   = cosmology.lnP_nk;
  const int nz   = cosmology.lnP_nz;
  if (static_cast<int>(z_chi.n_elem) != nchi ||
      static_cast<int>(dchi_dX.n_elem) != nchi ||
      static_cast<int>(z_G.n_elem) != nG ||
      static_cast<int>(dlnG_dX.n_elem) != nG ||
      static_cast<int>(log10k.n_elem) != nk ||
      static_cast<int>(z_P.n_elem) != nz ||
      static_cast<int>(dlnPNL_dX.n_elem) != nk*nz) {
    spdlog::critical("{}: response sizes do not match the set_cosmology "
                     "tables (chi {}, G {}, P {} x {})", fname, nchi, nG, nk, nz);
    exit(1);
  }
  for (int j=0; j<nchi; j++) {
    if (fdiff(cosmology.chi[0][j], z_chi(j))) {
      spdlog::critical("{}: chi-table z grid differs at node {}", fname, j);
      exit(1);
    }
  }
  for (int j=0; j<nG; j++) {
    if (fdiff(cosmology.G[0][j], z_G(j))) {
      spdlog::critical("{}: growth-table z grid differs at node {}", fname, j);
      exit(1);
    }
  }
  for (int i=0; i<nk; i++) {
    if (fdiff(cosmology.lnP[i][nz], log10k(i))) {
      spdlog::critical("{}: P-table log10k grid differs at node {} (for "
                       "X = h, interpolate the perturbed tables onto the "
                       "fiducial h/Mpc grid before differencing)", fname, i);
      exit(1);
    }
  }
  for (int j=0; j<nz; j++) {
    if (fdiff(cosmology.lnP[nk][j], z_P(j))) {
      spdlog::critical("{}: P-table z grid differs at node {}", fname, j);
      exit(1);
    }
  }
  set_fisher_response(ip,
                      dlnOm_dX,
                      dchi_dX.memptr(),
                      nchi,
                      dlnG_dX.memptr(),
                      nG,
                      dlnPNL_dX.memptr(),
                      nk,
                      nz);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void reset_fisher_response_cpp()
{
  reset_fisher_response();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// C_ss and dC_ss/dX: one batch call fills every enumerated pair; the cubes
// are symmetrized in (ni, nj) like C_ss_tomo_limber_cpp's
py::tuple dC_ss_dX_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  )
{
  const int nell = static_cast<int>(l.n_elem);
  if (!(nell > 0)) {
    spdlog::critical("{}: l array size = {}", "dC_ss_dX_tomo_limber_cpp", nell);
    exit(1);
  }
  const int np = fisher_nparam();
  const int NSIZE = tomo.shear_Npowerspectra;
  double*** tmp = (double***) malloc3d(1 + np, NSIZE, nell);
  dC_ss_dX_tomo_limber_nointerp_ells(l.memptr(), nell, NSIZE, tmp);

  arma::field<arma::Cube<double>> out(1 + np);
  for (int m=0; m<1 + np; m++) {
    arma::Cube<double> c(nell,
                         redshift.shear_nbin,
                         redshift.shear_nbin,
                         arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      for (int i=0; i<nell; i++) {
        c(i, z1, z2) = tmp[m][nz][i];
        c(i, z2, z1) = tmp[m][nz][i];
      }
    }
    out(m) = c;
  }
  free(tmp);
  arma::field<arma::Cube<double>> dC(np);
  for (int ip=0; ip<np; ip++) {
    dC(ip) = out(1 + ip);
  }
  return py::make_tuple(carma::cube_to_arr(out(0)), to_np4d(dC));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// xi_pm and dxi_pm/dX on the Ntable binning (symmetrized in (ni, nj))
py::tuple dxi_pm_dX_tomo_cpp()
{
  const int np = fisher_nparam();
  const int NSIZE = tomo.shear_Npowerspectra;
  double**** tmp = (double****) malloc4d(1 + np, 2, NSIZE, Ntable.Ntheta);
  dxi_pm_dX_tomo(tmp);

  arma::field<arma::Cube<double>> xp(1 + np);
  arma::field<arma::Cube<double>> xm(1 + np);
  for (int m=0; m<1 + np; m++) {
    arma::Cube<double> cp(Ntable.Ntheta,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    arma::Cube<double> cm(Ntable.Ntheta,
                          redshift.shear_nbin,
                          redshift.shear_nbin,
                          arma::fill::zeros);
    for (int nz=0; nz<NSIZE; nz++) {
      const int z1 = Z1(nz);
      const int z2 = Z2(nz);
      for (int i=0; i<Ntable.Ntheta; i++) {
        cp(i, z1, z2) = tmp[m][0][nz][i];
        cp(i, z2, z1) = tmp[m][0][nz][i];
        cm(i, z1, z2) = tmp[m][1][nz][i];
        cm(i, z2, z1) = tmp[m][1][nz][i];
      }
    }
    xp(m) = cp;
    xm(m) = cm;
  }
  free(tmp);
  arma::field<arma::Cube<double>> dxp(np);
  arma::field<arma::Cube<double>> dxm(np);
  for (int ip=0; ip<np; ip++) {
    dxp(ip) = xp(1 + ip);
    dxm(ip) = xm(1 + ip);
  }
  return py::make_tuple(carma::cube_to_arr(xp(0)),
                        carma::cube_to_arr(xm(0)),
                        to_np4d(dxp),
                        to_np4d(dxm));
}

}  // namespace cosmolike_interface
