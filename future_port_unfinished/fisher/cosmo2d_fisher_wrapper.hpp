#include <carma.h>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#ifndef __COSMOLIKE_COSMO2D_FISHER_WRAPPER_HPP
#define __COSMOLIKE_COSMO2D_FISHER_WRAPPER_HPP

// EXPERIMENTAL (future_port_unfinished/fisher): python side of
// cosmo2d_fisher.c. See fisher_derivatives.pdf and README.md.

namespace cosmolike_interface
{

// Load the response of the set_cosmology tables to the parameter in slot
// ip; the grids must be the ones of the current set_cosmology call.
void set_fisher_response_cpp(
    const int ip,                      // parameter slot (0..FISHER_NPARAM_MAX-1)
    const double dlnOm_dX,             // explicit dlnOmega_m/dX
    const arma::Col<double> z_chi,     // chi-table redshifts
    const arma::Col<double> dchi_dX,   // dchi/dX in Mpc/h
    const arma::Col<double> z_G,       // growth-table redshifts
    const arma::Col<double> dlnG_dX,   // dlnG/dX
    const arma::Col<double> log10k,    // log10(k [h/Mpc]) of the P table
    const arma::Col<double> z_P,       // redshifts of the P table
    const arma::Col<double> dlnPNL_dX  // dlnP_NL/dX, (z, k) table flattened
                                       // in Fortran order (as lnP_nonlinear)
  );

// Forget every loaded response.
void reset_fisher_response_cpp();

// C_ss (NLA E-mode) and dC_ss/dX for every loaded parameter.
py::tuple dC_ss_dX_tomo_limber_cpp(
    const arma::Col<double> l    // multipoles
  ); // returns (C (nl, nbin, nbin), dC (nparam, nl, nbin, nbin))

// xi_pm and dxi_pm/dX for every loaded parameter on the Ntable binning.
py::tuple dxi_pm_dX_tomo_cpp(
  ); // returns (xi+, xi-, dxi+, dxi-): xi (Ntheta, nbin, nbin),
     // dxi (nparam, Ntheta, nbin, nbin)

}  // namespace cosmolike_interface
#endif // HEADER GUARD
