#ifndef __COSMOLIKE_COSMO2D_FISHER_H
#define __COSMOLIKE_COSMO2D_FISHER_H
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// EXPERIMENTAL (future_port_unfinished/fisher): analytic derivatives of the
// Limber C_ss and of xi_pm with respect to cosmological parameters, from
// response tables supplied by Python. Not compiled by any project. The math
// and its verification are in fisher_derivatives.pdf next to this file.
//
// For a parameter X the chain rule needs four inputs (the "response"):
//   dlnP_NL/dX (k, z)  at fixed (k, z), on the cosmology.lnP grid
//   dchi/dX (z)        at fixed z, on the cosmology.chi grid (Mpc/h)
//   dlnG/dX (z)        at fixed z, on the cosmology.G grid
//   dlnOmega_m/dX      the explicit Omega_m of the lensing and NLA kernels
// Everything else (H(z) and its response, the lensing efficiency, the
// k = l/chi shift of P_NL, the measure, the NLA amplitude) is analytic.
// Scope: NLA intrinsic alignments; TATT is rejected (its one-loop kernels
// would need FAST-PT FFTs of the response inside cosmolike).
// ----------------------------------------------------------------------------

// Maximum number of parameter slots with a loaded response.
#define FISHER_NPARAM_MAX 8

// Load the response of the cosmology tables to the parameter in slot ip.
// The tables must live on the grids of the current set_cosmology call (the
// C++ wrapper checks the grids; this function only checks the sizes).
void set_fisher_response(
    const int ip,             // parameter slot (0 .. FISHER_NPARAM_MAX-1)
    const double dlnOm_dX,    // explicit dlnOmega_m/dX (Cocoa basis: 1/Omega_m
                              // for X = Omega_m, 0 for every other parameter)
    const double* dchi_dX,    // dchi/dX in Mpc/h, on the cosmology.chi z grid
    const int nz_chi,         // number of chi nodes (= cosmology.chi_nz)
    const double* dlnG_dX,    // dlnG/dX on the cosmology.G z grid
    const int nz_G,           // number of growth nodes (= cosmology.G_nz)
    const double* dlnPNL_dX,  // dlnP_NL/dX, element (ik, jz) at ik*nz_P + jz
    const int nk,             // number of log10k nodes (= cosmology.lnP_nk)
    const int nz_P            // number of P-table z nodes (= cosmology.lnP_nz)
  );

// Forget every loaded response.
void reset_fisher_response(void);

// Number of loaded parameter slots; slots 0 .. n-1 must all be loaded.
int fisher_nparam(void);

// Batch C_ss (NLA E-mode; B-mode vanishes) and its derivative with respect
// to every loaded parameter, at arbitrary multipoles. One pass computes the
// spectrum and all derivatives: the quadrature nodes, kernels and P_NL are
// shared, only the per-parameter bracket of the integrand differs.
void dC_ss_dX_tomo_limber_nointerp_ells(
    const double* ells,  // multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of tomo shear power spectra
    double*** out        // output [1 + nparam][NSIZE][nell]:
                         // out[0] = C_EE, out[1 + ip] = dC_EE/dX_ip
  );

// xi_pm and its derivative with respect to every loaded parameter, on the
// Ntable angular binning: the same C_ell -> xi pipeline as xi_pm_tomo
// (exact low multipoles, log-ell table gathered to every integer multipole,
// bin-averaged Legendre sums), applied to C and to each dC/dX.
void dxi_pm_dX_tomo(
    double**** out       // output [1 + nparam][2][NSIZE][Ntheta]:
                         // out[0][0] = xi+, out[0][1] = xi-,
                         // out[1 + ip][pm] = dxi_pm/dX_ip
  );

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
