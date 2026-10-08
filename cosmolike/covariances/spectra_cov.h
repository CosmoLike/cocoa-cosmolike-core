#ifndef COSMOLIKE_SPECTRA_COV_H
#define COSMOLIKE_SPECTRA_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// A snapshot of the radial inputs to a covariance calculation. Distances
// and the dchi quadrature weights are in c/H0; every window is in
// (c/H0)^-1. Field order is lenses, then sources. No data-vector exclusion
// or scale cut is applied to these fields. Window roles a field does not
// use stay zero: lens window[2] always, source window[2] without IA.
struct radial_cov {
  int nnode;            // number of common radial samples (nodes)
  int nlens;            // number of lens fields
  int nsource;          // number of source fields
  double** geometry;   // [4][node]: a, chi, f_K, positive dchi weight
  double*** window;    // [3][field][node]: density, lensing/magnification, NLA
};

// Construct a snapshot from the current core cosmology and nuisance state.
// Recreate it after changing that state. Edges ascend in a, strictly inside
// (0,1); each interval gets a tabulated Gauss-Legendre rule. The caller
// must include the foreground and the complete redshift support. Release
// the snapshot with free_radial_cov.
struct radial_cov* radial_inputs_cov(
    const int npanel,       // number of integration intervals
    const double* a_edges,  // npanel+1 increasing scale factors
    const int nquad,        // nodes per panel: 64, 96, 128, 256, 512, 1024
    const int nwindow,      // uniform-a nodes for cumulative efficiencies
    const int include_ia    // 0 ignores IA; 1 includes the core NLA amplitude
  );

// Internal non-Limber sampling, after radial_inputs_cov validates the
// cosmology and catalog setup. Same windows; uniform ln(chi), no weights.
// Release the snapshot with free_radial_cov.
struct radial_cov* radial_logchi_cov(
    const double amin,      // scale factor at the far radial boundary
    const double chi_min,   // near distance in c/H0, positive
    const int nchi,         // logarithmic grid samples
    const int nwindow,      // cumulative efficiency samples
    const int include_ia    // include signed NLA
  );

void free_radial_cov(struct radial_cov* radial); // release this snapshot

// Read independent wavenumber rows at one scale factor. Inputs and outputs
// have shape [nrow][ncol], in core c/H0 units, and cannot overlap.
// Initialize the cosmology tables before calling; no table is changed.
// Call outside an OpenMP region: the function divides the rows among its
// own OpenMP workers.
void power_rows_cov(
    const double a,                  // shared scale factor
    const int nrow,                  // independent wavenumber rows
    const int ncol,                  // samples per row
    const double* const* k,          // positive physical wavenumbers
    const int linear,               // linear or configured nonlinear power
    double* const* power            // caller-owned output rows
  );

// Evaluate every field pair on one common radial rule. This is a Limber
// spectrum builder, not a non-Limber approximation at small multipoles.
// Output rows use i-major triangular order: (0,0),(0,1),...,(1,1),...
// Spectra are in the core harmonic convention, before the extra spin
// conversion in the real-space data-vector kernels. Noise is not included.
// No cache: the caller owns the ell grid and output, and calls serially.
// linear selects the power: 0 = the core's run-mode Pdelta(k,a),
// 1 = p_lin(k,a), 2 = D(a)^2 p_lin(k,1), the separable linear field that
// the non-Limber correction subtracts.
void limber_spectra_cov(
    const struct radial_cov* radial, // immutable snapshot
    const int nell,                 // number of supplied ell nodes
    const double* ell,              // finite multipoles >= 1
    const int linear,               // 0=Pdelta, 1=p_lin(k,a), 2=D^2 p_lin(k,1)
    const int include_rsd,          // same lens RSD field for every pair
    double* const* spectra          // output [nfield*(nfield+1)/2][nell]
  );

#ifdef __cplusplus
}
#endif
#endif
