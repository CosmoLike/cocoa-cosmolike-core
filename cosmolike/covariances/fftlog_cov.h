#ifndef COSMOLIKE_FFTLOG_COV_H
#define COSMOLIKE_FFTLOG_COV_H

#ifdef __cplusplus
extern "C" {
#endif

struct fftlog_workspace_cov;

// Prepare F_l(k) = integral dlnchi f(chi) j_l(k chi)/(k chi)^p.
// chi[i] = chi_min*exp(i*dlnchi); p is 0 (density) or 2 (lensing).
// f already includes the dchi -> dlnchi Jacobian. No spin factor is
// supplied here. Distances and inverse wavenumbers use the same units.
// The owner retains this workspace across multipole blocks and frees it.
// Create, execute and free outside OpenMP regions; these functions manage
// their own workers. Input arrays are copied, never retained or modified.
struct fftlog_workspace_cov* fftlog_create_cov(
    const int nchi,                  // physical radial samples
    const int padding,               // zero samples before physical input
    const int nfft,                  // even FFT length with 2/3/5/7 factors
    const int extra,                 // k samples beyond each radial end
    const double chi_min,            // positive first physical distance
    const double dlnchi,             // positive uniform logarithmic step
    const int nfield,                // independent radial functions
    const int* inverse_power,        // [field], each 0 or 2
    const double* const* radial      // [field][nchi], finite signed f
  );

// Return consecutive integer multipoles, 2 <= first <= first+nell-1.
// nell <= 16 bounds the temporary Mellin kernels. The output k grid has
// nk=nchi+2*extra nodes and is shared by every field at a given ell.
// All output entries are overwritten; arrays must not overlap.
void fftlog_execute_cov(
    struct fftlog_workspace_cov* work, // prepared radial transforms
    const int first,                  // first integer ell, >= 2
    const int nell,                   // block length, 1..16
    double* const* wave,              // [nell][nk], increasing positive k
    double** const* transfer          // [nell][field][nk], F_l(k)
  );

void fftlog_free_cov(struct fftlog_workspace_cov* work);

#ifdef __cplusplus
}
#endif
#endif
