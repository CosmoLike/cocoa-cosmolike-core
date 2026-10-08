#ifndef COSMOLIKE_FFTLOG_COV_H
#define COSMOLIKE_FFTLOG_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// An opaque workspace: callers hold only a pointer. It owns the FFTW
// plans, the saved forward transform of every radial input and one
// scratch row per OpenMP worker (members documented in fftlog_cov.c).
struct fftlog_workspace_cov;

// Prepare F_l(k) = integral dlnchi f(chi) j_l(k chi)/(k chi)^p.
// chi[i] = chi_min*exp(i*dlnchi); p is 0 (density) or 2 (lensing).
// f already includes the dchi -> dlnchi Jacobian: f = chi*W for a radial
// weight W per unit distance (growth included), so
// F = integral dchi W j_l(k chi)/(k chi)^p.
// No spin factor is supplied here; the caller multiplies F by its angular
// factor. Distances and inverse wavenumbers use the same units.
//
// Grid contract: nchi >= 3; padding >= 1 zero samples below chi_min;
// nfft >= 8 and nfft >= nchi+2*padding, even, with no prime factor
// above 7; 0 <= extra <= padding. The padding fixes the transform's
// phase origin.
//
// The owner retains this workspace across multipole blocks and frees it.
// Create, execute and free outside OpenMP regions; these functions manage
// their own workers. The workspace holds one scratch row per thread of
// omp_get_max_threads() at creation, so later executions must not use a
// larger team. Input arrays are copied, never retained or modified.
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

// Return F_l(k) for the consecutive integer multipoles
// l = first, ..., first+nell-1, with first >= 2.
// nell <= 16 bounds the temporary Mellin kernels. The output k grid has
// nk=nchi+2*extra nodes and is shared by every field at a given ell:
//   wave[index][q] = k0*exp((q-extra)*dlnchi),  k0 = (l+1)/chi_max,
// with chi_max the last radial sample. Its logarithmic step equals
// dlnchi, so a later k integral uses dlnk = dlnchi.
// transfer[index][field][q] is that field's F_l at wave[index][q].
// All output entries are overwritten; arrays must not overlap.
void fftlog_execute_cov(
    struct fftlog_workspace_cov* work, // prepared radial transforms
    const int first,                  // first integer ell, >= 2
    const int nell,                   // block length, 1..16
    double* const* wave,              // [nell][nk], increasing positive k
    double** const* transfer          // [nell][field][nk], F_l(k)
  );

// Destroy the plan and release every workspace array. Call once, serially,
// after the last execution; returned transfers stay valid.
void fftlog_free_cov(struct fftlog_workspace_cov* work);

#ifdef __cplusplus
}
#endif
#endif
