#include <cmath>
#include <stdexcept>

#include <carma.h>
#include <armadillo>
#include "covariance_wrapper_cov.hpp"
#include "gaussian_cov.h"
#include "assembly_cov.h"
#include "cosmolike/basics.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Assemble Cov(AB,CD) from the Wick pairings AC*BD and AD*BC.
//
// spectra(ell,A,B) includes all internal field crosses, even those absent
// from the observable list. Each observable specifies the fields measured
// together. A bin operator converts its harmonic spectrum into the chosen
// angular statistic or Fourier bandpower.
//
// Real-space noise has an infinite multipole tail: integrate the signal
// and mixed terms, then add the exact pure-noise pair-count expression.
// Fourier bands instead keep all noise terms in their finite harmonic sum.
//
// Public arrays use Armadillo axes. The short C workspace copies below
// follow cosmo2D_wrapper.cpp: C requires contiguous multipole rows, whereas
// Armadillo stores columns contiguously. No pointer vectors describe user
// arrays. The shared C assembler retains SIMD arithmetic and sum order;
// complete independent observable blocks share an OpenMP team.
// ---------------------------------------------------------------------------
static arma::Mat<double> gaussian_matrix_cpp(
    const arma::Cube<double>& spectra,   // (ell, field, field), signal only
    const arma::Col<double>& noise,      // (field), white-noise powers
    const arma::Mat<int>& rows,          // real (probe,A,B); Fourier (A,B)
    const arma::Cube<double>& operators, // (probe, bin, ell)
    const int ell_min,                   // first consecutive multipole
    const double area_sr,                // survey solid angle, sr
    const arma::Col<double>& pair_area,  // real-space ordered-pair area, sr^2
    const bool realspace                 // pure-noise convention
  )
{
  // --- 1. CHECK THE PHYSICAL AXES BEFORE ENTERING C ---

  if (spectra.is_empty()
      || spectra.n_cols != spectra.n_slices
      || noise.n_elem != spectra.n_cols
      || rows.n_rows < 1
      || rows.n_cols != (realspace ? 3 : 2)
      || operators.n_rows != (realspace ? 4 : 1)
      || operators.n_cols < 1
      || operators.n_slices != spectra.n_rows) {
    throw std::invalid_argument("inconsistent spectra, fields or bin axes");
  }
  if (ell_min < (realspace ? 2 : 0)
      || !std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    throw std::invalid_argument("invalid ell_min or survey area");
  }
  if (!spectra.is_finite()
      || !noise.is_finite()
      || !operators.is_finite()
      || !pair_area.is_finite()
      || arma::any(noise < 0.0)) {
    throw std::invalid_argument("need finite inputs and nonnegative noise");
  }
  if (realspace
      && (pair_area.n_elem != operators.n_cols
          || arma::any(pair_area <= 0.0))) {
    throw std::invalid_argument("need one positive pair area per angular bin");
  }

  const int nell = spectra.n_rows;   // integer multipoles
  const int nobs = rows.n_rows;      // measured field pairs
  const int nbin = operators.n_cols; // bins within each observable
  const int offset = realspace ? 1 : 0; // first field column in rows
  const int ndata = nobs*nbin;       // dimension of the covariance

  for (int observable=0; observable<nobs; observable++) {
    if (realspace
        && (rows(observable, 0) < XI_PLUS_COV
            || rows(observable, 0) > W_THETA_COV)) {
      throw std::invalid_argument("real-space probe IDs must lie in 0..3");
    }
    for (int leg=0; leg<2; leg++) {
      const int field = rows(observable, offset+leg);
      if (field < 0
          || field >= (int) noise.n_elem) {
        throw std::invalid_argument("observable field ID exceeds spectra");
      }
    }
  }

  // --- 2. COPY NOTEBOOK AXES TO THE SHARED C ROW LAYOUT ---

  // Armadillo stores columns contiguously; C sums contiguous ell rows.
  // These copies preserve physical axes and keep pointer bookkeeping out
  // of the notebook API. The C assembler owns every mathematical loop.
  arma::Mat<double> output(ndata, ndata);
  const int nfield = noise.n_elem; // catalogs in each spectrum axis
  const int nprobe = realspace ? 4 : 1; // operator roles
  double** power = (double**) malloc2d(nfield*nfield, nell);
  double** kernels = (double**) malloc2d(nprobe*nbin, nell);
  double** result = (double**) malloc2d(ndata, ndata);
  arma::Mat<int> layout(3, nobs); // column-major triples, flat (probe,A,B)

  for (int first=0; first<nfield; first++) {
    for (int second=0; second<nfield; second++) {
      for (int ell=0; ell<nell; ell++) {
        power[first*nfield+second][ell] = spectra(ell, first, second);
      }
    }
  }
  for (int probe=0; probe<nprobe; probe++) {
    for (int bin=0; bin<nbin; bin++) {
      for (int ell=0; ell<nell; ell++) {
        kernels[probe*nbin+bin][ell] = operators(probe, bin, ell);
      }
    }
  }
  for (int row=0; row<nobs; row++) {
    layout(0, row) = realspace ? rows(row, 0) : 0;
    layout(1, row) = rows(row, offset);
    layout(2, row) = rows(row, offset+1);
  }

  gaussian_matrix_cov(nell, nfield, nobs, nbin, layout.memptr(), power,
      noise.memptr(), kernels, ell_min, area_sr,
      realspace ? pair_area.memptr() : nullptr, realspace, result);

  for (int row=0; row<ndata; row++) {
    for (int col=0; col<ndata; col++) {
      output(row, col) = result[row][col];
    }
  }
  free(power);
  free(kernels);
  free(result);
  return output;
}

// Separate typed entry points keep the notebook's real-space cube and
// Fourier band matrix distinct. The Fourier helper adds a single common
// operator role solely for the shared Gaussian calculation above.
arma::Mat<double> covariance_gaussian_real_cpp(
    const arma::Cube<double>& spectra,
    const arma::Col<double>& noise,
    const arma::Mat<int>& rows,
    const arma::Cube<double>& operators,
    const int ell_min,
    const double area_sr,
    const arma::Col<double>& pair_area_sr2
  )
{
  return gaussian_matrix_cpp(spectra, noise, rows, operators, ell_min,
                             area_sr, pair_area_sr2, true);
}

arma::Mat<double> covariance_gaussian_fourier_cpp(
    const arma::Cube<double>& spectra,
    const arma::Col<double>& noise,
    const arma::Mat<int>& pairs,
    const arma::Mat<double>& operators,
    const int ell_min,
    const double area_sr
  )
{
  arma::Cube<double> kernels(1, operators.n_rows, operators.n_cols);
  for (arma::uword bin=0; bin<operators.n_rows; bin++) {
    for (arma::uword ell=0; ell<operators.n_cols; ell++) {
      kernels(0, bin, ell) = operators(bin, ell);
    }
  }
  const arma::Col<double> unused;
  return gaussian_matrix_cpp(spectra, noise, pairs, kernels, ell_min,
                             area_sr, unused, false);
}

// ---------------------------------------------------------------------------
// Project the connected matter trispectrum through catalog windows.
//
// For an observable r measuring fields A,B, pair_window(r,node)=W_A W_B.
// projected(p*nbin+i,q*nbin+j,node) already contains both angular transforms
// of the matter trispectrum. The remaining integral is
//
//   C[(r,i),(s,j)] = sum_node W_r W_s projected measure.
//
// Group observables by probe because catalog pairs with the same probes
// share this transformed matter function. A task handles one pair of
// angular bins and all its catalog pairs. The C projection retains its
// ordered radial sum and SIMD arithmetic; Armadillo gives the notebook
// explicit observable, angular-bin and radial-node axes.
// ---------------------------------------------------------------------------
arma::Mat<double> covariance_project_connected_cpp(
    const arma::Col<int>& probes,        // (observable), xi+,xi-,gamma_t,w
    const arma::Mat<double>& pair_window,// (observable,node), W_A W_B
    const arma::Cube<double>& projected, // (4*bin,4*bin,node), transformed T
    const arma::Col<double>& measure     // (node), dchi/(area*f_K^6)
  )
{
  if (probes.is_empty()
      || measure.is_empty()
      || pair_window.n_rows != probes.n_elem
      || pair_window.n_cols != measure.n_elem
      || projected.n_rows < 4
      || projected.n_rows%4 != 0
      || projected.n_cols != projected.n_rows
      || projected.n_slices != measure.n_elem
      || !pair_window.is_finite()
      || !projected.is_finite()
      || !measure.is_finite()) {
    throw std::invalid_argument("inconsistent connected projection axes");
  }
  if (arma::any(probes < XI_PLUS_COV)
      || arma::any(probes > W_THETA_COV)) {
    throw std::invalid_argument("connected probe IDs must lie in 0..3");
  }

  const int nobs = probes.n_elem;   // measured catalog pairs
  const int nnode = measure.n_elem; // common radial quadrature
  const int nbin = projected.n_rows/4; // angular bins per statistic
  arma::Mat<double> output(nobs*nbin, nobs*nbin);
  const int ntransform = 4*nbin; // combined probe and angular-bin axis
  double** windows = (double**) malloc2d(nobs, nnode);
  double** matter = (double**) malloc2d(ntransform*ntransform, nnode);
  double** result = (double**) malloc2d(nobs*nbin, nobs*nbin);

  // Copy by physical indices. C expects the radial node to vary fastest;
  // its assembler supplies probe grouping, integration and parallel work.
  for (int row=0; row<nobs; row++) {
    for (int node=0; node<nnode; node++) {
      windows[row][node] = pair_window(row, node);
    }
  }
  for (int first=0; first<ntransform; first++) {
    for (int second=0; second<ntransform; second++) {
      for (int node=0; node<nnode; node++) {
        matter[first*ntransform+second][node] = projected(first, second, node);
      }
    }
  }

  connected_matrix_cov(nobs, nbin, nnode, probes.memptr(), windows,
      matter, measure.memptr(), result);

  for (int row=0; row<nobs*nbin; row++) {
    for (int col=0; col<nobs*nbin; col++) {
      output(row, col) = result[row][col];
    }
  }
  free(windows);
  free(matter);
  free(result);
  return output;
}

} // namespace cosmolike_interface
