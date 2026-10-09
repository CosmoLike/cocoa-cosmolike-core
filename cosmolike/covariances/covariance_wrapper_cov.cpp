#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <spdlog/spdlog.h>

// Abort on invalid input like the data-vector layer: print through
// the shared logger, then end the process. No C++ exceptions.
using spdlog::critical;
using std::exit;

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
// angular statistic or Fourier bandpower. For observables r=AB and s=CD
// in bins i and j, the shared C assembler gaussian_matrix_cov computes
//
//   Cov[(r,i),(s,j)] = sum_ell K_i(ell) G_AB,CD(ell) K_j(ell),
//   G_AB,CD(ell)     = [(C_AC+N_AC)(C_BD+N_BD) + (C_AD+N_AD)(C_BC+N_BC)]
//                      / [(2 ell+1) fsky],   fsky = area_sr/(4 pi).
//
// N_AC equals noise(A) only when A and C are the same catalog: different
// catalogs have independent white noise. K_i is the bin-i row of the
// observable's probe operator; Fourier bands have a single probe row.
// Real-space spectra must use the convention of the spin operators: each
// source leg of a core C_ell carries sqrt((ell-1)(ell+2)/(ell(ell+1))),
// as observed_spectra applies in Python. Fourier bands average the core
// C_ell directly, and white noise never takes this factor.
//
// Real-space noise has an infinite multipole tail: integrate the signal
// and mixed terms, then add the exact pure-noise pair-count expression.
// Fourier bands instead keep all noise terms in their finite harmonic sum.
// With b_spectra, xi+/xi- blocks also add the BB Wick term (signal and
// mixed noise), with sign +1 for equal and -1 for different xi estimators:
// xi+ measures EE+BB and xi- measures EE-BB.
//
// The result is an arma::Mat [nobs*nbin,nobs*nbin], bin varying fastest
// inside each observable; C writes both triangles from one number. The
// estimators are dimensionless, and so is their covariance. All inputs
// are const Armadillo copies made by the binding, so the caller's NumPy
// arrays, in any memory order or slice, are never modified.
//
// Public arrays use Armadillo axes. The short C workspace copies below
// follow cosmo2D_wrapper.cpp: C requires contiguous multipole rows, whereas
// Armadillo stores columns contiguously. No pointer vectors describe user
// arrays. The shared C assembler retains SIMD arithmetic and sum order;
// complete independent observable blocks share an OpenMP team.
// ---------------------------------------------------------------------------
static arma::Mat<double> gaussian_matrix_cpp(
    const arma::Cube<double>& spectra,   // (ell, field, field), signal only
    const arma::Col<double>& noise,      // (field), white-noise powers, sr
    const arma::Mat<int>& rows,          // real (probe,A,B); Fourier (A,B)
    const arma::Cube<double>& operators, // (probe, bin, ell)
    const int ell_min,                   // first consecutive multipole
    const double area_sr,                // survey solid angle, sr
    const arma::Col<double>& pair_area,  // (bin), real-space pair area, sr^2
    const bool realspace,                // true adds pair-count pure noise
    const arma::Cube<double>& b_spectra   // optional BB, empty omits it
  )
{
  // --- 1. CHECK THE PHYSICAL AXES BEFORE ENTERING C ---

  // The C assembler trusts its sizes and row pointers: a wrong field ID
  // would read outside the spectra. These checks catch that first, print
  // through the shared logger and end the process, like every validation
  // in this module.
  if (spectra.is_empty()
      || spectra.n_cols != spectra.n_slices
      || noise.n_elem != spectra.n_cols
      || rows.n_rows < 1
      || rows.n_cols != (realspace ? 3 : 2)
      || operators.n_rows != (realspace ? 4 : 1)
      || operators.n_cols < 1
      || operators.n_slices != spectra.n_rows) {
    critical("{}: inconsistent spectra, fields or bin axes",
      "gaussian_matrix_cpp");
    exit(1);
  }
  if (ell_min < (realspace ? 2 : 0)
      || !std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    critical("{}: invalid ell_min or survey area", "gaussian_matrix_cpp");
    exit(1);
  }
  if (!spectra.is_finite()
      || !noise.is_finite()
      || !operators.is_finite()
      || !pair_area.is_finite()
      || arma::any(noise < 0.0)) {
    critical("{}: need finite inputs and nonnegative noise",
      "gaussian_matrix_cpp");
    exit(1);
  }
  if (realspace
      && (pair_area.n_elem != operators.n_cols
          || arma::any(pair_area <= 0.0))) {
    critical("{}: need one positive pair area per angular bin",
      "gaussian_matrix_cpp");
    exit(1);
  }

  const int nell = spectra.n_rows;   // integer multipoles
  const int nobs = rows.n_rows;      // measured field pairs
  const int nbin = operators.n_cols; // bins within each observable
  const int offset = realspace ? 1 : 0; // first field column in rows
  const int ndata = nobs*nbin;       // dimension of the covariance

  // Each observable names its probe (real space only) and two fields. An
  // ID outside the field axis would select a spectrum row that is absent.
  for (int observable=0; observable<nobs; observable++) {
    if (realspace
        && (rows(observable, 0) < XI_PLUS_COV
            || rows(observable, 0) > W_THETA_COV)) {
      critical("{}: real-space probe IDs must lie in 0..3",
        "gaussian_matrix_cpp");
      exit(1);
    }
    for (int leg=0; leg<2; leg++) {
      const int field = rows(observable, offset+leg);
      if (field < 0
          || field >= (int) noise.n_elem) {
        critical("{}: observable field ID exceeds spectra",
          "gaussian_matrix_cpp");
        exit(1);
      }
    }
  }

  if (!b_spectra.is_empty()
      && (b_spectra.n_rows != spectra.n_rows
          || b_spectra.n_cols != spectra.n_cols
          || b_spectra.n_slices != spectra.n_slices
          || !b_spectra.is_finite())) {
    critical("{}: b_spectra must be finite and match spectra",
      "gaussian_matrix_cpp");
    exit(1);
  }

  // --- 2. COPY NOTEBOOK AXES TO THE SHARED C ROW LAYOUT ---

  // Armadillo stores columns contiguously; C sums contiguous ell rows.
  // These copies preserve physical axes and keep pointer bookkeeping out
  // of the notebook API. The C assembler owns every mathematical loop.
  // malloc2d returns one padded block holding its row pointers and rows,
  // so one free releases each workspace.
  arma::Mat<double> output(ndata, ndata);
  const int nfield = noise.n_elem; // catalogs in each spectrum axis
  const int nprobe = realspace ? 4 : 1; // operator roles
  double** power = (double**) malloc2d(nfield*nfield, nell);
  double** b_power = nullptr;
  if (!b_spectra.is_empty()) {
    b_power = (double**) malloc2d(nfield*nfield, nell);
  }
  double** kernels = (double**) malloc2d(nprobe*nbin, nell);
  double** result = (double**) malloc2d(ndata, ndata);

  // C reads the observables as one flat int array [nobs][3]. Armadillo
  // stores a 3 x nobs matrix column by column, so each observable's
  // (probe,A,B) triple is contiguous and layout.memptr() is that array.
  arma::Mat<int> layout(3, nobs); // column-major triples, flat (probe,A,B)

  // Row A*nfield+B of power holds C_AB at every multipole; the optional
  // BB rows use the same field-pair index.
  for (int first=0; first<nfield; first++) {
    for (int second=0; second<nfield; second++) {
      for (int ell=0; ell<nell; ell++) {
        power[first*nfield+second][ell] = spectra(ell, first, second);
        if (b_power != nullptr) {
          b_power[first*nfield+second][ell] = b_spectra(ell, first, second);
        }
      }
    }
  }

  // Row probe*nbin+bin of kernels holds that operator row at every ell.
  for (int probe=0; probe<nprobe; probe++) {
    for (int bin=0; bin<nbin; bin++) {
      for (int ell=0; ell<nell; ell++) {
        kernels[probe*nbin+bin][ell] = operators(probe, bin, ell);
      }
    }
  }

  // Fourier rows carry no probe column; their single operator role is 0.
  for (int row=0; row<nobs; row++) {
    layout(0, row) = realspace ? rows(row, 0) : 0;
    layout(1, row) = rows(row, offset);
    layout(2, row) = rows(row, offset+1);
  }

  // One call projects every observable block, adds real-space pure noise
  // and overwrites both triangles of result. Fourier needs no pair areas.
  gaussian_matrix_cov(nell, nfield, nobs, nbin, layout.memptr(), power,
      b_power, noise.memptr(), kernels, ell_min, area_sr,
      realspace ? pair_area.memptr() : nullptr, realspace, result);

  // Copy back by (row, column).
  for (int row=0; row<ndata; row++) {
    for (int col=0; col<ndata; col++) {
      output(row, col) = result[row][col];
    }
  }

  // free(nullptr) does nothing, so the optional BB workspace needs no test.
  free(power);
  free(b_power);
  free(kernels);
  free(result);
  return output;
}

// Separate typed entry points keep the notebook's real-space cube and
// Fourier band matrix distinct. The Fourier helper adds a single common
// operator role solely for the shared Gaussian calculation above.
arma::Mat<double> covariance_gaussian_real_cpp(
    const arma::Cube<double>& spectra,   // [nell,nfield,nfield], signal
    const arma::Col<double>& noise,      // [nfield], white noise, sr
    const arma::Mat<int>& rows,          // [nobs,3], (probe,A,B)
    const arma::Cube<double>& operators, // [4,nbin,nell], probe kernels
    const int ell_min,                   // first multipole, >= 2
    const double area_sr,                // survey solid angle, sr
    const arma::Col<double>& pair_area_sr2, // [nbin], ordered pairs, sr^2
    const arma::Cube<double>& b_spectra  // like spectra, BB; empty: E only
  )
{
  return gaussian_matrix_cpp(spectra, noise, rows, operators, ell_min,
                             area_sr, pair_area_sr2, true, b_spectra);
}

arma::Mat<double> covariance_gaussian_fourier_cpp(
    const arma::Cube<double>& spectra,   // [nell,nfield,nfield], signal
    const arma::Col<double>& noise,      // [nfield], white noise, sr
    const arma::Mat<int>& pairs,         // [nobs,2], (A,B)
    const arma::Mat<double>& operators,  // [nband,nell], band weights
    const int ell_min,                   // first multipole, >= 0
    const double area_sr                 // survey solid angle, sr
  )
{
  // Add a probe axis of length one, kernels(0,band,ell) =
  // operators(band,ell): every Fourier observable uses the same bands.
  arma::Cube<double> kernels(1, operators.n_rows, operators.n_cols);
  for (arma::uword bin=0; bin<operators.n_rows; bin++) {
    for (arma::uword ell=0; ell<operators.n_cols; ell++) {
      kernels(0, bin, ell) = operators(bin, ell);
    }
  }

  // Fourier noise stays in the harmonic sum, so no pair areas are needed.
  const arma::Col<double> unused;
  return gaussian_matrix_cpp(spectra, noise, pairs, kernels, ell_min,
                             area_sr, unused, false, arma::Cube<double>());
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
//
// Units: W_A W_B in (c/H0)^-2, the transformed trispectrum in (c/H0)^9
// and measure = dchi/(area f_K^6) in (c/H0)^-5 per sr, so the result, an
// arma::Mat [nobs*nbin,nobs*nbin] with bin inside observable, is
// dimensionless. C reads only the upper triangle i <= j of the combined
// (probe,bin) index of projected and writes both triangles of the result.
// Inputs are const Armadillo copies; the caller's arrays are untouched.
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
    critical("{}: inconsistent connected projection axes",
      "covariance_project_connected_cpp");
    exit(1);
  }
  if (arma::any(probes < XI_PLUS_COV)
      || arma::any(probes > W_THETA_COV)) {
    critical("{}: connected probe IDs must lie in 0..3",
      "covariance_project_connected_cpp");
    exit(1);
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

  // One call groups the observables by probe, integrates every angular
  // block over the radial nodes and overwrites both triangles of result.
  connected_matrix_cov(nobs, nbin, nnode, probes.memptr(), windows,
      matter, measure.memptr(), result);

  // Copy back by (row, column) and release the C workspaces.
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
