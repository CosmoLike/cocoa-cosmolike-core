#include <cmath>
#include <stdexcept>

#include <carma.h>
#include <armadillo>
#include "covariance_wrapper_cov.hpp"
#include "gaussian_cov.h"
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
// arrays. The existing C routines retain their SIMD arithmetic and sum
// order; complete independent observable blocks share an OpenMP team.
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

  // --- 2. SHARE THE BIN OPERATORS AND ENUMERATE DISTINCT BLOCKS ---

  // Every pair of catalogs uses the same angular/band operator. Copy it
  // once, outside the observable loop. Each C row sums consecutive ell.
  arma::Mat<double> output(ndata, ndata);
  arma::Mat<int> tasks(nobs*(nobs+1)/2, 2);
  double*** kernels = (double***) malloc3d(operators.n_rows, nbin, nell);
  for (arma::uword probe=0; probe<operators.n_rows; probe++) {
    for (int bin=0; bin<nbin; bin++) {
      for (int ell=0; ell<nell; ell++) {
        kernels[probe][bin][ell] = operators(probe, bin, ell);
      }
    }
  }

  // Cov(AB,CD)=Cov(CD,AB). A task owns both a block and its transpose;
  // enumerating the triangle explicitly gives the workers equal task counts.
  int task = 0;
  for (int first=0; first<nobs; first++) {
    for (int second=first; second<nobs; second++) {
      tasks(task, 0) = first;
      tasks(task, 1) = second;
      task++;
    }
  }

  // --- 3. APPLY THE C WICK AND PROJECTION ROUTINES TO EACH BLOCK ---

  // Each worker keeps scratch for a complete bin-by-bin block. Different
  // tasks write disjoint matrix cells, so the ell sum never gets split
  // between workers. A single-observable call lets C parallelize its bins.
  #pragma omp parallel if(nobs > 1)
  {
    arma::Col<double> harmonic(nell); // Wick covariance per multipole
    double** cross = (double**) malloc2d(4, nell);
    double** weighted = (double**) malloc2d(nbin, nell);
    double** block = (double**) malloc2d(nbin, nbin);

    #pragma omp for schedule(static)
    for (arma::uword index=0; index<tasks.n_rows; index++) {
      const int first = tasks(index, 0);
      const int second = tasks(index, 1);
      const int a = rows(first, offset);
      const int b = rows(first, offset+1);
      const int c = rows(second, offset);
      const int d = rows(second, offset+1);
      const int left_probe = realspace ? rows(first, 0) : 0;
      const int right_probe = realspace ? rows(second, 0) : 0;
      const arma::Col<int> fields = {a, b, c, d};
      const arma::Col<double> noise_ab = {noise(a), noise(b)};
      const arma::Col<double> cross_noise = {
        a == c ? noise(a) : 0.0,
        b == d ? noise(b) : 0.0,
        a == d ? noise(a) : 0.0,
        b == c ? noise(b) : 0.0
      };

      // Preserve AC,BD,AD,BC order: the Wick kernel multiplies rows 0*1
      // and 2*3. Catalog noise contributes only for identical fields.
      for (int ell=0; ell<nell; ell++) {
        cross[0][ell] = spectra(ell, a, c);
        cross[1][ell] = spectra(ell, b, d);
        cross[2][ell] = spectra(ell, a, d);
        cross[3][ell] = spectra(ell, b, c);
      }
      gaussian_wick_cov(ell_min, nell, area_sr/(4.0*M_PI), cross,
          cross_noise.memptr(), !realspace, harmonic.memptr());
      gaussian_project_cov(nbin, nbin, nell, kernels[left_probe],
          kernels[right_probe], harmonic.memptr(), weighted, block);

      if (realspace) {
        // Disjoint angular bins share pure pair noise only on the
        // diagonal. This term includes modes beyond the finite ell grid.
        for (int bin=0; bin<nbin; bin++) {
          block[bin][bin] += gaussian_noise_pair_cov(
              (probe_cov) left_probe, (probe_cov) right_probe,
              fields.memptr(), noise_ab.memptr(), pair_area(bin));
        }
      }

      // Select one triangle even for a diagonal block: roundoff in a
      // reverse product must not change which value gets mirrored.
      for (int left=0; left<nbin; left++) {
        const int start = first == second ? left : 0;
        for (int right=start; right<nbin; right++) {
          const int i = first*nbin+left;
          const int j = second*nbin+right;
          output(i, j) = block[left][right];
          output(j, i) = block[left][right];
        }
      }
    }
    free(cross);
    free(weighted);
    free(block);
  }
  free(kernels);
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
  arma::Col<int> counts(4, arma::fill::zeros);
  arma::Mat<int> groups(4, nobs); // input observable IDs within each probe

  // Preserve catalog order within each probe. On a diagonal angular
  // block, this determines which value supplies both covariance triangles.
  for (int row=0; row<nobs; row++) {
    const int probe = probes(row);
    groups(probe, counts(probe)) = row;
    counts(probe)++;
  }
  const int largest = counts.max(); // largest catalog group
  arma::Mat<int> tasks(10*nbin*nbin, 4); // probes and bins for each task
  int ntask = 0;
  for (int left=0; left<4; left++) {
    if (counts(left) == 0) continue;
    for (int right=left; right<4; right++) {
      if (counts(right) == 0) continue;
      for (int first=0; first<nbin; first++) {
        const int start = left == right ? first : 0;
        for (int second=start; second<nbin; second++) {
          tasks(ntask, 0) = left;
          tasks(ntask, 1) = right;
          tasks(ntask, 2) = first;
          tasks(ntask, 3) = second;
          ntask++;
        }
      }
    }
  }

  // C integrates complete radial rows. Copy each group once; unused
  // rows in smaller groups are never passed to the projection routine.
  double*** windows = (double***) malloc3d(4, largest, nnode);
  for (int probe=0; probe<4; probe++) {
    for (int row=0; row<counts(probe); row++) {
      for (int node=0; node<nnode; node++) {
        windows[probe][row][node] = pair_window(groups(probe, row), node);
      }
    }
  }

  // Each task owns all catalog pairings for two angular bins, including
  // their transposes. Round-robin scheduling distributes large and small
  // catalog groups across the team without splitting a radial sum.
  #pragma omp parallel if(ntask > 1)
  {
    arma::Col<double> weight(nnode);
    double** weighted = (double**) malloc2d(largest, nnode);
    double** block = (double**) malloc2d(largest, largest);

    #pragma omp for schedule(static, 1)
    for (int task=0; task<ntask; task++) {
      const int left = tasks(task, 0);
      const int right = tasks(task, 1);
      const int first = tasks(task, 2);
      const int second = tasks(task, 3);

      // First combine the matter function with its radial measure.
      // The C kernel then multiplies the left window before summing
      // against the right, preserving the original arithmetic order.
      for (int node=0; node<nnode; node++) {
        weight(node) = measure(node)
            *projected(left*nbin+first, right*nbin+second, node);
      }
      gaussian_project_cov(counts(left), counts(right), nnode,
          windows[left], windows[right], weight.memptr(), weighted, block);

      for (int i=0; i<counts(left); i++) {
        const int start = (left == right
                           && first == second) ? i : 0;
        for (int j=start; j<counts(right); j++) {
          const int row = groups(left, i)*nbin+first;
          const int col = groups(right, j)*nbin+second;
          output(row, col) = block[i][j];
          output(col, row) = block[i][j];
        }
      }
    }
    free(weighted);
    free(block);
  }
  free(windows);
  return output;
}

} // namespace cosmolike_interface
