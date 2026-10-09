#include <cmath>
#include <vector>
#include <cstdlib>
#include <spdlog/spdlog.h>

// Abort on invalid input like the data-vector layer: print through
// the shared logger, then end the process. No C++ exceptions.
using spdlog::critical;
using std::exit;

#include <pybind11/numpy.h>
#include "production_interface_cov.hpp"
#include "assembly_cov.h"
#include "cosmolike/basics.h"
#include "gaussian_cov.h"

namespace py = pybind11;

namespace cosmolike_interface {
// Every array argument below is registered with noconvert(): pybind11
// accepts only C-contiguous float64 (matrix_array_cov) or int32
// (index_array_cov) arrays, and the C assemblers read the caller's memory
// directly. Each returned matrix is a new NumPy array owned by Python.
using matrix_array_cov = py::array_t<double, py::array::c_style>;
using index_array_cov = py::array_t<int, py::array::c_style>;

// ---------------------------------------------------------------------------
// Assemble a Gaussian matrix from supplied field spectra and bin operators.
//
// An observable AB fluctuates with CD through the Wick pairings AC*BD and
// AD*BC. Each pairing needs the internal crossed field spectrum, including
// pairs absent from the observable list. The input therefore contains the
// complete field matrix at every consecutive integer multipole.
//
// Real-space white noise has an infinite multipole tail. The harmonic sum
// keeps signal and mixed noise, while the exact pair-area expression adds
// pure noise. Fourier bandpowers instead retain noise in the harmonic sum.
// The existing C primitives perform both calculations, including their
// SIMDe contractions. The shared C assembler distributes observable blocks
// across OpenMP workers; each worker retains the same ordered ell sums.
//
// This boundary validates the inputs once and transposes the spectra once
// into contiguous field-pair rows. gaussian_matrix_cov then allocates its
// scratch once per worker. NumPy owns the returned matrix, so later calls
// or cosmology changes cannot alter it. No cosmology or likelihood state
// is read or changed in this calculation.
//
// ARRAYS CROSSING THE BOUNDARY
//   spectra[nell,nfield,nfield]  signal C_AB at ell_min..ell_min+nell-1,
//                                with shear legs in the observed-shear
//                                convention; dimensionless
//   noise[nfield]                white noise 1/n or sigma_component^2/n,
//                                n per steradian; nonnegative
//   rows[nobs,3] or [nobs,2]     int32 (probe,A,B) in real space, (A,B)
//                                for Fourier bands; probe 0..3 is xi+,
//                                xi-, gamma_t, w
//   operators[4,nbin,nell]       real-space bin operators, or
//   operators[nbin,nell]         Fourier band weights
//   pair_area[nbin]              real space: ordered-pair area in sr^2
//   b_spectra                    empty, or [nell,nfield,nfield] BB
//                                signal (real space only)
//   output[nobs*nbin,nobs*nbin]  owned matrix, bin inside observable
// Validation: ranks and shapes, finite values, ell_min >= 0 (>= 2 in real
// space), 0 < area_sr <= 4 pi, nonnegative noise, probe IDs in 0..3,
// field IDs below nfield and positive pair areas. No lazy core table is
// read, so no warm-up is needed; the conversion calls no BLAS routine.
// ---------------------------------------------------------------------------
static matrix_array_cov gaussian_matrix_cpp(
    const matrix_array_cov& spectra,   // [ell][field][field], observed signal
    const matrix_array_cov& noise,     // [field], nonnegative white powers
    const index_array_cov& rows,       // real: [probe,A,B]; Fourier: [A,B]
    const matrix_array_cov& operators, // real [4,bin,ell], Fourier [bin,ell]
    const int ell_min,                 // first consecutive input multipole
    const double area_sr,              // common footprint area, steradians
    const matrix_array_cov& pair_area, // real: [bin], ordered-pair area in sr^2
    const bool realspace,              // selects the stated noise convention
    const matrix_array_cov& b_spectra   // empty or [ell,field,field] BB
  )
{
  // --- 1. CHECK THE ARRAY CONTRACT BEFORE ALLOCATION OR C CALLS ---

  if (spectra.ndim() != 3
      || spectra.shape(0) < 1
      || spectra.shape(1) < 1
      || spectra.shape(1) != spectra.shape(2)
      || noise.ndim() != 1
      || noise.size() != spectra.shape(1)
      || rows.ndim() != 2
      || rows.shape(0) < 1
      || rows.shape(1) != (realspace ? 3 : 2)) {
    critical("{}: need spectra[ell,field,field], "
      "noise[field] and observable rows", "gaussian_matrix_cpp");
    exit(1);
  }
  if (ell_min < 0
      || !std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    critical("{}: need ell_min>=0 and 0<area_sr<=4*pi", "gaussian_matrix_cpp");
    exit(1);
  }
  if (realspace) {
    if (operators.ndim() != 3
        || operators.shape(0) != 4
        || operators.shape(1) < 1
        || operators.shape(2) != spectra.shape(0)
        || pair_area.ndim() != 1
        || pair_area.size() != operators.shape(1)
        || ell_min < 2) {
      critical("{}: real-space needs operators[4,bin,ell], "
        "pair_area[bin], ell_min>=2", "gaussian_matrix_cpp");
      exit(1);
    }
  } else if (operators.ndim() != 2
             || operators.shape(0) < 1
             || operators.shape(1) != spectra.shape(0)) {
    critical("{}: Fourier operators must have shape [bin,ell]",
      "gaussian_matrix_cpp");
    exit(1);
  }

  const int nell = spectra.shape(0); // shared integer-multipole count
  const int nfield = noise.size();  // number of catalogs/observed fields
  const int nobs = rows.shape(0);   // number of measured field pairs
  const int nbin = operators.shape(realspace ? 1 : 0); // bins per observable
  const int offset = realspace ? 1 : 0; // first field column in the row map
  const int ndata = nobs*nbin;      // dimension of the returned covariance

  // These four numeric arrays enter arithmetic directly. Reject NaN/Inf
  // here so the production caller gets an exception before a C primitive
  // runs. The outer loop visits each array once; the inner loop scans its
  // flat C-order storage.
  for (const auto* array : {&spectra, &noise, &operators, &pair_area}) {
    for (py::ssize_t index=0; index<array->size(); index++) {
      if (!std::isfinite(array->data()[index])) {
        critical("{}: Gaussian matrix inputs must be finite",
          "gaussian_matrix_cpp");
        exit(1);
      }
    }
  }
  for (int field=0; field<nfield; field++) {
    if (noise.data()[field] < 0.0) {
      critical("{}: white noise powers must be nonnegative",
        "gaussian_matrix_cpp");
      exit(1);
    }
  }

  // C indexes spectra[A*nfield+B] and the operator role by probe, so an
  // out-of-range ID would read outside the supplied arrays.
  for (int observable=0; observable<nobs; observable++) {
    if (realspace
        && (rows.at(observable, 0) < XI_PLUS_COV
            || rows.at(observable, 0) > W_THETA_COV)) {
      critical("{}: real-space probe IDs must lie in 0..3",
        "gaussian_matrix_cpp");
      exit(1);
    }
    for (int leg=0; leg<2; leg++) {
      const int field = rows.at(observable, offset+leg);
      if (field < 0
          || field >= nfield) {
        critical("{}: observable field ID exceeds spectra",
          "gaussian_matrix_cpp");
        exit(1);
      }
    }
  }
  if (realspace) {
    for (int bin=0; bin<nbin; bin++) {
      if (pair_area.data()[bin] <= 0.0) {
        critical("{}: ordered-pair areas must be positive",
          "gaussian_matrix_cpp");
        exit(1);
      }
    }
  }

  if (b_spectra.size() != 0) {
    if (b_spectra.ndim() != 3
        || b_spectra.shape(0) != nell
        || b_spectra.shape(1) != nfield
        || b_spectra.shape(2) != nfield) {
      critical("{}: b_spectra must match spectra", "gaussian_matrix_cpp");
      exit(1);
    }
    for (py::ssize_t index=0; index<b_spectra.size(); index++) {
      if (!std::isfinite(b_spectra.data()[index])) {
        critical("{}: b_spectra must be finite", "gaussian_matrix_cpp");
        exit(1);
      }
    }
  }

  // --- 2. PREPARE CONTIGUOUS SPECTRA AND REUSABLE BLOCK SCRATCH ---

  // C integrates along ell. In the input, adjacent values instead belong
  // to different fields. This one transpose gives each Wick input a full
  // contiguous ell row, reused by every observable that needs that pair.
  // The transposed copies are temporary vectors owned by this call; row
  // A*nfield+B of power holds C_AB(ell) for every ell (B modes likewise).
  std::vector<double> power((size_t) nfield*nfield*nell);
  std::vector<double> b_power;
  std::vector<const double*> b_rows;
  if (b_spectra.size() != 0) {
    b_power.resize(power.size());
    b_rows.resize(nfield*nfield);
  }
  for (int first=0; first<nfield; first++) {
    for (int second=0; second<nfield; second++) {
      double* row = power.data()+((size_t) first*nfield+second)*nell;
      for (int ell=0; ell<nell; ell++) {
        row[ell] = spectra.at(ell, first, second);
        if (!b_power.empty()) {
          b_power[((size_t) first*nfield+second)*nell+ell] =
              b_spectra.at(ell, first, second);
        }
      }
    }
  }

  // Row maps borrow the contiguous values above. Kernel rows already
  // have ell as their last axis, so only their addresses are collected.
  std::vector<const double*> power_rows(nfield*nfield);
  std::vector<const double*> kernel_rows((realspace ? 4 : 1)*nbin);
  std::vector<int> layout(3*nobs);
  matrix_array_cov output({ndata, ndata});
  std::vector<double*> output_rows(ndata);
  for (int pair=0; pair<nfield*nfield; pair++) {
    power_rows[pair] = power.data()+(size_t) pair*nell;
    if (!b_rows.empty()) {
      b_rows[pair] = b_power.data()+(size_t) pair*nell;
    }
  }
  for (size_t row=0; row<kernel_rows.size(); row++) {
    kernel_rows[row] = operators.data()+row*nell;
  }

  // C reads one flat [nobs,3] (probe,A,B) map in both modes. Fourier rows
  // carry no probe column, so their probe slot is 0: the single operator
  // role [nbin,nell].
  for (int row=0; row<nobs; row++) {
    layout[3*row] = realspace ? rows.at(row, 0) : 0;
    layout[3*row+1] = rows.at(row, offset);
    layout[3*row+2] = rows.at(row, offset+1);
  }
  for (int row=0; row<ndata; row++) {
    output_rows[row] = output.mutable_data(row, 0);
  }

  gaussian_matrix_cov(nell, nfield, nobs, nbin, layout.data(),
      power_rows.data(), b_rows.empty() ? nullptr : b_rows.data(),
      noise.data(), kernel_rows.data(), ell_min,
      area_sr, realspace ? pair_area.data() : nullptr, realspace,
      output_rows.data());
  return output;
}

// ---------------------------------------------------------------------------
// Add catalog windows to an already angularly projected matter trispectrum.
//
// Observable r measures a pair of fields A,B. At radial shell alpha it
// carries the product W_r = W_A W_B. The connected covariance is therefore
//
//   C[(r,i),(s,j)] = sum_alpha W_r(alpha) W_s(alpha)
//                   T[t(r,i),t(s,j),alpha] measure(alpha).
//
// The combined index t(r,i) = p_r*nbin+i places bin inside probe.
// T already contains both angular/band transforms; measure contains
// dchi/(survey_area*f_K^6). A different catalog pair changes W, while
// every pair with the same two probes and angular bins shares T.
//
// The shared connected_matrix_cov C routine owns grouping, parallel work
// and SIMD radial sums. This boundary only checks the arrays and borrows
// their rows. NumPy owns the result; neither interface changes the input.
//
// ARRAYS CROSSING THE BOUNDARY
//   probes[nobs]                    int32 probe of each observable, 0..3
//   pair_window[nobs,nnode]         W_A W_B, in (c/H0)^-2
//   projected[4*nbin,4*nbin,nnode]  T after both angular/band transforms,
//                                   in (c/H0)^9; C reads only entries whose
//                                   combined row index t <= column index
//   measure[nnode]                  dchi/(area*f_K^6), in (c/H0)^-5
//   output[nobs*nbin,nobs*nbin]     dimensionless, bin inside observable;
//                                   each computed entry is mirrored
// Validation: ranks and shapes (projected square with a multiple of four
// rows), probe IDs in 0..3 and finite values. C runs one OpenMP team over
// (probe,bin) blocks with private scratch and reads no core table; the
// conversion calls no BLAS routine.
// ---------------------------------------------------------------------------
static matrix_array_cov connected_matrix_cpp(
    const index_array_cov& probes,       // [observable], probe IDs 0..3
    const matrix_array_cov& pair_window, // [observable,node], W_A W_B
    const matrix_array_cov& projected,   // [4*bin,4*bin,node], transformed T
    const matrix_array_cov& measure      // [node], dchi/(area*f_K^6)
  )
{
  // --- 1. CHECK SHAPES AND VALUES BEFORE ALLOCATION OR PARALLEL WORK ---

  if (probes.ndim() != 1
      || probes.size() < 1
      || measure.ndim() != 1
      || measure.size() < 1
      || pair_window.ndim() != 2
      || pair_window.shape(0) != probes.size()
      || pair_window.shape(1) != measure.size()
      || projected.ndim() != 3
      || projected.shape(0) < 4
      || projected.shape(0)%4 != 0
      || projected.shape(1) != projected.shape(0)
      || projected.shape(2) != measure.size()) {
    critical("{}: need probes[observable], "
      "pair_window[observable,node], projected[4*bin,4*bin,node] and measure[node]", "connected_matrix_cpp");
    exit(1);
  }
  for (py::ssize_t row=0; row<probes.size(); row++) {
    if (probes.data()[row] < XI_PLUS_COV
        || probes.data()[row] > W_THETA_COV) {
      critical("{}: connected probe IDs must lie in 0..3",
        "connected_matrix_cpp");
      exit(1);
    }
  }
  for (const auto* array : {&pair_window, &projected, &measure}) {
    for (py::ssize_t index=0; index<array->size(); index++) {
      if (!std::isfinite(array->data()[index])) {
        critical("{}: connected projection inputs must be finite",
          "connected_matrix_cpp");
        exit(1);
      }
    }
  }

  const int nobs = probes.size();           // measured field-pair count
  const int nnode = measure.size();         // common radial sample count
  const int ntransform = projected.shape(0); // four probes times bin count
  const int nbin = ntransform/4;            // bins within each observable
  const int ndata = nobs*nbin;              // complete covariance dimension

  // Borrow radial rows in their original order. Grouping catalogs and
  // assigning blocks to workers belong to C, shared with the notebook API.
  // Row t*ntransform+u of matter is the radial T row of the combined
  // indices (t,u), which is how C addresses projected.
  std::vector<const double*> windows(nobs);
  std::vector<const double*> matter(ntransform*ntransform);
  matrix_array_cov output({ndata, ndata});
  std::vector<double*> result(ndata);
  for (int row=0; row<nobs; row++) {
    windows[row] = pair_window.data(row, 0);
  }
  for (int row=0; row<ntransform*ntransform; row++) {
    matter[row] = projected.data()+(size_t) row*nnode;
  }
  for (int row=0; row<ndata; row++) {
    result[row] = output.mutable_data(row, 0);
  }
  connected_matrix_cov(nobs, nbin, nnode, probes.data(), windows.data(),
      matter.data(), measure.data(), result.data());
  return output;
}

// Register the connected projection and the two Gaussian assemblers on the
// production submodule. The real-space lambda selects the pair-count pure
// noise and optional BB spectra; the default b_spectra is an empty array,
// meaning E modes only. The Fourier lambda passes empty arrays for the
// unused pair areas and B modes. Every array argument is noconvert().
void bind_production_matrices_cov(py::module_& module)
{
  module.def("covariance_project_connected", &connected_matrix_cpp,
      R"doc(Project a connected matter table through every catalog pair.

probes[observable] is contiguous int32: 0 xi+, 1 xi-, 2 gamma_t, 3 w.
pair_window[observable,node] contains W_A*W_B in (c/H0)^-2.
projected[4*nbin,4*nbin,node] contains the already angularly/band-projected
matter trispectrum in (c/H0)^9, with bin inside probe on both axes.
Only its probe/bin upper triangle is consumed and mirrored in the result.
measure[node] supplies dchi/(area*f_K^6), in (c/H0)^-5. These three arrays
are contiguous float64 and share their radial nodes. Signed inputs are
retained. The common matter model and its approximations belong to the
caller; this operation does not compute halo physics or add SSC/noise.

Returns an owned symmetric [nobservable*nbin,nobservable*nbin] matrix,
with bin inside observable. Every entry retains increasing radial-node
sum order. No input or cosmology/likelihood state is changed.
)doc",
      py::arg("probes").noconvert(), py::arg("pair_window").noconvert(),
      py::arg("projected").noconvert(), py::arg("measure").noconvert());

  module.def("covariance_gaussian_real",
      [](const matrix_array_cov& spectra, const matrix_array_cov& noise,
         const index_array_cov& rows, const matrix_array_cov& operators,
         const int ell_min, const double area_sr,
         const matrix_array_cov& pair_area_sr2,
         const matrix_array_cov& b_spectra) {
        return gaussian_matrix_cpp(spectra, noise, rows, operators, ell_min,
                                   area_sr, pair_area_sr2, true, b_spectra);
      },
      R"doc(Compute a real-space Gaussian matrix from supplied field spectra.

spectra[ell,field,field] contains signal in the observed-shear convention;
noise[field] gives independent white shot/shape powers per steradian.
Optional b_spectra has the same axes and units as spectra and contains
source-source BB signals, with galaxy rows/columns zero. EB and gB vanish
by parity. The xi+/xi- BB signs are applied in C; pure noise is not doubled.
rows[observable,3] contains (probe,A,B), with probe=0 xi+, 1 xi-,
2 gamma_t, 3 w. operators[4,bin,ell] covers the same consecutive ell
values as spectra, starting at ell_min>=2. Supply angular-bin-averaged
kernels and positive ordered-pair areas pair_area_sr2[bin]. area_sr is
survey area in steradians. All numeric arrays are contiguous float64;
rows is contiguous int32. Every internal crossed field pair is required.

Returns an owned symmetric [nobservable*nbin,nobservable*nbin] matrix,
with bins inside each observable. Pure noise is added analytically;
the multipole sum contains signal and mixed noise only. No likelihood
state is changed and no input is modified. This is a Gaussian fsky
calculation, not exact cut-sky mode coupling or non-Gaussian covariance.
)doc",
      py::arg("spectra").noconvert(), py::arg("noise").noconvert(),
      py::arg("rows").noconvert(), py::arg("operators").noconvert(),
      py::arg("ell_min"), py::arg("area_sr"),
      py::arg("pair_area_sr2").noconvert(),
      py::arg("b_spectra").noconvert() = matrix_array_cov(0));

  module.def("covariance_gaussian_fourier",
      [](const matrix_array_cov& spectra, const matrix_array_cov& noise,
         const index_array_cov& pairs, const matrix_array_cov& operators,
         const int ell_min, const double area_sr) {
        const matrix_array_cov unused((py::ssize_t) 0);
        return gaussian_matrix_cpp(spectra, noise, pairs, operators, ell_min,
                                   area_sr, unused, false, unused);
      },
      R"doc(Compute a Gaussian bandpower matrix from supplied field spectra.

spectra[ell,field,field] contains observed signal on consecutive integer
multipoles starting at ell_min>=0. noise[field] is independent white
shot/shape power. pairs[observable,2] contains field IDs (A,B).
operators[band,ell] supplies normalized band weights; the supplied bands
may overlap. area_sr is the common survey area in steradians.
Use contiguous float64 arrays and int32 pairs. Internal crossed spectra
are required even when excluded from the measured bandpower vector.

Returns an owned symmetric [nobservable*nband,nobservable*nband] matrix,
with bands inside each observable. The harmonic Wick sum includes pure
noise. No state or input is changed. This fsky Gaussian calculation does
not include SSC, cNG or exact cut-sky mode coupling.
)doc",
      py::arg("spectra").noconvert(), py::arg("noise").noconvert(),
      py::arg("pairs").noconvert(), py::arg("operators").noconvert(),
      py::arg("ell_min"), py::arg("area_sr"));
}
}
