#include <cmath>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include "covariance_wrapper_cov.hpp"
#include "gaussian_cov.h"
#include "simde/x86/sse2.h"

namespace py = pybind11;

namespace cosmolike_interface {
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
// SIMDe contractions. This wrapper distributes complete observable blocks
// across OpenMP workers; each worker retains the same ordered ell sums.
//
// Validate inputs once, transpose spectra once into contiguous field-pair
// rows, and allocate scratch once per worker. NumPy owns the
// returned matrix, so later calls or cosmology changes cannot alter it.
// No cosmology or likelihood state is read or changed in this calculation.
// ---------------------------------------------------------------------------
static matrix_array_cov gaussian_matrix_cpp(
    const matrix_array_cov& spectra,   // [ell][field][field], observed signal
    const matrix_array_cov& noise,     // [field], nonnegative white powers
    const index_array_cov& rows,       // real: [probe,A,B]; Fourier: [A,B]
    const matrix_array_cov& operators, // real [4,bin,ell], Fourier [bin,ell]
    const int ell_min,                 // first consecutive input multipole
    const double area_sr,              // common footprint area, steradians
    const matrix_array_cov& pair_area, // real: [bin], ordered-pair area in sr^2
    const bool realspace               // selects the stated noise convention
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
    throw std::invalid_argument(
        "need spectra[ell,field,field], noise[field] and observable rows");
  }
  if (ell_min < 0
      || !std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    throw std::invalid_argument("need ell_min>=0 and 0<area_sr<=4*pi");
  }
  if (realspace) {
    if (operators.ndim() != 3
        || operators.shape(0) != 4
        || operators.shape(1) < 1
        || operators.shape(2) != spectra.shape(0)
        || pair_area.ndim() != 1
        || pair_area.size() != operators.shape(1)
        || ell_min < 2) {
      throw std::invalid_argument(
          "real-space needs operators[4,bin,ell], pair_area[bin], ell_min>=2");
    }
  } else if (operators.ndim() != 2
             || operators.shape(0) < 1
             || operators.shape(1) != spectra.shape(0)) {
    throw std::invalid_argument("Fourier operators must have shape [bin,ell]");
  }

  const int nell = spectra.shape(0); // shared integer-multipole count
  const int nfield = noise.size();  // number of catalogs/observed fields
  const int nobs = rows.shape(0);   // number of measured field pairs
  const int nbin = operators.shape(realspace ? 1 : 0); // bins per observable
  const int offset = realspace ? 1 : 0; // first field column in the row map
  const int ndata = nobs*nbin;      // dimension of the returned covariance

  // These four numeric arrays enter arithmetic directly. Reject NaN/Inf
  // here so the notebook gets an exception before a C primitive runs.
  for (const auto* array : {&spectra, &noise, &operators, &pair_area}) {
    for (py::ssize_t index=0; index<array->size(); index++) {
      if (!std::isfinite(array->data()[index])) {
        throw std::invalid_argument("Gaussian matrix inputs must be finite");
      }
    }
  }
  for (int field=0; field<nfield; field++) {
    if (noise.data()[field] < 0.0) {
      throw std::invalid_argument("white noise powers must be nonnegative");
    }
  }
  for (int observable=0; observable<nobs; observable++) {
    if (realspace
        && (rows.at(observable, 0) < XI_PLUS_COV
            || rows.at(observable, 0) > W_THETA_COV)) {
      throw std::invalid_argument("real-space probe IDs must lie in 0..3");
    }
    for (int leg=0; leg<2; leg++) {
      const int field = rows.at(observable, offset+leg);
      if (field < 0
          || field >= nfield) {
        throw std::invalid_argument("observable field ID exceeds spectra");
      }
    }
  }
  if (realspace) {
    for (int bin=0; bin<nbin; bin++) {
      if (pair_area.data()[bin] <= 0.0) {
        throw std::invalid_argument("ordered-pair areas must be positive");
      }
    }
  }

  // --- 2. PREPARE CONTIGUOUS SPECTRA AND REUSABLE BLOCK SCRATCH ---

  // C integrates along ell. In the input, adjacent values instead belong
  // to different fields. This one transpose gives each Wick input a full
  // contiguous ell row, reused by every observable that needs that pair.
  std::vector<double> power((size_t) nfield*nfield*nell);
  for (int first=0; first<nfield; first++) {
    for (int second=0; second<nfield; second++) {
      double* row = power.data()+((size_t) first*nfield+second)*nell;
      for (int ell=0; ell<nell; ell++) {
        row[ell] = spectra.at(ell, first, second);
      }
    }
  }

  // Cov(AB,CD)=Cov(CD,AB). List the upper-triangle observable pairs once,
  // so workers get equal numbers of blocks even though triangular rows
  // have different lengths. Each task also owns its mirrored output block.
  // This list changes only work assignment, not any multipole or sum order.
  std::vector<int> first_observable;  // left measured row of each task
  std::vector<int> second_observable; // right measured row of each task
  for (int first=0; first<nobs; first++) {
    for (int second=first; second<nobs; second++) {
      first_observable.push_back(first);
      second_observable.push_back(second);
    }
  }

  matrix_array_cov output({ndata, ndata}); // owned complete matrix
  double* result = output.mutable_data(); // shared disjoint output blocks

  // --- 3. COMPUTE EACH OBSERVABLE BLOCK ONCE AND COPY ITS TRANSPOSE ---

  // Many measured pairs provide many independent covariance blocks. Give
  // each worker complete blocks, avoiding a team synchronization after
  // every small angular-bin sum. C detects this outer parallel region and
  // keeps its inner loops on the calling worker; its SIMDe lanes still
  // calculate distinct angular-bin outputs in the same ell order.
  // A single-observable call leaves bin-level parallelism to C instead.
  #pragma omp parallel if(nobs > 1)
  {
    std::vector<double> harmonic(nell); // covariance per integer multipole
    std::vector<double> weighted((size_t) nbin*nell); // weighted left kernels
    std::vector<double> block((size_t) nbin*nbin); // one angular/band block
    std::vector<double*> weighted_rows(nbin); // C pointers into scratch
    std::vector<double*> block_rows(nbin); // C pointers into block output
    std::vector<const double*> left_kernel(nbin); // selected left operator
    std::vector<const double*> right_kernel(nbin); // selected right operator

    // Each worker reuses its own arrays for successive blocks. Different
    // workers must never overwrite the weighted kernels of an active sum.
    for (int bin=0; bin<nbin; bin++) {
      weighted_rows[bin] = weighted.data()+(size_t) bin*nell;
      block_rows[bin] = block.data()+(size_t) bin*nbin;
    }

    // A task couples measured AB to CD. Its Wick inputs AC, BD, AD, BC
    // come from the complete field matrix, including unmeasured pairs.
    #pragma omp for schedule(static)
    for (size_t task=0; task<first_observable.size(); task++) {
      const int first = first_observable[task];
      const int second = second_observable[task];
      const int a = rows.at(first, offset);
      const int b = rows.at(first, offset+1);
      const int left_probe = realspace ? rows.at(first, 0) : 0;
      const int c = rows.at(second, offset);
      const int d = rows.at(second, offset+1);
      const int right_probe = realspace ? rows.at(second, 0) : 0;
      const int fields[4] = {a, b, c, d};
      const double noise_ab[2] = {noise.at(a), noise.at(b)};
      const double cross_noise[4] = {
        a == c ? noise.at(a) : 0.0,
        b == d ? noise.at(b) : 0.0,
        a == d ? noise.at(a) : 0.0,
        b == c ? noise.at(b) : 0.0
      };
      const double* cross[4] = {
        power.data()+((size_t) a*nfield+c)*nell,
        power.data()+((size_t) b*nfield+d)*nell,
        power.data()+((size_t) a*nfield+d)*nell,
        power.data()+((size_t) b*nfield+c)*nell
      };

      // Select the spin operator in real space. Fourier bands use one
      // common operator because the supplied spectra already describe
      // the observed fields, including the shear convention.
      for (int bin=0; bin<nbin; bin++) {
        left_kernel[bin] = operators.data()
                          +((size_t) left_probe*nbin+bin)*nell;
        right_kernel[bin] = operators.data()
                           +((size_t) right_probe*nbin+bin)*nell;
      }
      gaussian_wick_cov(ell_min, nell, area_sr/(4.0*M_PI), cross,
          cross_noise, !realspace, harmonic.data());
      gaussian_project_cov(nbin, nbin, nell, left_kernel.data(),
          right_kernel.data(), harmonic.data(), weighted_rows.data(),
          block_rows.data());

      if (realspace) {
        // Disjoint angular bins share pure pair noise only on matching
        // bins. Its analytic expression includes modes above ell_max.
        for (int bin=0; bin<nbin; bin++) {
          block_rows[bin][bin] += gaussian_noise_pair_cov(
              (enum probe_cov) left_probe, (enum probe_cov) right_probe,
              fields, noise_ab, pair_area.at(bin));
        }
      }

      // Global row order is observable first, then angular/Fourier bin.
      // On a diagonal block, retain its upper triangle so independently
      // rounded reverse products cannot overwrite the chosen entry.
      for (int left=0; left<nbin; left++) {
        const int start = first == second ? left : 0;
        for (int right=start; right<nbin; right++) {
          const int i = first*nbin+left;
          const int j = second*nbin+right;
          result[(size_t) i*ndata+j] = block_rows[left][right];
          result[(size_t) j*ndata+i] = block_rows[left][right];
        }
      }
    }
  }
  return output;
}

// One angular block shares its matter trispectrum among all catalog pairs.
// Catalog indices live in the separate per-probe lists below.
struct angular_block_cov {
  int left_probe;  // left statistic: xi+, xi-, gamma_t or w
  int right_probe; // right statistic in that same 0..3 convention
  int left_bin;    // left angular or Fourier bin
  int right_bin;   // right angular or Fourier bin
};

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
// Compute that common radial weight once per angular block, then call
// the existing SIMD projection for every catalog pairing in that block.
// Distribute whole blocks over one OpenMP team. This avoids starting two
// new parallel regions for each of thousands of small Python calls.
// Each worker has its own weight, weighted windows and output scratch.
// gaussian_project_cov supplies the common weighted-dot-product kernel:
// here its integration nodes are radial shells, rather than multipoles.
//
// Keep the previous arithmetic: first measure*T, then W_left*weight,
// then the C primitive's ordered sum against W_right. No mass, distance
// or angular node changes. Signed trispectra remain signed. Copy the
// chosen triangular result to its transpose, without averaging entries.
// The supplied arrays are read-only; NumPy owns the returned matrix.
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
    throw std::invalid_argument(
        "need probes[observable], pair_window[observable,node], "
        "projected[4*bin,4*bin,node] and measure[node]");
  }
  for (py::ssize_t row=0; row<probes.size(); row++) {
    if (probes.data()[row] < XI_PLUS_COV
        || probes.data()[row] > W_THETA_COV) {
      throw std::invalid_argument("connected probe IDs must lie in 0..3");
    }
  }
  for (const auto* array : {&pair_window, &projected, &measure}) {
    for (py::ssize_t index=0; index<array->size(); index++) {
      if (!std::isfinite(array->data()[index])) {
        throw std::invalid_argument(
            "connected projection inputs must be finite");
      }
    }
  }

  const int nobs = probes.size();           // measured field-pair count
  const int nnode = measure.size();         // common radial sample count
  const int ntransform = projected.shape(0); // four probes times bin count
  const int nbin = ntransform/4;            // bins within each observable
  const int ndata = nobs*nbin;              // complete covariance dimension

  // --- 2. GROUP CATALOG PAIRS AND ENUMERATE INDEPENDENT ANGULAR BLOCKS ---

  // Preserve input order within a probe. On a diagonal angular block,
  // that order chooses which rounded product supplies both triangles.
  // Pointers reference the supplied windows directly; no table is copied.
  std::vector<std::vector<int>> groups(4); // observable IDs for each probe
  std::vector<std::vector<const double*>> windows(4); // matching C rows
  int largest = 0; // maximum catalog-pair count in any one probe

  for (int row=0; row<nobs; row++) {
    const int probe = probes.data()[row]; // this observable's statistic
    groups[probe].push_back(row);
    windows[probe].push_back(pair_window.data(row, 0));
    if ((int) groups[probe].size() > largest) {
      largest = groups[probe].size();
    }
  }

  std::vector<angular_block_cov> tasks; // each also owns its transpose

  // Different probes require every bin pairing. Equal probes need only
  // the angular upper triangle, because exchanging both observables
  // gives the transpose. Empty probe groups contribute no matrix entries.
  for (int left=0; left<4; left++) {
    if (groups[left].empty()) continue;
    for (int right=left; right<4; right++) {
      if (groups[right].empty()) continue;
      for (int first=0; first<nbin; first++) {
        const int start = left == right ? first : 0; // first distinct bin
        for (int second=start; second<nbin; second++) {
          tasks.push_back({left, right, first, second});
        }
      }
    }
  }

  matrix_array_cov output({ndata, ndata}); // owned complete covariance
  double* result = output.mutable_data(); // disjoint blocks written below
  const double* matter = projected.data(); // shared transformed trispectra
  const double* radial_measure = measure.data(); // shared radial weights

  // --- 3. PROJECT COMPLETE ANGULAR BLOCKS WITH PRIVATE WORKER SCRATCH ---

  // Blocks have different numbers of catalog pairs. Hand successive
  // blocks to successive workers, cycling through the team: every worker
  // then receives blocks from both large and small probe groups. Within
  // a block, C retains its SIMDe sums and suppresses nested parallelism.
  #pragma omp parallel if(tasks.size() > 1)
  {
    std::vector<double> weight(nnode); // measure times this block's T
    std::vector<double> weighted((size_t) largest*nnode); // weighted W_left
    std::vector<double> block((size_t) largest*largest); // catalog-pair block
    std::vector<double*> weighted_rows(largest); // C scratch row pointers
    std::vector<double*> block_rows(largest); // C output row pointers

    for (int row=0; row<largest; row++) {
      weighted_rows[row] = weighted.data()+(size_t) row*nnode;
      block_rows[row] = block.data()+(size_t) row*largest;
    }

    // Each task owns all catalog pairings for these two angular bins,
    // including their mirrored entries. No two tasks update the same cell.
    #pragma omp for schedule(static, 1)
    for (size_t task=0; task<tasks.size(); task++) {
      const angular_block_cov& item = tasks[task]; // two probes and bins
      const auto& left = groups[item.left_probe]; // left observable IDs
      const auto& right = groups[item.right_probe]; // right observable IDs
      const int first = item.left_probe*nbin+item.left_bin; // T row
      const int second = item.right_probe*nbin+item.right_bin; // T column
      const double* row = matter
          +((size_t) first*ntransform+second)*nnode; // this block's T(alpha)

      // Form measure*T at each shell before weighting any catalog. SIMD
      // lanes 0 and 1 hold two adjacent shells here, not parts of a sum.
      // The projection below still adds shells in their original order.
      int node = 0; // first radial sample not yet weighted
      for (; node+1<nnode; node+=2) {
        // Read two adjacent radial measures. loadu needs no special
        // alignment of the supplied NumPy array.
        const simde__m128d vm = simde_mm_loadu_pd(radial_measure+node);

        // Read T at the same two shells into the matching SIMD lanes.
        const simde__m128d vt = simde_mm_loadu_pd(row+node);

        // Multiply within each lane: [measure_0*T_0, measure_1*T_1].
        // No shell is added to another at this preparation stage.
        const simde__m128d vw = simde_mm_mul_pd(vm, vt);

        // Store both weights in this worker's ordinary contiguous array.
        simde_mm_storeu_pd(weight.data()+node, vw);
      }
      if (node < nnode) {
        weight[node] = radial_measure[node]*row[node];
      }

      gaussian_project_cov(left.size(), right.size(), nnode,
          windows[item.left_probe].data(), windows[item.right_probe].data(),
          weight.data(), weighted_rows.data(), block_rows.data());

      // At equal probe and angular bin, choose the catalog upper triangle.
      // All other blocks retain every catalog pairing. Copying each value
      // to its transpose makes symmetry exact even for signed spectra.
      const bool diagonal = item.left_probe == item.right_probe
                            && item.left_bin == item.right_bin;
      for (size_t i=0; i<left.size(); i++) {
        const size_t start = diagonal ? i : 0; // first distinct catalog
        for (size_t j=start; j<right.size(); j++) {
          const int global_left = left[i]*nbin+item.left_bin; // output row
          const int global_right = right[j]*nbin+item.right_bin; // column
          result[(size_t) global_left*ndata+global_right] = block_rows[i][j];
          result[(size_t) global_right*ndata+global_left] = block_rows[i][j];
        }
      }
    }
  }
  return output;
}

void bind_covariance_wrappers(py::module_& module)
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
         const matrix_array_cov& pair_area_sr2) {
        return gaussian_matrix_cpp(spectra, noise, rows, operators, ell_min,
                                   area_sr, pair_area_sr2, true);
      },
      R"doc(Compute a real-space Gaussian matrix from supplied field spectra.

spectra[ell,field,field] contains signal in the observed-shear convention;
noise[field] gives independent white shot/shape powers per steradian.
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
      py::arg("pair_area_sr2").noconvert());

  module.def("covariance_gaussian_fourier",
      [](const matrix_array_cov& spectra, const matrix_array_cov& noise,
         const index_array_cov& pairs, const matrix_array_cov& operators,
         const int ell_min, const double area_sr) {
        const matrix_array_cov unused((py::ssize_t) 0);
        return gaussian_matrix_cpp(spectra, noise, pairs, operators, ell_min,
                                   area_sr, unused, false);
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
