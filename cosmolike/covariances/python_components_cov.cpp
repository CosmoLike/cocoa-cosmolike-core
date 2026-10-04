#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include "generic_interface_cov.hpp"
#include "gaussian_cov.h"
#include "halo_cov.h"
#include "mask_cov.h"
#include "non_gaussian_cov.h"
#include "operators_cov.h"
#include "perturbation_cov.h"
#include "ssc_cov.h"
#include "spectra_cov.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"

namespace py = pybind11;

namespace cosmolike_interface {

// NumPy owns all arrays. A C-style array has adjacent elements within each
// row, so its rows can be passed directly to the existing C components.
// The small pointer vectors below describe those rows; they copy no data.
using cov_array = py::array_t<double, py::array::c_style>;
using cov_int_array = py::array_t<int, py::array::c_style>;

static void finite_cov(const cov_array& values, const char* name)
{
  for (py::ssize_t index=0; index<values.size(); index++) {
    if (!std::isfinite(values.data()[index])) {
      throw std::invalid_argument(std::string(name)+" must be finite");
    }
  }
}

static void vector_cov(const cov_array& values, const char* name)
{
  if (values.ndim() != 1
      || values.size() < 1) {
    throw std::invalid_argument(
        std::string(name)+" must be a nonempty 1D array");
  }
  finite_cov(values, name);
}

static void matrix_cov(const cov_array& values, const char* name)
{
  if (values.ndim() != 2
      || values.shape(0) < 1
      || values.shape(1) < 1) {
    throw std::invalid_argument(
        std::string(name)+" must be a nonempty 2D array");
  }
  finite_cov(values, name);
}

static std::vector<const double*> input_rows_cov(const cov_array& values)
{
  std::vector<const double*> rows(values.shape(0));
  for (py::ssize_t row=0; row<values.shape(0); row++) {
    rows[row] = values.data(row, 0);
  }
  return rows;
}

static std::vector<double*> output_rows_cov(cov_array& values)
{
  std::vector<double*> rows(values.shape(0));
  for (py::ssize_t row=0; row<values.shape(0); row++) {
    rows[row] = values.mutable_data(row, 0);
  }
  return rows;
}

static void quadrature_cov(const int nquad)
{
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    throw std::invalid_argument("nquad must be 64,96,128,256,512 or 1024");
  }
}

static void angles_cov(const cov_array& edges)
{
  vector_cov(edges, "edges_rad");
  if (edges.size() < 2
      || edges.data()[0] < 0.0
      || edges.data()[edges.size()-1] > M_PI) {
    throw std::invalid_argument("need at least two angle edges inside [0,pi]");
  }
  for (py::ssize_t edge=1; edge<edges.size(); edge++) {
    if (edges.data()[edge] <= edges.data()[edge-1]) {
      throw std::invalid_argument("angle edges must increase strictly");
    }
  }
}

// ---------------------------------------------------------------------------
// Project supplied rows through a shared weighted sum.
//
// Every output is sum_node left[i,node]*weight[node]*right[j,node].
// For Gaussian covariance the nodes are multipoles. For density SSC they
// can instead be radial shells. In the latter case, using the same
// responses on both sides and positive weights preserves nonnegative
// variance. The caller can request a rectangular matrix subblock without
// changing the summation order. All parallel work remains in the C code.
// ---------------------------------------------------------------------------
static cov_array covariance_project(
    const cov_array& left,   // [nleft,nnode], left response/operator rows
    const cov_array& right,  // [nright,nnode], right rows
    const cov_array& weight  // [nnode], common integration weights
  )
{
  matrix_cov(left, "left");
  matrix_cov(right, "right");
  vector_cov(weight, "weight");
  if (left.shape(1) != weight.size()
      || right.shape(1) != weight.size()) {
    throw std::invalid_argument("left/right columns must match weight length");
  }

  cov_array scratch({left.shape(0), weight.size()});
  cov_array output({left.shape(0), right.shape(0)});
  auto left_rows = input_rows_cov(left);
  auto right_rows = input_rows_cov(right);
  auto scratch_rows = output_rows_cov(scratch);
  auto output_rows = output_rows_cov(output);

  gaussian_project_cov(left.shape(0), right.shape(0), weight.size(),
      left_rows.data(), right_rows.data(), weight.data(),
      scratch_rows.data(), output_rows.data());
  return output;
}

// The four rows are the AC, BD, AD and BC pairings of an AB-by-CD block.
// Noise is supplied separately so real-space calculations can replace the
// pure noise product by the analytic number of available galaxy pairs.
static cov_array covariance_gaussian_wick(
    const cov_array& cross_spectra, // [4,nell], signal only
    const cov_array& cross_noise,   // [4], matching white-noise powers
    const int ell_min,              // first consecutive integer multipole
    const double fsky,              // survey area / (4*pi)
    const bool include_noise_noise  // retain the pure noise product
  )
{
  matrix_cov(cross_spectra, "cross_spectra");
  vector_cov(cross_noise, "cross_noise");
  if (cross_spectra.shape(0) != 4
      || cross_noise.size() != 4
      || ell_min < 0
      || !std::isfinite(fsky)
      || fsky <= 0.0
      || fsky > 1.0) {
    throw std::invalid_argument("need four pairings, ell_min>=0 and 0<fsky<=1");
  }

  auto rows = input_rows_cov(cross_spectra);
  cov_array output(cross_spectra.shape(1));
  gaussian_wick_cov(ell_min, output.size(), fsky, rows.data(),
      cross_noise.data(), include_noise_noise, output.mutable_data());
  return output;
}

// Build all four estimator operators together. The flattened C row order
// is probe*nbin+bin; reshape on return without copying the owned data.
static cov_array covariance_realspace_operator(
    const cov_array& edges_rad, // angular-bin boundaries in radians
    const int ell_max,          // last integer multipole, inclusive
    const int nquad             // integration nodes per angular bin
  )
{
  angles_cov(edges_rad);
  quadrature_cov(nquad);
  if (ell_max < 2) {
    throw std::invalid_argument("ell_max must be at least 2");
  }
  const py::ssize_t nbin = edges_rad.size()-1;
  cov_array output({4*nbin, (py::ssize_t) ell_max+1});
  auto rows = output_rows_cov(output);
  realspace_operator_cov(nbin, edges_rad.data(), ell_max, nquad, rows.data());
  output.resize({(py::ssize_t) 4, nbin, (py::ssize_t) ell_max+1});
  return output;
}

static cov_array covariance_bandpower_operator(
    const cov_int_array& first, // inclusive lower multipole of each band
    const cov_int_array& last,  // inclusive upper multipole
    const int ell_min,          // first multipole of the shared output grid
    const int nell             // number of consecutive output multipoles
  )
{
  if (first.ndim() != 1
      || last.ndim() != 1
      || first.size() < 1
      || first.size() != last.size()
      || ell_min < 0
      || nell < 1) {
    throw std::invalid_argument(
        "need equal 1D band bounds and a valid ell grid");
  }
  for (py::ssize_t band=0; band<first.size(); band++) {
    if (first.data()[band] < ell_min
        || last.data()[band] < first.data()[band]
        || last.data()[band] >= ell_min+nell) {
      throw std::invalid_argument("band bounds must lie inside the ell grid");
    }
  }

  cov_array output({first.size(), (py::ssize_t) nell});
  auto rows = output_rows_cov(output);
  bandpower_operator_cov(first.size(), ell_min, nell, first.data(),
      last.data(), rows.data());
  return output;
}

static double covariance_noise_pair(
    const int probe_left,       // xi+, xi-, gamma_t, w: 0,1,2,3
    const int probe_right,      // right estimator in the same convention
    const cov_int_array& fields,// [4], A,B,C,D global catalog indices
    const cov_array& noise_ab,  // [2], powers of catalogs A and B
    const double pair_area_sr2 // ordered-pair area within the angular bin
  )
{
  vector_cov(noise_ab, "noise_ab");
  if (fields.ndim() != 1
      || fields.size() != 4
      || noise_ab.size() != 2
      || probe_left < 0
      || probe_left > 3
      || probe_right < 0
      || probe_right > 3
      || !std::isfinite(pair_area_sr2)
      || pair_area_sr2 <= 0.0) {
    throw std::invalid_argument(
        "invalid estimator, catalog or pair-area input");
  }
  for (int index=0; index<4; index++) {
    if (fields.data()[index] < 0) {
      throw std::invalid_argument("catalog indices must be nonnegative");
    }
  }
  if (noise_ab.data()[0] < 0.0
      || noise_ab.data()[1] < 0.0) {
    throw std::invalid_argument("noise powers must be nonnegative");
  }
  return gaussian_noise_pair_cov((probe_cov) probe_left,
      (probe_cov) probe_right, fields.data(), noise_ab.data(), pair_area_sr2);
}


// A raw mask power is nonnegative and keeps C0=area^2/(4*pi). Checking
// that convention here prevents interpreting a normalized mask as raw.
static void raw_mask_cov(const cov_array& mask, const double area)
{
  vector_cov(mask, "mask_cl");
  if (!std::isfinite(area)
      || area <= 0.0
      || area > 4.0*M_PI) {
    throw std::invalid_argument("area_sr must lie in (0,4*pi]");
  }
  for (py::ssize_t ell=0; ell<mask.size(); ell++) {
    if (mask.data()[ell] < 0.0) {
      throw std::invalid_argument("mask_cl must be nonnegative");
    }
  }
  const double monopole = area*area/(4.0*M_PI);
  if (std::fabs(mask.data()[0]/monopole-1.0) > 1.e-8) {
    throw std::invalid_argument("raw mask C0 must equal area_sr^2/(4*pi)");
  }
}

static cov_array covariance_mask_pair_area(
    const cov_array& edges_rad,    // angular-bin edges in radians
    const cov_array& mask_cl,      // raw footprint spectrum, L=0 onward
    const double area_sr,          // footprint area
    const cov_array& scalar_kernel// [nbin,nmask], bin-averaged w operator
  )
{
  angles_cov(edges_rad);
  raw_mask_cov(mask_cl, area_sr);
  matrix_cov(scalar_kernel, "scalar_kernel");
  if (scalar_kernel.shape(0) != edges_rad.size()-1
      || scalar_kernel.shape(1) != mask_cl.size()) {
    throw std::invalid_argument("scalar_kernel must have shape [nbin,nmask]");
  }

  cov_array output(edges_rad.size()-1);
  auto rows = input_rows_cov(scalar_kernel);
  mask_pair_area_cov(output.size(), mask_cl.size(), area_sr,
      edges_rad.data(), mask_cl.data(), rows.data(), output.mutable_data());
  return output;
}

// The output describes the background power of a radial shell in the
// long-mode Limber approximation. It has units of length; the radial
// quadrature weight is applied later when the observable responses meet.
static cov_array covariance_ssc_mask_variance(
    const cov_array& mask_cl, // raw footprint spectrum
    const double area_sr,    // footprint area in steradians
    const cov_array& distance,// positive transverse distances, c/H0
    const cov_array& power   // [nnode,nmask], linear power, (c/H0)^3
  )
{
  raw_mask_cov(mask_cl, area_sr);
  vector_cov(distance, "distance");
  matrix_cov(power, "power");
  if (power.shape(0) != distance.size()
      || power.shape(1) != mask_cl.size()) {
    throw std::invalid_argument("power must have shape [nnode,nmask]");
  }
  for (py::ssize_t node=0; node<distance.size(); node++) {
    if (distance.data()[node] <= 0.0) {
      throw std::invalid_argument("distance must be positive");
    }
  }

  cov_array output(distance.size());
  auto rows = input_rows_cov(power);
  ssc_mask_variance_cov(distance.size(), mask_cl.size(), area_sr,
      mask_cl.data(), distance.data(), rows.data(), output.mutable_data());
  return output;
}

// Each row describes one observable at one multipole. Its shell response
// includes the change of matter clustering and, when present, the change
// in the catalog mean used to define observed galaxy density.
static cov_array covariance_ssc_shell_response(
    const cov_array& distance,       // [nnode], transverse distances
    const cov_array& signal,         // [nrow], complete projected spectra
    const cov_array& pair_window,    // [nrow,nnode], W_A*W_B
    const cov_array& mean_window,    // [nrow,nnode], U_A+U_B
    const cov_array& power_response  // [nrow,nnode], dP/d(delta_b)
  )
{
  vector_cov(distance, "distance");
  vector_cov(signal, "signal");
  matrix_cov(pair_window, "pair_window");
  matrix_cov(mean_window, "mean_window");
  matrix_cov(power_response, "power_response");
  const py::ssize_t nrow = signal.size();
  const py::ssize_t nnode = distance.size();

  if (pair_window.shape(0) != nrow
      || mean_window.shape(0) != nrow
      || power_response.shape(0) != nrow
      || pair_window.shape(1) != nnode
      || mean_window.shape(1) != nnode
      || power_response.shape(1) != nnode) {
    throw std::invalid_argument("response inputs must have shape [nrow,nnode]");
  }
  for (py::ssize_t node=0; node<nnode; node++) {
    if (distance.data()[node] <= 0.0) {
      throw std::invalid_argument("distance must be positive");
    }
  }

  cov_array output({nrow, nnode});
  auto pair_rows = input_rows_cov(pair_window);
  auto mean_rows = input_rows_cov(mean_window);
  auto power_rows = input_rows_cov(power_response);
  auto rows = output_rows_cov(output);
  ssc_shell_response_cov(nrow, nnode, distance.data(), signal.data(),
      pair_rows.data(), mean_rows.data(), power_rows.data(), rows.data());
  return output;
}

// Return both the one-profile moment and the five pair moments. Their
// NumPy arrays own the values after the C routine releases its workspace.
// The temporary pointers merely give C access to [role][a][pair] rows;
// no moment is copied and no covariance approximation is chosen here.
// A response slope can request I11 alone: its other moments are already
// available at the central k. Return None for the omitted pair array.
static py::tuple covariance_halo_moments(
    const cov_array& a,         // scale factors
    const cov_array& k,         // [na,nk], inverse c/H0
    const cov_array& lnm_edges, // logarithmic halo-mass panel boundaries
    const int nquad,           // Gaussian mass nodes per panel
    const bool pair_moments    // also compute the five pair-moment roles
  )
{
  vector_cov(a, "a");
  matrix_cov(k, "k");
  vector_cov(lnm_edges, "lnm_edges");
  quadrature_cov(nquad);
  if (k.shape(0) != a.size()
      || lnm_edges.size() < 2) {
    throw std::invalid_argument(
        "k needs na rows and lnm_edges needs two edges");
  }
  if (cosmology.lnPL == nullptr
      || cosmology.G == nullptr
      || like.halo_model[3] != HALO_PROFILE_NFW) {
    throw std::invalid_argument("initialize cosmology and NFW halo profiles");
  }
  for (py::ssize_t row=0; row<a.size(); row++) {
    if (a.data()[row] < limits.a_min
        || a.data()[row] >= 1.0) {
      throw std::invalid_argument("a must lie in the initialized halo range");
    }
  }
  for (py::ssize_t index=0; index<k.size(); index++) {
    if (k.data()[index] < 0.0) {
      throw std::invalid_argument("halo wavenumbers must be nonnegative");
    }
  }
  for (py::ssize_t edge=0; edge<lnm_edges.size(); edge++) {
    if (lnm_edges.data()[edge] < std::log(limits.halo_m[RANGE_MIN])
        || lnm_edges.data()[edge] > std::log(limits.halo_m[RANGE_MAX])
        || (edge > 0
            && lnm_edges.data()[edge] <= lnm_edges.data()[edge-1])) {
      throw std::invalid_argument(
          "mass edges must increase inside halo limits");
    }
  }

  const py::ssize_t na = a.size();
  const py::ssize_t nk = k.shape(1);
  const py::ssize_t npair = nk*(nk+1)/2;
  cov_array i11({na, nk});
  auto k_rows = input_rows_cov(k);
  auto i11_rows = output_rows_cov(i11);

  if (!pair_moments) {
    halo_moments_cov(na, a.data(), nk, k_rows.data(), lnm_edges.size()-1,
        lnm_edges.data(), nquad, i11_rows.data(), nullptr);
    return py::make_tuple(i11, py::none());
  }

  cov_array moments({(py::ssize_t) 5, na, npair});
  std::vector<std::vector<double*>> moment_rows(5);
  std::vector<double**> roles(5);

  for (int role=0; role<5; role++) {
    moment_rows[role].resize(na);
    for (py::ssize_t row=0; row<na; row++) {
      moment_rows[role][row] = moments.mutable_data(role, row, 0);
    }
    roles[role] = moment_rows[role].data();
  }

  halo_moments_cov(na, a.data(), nk, k_rows.data(), lnm_edges.size()-1,
      lnm_edges.data(), nquad, i11_rows.data(), roles.data());
  return py::make_tuple(i11, moments);
}

// Read one vector or a matrix of physical wavenumbers at a shared a.
// Matrix rows are independent integration batches distributed by C over
// OpenMP workers. A vector retains the original serial core-reader path.
// The output has the input's shape; the covariance owns the input grid.
static cov_array covariance_power(
    const double a,      // scale factor inside the initialized range
    const cov_array& k,  // physical wavenumbers in inverse c/H0
    const bool linear   // linear total-matter P or the configured Pdelta
  )
{
  if (k.ndim() == 1) {
    vector_cov(k, "k");
  } else {
    matrix_cov(k, "k");
  }
  if (!std::isfinite(a)
      || a < limits.a_min
      || a >= 1.0
      || cosmology.lnPL == nullptr
      || (!linear
          && cosmology.lnP == nullptr)) {
    throw std::invalid_argument("initialize power tables and use a_min<=a<1");
  }
  for (py::ssize_t index=0; index<k.size(); index++) {
    if (k.data()[index] <= 0.0) {
      throw std::invalid_argument("power wavenumbers must be positive");
    }
  }

  cov_array output(k.request().shape);
  if (k.ndim() == 2) {
    auto k_rows = input_rows_cov(k);
    auto rows = output_rows_cov(output);
    power_rows_cov(a, k.shape(0), k.shape(1), k_rows.data(), linear,
                    rows.data());
  } else if (linear) {
    p_lin_at_a(a, k.data(), k.size(), output.mutable_data());
  } else {
    Pdelta_at_a(a, k.data(), k.size(), output.mutable_data());
  }
  return output;
}


// The angular rule and its power samples are supplied together. The
// arrays must describe the same K,Q pairs and the same angular nodes;
// otherwise the cancellations between perturbation diagrams are lost.
static cov_array covariance_tree_averages(
    const cov_array& k,      // [2,npair], positive K and Q
    const cov_array& pk,     // [2,npair], matching linear power
    const cov_array& corner, // [nangle], stable 1+cos(theta)
    const cov_array& weight, // [nangle], normalized dtheta/pi weights
    const cov_array& ps      // [npair,nangle], P(|K+Q|)
  )
{
  matrix_cov(k, "k");
  matrix_cov(pk, "pk");
  matrix_cov(ps, "ps");
  vector_cov(corner, "corner");
  vector_cov(weight, "weight");
  if (k.shape(0) != 2
      || pk.shape(0) != 2
      || pk.shape(1) != k.shape(1)
      || ps.shape(0) != k.shape(1)
      || ps.shape(1) != corner.size()
      || corner.size() != weight.size()) {
    throw std::invalid_argument(
        "tree input pair and angle dimensions disagree");
  }
  for (py::ssize_t point=0; point<k.size(); point++) {
    if (k.data()[point] <= 0.0) {
      throw std::invalid_argument("tree wavenumbers must be positive");
    }
  }
  double normalization = 0.0;
  for (py::ssize_t node=0; node<corner.size(); node++) {
    if (corner.data()[node] <= 0.0
        || corner.data()[node] > 2.0
        || weight.data()[node] <= 0.0) {
      throw std::invalid_argument("need 0<corner<=2 and positive weights");
    }
    normalization += weight.data()[node];
  }
  if (std::fabs(normalization-1.0) > 1.e-10) {
    throw std::invalid_argument("angular weights must sum to one");
  }

  cov_array output({(py::ssize_t) 3, k.shape(1)});
  auto k_rows = input_rows_cov(k);
  auto pk_rows = input_rows_cov(pk);
  auto ps_rows = input_rows_cov(ps);
  auto rows = output_rows_cov(output);
  tree_averages_cov(k.shape(1), corner.size(), k_rows.data(),
      pk_rows.data(), corner.data(), weight.data(), ps_rows.data(),
      rows.data());
  return output;
}

// Keep the five halo contributions separate in the returned array so
// callers can examine their scale dependence before projecting their sum.
// All supplied moments and powers must refer to the same density field.
static cov_array covariance_halo_trispectrum(
    const cov_array& pk,      // [2,npoint], linear power at K,Q
    const cov_array& i11,     // [2,npoint], one-profile moments
    const cov_array& moments, // [5,npoint], pair moments in halo_cov order
    const cov_array& tree     // [3,npoint], angular P/B/T averages
  )
{
  matrix_cov(pk, "pk");
  matrix_cov(i11, "i11");
  matrix_cov(moments, "moments");
  matrix_cov(tree, "tree");
  const py::ssize_t npoint = pk.shape(1);
  if (pk.shape(0) != 2
      || i11.shape(0) != 2
      || moments.shape(0) != 5
      || tree.shape(0) != 3
      || i11.shape(1) != npoint
      || moments.shape(1) != npoint
      || tree.shape(1) != npoint) {
    throw std::invalid_argument(
        "trispectrum inputs need 2,2,5,3 matching rows");
  }

  cov_array output({(py::ssize_t) 5, npoint});
  auto pk_rows = input_rows_cov(pk);
  auto i11_rows = input_rows_cov(i11);
  auto moment_rows = input_rows_cov(moments);
  auto tree_rows = input_rows_cov(tree);
  auto rows = output_rows_cov(output);
  halo_trispectrum_cov(npoint, pk_rows.data(), i11_rows.data(),
      moment_rows.data(), tree_rows.data(), rows.data());
  return output;
}

// The two coefficients and the differentiated spectrum are explicit:
// the Python workflow selects a response prescription, not the binding.
// Row 0 returns halo power; row 1 returns its dimensional response or
// the fractional response transferred to the supplied target power.
static cov_array covariance_halo_response(
    const cov_array& inputs,            // [6,npoint], documented halo inputs
    const double growth_coefficient,   // constant growth contribution
    const double dilation_coefficient, // coefficient of logarithmic slope
    const bool fractional              // transfer D_halo/P_halo to P_target
  )
{
  matrix_cov(inputs, "inputs");
  if (inputs.shape(0) != 6
      || !std::isfinite(growth_coefficient)
      || !std::isfinite(dilation_coefficient)) {
    throw std::invalid_argument(
        "need six response rows and finite coefficients");
  }
  for (py::ssize_t point=0; point<inputs.shape(1); point++) {
    const double i11 = *inputs.data(2, point);
    const double phalo = i11*i11*(*inputs.data(0, point))
                         +*inputs.data(3, point);
    if (!std::isfinite(phalo)
        || phalo <= 0.0) {
      throw std::invalid_argument("supplied moments must give positive halo P");
    }
  }

  cov_array output({(py::ssize_t) 2, inputs.shape(1)});
  auto input_rows = input_rows_cov(inputs);
  auto rows = output_rows_cov(output);
  halo_response_cov(inputs.shape(1), growth_coefficient,
      dilation_coefficient, fractional, input_rows.data(), rows.data());
  return output;
}

void bind_covariance_components(py::module_& module)
{

  module.def("covariance_mask_pair_area", &covariance_mask_pair_area,
      "Ordered-pair area [nbin] in sr^2 from a raw mask and bin kernel.",
      py::arg("edges_rad").noconvert(), py::arg("mask_cl").noconvert(),
      py::arg("area_sr"), py::arg("scalar_kernel").noconvert());

  module.def("covariance_ssc_mask_variance", &covariance_ssc_mask_variance,
      "Long-mode Limber background strength [nnode], in c/H0 length units.",
      py::arg("mask_cl").noconvert(), py::arg("area_sr"),
      py::arg("distance").noconvert(), py::arg("power").noconvert());

  module.def("covariance_ssc_shell_response", &covariance_ssc_shell_response,
      "Observable response [nrow,nnode], including projected mean subtraction.",
      py::arg("distance").noconvert(), py::arg("signal").noconvert(),
      py::arg("pair_window").noconvert(), py::arg("mean_window").noconvert(),
      py::arg("power_response").noconvert());

  module.def("covariance_halo_moments", &covariance_halo_moments,
      "Return I11 [na,nk] and moments [5,na,nk*(nk+1)/2]. "
      "pair_moments=False omits pair sums and returns (I11, None). "
      "Both use the cb halo convention.",
      py::arg("a").noconvert(), py::arg("k").noconvert(),
      py::arg("lnm_edges").noconvert(), py::arg("nquad"),
      py::arg("pair_moments") = true);

  module.def("covariance_power", &covariance_power,
      "Read power for a k vector or matrix at one a; retain its shape. "
      "Matrix rows use OpenMP. Input/output use the core c/H0 units.",
      py::arg("a"), py::arg("k").noconvert(), py::arg("linear") = true);

  module.def("covariance_tree_averages", &covariance_tree_averages,
      "Planar P/B/T averages [3,npair] from supplied linear-power inputs.",
      py::arg("k").noconvert(), py::arg("pk").noconvert(),
      py::arg("corner").noconvert(), py::arg("weight").noconvert(),
      py::arg("ps").noconvert());

  module.def("covariance_halo_trispectrum", &covariance_halo_trispectrum,
      "Return five halo terms [5,npoint]: 1h,2h(13),2h(22),3h,4h.",
      py::arg("pk").noconvert(), py::arg("i11").noconvert(),
      py::arg("moments").noconvert(), py::arg("tree").noconvert());

  module.def("covariance_halo_response", &covariance_halo_response,
      "Return halo power and dimensional response [2,npoint].",
      py::arg("inputs").noconvert(), py::arg("growth_coefficient"),
      py::arg("dilation_coefficient"), py::arg("fractional") = true);

  module.def("covariance_project", &covariance_project,
      "Contract rows through common weights; return an owned matrix.",
      py::arg("left").noconvert(), py::arg("right").noconvert(),
      py::arg("weight").noconvert());

  module.def("covariance_gaussian_wick", &covariance_gaussian_wick,
      "Gaussian AB,CD covariance from AC,BD,AD,BC rows on consecutive ell.",
      py::arg("cross_spectra").noconvert(), py::arg("cross_noise").noconvert(),
      py::arg("ell_min"), py::arg("fsky"),
      py::arg("include_noise_noise") = false);

  module.def("covariance_realspace_operator", &covariance_realspace_operator,
      "Return [4,nbin,ell_max+1] full-sky xi+,xi-,gamma_t,w operators.",
      py::arg("edges_rad").noconvert(), py::arg("ell_max"), py::arg("nquad"));

  module.def("covariance_bandpower_operator", &covariance_bandpower_operator,
      "Return normalized integer-mode weights [nband,nell]; bounds inclusive.",
      py::arg("first").noconvert(), py::arg("last").noconvert(),
      py::arg("ell_min"), py::arg("nell"));

  module.def("covariance_noise_pair", &covariance_noise_pair,
      "Pure real-space count/shape noise for one angular-bin pair area.",
      py::arg("probe_left"), py::arg("probe_right"),
      py::arg("fields").noconvert(), py::arg("noise_ab").noconvert(),
      py::arg("pair_area_sr2"));
}
}
