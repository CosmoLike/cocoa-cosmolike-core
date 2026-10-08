#include <cmath>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include "production_interface_cov.hpp"
#include "cosmolike/halo_cluster.h"
#include "cosmolike/redshift_spline_cluster.h"
#include "counts_cluster_cov.h"
#include "spectra_cluster_cov.h"
#include "moments_cluster_cov.h"
#include "halo_cluster_cov.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"
#include "cosmolike/structs_cluster.h"

namespace py = pybind11;

namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Production bindings for the cluster parts of the joint covariance.
//
// Inputs are borrowed: every array argument is noconvert(), so pybind11
// accepts only C-contiguous float64 arrays (int32 for richness indices)
// and C reads the caller's memory without copying or modifying it. Every
// output is a new NumPy array owned by Python.
//
// Threads and lazy tables. The conversions open no OpenMP region and call
// no BLAS routine. counts_shell_cluster_cov, limber_cluster_cov and
// moments_cluster_cov read only their arguments and run their own OpenMP
// loops. halo_samples_cluster_cov warms the core readers it needs on the
// calling thread before its OpenMP loops. The catalog readers at the end
// of this file use serial loops; ncl_richness, bcl_richness and
// pcm_1h_richness call cluster_warmup() first.
// ---------------------------------------------------------------------------
using cluster_cov_array = py::array_t<double, py::array::c_style>;

// ---------------------------------------------------------------------------
// Expose count-shell quantities without assigning a mass-selection model.
//
// NumPy supplies selected abundances and their background-density
// derivatives. C converts them to counts per radial distance, using the
// shell volume. Shape and value checks precede C, so a malformed input
// raises a Python exception instead of stopping the Python process.
// Returned arrays own their data and remain valid after subsequent calls.
// No likelihood, nuisance parameter or cosmology state is read or changed.
//
// A shell of thickness dchi subtends the volume area_sr f_K^2 dchi, so
//
//   shell_density  S_i(chi)   = area_sr f_K^2 n_i         (dN_i/dchi),
//   shell_response Phi_i(chi) = area_sr f_K^2 dn_i/d(delta_b).
//
// Arrays: distance[nnode] f_K in one length unit L; density and
// derivative [ncount,nnode] in L^-3; outputs [ncount,nnode] in L^-1,
// without a dchi weight. Validation: shapes, 0 < area_sr <= 4 pi, finite
// positive distances, finite nonnegative densities, finite derivatives
// of either sign. counts_shell_cluster_cov collapses (bin, node pair)
// over the OpenMP team.
// ---------------------------------------------------------------------------
static py::dict covariance_counts_shell(
    const cluster_cov_array& distance, // [nnode], transverse distances
    const cluster_cov_array& density,  // [ncount,nnode], selected n_i
    const cluster_cov_array& derivative, // matching dn_i/d(delta_b)
    const double area_sr               // survey solid angle
  )
{
  if (distance.ndim() != 1
      || distance.size() < 1
      || density.ndim() != 2
      || derivative.ndim() != 2
      || density.shape(0) < 1
      || density.shape(1) != distance.size()
      || derivative.shape(0) != density.shape(0)
      || derivative.shape(1) != density.shape(1)) {
    throw std::invalid_argument(
        "distance must be nonempty [nnode]; density and derivative "
        "must both have shape [ncount,nnode], with ncount > 0");
  }
  if (!std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    throw std::invalid_argument("area_sr must be finite and in (0,4*pi]");
  }

  // A positive distance gives a physical shell volume. Zero abundances
  // are allowed where a bin's selection vanishes; signed responses permit
  // a selection model whose environmental change reduces the abundance.
  for (py::ssize_t node=0; node<distance.size(); node++) {
    if (!std::isfinite(distance.data()[node])
        || distance.data()[node] <= 0.0) {
      throw std::invalid_argument("distance must be finite and positive");
    }
  }
  for (py::ssize_t entry=0; entry<density.size(); entry++) {
    if (!std::isfinite(density.data()[entry])
        || density.data()[entry] < 0.0
        || !std::isfinite(derivative.data()[entry])) {
      throw std::invalid_argument(
          "density must be finite and nonnegative; derivative must "
          "be finite and may have either sign");
    }
  }

  // The pointer vectors describe NumPy's rows without copying values.
  // Separate output arrays ensure that C never overwrites an input.
  // One loop iteration records the four row addresses of one count bin.
  const py::ssize_t ncount = density.shape(0);
  const py::ssize_t nnode = distance.size();
  cluster_cov_array shell({ncount, nnode});
  cluster_cov_array response({ncount, nnode});
  std::vector<const double*> density_rows(ncount);
  std::vector<const double*> derivative_rows(ncount);
  std::vector<double*> shell_rows(ncount);
  std::vector<double*> response_rows(ncount);

  for (py::ssize_t bin=0; bin<ncount; bin++) {
    density_rows[bin] = density.data(bin, 0);
    derivative_rows[bin] = derivative.data(bin, 0);
    shell_rows[bin] = shell.mutable_data(bin, 0);
    response_rows[bin] = response.mutable_data(bin, 0);
  }

  counts_shell_cluster_cov(ncount, nnode, area_sr, distance.data(),
      density_rows.data(), derivative_rows.data(), shell_rows.data(),
      response_rows.data());

  py::dict result;
  result["shell_density"] = shell;
  result["shell_response"] = response;
  return result;
}


// ---------------------------------------------------------------------------
// Project supplied cluster ingredients without importing a survey model.
//
// All arrays share the same radial nodes. The caller obtains normalized
// cluster windows, selected biases and one-halo profiles from its model.
// This interface checks shapes before passing row pointers to C; returned
// cross and auto spectra own their data. The galaxy/shear block is supplied
// separately by the ordinary covariance spectrum builder.
//
// In the Limber approximation, with q_c the normalized cluster window and
// b_c its selected bias, limber_cluster_cov evaluates
//
//   C_cc'(ell) = integral dchi q_c b_c q_c' b_c' P_NL / f_K^2,
//   C_cg(ell)  = integral dchi q_c b_c W_g P_NL / f_K^2,
//   C_cs(ell)  = F_ell integral dchi q_c W_s (b_c P_NL + P_cm^1h) / f_K^2,
//
// at k=(ell+1/2)/f_K, with the core shear factor F_ell for source fields.
// Arrays (one length unit L): ell[nell] >= 2; distance, dchi [nnode] in L;
// base[nbase,nnode] = nlens biased galaxy windows then source lensing
// windows, in L^-1; window[ncluster,nnode] in L^-1 (integral q_c dchi =
// 1); bias[ncluster,nnode] dimensionless; power[nell,nnode] and
// profile[nrichness,nell,nnode] in L^3; richness[ncluster] int32 rows of
// profile. Outputs: cluster_base[nell,ncluster,nbase] and
// cluster_cluster[nell,ncluster,ncluster], dimensionless.
// Validation: ranks and shapes, finite values, ell >= 2, positive distance
// and dchi, 0 <= nlens <= nbase, richness indices in [0,nrichness).
// limber_cluster_cov reads no global state or lazy table and spreads
// (ell, pair group) outputs over the OpenMP team.
// ---------------------------------------------------------------------------
static py::dict covariance_cluster_spectra(
    const cluster_cov_array& ell,       // [nell] multipoles
    const cluster_cov_array& distance,  // [nnode] transverse distances f_K
    const cluster_cov_array& dchi,      // [nnode] radial integration weights
    const cluster_cov_array& base,      // [nbase,nnode] galaxy, then lensing
    const cluster_cov_array& window,    // [ncluster,nnode] normalized q_c
    const cluster_cov_array& bias,      // [ncluster,nnode] selected b_c
    const cluster_cov_array& power,     // [nell,nnode] nonlinear matter P
    const cluster_cov_array& profile,   // [nrichness,nell,nnode] P_cm^1h
    const py::array_t<int, py::array::c_style>& richness, // profile map
    const int nlens                     // leading galaxy fields in base
  )
{
  if (ell.ndim() != 1
      || ell.size() < 1
      || distance.ndim() != 1
      || distance.size() < 1
      || dchi.ndim() != 1
      || dchi.size() != distance.size()
      || base.ndim() != 2
      || window.ndim() != 2
      || bias.ndim() != 2
      || power.ndim() != 2
      || profile.ndim() != 3
      || richness.ndim() != 1) {
    throw std::invalid_argument(
        "ell/distance/dchi/richness must be vectors; base/window/bias/"
        "power matrices; profile a 3D array; ell/distance nonempty");
  }
  const py::ssize_t nell = ell.size();
  const py::ssize_t nnode = distance.size();
  const py::ssize_t nbase = base.shape(0);
  const py::ssize_t ncluster = window.shape(0);
  const py::ssize_t nrichness = profile.shape(0);
  if (nbase < 1
      || ncluster < 1
      || nrichness < 1
      || nlens < 0
      || nlens > nbase
      || base.shape(1) != nnode
      || window.shape(1) != nnode
      || bias.shape(0) != ncluster
      || bias.shape(1) != nnode
      || power.shape(0) != nell
      || power.shape(1) != nnode
      || profile.shape(1) != nell
      || profile.shape(2) != nnode
      || richness.size() != ncluster) {
    throw std::invalid_argument(
        "need base[nbase,nnode], window/bias[ncluster,nnode], "
        "power[nell,nnode], profile[nrichness,nell,nnode], "
        "richness[ncluster], positive field counts and 0<=nlens<=nbase");
  }

  // Reject nonfinite values before any output allocation or C operation.
  // Signed windows and profiles can be supplied; they are never clipped.
  for (const auto* input : {&ell, &distance, &dchi, &base, &window,
                            &bias, &power, &profile}) {
    for (py::ssize_t entry=0; entry<input->size(); entry++) {
      if (!std::isfinite(input->data()[entry])) {
        throw std::invalid_argument("cluster spectrum inputs must be finite");
      }
    }
  }
  for (py::ssize_t index=0; index<nell; index++) {
    if (ell.data()[index] < 2.0) {
      throw std::invalid_argument("cluster spectrum ell must be >= 2");
    }
  }
  for (py::ssize_t node=0; node<nnode; node++) {
    if (distance.data()[node] <= 0.0
        || dchi.data()[node] <= 0.0) {
      throw std::invalid_argument("distance and dchi must be positive");
    }
  }
  for (py::ssize_t field=0; field<ncluster; field++) {
    if (richness.data()[field] < 0
        || richness.data()[field] >= nrichness) {
      throw std::invalid_argument("richness indices must be in [0,nrichness)");
    }
  }

  // Row-pointer vectors describe the supplied contiguous arrays without
  // copying their values. They remain alive until C finishes reading them.
  // The profile cube needs two levels: profile_planes[bin] points at the
  // nell row addresses of that richness bin, stored in profile_rows.
  std::vector<const double*> base_rows(nbase);
  std::vector<const double*> window_rows(ncluster);
  std::vector<const double*> bias_rows(ncluster);
  std::vector<const double*> power_rows(nell);
  std::vector<const double*> profile_rows(nrichness*nell);
  std::vector<const double* const*> profile_planes(nrichness);
  for (py::ssize_t field=0; field<nbase; field++) {
    base_rows[field] = base.data(field, 0);
  }
  for (py::ssize_t field=0; field<ncluster; field++) {
    window_rows[field] = window.data(field, 0);
    bias_rows[field] = bias.data(field, 0);
  }
  for (py::ssize_t index=0; index<nell; index++) {
    power_rows[index] = power.data(index, 0);
  }
  for (py::ssize_t bin=0; bin<nrichness; bin++) {
    profile_planes[bin] = profile_rows.data()+bin*nell;
    for (py::ssize_t index=0; index<nell; index++) {
      profile_rows[bin*nell+index] = profile.data(bin, index, 0);
    }
  }

  // C writes one row per pair: first the ncluster*nbase cluster-major
  // (cluster,base) pairs, then the cluster upper triangle (0,0),(0,1),...
  // The temporary triangular array holds them until the copy below.
  const py::ssize_t ncross = ncluster*nbase;
  const py::ssize_t npair = ncross+ncluster*(ncluster+1)/2;
  cluster_cov_array triangular({npair, nell});
  std::vector<double*> output_rows(npair);
  for (py::ssize_t pair=0; pair<npair; pair++) {
    output_rows[pair] = triangular.mutable_data(pair, 0);
  }
  limber_cluster_cov(nell, ell.data(), nnode, distance.data(), dchi.data(),
      nbase, nlens, base_rows.data(), ncluster, window_rows.data(),
      bias_rows.data(), power_rows.data(), profile_planes.data(),
      richness.data(), output_rows.data());

  // Python receives explicit field axes, not the C triangular row map.
  // Copy both auto-spectrum triangles from the same integrated value.
  // pair walks the C rows in their written order: the first double loop
  // consumes the cross rows, the second the cluster upper triangle.
  cluster_cov_array cross({nell, ncluster, nbase});
  cluster_cov_array auto_spectra({nell, ncluster, ncluster});
  py::ssize_t pair = 0;
  for (py::ssize_t field=0; field<ncluster; field++) {
    for (py::ssize_t other=0; other<nbase; other++) {
      for (py::ssize_t index=0; index<nell; index++) {
        *cross.mutable_data(index, field, other) = output_rows[pair][index];
      }
      pair++;
    }
  }
  for (py::ssize_t field=0; field<ncluster; field++) {
    for (py::ssize_t other=field; other<ncluster; other++) {
      for (py::ssize_t index=0; index<nell; index++) {
        const double value = output_rows[pair][index];
        *auto_spectra.mutable_data(index, field, other) = value;
        *auto_spectra.mutable_data(index, other, field) = value;
      }
      pair++;
    }
  }
  py::dict result;
  result["cluster_base"] = cross;
  result["cluster_cluster"] = auto_spectra;
  return result;
}


// ---------------------------------------------------------------------------
// Keep the selection and mass rule explicit at the Python boundary.
//
// weight already contains the selected abundance and mass quadrature.
// Profiles are shared by every selection at a given state, so C can reuse
// them for single and pair moments without choosing a mass function or
// reading cluster globals. Outputs expose state and selection separately;
// only the temporary C row map combines them into a flat population index.
//
// With dn S_i = weight (dlnM dn/dlnM times one membership probability)
// and p(k) = (M/rho) u(k|M), moments_cluster_cov sums over mass nodes
//
//   density = sum dn S_i,  biased_density = sum dn S_i b,
//   J01(K) = sum dn S_i p(K),  J11(K) = sum dn S_i b p(K),
//   J02(K,Q) = sum dn S_i p(K) p(Q),
//   J03_KKQ = sum dn S_i p(K)^2 p(Q),  J03_KQQ = sum dn S_i p(K) p(Q)^2.
//
// Arrays (one length unit L): weight[state,selection,mass] in L^-3,
// bias[state,mass], profile[state,k,mass] in L^3. Outputs: density and
// biased_density [state,selection] in L^-3; J01, J11 [state,selection,k],
// dimensionless; J02, J03_KKQ, J03_KQQ [state,selection,kpair] in
// L^3, L^6, L^6, with kpair in (0,0),(0,1),...,(1,1),... order of k.
// Validation: ranks, positive counts, shared state and mass axes, finite
// values, weight >= 0. C collapses (state, selection, k or pair) over the
// OpenMP team and keeps each mass sum in increasing node order.
// ---------------------------------------------------------------------------
static py::dict covariance_cluster_moments(
    const cluster_cov_array& weight,  // [state,selection,mass], selected dn
    const cluster_cov_array& bias,    // [state,mass], linear halo bias
    const cluster_cov_array& profile  // [state,k,mass], (M/rho)*u(k|M)
  )
{
  if (weight.ndim() != 3
      || bias.ndim() != 2
      || profile.ndim() != 3) {
    throw std::invalid_argument(
        "need weight[state,selection,mass], bias[state,mass] and "
        "profile[state,k,mass]");
  }
  const py::ssize_t na = weight.shape(0); // independent radial states
  const py::ssize_t nselection = weight.shape(1); // observed categories
  const py::ssize_t nmass = weight.shape(2); // mass quadrature nodes
  const py::ssize_t nk = profile.shape(1); // profiles per radial state

  if (na < 1
      || nselection < 1
      || nmass < 1
      || nk < 1
      || bias.shape(0) != na
      || bias.shape(1) != nmass
      || profile.shape(0) != na
      || profile.shape(2) != nmass) {
    throw std::invalid_argument(
        "all counts must be positive; weight, bias and profile must "
        "share their state and mass dimensions");
  }
  for (const auto* input : {&weight, &bias, &profile}) {
    for (py::ssize_t entry=0; entry<input->size(); entry++) {
      if (!std::isfinite(input->data()[entry])) {
        throw std::invalid_argument("cluster moment inputs must be finite");
      }
    }
  }
  for (py::ssize_t entry=0; entry<weight.size(); entry++) {
    if (weight.data()[entry] < 0.0) {
      throw std::invalid_argument("selected mass weights must be nonnegative");
    }
  }

  // Pointer maps retain the original mass rows without copying numerical
  // inputs. Their vectors own the maps until the synchronous C call ends.
  // One iteration of the state loop builds that state's plane views:
  // weight_planes[state][bin] and profile_planes[state][mode] are mass rows.
  const py::ssize_t nrow = na*nselection; // flattened populations
  const py::ssize_t npair = nk*(nk+1)/2; // triangular k-pair count
  std::vector<const double*> weight_rows(nrow);
  std::vector<const double* const*> weight_planes(na);
  std::vector<const double*> bias_rows(na);
  std::vector<const double*> profile_rows(na*nk);
  std::vector<const double* const*> profile_planes(na);
  for (py::ssize_t state=0; state<na; state++) {
    weight_planes[state] = weight_rows.data()+state*nselection;
    profile_planes[state] = profile_rows.data()+state*nk;
    bias_rows[state] = bias.data(state, 0);
    for (py::ssize_t bin=0; bin<nselection; bin++) {
      weight_rows[state*nselection+bin] = weight.data(state, bin, 0);
    }
    for (py::ssize_t mode=0; mode<nk; mode++) {
      profile_rows[state*nk+mode] = profile.data(state, mode, 0);
    }
  }

  // Named arrays keep the notebook API's physical axes without an
  // Armadillo allocation or a packed fourth numerical axis. C addresses
  // populations by the flat row = state*nselection+selection. In these
  // C-order arrays that row is exactly element (state,selection), so only
  // the row pointers below use the flat index.
  cluster_cov_array density({na, nselection});
  cluster_cov_array biased_density({na, nselection});
  cluster_cov_array j01({na, nselection, nk});
  cluster_cov_array j11({na, nselection, nk});
  cluster_cov_array j02({na, nselection, npair});
  cluster_cov_array j03_kkq({na, nselection, npair});
  cluster_cov_array j03_kqq({na, nselection, npair});
  double* density_rows[2] = {
    density.mutable_data(), biased_density.mutable_data()
  };
  cluster_cov_array* singles[2] = {&j01, &j11};
  cluster_cov_array* pairs[3] = {&j02, &j03_kkq, &j03_kqq};
  std::vector<double*> single_rows(2*nrow);
  double** single_planes[2];
  std::vector<double*> pair_rows(3*nrow);
  double** pair_planes[3];

  // Only addresses are copied. The NumPy values remain in their owned
  // output arrays for the entire synchronous mass integration. Roles
  // follow moments_cluster_cov: single 0,1 = J01, J11; pair 0,1,2 = J02,
  // J03_KKQ, J03_KQQ. Each role's plane holds nrow population rows.
  for (int role=0; role<3; role++) {
    pair_planes[role] = pair_rows.data()+role*nrow;
    if (role < 2) {
      single_planes[role] = single_rows.data()+role*nrow;
    }
    for (py::ssize_t row=0; row<nrow; row++) {
      pair_rows[role*nrow+row] = pairs[role]->mutable_data()+row*npair;
      if (role < 2) {
        single_rows[role*nrow+row] = singles[role]->mutable_data()+row*nk;
      }
    }
  }
  moments_cluster_cov(na, nselection, nk, nmass, weight_planes.data(),
      bias_rows.data(), profile_planes.data(), density_rows,
      single_planes, pair_planes);

  py::dict result;
  result["density"] = density;
  result["biased_density"] = biased_density;
  result["J01"] = j01;
  result["J11"] = j11;
  result["J02"] = j02;
  result["J03_KKQ"] = j03_kkq;
  result["J03_KQQ"] = j03_kqq;
  return result;
}


// ---------------------------------------------------------------------------
// Read physical halo inputs on a caller-owned covariance mass rule.
//
// This is separate from the supplied-weight moment integrator: notebooks
// can inspect the abundance/profile samples or pass an independent model
// to that integrator. All state and shape guards precede lazy core reads,
// allocation and parallel work. The output arrays own their storage.
//
// At each scale factor and mass node, halo_samples_cluster_cov returns
//
//   weight[state,lambda,mass]  = dlnM (dn/dlnM) S_lambda(M,z),
//   bias[state,mass]           = linear halo bias b_h(M,a),
//   profile[state,k,mass]      = (M/rho_m) u_NFW(k|M,a),
//
// with the Tinker mass function, its initialized amplitude convention and
// the lognormal richness selection S_lambda of each richness bin. Units:
// a dimensionless; k in (c/H0)^-1; lnm = ln(M/[Msun/h]); weight in
// (c/H0)^-3; bias dimensionless; profile in (c/H0)^3. No redshift-bin
// selection or catalog normalization is included.
// Validation: shapes and finite values; massless neutrinos, Omega_m > 0,
// Tinker 2010 mass function and NFW profiles; at least one richness bin,
// a lognormal mass-observable relation with positive scatter and pivots,
// selection_model 0 and hmf_alpha_mode 0 or 1; limits.a_min <= a < 1;
// k >= 0; lnm inside [ln halo_m[RANGE_MIN], ln halo_m[RANGE_MAX]], the
// halo.c mass range; dlnm > 0. halo_samples_cluster_cov warms the core
// readers it uses on the calling thread, then distributes (state, mass)
// and (state, k, mass) work over the OpenMP team.
// ---------------------------------------------------------------------------
static py::dict covariance_cluster_halo_samples(
    const cluster_cov_array& a,       // scale factors [state]
    const cluster_cov_array& k,       // core wavenumbers [state,k]
    const cluster_cov_array& lnm,     // log masses [mass]
    const cluster_cov_array& dlnm     // positive quadrature measures [mass]
  )
{
  if (a.ndim() != 1
      || a.size() < 1
      || k.ndim() != 2
      || k.shape(0) != a.size()
      || k.shape(1) < 1
      || lnm.ndim() != 1
      || lnm.size() < 1
      || dlnm.ndim() != 1
      || dlnm.size() != lnm.size()) {
    throw std::invalid_argument(
        "need nonempty a[state], k[state,k], lnm[mass], dlnm[mass]");
  }
  if (cosmology.Omega_nu != 0.0
      || cosmology.Omega_m <= 0.0
      || like.halo_model[0] != HMF_TINKER_2010
      || like.halo_model[3] != HALO_PROFILE_NFW
      || cluster.richness_nbin < 1
      || cluster.mor_model != CLUSTER_MOR_LOGNORMAL
      || cluster.selection_model != CLUSTER_SELECTION_NONE
      || cluster.mor[2] <= 0.0
      || cluster.mor_pivot_mass <= 0.0
      || cluster.mor_pivot_1pz <= 0.0) {
    throw std::invalid_argument(
        "initialize massless cosmology, NFW halos, richness bins and a "
        "lognormal MOR with positive scatter and selection_model=0");
  }
  if (cluster.hmf_alpha_mode != CLUSTER_HMF_ALPHA_FIXED
      && cluster.hmf_alpha_mode != CLUSTER_HMF_ALPHA_NORMALIZED) {
    throw std::invalid_argument("cluster hmf_alpha_mode must be 0 or 1");
  }
  for (const auto* input : {&a, &k, &lnm, &dlnm}) {
    for (py::ssize_t entry=0; entry<input->size(); entry++) {
      if (!std::isfinite(input->data()[entry])) {
        throw std::invalid_argument("halo sample inputs must be finite");
      }
    }
  }
  for (py::ssize_t state=0; state<a.size(); state++) {
    if (a.data()[state] < limits.a_min
        || a.data()[state] >= 1.0) {
      throw std::invalid_argument("a must lie in [limits.a_min,1)");
    }
  }
  for (py::ssize_t entry=0; entry<k.size(); entry++) {
    if (k.data()[entry] < 0.0) {
      throw std::invalid_argument("k must be nonnegative in core units");
    }
  }
  for (py::ssize_t node=0; node<lnm.size(); node++) {
    if (lnm.data()[node] < std::log(limits.halo_m[RANGE_MIN])
        || lnm.data()[node] > std::log(limits.halo_m[RANGE_MAX])
        || dlnm.data()[node] <= 0.0) {
      throw std::invalid_argument(
          "lnm must lie in the core sigma mass range; dlnm must be > 0");
    }
  }

  const py::ssize_t na = a.size(); // independent scale-factor states
  const py::ssize_t nk = k.shape(1); // wavenumbers per state
  const py::ssize_t nmass = lnm.size(); // mass quadrature nodes
  const py::ssize_t nrichness = cluster.richness_nbin; // selection bins
  cluster_cov_array weight({na, nrichness, nmass}); // dn S, (c/H0)^-3
  cluster_cov_array bias({na, nmass}); // dimensionless linear halo bias
  cluster_cov_array profile({na, nk, nmass}); // (M/rho_m)*u, (c/H0)^3
  std::vector<const double*> k_rows(na); // input row views
  std::vector<double*> weight_rows(na*nrichness); // output mass rows
  std::vector<double**> weight_planes(na); // state views of richness rows
  std::vector<double*> bias_rows(na); // output bias mass rows
  std::vector<double*> profile_rows(na*nk); // output profile mass rows
  std::vector<double**> profile_planes(na); // state views of k rows

  // Numerical values stay in NumPy-owned contiguous arrays. Only these
  // small pointer maps translate their axes into C's row-pointer format.
  // One iteration fills every row address of one scale-factor state.
  for (py::ssize_t state=0; state<na; state++) {
    k_rows[state] = k.data(state, 0);
    weight_planes[state] = weight_rows.data()+state*nrichness;
    bias_rows[state] = bias.mutable_data(state, 0);
    profile_planes[state] = profile_rows.data()+state*nk;
    for (py::ssize_t bin=0; bin<nrichness; bin++) {
      weight_rows[state*nrichness+bin] = weight.mutable_data(state, bin, 0);
    }
    for (py::ssize_t mode=0; mode<nk; mode++) {
      profile_rows[state*nk+mode] = profile.mutable_data(state, mode, 0);
    }
  }
  halo_samples_cluster_cov(na, a.data(), nk, k_rows.data(), nmass,
      lnm.data(), dlnm.data(), weight_planes.data(), bias_rows.data(),
      profile_planes.data());

  py::dict result;
  result["weight"] = weight;
  result["bias"] = bias;
  result["profile"] = profile;
  return result;
}


// ---------------------------------------------------------------------------
// Sample existing catalog tables directly into production-owned arrays.
//
// These loops only call the public C readers; the halo integrals and
// interpolation remain in the original C implementations. Every loop here
// is serial. ncl_richness, bcl_richness and pcm_1h_richness first call
// cluster_warmup(), which builds the lazy cluster tables on this thread,
// as the data-vector path does before its threaded loops. phi_cluster
// calls no warm-up: its first read builds the selection table on this
// same thread. A table that cluster_warmup skips (the one-halo table when
// cluster lensing is off) is likewise built by its first serial read.
// The readers return zero outside their tables: phi outside a bin's
// redshift support, abundance, bias and P1h outside the cluster a grid.
// Inputs are borrowed C-contiguous float64 vectors; outputs are owned.
// ---------------------------------------------------------------------------
static void bind_production_catalog_cov(py::module_& module)
{
  // <phi_i|z>: probability that a cluster at true redshift z is assigned
  // to cluster redshift bin i. z[node] must be finite and >= 0; the
  // output [node,zbin] is dimensionless. One iteration of the outer read
  // loop evaluates every redshift bin at one z node.
  module.def("phi_cluster", [](const cluster_cov_array& z) {
    if (z.ndim() != 1
        || z.size() < 1
        || cluster.zdist_nbin < 1) {
      throw std::invalid_argument("initialize cluster bins and supply z[node]");
    }
    cluster_cov_array result({z.size(), py::ssize_t(cluster.zdist_nbin)});
    for (py::ssize_t node=0; node<z.size(); node++) {
      if (!std::isfinite(z.data()[node])
          || z.data()[node] < 0.0) {
        throw std::invalid_argument("cluster redshifts must be finite and >=0");
      }
    }
    for (py::ssize_t node=0; node<z.size(); node++) {
      for (int bin=0; bin<cluster.zdist_nbin; bin++) {
        *result.mutable_data(node, bin) = phi_cluster(z.data()[node], bin);
      }
    }
    return result;
  }, "Redshift-selection probability [node,zbin] from core C tables.",
      py::arg("z").noconvert());

  // Abundance and bias share the same (a,richness) table axes. The flag
  // selects which public C reader fills the output, without mixing them.
  // The loop registers two Python functions; each lambda keeps its own
  // copy of the flag. ncl_richness returns the selected comoving abundance
  // n_lambda(a) in (c/H0)^-3, bcl_richness the dimensionless selected bias
  // b_lambda(a), both [node,richness] for a[node] inside (0,1).
  for (const bool bias : {false, true}) {
    const char* name = bias ? "bcl_richness" : "ncl_richness";
    module.def(name, [bias](const cluster_cov_array& a) {
      if (a.ndim() != 1
          || a.size() < 1
          || cluster.richness_nbin < 1) {
        throw std::invalid_argument("initialize richness bins and supply a[node]");
      }
      for (py::ssize_t node=0; node<a.size(); node++) {
        if (!std::isfinite(a.data()[node])
            || a.data()[node] <= 0.0
            || a.data()[node] >= 1.0) {
          throw std::invalid_argument("cluster scale factors must lie in (0,1)");
        }
      }
      cluster_cov_array result({a.size(), py::ssize_t(cluster.richness_nbin)});
      cluster_warmup();
      for (py::ssize_t node=0; node<a.size(); node++) {
        for (int bin=0; bin<cluster.richness_nbin; bin++) {
          *result.mutable_data(node, bin) = bias
              ? bcl_richness(a.data()[node], bin)
              : ncl_richness(a.data()[node], bin);
        }
      }
      return result;
    }, "Cached selected abundance or bias [a,richness] from core C readers.",
        py::arg("a").noconvert());
  }

  // P_cm^1h(k,a): the one-halo cluster-matter power of each richness bin,
  // the selected halos' own mass profile averaged within the bin, in
  // (c/H0)^3. k[nk] > 0 in (c/H0)^-1 and a[na] inside (0,1); the output
  // is [k,a,richness]. The three read loops visit every combination once.
  module.def("pcm_1h_richness", [](
      const cluster_cov_array& k, const cluster_cov_array& a) {
    if (k.ndim() != 1
        || a.ndim() != 1
        || k.size() < 1
        || a.size() < 1
        || cluster.richness_nbin < 1) {
      throw std::invalid_argument("initialize richness bins and supply k,a vectors");
    }
    for (py::ssize_t mode=0; mode<k.size(); mode++) {
      if (!std::isfinite(k.data()[mode])
          || k.data()[mode] <= 0.0) {
        throw std::invalid_argument("cluster profile wavenumbers must be >0");
      }
    }
    for (py::ssize_t node=0; node<a.size(); node++) {
      if (!std::isfinite(a.data()[node])
          || a.data()[node] <= 0.0
          || a.data()[node] >= 1.0) {
        throw std::invalid_argument("cluster scale factors must lie in (0,1)");
      }
    }
    cluster_cov_array result({k.size(), a.size(),
                              py::ssize_t(cluster.richness_nbin)});
    cluster_warmup();
    for (py::ssize_t mode=0; mode<k.size(); mode++) {
      for (py::ssize_t node=0; node<a.size(); node++) {
        for (int bin=0; bin<cluster.richness_nbin; bin++) {
          *result.mutable_data(mode, node, bin) =
              pcm_1h_richness(k.data()[mode], a.data()[node], bin);
        }
      }
    }
    return result;
  }, "Selected one-halo power [k,a,richness], in (c/H0)^3.",
      py::arg("k").noconvert(), py::arg("a").noconvert());
}

// Add the cluster bindings to the production submodule parent.covariance.
// That submodule must already exist: bind_covariance_production, called
// through bind_covariance, creates it, so a project interface calls
// bind_covariance before bind_covariance_cluster. Without it, the
// attribute lookup below raises AttributeError during module import.
void bind_production_cluster_cov(py::module_& parent)
{
  py::module_ module = parent.attr("covariance").cast<py::module_>();
  bind_production_catalog_cov(module);
  module.def("covariance_cluster_halo_samples",
      &covariance_cluster_halo_samples,
      R"doc(Sample the initialized halo/richness model on covariance nodes.

Arguments (contiguous float64):
    a: [state], limits.a_min <= a < 1.
    k: [state,k], nonnegative wavenumbers in inverse c/H0.
    lnm: [mass], ln(M/[Msun/h]) inside the core sigma mass range.
    dlnm: [mass], positive quadrature weights for dlnM.
Returns:
    Owned weight[state,richness,mass], selected dn in (c/H0)^-3;
    bias[state,mass], dimensionless linear halo bias;
    profile[state,k,mass], (M/rho_m)*u_NFW in (c/H0)^3.
    Pass these arrays to covariance_cluster_moments for mass integration.
Scope:
    Requires initialized massless cosmology, NFW, lognormal richness
    selection and selection_model=0. The HMF amplitude follows the
    initialized cluster hmf_alpha_mode. No redshift-bin selection, low-mass
    completion, catalog normalization or environmental response is added.
    Core reader tables are warmed serially; model settings are not changed.
)doc",
      py::arg("a").noconvert(), py::arg("k").noconvert(),
      py::arg("lnm").noconvert(), py::arg("dlnm").noconvert());

  module.def("covariance_cluster_moments", &covariance_cluster_moments,
      R"doc(Integrate selected halo moments with supplied mass quadrature.

Arguments (float64; one consistent length unit L):
    weight: [state,selection,mass], dlnM*(dn/dlnM)*S_i, in L^-3.
        S_i is the membership probability, included ONCE, even for a
        same-halo correlation with multiple cluster legs. Weight >= 0.
    bias: [state,mass], finite linear halo bias, dimensionless.
    profile: [state,k,mass], finite (M/rho)*u(k|M), in L^3.
Returns:
    density[state,selection]: n_i; biased_density[state,selection]:
        integral dn S_i b. Both have units L^-3.
    J01[state,selection,k] and J11[state,selection,k], dimensionless.
    J02[state,selection,kpair], J03_KKQ[state,selection,kpair] and
        J03_KQQ[state,selection,kpair], with units L^3,L^6,L^6.
        Each named array owns its values. Pair order is the k-grid's
        upper triangle: (0,0),(0,1),...,(1,1),... .
    J_beta_mu = integral dn S_i b_beta product(profile), b_0=1, b_1=b.
Scope:
    These are unnormalized ingredients, not a covariance. No mass function,
    low-mass completion, survey mask, shot noise or catalog normalization is
    chosen. The fixed-selection interpretation requires S_i not to respond
    to the background overdensity; an environmental derivative is separate.
    Distinct exclusive observed categories share no same-halo term, even
    when their true-mass distributions overlap. Their different-halo and
    super-sample correlations must still be computed.
)doc",
      py::arg("weight").noconvert(), py::arg("bias").noconvert(),
      py::arg("profile").noconvert());

  module.def("covariance_cluster_spectra", &covariance_cluster_spectra,
      R"doc(Project every cluster-galaxy, cluster-source and cluster pair.

Arguments (contiguous float64, except richness as int32):
    ell: [nell], multipoles >= 2.
    distance, dchi: [nnode], positive transverse distances/radial weights.
    base: [nbase,nnode], biased galaxy windows followed by lensing windows.
    window, bias: [ncluster,nnode], normalized q_c and selected bias b_c.
    power: [nell,nnode], nonlinear matter P at k=(ell+1/2)/f_K.
    profile: [nrichness,nell,nnode], selected P_cm^1h on the same grid.
    richness: [ncluster], index of each category's selected profile.
    nlens: number of leading galaxy fields in base, from 0 to nbase.
Units:
    One length unit L for distance/dchi; windows L^-1; power/profile L^3.
    Bias and the returned angular spectra are dimensionless.
Returns:
    Owned arrays cluster_base[nell,ncluster,nbase] and
    cluster_cluster[nell,ncluster,ncluster], the latter exactly symmetric.
Model:
    cc and cg use biased nonlinear matter power. Only cluster lensing
    adds the selected halo's own mass profile. Source spectra include the
    core harmonic spin factor. No noise, IA, magnification, RSD, non-Limber
    correction or estimator transform is included. No global state changes.
    Validate field positivity with catalog noise before Gaussian assembly.
)doc",
      py::arg("ell").noconvert(), py::arg("distance").noconvert(),
      py::arg("dchi").noconvert(), py::arg("base").noconvert(),
      py::arg("window").noconvert(), py::arg("bias").noconvert(),
      py::arg("power").noconvert(), py::arg("profile").noconvert(),
      py::arg("richness").noconvert(), py::arg("nlens"));

  module.def("covariance_counts_shell", &covariance_counts_shell,
      R"doc(Convert selected abundances into count densities and responses.

Arguments:
    distance: float64 [nnode], positive f_K in one chosen length unit L.
    density: float64 [ncount,nnode], selected abundance in L^-3,
        including richness probability, completeness and redshift selection.
    derivative: same shape, d(density)/d(delta_b), in L^-3. For a fixed
        selection this is the selected, bias-weighted halo abundance.
    area_sr: angular area, in steradians, inside (0,4*pi].
Returns:
    Dict with owned shell_density and shell_response arrays, both
    [ncount,nnode] in L^-1. Neither contains radial quadrature weights.
    Integrating shell_density over dchi gives the expected counts.
    Use shell_response with the same background kernel and two-point
    responses to assemble count-count and count-two-point SSC.
Scope:
    No mass-selection model, Poisson noise, non-SSC cross term or radial
    integration is added. No global interface state is changed.
)doc",
      py::arg("distance").noconvert(),
      py::arg("density").noconvert(),
      py::arg("derivative").noconvert(),
      py::arg("area_sr"));
}

} // namespace cosmolike_interface
