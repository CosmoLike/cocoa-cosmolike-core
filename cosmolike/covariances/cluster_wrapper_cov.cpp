#include <cmath>
#include <stdexcept>

#include <carma.h>
#include <armadillo>
#include "cluster_wrapper_cov.hpp"
#include "counts_cluster_cov.h"
#include "spectra_cluster_cov.h"
#include "moments_cluster_cov.h"
#include "halo_cluster_cov.h"
#include "cosmolike/halo.h"
#include "cosmolike/basics.h"
#include "cosmolike/structs.h"
#include "cosmolike/structs_cluster.h"

namespace py = pybind11;

namespace cosmolike_interface {


// ---------------------------------------------------------------------------
// Expose count-shell quantities without assigning a mass-selection model.
//
// NumPy supplies selected abundances and their background-density
// derivatives. C converts them to counts per radial distance, using the
// shell volume. Shape and value checks precede C, so a malformed notebook
// input raises a Python exception instead of stopping the Python process.
// Returned arrays own their data and remain valid after subsequent calls.
// No likelihood, nuisance parameter or cosmology state is read or changed.
// ---------------------------------------------------------------------------
py::dict covariance_counts_shell_cpp(
    const arma::Col<double>& distance, // [nnode], transverse distances
    const arma::Mat<double>& density,  // [ncount,nnode], selected n_i
    const arma::Mat<double>& derivative, // matching dn_i/d(delta_b)
    const double area_sr               // survey solid angle
  )
{
  if (distance.n_elem < 1
      || density.n_rows < 1
      || density.n_cols != distance.n_elem
      || derivative.n_rows != density.n_rows
      || derivative.n_cols != density.n_cols) {
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
  for (arma::uword node=0; node<distance.n_elem; node++) {
    if (!std::isfinite(distance(node))
        || distance(node) <= 0.0) {
      throw std::invalid_argument("distance must be finite and positive");
    }
  }
  for (arma::uword entry=0; entry<density.n_elem; entry++) {
    if (!std::isfinite(density(entry))
        || density(entry) < 0.0
        || !std::isfinite(derivative(entry))) {
      throw std::invalid_argument(
          "density must be finite and nonnegative; derivative must "
          "be finite and may have either sign");
    }
  }

  const arma::uword ncount = density.n_rows;
  const arma::uword nnode = distance.n_elem;
  arma::Mat<double> shell(ncount, nnode);
  arma::Mat<double> response(ncount, nnode);
  double*** work = (double***) malloc3d(4, ncount, nnode);

  // Abundance and its response share the same catalog and radial axes.
  // C consumes contiguous shell rows; the notebook retains matrix axes.
  for (arma::uword bin=0; bin<ncount; bin++) {
    for (arma::uword node=0; node<nnode; node++) {
      work[0][bin][node] = density(bin, node);
      work[1][bin][node] = derivative(bin, node);
    }
  }
  counts_shell_cluster_cov(ncount, nnode, area_sr, distance.memptr(),
      work[0], work[1], work[2], work[3]);

  for (arma::uword bin=0; bin<ncount; bin++) {
    for (arma::uword node=0; node<nnode; node++) {
      shell(bin, node) = work[2][bin][node];
      response(bin, node) = work[3][bin][node];
    }
  }
  free(work);
  py::dict result;
  result["shell_density"] = carma::mat_to_arr(shell);
  result["shell_response"] = carma::mat_to_arr(response);
  return result;
}


// ---------------------------------------------------------------------------
// Project supplied cluster ingredients without importing a survey model.
//
// All arrays share the same radial nodes. The caller obtains normalized
// cluster windows, selected biases and one-halo profiles from its model.
// This wrapper checks shapes before copying the C workspace; returned
// cross and auto spectra own their data. The galaxy/shear block is supplied
// separately by the ordinary covariance spectrum builder.
// ---------------------------------------------------------------------------
py::dict covariance_cluster_spectra_cpp(
    const arma::Col<double>& ell,       // multipole samples
    const arma::Col<double>& distance,  // common transverse distances
    const arma::Col<double>& dchi,      // radial integration weights
    const arma::Mat<double>& base,      // galaxy and lensing windows
    const arma::Mat<double>& window,    // normalized cluster windows
    const arma::Mat<double>& bias,      // selected cluster bias
    const arma::Mat<double>& power,     // nonlinear matter power
    const arma::Cube<double>& profile,   // selected one-halo spectra
    const arma::Col<int>& richness, // profile map
    const int nlens                     // leading galaxy fields in base
  )
{
  if (ell.n_elem < 1
      || distance.n_elem < 1
      || dchi.n_elem != distance.n_elem) {
    throw std::invalid_argument(
        "ell/distance/dchi/richness must be vectors; base/window/bias/"
        "power matrices; profile a 3D array; ell/distance nonempty");
  }
  const arma::uword nell = ell.n_elem;
  const arma::uword nnode = distance.n_elem;
  const arma::uword nbase = base.n_rows;
  const arma::uword ncluster = window.n_rows;
  const arma::uword nrichness = profile.n_rows;
  if (nbase < 1
      || ncluster < 1
      || nrichness < 1
      || nlens < 0
      || nlens > nbase
      || base.n_cols != nnode
      || window.n_cols != nnode
      || bias.n_rows != ncluster
      || bias.n_cols != nnode
      || power.n_rows != nell
      || power.n_cols != nnode
      || profile.n_cols != nell
      || profile.n_slices != nnode
      || richness.n_elem != ncluster) {
    throw std::invalid_argument(
        "need base[nbase,nnode], window/bias[ncluster,nnode], "
        "power[nell,nnode], profile[nrichness,nell,nnode], "
        "richness[ncluster], positive field counts and 0<=nlens<=nbase");
  }

  // Reject nonfinite values before any output allocation or C operation.
  // Signed windows and profiles can be supplied; they are never clipped.
  if (!ell.is_finite()
      || !distance.is_finite()
      || !dchi.is_finite()
      || !base.is_finite()
      || !window.is_finite()
      || !bias.is_finite()
      || !power.is_finite()
      || !profile.is_finite()) {
    throw std::invalid_argument("covariance inputs must be finite");
  }
  for (arma::uword index=0; index<nell; index++) {
    if (ell(index) < 2.0) {
      throw std::invalid_argument("cluster spectrum ell must be >= 2");
    }
  }
  for (arma::uword node=0; node<nnode; node++) {
    if (distance(node) <= 0.0
        || dchi(node) <= 0.0) {
      throw std::invalid_argument("distance and dchi must be positive");
    }
  }
  for (arma::uword field=0; field<ncluster; field++) {
    if (richness(field) < 0
        || richness(field) >= nrichness) {
      throw std::invalid_argument("richness indices must be in [0,nrichness)");
    }
  }

  const arma::uword ncross = ncluster*nbase;
  const arma::uword npair = ncross+ncluster*(ncluster+1)/2;
  arma::Cube<double> cross(nell, ncluster, nbase);
  arma::Cube<double> auto_spectra(nell, ncluster, ncluster);
  double** base_c = (double**) malloc2d(nbase, nnode);
  double*** cluster_c = (double***) malloc3d(2, ncluster, nnode);
  double** power_c = (double**) malloc2d(nell, nnode);
  double*** profile_c = (double***) malloc3d(nrichness, nell, nnode);
  double** triangular = (double**) malloc2d(npair, nell);

  // Match C's radial rows by copying physical indices. Each catalog has
  // its own window/bias; a richness category selects its one-halo profile.
  for (arma::uword node=0; node<nnode; node++) {
    for (arma::uword field=0; field<nbase; field++) {
      base_c[field][node] = base(field, node);
    }
    for (arma::uword field=0; field<ncluster; field++) {
      cluster_c[0][field][node] = window(field, node);
      cluster_c[1][field][node] = bias(field, node);
    }
    for (arma::uword index=0; index<nell; index++) {
      power_c[index][node] = power(index, node);
      for (arma::uword bin=0; bin<nrichness; bin++) {
        profile_c[bin][index][node] = profile(bin, index, node);
      }
    }
  }
  limber_cluster_cov(nell, ell.memptr(), nnode, distance.memptr(),
      dchi.memptr(), nbase, nlens, base_c, ncluster, cluster_c[0],
      cluster_c[1], power_c, profile_c, richness.memptr(), triangular);

  // Expose explicit field axes instead of C's triangular pair index.
  // Both auto-spectrum triangles receive exactly the same value.
  arma::uword pair = 0;
  for (arma::uword field=0; field<ncluster; field++) {
    for (arma::uword other=0; other<nbase; other++) {
      for (arma::uword index=0; index<nell; index++) {
        cross(index, field, other) = triangular[pair][index];
      }
      pair++;
    }
  }
  for (arma::uword field=0; field<ncluster; field++) {
    for (arma::uword other=field; other<ncluster; other++) {
      for (arma::uword index=0; index<nell; index++) {
        const double value = triangular[pair][index];
        auto_spectra(index, field, other) = value;
        auto_spectra(index, other, field) = value;
      }
      pair++;
    }
  }
  free(base_c);
  free(cluster_c);
  free(power_c);
  free(profile_c);
  free(triangular);
  py::dict result;
  result["cluster_base"] = carma::cube_to_arr(cross);
  result["cluster_cluster"] = carma::cube_to_arr(auto_spectra);
  return result;
}


// ---------------------------------------------------------------------------
// Keep the selection and mass rule explicit at the notebook boundary.
//
// weight already contains the selected abundance and mass quadrature.
// Profiles are shared by every selection at a given state, so C can reuse
// them for single and pair moments without choosing a mass function or
// reading cluster globals. Outputs expose state and selection separately;
// only the temporary C row map combines them into a flat population index.
// ---------------------------------------------------------------------------
py::dict covariance_cluster_moments_cpp(
    const arma::Cube<double>& weight,  // [state,selection,mass], selected dn
    const arma::Mat<double>& bias,    // [state,mass], linear halo bias
    const arma::Cube<double>& profile  // [state,k,mass], (M/rho)*u(k|M)
  )
{
  const arma::uword na = weight.n_rows; // independent radial states
  const arma::uword nselection = weight.n_cols; // observed categories
  const arma::uword nmass = weight.n_slices; // mass quadrature nodes
  const arma::uword nk = profile.n_cols; // profiles per radial state

  if (na < 1
      || nselection < 1
      || nmass < 1
      || nk < 1
      || bias.n_rows != na
      || bias.n_cols != nmass
      || profile.n_rows != na
      || profile.n_slices != nmass) {
    throw std::invalid_argument(
        "all counts must be positive; weight, bias and profile must "
        "share their state and mass dimensions");
  }
  if (!weight.is_finite()
      || !bias.is_finite()
      || !profile.is_finite()) {
    throw std::invalid_argument("covariance inputs must be finite");
  }
  for (arma::uword entry=0; entry<weight.n_elem; entry++) {
    if (weight(entry) < 0.0) {
      throw std::invalid_argument("selected mass weights must be nonnegative");
    }
  }

  const arma::uword nrow = na*nselection;
  const arma::uword npair = nk*(nk+1)/2;
  arma::Mat<double> density(na, nselection);
  arma::Mat<double> biased_density(na, nselection);
  arma::Cube<double> j01(na, nselection, nk);
  arma::Cube<double> j11(na, nselection, nk);
  arma::Cube<double> j02(na, nselection, npair);
  arma::Cube<double> j03_kkq(na, nselection, npair);
  arma::Cube<double> j03_kqq(na, nselection, npair);
  double*** weight_c = (double***) malloc3d(na, nselection, nmass);
  double** bias_c = (double**) malloc2d(na, nmass);
  double*** profile_c = (double***) malloc3d(na, nk, nmass);
  double** density_c = (double**) malloc2d(2, nrow);
  double*** single_c = (double***) malloc3d(2, nrow, nk);
  double*** pair_c = (double***) malloc3d(3, nrow, npair);

  // Preserve state, selection and mass as distinct physical coordinates.
  // The membership probability is already in weight and enters once.
  for (arma::uword state=0; state<na; state++) {
    for (arma::uword mass=0; mass<nmass; mass++) {
      bias_c[state][mass] = bias(state, mass);
      for (arma::uword bin=0; bin<nselection; bin++) {
        weight_c[state][bin][mass] = weight(state, bin, mass);
      }
      for (arma::uword mode=0; mode<nk; mode++) {
        profile_c[state][mode][mass] = profile(state, mode, mass);
      }
    }
  }
  moments_cluster_cov(na, nselection, nk, nmass, weight_c, bias_c,
      profile_c, density_c, single_c, pair_c);

  // C combines state and selection into a population row. Unpack that
  // index, and name the physical moments instead of adding a fourth axis.
  // The two J03 quantities repeat K or Q respectively in their profiles.
  for (arma::uword state=0; state<na; state++) {
    for (arma::uword bin=0; bin<nselection; bin++) {
      const arma::uword row = state*nselection+bin;
      density(state, bin) = density_c[0][row];
      biased_density(state, bin) = density_c[1][row];
      for (arma::uword mode=0; mode<nk; mode++) {
        j01(state, bin, mode) = single_c[0][row][mode];
        j11(state, bin, mode) = single_c[1][row][mode];
      }
      for (arma::uword pair=0; pair<npair; pair++) {
        j02(state, bin, pair) = pair_c[0][row][pair];
        j03_kkq(state, bin, pair) = pair_c[1][row][pair];
        j03_kqq(state, bin, pair) = pair_c[2][row][pair];
      }
    }
  }
  free(weight_c);
  free(bias_c);
  free(profile_c);
  free(density_c);
  free(single_c);
  free(pair_c);
  py::dict result;
  result["density"] = carma::mat_to_arr(density);
  result["biased_density"] = carma::mat_to_arr(biased_density);
  result["J01"] = carma::cube_to_arr(j01);
  result["J11"] = carma::cube_to_arr(j11);
  result["J02"] = carma::cube_to_arr(j02);
  result["J03_KKQ"] = carma::cube_to_arr(j03_kkq);
  result["J03_KQQ"] = carma::cube_to_arr(j03_kqq);
  return result;
}


// ---------------------------------------------------------------------------
// Read physical halo inputs on a caller-owned covariance mass rule.
//
// This is separate from the supplied-weight moment integrator: notebooks
// can inspect the abundance/profile samples or pass an independent model
// to that integrator. All state and shape guards precede lazy core reads,
// allocation and parallel work. The output arrays own their storage.
// ---------------------------------------------------------------------------
py::dict covariance_cluster_halo_samples_cpp(
    const arma::Col<double>& a,       // scale factors [state]
    const arma::Mat<double>& k,       // core wavenumbers [state,k]
    const arma::Col<double>& lnm,     // log masses [mass]
    const arma::Col<double>& dlnm     // positive quadrature measures [mass]
  )
{
  if (a.n_elem < 1
      || k.n_rows != a.n_elem
      || k.n_cols < 1
      || lnm.n_elem < 1
      || dlnm.n_elem != lnm.n_elem) {
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
  if (!a.is_finite()
      || !k.is_finite()
      || !lnm.is_finite()
      || !dlnm.is_finite()) {
    throw std::invalid_argument("covariance inputs must be finite");
  }
  for (arma::uword state=0; state<a.n_elem; state++) {
    if (a(state) < limits.a_min
        || a(state) >= 1.0) {
      throw std::invalid_argument("a must lie in [limits.a_min,1)");
    }
  }
  for (arma::uword entry=0; entry<k.n_elem; entry++) {
    if (k(entry) < 0.0) {
      throw std::invalid_argument("k must be nonnegative in core units");
    }
  }
  for (arma::uword node=0; node<lnm.n_elem; node++) {
    if (lnm(node) < std::log(limits.halo_m[RANGE_MIN])
        || lnm(node) > std::log(limits.halo_m[RANGE_MAX])
        || dlnm(node) <= 0.0) {
      throw std::invalid_argument(
          "lnm must lie in the core sigma mass range; dlnm must be > 0");
    }
  }

  const arma::uword na = a.n_elem; // independent scale-factor states
  const arma::uword nk = k.n_cols; // wavenumbers per state
  const arma::uword nmass = lnm.n_elem; // mass quadrature nodes
  const arma::uword nrichness = cluster.richness_nbin; // selection bins
  arma::Cube<double> weight(na, nrichness, nmass);
  arma::Mat<double> bias(na, nmass);
  arma::Cube<double> profile(na, nk, nmass);
  double** k_c = (double**) malloc2d(na, nk);
  double*** weight_c = (double***) malloc3d(na, nrichness, nmass);
  double** bias_c = (double**) malloc2d(na, nmass);
  double*** profile_c = (double***) malloc3d(na, nk, nmass);

  // C samples all requested states in one batch. Its contiguous mass
  // rows are copied into Armadillo without changing any physical axes.
  for (arma::uword state=0; state<na; state++) {
    for (arma::uword mode=0; mode<nk; mode++) {
      k_c[state][mode] = k(state, mode);
    }
  }
  halo_samples_cluster_cov(na, a.memptr(), nk, k_c, nmass,
      lnm.memptr(), dlnm.memptr(), weight_c, bias_c, profile_c);

  for (arma::uword state=0; state<na; state++) {
    for (arma::uword mass=0; mass<nmass; mass++) {
      bias(state, mass) = bias_c[state][mass];
      for (arma::uword bin=0; bin<nrichness; bin++) {
        weight(state, bin, mass) = weight_c[state][bin][mass];
      }
      for (arma::uword mode=0; mode<nk; mode++) {
        profile(state, mode, mass) = profile_c[state][mode][mass];
      }
    }
  }
  free(k_c);
  free(weight_c);
  free(bias_c);
  free(profile_c);
  py::dict result;
  result["weight"] = carma::cube_to_arr(weight);
  result["bias"] = carma::mat_to_arr(bias);
  result["profile"] = carma::cube_to_arr(profile);
  return result;
}


} // namespace cosmolike_interface
