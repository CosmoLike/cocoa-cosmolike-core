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
// Armadillo notebook wrappers of the cluster covariance components.
//
// generic_interface_cluster_cov.cpp copies each NumPy argument into an
// owning arma::Col, arma::Mat or arma::Cube with notebook_input_cov, so
// C-order, Fortran-order, sliced and read-only arrays are all accepted and
// never modified. Each wrapper below checks shapes and physical domains,
// copies its inputs into padded malloc2d/malloc3d C workspaces with
// contiguous rows, and calls one shared *_cluster_cov.c routine. Every
// integral, SIMD loop and OpenMP team stays in C, in the same routines
// that the production interface (cluster_interface_cov.cpp) calls. The
// results return to Armadillo by physical indices and leave as a Python
// dict of NumPy arrays that own their memory through CARMA.
//
// Checks print through the shared logger and end the process (critical
// and exit, the data-vector layer's pattern); the C routines trust the
// sizes these checks establish.
// L denotes one consistent length unit, c/H0 in the survey workflow.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Expose count-shell quantities without assigning a mass-selection model.
//
// NumPy supplies selected abundances and their background-density
// derivatives. C converts them to counts per radial distance, using the
// shell volume. Shape and value checks precede C, so a malformed notebook
// input raises a Python exception instead of stopping the Python process.
// Returned arrays own their data and remain valid after subsequent calls.
// No likelihood, nuisance parameter or cosmology state is read or changed.
//
// With n_i the selected comoving abundance of count bin i and
// B_i = dn_i/d(delta_b), a shell of thickness dchi holds the volume
// area_sr f_K^2 dchi, so counts_shell_cluster_cov returns
//
//   S_i(chi)   = dN_i/dchi = area_sr f_K^2 n_i,
//   Phi_i(chi) = area_sr f_K^2 B_i,
//
// both in L^-1. integral dchi S_i gives the mean counts, and
// integral dchi sigma_b^2 Phi_i Phi_j their SSC. Inputs: distance
// arma::Col [nnode] (f_K in L); density and derivative arma::Mat
// [ncount,nnode] in L^-3. Returns a dict of arma::Mat [ncount,nnode]:
// shell_density (S_i) and shell_response (Phi_i).
// ---------------------------------------------------------------------------
py::dict covariance_counts_shell_cpp(
    const arma::Col<double>& distance, // [nnode], f_K in L
    const arma::Mat<double>& density,  // [ncount,nnode], selected n_i, L^-3
    const arma::Mat<double>& derivative, // matching dn_i/d(delta_b), L^-3
    const double area_sr               // survey solid angle, sr
  )
{
  if (distance.n_elem < 1
      || density.n_rows < 1
      || density.n_cols != distance.n_elem
      || derivative.n_rows != density.n_rows
      || derivative.n_cols != density.n_cols) {
    critical("{}: distance must be nonempty [nnode]; "
      "density and derivative must both have shape [ncount,nnode], with ncount > 0", "covariance_counts_shell_cpp");
    exit(1);
  }
  if (!std::isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    critical("{}: area_sr must be finite and in (0,4*pi]",
      "covariance_counts_shell_cpp");
    exit(1);
  }

  // A positive distance gives a physical shell volume. Zero abundances
  // are allowed where a bin's selection vanishes; signed responses permit
  // a selection model whose environmental change reduces the abundance.
  for (arma::uword node=0; node<distance.n_elem; node++) {
    if (!std::isfinite(distance(node))
        || distance(node) <= 0.0) {
      critical("{}: distance must be finite and positive",
        "covariance_counts_shell_cpp");
      exit(1);
    }
  }
  for (arma::uword entry=0; entry<density.n_elem; entry++) {
    if (!std::isfinite(density(entry))
        || density(entry) < 0.0
        || !std::isfinite(derivative(entry))) {
      critical("{}: density must be finite and nonnegative; "
        "derivative must be finite and may have either sign", "covariance_counts_shell_cpp");
      exit(1);
    }
  }

  const arma::uword ncount = density.n_rows;
  const arma::uword nnode = distance.n_elem;
  arma::Mat<double> shell(ncount, nnode);
  arma::Mat<double> response(ncount, nnode);
  double*** work = (double***) malloc3d(4, ncount, nnode);

  // Abundance and its response share the same catalog and radial axes.
  // C consumes contiguous shell rows; the notebook retains matrix axes.
  // Planes 0 and 1 hold n_i and B_i; planes 2 and 3 receive S_i and Phi_i.
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
//
// With q_c the normalized cluster window (integral q_c dchi = 1), b_c its
// selected bias and F_ell the core harmonic shear spin factor, the C
// routine limber_cluster_cov evaluates the Limber spectra
//
//   C_cc' = sum_node dchi q_c q_c' b_c b_c' P_NL / f_K^2,
//   C_cg  = sum_node dchi q_c b_c W_g P_NL / f_K^2,
//   C_cs  = F_ell sum_node dchi q_c W_s (b_c P_NL + P_cm^1h) / f_K^2.
//
// W_g already contains the galaxy bias. Only cluster-source spectra see
// the selected halo's own mass profile P_cm^1h. No noise, IA,
// magnification or RSD enters.
//
// Inputs: ell arma::Col [nell]; distance and dchi arma::Col [nnode] in L;
// base arma::Mat [nbase,nnode] (nlens galaxy windows, then lensing
// windows) and window arma::Mat [ncluster,nnode], both in L^-1; bias
// arma::Mat [ncluster,nnode]; power arma::Mat [nell,nnode] (P_NL at
// k=(ell+1/2)/f_K) and profile arma::Cube [nrichness,nell,nnode], both in
// L^3; richness arma::Col<int> [ncluster], the profile row of each
// category. Returns a dict of dimensionless arma::Cube: cluster_base
// [nell,ncluster,nbase] and cluster_cluster [nell,ncluster,ncluster].
// ---------------------------------------------------------------------------
py::dict covariance_cluster_spectra_cpp(
    const arma::Col<double>& ell,       // [nell], multipoles >= 2
    const arma::Col<double>& distance,  // [nnode], f_K in L
    const arma::Col<double>& dchi,      // [nnode], radial weights in L
    const arma::Mat<double>& base,      // [nbase,nnode], W_g then W_s, L^-1
    const arma::Mat<double>& window,    // [ncluster,nnode], q_c, L^-1
    const arma::Mat<double>& bias,      // [ncluster,nnode], selected b_c
    const arma::Mat<double>& power,     // [nell,nnode], P_NL at (ell+1/2)/f_K
    const arma::Cube<double>& profile,   // [nrichness,nell,nnode], P_cm^1h, L^3
    const arma::Col<int>& richness, // [ncluster], profile row of each q_c
    const int nlens                     // leading galaxy fields in base
  )
{
  if (ell.n_elem < 1
      || distance.n_elem < 1
      || dchi.n_elem != distance.n_elem) {
    critical("{}: ell/distance/dchi/richness must be "
      "vectors; base/window/bias/power matrices; profile a 3D array; ell/distance nonempty", "covariance_cluster_spectra_cpp");
    exit(1);
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
    critical("{}: need base[nbase,nnode], "
      "window/bias[ncluster,nnode], power[nell,nnode], profile[nrichness,nell,nnode], richness[ncluster], positive field counts and 0<=nlens<=nbase", "covariance_cluster_spectra_cpp");
    exit(1);
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
    critical("{}: covariance inputs must be finite",
      "covariance_cluster_spectra_cpp");
    exit(1);
  }
  for (arma::uword index=0; index<nell; index++) {
    if (ell(index) < 2.0) {
      critical("{}: cluster spectrum ell must be >= 2",
        "covariance_cluster_spectra_cpp");
      exit(1);
    }
  }
  for (arma::uword node=0; node<nnode; node++) {
    if (distance(node) <= 0.0
        || dchi(node) <= 0.0) {
      critical("{}: distance and dchi must be positive",
        "covariance_cluster_spectra_cpp");
      exit(1);
    }
  }
  for (arma::uword field=0; field<ncluster; field++) {
    if (richness(field) < 0
        || richness(field) >= nrichness) {
      critical("{}: richness indices must be in [0,nrichness)",
        "covariance_cluster_spectra_cpp");
      exit(1);
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
  // C rows list the ncluster*nbase (cluster,base) pairs, cluster-major,
  // then the cluster upper triangle (0,0),(0,1),...,(1,1),... .
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
//
// Write dn S_i for weight (dlnM dn/dlnM times the membership probability
// S_i, which enters once) and p(k) = (M/rho) u(k|M) for profile. The C
// routine moments_cluster_cov sums over the mass nodes
//
//   density = sum dn S_i,           biased_density = sum dn S_i b,
//   J01(K)  = sum dn S_i p(K),      J11(K) = sum dn S_i b p(K),
//   J02(K,Q) = sum dn S_i p(K) p(Q),
//   J03_KKQ = sum dn S_i p(K)^2 p(Q), J03_KQQ = sum dn S_i p(K) p(Q)^2.
//
// Inputs: weight arma::Cube [state,selection,mass] in L^-3, bias
// arma::Mat [state,mass], profile arma::Cube [state,k,mass] in L^3.
// Returns a dict: density and biased_density arma::Mat [state,selection]
// in L^-3; J01 and J11 arma::Cube [state,selection,k], dimensionless;
// J02, J03_KKQ and J03_KQQ arma::Cube [state,selection,kpair] in L^3,
// L^6 and L^6, kpair running over the k pairs (0,0),(0,1),...,(1,1),... .
// ---------------------------------------------------------------------------
py::dict covariance_cluster_moments_cpp(
    const arma::Cube<double>& weight,  // [state,selection,mass], dn S_i, L^-3
    const arma::Mat<double>& bias,    // [state,mass], linear halo bias
    const arma::Cube<double>& profile  // [state,k,mass], (M/rho)u(k|M), L^3
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
    critical("{}: all counts must be positive; weight, "
      "bias and profile must share their state and mass dimensions", "covariance_cluster_moments_cpp");
    exit(1);
  }
  if (!weight.is_finite()
      || !bias.is_finite()
      || !profile.is_finite()) {
    critical("{}: covariance inputs must be finite",
      "covariance_cluster_moments_cpp");
    exit(1);
  }
  for (arma::uword entry=0; entry<weight.n_elem; entry++) {
    if (weight(entry) < 0.0) {
      critical("{}: selected mass weights must be nonnegative",
        "covariance_cluster_moments_cpp");
      exit(1);
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
//
// For every state a, richness bin and mass node the C routine
// halo_samples_cluster_cov evaluates the selected abundance measure
//
//   weight = dlnM (rho_cb/M) f(nu,a) nu (dlnnu/dlnM) S_lambda(M,z),
//   nu     = 1.686/sigma_cb(M,a),
//
// with the initialized multiplicity f of HMF_TINKER_2010 and lognormal
// richness selection S_lambda, the linear halo bias b_h(M,a) and the profile
// (M/rho_m) u_NFW(k|M). Massless neutrinos give rho_cb = rho_m.
// Inputs: a arma::Col [state]; k arma::Mat [state,k] in (c/H0)^-1; lnm
// and dlnm arma::Col [mass], ln(M/[Msun/h]) and positive dlnM weights.
// Returns a dict: weight arma::Cube [state,richness,mass] in (c/H0)^-3,
// bias arma::Mat [state,mass] and profile arma::Cube [state,k,mass] in
// (c/H0)^3, ready for covariance_cluster_moments_cpp.
// ---------------------------------------------------------------------------
py::dict covariance_cluster_halo_samples_cpp(
    const arma::Col<double>& a,       // scale factors [state]
    const arma::Mat<double>& k,       // wavenumbers [state,k], (c/H0)^-1
    const arma::Col<double>& lnm,     // ln(M/[Msun/h]) [mass]
    const arma::Col<double>& dlnm     // positive quadrature measures [mass]
  )
{
  if (a.n_elem < 1
      || k.n_rows != a.n_elem
      || k.n_cols < 1
      || lnm.n_elem < 1
      || dlnm.n_elem != lnm.n_elem) {
    critical("{}: need nonempty a[state], k[state,k], lnm[mass], dlnm[mass]",
      "covariance_cluster_halo_samples_cpp");
    exit(1);
  }

  // The C routine supports only this initialized model: massless
  // neutrinos, the HMF_TINKER_2010 abundance, NFW profiles and a lognormal
  // mass-richness relation with positive scatter and no extra selection.
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
    critical("{}: initialize massless cosmology, NFW "
      "halos, richness bins and a lognormal MOR with positive scatter and selection_model=0", "covariance_cluster_halo_samples_cpp");
    exit(1);
  }
  if (cluster.hmf_alpha_mode != CLUSTER_HMF_ALPHA_FIXED
      && cluster.hmf_alpha_mode != CLUSTER_HMF_ALPHA_NORMALIZED) {
    critical("{}: cluster hmf_alpha_mode must be 0 or 1",
      "covariance_cluster_halo_samples_cpp");
    exit(1);
  }
  if (!a.is_finite()
      || !k.is_finite()
      || !lnm.is_finite()
      || !dlnm.is_finite()) {
    critical("{}: covariance inputs must be finite",
      "covariance_cluster_halo_samples_cpp");
    exit(1);
  }
  for (arma::uword state=0; state<a.n_elem; state++) {
    if (a(state) < limits.a_min
        || a(state) >= 1.0) {
      critical("{}: a must lie in [limits.a_min,1)",
        "covariance_cluster_halo_samples_cpp");
      exit(1);
    }
  }
  for (arma::uword entry=0; entry<k.n_elem; entry++) {
    if (k(entry) < 0.0) {
      critical("{}: k must be nonnegative in core units",
        "covariance_cluster_halo_samples_cpp");
      exit(1);
    }
  }
  for (arma::uword node=0; node<lnm.n_elem; node++) {
    if (lnm(node) < std::log(limits.halo_m[RANGE_MIN])
        || lnm(node) > std::log(limits.halo_m[RANGE_MAX])
        || dlnm(node) <= 0.0) {
      critical("{}: lnm must lie in the core sigma mass "
        "range; dlnm must be > 0", "covariance_cluster_halo_samples_cpp");
      exit(1);
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
