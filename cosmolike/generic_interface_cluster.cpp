#include "cosmolike/generic_interface_cluster.hpp"
#include <algorithm>
#include <string_view>
#include <unordered_map>
using namespace std::literals; // enables "sv" literal

// boost library
#include <boost/algorithm/string.hpp>

// std::isnan: no compile w/ -O3 or -fast-math stackoverflow.com/a/47703550/2472169

static constexpr std::string_view errbegins = "Begins Execution"sv;
static constexpr std::string_view errends = "Ends Execution"sv;
static constexpr std::string_view errleii = "logical error, internal inconsistent"sv;
static constexpr std::string_view erriiwz = "incompatible input vector with size = "sv;
static constexpr std::string_view errnance = "common error if `params_values.get(p, None)` return None"sv;
static constexpr std::string_view errnance2 = "{}: NaN found on index {} ({})."sv;
static constexpr std::string_view errorns = "{}: {}={} not supported (max={})"sv;
static constexpr std::string_view errorns2 = "{}: {} = {} not supported"sv;
static constexpr std::string_view debugsel = "{}: {} = {} selected."sv;
static constexpr std::string_view errornset = "{}: {} not set (?ill-defined) prior to this function call"sv;
static constexpr std::string_view errorsz1d = "{}: {} {} (!= {})"sv;

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using spdlog::info;
using spdlog::debug;
using spdlog::critical;

namespace cosmolike_interface
{

// ============================================================================
// [SECTION] INTERFACE STATE AND PRIVATE HELPERS
// ============================================================================

// Limber switches of w_cc and w_cg. They are interface choices, not model
// parameters, so they live here and not in the frozen `cluster` struct.
static int cluster_adopt_limber_cc = 1;
static int cluster_adopt_limber_cg = 1;

// Lowest redshift of the selection-kernel table (set_lens_sample: z = 0
// would put a cluster at chi = 0).
static constexpr double cluster_zdist_zmin_floor = 1.0e-5;

// (1 + z_piv) of the redshift factor of the Y6 selection bias,
// ((1 + zbar)/1.45)^s3 (lighthouse SELECTIONB; the paper fixes s3 = 0).
static constexpr double cluster_selection_pivot_1pz = 1.45;

// Guard of the float-to-int conversion of the mask column (the IP recipe).
static constexpr double mask_rounding_guard = 1.0e-13;

// Stencils of the Y transform need five theta bins (the one-sided rows).
static constexpr int ytransform_min_ntheta = 5;


// Number of richness pairs (nl1 <= nl2) of the w_cc block.
static int cluster_richness_npairs()
{
  return cluster.richness_nbin * (cluster.richness_nbin + 1) / 2;
}


// 1 when any cluster probe is on.
static int any_cluster_probe()
{
  if (cluster.probe_N || cluster.probe_cs ||
      cluster.probe_cc || cluster.probe_cg) {
    return 1;
  }
  return 0;
}


// ---------------------------------------------------------------------------
// Build the pair maps of redshift_spline_cluster.c here, single-threaded,
// and never inside an OpenMP region of a C function (the ZL/ZS race of
// pitfalls #2b). The maps own the pair counts: their builder rebuilds on a
// new cluster.random_pairs or on any bin-count change (cluster, source,
// lens, richness) and writes cluster.cs/cg/cc_npowerspectra, so every
// reader of those counts calls this first. Any accessor runs the builder;
// the calls below respect each accessor's range checks.
// ---------------------------------------------------------------------------
static void warmup_cluster_pair_maps()
{
  if (cluster.zdist_nbin > 0 && redshift.shear_nbin > 0) {
    (void) N_cs(0, 0);
  }
  if (cluster.richness_nbin > 0) {
    (void) N_cc_richness(0, 0);
  }
  if (cluster.cg_npowerspectra > 0) {  // count written by the builder above
    (void) ZC_cg(0);
  }
}


// ---------------------------------------------------------------------------
// Consistency of the cluster setters, checked (single-threaded) before the
// data files are read and before any theory vector is computed; it also
// brings the pair maps and their counts up to date.
// ---------------------------------------------------------------------------
static void check_cluster_state(std::string_view fname)
{
  if (0 == any_cluster_probe()) {
    return;
  }
  if (!(cluster.zdist_nbin > 0) || NULL == cluster.zdist_table) [[unlikely]] {
    critical(errornset, fname, "cluster selection kernels (set_cluster_zdist)");
    exit(1);
  }
  if (!(cluster.richness_nbin > 0)) [[unlikely]] {
    critical(errornset, fname, "richness bins (init_cluster_richness_bins)");
    exit(1);
  }
  warmup_cluster_pair_maps();
  if (1 == cluster.probe_cg && 0 == cluster.cg_npowerspectra) [[unlikely]] {
    critical("{}: w_cg is on but no cluster bin has a lens bin (call "
      "init_cluster_pairs after set_cluster_zdist)", fname);
    exit(1);
  }
  if (1 == cluster.probe_cs && 1 == cluster.ytransform &&
      Ntable.Ntheta < ytransform_min_ntheta) [[unlikely]] {
    critical("{}: the Y transform needs Ntheta >= {} (Ntheta = {})",
      fname, ytransform_min_ntheta, Ntable.Ntheta);
    exit(1);
  }
}



// ============================================================================
// [SECTION] INIT AND SET FUNCTIONS
// ============================================================================

// ---------------------------------------------------------------------------
// Reset the cluster state to the defaults of structs_cluster.c.
//
// reset_cluster_struct() leaves every cache key at 0; fresh draws follow
// so that no table built by a previous model in the same process can
// match them. The IPCluster measurement state and the interface Limber
// switches are reset too. Call after initial_setup().
// ---------------------------------------------------------------------------
void reset_cluster()
{
  static constexpr std::string_view fname = "reset_cluster"sv;
  debug("{}: {}", fname, errbegins);

  reset_cluster_struct();

  RandomNumber& random = RandomNumber::get_instance();
  cluster.random_model     = random.get();
  cluster.random_zdist     = random.get();
  cluster.random_mor       = random.get();
  cluster.random_selection = random.get();
  cluster.random_pairs     = random.get();

  IPCluster::get_instance().reset();

  cluster_adopt_limber_cc = 1;
  cluster_adopt_limber_cg = 1;

  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Probe combination by name. Flags in the cluster_block order
// (ss, gs, gg, cg, N, cc, cs). The CMB-lensing probes (gk, ks, kk) are
// always off: they have no slot in the joint vector.
//
//   4x2pt_N = CL+GC of arXiv 2503.13631: gg + cg + N + cc + cs
//   6x2pt_N = CL+3x2pt: every block
//   n, n_cc, n_cs, cs, cc, cg: subsets for the milestones and the tests
//   3x2pt   = the joint layout with every cluster probe off
// ---------------------------------------------------------------------------
void init_probes_cluster(std::string possible_probes)
{
  static constexpr std::string_view fname = "init_probes_cluster"sv;
  debug("{}: {}", fname, errbegins);

  using flags_t = arma::Col<int>::fixed<cluster_block::count>;
  static const std::unordered_map<std::string, flags_t> probe_map = {
      //                          ss gs gg cg  N cc cs
      { "4x2pt_n",  flags_t{{ 0, 0, 1, 1, 1, 1, 1 }} },
      { "6x2pt_n",  flags_t{{ 1, 1, 1, 1, 1, 1, 1 }} },
      { "n",        flags_t{{ 0, 0, 0, 0, 1, 0, 0 }} },
      { "n_cc",     flags_t{{ 0, 0, 0, 0, 1, 1, 0 }} },
      { "n_cs",     flags_t{{ 0, 0, 0, 0, 1, 0, 1 }} },
      { "cs",       flags_t{{ 0, 0, 0, 0, 0, 0, 1 }} },
      { "cc",       flags_t{{ 0, 0, 0, 0, 0, 1, 0 }} },
      { "cg",       flags_t{{ 0, 0, 0, 1, 0, 0, 0 }} },
      { "3x2pt",    flags_t{{ 1, 1, 1, 0, 0, 0, 0 }} },
    };

  boost::trim_if(possible_probes, boost::is_any_of("\t "));
  const std::string probe_key =
    boost::algorithm::to_lower_copy(possible_probes);
  auto it = probe_map.find(probe_key);
  if (it == probe_map.end()) [[unlikely]] {
    critical(errorns2, fname, "possible_probes", possible_probes);
    exit(1);
  }
  const flags_t& flags = it->second;

  like.probe[PROBE_SS] = flags(cluster_block::ss);
  like.probe[PROBE_GS]   = flags(cluster_block::gs);
  like.probe[PROBE_GG]     = flags(cluster_block::gg);
  like.probe[PROBE_GK] = 0;
  like.probe[PROBE_KS] = 0;
  like.probe[PROBE_KK] = 0;

  cluster.probe_cg = flags(cluster_block::cg);
  cluster.probe_N  = flags(cluster_block::N);
  cluster.probe_cc = flags(cluster_block::cc);
  cluster.probe_cs = flags(cluster_block::cs);

  debug(debugsel, fname, "possible_probes", probe_key);
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// The four cluster probe flags one by one (0 or 1); like.* is untouched.
// ---------------------------------------------------------------------------
void init_cluster_probes(
    const int N,
    const int cs,
    const int cc,
    const int cg
  )
{
  static constexpr std::string_view fname = "init_cluster_probes"sv;
  debug("{}: {}", fname, errbegins);

  const int flags[4] = {N, cs, cc, cg};
  const char* names[4] = {"N", "cs", "cc", "cg"};
  for (int i=0; i<4; i++) {
    if (!(0 == flags[i] || 1 == flags[i])) [[unlikely]] {
      critical(errorns2, fname, names[i], flags[i]);
      exit(1);
    }
  }
  cluster.probe_N  = N;
  cluster.probe_cs = cs;
  cluster.probe_cc = cc;
  cluster.probe_cg = cg;

  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Model choices of the cluster analysis (structs_cluster.h).
//
// Cache invalidation: draws cluster.random_model when any choice changed,
// so every cluster table keyed on it (halo_cluster.c, cosmo2D_cluster.c)
// rebuilds; unchanged input leaves the key alone.
//
// Validation: each choice within its enumeration, a finite magnification,
// else critical() + exit(1).
// ---------------------------------------------------------------------------
void init_cluster_model(
    const int mor_model,
    const int kernel_mode,
    const int selection_model,
    const int ytransform,
    const int include_ia,
    const double magnification
  )
{
  static constexpr std::string_view fname = "init_cluster_model"sv;
  debug("{}: {}", fname, errbegins);

  if (mor_model != CLUSTER_MOR_LOGNORMAL) [[unlikely]] {
    critical(errorns2, fname, "mor_model", mor_model);
    exit(1);
  }
  if (kernel_mode != CLUSTER_KERNEL_VOLUME &&
      kernel_mode != CLUSTER_KERNEL_ABUNDANCE) [[unlikely]] {
    critical(errorns2, fname, "kernel_mode", kernel_mode);
    exit(1);
  }
  if (selection_model != CLUSTER_SELECTION_NONE &&
      selection_model != CLUSTER_SELECTION_Y1 &&
      selection_model != CLUSTER_SELECTION_Y6) [[unlikely]] {
    critical(errorns2, fname, "selection_model", selection_model);
    exit(1);
  }
  if (!(0 == ytransform || 1 == ytransform)) [[unlikely]] {
    critical(errorns2, fname, "ytransform", ytransform);
    exit(1);
  }
  if (!(0 == include_ia || 1 == include_ia)) [[unlikely]] {
    critical(errorns2, fname, "include_ia", include_ia);
    exit(1);
  }
  if (!std::isfinite(magnification)) [[unlikely]] {
    critical(errorns2, fname, "magnification", magnification);
    exit(1);
  }

  const bool changed = (cluster.mor_model != mor_model) ||
                       (cluster.kernel_mode != kernel_mode) ||
                       (cluster.selection_model != selection_model) ||
                       (cluster.ytransform != ytransform) ||
                       (cluster.include_ia != include_ia) ||
                       fdiff(cluster.magnification, magnification);

  cluster.mor_model       = mor_model;
  cluster.kernel_mode     = kernel_mode;
  cluster.selection_model = selection_model;
  cluster.ytransform      = ytransform;
  cluster.include_ia      = include_ia;
  cluster.magnification   = magnification;

  if (changed) {
    cluster.random_model = RandomNumber::get_instance().get();
  }

  debug(debugsel, fname, "mor_model", mor_model);
  debug(debugsel, fname, "kernel_mode", kernel_mode);
  debug(debugsel, fname, "selection_model", selection_model);
  debug(debugsel, fname, "ytransform", ytransform);
  debug(debugsel, fname, "include_ia", include_ia);
  debug(debugsel, fname, "magnification", magnification);
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Amplitude alpha of the Tinker 2010 multiplicity in the cluster mass
// function (structs_cluster.h): CLUSTER_HMF_ALPHA_FIXED (0.368 at every z,
// 1001.3162 Table 4; the DES convention and the default) or
// CLUSTER_HMF_ALPHA_NORMALIZED (halo.c's alpha(a) from int b f dnu = 1).
//
// Cache invalidation: draws cluster.random_model when the mode changed, so
// the n_nl, b_nl and P1h tables (halo_cluster.c) and everything built on
// them refill; the same mode leaves the key alone.
//
// Validation: one of the two modes, else critical() + exit(1).
// ---------------------------------------------------------------------------
void init_cluster_hmf_alpha_mode(const int hmf_alpha_mode)
{
  static constexpr std::string_view fname = "init_cluster_hmf_alpha_mode"sv;
  debug("{}: {}", fname, errbegins);

  if (hmf_alpha_mode != CLUSTER_HMF_ALPHA_FIXED &&
      hmf_alpha_mode != CLUSTER_HMF_ALPHA_NORMALIZED) [[unlikely]] {
    critical(errorns2, fname, "hmf_alpha_mode", hmf_alpha_mode);
    exit(1);
  }

  if (cluster.hmf_alpha_mode != hmf_alpha_mode) {
    cluster.hmf_alpha_mode = hmf_alpha_mode;
    cluster.random_model = RandomNumber::get_instance().get();
  }

  debug(debugsel, fname, "hmf_alpha_mode", hmf_alpha_mode);
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Limber (1) or non-Limber (0) for the w_cc and w_cg blocks.
// ---------------------------------------------------------------------------
void init_cluster_adopt_limber(
    const int adopt_limber_cc,
    const int adopt_limber_cg
  )
{
  static constexpr std::string_view fname = "init_cluster_adopt_limber"sv;
  if (!(0 == adopt_limber_cc || 1 == adopt_limber_cc)) [[unlikely]] {
    critical(errorns2, fname, "adopt_limber_cc", adopt_limber_cc);
    exit(1);
  }
  if (!(0 == adopt_limber_cg || 1 == adopt_limber_cg)) [[unlikely]] {
    critical(errorns2, fname, "adopt_limber_cg", adopt_limber_cg);
    exit(1);
  }
  cluster_adopt_limber_cc = adopt_limber_cc;
  cluster_adopt_limber_cg = adopt_limber_cg;
}


// ---------------------------------------------------------------------------
// Observed-richness bins [lambda_min(nl), lambda_max(nl)).
//
// Cache invalidation: draws cluster.random_model when any edge changed
// (the richness binning belongs to that key); a new number of bins also
// draws cluster.random_pairs (the w_cc richness-pair map).
//
// Validation: equal sizes within MAX_SIZE_ARRAYS, finite edges with
// 0 < lambda_min < lambda_max (the MOR works in ln lambda).
// ---------------------------------------------------------------------------
void init_cluster_richness_bins(vector lambda_min, vector lambda_max)
{
  static constexpr std::string_view fname = "init_cluster_richness_bins"sv;
  debug("{}: {}", fname, errbegins);

  const int nbin = static_cast<int>(lambda_min.n_elem);
  if (!(nbin > 0) || nbin > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "richness nbin", nbin, MAX_SIZE_ARRAYS);
    exit(1);
  }
  if (nbin != static_cast<int>(lambda_max.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, lambda_max.n_elem, nbin);
    exit(1);
  }
  for (int nl=0; nl<nbin; nl++) {
    if (std::isnan(lambda_min(nl)) || std::isnan(lambda_max(nl))) [[unlikely]] {
      critical(errnance2, fname, nl, errnance);
      exit(1);
    }
    if (!(lambda_min(nl) > 0.0) || !(lambda_max(nl) > lambda_min(nl))) [[unlikely]] {
      critical("{}: richness bin {} = [{}, {}] not supported",
        fname, nl, lambda_min(nl), lambda_max(nl));
      exit(1);
    }
  }

  bool changed = (cluster.richness_nbin != nbin);
  const bool new_nbin = changed;
  for (int nl=0; nl<nbin; nl++) {
    if (fdiff(cluster.richness_min[nl], lambda_min(nl)) ||
        fdiff(cluster.richness_max[nl], lambda_max(nl))) {
      changed = true;
    }
    cluster.richness_min[nl] = lambda_min(nl);
    cluster.richness_max[nl] = lambda_max(nl);
  }
  cluster.richness_nbin = nbin;

  if (changed) {
    cluster.random_model = RandomNumber::get_instance().get();
  }
  if (new_nbin) {
    cluster.random_pairs = RandomNumber::get_instance().get();
  }
  for (int nl=0; nl<nbin; nl++) {
    debug("{}: richness bin {} = [{}, {}]", fname, nl,
      cluster.richness_min[nl], cluster.richness_max[nl]);
  }
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Install the cluster selection kernels <phi_i|z_true> and the nominal
// z_lambda edges of each cluster redshift bin.
//
// Layout (the set_lens_sample design): cluster.zdist_table is
// (nbin + 1) x nz with <phi_i|z> in row i and the z grid in row nbin.
// Unlike a galaxy n(z) histogram, the z column holds sample points of a
// function (no left-edge convention): the table covers [z_0, z_last].
//
// Support of bin i: the zero nodes that bracket the nonzero entries of
// <phi_i|z> (the first nonzero node and the last one, each widened by one
// grid point, clamped to the table). The kernel is read by linear
// interpolation of the table, which ramps from the last nonzero node to
// the next (zero) one; the widening keeps that whole ramp inside the
// integration range, so the cut removes nothing of the interpolant (it
// matters for a top-hat kernel, whose ramp is the bin edge itself). No
// tail is truncated here: a kernel table should be exactly zero outside
// the range it means (the table generator's +- n sigma_z cut), or the
// support, and with it the Limber integration range, spans the table.
//
// The nominal edges zbin_min/zbin_max are kept separately: the selection
// bias (zbar, eq 23) and the physical scale cuts read them, and no kernel
// table redefines them.
//
// Cache invalidation: when the table or an edge changed (fdiff), draws
// cluster.random_zdist; a new number of bins also draws
// cluster.random_pairs and clears the w_cg pairing (cg_lens_bin), so
// init_cluster_pairs must run again before the next data vector. The fine-z kernel table of
// redshift_spline_cluster.c is rebuilt by cluster_warmup at the next
// computation, single-threaded.
//
// Validation: 0 < nbin <= MAX_SIZE_ARRAYS, at least two z rows, strictly
// increasing z >= 0, finite non-negative kernels with a positive maximum,
// finite edges with zbin_min < zbin_max.
// ---------------------------------------------------------------------------
void set_cluster_zdist(matrix input_table, vector zbin_min, vector zbin_max)
{
  static constexpr std::string_view fname = "set_cluster_zdist"sv;
  debug("{}: {}", fname, errbegins);

  // --- 1. VALIDATION ---

  const int nbin = static_cast<int>(input_table.n_cols) - 1;
  const int nz = static_cast<int>(input_table.n_rows);
  if (!(nbin > 0) || nbin > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "cluster nbin", nbin, MAX_SIZE_ARRAYS);
    exit(1);
  }
  if (nz < 2) [[unlikely]] {
    critical(errorns2, fname, "number of z rows", nz);
    exit(1);
  }
  if (nbin != static_cast<int>(zbin_min.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, zbin_min.n_elem, nbin);
    exit(1);
  }
  if (nbin != static_cast<int>(zbin_max.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, zbin_max.n_elem, nbin);
    exit(1);
  }
  for (int i=0; i<nz; i++) {
    for (int k=0; k<nbin+1; k++) {
      if (!std::isfinite(input_table(i,k))) [[unlikely]] {
        critical("{}: non-finite entry at row {}, column {}", fname, i, k);
        exit(1);
      }
    }
    if (input_table(i,0) < 0.0) [[unlikely]] {
      critical("{}: negative redshift at row {}", fname, i);
      exit(1);
    }
    if (i > 0 && !(input_table(i,0) > input_table(i-1,0))) [[unlikely]] {
      critical("{}: z column not strictly increasing at row {}", fname, i);
      exit(1);
    }
    for (int k=0; k<nbin; k++) {
      if (input_table(i,k+1) < 0.0) [[unlikely]] {
        critical("{}: negative <phi|z> of bin {} at row {}", fname, k, i);
        exit(1);
      }
    }
  }
  for (int k=0; k<nbin; k++) {
    if (!std::isfinite(zbin_min(k)) || !std::isfinite(zbin_max(k)) ||
        zbin_min(k) < 0.0 || !(zbin_max(k) > zbin_min(k))) [[unlikely]] {
      critical("{}: cluster z bin {} = [{}, {}] not supported",
        fname, k, zbin_min(k), zbin_max(k));
      exit(1);
    }
  }

  // --- 2. DID ANYTHING CHANGE? ---

  bool changed = (cluster.zdist_nbin != nbin) ||
                 (cluster.zdist_nz != nz) ||
                 (NULL == cluster.zdist_table);
  if (!changed) {
    const double* z_v = cluster.zdist_table[nbin];
    for (int i=0; i<nz && !changed; i++) {
      if (fdiff(z_v[i], input_table(i,0))) {
        changed = true;
      }
      for (int k=0; k<nbin && !changed; k++) {
        if (fdiff(cluster.zdist_table[k][i], input_table(i,k+1))) {
          changed = true;
        }
      }
    }
    for (int k=0; k<nbin && !changed; k++) {
      if (fdiff(cluster.zbin_min[k], zbin_min(k)) ||
          fdiff(cluster.zbin_max[k], zbin_max(k))) {
        changed = true;
      }
    }
  }
  if (!changed) {
    debug("{}: {}", fname, errends);
    return;
  }

  // --- 3. A NEW NUMBER OF BINS INVALIDATES THE w_cg PAIRING ---

  if (cluster.zdist_nbin != nbin) {
    for (int k=0; k<MAX_SIZE_ARRAYS; k++) {
      cluster.cg_lens_bin[k] = -1;
    }
    cluster.random_pairs = RandomNumber::get_instance().get();
  }

  // --- 4. COPY THE TABLE (z IN THE LAST ROW) ---

  if (cluster.zdist_table != NULL) {
    free(cluster.zdist_table);
  }
  cluster.zdist_table = (double**) malloc2d(nbin + 1, nz);
  cluster.zdist_nbin = nbin;
  cluster.zdist_nz = nz;

  double** tab = cluster.zdist_table;  // alias
  double* z_v = cluster.zdist_table[nbin];  // alias
  for (int i=0; i<nz; i++) {
    z_v[i] = input_table(i,0);
    for (int k=0; k<nbin; k++) {
      tab[k][i] = input_table(i,k+1);
    }
  }
  cluster.zdist_zmin_all = fmax(z_v[0], cluster_zdist_zmin_floor);
  cluster.zdist_zmax_all = z_v[nz-1];

  // --- 5. SUPPORT OF EACH KERNEL ---

  for (int k=0; k<nbin; k++) {
    double phi_max = tab[k][0];
    for (int i=1; i<nz; i++) {
      phi_max = fmax(phi_max, tab[k][i]);
    }
    if (!(phi_max > 0.0)) [[unlikely]] {
      critical("{}: <phi|z> of cluster bin {} has no positive entry", fname, k);
      exit(1);
    }
    int first = -1;
    int last  = -1;
    for (int i=0; i<nz; i++) {
      if (tab[k][i] > 0.0) {
        if (first < 0) {
          first = i;
        }
        last = i;
      }
    }
    // widen by one grid point: the linear interpolant ramps to zero there
    // (the column holds sample points, passed through unchanged)
    const int first_widened = (first > 0) ? first - 1 : 0;
    const int last_widened  = (last < nz - 1) ? last + 1 : nz - 1;

    cluster.zdist_zmin[k] = fmax(z_v[first_widened], cluster.zdist_zmin_all);
    cluster.zdist_zmax[k] = z_v[last_widened];
    cluster.zbin_min[k] = zbin_min(k);
    cluster.zbin_max[k] = zbin_max(k);

    debug("{}: cluster bin {}: nominal [{}, {}], support [{}, {}]", fname, k,
      cluster.zbin_min[k], cluster.zbin_max[k],
      cluster.zdist_zmin[k], cluster.zdist_zmax[k]);
  }

  cluster.random_zdist = RandomNumber::get_instance().get();
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Tomographic pairs of the cluster blocks (the rules live in the pair maps
// of redshift_spline_cluster.c, which own the counts
// cluster.cs/cg/cc_npowerspectra):
//
//   cs: every (cluster bin, source bin) pair, cluster-major (lighthouse);
//       unwanted pairs are masked, not removed
//   cg: cluster bin ni with lens bin cg_lens_bin(ni); -1 = no w_cg for ni
//   cc: every cluster bin (auto z bin), all richness pairs nl1 <= nl2
//
// Cache invalidation: sets cluster.cg_lens_bin and always draws
// cluster.random_pairs (the bins can differ from a previous model in the
// same process, the init_ntomo_powerspectra rule), then builds the pair
// maps here, single-threaded, and reads the counts they wrote.
//
// Validation: cluster bins and richness bins set; one entry per cluster
// bin, each -1 or a lens bin in [0, clustering_nbin).
// ---------------------------------------------------------------------------
void init_cluster_pairs(arma::Col<int> cg_lens_bin)
{
  static constexpr std::string_view fname = "init_cluster_pairs"sv;
  debug("{}: {}", fname, errbegins);

  if (!(cluster.zdist_nbin > 0)) [[unlikely]] {
    critical(errornset, fname, "cluster bins (set_cluster_zdist)");
    exit(1);
  }
  if (!(cluster.richness_nbin > 0)) [[unlikely]] {
    critical(errornset, fname, "richness bins (init_cluster_richness_bins)");
    exit(1);
  }
  if (cluster.zdist_nbin != static_cast<int>(cg_lens_bin.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, cg_lens_bin.n_elem, cluster.zdist_nbin);
    exit(1);
  }
  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    const int ng = cg_lens_bin(ni);
    if (ng < -1 || ng >= redshift.clustering_nbin) [[unlikely]] {
      critical("{}: cg_lens_bin({}) = {} not supported (lens nbin = {})",
        fname, ni, ng, redshift.clustering_nbin);
      exit(1);
    }
  }

  for (int k=0; k<MAX_SIZE_ARRAYS; k++) {
    cluster.cg_lens_bin[k] = -1;
  }
  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    cluster.cg_lens_bin[ni] = cg_lens_bin(ni);
  }

  // the maps own the pair counts: build them, then read the counts
  cluster.random_pairs = RandomNumber::get_instance().get();
  warmup_cluster_pair_maps();

  debug("{}: cluster.cs_npowerspectra = {}", fname, cluster.cs_npowerspectra);
  debug("{}: cluster.cg_npowerspectra = {}", fname, cluster.cg_npowerspectra);
  debug("{}: cluster.cc_npowerspectra = {}", fname, cluster.cc_npowerspectra);
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Mass-observable relation (eqs 18-19), lighthouse order:
//   MOR = {ln lambda_0, A_lambda, sigma_int, B_lambda}
//
// Cache invalidation: draws cluster.random_mor when any value changed.
// Validation: length of the mor_model, no NaN.
// ---------------------------------------------------------------------------
void set_nuisance_cluster_mor(vector MOR)
{
  static constexpr std::string_view fname = "set_nuisance_cluster_mor"sv;
  debug("{}: {}", fname, errbegins);

  const int nmor = cluster_nmor_lognormal;  // CLUSTER_MOR_LOGNORMAL only
  if (nmor != static_cast<int>(MOR.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, MOR.n_elem, nmor);
    exit(1);
  }
  int cache_update = 0;
  for (int i=0; i<nmor; i++) {
    if (std::isnan(MOR(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance);
      exit(1);
    }
    if (fdiff(cluster.mor[i], MOR(i))) {
      cache_update = 1;
      cluster.mor[i] = MOR(i);
    }
  }
  if (1 == cache_update) {
    cluster.random_mor = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Selection-bias parameters (structs_cluster.h):
//   CLUSTER_SELECTION_Y6: {b_s1, b_s2, r_0 [comoving Mpc/h], s3}
//   CLUSTER_SELECTION_Y1: {b_s0, b_s1, b_s2, unused}
//
// Cache invalidation: draws cluster.random_selection when any value
// changed (only the Y1 bias tables read it; the Y6 factor is applied on
// the data vector at every call).
// Validation: four entries, no NaN; r_0 > 0 under CLUSTER_SELECTION_Y6
// (it divides theta chi(zbar) in eq 23).
// ---------------------------------------------------------------------------
void set_nuisance_cluster_selection(vector SEL)
{
  static constexpr std::string_view fname = "set_nuisance_cluster_selection"sv;
  debug("{}: {}", fname, errbegins);

  if (cluster_nselection != static_cast<int>(SEL.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, SEL.n_elem, cluster_nselection);
    exit(1);
  }
  for (int i=0; i<cluster_nselection; i++) {
    if (std::isnan(SEL(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance);
      exit(1);
    }
  }
  if (CLUSTER_SELECTION_Y6 == cluster.selection_model &&
      !(SEL(2) > 0.0)) [[unlikely]] {
    critical("{}: r_0 = {} must be positive (CLUSTER_SELECTION_Y6)",
      fname, SEL(2));
    exit(1);
  }
  int cache_update = 0;
  for (int i=0; i<cluster_nselection; i++) {
    if (fdiff(cluster.selection[i], SEL(i))) {
      cache_update = 1;
      cluster.selection[i] = SEL(i);
    }
  }
  if (1 == cache_update) {
    cluster.random_selection = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Measurement side of the joint vector: mask, then data, then covariance
// (IPCluster). The block sizes come from the current setters, so every
// init/set function above (and init_ntomo_powerspectra, init_binning)
// must have run first.
// ---------------------------------------------------------------------------
void init_data_cluster(std::string cov, std::string mask, std::string data)
{
  static constexpr std::string_view fname = "init_data_cluster"sv;
  debug("{}: {}", fname, errbegins);
  check_cluster_state(fname);
  IPCluster& ipc = IPCluster::get_instance();
  ipc.set_mask(mask);  // set_mask must be called first
  ipc.set_data(data);
  ipc.set_inv_cov(cov);
  debug("{}: {}", fname, errends);
}



// ============================================================================
// [SECTION] BLOCK SIZES AND STARTS
// ============================================================================

// ---------------------------------------------------------------------------
// Length of each block (cluster_block order). The ss, gs, gg lengths come
// from the core (compute_data_vector_Mx2pt_N_sizes<0,3>: real space, xi+
// and xi- stacked). Sizes do not depend on the probe flags.
// ---------------------------------------------------------------------------
arma::Col<int>::fixed<cluster_block::count> compute_data_vector_cluster_sizes()
{
  static constexpr std::string_view fname = "compute_data_vector_cluster_sizes"sv;
  debug("{}: {}", fname, errbegins);

  // the pair counts below are written by the pair maps: bring them up to date
  warmup_cluster_pair_maps();

  const arma::Col<int>::fixed<3> sizes_3x2pt =
    compute_data_vector_Mx2pt_N_sizes<0,3>();

  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;

  arma::Col<int>::fixed<cluster_block::count> sizes;
  sizes(cluster_block::ss) = sizes_3x2pt(0);
  sizes(cluster_block::gs) = sizes_3x2pt(1);
  sizes(cluster_block::gg) = sizes_3x2pt(2);
  sizes(cluster_block::cg) = ntheta*nrichness*cluster.cg_npowerspectra;
  sizes(cluster_block::N)  = cluster.zdist_nbin*nrichness;
  sizes(cluster_block::cc) = ntheta*cluster_richness_npairs()*
                             cluster.cc_npowerspectra;
  sizes(cluster_block::cs) = ntheta*nrichness*cluster.cs_npowerspectra;

  debug("{}: {}", fname, errends);
  return sizes;
}


// Offset of each block: the summed sizes of the blocks before it.
arma::Col<int>::fixed<cluster_block::count> compute_data_vector_cluster_starts()
{
  const arma::Col<int>::fixed<cluster_block::count> sizes =
    compute_data_vector_cluster_sizes();
  arma::Col<int>::fixed<cluster_block::count> start(arma::fill::zeros);
  for (int b=1; b<cluster_block::count; b++) {
    start(b) = start(b-1) + sizes(b-1);
  }
  return start;
}



// ============================================================================
// [SECTION] Y TRANSFORM AND SELECTION BIAS (DATA-VECTOR LEVEL)
// ============================================================================

// ---------------------------------------------------------------------------
// The Park, Rozo & Krause (2021, arXiv 2004.07504) localization matrix on
// the analysis theta grid, eq (15) of 2503.13631: Sigma = T gamma_t.
//
/* PHYSICAL DERIVATION & LOGIC FLOW
   1. Y(R) = Sigma(R) - Sigma(R_max)
           = int_{ln R}^{ln R_max} dln R' [2 DSigma + dDSigma/dln R']
      (P21 eq 9). The integrand removes the enclosed mass: Y at R depends
      only on the profile between R and R_max.
   2. With R = f_K(chi(zbar)) theta the map is the same in ln theta, and
      gamma_t = DSigma/Sigma_crit with Sigma_crit constant per block, so
      T acts on gamma_t directly (P21 eq 12) and Sigma_crit cancels.
   3. The bins are uniform in ln theta, Delta = ln(theta_max/theta_min)/N
      (the area-weighted centers are uniform in ln theta too).
   4. S = trapezoid rule from theta_i to theta_max (P21 eq 10):
        S[i][i] = Delta/2, S[i][j] = Delta (i < j < N-1),
        S[i][N-1] = Delta/2, row N-1 = 0 (Y(R_max) = 0).
   5. D = d/dln theta by finite differences, divided by Delta: central
      stencils of half-width min(i, N-1-i, 4) (orders 2 to 8), the
      one-sided 5-point stencil in rows 0 and N-1.
   6. T = 2 S + S D (lighthouse cluster_util.c T_Ytransform).
   Consequence: row N-1 of T is zero (the last theta bin of every cs row
   carries no information), and Y at bin i reads gamma_t down to bin
   i - 4: gamma_t is needed at every bin, masked or not. */
// ---------------------------------------------------------------------------
matrix compute_cluster_ytransform_matrix()
{
  static constexpr std::string_view fname = "compute_cluster_ytransform_matrix"sv;
  debug("{}: {}", fname, errbegins);

  const int N = Ntable.Ntheta;
  if (N < ytransform_min_ntheta) [[unlikely]] {
    critical("{}: the Y transform needs Ntheta >= {} (Ntheta = {})",
      fname, ytransform_min_ntheta, N);
    exit(1);
  }
  if (!(Ntable.vt[RANGE_MAX] > Ntable.vt[RANGE_MIN]) || !(Ntable.vt[RANGE_MIN] > 0.0)) [[unlikely]] {
    critical(errornset, fname, "Ntable.vt[RANGE_MAX] and Ntable.vt[RANGE_MIN]");
    exit(1);
  }
  const double Delta = std::log(Ntable.vt[RANGE_MAX]/Ntable.vt[RANGE_MIN])/N;

  // --- 1. TRAPEZOID MATRIX S ---

  matrix S(N, N, arma::fill::zeros);
  for (int i=0; i<N-1; i++) {
    S(i,i) = 0.5*Delta;
    for (int j=i+1; j<N-1; j++) {
      S(i,j) = Delta;
    }
    S(i,N-1) = 0.5*Delta;
  }

  // --- 2. FINITE-DIFFERENCE MATRIX D ---

  // one-sided 5-point first derivative (4th order): coefficients of
  // f(0), f(1), ..., f(4) in row 0; row N-1 mirrors it with the sign flipped
  const double one_sided[5] = {-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0};

  // central first derivative of half-width w: coefficient of f(i + m),
  // m = 1..w (f(i - m) gets the opposite sign)
  const double central[5][4] = {
    {0.0,         0.0,         0.0,         0.0},        // w = 0 (unused)
    {1.0/2.0,     0.0,         0.0,         0.0},        // 2nd order
    {2.0/3.0,    -1.0/12.0,    0.0,         0.0},        // 4th order
    {3.0/4.0,    -3.0/20.0,    1.0/60.0,    0.0},        // 6th order
    {4.0/5.0,    -1.0/5.0,     4.0/105.0,  -1.0/280.0}   // 8th order
  };
  const int max_half_width = 4;

  matrix D(N, N, arma::fill::zeros);
  for (int m=0; m<5; m++) {
    D(0, m) = one_sided[m];
    D(N-1, N-1-m) = -one_sided[m];
  }
  for (int i=1; i<N-1; i++) {
    const int width = std::min(std::min(i, N - 1 - i), max_half_width);
    for (int m=1; m<=width; m++) {
      D(i, i + m) =  central[width][m-1];
      D(i, i - m) = -central[width][m-1];
    }
  }
  D /= Delta;

  // --- 3. T = 2 S + S D ---

  matrix T = 2.0*S + S*D;

  debug("{}: {}", fname, errends);
  return T;
}


// ---------------------------------------------------------------------------
// Selection-bias factor of eq (23), applied on the data vector:
//
/* PHYSICAL DERIVATION & LOGIC FLOW
   1. Clusters selected by richness sit preferentially in filaments
      aligned with the line of sight, which boosts their lensing and
      clustering signal on large scales (Wu et al. 2022). Eq (23) models
      the ratio to the unbiased prediction:
        B(theta) = [b_s1 + b_s2 exp(-R/r_0)] ((1 + zbar)/1.45)^s3,
        R = theta f_K(chi(zbar)): the comoving transverse separation in
        Mpc/h (= chi in a flat universe), r_0 in comoving Mpc/h.
      The redshift factor is the lighthouse s3 extension (0 in the paper).
   2. theta = the area-weighted center of each angular bin (radians),
      zbar = zmid_cluster(ni), the nominal bin midpoint.
   3. Sigma and w_cg carry B, w_cc carries B^2 (one factor per cluster
      leg); the counts carry none. */
//
// Returns the (cluster z bin) x (theta bin) matrix; ones unless
// cluster.selection_model == CLUSTER_SELECTION_Y6.
// ---------------------------------------------------------------------------
matrix compute_cluster_selection_factor()
{
  static constexpr std::string_view fname = "compute_cluster_selection_factor"sv;
  debug("{}: {}", fname, errbegins);

  const int ntheta = Ntable.Ntheta;
  matrix B(cluster.zdist_nbin, ntheta, arma::fill::ones);
  if (CLUSTER_SELECTION_Y6 != cluster.selection_model) {
    return B;
  }

  const double b_s1 = cluster.selection[0];
  const double b_s2 = cluster.selection[1];
  const double r_0  = cluster.selection[2];  // comoving Mpc/h
  const double s3   = cluster.selection[3];
  if (!(r_0 > 0.0)) [[unlikely]] {
    critical("{}: r_0 = {} must be positive (CLUSTER_SELECTION_Y6)", fname, r_0);
    exit(1);
  }

  const vector theta = compute_binning_real_space();  // radians

  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    // comoving transverse distance to the bin midpoint in Mpc/h
    const double zbar = zmid_cluster(ni);
    const double a_bar = 1.0/(1.0 + zbar);
    const double D_M = f_K(chi(a_bar))*cosmology.coverH0;

    const double redshift_factor =
      std::pow((1.0 + zbar)/cluster_selection_pivot_1pz, s3);

    for (int nt=0; nt<ntheta; nt++) {
      const double R = theta(nt)*D_M;  // Mpc/h
      B(ni, nt) = (b_s1 + b_s2*std::exp(-R/r_0))*redshift_factor;
    }
  }
  debug("{}: {}", fname, errends);
  return B;
}



// ============================================================================
// [SECTION] BLOCK FILLERS OF THE JOINT THEORY VECTOR
// ============================================================================
//
// Each filler computes the model at the unmasked entries of its block
// (the core compute_X_N_masked engines, read through the IPCluster mask),
// applies the data-vector-level factors, and leaves masked entries at 0.

// ---------------------------------------------------------------------------
// ss: xi+ half-block then xi- half-block, times (1+m_z1)(1+m_z2).
// ---------------------------------------------------------------------------
static void compute_ss_block_masked(vector& dv, const int start)
{
  IPCluster& ipc = IPCluster::get_instance();
  const int ntheta = Ntable.Ntheta;
  const int xim_offset = ntheta*tomo.shear_Npowerspectra;

  for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
    const int z1 = Z1(nz);
    const int z2 = Z2(nz);
    const double shear_calib = (1.0 + nuisance.shear_calibration_m[z1])*
                               (1.0 + nuisance.shear_calibration_m[z2]);
    for (int i=0; i<ntheta; i++) {
      const int index_xip = start + ntheta*nz + i;
      const int index_xim = index_xip + xim_offset;
      if (ipc.get_mask(index_xip)) {
        dv(index_xip) = xi_pm_tomo(1, i, z1, z2, 1)*shear_calib;
      }
      if (ipc.get_mask(index_xim)) {
        dv(index_xim) = xi_pm_tomo(-1, i, z1, z2, 1)*shear_calib;
      }
    }
  }
}


// ---------------------------------------------------------------------------
// gs: gamma_t plus the point-mass term, times (1+m_zs) (the core real-space
// recipe, add_calib_and_set_mask_X_N<0,1,1>).
// ---------------------------------------------------------------------------
static void compute_gs_block_masked(vector& dv, const int start)
{
  IPCluster& ipc = IPCluster::get_instance();
  const int ntheta = Ntable.Ntheta;
  const vector theta = compute_binning_real_space();  // radians

  for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
    const int zl = ZL(nz);
    const int zs = ZS(nz);
    const double shear_calib = 1.0 + nuisance.shear_calibration_m[zs];
    for (int i=0; i<ntheta; i++) {
      const int index = start + ntheta*nz + i;
      if (ipc.get_mask(index)) {
        const double gammat = w_gammat_tomo(i, zl, zs, like.adopt_limber[LIMBER_GS]);
        const double point_mass =
          PointMass::get_instance().get_pm(zl, zs, theta(i));
        dv(index) = (gammat + point_mass)*shear_calib;
      }
    }
  }
}


// ---------------------------------------------------------------------------
// gg: w_gg auto spectra of the lens bins.
// ---------------------------------------------------------------------------
static void compute_gg_block_masked(vector& dv, const int start)
{
  IPCluster& ipc = IPCluster::get_instance();
  const int ntheta = Ntable.Ntheta;

  for (int nz=0; nz<tomo.clustering_Npowerspectra; nz++) {
    for (int i=0; i<ntheta; i++) {
      const int index = start + ntheta*nz + i;
      if (ipc.get_mask(index)) {
        dv(index) = w_gg_tomo(i, nz, nz, like.adopt_limber[LIMBER_GG]);
      }
    }
  }
}


// ---------------------------------------------------------------------------
// cg: w_cg times the selection factor B(theta) of its cluster bin.
// Layout [cg pair][lambda][theta].
// ---------------------------------------------------------------------------
static void compute_cg_block_masked(
    vector& dv,
    const int start,
    const matrix& selection
  )
{
  IPCluster& ipc = IPCluster::get_instance();
  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;

  for (int n=0; n<cluster.cg_npowerspectra; n++) {
    const int ni = ZC_cg(n);
    const int ng = ZG_cg(n);
    for (int nl=0; nl<nrichness; nl++) {
      const int row_start = start + ntheta*(nrichness*n + nl);
      for (int i=0; i<ntheta; i++) {
        const int index = row_start + i;
        if (ipc.get_mask(index)) {
          const double w_cg = w_cg_tomo(i, nl, ni, ng, cluster_adopt_limber_cg);
          dv(index) = selection(ni, i)*w_cg;
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// N: expected counts, layout [cluster z bin][lambda]. No selection bias
// and no calibration: eq (16) as it is.
// ---------------------------------------------------------------------------
static void compute_N_block_masked(vector& dv, const int start)
{
  IPCluster& ipc = IPCluster::get_instance();
  const int nrichness = cluster.richness_nbin;

  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    for (int nl=0; nl<nrichness; nl++) {
      const int index = start + nrichness*ni + nl;
      if (ipc.get_mask(index)) {
        dv(index) = N_cluster_tomo(nl, ni);
      }
    }
  }
}


// ---------------------------------------------------------------------------
// cc: w_cc times B(theta)^2 (one selection factor per cluster leg).
// Layout [cluster z bin][richness pair nl1 <= nl2][theta].
// ---------------------------------------------------------------------------
static void compute_cc_block_masked(
    vector& dv,
    const int start,
    const matrix& selection
  )
{
  IPCluster& ipc = IPCluster::get_instance();
  const int ntheta = Ntable.Ntheta;
  const int npairs = cluster_richness_npairs();

  for (int ni=0; ni<cluster.cc_npowerspectra; ni++) {
    for (int n=0; n<npairs; n++) {
      const int nl1 = NL1_cc(n);
      const int nl2 = NL2_cc(n);
      const int row_start = start + ntheta*(npairs*ni + n);
      for (int i=0; i<ntheta; i++) {
        const int index = row_start + i;
        if (ipc.get_mask(index)) {
          const double w_cc =
            w_cc_tomo(i, nl1, nl2, ni, cluster_adopt_limber_cc);
          dv(index) = selection(ni, i)*selection(ni, i)*w_cc;
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// cs: cluster lensing. Layout [cs pair][lambda][theta].
//
/* PHYSICAL DERIVATION & LOGIC FLOW
   1. gamma_t(theta_k) of the (cluster bin, source bin, richness bin) row
      at EVERY theta bin: T couples bin i to bins i-4 .. N-1, so a masked
      bin still feeds the unmasked ones (only a fully masked row is
      skipped).
   2. Localization, eq (15): Sigma_i = sum_k T_ik gamma_t(theta_k)
      (cluster.ytransform = 1); Sigma = gamma_t otherwise (Y1).
   3. Selection bias, eq (23), after the transform: Sigma_i *= B_i.
   4. Shear calibration of the source leg: Sigma_i *= (1 + m_ns).
   5. Mask. */
// ---------------------------------------------------------------------------
static void compute_cs_block_masked(
    vector& dv,
    const int start,
    const matrix& selection,
    const matrix& T
  )
{
  IPCluster& ipc = IPCluster::get_instance();
  const int ntheta = Ntable.Ntheta;
  const int nrichness = cluster.richness_nbin;
  const bool ytransform = (1 == cluster.ytransform);

  vector gammat(ntheta, arma::fill::zeros);
  vector sigma(ntheta, arma::fill::zeros);

  for (int n=0; n<cluster.cs_npowerspectra; n++) {
    const int ni = ZC_cs(n);
    const int ns = ZS_cs(n);
    const double shear_calib = 1.0 + nuisance.shear_calibration_m[ns];

    for (int nl=0; nl<nrichness; nl++) {
      const int row_start = start + ntheta*(nrichness*n + nl);

      int nkept = 0;
      for (int i=0; i<ntheta; i++) {
        nkept += ipc.get_mask(row_start + i);
      }
      if (0 == nkept) {
        continue;
      }

      // --- 1. GAMMA_T (EVERY BIN WITH THE Y TRANSFORM) ---
      gammat.zeros();
      for (int i=0; i<ntheta; i++) {
        if (ytransform || ipc.get_mask(row_start + i)) {
          gammat(i) = w_gammat_cluster_tomo(i, nl, ni, ns);
        }
      }

      // --- 2. LOCALIZATION ---
      if (ytransform) {
        sigma = T*gammat;
      }
      else {
        sigma = gammat;
      }

      // --- 3.-5. SELECTION BIAS, CALIBRATION, MASK ---
      for (int i=0; i<ntheta; i++) {
        const int index = row_start + i;
        if (ipc.get_mask(index)) {
          dv(index) = selection(ni, i)*sigma(i)*shear_calib;
        }
        else {
          dv(index) = 0.0;
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// The cobaya-facing theory vector of the joint analysis, full layout
// (zeros off-mask), to be compared with IPCluster's data by get_chi2.
//
//   sizes/starts -> dv = 0
//     -> ss, gs, gg (like.* flags)
//     -> pair maps + cluster_warmup(), single-threaded, then
//        B(theta), T -> cg, N, cc, cs (cluster.probe_* flags)
// ---------------------------------------------------------------------------
vector compute_data_vector_cluster_masked()
{
  static constexpr std::string_view fname = "compute_data_vector_cluster_masked"sv;
  debug("{}: {}", fname, errbegins);

  IPCluster& ipc = IPCluster::get_instance();
  if (!ipc.is_mask_set()) [[unlikely]] {
    critical(errornset, fname, "mask (init_data_cluster)");
    exit(1);
  }
  check_cluster_state(fname);

  const arma::Col<int>::fixed<cluster_block::count> sizes =
    compute_data_vector_cluster_sizes();
  const arma::Col<int>::fixed<cluster_block::count> start =
    compute_data_vector_cluster_starts();
  const int ndata = arma::accu(sizes);
  if (ndata != ipc.get_ndata()) [[unlikely]] {
    critical("{}: {} (data vector size {} != mask size {})",
      fname, errleii, ndata, ipc.get_ndata());
    exit(1);
  }

  vector dv(ndata, arma::fill::zeros);

  // --- 1. GALAXY BLOCKS ---

  if (1 == like.probe[PROBE_SS]) {
    compute_ss_block_masked(dv, start(cluster_block::ss));
  }
  if (1 == like.probe[PROBE_GS]) {
    compute_gs_block_masked(dv, start(cluster_block::gs));
  }
  if (1 == like.probe[PROBE_GG]) {
    compute_gg_block_masked(dv, start(cluster_block::gg));
  }

  // --- 2. CLUSTER BLOCKS ---

  if (1 == any_cluster_probe()) {
    // every lazily filled cluster table is built here, single-threaded,
    // before any threaded loop of a C function reads it
    warmup_cluster_pair_maps();
    cluster_warmup();

    const matrix selection = compute_cluster_selection_factor();

    if (1 == cluster.probe_cg) {
      compute_cg_block_masked(dv, start(cluster_block::cg), selection);
    }
    if (1 == cluster.probe_N) {
      compute_N_block_masked(dv, start(cluster_block::N));
    }
    if (1 == cluster.probe_cc) {
      compute_cc_block_masked(dv, start(cluster_block::cc), selection);
    }
    if (1 == cluster.probe_cs) {
      matrix T;
      if (1 == cluster.ytransform) {
        T = compute_cluster_ytransform_matrix();
      }
      compute_cs_block_masked(dv, start(cluster_block::cs), selection, T);
    }
  }

  debug("{}: {}", fname, errends);
  return dv;
}



// ============================================================================
// [SECTION] CLASS IPCluster MEMBER FUNCTIONS
// ============================================================================

void IPCluster::reset()
{
  this->is_mask_set_ = false;
  this->is_data_set_ = false;
  this->is_inv_cov_set_ = false;
  this->ndata_ = 0;
  this->ndata_sqzd_ = 0;
  this->mask_filename_.clear();
  this->cov_filename_.clear();
  this->data_filename_.clear();
  this->mask_.reset();
  this->index_sqzd_.reset();
  this->data_masked_.reset();
  this->cov_masked_.reset();
  this->inv_cov_masked_.reset();
  this->data_masked_sqzd_.reset();
  this->cov_masked_sqzd_.reset();
  this->inv_cov_masked_sqzd_.reset();
}


// ---------------------------------------------------------------------------
// Build the mask of the joint vector and the full <-> sqzd index map.
//
// Stages:
//   read column 1 of the mask file (one row per joint-vector entry)
//     -> validate: every entry is 0 or 1
//     -> zero the blocks whose probe is off (like.* for ss/gs/gg,
//        cluster.probe_* for cg/N/cc/cs)
//     -> with the Y transform, zero the last theta bin of every cs row:
//        the last row of T vanishes, so model, data and covariance are
//        identically zero there
//     -> ndata_sqzd_ = number of surviving 1s (> 0)
//     -> index_sqzd_(i) = running count of 1s before i, -1 if masked
// ---------------------------------------------------------------------------
void IPCluster::set_mask(std::string mask_filename)
{
  static constexpr std::string_view fname = "IPCluster::set_mask"sv;
  debug("{}: {}", fname, errbegins);

  const arma::Col<int>::fixed<cluster_block::count> sizes =
    compute_data_vector_cluster_sizes();
  const arma::Col<int>::fixed<cluster_block::count> start =
    compute_data_vector_cluster_starts();

  this->ndata_ = arma::accu(sizes);
  if (!(this->ndata_ > 0)) [[unlikely]] {
    critical(errornset, fname, "data-vector size");
    exit(1);
  }
  this->mask_filename_ = mask_filename;
  this->mask_.set_size(this->ndata_);

  // --- 1. READ AND VALIDATE ---

  matrix table = read_table(mask_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_) [[unlikely]] {
    critical("{}: mask file {} has {} rows (joint data vector: {})",
      fname, mask_filename, table.n_rows, this->ndata_);
    exit(1);
  }
  if (table.n_cols < 2) [[unlikely]] {
    critical("{}: mask file {} needs two columns (index, mask)",
      fname, mask_filename);
    exit(1);
  }
  for (int i=0; i<this->ndata_; i++) {
    this->mask_(i) = static_cast<int>(table(i,1) + mask_rounding_guard);
    if (!(0 == this->mask_(i) || 1 == this->mask_(i))) [[unlikely]] {
      critical("{}: inconsistent mask (entry {} = {})", fname, i, table(i,1));
      exit(1);
    }
  }

  // --- 2. ZERO THE BLOCKS OF DISABLED PROBES ---

  arma::Col<int>::fixed<cluster_block::count> enabled;
  enabled(cluster_block::ss) = like.probe[PROBE_SS];
  enabled(cluster_block::gs) = like.probe[PROBE_GS];
  enabled(cluster_block::gg) = like.probe[PROBE_GG];
  enabled(cluster_block::cg) = cluster.probe_cg;
  enabled(cluster_block::N)  = cluster.probe_N;
  enabled(cluster_block::cc) = cluster.probe_cc;
  enabled(cluster_block::cs) = cluster.probe_cs;
  for (int b=0; b<cluster_block::count; b++) {
    if (0 == enabled(b)) {
      for (int i=start(b); i<start(b) + sizes(b); i++) {
        this->mask_(i) = 0;
      }
    }
  }

  // --- 3. Y SPACE: THE LAST THETA BIN OF EVERY cs ROW ---

  if (1 == cluster.probe_cs && 1 == cluster.ytransform) {
    const int ntheta = Ntable.Ntheta;
    const int nrows = cluster.cs_npowerspectra*cluster.richness_nbin;
    int nforced = 0;
    for (int row=0; row<nrows; row++) {
      const int index = start(cluster_block::cs) + ntheta*row + (ntheta - 1);
      nforced += this->mask_(index);
      this->mask_(index) = 0;
    }
    if (nforced > 0) {
      info("{}: masked the last theta bin of {} cs rows (zero in Y space)",
        fname, nforced);
    }
  }

  // --- 4. SQUEEZED INDEX MAP ---

  this->ndata_sqzd_ = arma::accu(this->mask_);
  if (!(this->ndata_sqzd_ > 0)) [[unlikely]] {
    critical("{}: mask file {} left no data points after masking",
      fname, mask_filename);
    exit(1);
  }
  this->index_sqzd_.set_size(this->ndata_);
  int j = 0;
  for (int i=0; i<this->ndata_; i++) {
    if (this->mask_(i) > 0) {
      this->index_sqzd_(i) = j;
      j++;
    }
    else {
      this->index_sqzd_(i) = -1;
    }
  }
  if (j != this->ndata_sqzd_) [[unlikely]] {
    critical("{}: {} mask operation", fname, errleii);
    exit(1);
  }

  // the data and covariance of a previous mask are no longer valid
  this->is_data_set_ = false;
  this->is_inv_cov_set_ = false;
  this->is_mask_set_ = true;

  debug("{}: mask file {} left {} non-masked elements",
    fname, mask_filename, this->ndata_sqzd_);
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Load the measured joint vector (column 1 of the file), zeroed off-mask,
// in both layouts. Requires set_mask.
// ---------------------------------------------------------------------------
void IPCluster::set_data(std::string datavector_filename)
{
  static constexpr std::string_view fname = "IPCluster::set_data"sv;
  debug("{}: {}", fname, errbegins);
  if (!(this->is_mask_set_)) [[unlikely]] {
    critical(errornset, fname, "mask");
    exit(1);
  }
  this->data_filename_ = datavector_filename;

  matrix table = read_table(datavector_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_ ||
      table.n_cols < 2) [[unlikely]] {
    critical("{}: inconsistent data vector {} ({} rows, joint vector {})",
      fname, datavector_filename, table.n_rows, this->ndata_);
    exit(1);
  }

  this->data_masked_.set_size(this->ndata_);
  this->data_masked_sqzd_.set_size(this->ndata_sqzd_);
  for (int i=0; i<this->ndata_; i++) {
    this->data_masked_(i) = table(i,1)*this->mask_(i);
    if (1 == this->mask_(i)) {
      this->data_masked_sqzd_(this->index_sqzd_(i)) = this->data_masked_(i);
    }
  }
  this->is_data_set_ = true;
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// Load, mask and invert the covariance of the joint vector.
//
// Accepted formats (read_table columns): 3 = (i, j, cov); 4 = (i, j,
// gauss, non-gauss), summed; 10 = CosmoCov, cov = col 8 + col 9. Each
// stored element is mirrored; off-diagonal elements are zeroed when either
// index is masked, masked diagonals are kept (the IP recipe, so
// get_cov_masked shows the file's variances).
//
// Stages after assembly (see the class header for why the squeezed
// matrix is the one inverted):
//   squeeze to the unmasked entries -> positive diagonal check -> eig_sym
//   check of the CORRELATION matrix (every eigenvalue > 0) -> invert it and
//   rescale by the standard deviations -> expand the inverse to the
//   full layout (zero rows and columns at masked entries).
// ---------------------------------------------------------------------------
void IPCluster::set_inv_cov(std::string cov_filename)
{
  static constexpr std::string_view fname = "IPCluster::set_inv_cov"sv;
  debug("{}: {}", fname, errbegins);
  if (!(this->is_mask_set_)) [[unlikely]] {
    critical(errornset, fname, "mask");
    exit(1);
  }
  this->cov_filename_ = cov_filename;
  matrix table = read_table(cov_filename);

  // --- 1. COLUMNS OF THE FILE FORMAT ---

  const int ncols = static_cast<int>(table.n_cols);
  int col_a = -1;  // covariance = table(r, col_a) [+ table(r, col_b)]
  int col_b = -1;
  switch (ncols)
  {
    case 3:
    {
      col_a = 2;
      break;
    }
    case 4:
    {
      col_a = 2;
      col_b = 3;
      break;
    }
    case 10:
    {
      col_a = 8;
      col_b = 9;
      break;
    }
    default:
    {
      critical("{}: invalid format for cov file = {}", fname, cov_filename);
      exit(1);
    }
  }

  // --- 2. FULL MASKED MATRIX ---

  this->cov_masked_.zeros(this->ndata_, this->ndata_);
  for (int r=0; r<static_cast<int>(table.n_rows); r++) {
    const long j = std::lround(table(r,0));
    const long k = std::lround(table(r,1));
    if (j < 0 || k < 0 || j >= this->ndata_ || k >= this->ndata_) [[unlikely]] {
      critical("{}: cov file {} row {}: index ({}, {}) outside [0, {})",
        fname, cov_filename, r, j, k, this->ndata_);
      exit(1);
    }
    double value = table(r, col_a);
    if (col_b >= 0) {
      value += table(r, col_b);
    }
    if (j == k) {
      this->cov_masked_(j,k) = value;
    }
    else {
      const double masked = value*this->mask_(j)*this->mask_(k);
      this->cov_masked_(j,k) = masked;
      this->cov_masked_(k,j) = masked;
    }
  }

  // --- 3. SQUEEZE ---

  this->cov_masked_sqzd_.zeros(this->ndata_sqzd_, this->ndata_sqzd_);
  for (int i=0; i<this->ndata_; i++) {
    if (0 == this->mask_(i)) {
      continue;
    }
    for (int j=0; j<this->ndata_; j++) {
      if (1 == this->mask_(j)) {
        const int a = this->index_sqzd_(i);
        const int b = this->index_sqzd_(j);
        this->cov_masked_sqzd_(a,b) = this->cov_masked_(i,j);
      }
    }
  }

  // --- 4. CHECK AND INVERT THE SQUEEZED MATRIX ---

  for (int a=0; a<this->ndata_sqzd_; a++) {
    if (!(this->cov_masked_sqzd_(a,a) > 0.0)) [[unlikely]] {
      critical("{}: non-positive variance {} at unmasked squeezed entry {}",
        fname, this->cov_masked_sqzd_(a,a), a);
      exit(1);
    }
  }
  // The joint vector's variances span ~19 orders of magnitude (xi- ~1e-15,
  // counts ~1e4), so eigenvalues of the raw matrix carry round-off of order
  // 1e-16 * lambda_max (a positive-definite covariance showed raw
  // eigenvalues down to -1e-12 next to lambda_max = 7.6e3). The test and
  // the inversion run on the correlation matrix R = D^-1/2 C D^-1/2
  // (scale invariant, unit diagonal), and C^-1 = D^-1/2 R^-1 D^-1/2.
  const vector inv_sigma = 1.0/arma::sqrt(this->cov_masked_sqzd_.diag());
  const matrix corr = this->cov_masked_sqzd_ % (inv_sigma*inv_sigma.t());
  const vector eigvals = arma::eig_sym(corr);
  for (int a=0; a<this->ndata_sqzd_; a++) {
    if (!(eigvals(a) > 0.0)) [[unlikely]] {
      critical("{}: masked correlation matrix not positive definite "
        "(eigenvalue {} = {})", fname, a, eigvals(a));
      exit(1);
    }
  }
  matrix inv_corr;
  if (!arma::inv(inv_corr, corr)) [[unlikely]] {
    critical("{}: inversion of the masked correlation matrix failed", fname);
    exit(1);
  }
  this->inv_cov_masked_sqzd_ = inv_corr % (inv_sigma*inv_sigma.t());

  // --- 5. EXPAND THE INVERSE ---

  this->inv_cov_masked_.zeros(this->ndata_, this->ndata_);
  for (int i=0; i<this->ndata_; i++) {
    if (0 == this->mask_(i)) {
      continue;
    }
    for (int j=0; j<this->ndata_; j++) {
      if (1 == this->mask_(j)) {
        const int a = this->index_sqzd_(i);
        const int b = this->index_sqzd_(j);
        this->inv_cov_masked_(i,j) = this->inv_cov_masked_sqzd_(a,b);
      }
    }
  }

  this->is_inv_cov_set_ = true;
  debug("{}: {}", fname, errends);
}


// ---------------------------------------------------------------------------
// chi2 = delta^T C^-1 delta on the squeezed views,
// delta = sqzd(theory) - data.
// ---------------------------------------------------------------------------
double IPCluster::get_chi2(vector datavector) const
{
  static constexpr std::string_view fname = "IPCluster::get_chi2"sv;
  debug("{}: {}", fname, errbegins);
  if (!(this->is_data_set_)) [[unlikely]] {
    critical(errornset, fname, "data_vector");
    exit(1);
  }
  if (!(this->is_inv_cov_set_)) [[unlikely]] {
    critical(errornset, fname, "inv_cov");
    exit(1);
  }
  if (static_cast<int>(datavector.n_elem) != this->ndata_) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, datavector.n_elem, this->ndata_);
    exit(1);
  }
  const vector delta = this->sqzd_theory_data_vector(datavector) -
                       this->data_masked_sqzd_;
  const double chi2 = arma::dot(delta, this->inv_cov_masked_sqzd_*delta);
  if (chi2 < 0.0) [[unlikely]] {
    critical("{}: chi2 = {} (invalid)", fname, chi2);
    exit(1);
  }
  debug("{}: {}", fname, errends);
  return chi2;
}


// Scatter a squeezed vector back to the full layout (zeros off-mask).
vector IPCluster::expand_theory_data_vector_from_sqzd(vector input) const
{
  static constexpr std::string_view fname =
    "IPCluster::expand_theory_data_vector_from_sqzd"sv;
  if (this->ndata_sqzd_ != static_cast<int>(input.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, input.n_elem, this->ndata_sqzd_);
    exit(1);
  }
  vector result(this->ndata_, arma::fill::zeros);
  for (int i=0; i<this->ndata_; i++) {
    if (1 == this->mask_(i)) {
      result(i) = input(this->index_sqzd_(i));
    }
  }
  return result;
}


// Compact a full-layout vector to its unmasked entries.
vector IPCluster::sqzd_theory_data_vector(vector input) const
{
  static constexpr std::string_view fname =
    "IPCluster::sqzd_theory_data_vector"sv;
  if (this->ndata_ != static_cast<int>(input.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, input.n_elem, this->ndata_);
    exit(1);
  }
  vector result(this->ndata_sqzd_, arma::fill::zeros);
  for (int i=0; i<this->ndata_; i++) {
    if (1 == this->mask_(i)) {
      result(this->index_sqzd_(i)) = input(i);
    }
  }
  return result;
}

}  // namespace cosmolike_interface
