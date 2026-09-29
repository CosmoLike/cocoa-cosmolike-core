#include "cosmolike/generic_interface.hpp"
#include <string_view>
using namespace std::literals; // enables "sv" literal

// Python Binding
namespace py = pybind11;

// boost library
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/lexical_cast.hpp>

// std::isnan: no compile w/ -O3 or -fast-math stackoverflow.com/a/47703550/2472169

static constexpr std::string_view errbegins = "Begins Execution"sv;
static constexpr std::string_view errends = "Ends Execution"sv;
static constexpr std::string_view errleii = "logical error, internal inconsistent"sv;
static constexpr std::string_view erriiwz = "incompatible input vector with size = "sv;
static constexpr std::string_view errnanit = "NaN found on interpolation table"sv;
static constexpr std::string_view errnance = "common error if `params_values.get(p, None)` return None"sv;
static constexpr std::string_view errnance2 = "{}: NaN found on index {} ({})."sv;
static constexpr std::string_view errorns = "{}: {}={} not supported (max={})"sv;
static constexpr std::string_view errorns2 = "{}: {} = {} not supported"sv;
static constexpr std::string_view debugsel = "{}: {} = {} selected."sv;
static constexpr std::string_view errornset = "{}: {} not set (?ill-defined) prior to this function call"sv;
static constexpr std::string_view errorsz1d = "{}: {} {} (!= {})"sv;
static constexpr std::string_view erroric0 ="{}: {} incompatible input"sv;

static const int force_cache_update_test = 0;

using vector = arma::Col<double>;
using matrix = arma::Mat<double>;
using cube = arma::Cube<double>;
using spdlog::info;
using spdlog::debug;
using spdlog::critical;
// Interface functions take and return arma types (arma::Col, arma::Mat,
// arma::Cube); the carma headers pulled in by generic_interface.hpp
// convert them to and from numpy arrays at the pybind11 boundary.

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// AUX FUNCTIONS (PRIVATE)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Parse one whitespace-trimmed token as a double, accepting underflow and
// rejecting overflow.
//
// Cosmolike covariance and data-vector tables routinely contain entries
// well below DBL_MIN (~2.2e-308), e.g. deep-tail covariance off-diagonals
// or noise-model entries pre-multiplied by tiny prefactors. Such values are
// numerically fine: they round to a subnormal or to 0.0, which is exactly
// what the downstream Cholesky / matrix-vector code expects. std::stod and
// std::stold throw std::out_of_range on any ERANGE, with no distinction
// between
//
//     (a) overflow  -> result is +/-HUGE_VAL, truly unusable, rejected here;
//     (b) underflow -> result is a finite subnormal or 0.0, perfectly safe;
//
// under which a table holding a legitimate 1e-320 aborts the run before a
// single likelihood is evaluated. std::strtod keeps the two cases apart:
// errno is set on either range error, and std::isfinite(v) is false for
// +/-HUGE_VAL and NaN but true for every normal, subnormal and zero value,
// so only the non-finite branch is treated as fatal.
//
// Implementation notes:
//   - errno must be cleared before the call. strtod sets errno on range
//     errors but never clears it, so a stale ERANGE from elsewhere would
//     otherwise be misattributed to this parse.
//   - end == tok.c_str() is the canonical "no digits consumed" check;
//     this catches empty tokens, pure whitespace, and garbage like "abc".
//   - Tokens like "nan" parse successfully and are not finite, so they are
//     rejected here -- the right outcome for table data.
//
// Error behavior: throws std::runtime_error when the token contains no
// numeric prefix (strtod consumes zero characters) or on overflow
// (errno == ERANGE with a non-finite result).
//
// Parameters:
//   tok - the token to parse
//
// Returns:
//   the parsed double; underflowed input comes back as subnormal or 0.0
// ---------------------------------------------------------------------------
double parse_double_or_throw(const std::string& tok) {
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(tok.c_str(), &end);

  if (end == tok.c_str()) {
    throw std::runtime_error(
      fmt::format("read_table: cannot parse '{}' as double", tok));
  }
  if (errno == ERANGE && !std::isfinite(v)) {
    throw std::runtime_error(
      fmt::format("read_table: range error parsing '{}' (non-finite)", tok));
  }
  return v;
}

// ---------------------------------------------------------------------------
// Read a whitespace-delimited numeric ASCII table into an arma matrix.
//
// Stages:
//   1. Slurp the whole file into one string (single read).
//   2. Split into lines; drop lines starting with "#".
//   3. Tokenize the first line to fix the column count, then parse the
//      remaining lines in parallel (OpenMP) with parse_double_or_throw.
//
// Validation / error behavior: critical() + exit(1) when the file cannot be
// opened, is empty, or a row disagrees with the first-row column count;
// parse_double_or_throw throws std::runtime_error on unparsable or
// overflowing tokens (underflow to subnormal/zero is accepted).
//
// Parameters:
//   file_name - path of the ASCII table
//
// Returns:
//   (nrows x ncols) matrix of the parsed values
// ---------------------------------------------------------------------------
arma::Mat<double> read_table(const std::string file_name)
{
  std::ifstream input_file(file_name);
  if (!input_file.is_open()) {
    critical("{}: file {} cannot be opened", "read_table", file_name);
    exit(1);
  }

  // --------------------------------------------------------
  // Read the entire file into memory
  // --------------------------------------------------------

  std::string tmp;
  
  input_file.seekg(0,std::ios::end);
  
  tmp.resize(static_cast<size_t>(input_file.tellg()));
  
  input_file.seekg(0,std::ios::beg);
  
  input_file.read(&tmp[0],tmp.size());
  
  input_file.close();
  
  if (tmp.empty())
  {
    critical("{}: file {} is empty", "read_table", file_name);
    exit(1);
  }
  
  // --------------------------------------------------------
  // Second: Split file into lines
  // --------------------------------------------------------
  
  std::vector<std::string> lines;
  lines.reserve(50000);

  boost::trim_if(tmp, boost::is_any_of("\t "));
  
  boost::trim_if(tmp, boost::is_any_of("\n"));
  
  boost::split(lines, tmp,boost::is_any_of("\n"), boost::token_compress_on);
  
  // Erase comment/blank lines
  auto check = [](std::string mystr) -> bool
  {
    return boost::starts_with(mystr, "#");
  };
  lines.erase(std::remove_if(lines.begin(), lines.end(), check), lines.end());
  
  // --------------------------------------------------------
  // Third: Split line into words
  // --------------------------------------------------------

  arma::Mat<double> result;
  size_t ncols = 0;
  
  { // first line
    std::vector<std::string> words;
    words.reserve(100);
    
    boost::trim_left(lines[0]);
    boost::trim_right(lines[0]);

    boost::split(
      words,lines[0], 
      boost::is_any_of(" \t"),
      boost::token_compress_on
    );
    
    ncols = words.size();

    result.set_size(lines.size(), ncols);
    
    for (size_t j=0; j<ncols; j++)
      result(0,j) = parse_double_or_throw(words[j]);
  }

  #pragma omp parallel for schedule(static)
  for (size_t i=1; i<lines.size(); i++)
  {
    std::vector<std::string> words;
    
    boost::trim_left(lines[i]);
    boost::trim_right(lines[i]);

    boost::split(
      words, 
      lines[i], 
      boost::is_any_of(" \t"),
      boost::token_compress_on
    );
    
    if (words.size() != ncols)
    {
      critical("{}: file {} is not well formatted"
                       " (regular table required)", 
                       "read_table", 
                       file_name
                      );
      exit(1);
    }
    
    for (size_t j=0; j<ncols; j++)
      result(i,j) = parse_double_or_throw(words[j]);
  };
  
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Normalize a user-facing baryon-simulation label into (name, tag).
//
// Stages: trim and lowercase; restore the canonical family spelling
// (owls_AGN, BAHAMAS, HzAGN, TNG) and translate the legacy temperature
// suffixes (_t80/_t85/_t87, _t78/_t76) into numeric tags; split at the last
// "-" into name and integer tag; a label without "-" gets tag = 1.
//
// Validation: more than one "-" is rejected with critical() + exit(1)
// (the two-dash range syntax is expanded earlier, in
// BaryonScenario::set_scenarios).
//
// Parameters:
//   sim - scenario label from python, case-insensitive (e.g. "owls_agn_t80")
//
// Returns:
//   std::tuple(name, tag), e.g. ("owls_AGN", 1)
// ---------------------------------------------------------------------------
std::tuple<std::string,int> get_baryon_sim_name_and_tag(std::string sim)
{
  static constexpr std::string_view fname = "get_baryon_sim_name_and_tag"sv;
  // Desired Convention:
  // (1) Python input: not be case sensitive
  // (2) simulation names only have "_" as deliminator, e.g., owls_AGN.
  // (3) simulation IDs are indicated by "-", e.g., antilles-1.
 
  boost::trim_if(sim, boost::is_any_of("\t "));
  sim = boost::algorithm::to_lower_copy(sim);
  
  { // Count occurrences of - (dashes)
    size_t pos = 0; 
    size_t count = 0; 
    std::string tmp = sim;
    while ((pos = tmp.rfind("-")) != std::string::npos) {
      tmp = tmp.substr(0, pos);
      count++;
    }
    if (count > 1) {
      critical("{}: Scenario {} not supported (too many dashes)", fname, sim);
      exit(1);
    }
  }

  if (sim.rfind("owls_agn") != std::string::npos) {
    boost::replace_all(sim, "owls_agn", "owls_AGN");
    boost::replace_all(sim, "_t80", "-1");
    boost::replace_all(sim, "_t85", "-2");
    boost::replace_all(sim, "_t87", "-3");
  } 
  else if (sim.rfind("bahamas") != std::string::npos) {
    boost::replace_all(sim, "bahamas", "BAHAMAS");
    boost::replace_all(sim, "_t78", "-1");
    boost::replace_all(sim, "_t76", "-2");
    boost::replace_all(sim, "_t80", "-3");
  } 
  else if (sim.rfind("hzagn") != std::string::npos) {
    boost::replace_all(sim, "hzagn", "HzAGN");
  }
  else if (sim.rfind("tng") != std::string::npos) {
    boost::replace_all(sim, "tng", "TNG");
  }
  
  std::string name;
  int tag;
  if (sim.rfind('-') != std::string::npos) {
    const size_t pos = sim.rfind('-');
    name = sim.substr(0, pos);
    tag = std::stoi(sim.substr(pos + 1));
  } 
  else { 
    name = sim;
    tag = 1; 
  }

  return std::make_tuple(name, tag);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INIT FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// How a likelihood option reaches the C kernels (representative rows;
// each project's interface.cpp binds the python names):
//
//   yaml option      -> init_* here            -> C struct field
//                                              -> consumer
//   probe            -> init_probes            -> like.shear_shear..kk
//                       -> gates every Mx2pt block (the hpp templates)
//   n_theta, theta_* -> init_binning_real_space-> Ntable.Ntheta/vtmin/
//                       vtmax -> real-space kernels (cosmo2D.c)
//   accuracyboost    -> init_accuracy_boost    -> Ntable.N_a/N_ell/...
//                       -> every interpolation-table resolution
//   lmax             -> init_ntable_lmax       -> Ntable.LMAX
//                       -> Legendre sums (cosmo2D.c)
//   mask/cov/data    -> init_data_Mx2pt_N      -> IP singleton
//                       -> IP::get_chi2
//   lens/source file -> init_redshift_distributions_from_files
//                       -> redshift.*_zdist_table -> redshift_spline.c
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reset every Cosmolike global struct to a defined startup state.
//
// Zeroes the probe flags (like.shear_shear/shear_pos/pos_pos, gk/kk/ks),
// the Fourier binning (like.Ncl/lmin/lmax) and the cluster flags, then runs
// the reset_*_struct family (redshift, nuisance, cosmology, tomo, Ntable,
// like, cmb). Afterwards sets the defaults like.adopt_limber_gg = 0,
// like.adopt_limber_gs = 1 and pdeltaparams.runmode = "Halofit", and loads
// the spdlog verbosity from the environment (SPDLOG_LEVEL).
//
// Runs once, before any other init_/set_ call, so later writes land on a
// defined state.
//
// Parameters:
//   (none)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void initial_setup()
{
  static constexpr std::string_view fname = "initial_setup"sv;
  spdlog::cfg::load_env_levels();
  debug("{}: {}", fname, errbegins);

  like.shear_shear = 0;
  like.shear_pos = 0;
  like.pos_pos = 0;

  like.Ncl = 0;
  like.lmin = 0;
  like.lmax = 0;

  like.gk = 0;
  like.kk = 0;
  like.ks = 0;
  
  // cluster probes off (this interface carries no cluster likelihood)
  like.clusterN = 0;
  like.clusterWL = 0;
  like.clusterCG = 0;
  like.clusterCC = 0;

  // reset bias - pretty important to setup variables to zero or 1 via reset
  reset_redshift_struct();
  reset_nuisance_struct();
  reset_cosmology_struct();
  reset_tomo_struct();
  reset_Ntable_struct();
  reset_like_struct();
  reset_cmb_struct();

  like.adopt_limber_gg = 0;
  like.adopt_limber_gs = 1;

  std::string mode = "Halofit";
  memcpy(pdeltaparams.runmode, mode.c_str(), mode.size() + 1);
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set Ntable.LMAX, the highest multipole of the 2D projection tables, and
// bump Ntable.random so every table keyed on it rebuilds.
//
// Parameters:
//   lmax - new Ntable.LMAX
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_ntable_lmax(const int lmax) {
  static constexpr std::string_view fname = "init_ntable_lmax"sv;
  debug("{}: {}", fname, errbegins);
  Ntable.LMAX = lmax;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// Set the internal coarse ell grid of the Limber C_l tables.
//
// The C_ss, C_gs, C_gk and C_ks tables keep Ntable.N_ell nodes with
// linear interpolation (the real-space Legendre sums interpolate
// ~100k multipoles through the vectorized gather fill), but their
// construction cost is N_ell x N_pairs Limber quadratures. Those
// spectra are smooth in ln l, so the exact quadrature runs on this
// coarse grid and a cubic spline upsamples to the unchanged N_ell
// nodes at cache-build time. The same knob coarsens the ell axis of
// the dC_X/dlnk scale-cut tables (their ln k axis has its own knob,
// init_ntable_dcx_dlnk_nlnk_internal). 0 disables the trick (exact
// quadrature at every node): the A/B switch for validation. Galaxy
// clustering never uses it (BAO wiggles; see C_gg_tomo_limber).
//
//
// init_accuracy_boost (the catch-all) also scales this knob from its
// first-boost-call baseline; calling this setter afterwards
// overwrites the boosted value.
// Cache invalidation:
// bumps Ntable.random so every table rebuilds.
//
// Parameters:
//   nell_internal - coarse node count (4 <= n <= Ntable.N_ell), or 0
//                   for exact
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_ntable_ell_internal(const int nell_internal) {
  static constexpr std::string_view fname = "init_ntable_ell_internal"sv;
  debug("{}: {}", fname, errbegins);
  if (nell_internal != 0 &&
      (nell_internal < 4 || nell_internal > Ntable.N_ell)) [[unlikely]] {
    critical("{}: nell_internal = {} not 0 and outside [4, {}]",
             fname, nell_internal, Ntable.N_ell);
    exit(1);
  }
  Ntable.N_ell_internal = nell_internal;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// Set the internal coarse ln k grid of the dC_X/dlnk scale-cut tables.
//
// The (ln k, ln l) tables in cosmo2D_scuts.c cost one exact
// single-node Limber evaluation per entry. When a coarse axis is
// active, the exact evaluations run on the coarse nodes and a
// tensor-product bicubic (spline2d_upsample_uniform, basics.c) fills
// the unchanged dense table. This knob coarsens the ln k axis; the
// ell axis follows Ntable.N_ell_internal. The ln k direction carries
// the BAO wiggles of P(k); the default (128 of the 256-node grid)
// keeps the measured response error at or below what the retired
// fixed quadrature imposed (max |dRF| 5.9e-3, medians ~1e-6) at
// twice the refill speed. 0 = exact: the A/B switch.
//
//
// init_accuracy_boost (the catch-all) also scales this knob from its
// first-boost-call baseline; calling this setter afterwards
// overwrites the boosted value.
// Cache invalidation:
// bumps Ntable.random so every table rebuilds.
//
// Parameters:
//   nlnk_internal - coarse ln k node count
//                   (4 <= n <= Ntable.dCX_dlnk_nlnk), or 0 for exact
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_ntable_dcx_dlnk_nlnk_internal(const int nlnk_internal) {
  static constexpr std::string_view fname =
      "init_ntable_dcx_dlnk_nlnk_internal"sv;
  debug("{}: {}", fname, errbegins);
  if (nlnk_internal != 0 &&
      (nlnk_internal < 4 ||
       nlnk_internal > Ntable.dCX_dlnk_nlnk)) [[unlikely]] {
    critical("{}: nlnk_internal = {} not 0 and outside [4, {}]",
             fname, nlnk_internal, Ntable.dCX_dlnk_nlnk);
    exit(1);
  }
  Ntable.dCX_dlnk_nlnk_internal = nlnk_internal;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// Set the internal coarse mass grid of the sigma^2(M) halo-model table.
//
// sigma^2(M)'s cached table keeps Ntable.N_M nodes in ln M; when this
// knob is active the exact lobe-summed quadratures run on the coarse
// nodes only and the house cubic spline upsamples ln sigma^2 onto the
// unchanged dense table (ln sigma^2 is smooth and monotone in ln M).
// 0 disables the trick: the A/B switch for validation.
//
// init_accuracy_boost (the catch-all) also scales this knob from its
// first-boost-call baseline; calling this setter afterwards
// overwrites the boosted value.
//
// Cache invalidation:
// bumps Ntable.random so every table rebuilds.
//
// Parameters:
//   nm_internal - coarse node count (4 <= n <= Ntable.N_M), or 0 for
//                 exact
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_ntable_nm_internal(const int nm_internal) {
  static constexpr std::string_view fname = "init_ntable_nm_internal"sv;
  debug("{}: {}", fname, errbegins);
  if (nm_internal != 0 &&
      (nm_internal < 4 || nm_internal > Ntable.N_M)) [[unlikely]] {
    critical("{}: nm_internal = {} not 0 and outside [4, {}]",
             fname, nm_internal, Ntable.N_M);
    exit(1);
  }
  Ntable.N_M_internal = nm_internal;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// Diagnostic read of the halo-model mass variance sigma^2(M) at a = 1:
// the cached table (lobe-summed, and coarse-M upsampled when
// Ntable.N_M_internal is active). M in M_sun/h.
//
// Cache invalidation:
// none here; the cached table rebuilds on cosmology.random /
// Ntable.random as usual.
//
// Parameters:
//   M - halo mass in M_sun/h
//
// Returns:
//   sigma^2(M)
// ---------------------------------------------------------------------------
double compute_sigma2(const double M)
{
  return sigma2(M);
}


// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Select the n(z) ingestion conventions.
//
// Writes Ntable.photoz_interpolation_type, the n(z) stage-1 interpolant
// (0 = cspline, 1 = linear, 2+ = steffen), and
// Ntable.photoz_zmid_convention, the reading of the n(z) file z column
// (0 = Z_LOW left bin edges, values at cell centers z + dz/2; 1 = Z_MID
// sample points). The n(z) table caches watch both values, so a runtime
// change rebuilds the tables.
//
// Cache invalidation:
// bumps Ntable.random.
//
// Parameters:
//   interpolation_type - 0 = cspline, 1 = linear, 2+ = steffen
//   zmid_convention    - 0 = Z_LOW left edges, 1 = Z_MID sample points
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_photoz_conventions(
    const int interpolation_type,
    const int zmid_convention
  )
{
  static constexpr std::string_view fname = "init_photoz_conventions"sv;
  debug("{}: {}", fname, errbegins);
  Ntable.photoz_interpolation_type = interpolation_type;
  Ntable.photoz_zmid_convention = zmid_convention;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the C-FAST-PT internal (convolution) grid as a fraction of the output
// table: Ntable.FPT_internal_accuracy_boost = internal_boost. 1.0 keeps the
// two grids equal (the exact reference path); smaller values run the FFTLog
// convolutions on fewer points and cubic-spline upsample onto the output
// table (see pt_cfastpt.c: fpt_regrid).
//
// Cache invalidation:
// bumps Ntable.random.
//
// Validation: internal_boost > 0, else critical() + exit(1).
//
//
// init_accuracy_boost (the catch-all) also scales this knob from its
// first-boost-call baseline; calling this setter afterwards
// overwrites the boosted value.
// Parameters:
//   internal_boost - internal-grid fraction of the output grid (> 0)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_fpt_internal_boost(const double internal_boost)
{
  static constexpr std::string_view fname = "init_fpt_internal_boost"sv;
  debug("{}: {}", fname, errbegins);
  if (!(internal_boost > 0)) {
    critical("{}: invalid internal_boost = {}", fname, internal_boost);
    exit(1);
  }
  Ntable.FPT_internal_accuracy_boost = internal_boost;
  Ntable.random = RandomNumber::get_instance().get(); // update cache
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Choose the galaxy-galaxy lensing C_l^gs computation, writing
// like.adopt_limber_gs: 1 = Limber at every multipole (the default); 0 =
// the non-Limber C_gs_tomo below limits.LMAX_NOLIMBER, in gamma_t
// (w_gammat_tomo) and in the Fourier-space data vectors (C_gs_tomo_ells).
// Likelihood yaml key: adopt_limber_gs. Example: adopt_limber_gs: 0 in
// combo_3x2pt.yaml -> the likelihood calls init_adopt_limber_gs(0) and the
// next data vector uses the non-Limber path. No cache key is bumped here:
// w_gammat_tomo keys its cache on the flag itself.
//
// Validation: the value must be 0 or 1, else critical() + exit(1).
//
// Parameters:
//   adopt_limber_gs - 1 = Limber everywhere, 0 = non-Limber at low ell
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_adopt_limber_gs(const int adopt_limber_gs)
{
  static constexpr std::string_view fname = "init_adopt_limber_gs"sv;
  debug("{}: {}", fname, errbegins);
  if (adopt_limber_gs != 0 && adopt_limber_gs != 1) {
    critical("{}: invalid adopt_limber_gs = {}", fname, adopt_limber_gs);
    exit(1);
  }
  like.adopt_limber_gs = adopt_limber_gs;
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Choose the galaxy clustering C_l^gg computation, writing
// like.adopt_limber_gg: 0 = the non-Limber C_cl_tomo below
// limits.LMAX_NOLIMBER (the default of the real-space projects), in
// w(theta) (w_gg_tomo) and in the Fourier-space data vectors
// (C_gg_tomo_ells); 1 = Limber at every multipole (the default of the
// Fourier-space projects, set in their likelihood yamls). Likelihood yaml
// key: adopt_limber_gg. Example: adopt_limber_gg: 1 in combo_3x2pt.yaml of
// lsst_y1 -> the likelihood calls init_adopt_limber_gg(1) and the next
// data vector uses Limber w(theta). No cache key is bumped here: w_gg_tomo
// keys its cache on the flag itself.
//
// Validation: the value must be 0 or 1, else critical() + exit(1).
//
// Parameters:
//   adopt_limber_gg - 0 = non-Limber at low ell, 1 = Limber everywhere
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_adopt_limber_gg(const int adopt_limber_gg)
{
  static constexpr std::string_view fname = "init_adopt_limber_gg"sv;
  debug("{}: {}", fname, errbegins);
  if (adopt_limber_gg != 0 && adopt_limber_gg != 1) {
    critical("{}: invalid adopt_limber_gg = {}", fname, adopt_limber_gg);
    exit(1);
  }
  like.adopt_limber_gg = adopt_limber_gg;
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scale every Cosmolike sampling knob with one accuracy boost (the
// catch-all: a single call shifts them all).
//
// The first call snapshots the incoming values of the knobs below in
// a static cache; every call rescales from those baselines, so
// repeated calls do not compound:
//
//   Ntable.N_a                     -> ceil(baseline * boost)
//   Ntable.N_ell                   -> ceil(baseline * boost)
//   Ntable.N_ell_internal          -> ceil(baseline * boost)
//   Ntable.dCX_dlnk_nlnk           -> ceil(baseline * boost)
//   Ntable.dCX_dlnk_nlnk_internal  -> ceil(baseline * boost)
//   Ntable.N_M_internal            -> ceil(baseline * boost)
//   Ntable.halo_uks_nc             -> ceil(baseline * boost)
//   Ntable.halo_uks_nz             -> ceil(baseline * boost)
//   Ntable.halo_nfw_n              -> ceil(baseline * boost)
//   Ntable.NL_Nchi                 -> ceil(baseline * boost)
//   Ntable.nz_fine_sampling_factor -> ceil(baseline * boost)
//   Ntable.FPT_internal_accuracy_boost -> baseline * boost (double)
//
// The internal coarse grids scale together with their dense tables,
// so the coarse/dense ratios are boost-invariant, and a knob whose
// baseline is 0 (disabled) stays 0 under any boost. The dedicated setters (init_ntable_ell_internal,
// init_ntable_dcx_dlnk_nlnk_internal, init_fpt_internal_boost) remain
// for individual overrides: called BEFORE the first boost call they
// define the baseline, called after they overwrite the boosted value.
//
// Also writes Ntable.FPTboost (int(boost - 1) for boost > 1, else 0;
// enlarges the FAST-PT grids in pt_cfastpt.c) and
// Ntable.high_def_integration = integration_accuracy (selects larger
// fixed-order quadrature tables downstream).
//
// Cache invalidation:
// bumps Ntable.random so all tables keyed on it rebuild.
//
// Parameters:
//   accuracy_boost       - multiplier on the baseline table sizes (ceil)
//   integration_accuracy - written to Ntable.high_def_integration
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_accuracy_boost(
    const double accuracy_boost,
    const int integration_accuracy
  )
{
  static constexpr std::string_view fname = "init_accuracy_boost"sv;
  static int cache[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS]; // standard: static vars init to zero
  static double fptcache = 0.0; // FPT_internal_accuracy_boost baseline
  debug("{}: {}", fname, errbegins);

  if (0 == cache[0]) cache[0] = Ntable.N_a;
  Ntable.N_a = static_cast<int>(ceil(cache[0]*accuracy_boost));
  
  if (0 == cache[1]) cache[1] = Ntable.N_ell;
  Ntable.N_ell = static_cast<int>(ceil(cache[1]*accuracy_boost));

  if (0 == cache[2]) cache[2] = Ntable.dCX_dlnk_nlnk;
  Ntable.dCX_dlnk_nlnk = static_cast<int>(ceil(cache[2]*accuracy_boost));

  if (0 == cache[3]) cache[3] = Ntable.NL_Nchi;
  Ntable.NL_Nchi = static_cast<int>(ceil(cache[3]*accuracy_boost));

  if (0 == cache[4]) cache[4] = Ntable.nz_fine_sampling_factor;
  Ntable.nz_fine_sampling_factor = 
                                static_cast<int>(ceil(cache[4]*accuracy_boost));

  if (0 == cache[5]) cache[5] = Ntable.N_ell_internal;
  Ntable.N_ell_internal = static_cast<int>(ceil(cache[5]*accuracy_boost));

  if (0 == cache[6]) cache[6] = Ntable.dCX_dlnk_nlnk_internal;
  Ntable.dCX_dlnk_nlnk_internal =
      static_cast<int>(ceil(cache[6]*accuracy_boost));

  if (0 == cache[7]) cache[7] = Ntable.N_M_internal;
  Ntable.N_M_internal = static_cast<int>(ceil(cache[7]*accuracy_boost));

  if (0 == cache[8]) cache[8] = Ntable.halo_uks_nc;
  Ntable.halo_uks_nc = static_cast<int>(ceil(cache[8]*accuracy_boost));

  if (0 == cache[9]) cache[9] = Ntable.halo_uks_nz;
  Ntable.halo_uks_nz = static_cast<int>(ceil(cache[9]*accuracy_boost));

  if (0 == cache[10]) cache[10] = Ntable.halo_nfw_n;
  Ntable.halo_nfw_n = static_cast<int>(ceil(cache[10]*accuracy_boost));

  if (0 == fptcache) fptcache = Ntable.FPT_internal_accuracy_boost;
  Ntable.FPT_internal_accuracy_boost = fptcache*accuracy_boost;

  if (accuracy_boost>1) {
    Ntable.FPTboost = static_cast<int>(accuracy_boost-1.0);
  }
  else {
    Ntable.FPTboost = 0.0;
  }

  /*  
  Ntable.N_k_lin = 
    static_cast<int>(ceil(Ntable.N_k_lin*sampling_boost));
  
  Ntable.N_k_nlin = 
    static_cast<int>(ceil(Ntable.N_k_nlin*sampling_boost));

  Ntable.N_M = 
    static_cast<int>(ceil(Ntable.N_M*sampling_boost));
  */

  Ntable.high_def_integration = int(integration_accuracy);
  Ntable.random = RandomNumber::get_instance().get();
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Contaminate the matter power spectrum with a fixed baryon scenario from
// the library compiled into baryons.c. Parses sim with
// get_baryon_sim_name_and_tag and forwards "name-tag" to init_baryons,
// which fills the bary struct 2D interpolator read by PkRatio_baryons.
//
// Parameters:
//   sim - scenario label, case-insensitive (see get_baryon_sim_name_and_tag)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_baryons_contamination(std::string sim)
{
  static constexpr std::string_view fname = "init_baryons_contamination"sv;
  debug("{}: {}", fname, errbegins);
  auto [name, tag] = get_baryon_sim_name_and_tag(sim);
  debug("{}: Baryon simulation w/ Name = {} & Tag = {} selected",fname,name,tag);
  std::string tmp = name + "-" + std::to_string(tag);
  init_baryons(tmp.c_str());
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Overload of the above that reads the scenario from an HDF5 library file:
// forwards (name, tag, all_sims_file) to init_baryons_from_hdf5_file.
//
// Parameters:
//   sim           - scenario label, case-insensitive
//   all_sims_file - HDF5 library with the scenario suppression tables
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_baryons_contamination(std::string sim, std::string all_sims_file)
{
  static constexpr std::string_view fname = "init_baryons_contamination"sv;
  debug("{}: {}", fname, errbegins);
  auto [name, tag] = get_baryon_sim_name_and_tag(sim);
  debug("{}: Baryon simulation w/ Name = {} & Tag = {} selected", fname, name, tag);
  init_baryons_from_hdf5_file(name.c_str(), tag, all_sims_file.c_str());
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Select the redshift-evolution model of each galaxy-bias parameter.
//
// Writes like.galaxy_bias_model[i] (model codes, not amplitudes; bias.c
// dispatches on them, e.g. [0] = b1 evolution: B1_PER_BIN,
// B1_PER_BIN_EVOLV, B1_PER_BIN_PASS_EVOLV, B1_GROWTH_SCALING,
// B1_POWER_LAW). Amplitudes arrive separately via set_nuisance_*_bias.
// No cache key is bumped.
//
// Validation: input size <= MAX_SIZE_ARRAYS and no NaN entries, else
// critical() + exit(1).
//
// Parameters:
//   bias_z_evol_model - evolution-model code per bias slot (layout in the
//                       body comment); size <= MAX_SIZE_ARRAYS
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_bias(vector bias_z_evol_model)
{
  static constexpr std::string_view fname = "init_bias"sv;
  debug("{}: {}", fname, errbegins);
  const int nsz = static_cast<int>(bias_z_evol_model.n_elem);
  if (MAX_SIZE_ARRAYS < nsz) [[unlikely]] {
    critical("{}: {} = {:d} (>{:d})", fname, erriiwz, nsz, MAX_SIZE_ARRAYS);
    exit(1);
  }
  // like.galaxy_bias_model slot layout (parallel to the nuisance.gb
  // rows; bias.c dispatches gb1/gb2/gbs2/gb3/gbmag/gbK on these):
  //   [0] = b1    linear bias
  //   [1] = b2    quadratic bias
  //   [2] = bs2   tidal bias
  //   [3] = b3    third-order bias
  //   [4] = bmag  magnification bias
  //   [5] = bK    nonlocal bias
  for(int i=0; i<nsz; i++) {
    if (std::isnan(bias_z_evol_model(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    const double bias = bias_z_evol_model(i);
    like.galaxy_bias_model[i] = bias;
    debug("{}: {}[{}] = {} selected.", fname, "like.galaxy_bias_model", i, bias);
  }
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Define the Fourier-space band powers of the data vector.
//
// Writes like.Ncl, like.lmin, like.lmax, like.lmax_shear and reallocates
// like.ell with the Ncl log-spaced bin centers
//   ell_i = exp(ln(lmin) + (i + 0.5) dlnl),  dlnl = ln(lmax/lmin)/Ncl.
// No cache key is bumped.
//
// Validation: nells > 0, else critical() + exit(1).
//
// Parameters:
//   nells      - number of band powers (> 0), written to like.Ncl
//   lmin       - lowest multipole (like.lmin)
//   lmax       - highest multipole (like.lmax)
//   lmax_shear - highest shear-shear multipole (like.lmax_shear)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_binning_fourier(
    const int nells,
    const int lmin,
    const int lmax,
    const int lmax_shear
  )
{
  static constexpr std::string_view fname = "init_binning_fourier"sv;
  debug("{}: {}", fname, errbegins);
  if (!(nells > 0)) [[unlikely]] {
    critical(errorns2, fname, "Number of l modes (nells)", nells);
    exit(1);
  }
  debug(debugsel, fname, "nells", nells);
  debug(debugsel, fname, "l_min", lmin);
  debug(debugsel, fname, "l_max", lmax);
  debug(debugsel, fname, "l_max_shear", lmax_shear);

  like.Ncl = nells;
  like.lmin = lmin;
  like.lmax = lmax;
  like.lmax_shear = lmax_shear;
  
  const double logdl = (std::log(lmax) - std::log(lmin))/ (double) like.Ncl;
  if (like.ell != NULL) {
    free(like.ell);
  }
  like.ell = (double*) malloc(sizeof(double)*like.Ncl);
  
  for (int i=0; i<like.Ncl; i++) {
    like.ell[i] = std::exp(std::log(like.lmin) + (i + 0.5)*logdl);
    /*debug(
        "{}: Bin {:d}, {} = {:d}, {} = {:d} and {} = {:d}",
        "init_binning_fourier",
        i,
        "lmin",
        lmin,
        "ell",
        like.ell[i],
        "lmax",
        lmax
      );*/
  }
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Define the real-space angular binning of the data vector.
//
// Writes Ntable.Ntheta and the angular range Ntable.vtmin/vtmax (input in
// arcmin, stored in rad). Bin centers derive from these in
// compute_binning_real_space and in the real-space projections (cosmo2D.c).
// No cache key is bumped.
//
// Validation: Ntheta > 0, else critical() + exit(1).
//
// Parameters:
//   Ntheta           - number of angular bins (> 0)
//   theta_min_arcmin - lower edge of the angular range (arcmin)
//   theta_max_arcmin - upper edge of the angular range (arcmin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_binning_real_space(
    const int Ntheta,
    const double theta_min_arcmin,
    const double theta_max_arcmin
  )
{
  static constexpr std::string_view fname = "init_binning_real_space"sv;
  debug("{}: {}", fname, errbegins);
  if (!(Ntheta > 0)) [[unlikely]] {
    critical(errorns2, fname, "Ntheta", Ntheta);
    exit(1);
  }
  debug(debugsel, fname, "Ntheta", Ntheta);
  debug(debugsel, fname, "theta_min_arcmin", theta_min_arcmin);
  debug(debugsel, fname, "theta_max_arcmin", theta_max_arcmin);
  Ntable.Ntheta = Ntheta;
  Ntable.vtmin  = theta_min_arcmin * 2.90888208665721580e-4; // arcmin to rad conv
  Ntable.vtmax  = theta_max_arcmin * 2.90888208665721580e-4; // arcmin to rad conv  
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Configure the w_xk real-space cross correlation with CMB lensing through
// the IPCMB singleton (a front end to the C global struct cmb).
//
// Writes cmb.fwhm (input beam fwhm in arcmin, stored in rad),
// cmb.lmink_wxk/lmaxk_wxk, and the tabulated HealPix window
// (cmb.healpixwin = column 1 of healpixwin_filename).
//
// Cache invalidation:
// bumps cmb.random so tables keyed on it rebuild.
//
// Parameters:
//   lmin                - lowest multipole of the w_xk sum (cmb.lmink_wxk)
//   lmax                - highest multipole of the w_xk sum (cmb.lmaxk_wxk)
//   fwhm                - CMB beam fwhm (arcmin; stored in rad)
//   healpixwin_filename - table whose column 1 is the HealPix window
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_cmb_cross_correlation (
    const int lmin,
    const int lmax,
    const double fwhm, // fwhm = beam size in arcmin
    std::string healpixwin_filename
  )
{
  static constexpr std::string_view fname = "init_cmb_cross_correlation"sv;
  debug("{}: {}", fname, errbegins);
  IPCMB& cmb = IPCMB::get_instance();
  // fwhm = beam size in arcmin - cmb.fwhm = beam size in rad
  cmb.set_wxk_beam_size(fwhm*2.90888208665721580e-4);
  cmb.set_wxk_lminmax(lmin, lmax);
  cmb.set_wxk_healpix_window(healpixwin_filename);
  cmb.update_cache(RandomNumber::get_instance().get());
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Configure the CMB lensing auto-spectrum (kk) bandpower compression
// through the IPCMB singleton.
//
// Writes cmb.nbp_kk/lminbp_kk/lmaxbp_kk, the nbins x (lmax - lmin + 1)
// binning matrix, the per-band theory offsets (zeros when theory_offset is
// an empty string) and the Hartlap alpha that IP::set_inv_cov applies to
// the kkkk covariance block.
//
// The Hartlap alpha debiases an inverse covariance estimated from a
// finite set of simulated realizations: the unbiased estimate is
// alpha * (sample cov)^-1 with
//
//   alpha = (N_sim - N_data - 2) / (N_sim - 1) < 1
//
// (Hartlap et al. 2007). The caller computes alpha; IP::set_inv_cov
// applies it by dividing the kkkk covariance block by alpha before the
// joint inversion, which scales that block of the inverse by alpha.
//
// Cache invalidation:
// bumps cmb.random so tables keyed on it rebuild.
//
// Parameters:
//   nbins          - number of kk band powers (> 0)
//   lmin           - lowest multipole entering the bands (> 0)
//   lmax           - highest multipole entering the bands (> 0)
//   binning_matrix - file with the nbins x (lmax - lmin + 1) matrix
//   theory_offset  - file with per-band offsets ("" = zeros)
//   alpha          - Hartlap alpha for the kkkk covariance block
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_cmb_auto_bandpower (
    const int nbins,
    const int lmin,
    const int lmax,
    std::string binning_matrix,
    std::string theory_offset,
    const double alpha
  )
{
  static constexpr std::string_view fname = "init_cmb_auto_bandpower"sv;
  debug("{}: Begins", fname);
  IPCMB& cmb = IPCMB::get_instance();
  cmb.set_kk_binning_bandpower(nbins, lmin, lmax);
  cmb.set_kk_binning_mat(binning_matrix);
  cmb.set_kk_theory_offset(theory_offset);
  cmb.set_alpha_Hartlap_cov_kkkk(alpha);
  cmb.update_cache(RandomNumber::get_instance().get());
  debug("{}: Ends", fname);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Select the matter power spectrum consumed downstream:
// pdeltaparams.runmode = "linear" or "Halofit" (string parsed in cosmo3D.c).
//
// Parameters:
//   is_linear - true = "linear", false = "Halofit"
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_cosmo_runmode(const bool is_linear)
{
  static constexpr std::string_view fname = "init_cosmo_runmode"sv;
  debug("{}: {}", fname, errbegins);
  std::string mode = is_linear ? "linear" : "Halofit";
  const size_t size = mode.size();
  memcpy(pdeltaparams.runmode, mode.c_str(), size + 1);
  debug(debugsel, fname, "runmode", mode);
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Select the intrinsic-alignment model.
//
// Writes nuisance.IA_MODEL (0 = IA_MODEL_NLA, 1 = IA_MODEL_TATT),
// nuisance.IA (redshift dependence: NO_IA, IA_NLA_LF, IA_REDSHIFT_BINNING
// or IA_REDSHIFT_EVOLUTION, see IA.h) and nuisance.IA_code (0 = CFASTPT,
// 1 = python FAST-PT fed through set_IA_PS). No cache key is bumped.
//
// Validation: values outside these sets are critical() + exit(1).
//
// Parameters:
//   IA_MODEL         - 0 = IA_MODEL_NLA, 1 = IA_MODEL_TATT
//   IA_REDSHIFT_EVOL - NO_IA, IA_NLA_LF, IA_REDSHIFT_BINNING or
//                      IA_REDSHIFT_EVOLUTION
//   IA_code          - 0 = CFASTPT, 1 = python FAST-PT via set_IA_PS
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_IA_fastpt(const int IA_MODEL, const int IA_REDSHIFT_EVOL, const int IA_code)
{
  static constexpr std::string_view fname = "init_IA_fastpt"sv;
  debug("{}: {}", fname, errbegins);
  debug(debugsel, fname, "IA MODEL", IA_MODEL);
  debug(debugsel, fname, "IA REDSHIFT EVOLUTION", IA_REDSHIFT_EVOL);
  debug(debugsel, fname, "IA code", IA_code);
  
  if (0 == IA_MODEL || 1 == IA_MODEL) {
    nuisance.IA_MODEL = IA_MODEL;
  }
  else [[unlikely]] {
    critical(errorns2, fname, "nuisance.IA_MODEL", IA_MODEL);
    exit(1);
  }
  
  if (IA_REDSHIFT_EVOL == NO_IA                   || 
      IA_REDSHIFT_EVOL == IA_NLA_LF               ||
      IA_REDSHIFT_EVOL == IA_REDSHIFT_BINNING     || 
      IA_REDSHIFT_EVOL == IA_REDSHIFT_EVOLUTION)
  {
    nuisance.IA = IA_REDSHIFT_EVOL;
  }
  else [[unlikely]] {
    critical(errorns2, fname, "nuisance.IA", IA_REDSHIFT_EVOL);
    exit(1);
  }

  if (0 == IA_code || 1 == IA_code) {
    nuisance.IA_code = IA_code;
  }
  else [[unlikely]] {
    critical(errorns2, fname, "nuisance.IA_code", IA_code);
    exit(1);
  }
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// Backward-compatible alias: init_IA_fastpt with IA_code = 0 (CFASTPT).
//
// Parameters:
//   IA_MODEL         - 0 = IA_MODEL_NLA, 1 = IA_MODEL_TATT
//   IA_REDSHIFT_EVOL - redshift-dependence mode (see init_IA_fastpt)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_IA(const int IA_MODEL, const int IA_REDSHIFT_EVOL)
{
	init_IA_fastpt(IA_MODEL, IA_REDSHIFT_EVOL, 0);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Turn on the probes that enter the data vector.
//
// Writes the flags like.shear_shear, like.shear_pos, like.pos_pos, like.gk,
// like.ks, like.kk from a named combination (probe_map keys: "xi",
// "gammat", "wtheta", "2x2pt", "3x2pt", "5x2pt", "6x2pt" and the partial
// ss/sg/gg/gk/sk/kk combinations below; input trimmed and lowercased for
// the lookup). IP::set_mask reads these flags to zero the mask entries of
// disabled probes. No cache key is bumped.
//
// Validation: unknown names are critical() + exit(1).
//
// Parameters:
//   possible_probes - probe-combination name (probe_map key, case-insensitive)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_probes(std::string possible_probes)
{
  static constexpr std::string_view fname = "init_probes"sv;
  debug("{}: {}", fname, errbegins);
  
  static const std::unordered_map<std::string, arma::Col<int>::fixed<6>> 
    probe_map = {
        { "xi",     arma::Col<int>::fixed<6>{{1,0,0,0,0,0}} },
        { "gammat", arma::Col<int>::fixed<6>{{0,1,0,0,0,0}} },
        { "wtheta", arma::Col<int>::fixed<6>{{0,0,1,0,0,0}} },
        { "2x2pt",  arma::Col<int>::fixed<6>{{0,1,1,0,0,0}} },
        { "3x2pt",  arma::Col<int>::fixed<6>{{1,1,1,0,0,0}} },
        { "5x2pt",  arma::Col<int>::fixed<6>{{1,1,1,1,1,0}} },
        { "6x2pt",  arma::Col<int>::fixed<6>{{1,1,1,1,1,1}} },
        { "3x2pt_ks_gk_kk", arma::Col<int>::fixed<6>{{0,0,0,1,1,1}} },
        { "3x2pt_ss_sk_sk", arma::Col<int>::fixed<6>{{1,0,0,0,1,1}} },
        { "xi_ggl", arma::Col<int>::fixed<6>{{1,1,0,0,0,0}} },
        { "xi_gg", arma::Col<int>::fixed<6>{{1,0,1,0,0,0}} },
        { "2x2pt_ss_sg", arma::Col<int>::fixed<6>{{1,1,0,0,0,0}} },
        { "2x2pt_ss_gg", arma::Col<int>::fixed<6>{{1,0,1,0,0,0}} },
        { "2x2pt_ss_sk", arma::Col<int>::fixed<6>{{1,0,0,0,1,0}} },
        { "2x2pt_ss_gk", arma::Col<int>::fixed<6>{{1,0,0,1,0,0}} },
        { "2x2pt_ss_kk", arma::Col<int>::fixed<6>{{1,0,0,0,0,1}} },
    };
  static const std::unordered_map<std::string,std::string> 
    names = {
       {"xi", "cosmic shear"},
       {"gammat", "gammat"},
       {"wtheta", "wtheta"},
       {"2x2pt", "2x2pt"},
       {"3x2pt", "3x2pt"},
       {"xi_ggl", "xi + ggl (2x2pt)"},
       {"xi_gg",  "xi + gg (2x2pt)"},
       {"2x2pt_ss_sg", "ss + sg (2x2pt)"},
       {"2x2pt_ss_gg", "ss + gg (2x2pt)"},
       {"2x2pt_ss_sk", "ss + sk (2x2pt)"},
       {"2x2pt_ss_gk", "ss + gk (2x2pt)"},
       {"2x2pt_ss_kk", "ss + kk (2x2pt)"},
       {"5x2pt",  "5x2pt"},
       {"3x2pt_ks_gk_kk", "3x2pt (gk + sk + kk)"},
       {"3x2pt_ss_sk_sk", "3x2pt (ss + sk + kk)"},
       {"6x2pt",  "6x2pt"},
    };

  boost::trim_if(possible_probes, boost::is_any_of("\t "));
  const std::string probe_key =
      boost::algorithm::to_lower_copy(possible_probes);
  auto it = probe_map.find(probe_key);
  if (it == probe_map.end()) {
    critical(errorns2, fname, "possible_probes", possible_probes);
    std::exit(1);
  }
  const auto& flags = it->second;

  like.shear_shear = flags(0);
  like.shear_pos = flags(1);
  like.pos_pos = flags(2);
  like.gk = flags(3);
  like.ks = flags(4);
  like.kk = flags(5);
  debug(debugsel, fname, "possible_probes", names.at(probe_key));
  debug("{}: Ends", "init_probes");
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Read an n(z) file: column 0 = redshift grid, columns 1..Ntomo = per-bin
// histograms (read_table format, "#" comment lines allowed).
//
// Validation: non-empty filename, 0 < Ntomo <= MAX_SIZE_ARRAYS and a
// monotonically increasing z column, else critical() + exit(1).
//
// Parameters:
//   multihisto_file - path of the n(z) file
//   Ntomo           - number of tomographic bins (columns after z)
//
// Returns:
//   the parsed (nz x (Ntomo+1)) table
// ---------------------------------------------------------------------------
arma::Mat<double> read_nz_sample(std::string multihisto_file, const int Ntomo)
{
  static constexpr std::string_view fname = "read_nz_sample"sv;
  debug("{}: {}", fname, errbegins);
  if (!(multihisto_file.size() > 0)) [[unlikely]] {
    critical("{}: empty {} string not supported", fname, "multihisto_file");
    exit(1);
  }
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "Ntomo", Ntomo, MAX_SIZE_ARRAYS);
    exit(1);
  }  
  debug(debugsel, fname, "redshift file:", multihisto_file);
  debug(debugsel, fname, "Ntomo", Ntomo);
  // READ THE N(Z) FILE BEGINS ------------
  arma::Mat<double> input_table = read_table(multihisto_file);
  if (!input_table.col(0).eval().is_sorted("ascend")) {
    critical("bad n(z) file (z vector not monotonic)");
    exit(1);
  }
  debug("{}: {}", fname, errends);
  return input_table;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lens n(z) from file: set_lens_sample_size(Ntomo), then set_lens_sample on
// the table returned by read_nz_sample.
//
// Parameters:
//   multihisto_file - path of the lens n(z) file
//   Ntomo           - number of lens tomographic bins
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_lens_sample(std::string multihisto_file, const int Ntomo)
{
  static constexpr std::string_view fname = "init_lens_sample v2.0"sv;
  debug("{}: {}", fname, errbegins);
  set_lens_sample_size(Ntomo);
  set_lens_sample(read_nz_sample(multihisto_file, Ntomo));
  debug("{}: Ends", "init_lens_sample v2.0");
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Source n(z) from file: set_source_sample_size(Ntomo), then
// set_source_sample on the table returned by read_nz_sample.
//
// Parameters:
//   multihisto_file - path of the source n(z) file
//   Ntomo           - number of source tomographic bins
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_source_sample(std::string multihisto_file, const int Ntomo)
{
  static constexpr std::string_view fname = "init_source_sample"sv;
  debug("{}: {}", fname, errbegins);
  set_source_sample_size(Ntomo);
  set_source_sample(read_nz_sample(multihisto_file, Ntomo));
  debug("{}: Ends", "init_source_sample");
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Count the tomographic power spectra entering the data vector and warm the
// GGL pair maps.
//
// Writes tomo.shear_Npowerspectra = nbin (nbin + 1) / 2 (auto + cross),
// tomo.ggl_Npowerspectra = number of lens-source pairs with
// test_zoverlap(l, s) = 1 (all pairs minus the init_ggl_exclude list) and
// tomo.clustering_Npowerspectra = clustering_nbin (auto only).
//
// Cache invalidation:
// bumps tomo.random_ggl, then (when any pair survives)
// touches ZL(0)/ZS(0)/N_ggl so the static pair maps of redshift_spline.c
// rebuild here, single-threaded - never inside an OpenMP region (see the
// inline comments).
//
// Validation: redshift.shear_nbin and redshift.clustering_nbin must already
// be set, else critical() + exit(1).
//
// Parameters:
//   (none)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_ntomo_powerspectra()
{
  static constexpr std::string_view fname = "init_ntomo_powerspectra"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.shear_nbin) [[unlikely]] {
    critical(errornset, fname, "redshift.shear_nbin"); exit(1);
  }
  if (0 == redshift.clustering_nbin) [[unlikely]] {
    critical(errornset, fname, "redshift.clustering_nbin"); exit(1);
  }
  tomo.shear_Npowerspectra = redshift.shear_nbin * (redshift.shear_nbin + 1) / 2;
  // The bins (and possibly ggl_exclude) of this model can differ from the
  // previous model built in the same process (the unit tests build a 3x2pt
  // model, then a cosmic-shear model with another ggl_exclude list): a new
  // key makes the static pair maps of redshift_spline.c rebuild.
  tomo.random_ggl = RandomNumber::get_instance().get();
  int n = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    for (int j=0; j<redshift.shear_nbin; j++) {
      n += test_zoverlap(i, j);
      if(test_zoverlap(i,j) == 0) {
        spdlog::info("{}: GGL pair L{:d}-S{:d} is excluded", fname, i, j);
      }
    }
  }
  tomo.ggl_Npowerspectra = n;
  tomo.clustering_Npowerspectra = redshift.clustering_nbin;
  if (n > 0) { // rebuild the pair maps here, single-threaded, so no call
               // inside an OpenMP region ever writes them
    (void) ZL(0);
    (void) ZS(0);
    (void) N_ggl(ZL(0), ZS(0));
  }

  debug("{}: tomo.shear_Npowerspectra = {}", fname, tomo.shear_Npowerspectra);
  debug("{}: tomo.ggl_Npowerspectra = {}", fname, tomo.ggl_Npowerspectra);
  debug("{}: tomo.clustering_Npowerspectra = {}", fname, tomo.clustering_Npowerspectra);
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Read both n(z) files and return them to python as a (lens, source) tuple
// of numpy arrays. No C global is written; the init_ companion below
// installs the tables instead.
//
// Parameters:
//   lens_multihisto_file   - path of the lens n(z) file
//   lens_ntomo             - number of lens bins
//   source_multihisto_file - path of the source n(z) file
//   source_ntomo           - number of source bins
//
// Returns:
//   py::tuple(lens table, source table) as numpy arrays
// ---------------------------------------------------------------------------
py::tuple read_redshift_distributions_from_files(
  std::string lens_multihisto_file, const int lens_ntomo,
  std::string source_multihisto_file, const int source_ntomo)
{
  matrix ilt = read_nz_sample(lens_multihisto_file,lens_ntomo);
  matrix ist = read_nz_sample(source_multihisto_file,source_ntomo);
  return py::make_tuple(carma::mat_to_arr(ilt), carma::mat_to_arr(ist));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One-call n(z) setup: init_lens_sample, init_source_sample, then
// init_ntomo_powerspectra (pair counts + pair-map warm-up).
//
// Parameters:
//   lens_multihisto_file   - path of the lens n(z) file
//   lens_ntomo             - number of lens bins
//   source_multihisto_file - path of the source n(z) file
//   source_ntomo           - number of source bins
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_redshift_distributions_from_files(
  std::string lens_multihisto_file, const int lens_ntomo,
  std::string source_multihisto_file, const int source_ntomo)
{
  init_lens_sample(lens_multihisto_file, lens_ntomo);
  init_source_sample(source_multihisto_file, source_ntomo);
  init_ntomo_powerspectra();
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the survey identity: survey.name (trimmed, lowercased), survey.area
// (deg^2) and survey.sigma_e. No cache key is bumped.
//
// Validation: name non-empty and shorter than CHAR_MAX_SIZE, else
// critical() + exit(1).
//
// Parameters:
//   surveyname - survey label (trimmed and lowercased before storing)
//   area       - survey area (deg^2)
//   sigma_e    - shape-noise dispersion
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_survey(
    std::string surveyname,
    double area,
    double sigma_e)
{
  static constexpr std::string_view fname = "init_survey"sv;
  debug("{}: {}", fname, errbegins);
  boost::trim_if(surveyname, boost::is_any_of("\t "));
  surveyname = boost::algorithm::to_lower_copy(surveyname);
  if (surveyname.size() > CHAR_MAX_SIZE - 1) {
    critical("{}: survey name too large for Cosmolike (C char overflow)", fname);
    exit(1);
  }
  if (!(surveyname.size()>0)) {
    critical(erroric0, fname, "surveyname.size()"); exit(1);
  }
  memcpy(survey.name, surveyname.c_str(), surveyname.size() + 1);
  survey.area = area;
  survey.sigma_e = sigma_e;
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install the lens-source pairs excluded from galaxy-galaxy lensing.
//
// Writes tomo.ggl_exclude as a flat [lens0, src0, lens1, src1, ...] array
// and tomo.N_ggl_exclude = input size / 2; test_zoverlap(l, s) returns 0
// for listed pairs. tomo.ggl_Npowerspectra is not recounted here; that (and
// the single-threaded pair-map warm-up) happens in init_ntomo_powerspectra.
//
// Cache invalidation:
// bumps tomo.random_ggl so the static pair maps
// (test_zoverlap/ZL/ZS/N_ggl in redshift_spline.c) rebuild on next use.
//
// Validation: NaN entries and allocation failure are critical() + exit(1).
//
// Parameters:
//   ggl_exclude - flat (lens0, src0, lens1, src1, ...) pair list
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void init_ggl_exclude(arma::Col<int> ggl_exclude)
{
  static constexpr std::string_view fname = "init_ggl_exclude"sv;
  debug("{}: {}", fname, errbegins);
  const int nsize = static_cast<int>(ggl_exclude.n_elem);
  if (tomo.ggl_exclude != NULL) {
    free(tomo.ggl_exclude);
  }
  tomo.ggl_exclude = (int*) malloc(sizeof(int)*nsize);
  if (NULL == tomo.ggl_exclude) {
    critical("array allocation failed"); exit(1);
  }
  if (0 != nsize % 2) [[unlikely]] {
    critical("{}: ggl_exclude length = {} is odd (lens, source pairs "
             "required)", fname, nsize);
    exit(1);
  }
  tomo.N_ggl_exclude = int(nsize/2);
  tomo.random_ggl = RandomNumber::get_instance().get(); // pair maps rebuild
  debug("{}: {} ggl pairs excluded", fname, tomo.N_ggl_exclude);
  for(int i=0; i<nsize; i++) {
    if (std::isnan(ggl_exclude(i))) {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    tomo.ggl_exclude[i] = ggl_exclude(i);
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SET FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Update the background parameters Cosmolike keeps (Cobaya supplies P(k,z),
// distances and growth through the other set_ functions).
//
// When any input changed (fdiff): writes cosmology.Omega_m,
// Omega_v = 1 - Omega_m, Omega_b (the Compton-y halo-model sector reads
// it; the y spectra abort while it is 0), h0 = hubble/100 (input H0 in
// km/s/Mpc), a fixed nonzero Omega_nu placeholder and
// MGSigma = MGmu = 0, and bumps cosmology.random so every table keyed
// on the cosmology rebuilds. Unchanged inputs leave the cache key
// alone.
//
// Parameters:
//   omega_matter - Omega_m today
//   omega_baryon - Omega_b today (0 = not provided; only the
//                  Compton-y sector demands it)
//   hubble       - H0 (km/s/Mpc)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_cosmological_parameters(
    const double omega_matter,
    const double omega_baryon,
    const double hubble
  )
{
  static constexpr std::string_view fname = "set_cosmological_parameters"sv;
  debug("{}: {}", fname, errbegins);
  // Cosmolike should not need parameters from inflation or dark energy.
  // Cobaya provides P(k,z), H(z), D(z), Chi(z)...
  // It may require H0 to set scales and \Omega_M to set the halo model
  int cache_update = 0;
  if (fdiff(cosmology.Omega_m, omega_matter) ||
      fdiff(cosmology.Omega_b, omega_baryon) ||
      fdiff(cosmology.h0, hubble/100.0)) { // assuming H0 in km/s/Mpc 
    cache_update = 1;
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.Omega_m = omega_matter;
    cosmology.Omega_v = 1.0-omega_matter;
    cosmology.Omega_b = omega_baryon;
    // Cosmolike only needs to know that there are massive neutrinos (>0)
    cosmology.Omega_nu = 0.1;
    cosmology.h0 = hubble/100.0; 
    cosmology.MGSigma = 0.0;
    cosmology.MGmu = 0.0;
    cosmology.random = cosmolike_interface::RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// Install the python FAST-PT intrinsic-alignment tables (nuisance.IA_code
// = 1 path), replacing FPTIA.tab wholesale.
//
// Units: input k in h/Mpc, spectra in (Mpc/h)^3; stored as k * coverH0 and
// P / coverH0^3 (row 10 is the k row, every other row a spectrum).
//
// The rebuild is skipped when N, k_min, k_max, k_cutoff and every table
// entry match the stored values (fdiff). On update: frees a separate
// internal table only, then the output table (see the aliasing comment
// below), reallocates tab as 12 x N, re-aliases tab_int = tab and
// N_int = N, and bumps nuisance.random_ia so the C_ell caches recompute.
// get_FPT_IA (pt_cfastpt.c, the IA_code = 0 path) detects the replaced
// table through its owned-table pointer and rebuilds its own grid on its
// next call.
//
// Validation: PS must hold exactly 12 x N elements (out-of-bounds guard,
// see below) and no NaN entries, else critical() + exit(1).
//
// Parameters:
//   PS     - flattened 12 x N table, row-major (row 10 = k grid)
//   kmin   - lower k edge (h/Mpc)
//   kmax   - upper k edge (h/Mpc)
//   cutoff - high-k cutoff (h/Mpc)
//   N      - number of k points per row
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_IA_PS(
    vector PS,
    const double kmin,
    const double kmax,
    const double cutoff,
    const int N
  )
{
  static constexpr std::string_view fname = "set_IA_PS"sv;
  // Row count of the IA table the python fastpt theory block sends:
  // 10 spectra (tt_E, tt_B, ta_dE1, ta_dE2, ta_0E0E, ta_0B0B, mixA,
  // mixBtype2, mixDEE, mixDBB) plus the k row (index 10, rescaled as
  // a wavenumber by the i != 10 branches below) plus P_lin.
  constexpr int NIAPS = 12;
  const double coverH0 = cosmology.coverH0;
  const double coverH0cube = coverH0*coverH0*coverH0;

  // Both loops below index PS[i*N+j] up to NIAPS*N, and arma's
  // operator[] does not bounds-check in release builds: a sender
  // whose row count disagrees with NIAPS would be read out of
  // bounds silently (see the NBIAS comment in set_bias_PS for the
  // abort that mismatch caused). Refuse mismatched input instead.
  if (PS.n_elem != static_cast<arma::uword>(NIAPS) * N) [[unlikely]] {
    critical("{}: PS has {} elements; expected NIAPS x N = {} x {} = {}",
             fname, PS.n_elem, NIAPS, N, NIAPS * N);
    exit(1);
  }

  int cache_update = 0;
  if (NULL == FPTIA.tab ||
      FPTIA.N != N ||
      fdiff(FPTIA.k_min, kmin * coverH0 ) || 
      fdiff(FPTIA.k_max, kmax * coverH0) || 
      fdiff(FPTIA.k_cutoff, cutoff * coverH0)) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<NIAPS; i++) {
      for (int j=0; j<FPTIA.N; j++) {
        if (i != 10) {
          if (fdiff(FPTIA.tab[i][j],PS[i*FPTIA.N+j]/(coverH0cube))) {
            cache_update = 1; 
            break; 
          }
        }
        else {
          if (fdiff(FPTIA.tab[i][j],PS[i*FPTIA.N+j]*coverH0)) { // k
            cache_update = 1; 
            break; 
          }
        }
      }
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) { 
    FPTIA.k_min  = kmin * coverH0;     // input in units of h/Mpc
    FPTIA.k_max  = kmax * coverH0;     // input in units of h/Mpc
    FPTIA.N      = N;
    FPTIA.sigma4 = 0.0;                // Not relevant for IA
    FPTIA.k_cutoff = cutoff * coverH0; // input in units of h/Mpc
    // FPTIA.tab_int aliases FPTIA.tab after a cfastpt call at the default
    // internal boost; free a separate internal table only, and re-alias it
    // to the new table (freeing tab alone would leave tab_int dangling, and
    // the next cfastpt rebuild would free it a second time: a malloc abort).
    if (FPTIA.tab_int != NULL && FPTIA.tab_int != FPTIA.tab) {
      free(FPTIA.tab_int);
    }
    if (FPTIA.tab != NULL) {
      free(FPTIA.tab);
    }
    FPTIA.tab = (double**) malloc2d(NIAPS, FPTIA.N);
    FPTIA.tab_int = FPTIA.tab;
    FPTIA.N_int = FPTIA.N;
    
    for (int i=0; i<NIAPS; i++) {
      for (int j=0; j<FPTIA.N; j++) {
        if (std::isnan(PS[i*FPTIA.N+j])) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        if (i != 10) {
          FPTIA.tab[i][j] = PS[i*FPTIA.N+j] / (coverH0cube);
        }
        else {
          FPTIA.tab[i][j] = PS[i*FPTIA.N+j] * coverH0; // k
        }
      }
    }
    nuisance.random_ia = RandomNumber::get_instance().get();
  } 
}

// ---------------------------------------------------------------------------
// Install the python FAST-PT galaxy-bias tables (nuisance.IA_code = 1
// path), replacing FPTbias.tab wholesale.
//
// Units: input k in h/Mpc and spectra in (Mpc/h)^3; stored as k * coverH0,
// P / coverH0^3 (row 6 is the k row) and sigma4 / coverH0^3.
//
// The rebuild is skipped when N, k_min, k_max, k_cutoff, sigma4 and every
// table entry match the stored values (fdiff). On update: frees a separate
// internal table only, then the output table (see the aliasing comment
// below), reallocates tab as 8 x N, re-aliases tab_int = tab and N_int = N,
// and bumps nuisance.random_galaxy_bias so the C_ell caches recompute.
// get_FPT_bias (pt_cfastpt.c, the IA_code = 0 path) detects the replaced
// table through its owned-table pointer and rebuilds its own grid on its
// next call.
//
// Validation: PS must hold exactly 8 x N elements (see the NBIAS comment
// below for the abort a mismatch caused) and no NaN entries, else
// critical() + exit(1).
//
// Parameters:
//   PS     - flattened 8 x N table, row-major (row 6 = k grid)
//   kmin   - lower k edge (h/Mpc)
//   kmax   - upper k edge (h/Mpc)
//   cutoff - high-k cutoff (h/Mpc)
//   sigma4 - FAST-PT sigma^4 constant (stored / coverH0^3)
//   N      - number of k points per row
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_bias_PS(
    vector PS,
    const double kmin,
    const double kmax,
    const double cutoff,
    const double sigma4,
    const int N
  )
{
  static constexpr std::string_view fname = "set_bias_PS"sv;
  // Row count of the bias table the python fastpt theory block sends:
  // (d1d2, d2d2, d1s2, d2s2, s2s2, d1p3, k, P_lin) - 8 rows, with the
  // k row at index 6 (the i != 6 branches below rescale it as a
  // wavenumber). This constant was 12, copy-pasted from NIAPS in
  // set_IA_PS: the two scan loops then read PS[i*N+j] for i = 8..11,
  // 4*N doubles past the end of the input vector (arma operator[]
  // does not bounds-check in release builds). Whenever the heap
  // bytes there happened to encode a NaN, the isnan tripwire below
  // killed the process ("NaN found on interpolation table"): a
  // nondeterministic, machine-load-dependent abort, observed in the
  // lsst_y1 CFASTPT-vs-FASTPT sweep at OMP_NUM_THREADS 1 and 4
  // (2026-09-22). The garbage rows also made the cache comparison
  // below report a change on almost every call, so the table was
  // freed and rebuilt every evaluation. Downstream code reads rows
  // 0, 2, and 5 only (GS_BIAS_SRC in cosmo2D.c).
  constexpr int NBIAS = 8;
  const double coverH0  = cosmology.coverH0;
  const double coverH0cube = coverH0*coverH0*coverH0;

  // same refusal as set_IA_PS: a row-count mismatch must fail loudly
  // here, not as an out-of-bounds read inside the loops
  if (PS.n_elem != static_cast<arma::uword>(NBIAS) * N) [[unlikely]] {
    critical("{}: PS has {} elements; expected NBIAS x N = {} x {} = {}",
             fname, PS.n_elem, NBIAS, N, NBIAS * N);
    exit(1);
  }

  int cache_update = 0;
  if (NULL == FPTbias.tab ||
      FPTbias.N != N ||
      fdiff(FPTbias.k_min, kmin * coverH0) || 
      fdiff(FPTbias.k_max, kmax * coverH0) || 
      fdiff(FPTbias.k_cutoff, cutoff * coverH0) ||
      fdiff(FPTbias.sigma4, sigma4 / (coverH0cube))) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<NBIAS; i++)  {
      for (int j=0; j<FPTbias.N; j++) {
        if (i != 6) {
          if(fdiff(FPTbias.tab[i][j],PS[i*FPTbias.N+j]/(coverH0cube))) {
            cache_update = 1; 
            break; 
          }
        }
        else { 
          if(fdiff(FPTbias.tab[i][j],PS[i*FPTbias.N+j]*coverH0)) { // k
            cache_update = 1; 
            break; 
          }
        }
      }
    }
  }

  if (1 == cache_update || 1 == force_cache_update_test) { 
    FPTbias.N        = N;
    FPTbias.k_min    = kmin * coverH0;    // input in units of h/Mpc
    FPTbias.k_max    = kmax * coverH0;    // input in units of h/Mpc
    FPTbias.k_cutoff = cutoff *coverH0; // input in units of h/Mpc
    FPTbias.sigma4   = sigma4 / (coverH0cube);
    // FPTbias.tab_int aliases FPTbias.tab after a cfastpt call at the default
    // internal boost; free a separate internal table only, and re-alias it
    // to the new table (freeing tab alone would leave tab_int dangling, and
    // the next cfastpt rebuild would free it a second time: a malloc abort).
    if (FPTbias.tab_int != NULL && FPTbias.tab_int != FPTbias.tab) {
      free(FPTbias.tab_int);
    }
    if (FPTbias.tab != NULL) {
      free(FPTbias.tab);
    }
    FPTbias.tab = (double**) malloc2d(NBIAS, FPTbias.N);
    FPTbias.tab_int = FPTbias.tab;
    FPTbias.N_int = FPTbias.N;

    for (int i=0; i<NBIAS; i++)  {
      for (int j=0; j<FPTbias.N; j++) {
        if (std::isnan(PS[i*FPTbias.N + j])) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        if (i != 6) {
          FPTbias.tab[i][j] = PS[i*FPTbias.N+j] / (coverH0cube);
        }
        else { 
          FPTbias.tab[i][j] = PS[i*FPTbias.N+j] * coverH0; // k
        }
      }
    }
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install the comoving-distance table chi(z) from Cobaya.
//
// When the size or any entry changed (fdiff scan): writes cosmology.chi
// (row 0 = z, row 1 = chi) and cosmology.chi_nz, precomputes the
// direct-index segment metadata under COSMO3D_ASSUME_PIECEWISE_UNIFORM
// (detect_uniform_segments), NaN-scans during the parallel fill, and bumps
// cosmology.random. Unchanged input leaves the cache key alone.
//
// Validation: equal input sizes and at least 5 points, else critical() +
// exit(1); NaN entries abort inside the fill.
//
// Parameters:
//   io_z   - redshift grid (>= 5 points)
//   io_chi - comoving distance at io_z (same length)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_distances(vector io_z, vector io_chi)
{
  static constexpr std::string_view fname = "set_distances"sv;
  debug("{}: Begins", "set_distances");
  bool debug_fail = false;
  if (io_z.n_elem != io_chi.n_elem) [[unlikely]] {
    debug_fail = true;
  }
  else {
    if (io_z.n_elem == 0) [[unlikely]] {
      debug_fail = true;
    }
  }
  if (debug_fail) [[unlikely]] {
    critical("{}: {} = {:d} and G.size = {:d}", fname, erriiwz, io_z.n_elem, io_chi.n_elem);
    exit(1);
  }
  if(io_z.n_elem < 5) [[unlikely]] {
    critical("{}: {} = {:d} and chi.size = {:d}", fname, erriiwz, io_z.n_elem, io_chi.n_elem);
    exit(1);
  }

  int cache_update = 0;
  if (cosmology.chi_nz != static_cast<int>(io_z.n_elem) || NULL == cosmology.chi) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<cosmology.chi_nz; i++) {
      if (fdiff(cosmology.chi[0][i], io_z(i)) ||
          fdiff(cosmology.chi[1][i], io_chi(i))) {
        cache_update = 1; 
        break; 
      }    
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.chi_nz = static_cast<int>(io_z.n_elem);
    if (cosmology.chi != NULL) {
      free(cosmology.chi);
    }
    cosmology.chi = (double**) malloc2d(2, cosmology.chi_nz);

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM 
    cosmology.chi_z_nseg = detect_uniform_segments(
        io_z.memptr(), cosmology.chi_nz, 1e-9, MAX_GRID_SEGMENTS,
        cosmology.chi_z_seg_start, cosmology.chi_z_seg_len,
        cosmology.chi_z_seg_xmin,  cosmology.chi_z_seg_inv_dx,
        "chi z");
#endif

    #pragma omp parallel for schedule(static)
    for (int i=0; i<cosmology.chi_nz; i++) {
      if (std::isnan(io_z(i)) || std::isnan(io_chi(i))) [[unlikely]] {
        critical("{}: {}", fname, errnanit);
        exit(1);
      }
      cosmology.chi[0][i] = io_z(i);
      cosmology.chi[1][i] = io_chi(i);
    }
    cosmology.random = RandomNumber::get_instance().get();
  }
  debug("{}: Ends", "set_distances");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install the growth table G(z) from Cobaya (D = G * a).
//
// When the size or any entry changed (fdiff scan): writes cosmology.G
// (row 0 = z, row 1 = G) and cosmology.G_nz, precomputes the direct-index
// segment metadata under COSMO3D_ASSUME_PIECEWISE_UNIFORM (consumed by
// f_growth/growfac and friends), NaN-scans during the parallel fill, and
// bumps cosmology.random. Unchanged input leaves the cache key alone.
//
// Validation: equal input sizes, else critical() + exit(1); NaN entries
// abort inside the fill.
//
// Parameters:
//   io_z - redshift grid
//   io_G - growth G at io_z, with D = G * a (same length)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_growth(vector io_z, vector io_G)
{
  static constexpr std::string_view fname = "set_growth"sv;
  debug("{}: {}", fname, errbegins);
  if (io_z.n_elem != io_G.n_elem) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, io_z.n_elem, io_G.n_elem); exit(1);
  }

  int cache_update = 0;
  if (cosmology.G_nz != static_cast<int>(io_z.n_elem) || NULL == cosmology.G) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<cosmology.G_nz; i++) {
      if (fdiff(cosmology.G[0][i], io_z(i)) || fdiff(cosmology.G[1][i], io_G(i))) {
        cache_update = 1; 
        break;
      }    
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test)
  {
    cosmology.G_nz = static_cast<int>(io_z.n_elem);

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM    
    // -----------------------------------------------------------------
    // Validate grid uniformity and precompute direct-index metadata.
    // f_growth, growfac, norm_growfac, norm_growfac_all use these
    // fields to skip the per-call binary search on z.
    // -----------------------------------------------------------------
    cosmology.G_z_nseg = detect_uniform_segments(
        io_z.memptr(), cosmology.G_nz, 1e-9, MAX_GRID_SEGMENTS,
        cosmology.G_z_seg_start, cosmology.G_z_seg_len,
        cosmology.G_z_seg_xmin,  cosmology.G_z_seg_inv_dx,
        "G z");
#endif

    if (cosmology.G != NULL) { free(cosmology.G); }
    cosmology.G = (double**) malloc2d(2, cosmology.G_nz);
    
    #pragma omp parallel for schedule(static)
    for (int i=0; i<cosmology.G_nz; i++) {
      if (std::isnan(io_z(i)) || std::isnan(io_G(i))) [[unlikely]] {
        critical("{}: {}", fname, errnanit); exit(1);
      }
      cosmology.G[0][i] = io_z(i);
      cosmology.G[1][i] = io_G(i);
    }
    cosmology.random = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install ln P_lin(log10k, z) from Cobaya.
//
// Table layout (structs.h): (nk+1) x (nz+1), values in [i<nk][j<nz], the
// log10k axis in column nz and the z axis in row nk. When sizes, values or
// either axis changed (fdiff scans): reallocates cosmology.lnPL, writes
// lnPL_nk/lnPL_nz, NaN-scans during the parallel fill, and bumps
// cosmology.random. Unchanged input leaves the cache key alone.
//
// Under COSMO3D_ASSUME_PIECEWISE_UNIFORM the log10k axis must be one
// uniform segment (critical() otherwise) and the z axis may be piecewise
// uniform; p_lin uses the stored metadata for direct indexing.
//
// Validation: io_lnP size must equal nk * nz, else critical() + exit(1).
//
// Parameters:
//   io_log10k - log10 k grid
//   io_z      - redshift grid
//   io_lnP    - flattened ln P_lin, io_lnP(i*nz + j) = ln P(k_i, z_j)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_linear_power_spectrum(vector io_log10k, vector io_z, vector io_lnP)
{
  static constexpr std::string_view fname = "set_linear_power_spectrum"sv;
  debug("{}: {}", fname, errbegins);
  if (io_z.n_elem*io_log10k.n_elem != io_lnP.n_elem) [[unlikely]] {
    critical(errorsz1d,fname,erriiwz,io_z.n_elem*io_log10k.n_elem,io_lnP.n_elem); 
    exit(1);
  }
  
  int cache_update = 0;
  if (cosmology.lnPL_nk != static_cast<int>(io_log10k.n_elem) ||
      cosmology.lnPL_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.lnPL) {
    cache_update = 1;
  }
  else {
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      for (int j=0; j<cosmology.lnPL_nz; j++) {
        if (fdiff(cosmology.lnPL[i][j], io_lnP(i*cosmology.lnPL_nz+j))) {
          cache_update = 1; 
          goto jump;
        }
      }
    }
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      if (fdiff(cosmology.lnPL[i][cosmology.lnPL_nz], io_log10k(i))) {
        cache_update = 1; 
        goto jump;
      }
    }
    for (int j=0; j<cosmology.lnPL_nz; j++) {
      if (fdiff(cosmology.lnPL[cosmology.lnPL_nk][j], io_z(j))) {
        cache_update = 1; 
        goto jump;
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test) 
  {
    cosmology.lnPL_nk = static_cast<int>(io_log10k.n_elem);
    cosmology.lnPL_nz = static_cast<int>(io_z.n_elem);
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
    // -------------------------------------------------------------------------
    // Validate grid uniformity and precompute direct-index metadata.
    // p_lin uses these fields to skip the per-call binary search on log10k & z
    // -------------------------------------------------------------------------
    {
      // log10k axis: required to be a single uniform segment.
      int    s_start[MAX_GRID_SEGMENTS], s_len[MAX_GRID_SEGMENTS];
      double s_xmin[MAX_GRID_SEGMENTS],  s_inv_dx[MAX_GRID_SEGMENTS];
      int nseg = detect_uniform_segments(io_log10k.memptr(),
                                         cosmology.lnPL_nk,
                                         1e-9, MAX_GRID_SEGMENTS,
                                         s_start, s_len, s_xmin, s_inv_dx,
                                         "lnPL log10k");
      if (nseg != 1) [[unlikely]] {
        critical("{}: lnPL log10k expected single uniform segment, got {}",
                 fname, nseg);
        exit(1);
      }
      cosmology.lnPL_log10k_min    = s_xmin[0];
      cosmology.lnPL_log10k_inv_dx = s_inv_dx[0];
    }
    {
      // z axis: piecewise-uniform allowed (1..MAX_GRID_SEGMENTS segments).
      cosmology.lnPL_z_nseg = detect_uniform_segments(
          io_z.memptr(), cosmology.lnPL_nz, 1e-9, MAX_GRID_SEGMENTS,
          cosmology.lnPL_z_seg_start, cosmology.lnPL_z_seg_len,
          cosmology.lnPL_z_seg_xmin,  cosmology.lnPL_z_seg_inv_dx,
          "lnPL z");
    }
#endif
    if (cosmology.lnPL != NULL) { free(cosmology.lnPL); }
    cosmology.lnPL = (double**) malloc2d(cosmology.lnPL_nk+1,cosmology.lnPL_nz+1);

    #pragma omp parallel
    {
      #pragma omp for schedule(static) nowait
      for (int i = 0; i < cosmology.lnPL_nk; i++) {
        if (std::isnan(io_log10k(i))) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        cosmology.lnPL[i][cosmology.lnPL_nz] = io_log10k(i);
      }
      #pragma omp for schedule(static) nowait
      for (int j = 0; j < cosmology.lnPL_nz; j++) {
        if (std::isnan(io_z(j))) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        cosmology.lnPL[cosmology.lnPL_nk][j] = io_z(j);
      }
      #pragma omp for collapse(2) schedule(static) nowait
      for (int i = 0; i < cosmology.lnPL_nk; i++) {
        for (int j = 0; j < cosmology.lnPL_nz; j++) {
          if (std::isnan(io_lnP(i * cosmology.lnPL_nz + j))) [[unlikely]] {
            critical("{}: {}", fname, errnanit); exit(1);
          }
          cosmology.lnPL[i][j] = io_lnP(i * cosmology.lnPL_nz + j);
        }
      }
    }
    cosmology.random = RandomNumber::get_instance().get();
  }

  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install ln P_nonlin(log10k, z) from Cobaya.
//
// Same machinery as set_linear_power_spectrum, applied to cosmology.lnP /
// lnP_nk / lnP_nz: fdiff change scans, uniform-grid metadata under
// COSMO3D_ASSUME_PIECEWISE_UNIFORM (consumed by p_nonlin), NaN scan during
// the parallel fill, and a cosmology.random bump on update.
//
// Validation: io_lnP size must equal nk * nz, else critical() + exit(1).
//
// Parameters:
//   io_log10k - log10 k grid
//   io_z      - redshift grid
//   io_lnP    - flattened ln P_nonlin, io_lnP(i*nz + j) = ln P(k_i, z_j)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_non_linear_power_spectrum(vector io_log10k, vector io_z, vector io_lnP)
{
  static constexpr std::string_view fname = "set_non_linear_power_spectrum"sv;
  debug("{}: {}", fname, errbegins);
  if (io_z.n_elem*io_log10k.n_elem != io_lnP.n_elem) [[unlikely]] {
    critical(errorsz1d,fname,erriiwz,io_z.n_elem*io_log10k.n_elem,io_lnP.n_elem); 
    exit(1);
  }

  int cache_update = 0;
  if (cosmology.lnP_nk != static_cast<int>(io_log10k.n_elem) ||
      cosmology.lnP_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.lnP) {
    cache_update = 1;
  }
  else
  {
    for (int i=0; i<cosmology.lnP_nk; i++) {
      for (int j=0; j<cosmology.lnP_nz; j++) {
        if (fdiff(cosmology.lnP[i][j], io_lnP(i*cosmology.lnP_nz+j))) {
          cache_update = 1; 
          goto jump;
        }
      }
    }
    for (int i=0; i<cosmology.lnP_nk; i++) {
      if (fdiff(cosmology.lnP[i][cosmology.lnP_nz], io_log10k(i))) {
        cache_update = 1; 
        goto jump;
      }
    }
    for (int j=0; j<cosmology.lnP_nz; j++) {
      if (fdiff(cosmology.lnP[cosmology.lnP_nk][j], io_z(j))) {
        cache_update = 1; 
        goto jump;
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test) 
  {
    cosmology.lnP_nk = static_cast<int>(io_log10k.n_elem);
    cosmology.lnP_nz = static_cast<int>(io_z.n_elem);
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
    // -----------------------------------------------------------------------
    // Validate grid uniformity and precompute direct-index metadata.
    // p_nonlin uses these fields to skip the per-call binary search on
    // log10k & z
    // -----------------------------------------------------------------------
    {
      // log10k axis: required to be a single uniform segment.
      int    s_start[MAX_GRID_SEGMENTS], s_len[MAX_GRID_SEGMENTS];
      double s_xmin[MAX_GRID_SEGMENTS],  s_inv_dx[MAX_GRID_SEGMENTS];
      int nseg = detect_uniform_segments(io_log10k.memptr(),
                                         cosmology.lnP_nk,
                                         1e-9, MAX_GRID_SEGMENTS,
                                         s_start, s_len, s_xmin, s_inv_dx,
                                         "lnP log10k");
      if (nseg != 1) [[unlikely]] {
        critical("{}: lnP log10k expected single uniform segment, got {}",
                 fname, nseg);
        exit(1);
      }
      cosmology.lnP_log10k_min    = s_xmin[0];
      cosmology.lnP_log10k_inv_dx = s_inv_dx[0];
    }
    {
      // z axis: piecewise-uniform allowed (1..MAX_GRID_SEGMENTS segments).
      cosmology.lnP_z_nseg = detect_uniform_segments(
          io_z.memptr(), cosmology.lnP_nz, 1e-9, MAX_GRID_SEGMENTS,
          cosmology.lnP_z_seg_start, cosmology.lnP_z_seg_len,
          cosmology.lnP_z_seg_xmin,  cosmology.lnP_z_seg_inv_dx,
          "lnP z");
    }
#endif
    if (cosmology.lnP != NULL) { free(cosmology.lnP); }
    cosmology.lnP = (double**) malloc2d(cosmology.lnP_nk+1,cosmology.lnP_nz+1);

    #pragma omp parallel
    {
      #pragma omp for schedule(static) nowait
      for (int i = 0; i < cosmology.lnP_nk; i++) {
        if (std::isnan(io_log10k(i))) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        cosmology.lnP[i][cosmology.lnP_nz] = io_log10k(i);
      }
      #pragma omp for schedule(static) nowait
      for (int j = 0; j < cosmology.lnP_nz; j++) {
        if (std::isnan(io_z(j))) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        cosmology.lnP[cosmology.lnP_nk][j] = io_z(j);
      }
      #pragma omp for collapse(2) schedule(static) nowait
      for (int i = 0; i < cosmology.lnP_nk; i++) {
        for (int j = 0; j < cosmology.lnP_nz; j++) {
          if (std::isnan(io_lnP(i * cosmology.lnP_nz + j))) [[unlikely]] {
            critical("{}: {}", fname, errnanit); exit(1);
          }
          cosmology.lnP[i][j] = io_lnP(i * cosmology.lnP_nz + j);
        }
      }
    }
    cosmology.random = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin multiplicative shear calibration
// nuisance.shear_calibration_m[i]. No cache key exists for m: the (1 + m)
// factors are applied at data-vector assembly (compute_*_masked in
// generic_interface.hpp), not through cached tables.
//
// Validation: shear_nbin set and equal to the input size, no NaN entries,
// else critical() + exit(1).
//
// Parameters:
//   M - per-bin multiplicative shear bias m_i (length shear_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_shear_calib(vector M)
{
  static constexpr std::string_view fname = "set_nuisance_shear_calib"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.shear_nbin) [[unlikely]] {
    critical(errorns2, fname, "shear_Nbin", 0); exit(1);
  }
  if (redshift.shear_nbin != static_cast<int>(M.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, M.n_elem, redshift.shear_nbin); exit(1);
  }
  for (int i=0; i<redshift.shear_nbin; i++) {
    if (std::isnan(M(i))) [[unlikely]] { // can't compile w/ -O3 or -fast-math
      critical(errnance2, fname, i, errnance); exit(1);
    }
    nuisance.shear_calibration_m[i] = M(i);
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin source photo-z shift nuisance.photoz[0][0][i]
// (nz_source_photoz evaluates n(z - shift)).
//
// Cache invalidation:
// bumps nuisance.random_photoz_shear when any value
// changed (fdiff), so the source-side kernels and C_ell caches recompute;
// unchanged input leaves the key alone.
//
// Validation: shear_nbin set and equal to the input size, no NaN entries,
// else critical() + exit(1).
//
// Parameters:
//   SP - per-bin source photo-z shifts (length shear_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_shear_photoz(vector SP)
{
  static constexpr std::string_view fname = "set_nuisance_shear_photoz"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.shear_nbin) [[unlikely]] {
    critical(errorns2, fname, "shear_Nbin", 0); exit(1);
  }
  if (redshift.shear_nbin != static_cast<int>(SP.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, SP.n_elem, redshift.shear_nbin); exit(1);
  }
  int cache_update = 0;
  for (int i=0; i<redshift.shear_nbin; i++) {
    if (std::isnan(SP(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    if (fdiff(nuisance.photoz[0][0][i], SP(i))) {
      cache_update = 1;
      nuisance.photoz[0][0][i] = SP(i);
    } 
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_photoz_shear = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin lens photo-z shift nuisance.photoz[1][0][i]
// (nz_lens_photoz shifts z before the stretch transform).
//
// Cache invalidation:
// bumps nuisance.random_photoz_clustering when any
// value changed (fdiff); unchanged input leaves the key alone.
//
// Validation: clustering_nbin set and equal to the input size, no NaN
// entries, else critical() + exit(1).
//
// Parameters:
//   CP - per-bin lens photo-z shifts (length clustering_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_clustering_photoz(vector CP)
{
  static constexpr std::string_view fname = "set_nuisance_clustering_photoz"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.clustering_nbin) [[unlikely]] {
    critical(errorns2, fname, "clustering_Nbin", 0);
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(CP.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, CP.n_elem, redshift.clustering_nbin);
    exit(1);
  }

  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(CP(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance);
      exit(1);
    }
    if (fdiff(nuisance.photoz[1][0][i], CP(i))) { 
      cache_update = 1;
      nuisance.photoz[1][0][i] = CP(i);
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_photoz_clustering = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin lens photo-z stretch nuisance.photoz[1][1][i]
// (nz_lens_photoz rescales z around the fiducial bin mean
// redshift.clustering_zdist_zmean by 1/stretch).
//
// Cache invalidation:
// bumps nuisance.random_photoz_clustering when any
// value changed (fdiff); unchanged input leaves the key alone.
//
// Validation: clustering_nbin set and equal to the input size, no NaN
// entries, else critical() + exit(1).
//
// Parameters:
//   CPS - per-bin lens photo-z stretch factors (length clustering_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_clustering_photoz_stretch(vector CPS)
{
  static constexpr std::string_view fname = "set_nuisance_clustering_photoz_stretch"sv;
  debug("{}: {}", fname, errbegins);

  if (0 == redshift.clustering_nbin) [[unlikely]] {
    critical(errorns2, fname, "clustering_Nbin", 0);
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(CPS.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, CPS.n_elem, redshift.clustering_nbin);
    exit(1);
  }

  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(CPS(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance);
      exit(1);
    }
    if (fdiff(nuisance.photoz[1][1][i], CPS(i))) {
      cache_update = 1;
      nuisance.photoz[1][1][i] = CPS(i);
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_photoz_clustering = RandomNumber::get_instance().get();
  }
  debug("{}: Ends", "set_nuisance_clustering_photoz_stretch");
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin linear galaxy bias b1 = nuisance.gb[0][i] (row layout in
// the block comment below).
//
// Cache invalidation:
// bumps nuisance.random_galaxy_bias when any value
// changed (fdiff); unchanged input leaves the key alone.
//
// Validation: clustering_nbin set and equal to the input size, no NaN
// entries, else critical() + exit(1).
//
// Parameters:
//   B1 - per-bin linear bias b1 (length clustering_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_linear_bias(vector B1)
{
  static constexpr std::string_view fname = "set_nuisance_linear_bias"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.clustering_nbin) [[unlikely]] {
    critical(errorns2, fname, "clustering_Nbin", 0); exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B1.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, B1.n_elem, redshift.clustering_nbin);
    exit(1);
  }
  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias  in clustering bin i
  //            b[4][i]: amplitude of magnification bias in clustering bin i
  //            b[5][i]: nonlocal bK galaxy bias in clustering bin i
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B1(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    if(fdiff(nuisance.gb[0][i], B1(i))) {
      cache_update = 1;
      nuisance.gb[0][i] = B1(i);
    } 
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin quadratic bias b2 = nuisance.gb[1][i] and derive the
// coevolution tidal bias gb[2][i] = bs2 = -(4/7)(b1 - 1) (zero when b2 is
// zero). Both writes happen only for bins whose b2 changed (fdiff on B2).
//
// Cache invalidation:
// bumps nuisance.random_galaxy_bias when any b2
// changed; unchanged input leaves the key alone.
//
// Validation: clustering_nbin set, both inputs of that size, no NaN
// entries, else critical() + exit(1).
//
// Parameters:
//   B1 - per-bin linear bias (enters only the derived bs2)
//   B2 - per-bin quadratic bias b2 (length clustering_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_nonlinear_bias(vector B1, vector B2)
{
  static constexpr std::string_view fname = "set_nuisance_nonlinear_bias"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.clustering_nbin) [[unlikely]]{
    critical(errorns2, fname, "clustering_Nbin", 0); exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B1.n_elem)) [[unlikely]] {
    critical("{}: {} {}(!= {})",fname, erriiwz, B1.n_elem, redshift.clustering_nbin);
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B2.n_elem)) [[unlikely]] {
    critical(errorsz1d,fname, erriiwz, B2.n_elem, redshift.clustering_nbin); exit(1);
  }
  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias  in clustering bin i 
  //            b[4][i]: amplitude of magnification bias in clustering bin i 
  //            b[5][i]: nonlocal bK galaxy bias in clustering bin i
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B1(i)) || std::isnan(B2(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    // bs2 depends on BOTH inputs: recompute the candidate first, so a
    // B1-only change with B2 fixed still updates the derived tidal bias
    const double bs2 = almost_equal(B2(i), 0.) ? 0 : (-4./7.)*(B1(i)-1.0);
    if (fdiff(nuisance.gb[1][i], B2(i)) || fdiff(nuisance.gb[2][i], bs2)) {
      cache_update = 1;
      nuisance.gb[1][i] = B2(i);
      nuisance.gb[2][i] = bs2;
    }
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin magnification-bias amplitude b_mag = nuisance.gb[4][i].
//
// Cache invalidation:
// bumps nuisance.random_galaxy_bias when any value
// changed (fdiff); unchanged input leaves the key alone.
//
// Validation: clustering_nbin set and equal to the input size, no NaN
// entries, else critical() + exit(1).
//
// Parameters:
//   B_MAG - per-bin magnification-bias amplitude (length clustering_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_magnification_bias(vector B_MAG)
{
  static constexpr std::string_view fname = "set_nuisance_magnification_bias"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.clustering_nbin) [[unlikely]] {
    critical(errorns2, fname, "clustering_Nbin", 0); exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B_MAG.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, B_MAG.n_elem, redshift.clustering_nbin);
    exit(1);
  }
  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias  in clustering bin i 
  //            b[4][i]: amplitude of magnification bias in clustering bin i
  //            b[5][i]: nonlocal bK galaxy bias in clustering bin i
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B_MAG(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    if(fdiff(nuisance.gb[4][i], B_MAG(i))) {
      cache_update = 1;
      nuisance.gb[4][i] = B_MAG(i);
    }
  }
  if(1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the per-bin third-order bias b3nl = nuisance.gb[3][i] and nonlocal
// bias bK = nuisance.gb[5][i].
//
// Cache invalidation:
// bumps nuisance.random_galaxy_bias when any value
// changed (fdiff); unchanged input leaves the key alone.
//
// Validation: clustering_nbin set, both inputs of that size, no NaN
// entries, else critical() + exit(1).
//
// Parameters:
//   B3nl - per-bin third-order bias b3nl (length clustering_nbin)
//   BK   - per-bin nonlocal bias bK (length clustering_nbin)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_nonlocal_bias(vector B3nl, vector BK)
{
  static constexpr std::string_view fname = "set_nuisance_nonlocal_bias"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.clustering_nbin) [[unlikely]] {
    critical(errorns2, fname, "clustering_Nbin", 0); exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(B3nl.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, B3nl.n_elem, redshift.clustering_nbin);
    exit(1);
  }
  if (redshift.clustering_nbin != static_cast<int>(BK.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, BK.n_elem, redshift.clustering_nbin);
    exit(1);
  }
  // GALAXY BIAS ------------------------------------------
  // 1st index: b[0][i]: linear galaxy bias in clustering bin i
  //            b[1][i]: nonlinear b2 galaxy bias in clustering bin i
  //            b[2][i]: leading order tidal bs2 galaxy bias in clustering bin i
  //            b[3][i]: nonlinear b3 galaxy bias  in clustering bin i 
  //            b[4][i]: amplitude of magnification bias in clustering bin i
  //            b[5][i]: nonlocal bK galaxy bias in clustering bin i
  int cache_update = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    if (std::isnan(B3nl(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    if(fdiff(nuisance.gb[3][i], B3nl(i))) {
      cache_update = 1;
      nuisance.gb[3][i] = B3nl(i);
    }
    if (std::isnan(BK(i))) [[unlikely]] {
      critical(errnance2, fname, i, errnance); exit(1);
    }
    if(fdiff(nuisance.gb[5][i], BK(i))) {
      cache_update = 1;
      nuisance.gb[5][i] = BK(i);
    }
  }
  if(1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sampler-facing bias update: b1, b2 (+ derived bs2), b_mag, then b3nl/bK.
// Each stage bumps nuisance.random_galaxy_bias only on change.
//
// Parameters:
//   B1    - per-bin linear bias b1
//   B2    - per-bin quadratic bias b2
//   B_MAG - per-bin magnification-bias amplitude
//   B3nl  - per-bin third-order bias b3nl
//   BK    - per-bin nonlocal bias bK
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_bias_fastpt(vector B1, vector B2, vector B_MAG, vector B3nl, vector BK)
{
  set_nuisance_linear_bias(B1);
  set_nuisance_nonlinear_bias(B1, B2);
  set_nuisance_magnification_bias(B_MAG);
  set_nuisance_nonlocal_bias(B3nl, BK);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Backward-compatible bias update: as set_nuisance_bias_fastpt without the
// nonlocal b3nl/bK stage.
//
// Parameters:
//   B1    - per-bin linear bias b1
//   B2    - per-bin quadratic bias b2
//   B_MAG - per-bin magnification-bias amplitude
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_bias(vector B1, vector B2, vector B_MAG)
{
  set_nuisance_linear_bias(B1);
  set_nuisance_nonlinear_bias(B1, B2);
  set_nuisance_magnification_bias(B_MAG);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the intrinsic-alignment amplitudes for the active nuisance.IA mode
// (array layout in the block comment below).
//
// IA_REDSHIFT_BINNING: per-bin ia[0][i] = A1, ia[1][i] = A2, ia[2][i] =
// b_TA, with a NaN scan. IA_REDSHIFT_EVOLUTION: ia[0][0..1] = (A_ia,
// eta_ia), ia[1][0..1] = (A2_ia, eta_ia_tt), ia[2][0] = b_TA, and the
// pivot nuisance.oneplusz0_ia = 1.62. Every call writes
// nuisance.c1rhocrit_ia = 0.01389; other IA modes change nothing else.
//
// Cache invalidation:
// bumps nuisance.random_ia when any stored value
// changed (fdiff); unchanged input leaves the key alone.
//
// Validation: shear_nbin set and each input at least shear_nbin long, else
// critical() + exit(1).
//
// Parameters:
//   A1  - tidal-alignment amplitudes: per bin, or (A_ia, eta_ia) in
//         slots 0-1 for IA_REDSHIFT_EVOLUTION
//   A2  - tidal-torque amplitudes: per bin, or (A2_ia, eta_ia_tt)
//   BTA - b_TA amplitudes: per bin, or slot 0 only
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_IA(vector A1, vector A2, vector BTA)
{
  static constexpr std::string_view fname = "set_nuisance_IA"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == redshift.shear_nbin) [[unlikely]] {
    critical(errorns2, fname, "shear_Nbin", 0); exit(1);
  }
  if (redshift.shear_nbin > static_cast<int>(A1.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, A1.n_elem, redshift.shear_nbin); exit(1);
  }
  if (redshift.shear_nbin > static_cast<int>(A2.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, A2.n_elem, redshift.shear_nbin); exit(1);
  }
  if (redshift.shear_nbin > static_cast<int>(BTA.n_elem)) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, BTA.n_elem, redshift.shear_nbin); exit(1);
  }
  // INTRINSIC ALIGMENT ------------------------------------------  
  // ia[0][0] = A_ia          if(IA_NLA_LF || IA_REDSHIFT_EVOLUTION)
  // ia[0][1] = eta_ia        if(IA_NLA_LF || IA_REDSHIFT_EVOLUTION)
  // ia[0][2] = eta_ia_highz  if(IA_NLA_LF, Joachimi2012)
  // ia[0][3] = beta_ia       if(IA_NLA_LF, Joachimi2012)
  // ia[0][4] = LF_alpha      if(IA_NLA_LF, Joachimi2012)
  // ia[0][5] = LF_P          if(IA_NLA_LF, Joachimi2012)
  // ia[0][6] = LF_Q          if(IA_NLA_LF, Joachimi2012)
  // ia[0][7] = LF_red_alpha  if(IA_NLA_LF, Joachimi2012)
  // ia[0][8] = LF_red_P      if(IA_NLA_LF, Joachimi2012)
  // ia[0][9] = LF_red_Q      if(IA_NLA_LF, Joachimi2012)
  // ------------------
  // ia[1][0] = A2_ia        if IA_REDSHIFT_EVOLUTION
  // ia[1][1] = eta_ia_tt    if IA_REDSHIFT_EVOLUTION
  // ------------------
  // ia[2][MAX_SIZE_ARRAYS] = b_ta_z[MAX_SIZE_ARRAYS]

  int cache_update = 0;
  nuisance.c1rhocrit_ia = 0.01389;
  
  if (nuisance.IA == IA_REDSHIFT_BINNING)
  {
    for (int i=0; i<redshift.shear_nbin; i++) {
      if (std::isnan(A1(i)) || std::isnan(A2(i)) || std::isnan(BTA(i))) [[unlikely]] {
        critical(errnance2, fname, i, errnance); exit(1);
      }
      if (fdiff(nuisance.ia[0][i],A1(i)) ||
          fdiff(nuisance.ia[1][i],A2(i)) ||
          fdiff(nuisance.ia[2][i],BTA(i)))
      {
        nuisance.ia[0][i] = A1(i);
        nuisance.ia[1][i] = A2(i);
        nuisance.ia[2][i] = BTA(i);
        cache_update = 1;
      }
    }
  }
  else if (nuisance.IA == IA_REDSHIFT_EVOLUTION)
  {
    nuisance.oneplusz0_ia = 1.62;
    if (fdiff(nuisance.ia[0][0],A1(0)) ||
        fdiff(nuisance.ia[0][1],A1(1)) ||
        fdiff(nuisance.ia[1][0],A2(0)) ||
        fdiff(nuisance.ia[1][1],A2(1)) ||
        fdiff(nuisance.ia[2][0],BTA(0)))
    {
      nuisance.ia[0][0] = A1(0);
      nuisance.ia[0][1] = A1(1);
      nuisance.ia[1][0] = A2(0);
      nuisance.ia[1][1] = A2(1);
      nuisance.ia[2][0] = BTA(0);
      cache_update = 1;
    }
  }
  if(1 == cache_update || 1 == force_cache_update_test) {
    nuisance.random_ia = RandomNumber::get_instance().get();
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Declare the lens tomography: redshift.clustering_nbin = Ntomo and
// redshift.clustering_photoz = 4.
//
// Validation: 0 < Ntomo <= MAX_SIZE_ARRAYS, else critical() + exit(1).
//
// Parameters:
//   Ntomo - number of lens tomographic bins
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_lens_sample_size(const int Ntomo)
{
  static constexpr std::string_view fname = "set_lens_sample_size"sv;
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns,fname,"Ntomo",Ntomo,MAX_SIZE_ARRAYS);
    exit(1);
  }
  // photo-z mode flag; no code in cosmolike_core reads it
  redshift.clustering_photoz = 4;
  redshift.clustering_nbin = Ntomo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install the lens n(z) table and derive the per-bin support.
//
// When the table changed (size or any entry, fdiff): rebuilds
// redshift.clustering_zdist_table as (Ntomo + 1) x nzbins with the z grid
// in row Ntomo, sets clustering_zdist_zmin_all = max(z_0, 1e-5) and
// clustering_zdist_zmax_all one grid spacing beyond the last z, and scans
// each column for its support: entries above 0.999e-8 of the column
// maximum, zmin = z of the first such entry, zmax = z of the last (plain
// loops on purpose - see the inline comment). A column with no entry above
// the threshold is critical() + exit(1).
//
// Cache invalidation:
// bumps redshift.random_clustering first, then calls
// nz_lens_photoz(0.1, 0) so the static interpolant rebuilds here,
// single-threaded, and stores clustering_zdist_zmean[k] = zmean(k) (the
// fiducial means the photo-z stretch transform rescales around).
//
// Validation: redshift.clustering_nbin already set and within
// MAX_SIZE_ARRAYS, else critical() + exit(1).
//
// Parameters:
//   input_table - (nz x (Ntomo+1)) table: column 0 = z grid, column k+1 =
//                 n(z) of lens bin k (read_nz_sample layout)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_lens_sample(arma::Mat<double> input_table)
{
  static constexpr std::string_view fname = "set_lens_sample"sv;
  debug("{}: {}", fname, errbegins);

  const int Ntomo = redshift.clustering_nbin;
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "Ntomo", Ntomo, MAX_SIZE_ARRAYS); exit(1);
  }

  int cache_update = 0;
  if (redshift.clustering_nzbins != static_cast<int>(input_table.n_rows) ||
      NULL == redshift.clustering_zdist_table) {
    cache_update = 1;
  }
  else
  {
    for (int i=0; i<redshift.clustering_nzbins; i++) {
      double** tab = redshift.clustering_zdist_table;        // alias
      double* z_v = redshift.clustering_zdist_table[Ntomo];  // alias

      if (fdiff(z_v[i], input_table(i,0))) {
        cache_update = 1;
        break;
      }
      for (int k=0; k<Ntomo; k++) {  
        if (fdiff(tab[k][i], input_table(i,k+1))) {
          cache_update = 1;
          goto jump;
        }
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test)
  {
    redshift.clustering_nzbins = input_table.n_rows;
    const int nzbins = redshift.clustering_nzbins;    // alias

    if (redshift.clustering_zdist_table != NULL) {
      free(redshift.clustering_zdist_table);
    }
    redshift.clustering_zdist_table = (double**) malloc2d(Ntomo + 1, nzbins);
    
    double** tab = redshift.clustering_zdist_table;        // alias
    double* z_v = redshift.clustering_zdist_table[Ntomo];  // alias
    
    for (int i=0; i<nzbins; i++) {
      z_v[i] = input_table(i,0);
      for (int k=0; k<Ntomo; k++) {
        tab[k][i] = input_table(i,k+1);
      }
    }
    
    redshift.clustering_zdist_zmin_all = fmax(z_v[0], 1.e-5);
    
    redshift.clustering_zdist_zmax_all = z_v[nzbins-1] + 
      (z_v[nzbins-1] - z_v[0]) / ((double) nzbins - 1.);

    for (int k=0; k<Ntomo; k++) { // Set tomography bin boundaries
      // The bin support is where n(z) exceeds 0.999e-8 of its maximum.
      // Plain loops: under COSMOLIKE_AGGRESSIVE_MODE (-ffast-math, LTO)
      // the equivalent arma::find(nofz > c*nofz.max()) expression has
      // returned an empty index list for a valid n(z) column.
      double nzmax = input_table(0, k+1);
      for (int i=1; i<nzbins; i++) {
        nzmax = fmax(nzmax, input_table(i, k+1));
      }
      int first = -1;
      int last  = -1;
      for (int i=0; i<nzbins; i++) {
        if (input_table(i, k+1) > 0.999e-8*nzmax) {
          if (first < 0) first = i;
          last = i;
        }
      }
      if (first < 0) [[unlikely]] {
        critical("{}: n(z) of bin {} has no positive entry", fname, k);
        exit(1);
      }
      redshift.clustering_zdist_zmin[k] = z_v[first];
      redshift.clustering_zdist_zmax[k] = z_v[last];
    }
    // READ THE N(Z) FILE ENDS ------------
    redshift.random_clustering = RandomNumber::get_instance().get();

    nz_lens_photoz(0.1, 0); // init static variables

    for (int k=0; k<Ntomo; k++) {
      redshift.clustering_zdist_zmean[k] = zmean(k);
      debug("{}: bin {} - {} = {}.", fname, k, "<z_s>", zmean(k));
    }
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Declare the source tomography: redshift.shear_nbin = Ntomo and
// redshift.shear_photoz = 4.
//
// Validation: 0 < Ntomo <= MAX_SIZE_ARRAYS, else critical() + exit(1).
//
// Parameters:
//   Ntomo - number of source tomographic bins
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_source_sample_size(const int Ntomo)
{
  static constexpr std::string_view fname = "set_source_sample_size"sv;
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "Ntomo", Ntomo,  MAX_SIZE_ARRAYS);
    exit(1);
  } 
  // photo-z mode flag; no code in cosmolike_core reads it
  redshift.shear_photoz = 4;
  redshift.shear_nbin = Ntomo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Install the source n(z) table and derive the per-bin support (source
// twin of set_lens_sample).
//
// When the table changed (size or any entry, fdiff): rebuilds
// redshift.shear_zdist_table as (Ntomo + 1) x nzbins with the z grid in
// row Ntomo, sets shear_zdist_zmin_all = max(z_0, 1e-5) and
// shear_zdist_zmax_all one grid spacing beyond the last z, and scans each
// column for its support: entries above 0.999e-8 of the column maximum,
// zmin = z of the first such entry (clamped to at least 1.001e-5), zmax =
// z of the last (plain loops on purpose - see the inline comment). A
// column with no entry above the threshold is critical() + exit(1), and so
// is a per-bin range outside the global [zmin_all, zmax_all].
//
// Cache invalidation:
// warms the static interpolant with
// nz_source_photoz(0.1, 0), prints zmean_source(k) at debug level, then
// bumps redshift.random_shear.
//
// Validation: redshift.shear_nbin already set and within MAX_SIZE_ARRAYS,
// else critical() + exit(1).
//
// Parameters:
//   input_table - (nz x (Ntomo+1)) table: column 0 = z grid, column k+1 =
//                 n(z) of source bin k (read_nz_sample layout)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_source_sample(arma::Mat<double> input_table)
{
  static constexpr std::string_view fname = "set_source_sample"sv;
  debug("{}: {}", fname, errbegins);

  const int Ntomo = redshift.shear_nbin;
  if (!(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "Ntomo", Ntomo, MAX_SIZE_ARRAYS); exit(1);
  } 

  int cache_update = 0;
  if (redshift.shear_nzbins != static_cast<int>(input_table.n_rows) ||
      NULL == redshift.shear_zdist_table) {
    cache_update = 1;
  }
  else
  {
    double** tab = redshift.shear_zdist_table;         // alias  
    double* z_v  = redshift.shear_zdist_table[Ntomo];  // alias
    for (int i=0; i<redshift.shear_nzbins; i++)  {
      if (fdiff(z_v[i], input_table(i,0))) {
        cache_update = 1;
        goto jump;
      }
      for (int k=0; k<Ntomo; k++) {
        if (fdiff(tab[k][i], input_table(i,k+1))) {
          cache_update = 1;
          goto jump;
        }
      }
    }
  }

  jump:

  if (1 == cache_update || 1 == force_cache_update_test)
  {
    redshift.shear_nzbins = input_table.n_rows;
    const int nzbins = redshift.shear_nzbins; // alias

    if (redshift.shear_zdist_table != NULL) {
      free(redshift.shear_zdist_table);
    }
    redshift.shear_zdist_table = (double**) malloc2d(Ntomo + 1, nzbins);

    double** tab = redshift.shear_zdist_table;        // alias  
    double* z_v = redshift.shear_zdist_table[Ntomo];  // alias
    for (int i=0; i<nzbins; i++) {
      z_v[i] = input_table(i,0);
      for (int k=0; k<Ntomo; k++) {
        tab[k][i] = input_table(i,k+1);
      }
    }
  
    redshift.shear_zdist_zmin_all = fmax(z_v[0], 1.e-5);
    redshift.shear_zdist_zmax_all = z_v[nzbins-1] + (z_v[nzbins-1] - z_v[0]) / ((double) nzbins - 1.);

    for (int k=0; k<Ntomo; k++)  { // Set tomography bin boundaries
      // The bin support is where n(z) exceeds 0.999e-8 of its maximum.
      // Plain loops: under COSMOLIKE_AGGRESSIVE_MODE (-ffast-math, LTO)
      // the equivalent arma::find(nofz > c*nofz.max()) expression has
      // returned an empty index list for a valid n(z) column.
      double nzmax = input_table(0, k+1);
      for (int i=1; i<nzbins; i++) {
        nzmax = fmax(nzmax, input_table(i, k+1));
      }
      int first = -1;
      int last  = -1;
      for (int i=0; i<nzbins; i++) {
        if (input_table(i, k+1) > 0.999e-8*nzmax) {
          if (first < 0) first = i;
          last = i;
        }
      }
      if (first < 0) [[unlikely]] {
        critical("{}: n(z) of bin {} has no positive entry", fname, k);
        exit(1);
      }
      redshift.shear_zdist_zmin[k] = fmax(z_v[first], 1.001e-5);
      redshift.shear_zdist_zmax[k] = z_v[last];
    }
  
    // READ THE N(Z) FILE ENDS ------------
    if (redshift.shear_zdist_zmax_all < redshift.shear_zdist_zmax[Ntomo-1] || 
        redshift.shear_zdist_zmin_all > redshift.shear_zdist_zmin[0]) [[unlikely]] {
      critical("{}: {} = {}, {} = {}", fname, "zhisto_min", 
          redshift.shear_zdist_zmin_all, "zhisto_max", redshift.shear_zdist_zmax_all);
      critical("{}: {} = {}, {} = {}", fname, "shear_zdist_zmin[0]", 
          redshift.shear_zdist_zmin[0], "shear_zdist_zmax[redshift.shear_nbin-1]", 
          redshift.shear_zdist_zmax[Ntomo-1]);
      exit(1);
    } 
    // bump the key BEFORE the warm-up, as set_lens_sample does: the
    // warm-up and the debug prints must see the sample just installed
    redshift.random_shear = RandomNumber::get_instance().get();
    nz_source_photoz(0.1, 0); // init static variables
    for (int k=0; k<Ntomo; k++) {
      debug("{}: bin {} - {} = {}.", fname, k, "<z_s>", zmean_source(k));
    }
  }
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GET FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Baryonic suppression ratio P_hydro / P_DMO at (log10 k [h/Mpc], a), from
// the 2D interpolant loaded by init_baryons_contamination
// (PkRatio_baryons; returns 1 when no scenario is loaded).
//
// Parameters:
//   log10k - log10 of the wavenumber (h/Mpc)
//   a      - scale factor
//
// Returns:
//   P_hydro / P_DMO at (k, a); 1 when no scenario is loaded
// ---------------------------------------------------------------------------
double get_baryon_power_spectrum_ratio(const double log10k, const double a)
{
  const double KNL = pow(10.0, log10k)*cosmology.coverH0;
  return PkRatio_baryons(KNL, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// COMPUTE FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Point-mass contribution to gamma_t for lens bin zl, source bin zs at
// angle theta (rad); forwards to PointMass::get_pm with the amplitudes
// stored via PointMass::set_pm_vector.
//
// Parameters:
//   zl    - lens tomographic bin
//   zs    - source tomographic bin
//   theta - angular scale (rad)
//
// Returns:
//   the point-mass gamma_t contribution for the (zl, zs) pair at theta
// ---------------------------------------------------------------------------
double compute_pm(const int zl, const int zs, const double theta)
{
  return PointMass::get_instance().get_pm(zl, zs, theta);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Area-weighted centers of the log-spaced angular bins:
//
//   theta_i = (2/3) (th_max^3 - th_min^3) / (th_max^2 - th_min^2)
//
// over Ntable.Ntheta bins spanning [Ntable.vtmin, Ntable.vtmax].
//
// Validation: Ntheta and the (vtmin, vtmax) range must be set (via
// init_binning_real_space), else critical() + exit(1).
//
// Parameters:
//   (none)
//
// Returns:
//   vector of Ntable.Ntheta bin-center angles (rad)
// ---------------------------------------------------------------------------
vector compute_binning_real_space()
{
  static constexpr std::string_view fname = "compute_binning_real_space"sv;
  debug("{}: {}", fname, errbegins);
  if (0 == Ntable.Ntheta)  [[unlikely]] {
    critical(errornset, fname, "Ntable.Ntheta"); exit(1);
  }
  if (!(Ntable.vtmax > Ntable.vtmin))  [[unlikely]] {
    critical(errornset, fname, "Ntable.vtmax and Ntable.vtmin"); exit(1);
  }
  const double logvtmin = std::log(Ntable.vtmin);
  const double logvtmax = std::log(Ntable.vtmax);
  const double logdt=(logvtmax - logvtmin)/Ntable.Ntheta;
  constexpr double fac = (2./3.);

  vector theta(Ntable.Ntheta, arma::fill::zeros);
  for (int i=0; i<Ntable.Ntheta; i++) {
    const double thetamin = std::exp(logvtmin + (i + 0.)*logdt);
    const double thetamax = std::exp(logvtmin + (i + 1.)*logdt);
    theta(i) = fac * (std::pow(thetamax,3) - std::pow(thetamin,3)) /
                     (thetamax*thetamax    - thetamin*thetamin);
    debug("{}: Bin {:d} - {} = {:.4e}, {} = {:.4e} and {} = {:.4e}",
        fname, i, "theta_min [rad]", thetamin, "theta [rad]", 
        theta(i), "theta_max [rad]", thetamax);
  }
  debug("{}: {}", fname, errends);
  return theta;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Add the baryon principal-component expansion to a data vector:
//
//   dv(j) += sum_i Q(i) * PC(j, i)   wherever the IP mask is 1
//
// with the PCs from BaryonScenario::set_pcs; masked-out entries pass
// through unchanged.
//
// Validation: PCs set, PC column count >= size of Q, PC row count == size
// of dv, else critical() + exit(1).
//
// Parameters:
//   Q  - PC amplitudes (one per retained principal component)
//   dv - data vector to contaminate (full length)
//
// Returns:
//   the contaminated data vector (modified copy of dv)
// ---------------------------------------------------------------------------
vector compute_add_baryons_pcs(vector Q, vector dv)
{
  static constexpr std::string_view fname = "compute_add_baryons_pcs"sv;
  debug("{}: {}", fname, errbegins);
  BaryonScenario& bs = BaryonScenario::get_instance();
  if (!bs.is_pcs_set()) [[unlikely]] {
    critical(errornset, fname, "baryon PCs"); exit(1);
  }
  if (bs.get_pcs().row(0).n_elem < Q.n_elem) [[unlikely]] {
    critical("{}: invalid PC amplitude vector / eigenvectors", fname); exit(1);
  }
  if (bs.get_pcs().col(0).n_elem != dv.n_elem) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, bs.get_pcs().col(0).n_elem, dv.n_elem); 
    exit(1);
  }
  for (int j=0; j<static_cast<int>(dv.n_elem); j++) {
    for (int i=0; i<static_cast<int>(Q.n_elem); i++) {
      if (IP::get_instance().get_mask(j)) {
        dv(j) += Q(i) * bs.get_pcs(j, i);
      }
    }
  }
  debug("{}: {}", fname, errends);
  return dv;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IP MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load the measured data vector (column 1 of the file) into the
// full-length masked view and the squeezed (masked-entries-only) view.
//
// Writes data_masked_(i) = value * mask(i) over the full ndata_ range and
// data_masked_sqzd_ at the compact index get_index_sqzd(i) for unmasked
// entries. Requires set_mask first (ndata_ and the index map come from
// it). Sets is_data_set_.
//
// Validation: mask set, file row count == ndata_, consistent squeezed
// indices, else critical() + exit(1).
//
// Parameters:
//   datavector_filename - data-vector file (column 1 = value, rows = ndata_)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void IP::set_data(std::string datavector_filename)
{
  static constexpr std::string_view fname = "IP::set_data"sv;
  debug("{}: {}", fname, errbegins);
  if (!(this->is_mask_set_)) {
    critical(errornset, fname, "mask"); exit(1);
  }

  this->data_masked_.set_size(this->ndata_);
    
  this->data_masked_sqzd_.set_size(this->ndata_sqzd_);

  this->data_filename_ = datavector_filename;

  matrix table = read_table(datavector_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_) {
    critical("{}: inconsistent data vector", fname); exit(1);
  }
  for(int i=0; i<like.Ndata; i++) {
    this->data_masked_(i) = table(i,1);
    this->data_masked_(i) *= this->get_mask(i);
    if(this->get_mask(i) == 1) {
      if(this->get_index_sqzd(i) < 0) {
        critical("{}: {} mask operation", fname, errleii); exit(1);
      }
      this->data_masked_sqzd_(this->get_index_sqzd(i)) = this->data_masked_(i);
    }
  }
  this->is_data_set_ = true;
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// Load the covariance, mask and invert it, and build the squeezed copies.
//
// Accepted formats (read_table columns): 3 = (i, j, cov); 4 = (i, j,
// gauss, non-gauss) summed; 10 = legacy CosmoCov layout, cov = col 8 +
// col 9. The stored triangle is symmetrized; off-diagonal entries are
// zeroed when either index is masked.
//
// When the kk bandpower probe is configured, the trailing kk block (last
// nbins_kk rows/columns) is divided by the Hartlap alpha from
// init_cmb_auto_bandpower before inversion:
// alpha = (N_sim - N_data - 2)/(N_sim - 1) < 1 is the Hartlap et al.
// 2007 debias of a simulation-estimated inverse covariance, so
// inflating the covariance block by 1/alpha here shrinks its inverse
// by alpha after the joint inversion.
//
// Stages after assembly: eigenvalue scan (any negative eigenvalue is
// critical() + exit(1)), arma::inv, re-masking of the inverse (masked
// rows/columns zeroed, diagonal included, so they cannot leak into chi2),
// then compaction of covariance and inverse into the ndata_sqzd_ square
// matrices. Sets is_inv_cov_set_.
//
// Parameters:
//   cov_filename - covariance file in one of the accepted column layouts
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void IP::set_inv_cov(std::string cov_filename)
{
  static constexpr std::string_view fname = "IP::set_inv_cov"sv;
  debug("{}: {}", fname, errbegins);
  if (!(this->is_mask_set_)) [[unlikely]] {
    critical(errornset, fname, "mask"); exit(1);
  }

  this->cov_filename_ = cov_filename;
  matrix table = read_table(cov_filename); 
  
  this->cov_masked_.set_size(this->ndata_, this->ndata_);
  this->cov_masked_.zeros();
  this->cov_masked_sqzd_.set_size(this->ndata_sqzd_, this->ndata_sqzd_);
  this->inv_cov_masked_sqzd_.set_size(this->ndata_sqzd_, this->ndata_sqzd_);

  switch (table.n_cols)
  {
    case 3:
    {
      #pragma omp parallel for schedule(static)
      for (int i=0; i<static_cast<int>(table.n_rows); i++) {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));
        this->cov_masked_(j,k) = table(i,2);
        if (j!=k) {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);
          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
        }
      };
      break;
    }
    case 4:
    {
      #pragma omp parallel for schedule(static)
      for (int i=0; i<static_cast<int>(table.n_rows); i++) {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));
        this->cov_masked_(j,k) = table(i,2) + table(i,3);
        if (j!=k) {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);
          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
        }
      };
      break;
    }
    case 10:
    {
      #pragma omp parallel for schedule(static)
      for (int i=0; i<static_cast<int>(table.n_rows); i++) {
        const int j = static_cast<int>(table(i,0));
        const int k = static_cast<int>(table(i,1));
        this->cov_masked_(j,k) = table(i,8) + table(i,9);
        if (j!=k) {
          // apply mask to off-diagonal covariance elements
          this->cov_masked_(j,k) *= this->get_mask(j);
          this->cov_masked_(j,k) *= this->get_mask(k);
          // m(i,j) = m(j,i)
          this->cov_masked_(k,j) = this->cov_masked_(j,k);
        }
      }
      break;
    }
    default:
    {
      critical("{}: invalid format for cov file = {}", fname, cov_filename);
      exit(1);
    }
  }

  if (1 == IPCMB::get_instance().is_kk_bandpower())
  {
    IPCMB& cmb = IPCMB::get_instance();
    const int N5x2pt = this->ndata_ - cmb.get_nbins_kk_bandpower();
    if (!(N5x2pt>0)) [[unlikely]] {
      critical("{}, {}: inconsistent dv size and number of binning in (kk)",
        fname, this->ndata_, cmb.get_nbins_kk_bandpower()); exit(1);
    }
    const double hartlap_factor = cmb.get_alpha_Hartlap_cov_kkkk();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=N5x2pt; i<this->ndata_; i++) {
      for (int j=N5x2pt; j<this->ndata_; j++) {
        this->cov_masked_(i,j) /= hartlap_factor;
      }
    }
  }

  vector eigvals = arma::eig_sym(this->cov_masked_);
  for(int i=0; i<this->ndata_; i++) {
    if(eigvals(i) < 0) [[unlikely]] {
      critical("{}: masked cov not positive definite", fname); exit(1);
    }
  }

  this->inv_cov_masked_ = arma::inv(this->cov_masked_);

  // apply mask again to make sure numerical errors in matrix inversion don't 
  // cause problems. Also, set diagonal elements corresponding to datavector
  // elements outside mask to 0, so that they don't contribute to chi2
  #pragma omp parallel for schedule(static)
  for (int i=0; i<this->ndata_; i++) {
    this->inv_cov_masked_(i,i) *= this->get_mask(i)*this->get_mask(i);
    for (int j=0; j<i; j++) {
      this->inv_cov_masked_(i,j) *= this->get_mask(i)*this->get_mask(j);
      this->inv_cov_masked_(j,i) = this->inv_cov_masked_(i,j);
    }
  };
  
  #pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<this->ndata_; i++)
  {
    for(int j=0; j<this->ndata_; j++)
    {
      if((this->mask_(i)>0.99) && (this->mask_(j)>0.99)) {
        if(this->get_index_sqzd(i) < 0) [[unlikely]] {
          critical("{}: {} mask operation", fname, errleii); exit(1);
        }
        if(this->get_index_sqzd(j) < 0) [[unlikely]] {
          critical("{}: {} mask operation", fname, errleii); exit(1);
        }
        const int idxa = this->get_index_sqzd(i);
        const int idxb = this->get_index_sqzd(j);
        this->cov_masked_sqzd_(idxa,idxb) = this->cov_masked_(i,j);
        this->inv_cov_masked_sqzd_(idxa,idxb) = this->inv_cov_masked_(i,j);
      }
    }
  }
  this->is_inv_cov_set_ = true;
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// chi2 of a theory vector against the stored data on the squeezed views:
//
//   delta = sqzd(theory) - data_masked_sqzd_
//   chi2  = delta^T inv_cov_masked_sqzd_ delta
//
// Masked entries never enter (the squeezed views drop them).
//
// Validation: data, mask and inverse covariance set; theory vector of full
// length like.Ndata; a negative chi2 is critical() + exit(1).
//
// Parameters:
//   datavector - theory vector at full (unmasked) length like.Ndata
//
// Returns:
//   chi2 (non-negative)
// ---------------------------------------------------------------------------
double IP::get_chi2(vector datavector) const
{
  static constexpr std::string_view fname = "IP::get_chi2"sv;
  debug("{}: {}", fname, errbegins);
  if (!(this->is_data_set_)) [[unlikely]] {
    critical(errornset, fname, "data_vector"); exit(1);
  }
  if (!(this->is_mask_set_)) [[unlikely]] {
    critical(errornset, fname, "mask"); exit(1);
  }
  if (!(this->is_inv_cov_set_)) [[unlikely]] {
    critical(errornset, fname, "inv_cov"); exit(1);
  }
  if (static_cast<int>(datavector.n_elem) != like.Ndata) [[unlikely]] { 
    critical(errorsz1d, fname, erriiwz, datavector.n_elem, like.Ndata); exit(1);
  }
  /*
  double chi2 = 0.0;
  #pragma omp parallel for collapse (2) reduction(+:chi2) schedule(static)
  for (int i=0; i<like.Ndata; i++) {
    for (int j=0; j<like.Ndata; j++) {
      if (this->get_mask(i) && this->get_mask(j)) {
        const double x = datavector(i) - this->get_dv_masked(i);
        const double y = datavector(j) - this->get_dv_masked(j);
        chi2 += x*this->get_inv_cov_masked(i,j)*y;
      }
    }
  }*/

  const arma::Col<double> delta = this->sqzd_theory_data_vector(datavector) - 
                                  this->data_masked_sqzd_;
  const double chi2 = arma::dot(delta, this->inv_cov_masked_sqzd_ * delta);

  if (chi2 < 0.0) [[unlikely]] {
    critical("{}: chi2 = {} (invalid)", fname, chi2); exit(1);
  }
  debug("{}: {}", fname, errends);
  return chi2;
}

// ---------------------------------------------------------------------------
// Scatter a squeezed vector back to full length: unmasked entries return
// to their original positions, masked entries are 0 (inverse of
// sqzd_theory_data_vector).
//
// Validation: input length == ndata_sqzd_ and consistent squeezed indices,
// else critical() + exit(1).
//
// Parameters:
//   input - squeezed vector (length ndata_sqzd_)
//
// Returns:
//   the full-length vector (masked entries zero)
// ---------------------------------------------------------------------------
vector IP::expand_theory_data_vector_from_sqzd(vector input) const
{
  static constexpr std::string_view fname = "IP::expand_theory_data_vector_from_sqzd"sv;
  debug("{}: {}", fname, errbegins);
  if (this->ndata_sqzd_ != static_cast<int>(input.n_elem)) [[unlikely]] {
    critical("{}: invalid input data vector", fname); exit(1);
  }
  vector result(this->ndata_, arma::fill::zeros);
  for(int i=0; i<this->ndata_; i++) {
    if(this->mask_(i) > 0.99) {
      if(this->get_index_sqzd(i) < 0) [[unlikely]] {
        critical("{}: {} mask operation", fname, errleii); exit(1);
      }
      result(i) = input(this->get_index_sqzd(i));
    }
  }
  debug("{}: {}", fname, errends);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Compact a full-length vector to its unmasked entries, ordered by
// get_index_sqzd (the layout data_masked_sqzd_ and the squeezed
// covariance use).
//
// Validation: input length == ndata_, else critical() + exit(1).
//
// Parameters:
//   input - full-length vector (length ndata_)
//
// Returns:
//   the squeezed vector of unmasked entries (length ndata_sqzd_)
// ---------------------------------------------------------------------------
vector IP::sqzd_theory_data_vector(vector input) const
{
  static constexpr std::string_view fname = "IP::sqzd_theory_data_vector"sv;
  debug("{}: {}", fname, errbegins);
  if (this->ndata_ != static_cast<int>(input.n_elem)) [[unlikely]] {
    critical("{}: invalid input data vector", fname); exit(1);
  }
  vector result(this->ndata_sqzd_, arma::fill::zeros);
  for (int i=0; i<this->ndata_; i++) {
    if (this->get_mask(i) > 0.99) {
      result(this->get_index_sqzd(i)) = input(i);
    }
  }
  debug("{}: {}", fname, errends);
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/*
// ---------------------------------------------------------------------------
// Disabled code: the enclosing block comment keeps this RealData method out
// of the build.
//
// Point-mass marginalization of gamma_t: updates the masked inverse
// covariance in place with the Sherman-Morrison-Woodbury identity
//
//   invC -= invC U (I + U^T invC U)^-1 U^T invC
//
// with the (ndata x Nlens) template U read from U_PMmarg_file as (row,
// lens bin, value) triples, masked rows zeroed. Checks the central block
// and the corrected inverse for positive definiteness, then refreshes the
// reduced-dimension covariance and inverse copies.
//
// Parameters:
//   U_PMmarg_file - three-column (row index, lens bin, value) table for U
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void ima::RealData::set_PMmarg(std::string U_PMmarg_file)
{
  if (!(this->is_mask_set_))
  {
    critical(
      errornset, "set_PMmarg", "mask"
    );
    exit(1);
  }

  arma::Mat<double> table = ima::read_table(U_PMmarg_file);
  if (table.n_cols!=3){
    critical(
      "\x1b[90m{}\x1b[0m: U_PMmarg_file should has three columns, but has {}!"
      "set_PMmarg", table.n_cols);
    exit(1);
  }
  // U has shape of Ndata x Nlens
  arma::Mat<double> U;
  U.set_size(this->ndata_, tomo.clustering_Nbin);
  U.zeros();
  for (int i=0; i<static_cast<int>(table.n_rows); i++)
  {
    const int j = static_cast<int>(table(i,0));
    const int k = static_cast<int>(table(i,1));
    U(j,k) = static_cast<double>(table(i,2)) * this->get_mask(j);
  };
  // Calculate precision matrix correction
  // invC * U * (I+UT*invC*U)^-1 * UT * invC
  arma::Mat<double> iden = arma::eye<arma::Mat<double>>(tomo.clustering_Nbin, tomo.clustering_Nbin);
  arma::Mat<double> central_block = iden + U.t() * this->inv_cov_masked_ * U;
  // test positive-definite
  vector eigvals = arma::eig_sym(central_block);
  for(int i=0; i<tomo.clustering_Nbin; i++)
  {
    if(eigvals(i)<=0.0){
      critical("{}: central block not positive definite!", "set_PMmarg");
      exit(-1);
    }
  }
  arma::Mat<double> invcov_PMmarg = this->inv_cov_masked_ * U * arma::inv_sympd(central_block) * U.t() * this->inv_cov_masked_; 
  //invcov_PMmarg.save("PMmarg_invcov_corr.h5", arma::hdf5_binary);
  // add the PM correction to inverse covariance
  for (int i=0; i<this->ndata_; i++)
  {
    invcov_PMmarg(i,i) *= this->get_mask(i);
    this->inv_cov_masked_(i,i) -= invcov_PMmarg(i,i);
    for (int j=0; j<i; j++)
    {
      double corr = this->get_mask(i)*this->get_mask(j)*(invcov_PMmarg(i,j)+invcov_PMmarg(j,i))/2.0;
      this->inv_cov_masked_(i,j) -= corr;
      this->inv_cov_masked_(j,i) -= corr;
    }
  }
  // examine again the positive-definite-ness
  vector eigvals_corr = arma::eig_sym(this->inv_cov_masked_);
  for(int i=0; i<tomo.clustering_Nbin; i++)
  {
    if(eigvals(i)<0){
      critical("{}: PM-marged invcov not positive definite!", "set_PMmarg");
      exit(-1);
    }
  }

  // Update the reduced covariance and precision matrix
  for(int i=0; i<this->ndata_; i++)
  {
    for(int j=0; j<this->ndata_; j++)
    {
      if((this->mask_(i)>0.99) && (this->mask_(j)>0.99))
      {
        if(this->get_index_reduced_dim(i) < 0)
        {
          critical("\x1b[90m{}\x1b[0m: logical error, internal"
            " inconsistent mask operation", "set_PMmarg");
          exit(1);
        }
        if(this->get_index_reduced_dim(j) < 0)
        {
          critical("\x1b[90m{}\x1b[0m: logical error, internal"
            " inconsistent mask operation", "set_PMmarg");
          exit(1);
        }

        this->cov_masked_reduced_dim_(this->get_index_reduced_dim(i),
          this->get_index_reduced_dim(j)) = this->cov_masked_(i,j);

        this->inv_cov_masked_reduced_dim_(this->get_index_reduced_dim(i),
          this->get_index_reduced_dim(j)) = this->inv_cov_masked_(i,j);
      }
    }
  }
  //this->inv_cov_masked_.save("cocoa_invcov_PMmarg_masked.h5",arma::hdf5_binary);
}
*/

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IPCMB MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load the HealPix window applied to the w_xk real-space cross
// correlation: cmb.healpixwin[l] = column 1 of the file,
// cmb.healpixwin_ncls = row count. Sets is_wxk_healpix_window_set_.
//
// Parameters:
//   healpixwin_filename - table whose column 1 is the window (row = l)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void IPCMB::set_wxk_healpix_window(std::string healpixwin_filename) {
  static constexpr std::string_view fname = "IPCMB::set_wxk_healpix_window"sv;
  debug("{}: {}", fname, errbegins);
  matrix table = read_table(healpixwin_filename);
  this->params_->healpixwin_ncls = static_cast<int>(table.n_rows);
  if (this->params_->healpixwin != NULL) {
    free(this->params_->healpixwin);
  }
  this->params_->healpixwin = (double*) malloc1d(this->params_->healpixwin_ncls);
  for (int i=0; i<this->params_->healpixwin_ncls; i++) {
    this->params_->healpixwin[i] = static_cast<double>(table(i,1));
  }
  this->is_wxk_healpix_window_set_ = true;
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load the kk bandpower binning matrix, nbins x (lmax - lmin + 1), into
// cmb.binning_matrix_kk. Requires set_kk_binning_bandpower first (sizes
// and the bandpower flag), else critical() + exit(1). Sets
// is_kk_binning_matrix_set_.
//
// Parameters:
//   binned_matrix_filename - file with the nbins x (lmax - lmin + 1) matrix
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void IPCMB::set_kk_binning_mat(std::string binned_matrix_filename)
{
  static constexpr std::string_view fname = "IPCMB::set_kk_binning_mat"sv;
  debug("{}: {}", fname, errbegins);
  if(!this->is_kk_bandpower_) [[unlikely]] {
    critical(erroric0, fname, "is_kk_bandpower"); exit(1);
  }
  matrix table = read_table(binned_matrix_filename);

  const int nbp  = this->get_nbins_kk_bandpower();
  const int lmax = this->get_lmax_kk_bandpower();
  const int lmin = this->get_lmin_kk_bandpower();
  const int ncl  = lmax - lmin + 1;
  
  if (this->params_->binning_matrix_kk != NULL) {
    free(this->params_->binning_matrix_kk);
  }
  this->params_->binning_matrix_kk = (double**) malloc2d(nbp, ncl);
    
  #pragma omp parallel for schedule(static)
  for (int i=0; i<nbp; i++) {
    for (int j=0; j<ncl; j++) {
      this->params_->binning_matrix_kk[i][j] = table(i,j);
    }
  }
  debug("{}: kk binning matrix has {} x {} elements", fname, nbp, ncl);
  debug("{}: {}", fname, errends);
  this->is_kk_binning_matrix_set_ = true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load the per-band kk theory offsets (column 0 of the file), or zeros
// when the filename is empty. Requires set_kk_binning_bandpower first,
// else critical() + exit(1). Sets is_kk_offset_set_.
//
// Parameters:
//   theory_offset_filename - column-0 offsets file; "" installs zeros
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void IPCMB::set_kk_theory_offset(std::string theory_offset_filename)
{
  static constexpr std::string_view fname = "IPCMB::set_kk_theory_offset"sv;
  debug("{}: {}", fname, errbegins);
  if(!this->is_kk_bandpower_) [[unlikely]] {
    critical(erroric0, fname, "is_kk_bandpower"); exit(1);
  }
  const int nbp = this->get_nbins_kk_bandpower();
  if (this->params_->theory_offset_kk != NULL) {
    free(this->params_->theory_offset_kk);
  }
  this->params_->theory_offset_kk = (double*) malloc1d(nbp);

  if (!theory_offset_filename.empty()) {
    matrix table = read_table(theory_offset_filename);
    for (int i=0; i<nbp; i++) {
      this->params_->theory_offset_kk[i] = static_cast<double>(table(i,0));
    }
  }
  else {
    for (int i=0; i<nbp; i++) {
      this->params_->theory_offset_kk[i] = 0.0;
    }
  }
  debug("{}: CMB theory offset has {} elements", fname, nbp);
  debug("{}: {}", fname, errends);
  this->is_kk_offset_set_ = true;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Declare the kk bandpower compression: cmb.nbp_kk bands over multipoles
// [lmin, lmax], and raise the is_kk_bandpower_ flag that gates the other
// kk setters and the Hartlap scaling in IP::set_inv_cov.
//
// Validation: nb, lmin, lmax > 0, else critical() + exit(1).
//
// Parameters:
//   nb   - number of kk band powers (cmb.nbp_kk)
//   lmin - lowest multipole entering the bands (cmb.lminbp_kk)
//   lmax - highest multipole entering the bands (cmb.lmaxbp_kk)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void IPCMB::set_kk_binning_bandpower (
    const int nb,
    const int lmin,
    const int lmax
  )
{
  static constexpr std::string_view fname = "IPCMB::set_kk_binning_bandpower"sv;
  debug("{}: {}", fname, errbegins);
  if (!(nb > 0)) [[unlikely]] {
    critical(errorns2, fname, "nbins", nb); exit(1);
  }
  if (!(lmin > 0)) [[unlikely]] {
    critical(errorns2, fname, "lmin", lmin); exit(1);
  }
  if (!(lmax > 0)) [[unlikely]] {
    critical(errorns2, fname, "lmax", lmax); exit(1);
  }
  debug(debugsel, fname, "nbins", nb);
  debug(debugsel, fname, "lmin", lmin);
  debug(debugsel, fname, "lmax", lmax);
  this->is_kk_bandpower_ = 1;
  this->params_->nbp_kk  = nb;
  this->params_->lminbp_kk = lmin;
  this->params_->lmaxbp_kk = lmax;
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class PointMass MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Point-mass contribution to gamma_t(theta) for lens bin zl, source bin zs:
//
//   pm = 4 pi (G/c^2) B_zl 1e13 g_tomo(a_l, zs) / (theta^2 chi_l a_l^3)
//
// with a_l = 1/(1 + zmean(zl)), chi_l = chi(a_l), B_zl = pm_[zl] from
// set_pm_vector, g_tomo the lens efficiency of source bin zs, and
// Goverc2 = 1.6e-23. The a_l^3 (rather than a_l) in the denominator
// matches the DES y3_production convention.
//
// Units: chi is in cosmolike's c/H0 units (cosmo3D.c; coverH0 =
// 2997.92 Mpc/h), theta in rad, g_tomo dimensionless. Goverc2 =
// 1.6e-23 is G/c^2 = 4.79e-20 Mpc/Msun divided by coverH0, i.e. G/c^2
// in c/H0 distance units per Msun/h; the 1e13 then puts the sampled
// amplitude B_zl in units of 10^13 Msun/h, and the returned gamma_t
// is dimensionless.
//
// Parameters:
//   zl    - lens tomographic bin
//   zs    - source tomographic bin
//   theta - angular scale (rad)
//
// Returns:
//   the point-mass gamma_t contribution for the (zl, zs) pair at theta
// ---------------------------------------------------------------------------
double PointMass::get_pm(
    const int zl,
    const int zs,
    const double theta
  ) const
{
  static constexpr std::string_view fname = "PointMass::get_pm"sv;
  debug("{}: {}", fname, errbegins);
  constexpr double Goverc2 = 1.6e-23;
  const double a_lens = 1.0/(1.0 + zmean(zl));
  const double chi_lens = chi(a_lens);
  debug("{}: {}", fname, errends);
  return 4*M_PI*Goverc2*this->pm_[zl]*1.e+13*
    g_tomo(a_lens, zs)/(theta*theta)/(chi_lens*a_lens*a_lens*a_lens);
  
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BaryonScenario MEMBER FUNCTIONS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Register the baryon scenarios entering the PCA from one delimited string
// (separators: "/", space, tab). Each token is normalized by
// get_baryon_sim_name_and_tag and stored as "name-tag"; writes nscenarios_
// and sets is_scenarios_set_.
//
// Validation: an empty string is critical() + exit(1).
//
// Parameters:
//   scenarios - delimited scenario list (separators "/", space, tab)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void BaryonScenario::set_scenarios(std::string scenarios)
{
  static constexpr std::string_view fname = "BaryonScenario::set_scenarios"sv;
  debug("{}: {}", fname, errbegins);
  std::vector<std::string> lines;
  lines.reserve(50);
  boost::trim_if(scenarios, boost::is_any_of("\t "));
  boost::trim_if(scenarios, boost::is_any_of("\n"));
  if (scenarios.empty()) [[unlikely]] {
    critical("{}: invalid string input (empty)", fname); exit(1);
  }

  debug("{}: Selecting baryon scenarios for PCA", fname);

  boost::split(lines,scenarios,boost::is_any_of("/ \t"),boost::token_compress_on);
  int nscenarios = 0;
  for (auto it=lines.begin(); it != lines.end(); ++it) {
    auto [name, tag] = get_baryon_sim_name_and_tag(*it);
    this->scenarios_[nscenarios++] = name + "-" + std::to_string(tag);
  }
  this->nscenarios_ = nscenarios;
  this->is_scenarios_set_ = true;
  debug("{}: {} scenarios are registered", fname, this->nscenarios_);
  debug("{}: Registering baryon scenarios for PCA done!", fname);
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// Overload that also records the scenario library file (set_sims_file) and
// expands range tokens: "root-a-b" (two dashes) registers root-min(a,b)
// through root-(max(a,b) - 1), the upper tag exclusive. Single-tag tokens
// behave as in the one-argument overload.
//
// Parameters:
//   data_sims - scenario library file recorded via set_sims_file
//   scenarios - delimited scenario list; "root-a-b" expands a tag range
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void BaryonScenario::set_scenarios(std::string data_sims, std::string scenarios)
{
  static constexpr std::string_view fname = "BaryonScenario::set_scenarios"sv;
  debug("{}: {}", fname, errbegins);
  this->set_sims_file(data_sims);
  std::vector<std::string> lines;
  lines.reserve(50);
  boost::trim_if(scenarios, boost::is_any_of("\t "));
  boost::trim_if(scenarios, boost::is_any_of("\n"));
  if (scenarios.empty()) [[unlikely]] {
    critical("{}: invalid string input (empty)", fname);
    exit(1);
  }

  debug("{}: Selecting baryon scenarios for PCA", fname);

  boost::split(lines,scenarios,boost::is_any_of("/ \t"),boost::token_compress_on);

  int nscenarios = 0;
  for (auto it=lines.begin(); it != lines.end(); ++it)  {
    // check if the name contains 2 dashes (range) begins ----------------------
    std::vector<int> tags;
    std::string root = *it;
    size_t count = 0;
    size_t pos = 0; 
    while ((pos = root.rfind("-")) != std::string::npos) {
      const int tag = boost::lexical_cast<int>(root.substr(pos+1));
      tags.push_back(tag);
      root = root.substr(0, pos);
      count++;
    }
    // check if the name contains 2 dashes (range) ends ------------------------
    if (2 == count) {
      const int a = std::min(tags[0], tags[1]);
      const int b = std::max(tags[0], tags[1]);
      for (int i=a; i<b; i++) {
        std::string sim = root + "-" + std::to_string(i);
        auto [name, tag] = get_baryon_sim_name_and_tag(sim);
        this->scenarios_[nscenarios++] = name + "-" + std::to_string(tag);
      }
    } 
    else {
      auto [name, tag] = get_baryon_sim_name_and_tag(*it);
      this->scenarios_[nscenarios++] = name + "-" + std::to_string(tag);
    }
  } 
  this->nscenarios_ = nscenarios;
  this->is_scenarios_set_ = true;
  debug("{}: {} scenarios are registered", fname, this->nscenarios_);
  debug("{}: Registering baryon scenarios for PCA done!", fname);
  debug("{}: {}", fname, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

} // end namespace cosmolike_interface

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
