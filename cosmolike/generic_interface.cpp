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
// Why the cpp functions accept and return STL vectors (instead of arma:Col)?
// Answer: the conversion between STL vector and python np array is cleaner
// Answer: arma:Col is cast to 2D np array with 1 column (not as nice!)

namespace cosmolike_interface
{
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// AUX FUNCTIONS (PRIVATE)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
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
      result(0,j) = std::stod(words[j]);
  }

  #pragma omp parallel for
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
      result(i,j) = std::stod(words[j]);
  };
  
  return result;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
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
  
    // no priors
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

void init_ntable_lmax(const int lmax) {
  static constexpr std::string_view fname = "init_ntable_lmax"sv;
  debug("{}: {}", fname, errbegins);
  Ntable.LMAX = lmax;
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

void init_accuracy_boost(
    const double accuracy_boost, 
    const int integration_accuracy
  )
{
  static constexpr std::string_view fname = "init_accuracy_boost"sv;
  static int cache[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS]; // standard: static vars init to zero
  debug("{}: {}", fname, errbegins);

  if (0 == cache[0]) cache[0] = Ntable.N_a;
  Ntable.N_a = static_cast<int>(ceil(cache[0]*accuracy_boost));
  
  if (0 == cache[1]) cache[1] = Ntable.N_ell;
  Ntable.N_ell = static_cast<int>(ceil(cache[1]*accuracy_boost));

  if (0 == cache[2]) cache[2] = Ntable.dCX_dlnk_nlnk;
  Ntable.dCX_dlnk_nlnk = static_cast<int>(ceil(cache[2]*accuracy_boost));

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

void init_baryons_contamination(std::string sim)
{ // OLD API
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

void init_baryons_contamination(std::string sim, std::string all_sims_file)
{ // NEW API
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

void init_bias(vector bias_z_evol_model)
{
  static constexpr std::string_view fname = "init_bias"sv;
  debug("{}: {}", fname, errbegins);
  const int nsz = static_cast<int>(bias_z_evol_model.n_elem);
  if (MAX_SIZE_ARRAYS < nsz) [[unlikely]] {
    critical("{}: {} = {:d} (>{:d})", fname, erriiwz, nsz, MAX_SIZE_ARRAYS);
    exit(1);
  }
  /*
  int galaxy_bias_model[MAX_SIZE_ARRAYS]; // [0] = b1, 
                                          // [1] = b2, 
                                          // [2] = bs2, 
                                          // [3] = b3, 
                                          // [4] = bmag 
  */
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

void init_IA_fastpt(const int IA_MODEL, const int IA_REDSHIFT_EVOL, const int IA_code)
{
  static constexpr std::string_view fname = "init_IA"sv;
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

// backward compatibility
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
  auto it = probe_map.find(boost::algorithm::to_lower_copy(possible_probes));
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
  debug(debugsel, fname, "possible_probes", names.at(possible_probes));
  debug("{}: Ends", "init_probes");
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
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
  tomo.N_ggl_exclude = int(nsize/2);  
  debug("{}: {} ggl pairs excluded", fname, tomo.N_ggl_exclude);
  #pragma omp parallel for
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

void set_cosmological_parameters(
    const double omega_matter,
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
      fdiff(cosmology.h0, hubble/100.0)) { // assuming H0 in km/s/Mpc 
    cache_update = 1;
  }
  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.Omega_m = omega_matter;
    cosmology.Omega_v = 1.0-omega_matter;
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

void set_IA_PS(
    vector PS,
    const double kmin,
    const double kmax,
    const double cutoff,
    const int N
  )
{
  static constexpr std::string_view fname = "set_IA_PS"sv;
  constexpr int NIAPS = 12;
  const double coverH0 = cosmology.coverH0;
  const double coverH0cube = coverH0*coverH0*coverH0;
  
  int cache_update = 0;
  if (FPTIA.N != N ||
      fdiff(FPTIA.k_min, kmin) || 
      fdiff(FPTIA.k_max, kmax) || 
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
    if (FPTIA.tab != NULL) {
      free(FPTIA.tab);
    }
    FPTIA.tab = (double**) malloc2d(NIAPS, FPTIA.N);
    
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
  constexpr int NBIAS = 12;
  const double coverH0  = cosmology.coverH0;
  const double coverH0cube = coverH0*coverH0*coverH0;

  int cache_update = 0;
  if (FPTIA.N != N ||
      fdiff(FPTIA.k_min, kmin) || 
      fdiff(FPTIA.k_max, kmax) || 
      fdiff(FPTIA.k_cutoff, cutoff * coverH0) ||
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
    if (FPTbias.tab != NULL) {
      free(FPTbias.tab);
    }
    FPTbias.tab = (double**) malloc2d(NBIAS, FPTbias.N);

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

void set_distances(vector io_z, vector io_chi)
{
  static constexpr std::string_view fname = "set_cosmological_parameters"sv;
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
  if (cosmology.chi_nz != static_cast<int>(io_z.n_elem) || 
      NULL == cosmology.chi) {
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

    #pragma omp parallel for
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

void set_growth(vector io_z, vector io_G)
{ // Growth: D = G * a
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
    if (cosmology.G != NULL) { free(cosmology.G); }
    cosmology.G = (double**) malloc2d(2, cosmology.G_nz);
    #pragma omp parallel for
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

void set_linear_power_spectrum(vector io_log10k, vector io_z, vector io_lnP)
{
  static constexpr std::string_view fname = "set_linear_power_spectrum"sv;
  debug("{}: {}", fname, errbegins);
  if (io_z.n_elem*io_log10k.n_elem != io_lnP.n_elem) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, io_z.n_elem*io_z.n_elem, io_lnP.n_elem); 
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

  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.lnPL_nk = static_cast<int>(io_log10k.n_elem);
    cosmology.lnPL_nz = static_cast<int>(io_z.n_elem);

    if (cosmology.lnPL != NULL) { free(cosmology.lnPL); }
    cosmology.lnPL = (double**) malloc2d(cosmology.lnPL_nk+1,cosmology.lnPL_nz+1);

    #pragma omp parallel for
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      if (std::isnan(io_log10k(i))) [[unlikely]] {
        critical("{}: {}", fname, errnanit); exit(1);
      }
      cosmology.lnPL[i][cosmology.lnPL_nz] = io_log10k(i);
    }
    #pragma omp parallel for
    for (int j=0; j<cosmology.lnPL_nz; j++) {
      if (std::isnan(io_z(j))) [[unlikely]] {
        critical("{}: {}", fname, errnanit); exit(1);
      }
      cosmology.lnPL[cosmology.lnPL_nk][j] = io_z(j);
    }
    #pragma omp parallel for collapse(2)
    for (int i=0; i<cosmology.lnPL_nk; i++) {
      for (int j=0; j<cosmology.lnPL_nz; j++) {
        if (std::isnan(io_lnP(i*cosmology.lnPL_nz+j))) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        cosmology.lnPL[i][j] = io_lnP(i*cosmology.lnPL_nz+j);
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

void set_non_linear_power_spectrum(vector io_log10k, vector io_z, vector io_lnP)
{
  static constexpr std::string_view fname = "set_non_linear_power_spectrum"sv;
  debug("{}: {}", fname, errbegins);
  if (io_z.n_elem*io_log10k.n_elem != io_lnP.n_elem) [[unlikely]] {
    critical(errorsz1d, fname, erriiwz, io_z.n_elem*io_z.n_elem, io_lnP.n_elem); 
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

  if (1 == cache_update || 1 == force_cache_update_test) {
    cosmology.lnP_nk = static_cast<int>(io_log10k.n_elem);
    cosmology.lnP_nz = static_cast<int>(io_z.n_elem);
    if (cosmology.lnP != NULL) { free(cosmology.lnP); }
    cosmology.lnP = (double**) malloc2d(cosmology.lnP_nk+1,cosmology.lnP_nz+1);

    #pragma omp parallel for
    for (int i=0; i<cosmology.lnP_nk; i++) {
      if (std::isnan(io_log10k(i))) [[unlikely]] {
        critical("{}: {}", fname, errnanit); exit(1);
      }
      cosmology.lnP[i][cosmology.lnP_nz] = io_log10k(i);
    }
    #pragma omp parallel for
    for (int j=0; j<cosmology.lnP_nz; j++) {
      if (std::isnan(io_z(j))) [[unlikely]] {
        critical("{}: {}", fname, errnanit); exit(1);
      }
      cosmology.lnP[cosmology.lnP_nk][j] = io_z(j);
    }
    #pragma omp parallel for collapse(2)
    for (int i=0; i<cosmology.lnP_nk; i++) {
      for (int j=0; j<cosmology.lnP_nz; j++) {
        if (std::isnan(io_lnP(i*cosmology.lnP_nz+j))) [[unlikely]] {
          critical("{}: {}", fname, errnanit); exit(1);
        }
        cosmology.lnP[i][j] = io_lnP(i*cosmology.lnP_nz+j);
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
  // 1st index: b[0][i] = linear galaxy bias in clustering bin i (b1)
  //            b[1][i] = linear galaxy bias in clustering bin i (b2)
  //            b[2][i] = leading order tidal bias in clustering bin i (b3)
  //            b[3][i] = leading order tidal bias in clustering bin i
  //            b[4][i] = amplitude of magnification bias in clustering bin i
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
    if(fdiff(nuisance.gb[1][i], B2(i))) {
      cache_update = 1;
      nuisance.gb[1][i] = B2(i);
      nuisance.gb[2][i] = almost_equal(B2(i), 0.) ? 0 : (-4./7.)*(B1(i)-1.0);
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

// backwards compatibility
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

void set_lens_sample_size(const int Ntomo)
{
  static constexpr std::string_view fname = "set_lens_sample_size"sv;
  if (std::isnan(Ntomo) || !(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns,fname,"Ntomo",Ntomo,MAX_SIZE_ARRAYS);
    exit(1);
  }
  redshift.clustering_photoz = 4;
  redshift.clustering_nbin = Ntomo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_lens_sample(arma::Mat<double> input_table)
{
  static constexpr std::string_view fname = "set_lens_sample"sv;
  debug("{}: {}", fname, errbegins);

  const int Ntomo = redshift.clustering_nbin;
  if (std::isnan(Ntomo) || !(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
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
      auto nofz = input_table.col(k+1).eval();
      arma::uvec idx = arma::find(nofz > 0.999e-8*nofz.max());
      redshift.clustering_zdist_zmin[k] = z_v[idx(0)];
      redshift.clustering_zdist_zmax[k] = z_v[idx(idx.n_elem-1)];
    }
    // READ THE N(Z) FILE ENDS ------------
    redshift.random_clustering = RandomNumber::get_instance().get();

    pf_photoz(0.1, 0); // init static variables

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

void set_source_sample_size(const int Ntomo)
{
  static constexpr std::string_view fname = "set_source_sample_size"sv;
  if (std::isnan(Ntomo) || !(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
    critical(errorns, fname, "Ntomo", Ntomo,  MAX_SIZE_ARRAYS);
    exit(1);
  } 
  redshift.shear_photoz = 4;
  redshift.shear_nbin = Ntomo;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void set_source_sample(arma::Mat<double> input_table)
{
  static constexpr std::string_view fname = "set_source_sample"sv;
  debug("{}: {}", fname, errbegins);

  const int Ntomo = redshift.shear_nbin;
  if (std::isnan(Ntomo) || !(Ntomo > 0) || Ntomo > MAX_SIZE_ARRAYS) [[unlikely]] {
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
      auto nofz = input_table.col(k+1).eval();
      arma::uvec idx = arma::find(nofz > 0.999e-8*nofz.max());
      redshift.shear_zdist_zmin[k] = fmax(z_v[idx(0)], 1.001e-5);
      redshift.shear_zdist_zmax[k] = z_v[idx(idx.n_elem-1)];
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
    zdistr_photoz(0.1, 0); // init static variables
    for (int k=0; k<Ntomo; k++) {
      debug("{}: bin {} - {} = {}.", fname, k, "<z_s>", zmean_source(k));
    }
    redshift.random_shear = RandomNumber::get_instance().get();
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
      #pragma omp parallel for
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
      #pragma omp parallel for
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
      #pragma omp parallel for
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
    #pragma omp parallel for collapse(2)
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
  #pragma omp parallel for
  for (int i=0; i<this->ndata_; i++) {
    this->inv_cov_masked_(i,i) *= this->get_mask(i)*this->get_mask(i);
    for (int j=0; j<i; j++) {
      this->inv_cov_masked_(i,j) *= this->get_mask(i)*this->get_mask(j);
      this->inv_cov_masked_(j,i) = this->inv_cov_masked_(i,j);
    }
  };
  
  #pragma omp parallel for collapse(2)
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
  }
  if (chi2 < 0.0) [[unlikely]] {
    critical("{}: chi2 = {} (invalid)", fname, chi2); exit(1);
  }
  debug("{}: {}", fname, errends);
  return chi2;
}

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
    
  #pragma omp parallel for
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

void IPCMB::set_kk_binning_bandpower (
    const int nb, 
    const int lmin, 
    const int lmax
  )
{
  static constexpr std::string_view fname = "IPCMB::set_kk_binning"sv;
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

double PointMass::get_pm(
    const int zl, 
    const int zs, 
    const double theta
  ) const
{ // JX: add alens^2 in the den to be consistent with y3_production
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
