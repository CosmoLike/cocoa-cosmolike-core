#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <variant>
#include <cmath> 
#include <string_view>
#include <optional>
using namespace std::literals; // enables "sv" literal

// SPDLOG
//#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/baryons.h"
#include "cosmolike/cosmo2D.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/IA.h"
#include "cosmolike/halo.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_GENERIC_INTERFACE_HPP
#define __COSMOLIKE_GENERIC_INTERFACE_HPP

namespace cosmolike_interface
{
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class RandomNumber
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
class RandomNumber
{ // Singleton Class that holds a random number generator
  public:
    static RandomNumber& get_instance() {
      static RandomNumber instance;
      return instance;
    }
    double get() {
      return dist_(mt_);
    }
  protected:
    std::random_device rd_;
    std::mt19937 mt_;
    std::uniform_real_distribution<double> dist_;
  private:
    RandomNumber() :
      rd_(),
      mt_(rd_()),
      dist_(0.0, 1.0) {
      };
    RandomNumber(RandomNumber const&) = delete;
};
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IP (InterfaceProducts)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
class IP
{ // InterfaceProducts: Singleton Class that holds data vector, covariance...
private:
  static constexpr std::string_view errornset = "{}: {} not set (?ill-defined) prior to this function call"sv;
  static constexpr std::string_view errornv = "{}: idx i={} not supported (min={},max={})"sv;
  public:
    static IP& get_instance() {
      static IP instance;
      return instance;
    }
    bool is_mask_set() const {
      return this->is_mask_set_;
    }
    bool is_data_set() const {
      return this->is_data_set_;
    }
    bool is_inv_cov_set() const {
      return this->is_inv_cov_set_;
    }

    void set_data(std::string datavector_filename);

    template <int N, int M>
    void set_mask(std::string mask_filename, arma::Col<int>::fixed<M> ord);

    void set_inv_cov(std::string covariance_filename);

    //void set_PMmarg(std::string U_PMmarg_file);

    int get_mask(const int ci) const {
      static constexpr std::string_view fn = "IP::get_mask"sv;
      if (ci > like.Ndata || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, like.Ndata); exit(1);
      }
      return this->mask_(ci);
    }

    double get_dv_masked(const int ci) const {
      static constexpr std::string_view fn = "IP::get_dv_masked"sv;
      if (ci > like.Ndata || ci < 0) [[unlikely]] {
        spdlog::critical(errornv,fn,ci,0,like.Ndata); exit(1);
      }
      return this->data_masked_(ci);
    }

    double get_inv_cov_masked(const int ci, const int cj) const {
      static constexpr std::string_view fn = "IP::get_inv_cov_masked"sv;
      if (ci > like.Ndata || ci < 0) [[unlikely]] {
        spdlog::critical(errornv,fn,ci,0,like.Ndata);
        exit(1);
      }
      if (cj > like.Ndata || cj < 0) [[unlikely]] {
        spdlog::critical(errornv,fn,cj,0,like.Ndata); exit(1);
      }
      return this->inv_cov_masked_(ci,cj);
    }

    int get_index_sqzd(const int ci) const {
      static constexpr std::string_view fn = "IP::get_index_sqzd"sv;
      if (ci > like.Ndata || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, like.Ndata); exit(1);
      }
      return this->index_sqzd_(ci);
    }

    double get_dv_masked_sqzd(const int ci) const {
      static constexpr std::string_view fn = "IP::get_dv_masked_sqzd"sv;
      if (ci > like.Ndata || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, like.Ndata); exit(1);
      }
      return this->data_masked_sqzd_(ci);
    }

    double get_inv_cov_masked_sqzd(const int ci, const int cj) const {
      static constexpr std::string_view fn = "IP::get_dv_masked_sqzd"sv;
      if (ci > like.Ndata || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, like.Ndata); exit(1);
      }
      if (cj > like.Ndata || cj < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, cj, 0, like.Ndata); exit(1);
      }
      return this->inv_cov_masked_sqzd_(ci,cj);
    }

    arma::Col<double> expand_theory_data_vector_from_sqzd(arma::Col<double>) const;

    arma::Col<double> sqzd_theory_data_vector(arma::Col<double>) const;

    double get_chi2(arma::Col<double> datavector) const;

    int get_ndata() const {
      return this->ndata_;
    }
    arma::Col<int> get_mask() const {
      return this->mask_;
    }
    arma::Col<double> get_dv_masked() const {
      return this->data_masked_;
    }
    arma::Mat<double> get_cov_masked() const{
      return this->cov_masked_;
    }
    arma::Mat<double> get_inv_cov_masked() const {
      return this->inv_cov_masked_;
    }
    int get_ndata_sqzd() const {
      return this->ndata_sqzd_;
    }
    arma::Col<double> get_dv_masked_sqzd() const {
      return this->data_masked_sqzd_;
    }
    arma::Mat<double> get_cov_masked_sqzd() const {
      return this->cov_masked_sqzd_;
    }
    arma::Mat<double> get_inv_cov_masked_sqzd() const {
      return this->inv_cov_masked_sqzd_;
    }
  private:
    bool is_mask_set_ = false;
    bool is_data_set_ = false;
    bool is_inv_cov_set_ = false;
    int ndata_ = 0;
    int ndata_sqzd_ = 0;
    std::string mask_filename_;
    std::string cov_filename_;
    std::string data_filename_;
    arma::Col<int> mask_;
    arma::Col<double> data_masked_;
    arma::Mat<double> cov_masked_;
    arma::Col<int> index_sqzd_;
    arma::Mat<double> inv_cov_masked_;
    arma::Col<double> data_masked_sqzd_;
    arma::Mat<double> cov_masked_sqzd_; 
    arma::Mat<double> inv_cov_masked_sqzd_;
    IP() = default;
    IP(IP const&) = delete;
};
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class IPCMB (Interface to Cosmolike C glocal struct CMBParams cmb)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
class IPCMB
{
private:
  static constexpr std::string_view errornset = "{}: {} not set (?ill-defined) prior to this function call"sv;
  static constexpr std::string_view errornv = "{}: idx i={} not supported (min={},max={})"sv;
public:
    static IPCMB& get_instance() {
      static IPCMB instance;
      if (NULL == instance.params_) {
        instance.params_ = &cmb;
      }
      return instance;
    }
    ~IPCMB() = default;

    
    bool is_kk_bandpower() const {
      return this->is_kk_bandpower_;
    }

    void update_cache(const double random) {
      this->params_->random = random;
      return;
    }
    
    void set_wxk_beam_size(const double fwhm) {
      this->params_->fwhm = fwhm;
      this->is_wxk_fwhm_set_ = true;
      return;
    }
    
    void set_wxk_lminmax(const int lmin, const int lmax) {
      this->params_->lmink_wxk = lmin;
      this->params_->lmaxk_wxk = lmax;
      this->is_wxk_lminmax_set_ = true;
      return;
    }
    
    void set_alpha_Hartlap_cov_kkkk(const double alpha) {
      this->params_->alpha_Hartlap_cov_kkkk = alpha;
      this->is_alpha_Hartlap_cov_kkkk_set_ = true;
      return;
    }

    void set_wxk_healpix_window(std::string healpixwin_filename);

    void set_kk_binning_mat(std::string binned_matrix_filename);

    void set_kk_theory_offset(std::string theory_offset_filename);
    
    void set_kk_binning_bandpower(const int, const int, const int);

    double get_kk_binning_matrix(const int ci, const int cj) const {
      static constexpr std::string_view fn = "IPCMB::get_kk_binning_matrix"sv;
      if (!this->is_kk_binning_matrix_set_) [[unlikely]] {
        spdlog::critical(errornset, fn, "is_kk_binning_matrix_set_"); exit(1);
      }
      const int nbp  = this->get_nbins_kk_bandpower();
      const int lmax = this->get_lmax_kk_bandpower();
      const int lmin = this->get_lmin_kk_bandpower();
      const int ncl  = lmax - lmin + 1;
      if (ci > nbp || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, nbp); exit(1);
      }
      if (cj > ncl || cj < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, cj, 0, ncl); exit(1);
      }
      return this->params_->binning_matrix_kk[ci][cj];
    }

    double get_kk_theory_offset(const int ci) const {
      static constexpr std::string_view fn = "IPCMB::get_kk_theory_offset"sv;
      if (!this->is_kk_offset_set_) [[unlikely]] {
        spdlog::critical(errornset, fn, "is_kk_offset_set_"); exit(1);
      }
      const int nbp = this->get_nbins_kk_bandpower();
      if (ci > nbp || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, nbp); exit(1);
      }
      return this->params_->theory_offset_kk[ci];
    }

    double get_alpha_Hartlap_cov_kkkk() const {
      static constexpr std::string_view fn = "IPCMB::get_alpha_Hartlap_cov_kkkk"sv;
      if (!this->is_alpha_Hartlap_cov_kkkk_set_) [[unlikely]] {
        spdlog::critical(errornset , fn,"is_alpha_Hartlap_cov_kkkk_set_"); exit(1);
      }
      return this->params_->alpha_Hartlap_cov_kkkk;
    }
    
    int get_nbins_kk_bandpower() const {
      return this->params_->nbp_kk; 
    }
    
    int get_lmin_kk_bandpower() const {
      return this->params_->lminbp_kk; 
    }
    
    int get_lmax_kk_bandpower() const {
      return this->params_->lmaxbp_kk;
    }

  private: 
    CMBparams* params_ = NULL;
    bool is_wxk_fwhm_set_ = false;
    bool is_wxk_lminmax_set_ = false;
    bool is_wxk_healpix_window_set_ = false;
    bool is_kk_bandpower_ = false;
    bool is_kk_binning_matrix_set_ = false; 
    bool is_kk_offset_set_ = false; 
    bool is_alpha_Hartlap_cov_kkkk_set_ = false; 
    IPCMB() = default;
    IPCMB(IPCMB const&) = delete; 
};
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class PointMass
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
class PointMass
{// Singleton Class that Evaluate Point Mass Marginalization
  public:
    static PointMass& get_instance() {
      static PointMass instance;
      return instance;
    }
    ~PointMass() = default;

    void set_pm_vector(arma::Col<double> pm)  {
      this->pm_ = pm;
      return;
    }
    arma::Col<double> get_pm_vector() const {
      return this->pm_;
    }
    double get_pm(const int zl, const int zs, const double theta) const;
  private:
    arma::Col<double> pm_;
    PointMass() = default;
    PointMass(PointMass const&) = delete;
};
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Class BaryonScenario
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
class BaryonScenario
{ // Singleton Class that map Baryon Scenario (integer to name)
  public:
    static BaryonScenario& get_instance()
    {
      static BaryonScenario instance;
      return instance;
    }
    ~BaryonScenario() = default;

    int nscenarios() const {
      static constexpr std::string_view fn = "BaryonScenario::nscenarios"sv;
      if (!this->is_scenarios_set_) [[unlikely]] {
        spdlog::critical("{}: {} not set",fn,"Baryon Scenarios");
        exit(1);
      }
      return this->nscenarios_;
    }
    
    bool is_pcs_set() const {
      return this->is_pcs_set_;
    }

    bool is_allsims_file_set() const {
      return this->is_allsims_file_set_;
    }

    bool is_scenarios_set() const {
      return this->is_scenarios_set_;
    }

    void set_pcs(arma::Mat<double> eigenvectors) {
      this->eigenvectors_ = eigenvectors;
      this->is_pcs_set_ = true;
    }

    void set_sims_file(std::string data_sims) {
      this->allsims_file_ = data_sims;
      this->is_allsims_file_set_ = true;
    }

    void set_scenarios(std::string data_sims, std::string scenarios);

    void set_scenarios(std::string scenarios);

    std::tuple<std::string,int> select_baryons_sim(const std::string scenario);
    
    std::string get_scenario(const int i) const {
      static constexpr std::string_view fn = "BaryonScenario::get_scenario"sv;
      if (!this->is_scenarios_set_) [[unlikely]] {
        spdlog::critical("{}: {} not set", fn, "Baryon Scenarios");
        exit(1);
      }
      return this->scenarios_.at(i);
    }
    
    arma::Mat<double> get_pcs() const {
      static constexpr std::string_view fn = "BaryonScenario::get_pcs"sv;
      if (!this->is_pcs_set_) [[unlikely]] {
        spdlog::critical("{}: {} not set",fn,"PC eigenvectors");
        exit(1);
      }
      return this->eigenvectors_;
    }
    
    double get_pcs(const int ci, const int cj) const {
      static constexpr std::string_view fn = "BaryonScenario::get_pcs"sv;
      if (!this->is_pcs_set_) [[unlikely]] {
        spdlog::critical("{}: {} not set",fn,"PC eigenvectors");
        exit(1);
      }
      return this->eigenvectors_(ci, cj); 
    }

    std::string get_allsims_file() const {
      static constexpr std::string_view fn = "BaryonScenario::get_allsims_file"sv;
      if (!this->is_allsims_file_set_) [[unlikely]] {
        spdlog::critical("{}: {} not set", fn, "all sims HDF5 file");
        exit(1);
      }
      return this->allsims_file_;
    }
  private:
    bool is_allsims_file_set_, is_pcs_set_, is_scenarios_set_;
    int nscenarios_;
    std::string allsims_file_;
    std::map<int, std::string> scenarios_;
    arma::Mat<double> eigenvectors_;
    BaryonScenario() = default;
    BaryonScenario(BaryonScenario const&) = delete;
};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Global Functions
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

arma::Mat<double> read_table(const std::string file_name);

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type 
almost_equal(T x, T y, int ulp = 100)
{
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
      // unless the result is subnormal
      || std::fabs(x-y) < std::numeric_limits<T>::min();
}

void init_ntable_lmax(
    const int lmax
  );

void init_accuracy_boost(
    const double accuracy_boost, 
    const int integration_accuracy
  );

void init_baryons_contamination(std::string sim, std::string all_sims_file); // NEW API

void init_baryons_contamination(std::string sim); // OLD API

void init_bias(arma::Col<double> bias_z_evol_model);

void init_binning_fourier(
    const int Ncl, 
    const int lmin, 
    const int lmax, 
    const int lmax_shear
  );

void init_binning_real_space(
    const int Ntheta, 
    const double theta_min_arcmin, 
    const double theta_max_arcmin
  );

void init_cmb_auto_bandpower (
    const int nbins, 
    const int lmin, 
    const int lmax,
    std::string binning_matrix, 
    std::string theory_offset,
    const double alpha
  );

void init_cosmo_runmode(const bool is_linear);

void init_cmb_cross_correlation (
    const int lmin, 
    const int lmax, 
    const double fwhm, // fwhm = beam size in arcmin
    std::string healpixwin_filename
  );

void init_IA_fastpt(
    const int IA_MODEL, 
    const int IA_REDSHIFT_EVOL,
    const int IA_code
  );

void init_IA(
    const int IA_MODEL, 
    const int IA_REDSHIFT_EVOL
  );

void init_probes(
    std::string possible_probes
  );

void initial_setup();

py::tuple read_redshift_distributions_from_files(
    std::string lens_multihisto_file, 
    const int lens_ntomo,
    std::string source_multihisto_file, 
    const int source_ntomo
  );

void init_redshift_distributions_from_files(
    std::string lens_multihisto_file, 
    const int lens_ntomo,
    std::string source_multihisto_file, 
    const int source_ntomo
  );

void init_survey(
    std::string surveyname, 
    double area, 
    double sigma_e
  );

void init_ggl_exclude(
	arma::Col<int> ggl_exclude
  );

void set_cosmological_parameters(
    const double omega_matter,
    const double hubble
  );

void set_distances(
    arma::Col<double> io_z, 
    arma::Col<double> io_chi
  );

void set_growth(
    arma::Col<double> io_z, 
    arma::Col<double> io_G
  );

void set_linear_power_spectrum(
    arma::Col<double> io_log10k,
    arma::Col<double> io_z, 
    arma::Col<double> io_lnP
  );

void set_IA_PS(
    arma::Col<double> PS,
    const double kmin,
    const double kmax,
    const double cutoff,
    const int N
  );

void set_bias_PS(
    arma::Col<double> PS,
    const double kmin,
    const double kmax,
    const double cutoff,
    const double sigma4,
    const int N
  );

void set_non_linear_power_spectrum(
    arma::Col<double> io_log10k,
    arma::Col<double> io_z, 
    arma::Col<double> io_lnP
  );

void set_nuisance_bias_fastpt(
	arma::Col<double> B1, 
	arma::Col<double> B2, 
	arma::Col<double> B_MAG,
	arma::Col<double> B3nl,
	arma::Col<double> BK
);
  
void set_nuisance_bias(
    arma::Col<double> B1,
    arma::Col<double> B2,
    arma::Col<double> B_MAG
  );

void set_nuisance_clustering_photoz(
    arma::Col<double> CP
  );

void set_nuisance_clustering_photoz_stretch(arma::Col<double> CPS);

void set_nuisance_IA(
    arma::Col<double> A1, 
    arma::Col<double> A2,
    arma::Col<double> BTA
  );

void set_nuisance_magnification_bias(
    arma::Col<double> B_MAG
  );

void set_nuisance_nonlinear_bias(
    arma::Col<double> B1,
    arma::Col<double> B2
  );

void set_nuisance_shear_calib(
    arma::Col<double> M
  );

void set_nuisance_shear_photoz(
    arma::Col<double> SP
  );

void set_lens_sample_size(const int Ntomo);

void set_lens_sample(arma::Mat<double> input_table);

void set_source_sample_size(const int Ntomo);

void set_source_sample(arma::Mat<double> input_table);

void init_ntomo_powerspectra();

arma::Col<double> compute_binning_real_space();

arma::Col<double> compute_add_baryons_pcs(arma::Col<double> Q, arma::Col<double> dv);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M>
arma::Col<int>::fixed<M> compute_data_vector_Mx2pt_N_sizes() 
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "compute_data_vector_Mx2pt_N_sizes"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}: {}", fname, errbegins);
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  arma::Col<int>::fixed<2> Nlen = {Ntable.Ntheta, like.Ncl};
  arma::Col<int>::fixed<M> sizes;
  if constexpr (N == 0) {
    sizes(0) = 2*Ntable.Ntheta*tomo.shear_Npowerspectra;
  } 
  else {
    sizes(0) = like.Ncl*tomo.shear_Npowerspectra;
  } 
  sizes(1) = Nlen[N]*tomo.ggl_Npowerspectra;
  sizes(2) = Nlen[N]*tomo.clustering_Npowerspectra;
  if constexpr (6 == M) {
    IPCMB& cmb = IPCMB::get_instance();
    sizes(3) = Nlen[N]*redshift.clustering_nbin;
    sizes(4) = Nlen[N]*redshift.shear_nbin;
    sizes(5) = cmb.is_kk_bandpower() == 1 ? cmb.get_nbins_kk_bandpower() : like.Ncl;
  }
  debug("{}: {}", fname, errends);
  return sizes;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M> 
void init_data_Mx2pt_N(
    std::string cov, 
    std::string mask, 
    std::string data, 
    arma::Col<int>::fixed<M> ord
  )
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "init_data_Mx2pt_N"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}: {}", fname, errbegins);
  arma::Col<int>::fixed<M> ndv = compute_data_vector_Mx2pt_N_sizes<N,M>();
  for(int i=0; i<(int) ndv.n_elem; i++) {
    like.Ndata += ndv(i);
  }
  IP& survey = IP::get_instance();
  survey.set_mask<N,M>(mask, ord);  // set_mask must be called first
  survey.set_data(data);
  survey.set_inv_cov(cov);
  debug("{}: {}", fname, errends);
  return;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M>
arma::Col<int>::fixed<M> compute_data_vector_Mx2pt_N_starts(arma::Col<int>::fixed<M> ord) 
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "compute_data_vector_Mx2pt_N_starts"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}: {}", fname, errbegins);
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  using namespace arma;
  Col<int>::fixed<M> sizes = compute_data_vector_Mx2pt_N_sizes<N,M>();
  auto indices = conv_to<Col<int>>::from(stable_sort_index(ord, "ascend"));
  Col<int>::fixed<M> start(arma::fill::zeros);
  for(int i=0; i<M; i++) {
    for(int j=0; j<indices(i); j++) {
      start(i) += sizes(indices(j));
    }
  }
  debug("{}: {}", fname, errends);
  return start; 
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M, int P = 1> 
void add_calib_and_set_mask_X_N(arma::Col<double>& dv, const int start)
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "compute_add_calib_and_set_mask_Mx2pt_N"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}<{},{},{}>: {}", fname, N, M, P, errbegins);
  using vector = arma::Col<double>;
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(P == 0 || P == 1, "P must be 0/1 (include PM))");
  IP& survey = IP::get_instance();
  arma::Col<int>::fixed<2> Nlen = {Ntable.Ntheta, like.Ncl};

  if constexpr (0 == M) {
    if (1 == like.shear_shear) { 
      for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
        const int z1 = Z1(nz);
        const int z2 = Z2(nz);
        for (int i=0; i<Nlen[N]; i++) {
          int index = start + Nlen[N]*nz + i;
          if (survey.get_mask(index)) {
            dv(index) *= (1.0 + nuisance.shear_calibration_m[z1])*
                         (1.0 + nuisance.shear_calibration_m[z2]);
          }
          else {
            dv(index) = 0.0;
          }
          if constexpr (N == 0) { 
            index += Nlen[N]*tomo.shear_Npowerspectra;
            if (survey.get_mask(index)) {
              dv(index) *= (1.0 + nuisance.shear_calibration_m[z1])*
                           (1.0 + nuisance.shear_calibration_m[z2]);
            }
            else {
              dv(index) = 0.0;
            }
          }
        }
      }
    }
  }
  else if constexpr (1 == M) {
    if (1 == like.shear_pos) {
      for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
        const int zs = ZS(nz);
        for (int i=0; i<Nlen[N]; i++) {
          const int index = start + Nlen[N]*nz + i;
          if (survey.get_mask(index)) {
            if constexpr (0 == N && 1 == P) {
              vector theta = compute_binning_real_space();
              const int zl = ZL(nz);
              dv(index) += PointMass::get_instance().get_pm(zl,zs,theta(i));
            }
            dv(index) *= (1.0+nuisance.shear_calibration_m[zs]);
          }
          else {
            dv(index) = 0.0;
          }
        }
      }
    }
  }
  else if constexpr (2 == M) {
    if (1 == like.pos_pos) {
      for (int nz=0; nz<tomo.clustering_Npowerspectra; nz++) {
        for (int i=0; i<Nlen[N]; i++) {
          const int index = start + Nlen[N]*nz + i;
          if (!survey.get_mask(index)) {
            dv(index) = 0.0;
          }
        }
      }
    }
  }
  else if constexpr (3 == M) {
    if (1 == like.gk) {
      for (int nz=0; nz<redshift.clustering_nbin; nz++) {
        for (int i=0; i<Nlen[N]; i++) {
          const int index = start + Nlen[N]*nz + i;
          if (!survey.get_mask(index)) {
            dv(index) = 0.0;
          }
        }
      }
    }
  }
  else if constexpr (4 == M) {
    if (1 == like.ks) {
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        for (int i=0; i<Nlen[N]; i++) {
          const int index = start + Nlen[N]*nz + i; 
          if (survey.get_mask(index)) {
            dv(index) *= (1.0 + nuisance.shear_calibration_m[nz]);
          }
          else {
            dv(index) = 0.0;
          }
        }
      }
    }
  }
  else if constexpr (5 == M) {
    if (1 == like.kk) {
      IPCMB& cmb = IPCMB::get_instance();
      if (0 == cmb.is_kk_bandpower()) {
        for (int i=0; i<like.Ncl; i++) {
          const int index = start + i; 
          if (!survey.get_mask(index)) {
            dv(index) = 0.0;
          }
        }
      }
      else {
        const int nbp = cmb.get_nbins_kk_bandpower();
        for (int i=0; i<nbp; i++) { // Loop through bandpower bins
          const int index = start + i; 
          if (!survey.get_mask(index)) {        
            dv(index) = 0.0;
          }
        }
      }
    }
  }
  debug("{}<{},{},{}>: {}", fname, N, M, P, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M, int P> 
arma::Col<double> compute_add_calib_and_set_mask_Mx2pt_N(
    arma::Col<double> data_vector, 
    arma::Col<int>::fixed<M> ord
  )
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "compute_add_calib_and_set_mask_Mx2pt_N"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}<{},{},{}>: {}", fname, N, M, P, errbegins);
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  static_assert(P == 0 || P == 1, "P must be 0/1 (include PM))");
  arma::Col<int>::fixed<M> start = compute_data_vector_Mx2pt_N_starts<N,M>(ord);
  add_calib_and_set_mask_X_N<N,0,P>(data_vector, start(0));
  add_calib_and_set_mask_X_N<N,1,P>(data_vector, start(1));
  add_calib_and_set_mask_X_N<N,2,P>(data_vector, start(2));
  if constexpr (6 == M) {
    add_calib_and_set_mask_X_N<N,3,P>(data_vector, start(3));
    add_calib_and_set_mask_X_N<N,4,P>(data_vector, start(4));
    add_calib_and_set_mask_X_N<N,5,P>(data_vector, start(5));
  }
  debug("{}<{},{},{}>: {}", fname, N, M, P, errends);
  return data_vector;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M> 
void compute_X_N_masked(arma::Col<double>& dv, const int start)
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "compute_X_N_masked"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}<{},{}>: {}", fname, N, M, errbegins);
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  IP& survey = IP::get_instance();
  arma::Col<int>::fixed<2> Nlen = {Ntable.Ntheta, like.Ncl};

  if constexpr (0 == M) {
    if (1 == like.shear_shear) {    
      for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
        const int z1 = Z1(nz);
        const int z2 = Z2(nz);
        for (int i=0; i<Nlen[N]; i++) {
          int index = start + Nlen[N]*nz + i;
          if constexpr (N == 0) {
            if (survey.get_mask(index)) {
              dv(index) = xi_pm_tomo(1, i, z1, z2, 1);
            }  
            index += Nlen[N]*tomo.shear_Npowerspectra;
            if (survey.get_mask(index)) {
              dv(index) = xi_pm_tomo(-1, i, z1, z2, 1);
            } 
          }
          else {
            if (survey.get_mask(index) && (like.ell[i]<like.lmax_shear)) {
              dv(index) = C_ss_tomo_limber_nointerp(like.ell[i], z1, z2, 1, 0);
            }
          }
        }
      }
      add_calib_and_set_mask_X_N<N,M>(dv, start);
    }
  }
  else if constexpr (1 == M) {
    if (1 == like.shear_pos) {
      for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
        const int zl = ZL(nz);
        const int zs = ZS(nz);
        for (int i=0; i<Nlen[N]; i++) {
          const int index = start + Nlen[N]*nz + i;
          if (survey.get_mask(index)) {
            if constexpr (0 == N)
              dv(index) = w_gammat_tomo(i,zl,zs,1);
            else
              dv(index) = C_gs_tomo_limber_nointerp(like.ell[i], zl, zs, 0);
          }
        }
      }
      add_calib_and_set_mask_X_N<N,M>(dv, start);
    }
  }
  else if constexpr (2 == M) {
    if (1 == like.pos_pos) {
      for (int nz=0; nz<tomo.clustering_Npowerspectra; nz++) {
        for (int i=0; i<Nlen[N]; i++) {
          const int index = start + Nlen[N]*nz + i;
          if (survey.get_mask(index)) {
            if constexpr (N == 0) {  
              dv(index) = w_gg_tomo(i, nz, nz, like.adopt_limber_gg);
            }
            else {
              dv(index) = C_gg_tomo_limber_nointerp(like.ell[i], nz, nz, 0);
            }
          }
        }
      }
      add_calib_and_set_mask_X_N<N,M>(dv, start);
    }
  }
  else if constexpr (3 == M) {
    if (1 == like.gk) {
      for (int nz=0; nz<redshift.clustering_nbin; nz++) {
        if constexpr (N == 0) {
          for (int i=0; i<Ntable.Ntheta; i++) {
            const int index = start + Ntable.Ntheta*nz + i;
            if (survey.get_mask(index)) {
              if constexpr (N == 0) {  
                dv(index) = w_gk_tomo(i, nz, 1);
              }
              else {
                spdlog::critical("not implemented");
                exit(1);
              }
            }
          }
        }
      }
      add_calib_and_set_mask_X_N<N,M>(dv, start);
    }
  }
  else if constexpr (4 == M) {
    if (1 == like.ks) {
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        if constexpr (N == 0) {
          for (int i=0; i<Ntable.Ntheta; i++) {
            const int index = start + Ntable.Ntheta*nz + i; 
            if (survey.get_mask(index)) {
              if constexpr (N == 0) { 
                dv(index) = w_ks_tomo(i, nz, 1);
              }
              else {
                spdlog::critical("not implemented");
                exit(1);
              }
            }
          }
        }
      }
      add_calib_and_set_mask_X_N<N,M>(dv, start);
    }
  }
  else if constexpr (5 == M) {
    if (1 == like.kk) {
      IPCMB& cmb = IPCMB::get_instance();
      if (0 == cmb.is_kk_bandpower()) {
        for (int i=0; i<like.Ncl; i++) {
          const int index = start + i; 
          if (survey.get_mask(index)) {
            const double l = like.ell[i];
            dv(index) = (l<=limits.LMIN_tab) ? C_kk_limber_nointerp(l,0) : 
                                               C_kk_limber(l);
          }
        }
      }
      else {
        const int nbp = cmb.get_nbins_kk_bandpower();
        const int lminbp = cmb.get_lmin_kk_bandpower();
        const int lmaxbp = cmb.get_lmax_kk_bandpower();
        for (int j=0; j<nbp; j++) {
          const int index = start + j; 
          if (survey.get_mask(index)) {        
            dv(index) = 0.0;
          }
        }
        for (int L=lminbp; L<lmaxbp + 1; L++) {
          const double Ckk = (L <= limits.LMIN_tab) ? 
            C_kk_limber_nointerp((double) L, 0) : C_kk_limber((double) L);
          for (int j=0; j<nbp; j++) { // Loop through bandpower bins
            const int index = start + j; 
            if (survey.get_mask(index)) {        
              dv(index) += (Ckk*cmb.get_kk_binning_matrix(j, L-lminbp));
            }
          }
        }
        for (int j=0; j<nbp; j++) { // offset due to marginalizing over primary CMB
          const int index = start + j;
          if (survey.get_mask(index)) {
            dv(index) -= cmb.get_kk_theory_offset(j);
          }
        }
      }
      add_calib_and_set_mask_X_N<N,M>(dv, start);
    }
  }
  debug("{}<{},{}>: {}", fname, N, M, errends);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M> 
arma::Col<double> compute_Mx2pt_N_masked(arma::Col<int>::fixed<M> ord)
{
  static constexpr std::string_view errbegins = "Begins Execution"sv;
  static constexpr std::string_view errends = "Ends Execution"sv;
  static constexpr std::string_view fname = "compute_Mx2pt_N_masked"sv;
  using spdlog::debug; using spdlog::info;
  debug("{}<{},{}>: {}", fname, N, M, errbegins);
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  arma::Col<int>::fixed<M> start = compute_data_vector_Mx2pt_N_starts<N,M>(ord);
  arma::Col<double> data_vector(like.Ndata, arma::fill::zeros); 
  compute_X_N_masked<N,0>(data_vector, start(0));
  compute_X_N_masked<N,1>(data_vector, start(1));
  compute_X_N_masked<N,2>(data_vector, start(2));
  if constexpr (6 == M) {
    compute_X_N_masked<N,3>(data_vector, start(3));
    compute_X_N_masked<N,4>(data_vector, start(4));
    compute_X_N_masked<N,5>(data_vector, start(5));
  }
  debug("{}<{},{}>: {}", fname, N, M, errends);
  return data_vector;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M> 
arma::Mat<double> compute_baryon_pcas_Mx2pt_N(arma::Col<int>::fixed<M> ord)
{
  using spdlog::debug; using spdlog::info;
  using matrix = arma::Mat<double>; using vector = arma::Col<double>;
  static constexpr std::string_view fname = "compute_baryon_pcas_Mx2pt_N"sv;
  static_assert(0 == N || 1 == N, "N must be 0 (real) or 1 (fourier)");
  static_assert(3 == M || 6 == M, "M must be 3 (3x2pt) or 6 (6x2pt)");
  IP& ip = IP::get_instance();
  const int ndata = ip.get_ndata();
  const int ndata_sqzd = ip.get_ndata_sqzd();
  BaryonScenario& bs = BaryonScenario::get_instance();
  const int nscenarios = bs.nscenarios();

  // Compute Cholesky Decomposition of the Covariance Matrix --------------
  debug("{}: Cholesky Decomposition of the cov Matrix begins", fname);
  matrix L = arma::chol(ip.get_cov_masked_sqzd(), "lower");
  matrix inv_L = arma::inv(L);
  debug("{}: Cholesky Decomposition of the cov Matrix ends", fname);

  // Compute Dark Matter data vector --------------------------------------
  debug("{}: Comp. DM only data vector begins", fname);
  cosmology.random = RandomNumber::get_instance().get(); // clear cosmolike cache
  reset_bary_struct(); // make sure there is no baryon contamination
  vector dv_dm = ip.sqzd_theory_data_vector(compute_Mx2pt_N_masked<N,M>(ord));
  debug("{}: Comp. DM only data vector ends", fname);
  // Compute data vector for all Baryon scenarios -------------------------
  matrix D = matrix(ndata_sqzd, nscenarios);
  for (int i=0; i<nscenarios; i++) {
    debug("{}: comp. dv w/ scenario {} begins", fname, bs.get_scenario(i));
    cosmology.random = RandomNumber::get_instance().get(); // clear cosmolike cache
    if (bs.is_allsims_file_set()) {// new api
      init_baryons_contamination(bs.get_scenario(i), bs.get_allsims_file());
    } else { // old api
      init_baryons_contamination(bs.get_scenario(i));
    }
    vector dv = ip.sqzd_theory_data_vector(compute_Mx2pt_N_masked<N,M>(ord));
    D.col(i) = dv - dv_dm;
    debug("{}: comp. dv w/ scenario {} ends", fname, bs.get_scenario(i));
  }
  reset_bary_struct();
  cosmology.random = RandomNumber::get_instance().get();  // clear cosmolike cache

  // weight the diff matrix by inv_L; then SVD ----------------------------  
  matrix U, V;
  vector s;
  arma::svd(U, s, V, inv_L * D);

  // compute PCs ----------------------------------------------------------
  matrix PC = L * U; 

  // Expand the number of dims --------------------------------------------
  matrix R = matrix(ndata, nscenarios); 
  for (int i=0; i<nscenarios; i++) {
    R.col(i) = ip.expand_theory_data_vector_from_sqzd(PC.col(i));
  }
  return R;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <int N, int M> 
void IP::set_mask(std::string mask_filename, arma::Col<int>::fixed<M> ord)
{
  static constexpr std::string_view debug1 = "{}: mask file {} left {} non-masked elements after masking"sv;
  static constexpr std::string_view errleii = "logical error, internal inconsistent"sv;
  static constexpr std::string_view fname = "IP::set_mask"sv;
  static constexpr std::string_view errornset = "{}: {} not set (?ill-defined) prior to this function call"sv;
  static constexpr std::string_view errorndp = "{}: mask file {} left no data points after masking"sv;
  static constexpr std::string_view errorim = "{}: inconsistent mask"sv;
  using matrix = arma::Mat<double>;
  if (!(like.Ndata>0)) [[unlikely]] {
    spdlog::critical(errornset,fname, "like.Ndata"); exit(1);
  }

  this->ndata_ = like.Ndata;
  this->mask_.set_size(this->ndata_);
  this->mask_filename_ = mask_filename;
  
  matrix table = read_table(mask_filename);
  if (static_cast<int>(table.n_rows) != this->ndata_) [[unlikely]] {
    spdlog::critical(errorim, fname); exit(1);
  }

  for (int i=0; i<this->ndata_; i++) {
    this->mask_(i) = static_cast<int>(table(i,1)+1e-13);
    if (!(0 == this->mask_(i) || 1 == this->mask_(i))) [[unlikely]] {
      spdlog::critical(errorim, fname); exit(1);
    }
  }

  arma::Col<int>::fixed<M> sizes = compute_data_vector_Mx2pt_N_sizes<N,M>();
  arma::Col<int>::fixed<M> start = compute_data_vector_Mx2pt_N_starts<N,M>(ord);
  if (0 == like.shear_shear) {
    const int A = start(0);
    const int B = A + sizes(0);
    for (int i=A; i<B; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.shear_pos) {
    const int A = start(1);
    const int B = A + sizes(1);
    for (int i=A; i<B; i++) {
      this->mask_(i) = 0;
    }
  }
  if (0 == like.pos_pos) {
    const int A = start(2);
    const int B = A + sizes(2);
    for (int i=A; i<B; i++) {
      this->mask_(i) = 0;
    }
  }
  if constexpr (6 == M) {
    if (0 == like.gk) {
      const int A = start(3);
      const int B = A + sizes(3);;
      for (int i=A; i<B; i++) {
        this->mask_(i) = 0.0;
      }
    }
    if (0 == like.ks)  {
      const int A = start(4);
      const int B = A + sizes(4);
      for (int i=A; i<B; i++) {
        this->mask_(i) = 0.0;
      }
    }
    if (0 == like.kk) {
      const int A = start(5);
      const int B = A + sizes(5);
      for (int i=A; i<B; i++) {
        this->mask_(i) = 0.0;
      }
    }
  }
  
  this->ndata_sqzd_ = arma::accu(this->mask_);
  if(!(this->ndata_sqzd_>0)) [[unlikely]] {
    spdlog::critical(errorndp, fname, mask_filename); exit(1);
  }

  spdlog::debug(debug1, fname, mask_filename, this->ndata_sqzd_);

  this->index_sqzd_.set_size(this->ndata_);
  {
    int j=0;
    for(int i=0; i<this->ndata_; i++) {
      if(this->get_mask(i) > 0) {
        this->index_sqzd_(i) = j;
        j++;
      }
      else {
        this->index_sqzd_(i) = -1;
      }
    }
    if(j != this->ndata_sqzd_) [[unlikely]] {
      spdlog::critical("{}: {} mask operation", fname, errleii); exit(1);
    }
  }
  this->is_mask_set_ = true;
}

}  // namespace cosmolike_interface
#endif // HEADER GUARD
