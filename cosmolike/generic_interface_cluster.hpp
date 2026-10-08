#include <string>
#include <vector>
#include <cmath>
#include <string_view>
using namespace std::literals; // enables "sv" literal

// SPDLOG
#include <spdlog/spdlog.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// cosmolike: the core interface (RandomNumber, read_table, the Mx2pt sizes)
#include "cosmolike/generic_interface.hpp"

// cosmolike: the cluster contract (every header carries its extern "C" guard)
#include "cosmolike/structs_cluster.h"
#include "cosmolike/redshift_spline_cluster.h"
#include "cosmolike/radial_weights_cluster.h"
#include "cosmolike/halo_cluster.h"
#include "cosmolike/cosmo2D_cluster.h"

#ifndef __COSMOLIKE_GENERIC_INTERFACE_CLUSTER_HPP
#define __COSMOLIKE_GENERIC_INTERFACE_CLUSTER_HPP

// ============================================================================
// [SECTION] PYTHON INTERFACE OF THE 4x2pt + N CLUSTER ANALYSIS
// ============================================================================
//
// Everything the cluster likelihood needs on the C++ side, written next to
// (and without editing) generic_interface.cpp/.hpp:
//
//   1. setters of the global `cluster` (structs_cluster.h): model choices,
//      probes, richness bins, selection kernels <phi_i|z>, tomographic
//      pairs, mass-observable relation (MOR) and selection-bias parameters.
//      Each setter checks sizes and NaNs and draws a new cache key from
//      RandomNumber only when a value actually changed (fdiff), exactly as
//      the galaxy setters of generic_interface.cpp do (the pair setter
//      init_cluster_pairs always draws, as init_ntomo_powerspectra does);
//   2. the joint data vector ss, gs, gg, cg, N, cc, cs (lighthouse order):
//      block sizes and starts, and the masked theory vector, including the
//      data-vector-level parts of the model: the Y transform of cluster
//      lensing (eq 15 of arXiv 2503.13631, Park et al. 2021) and the
//      scale-dependent selection bias (eq 23);
//   3. IPCluster, the measurement side of the joint vector (mask, data,
//      covariance, chi2). The core IP singleton cannot hold it: its mask
//      setter is hard-wired to the Mx2pt block sizes, and it inverts the
//      full covariance, which is singular in Y space (see IPCluster).
//
// The Python arrays of the cluster statistics, indexed by the bins
// themselves, live in cosmo2D_wrapper_cluster.cpp/.hpp.
//
// Joint data-vector layout (Nt = Ntable.Ntheta, NL = cluster.richness_nbin,
// NRP = NL (NL + 1)/2 richness pairs; blocks in this order):
//
//   ss  xi+ then xi-   [shear pair][theta]              2 Nt Npower_ss
//   gs  gamma_t        [ggl pair][theta]                Nt Npower_gs
//   gg  w_gg           [lens bin][theta]                Nt Npower_gg
//   cg  w_cg           [cg pair][lambda][theta]         Nt NL Npower_cg
//   N   counts         [cluster z bin][lambda]          Nz_c NL
//   cc  w_cc           [cluster z bin][lambda pair][theta]  Nt NRP Npower_cc
//   cs  Sigma (or gamma_t without the Y transform)
//                      [cs pair][lambda][theta]         Nt NL Npower_cs
//
// Every block keeps its slots whether or not its probe is on (a disabled
// block is zeroed by the mask), so 4x2pt+N (gg + cg + N + cc + cs) and
// 6x2pt+N (all seven) read the same data, mask and covariance files.
//
// Files: the data vector and the covariance must be in the space of the
// model. With cluster.ytransform = 1 the cs rows are Sigma = T gamma_t
// and the covariance carries T C T^T on the cs blocks (eq 31); the last
// theta bin of every cs row is identically zero there and IPCluster always
// masks it.

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// Block indices of the joint data vector (the order of the layout above).
// ---------------------------------------------------------------------------
namespace cluster_block
{
  constexpr int ss = 0;     // cosmic shear xi+ and xi-
  constexpr int gs = 1;     // galaxy-galaxy lensing gamma_t
  constexpr int gg = 2;     // galaxy clustering w_gg
  constexpr int cg = 3;     // cluster-galaxy clustering w_cg
  constexpr int N  = 4;     // cluster counts
  constexpr int cc = 5;     // cluster-cluster clustering w_cc
  constexpr int cs = 6;     // cluster lensing (Sigma or gamma_t)
  constexpr int count = 7;  // number of blocks
}

// Number of parameters of each cluster nuisance vector.
constexpr int cluster_nmor_lognormal = 4;  // ln lambda_0, A, sigma_int, B
constexpr int cluster_nselection = 4;      // see structs_cluster.h

// ============================================================================
// [SECTION] CLASS IPCluster: MASK, DATA AND COVARIANCE OF THE JOINT VECTOR
// ============================================================================
//
// The IP singleton of generic_interface.hpp, rewritten for the joint vector
// (two layouts, the same member names):
//
//   full layout - length ndata_ (the sum of the seven block sizes), the
//                 layout of the files and of the theory vector; masked
//                 entries are kept as zeros;
//   sqzd layout - only the mask == 1 entries, compacted in order.
//
// One difference in the covariance recipe. The core IP keeps the file's
// diagonal at masked entries so that the full matrix stays invertible,
// then inverts the full matrix. In Y space the last theta bin of every cs
// row is identically zero (the last row of T is zero), so its variance is
// 0 and the full matrix is singular. IPCluster therefore checks and
// inverts the squeezed matrix, which contains only unmasked entries, and
// expands the inverse back to the full layout with zeros at masked
// entries. Unmasked entries give the same chi2 either way.
// ---------------------------------------------------------------------------
class IPCluster
{
  private:
    static constexpr std::string_view errornv =
      "{}: idx i={} not supported (min={},max={})"sv;
  public:
    static IPCluster& get_instance() {
      static IPCluster instance;
      return instance;
    }

    // forget mask, data and covariance (a new model in the same process)
    void reset();

    bool is_mask_set() const {
      return this->is_mask_set_;
    }
    bool is_data_set() const {
      return this->is_data_set_;
    }
    bool is_inv_cov_set() const {
      return this->is_inv_cov_set_;
    }

    // order: set_mask, then set_data and set_inv_cov (both use the mask)
    void set_mask(std::string mask_filename);

    void set_data(std::string datavector_filename);

    void set_inv_cov(std::string covariance_filename);

    int get_mask(const int ci) const {
      static constexpr std::string_view fn = "IPCluster::get_mask"sv;
      if (ci >= this->ndata_ || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, this->ndata_ - 1);
        exit(1);
      }
      return this->mask_(ci);
    }

    int get_index_sqzd(const int ci) const {
      static constexpr std::string_view fn = "IPCluster::get_index_sqzd"sv;
      if (ci >= this->ndata_ || ci < 0) [[unlikely]] {
        spdlog::critical(errornv, fn, ci, 0, this->ndata_ - 1);
        exit(1);
      }
      return this->index_sqzd_(ci);
    }

    arma::Col<double> expand_theory_data_vector_from_sqzd(
        arma::Col<double> input
      ) const;

    arma::Col<double> sqzd_theory_data_vector(arma::Col<double> input) const;

    double get_chi2(arma::Col<double> datavector) const;

    int get_ndata() const {
      return this->ndata_;
    }
    int get_ndata_sqzd() const {
      return this->ndata_sqzd_;
    }
    arma::Col<int> get_mask() const {
      return this->mask_;
    }
    arma::Col<double> get_dv_masked() const {
      return this->data_masked_;
    }
    arma::Mat<double> get_cov_masked() const {
      return this->cov_masked_;
    }
    arma::Mat<double> get_inv_cov_masked() const {
      return this->inv_cov_masked_;
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
    arma::Col<int> index_sqzd_;
    arma::Col<double> data_masked_;
    arma::Mat<double> cov_masked_;
    arma::Mat<double> inv_cov_masked_;
    arma::Col<double> data_masked_sqzd_;
    arma::Mat<double> cov_masked_sqzd_;
    arma::Mat<double> inv_cov_masked_sqzd_;
    IPCluster() = default;
    IPCluster(IPCluster const&) = delete;
};

// ============================================================================
// [SECTION] INIT AND SET FUNCTIONS
// ============================================================================

// reset_cluster_struct() plus fresh cache keys, the IPCluster state and the
// interface-level Limber switches; call after initial_setup()
void reset_cluster();

// probe combination by name ("4x2pt_N", "6x2pt_N", ...): sets like.* for
// ss/gs/gg (gk, ks, kk off) and cluster.probe_* for N/cs/cc/cg
void init_probes_cluster(std::string possible_probes);

// the four cluster probe flags (0 or 1) one by one
void init_cluster_probes(
    const int N,
    const int cs,
    const int cc,
    const int cg
  );

void init_cluster_model(
    const int mor_model,         // CLUSTER_MOR_*
    const int kernel_mode,       // CLUSTER_KERNEL_*
    const int selection_model,   // CLUSTER_SELECTION_*
    const int ytransform,        // 1: Sigma = Y gamma_t (eq 15)
    const int include_ia,        // 1: source IA in the 2-halo lensing term
    const double magnification   // cluster magnification C_c (eq 28: -2)
  );

// amplitude of the Tinker 2010 cluster mass function: CLUSTER_HMF_ALPHA_FIXED
// (0.368 at every z, DES; default) or CLUSTER_HMF_ALPHA_NORMALIZED (halo.c's
// alpha(a), int b f dnu = 1); draws cluster.random_model on a change
void init_cluster_hmf_alpha_mode(const int hmf_alpha_mode);

// Limber (1) or non-Limber (0) w_cc and w_cg (default: Limber, the paper's
// choice for w_cg; the paper runs w_cc non-Limber). The cosmo2D_cluster.c
// engines implement Limber only: 0 aborts at the first w_cc / w_cg
// evaluation.
void init_cluster_adopt_limber(
    const int adopt_limber_cc,
    const int adopt_limber_cg
  );

void init_cluster_richness_bins(
    arma::Col<double> lambda_min,
    arma::Col<double> lambda_max
  );

// selection kernels: table column 0 = z, column i+1 = <phi_i|z>; nominal
// z_lambda edges of each bin in zbin_min / zbin_max
void set_cluster_zdist(
    arma::Mat<double> input_table,
    arma::Col<double> zbin_min,
    arma::Col<double> zbin_max
  );

// cg_lens_bin(ni) = lens bin paired with cluster bin ni in w_cg (-1: none);
// builds the pair maps single-threaded (they own the pair counts
// cluster.cs/cg/cc_npowerspectra)
void init_cluster_pairs(arma::Col<int> cg_lens_bin);

void set_nuisance_cluster_mor(arma::Col<double> MOR);

void set_nuisance_cluster_selection(arma::Col<double> SEL);

// sizes -> IPCluster::set_mask -> set_data -> set_inv_cov
void init_data_cluster(
    std::string cov,
    std::string mask,
    std::string data
  );

// ============================================================================
// [SECTION] JOINT DATA VECTOR
// ============================================================================

// block sizes and starts in the cluster_block order
arma::Col<int>::fixed<cluster_block::count> compute_data_vector_cluster_sizes();

arma::Col<int>::fixed<cluster_block::count> compute_data_vector_cluster_starts();

// Park et al. 2021 matrix T = 2S + SD of the current theta binning
// (Ntheta x Ntheta): Sigma = T gamma_t
arma::Mat<double> compute_cluster_ytransform_matrix();

// selection-bias factor B(theta) of eq (23) on the data vector,
// (cluster z bin) x (theta bin); ones unless CLUSTER_SELECTION_Y6
arma::Mat<double> compute_cluster_selection_factor();

// masked theory vector in the full layout (zeros off-mask); probes whose
// flag is off stay at zero
arma::Col<double> compute_data_vector_cluster_masked();

}  // namespace cosmolike_interface
#endif // HEADER GUARD
