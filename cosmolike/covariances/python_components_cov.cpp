#include <carma.h>
#include "generic_interface_cov.hpp"
#include "covariance_wrapper_cov.hpp"
#include "notebook_bindings_cov.hpp"

namespace py = pybind11;
namespace cosmolike_interface {

// ---------------------------------------------------------------------------
// Notebook bindings of the covariance components and matrix assemblers.
//
// Python conversion is kept here, apart from the Armadillo calculations.
// Each binding checks rank and copies its arguments before calling C++.
//
// These functions are registered on the module that bind_covariance
// receives (generic_interface_cov.cpp). Its covariance submodule holds the
// production functions of the same names (components_interface_cov.cpp,
// matrix_interface_cov.cpp), which borrow C-contiguous float64 or int32
// NumPy arrays without copying and reject any other array. Both layers
// call the same C routines.
//
// Every lambda below follows the same three steps:
//   1. notebook_input_cov checks the rank of each array argument and
//      copies it into a private, owning arma::Col (rank 1), arma::Mat
//      (rank 2) or arma::Cube (rank 3), as float64 for physical values and
//      int32 for IDs. C-order, Fortran-order, sliced and read-only NumPy
//      arrays are all accepted, and the caller's array is never modified.
//   2. The *_cpp wrapper (components_wrapper_cov.cpp or
//      covariance_wrapper_cov.cpp) checks shapes and domains and calls C.
//   3. CARMA exports the returned Armadillo object as a NumPy array whose
//      memory belongs to a capsule, a small Python object that frees the
//      Armadillo storage when the last array using it disappears. The
//      result therefore stays valid after later calls. Its shape is the
//      Armadillo shape, for example [n_rows,n_cols], and it keeps
//      Armadillo's column-major (Fortran) memory order. Indexing is
//      unaffected, but the C-order production functions need
//      np.ascontiguousarray of a 2D or 3D result.
// CARMA exports an arma::Col as a 2D [n,1] array; reshape(-1) returns the
// 1D [n] view of it that Python callers expect. pybind11 converts scalar
// arguments itself, and py::arg("name") = value sets a keyword default.
// The comment above each binding lists its shapes; the docstrings, which
// Python's help() shows, add units and scope.
// ---------------------------------------------------------------------------
void bind_covariance_components(py::module_& module)
{

  // nquad -> [2,nquad]: Gauss-Legendre nodes (row 0) and weights (row 1).
  module.def("covariance_integration_rule", [](
          const int nquad) {

        return covariance_integration_rule_cpp(
            nquad);
      },
      "Precomputed GSL nodes and weights [2,nquad] on [-1,1]. "
      "Allowed sizes: 64,96,128,256,512,1024; smaller rules are rejected.",
      py::arg("nquad"));

  // edges_rad [nbin+1] in radians, mask_cl [nmask] (raw C_L^W from L=0),
  // scalar_kernel [nbin,nmask] (w operator rows) -> [nbin] in sr^2.
  module.def("covariance_mask_pair_area", [](
          const py::object& edges_rad,
          const py::object& mask_cl,
          const double area_sr,
          const py::object& scalar_kernel) {
        const arma::Col<double> edges_rad_input =
            notebook_input_cov<arma::Col<double>>(edges_rad, 1);
        const arma::Col<double> mask_cl_input =
            notebook_input_cov<arma::Col<double>>(mask_cl, 1);
        const arma::Mat<double> scalar_kernel_input =
            notebook_input_cov<arma::Mat<double>>(scalar_kernel, 2);
        const arma::Col<double> result = covariance_mask_pair_area_cpp(
            edges_rad_input,
            mask_cl_input, area_sr, scalar_kernel_input);
        return carma::col_to_arr(result).attr("reshape")(-1);
      },
      "Ordered-pair area [nbin] in sr^2 from a raw mask and bin kernel.",
      py::arg("edges_rad"), py::arg("mask_cl"),
      py::arg("area_sr"), py::arg("scalar_kernel"));

  // mask_cl [nmask], distance [nnode], power [nnode,nmask] -> [nnode].
  module.def("covariance_ssc_mask_variance", [](
          const py::object& mask_cl,
          const double area_sr,
          const py::object& distance,
          const py::object& power) {
        const arma::Col<double> mask_cl_input =
            notebook_input_cov<arma::Col<double>>(mask_cl, 1);
        const arma::Col<double> distance_input =
            notebook_input_cov<arma::Col<double>>(distance, 1);
        const arma::Mat<double> power_input =
            notebook_input_cov<arma::Mat<double>>(power, 2);
        const arma::Col<double> result = covariance_ssc_mask_variance_cpp(
            mask_cl_input,
            area_sr, distance_input, power_input);
        return carma::col_to_arr(result).attr("reshape")(-1);
      },
      "Long-mode Limber background strength [nnode], in c/H0 length units.",
      py::arg("mask_cl"), py::arg("area_sr"),
      py::arg("distance"), py::arg("power"));

  // distance [nnode], signal [nrow], and pair_window, mean_window and
  // power_response [nrow,nnode] -> response [nrow,nnode].
  module.def("covariance_ssc_shell_response", [](
          const py::object& distance,
          const py::object& signal,
          const py::object& pair_window,
          const py::object& mean_window,
          const py::object& power_response) {
        const arma::Col<double> distance_input =
            notebook_input_cov<arma::Col<double>>(distance, 1);
        const arma::Col<double> signal_input =
            notebook_input_cov<arma::Col<double>>(signal, 1);
        const arma::Mat<double> pair_window_input =
            notebook_input_cov<arma::Mat<double>>(pair_window, 2);
        const arma::Mat<double> mean_window_input =
            notebook_input_cov<arma::Mat<double>>(mean_window, 2);
        const arma::Mat<double> power_response_input =
            notebook_input_cov<arma::Mat<double>>(power_response, 2);
        return covariance_ssc_shell_response_cpp(
            distance_input,
            signal_input, pair_window_input, mean_window_input,
            power_response_input);
      },
      "Observable response [nrow,nnode], including projected mean subtraction.",
      py::arg("distance"), py::arg("signal"),
      py::arg("pair_window"), py::arg("mean_window"),
      py::arg("power_response"));

  // a [na], k [na,nk], lnm_edges [npanel+1] -> tuple (I11 [na,nk],
  // moments [5,na,nk*(nk+1)/2] or None).
  module.def("covariance_halo_moments", [](
          const py::object& a,
          const py::object& k,
          const py::object& lnm_edges,
          const int nquad,
          const bool pair_moments) {
        const arma::Col<double> a_input =
            notebook_input_cov<arma::Col<double>>(a, 1);
        const arma::Mat<double> k_input =
            notebook_input_cov<arma::Mat<double>>(k, 2);
        const arma::Col<double> lnm_edges_input =
            notebook_input_cov<arma::Col<double>>(lnm_edges, 1);
        return covariance_halo_moments_cpp(
            a_input, k_input,
            lnm_edges_input, nquad, pair_moments);
      },
      "Return I11 [na,nk] and moments [5,na,nk*(nk+1)/2]. "
      "pair_moments=False omits pair sums and returns (I11, None). "
      "Both use the cb halo convention.",
      py::arg("a"), py::arg("k"),
      py::arg("lnm_edges"), py::arg("nquad"),
      py::arg("pair_moments") = true);

  // k [nk] -> power [nk]; k [nrow,ncol] -> power [nrow,ncol]. The
  // trailing return type -> py::object lets one lambda return either rank;
  // py::cast converts the matrix result with CARMA.
  module.def("covariance_power",
      [](const double a, const py::object& k,
         const bool linear) -> py::object {
        if (py::cast<int>(k.attr("ndim")) == 1) {
          const arma::Col<double> grid =
              notebook_input_cov<arma::Col<double>>(k, 1);
          const arma::Col<double> result =
              covariance_power_vector_cpp(a, grid, linear);
          return carma::col_to_arr(result).attr("reshape")(-1);
        }
        const arma::Mat<double> grid =
            notebook_input_cov<arma::Mat<double>>(k, 2);
        return py::cast(covariance_power_cpp(a, grid, linear));
      },
      "Read a k vector or matrix at one a; preserve its physical axes.",
      py::arg("a"), py::arg("k"), py::arg("linear") = true);

  // log10k [nrow,ncol] -> linear power [nrow,ncol]; the scalar shift
  // carries the per-shell factor: k = 10^(log10k+shift). Same contract as
  // the production binding of the same name.
  module.def("covariance_power_logk",
      [](const double a, const py::object& log10k,
         const double shift) -> py::object {
        const arma::Mat<double> grid =
            notebook_input_cov<arma::Mat<double>>(log10k, 2);
        return py::cast(covariance_power_logk_cpp(a, grid, shift));
      },
      "Linear power for a base-10 log-wavenumber matrix plus one shift.",
      py::arg("a"), py::arg("log10k"), py::arg("shift"));

  // Fused form: P(|K+Q|) evaluated block by block from the log table;
  // bit-for-bit covariance_power_logk + covariance_tree_averages.
  module.def("covariance_tree_averages_logk",
      [](const py::object& k, const py::object& pk,
         const py::object& corner, const py::object& weight,
         const double a, const py::object& log10s,
         const double shift) -> py::object {
        return py::cast(covariance_tree_averages_logk_cpp(
            notebook_input_cov<arma::Mat<double>>(k, 2),
            notebook_input_cov<arma::Mat<double>>(pk, 2),
            notebook_input_cov<arma::Col<double>>(corner, 1),
            notebook_input_cov<arma::Col<double>>(weight, 1),
            a,
            notebook_input_cov<arma::Mat<double>>(log10s, 2),
            shift));
      },
      "Planar P/B/T averages [3,npair] from the log table plus one shift.",
      py::arg("k"), py::arg("pk"), py::arg("corner"), py::arg("weight"),
      py::arg("a"), py::arg("log10s"), py::arg("shift"));

  // k and pk [2,npair], corner and weight [nangle], ps [npair,nangle]
  // -> averages [3,npair]: <P>, <B_tree>, <T_tree>.
  module.def("covariance_tree_averages", [](
          const py::object& k,
          const py::object& pk,
          const py::object& corner,
          const py::object& weight,
          const py::object& ps) {
        const arma::Mat<double> k_input =
            notebook_input_cov<arma::Mat<double>>(k, 2);
        const arma::Mat<double> pk_input =
            notebook_input_cov<arma::Mat<double>>(pk, 2);
        const arma::Col<double> corner_input =
            notebook_input_cov<arma::Col<double>>(corner, 1);
        const arma::Col<double> weight_input =
            notebook_input_cov<arma::Col<double>>(weight, 1);
        const arma::Mat<double> ps_input =
            notebook_input_cov<arma::Mat<double>>(ps, 2);
        return covariance_tree_averages_cpp(
            k_input, pk_input,
            corner_input, weight_input, ps_input);
      },
      "Planar P/B/T averages [3,npair] from supplied linear-power inputs.",
      py::arg("k"), py::arg("pk"),
      py::arg("corner"), py::arg("weight"),
      py::arg("ps"));

  // pk and i11 [2,npoint], moments [5,npoint], tree [3,npoint]
  // -> terms [5,npoint].
  module.def("covariance_halo_trispectrum", [](
          const py::object& pk,
          const py::object& i11,
          const py::object& moments,
          const py::object& tree) {
        const arma::Mat<double> pk_input =
            notebook_input_cov<arma::Mat<double>>(pk, 2);
        const arma::Mat<double> i11_input =
            notebook_input_cov<arma::Mat<double>>(i11, 2);
        const arma::Mat<double> moments_input =
            notebook_input_cov<arma::Mat<double>>(moments, 2);
        const arma::Mat<double> tree_input =
            notebook_input_cov<arma::Mat<double>>(tree, 2);
        return covariance_halo_trispectrum_cpp(
            pk_input,
            i11_input, moments_input, tree_input);
      },
      "Return five halo terms [5,npoint]: 1h,2h(13),2h(22),3h,4h.",
      py::arg("pk"), py::arg("i11"),
      py::arg("moments"), py::arg("tree"));

  // inputs [6,npoint] -> [2,npoint]: P_halo and dP/d(delta_b).
  module.def("covariance_halo_response", [](
          const py::object& inputs,
          const double growth_coefficient,
          const double dilation_coefficient,
          const bool fractional) {
        const arma::Mat<double> inputs_input =
            notebook_input_cov<arma::Mat<double>>(inputs, 2);
        return covariance_halo_response_cpp(
            inputs_input,
            growth_coefficient, dilation_coefficient, fractional);
      },
      "Return halo power and dimensional response [2,npoint].",
      py::arg("inputs"), py::arg("growth_coefficient"),
      py::arg("dilation_coefficient"), py::arg("fractional") = true);

  // left [nleft,nnode], right [nright,nnode], weight [nnode]
  // -> [nleft,nright].
  module.def("covariance_project", [](
          const py::object& left,
          const py::object& right,
          const py::object& weight) {
        const arma::Mat<double> left_input =
            notebook_input_cov<arma::Mat<double>>(left, 2);
        const arma::Mat<double> right_input =
            notebook_input_cov<arma::Mat<double>>(right, 2);
        const arma::Col<double> weight_input =
            notebook_input_cov<arma::Col<double>>(weight, 1);
        return covariance_project_cpp(
            left_input, right_input,
            weight_input);
      },
      "Contract rows through common weights; return an owned matrix.",
      py::arg("left"), py::arg("right"),
      py::arg("weight"));

  // cross_spectra [4,nell], cross_noise [4] in steradians -> [nell], one
  // value per multipole ell_min, ell_min+1, ... .
  module.def("covariance_gaussian_wick", [](
          const py::object& cross_spectra,
          const py::object& cross_noise,
          const int ell_min,
          const double fsky,
          const bool include_noise_noise) {
        const arma::Mat<double> cross_spectra_input =
            notebook_input_cov<arma::Mat<double>>(cross_spectra, 2);
        const arma::Col<double> cross_noise_input =
            notebook_input_cov<arma::Col<double>>(cross_noise, 1);
        const arma::Col<double> result = covariance_gaussian_wick_cpp(
            cross_spectra_input,
            cross_noise_input, ell_min, fsky, include_noise_noise);
        return carma::col_to_arr(result).attr("reshape")(-1);
      },
      "Gaussian AB,CD covariance from AC,BD,AD,BC rows on consecutive ell.",
      py::arg("cross_spectra"), py::arg("cross_noise"),
      py::arg("ell_min"), py::arg("fsky"),
      py::arg("include_noise_noise") = false);

  // edges_rad [nbin+1] in radians -> operators [4,nbin,ell_max+1].
  module.def("covariance_realspace_operator", [](
          const py::object& edges_rad,
          const int ell_max,
          const int nquad) {
        const arma::Col<double> edges_rad_input =
            notebook_input_cov<arma::Col<double>>(edges_rad, 1);
        return covariance_realspace_operator_cpp(
            edges_rad_input,
            ell_max, nquad);
      },
      "Return [4,nbin,ell_max+1] full-sky xi+,xi-,gamma_t,w operators.",
      py::arg("edges_rad"), py::arg("ell_max"), py::arg("nquad"));

  // first and last [nband], int32 -> weights [nband,nell].
  module.def("covariance_bandpower_operator", [](
          const py::object& first,
          const py::object& last,
          const int ell_min,
          const int nell) {
        const arma::Col<int> first_input =
            notebook_input_cov<arma::Col<int>>(first, 1);
        const arma::Col<int> last_input =
            notebook_input_cov<arma::Col<int>>(last, 1);
        return covariance_bandpower_operator_cpp(
            first_input,
            last_input, ell_min, nell);
      },
      "Return normalized integer-mode weights [nband,nell]; bounds inclusive.",
      py::arg("first"), py::arg("last"),
      py::arg("ell_min"), py::arg("nell"));

  // fields [4] int32, noise_ab [2] in steradians, pair_area_sr2 in sr^2
  // -> one dimensionless covariance entry (a Python float).
  module.def("covariance_noise_pair", [](
          const int probe_left,
          const int probe_right,
          const py::object& fields,
          const py::object& noise_ab,
          const double pair_area_sr2) {
        const arma::Col<int> fields_input =
            notebook_input_cov<arma::Col<int>>(fields, 1);
        const arma::Col<double> noise_ab_input =
            notebook_input_cov<arma::Col<double>>(noise_ab, 1);
        return covariance_noise_pair_cpp(
            probe_left, probe_right,
            fields_input, noise_ab_input, pair_area_sr2);
      },
      "Pure real-space count/shape noise for one angular-bin pair area.",
      py::arg("probe_left"), py::arg("probe_right"),
      py::arg("fields"), py::arg("noise_ab"),
      py::arg("pair_area_sr2"));
}

// Matrix assemblers: the connected projection and the real-space and
// Fourier Gaussian matrices, declared in covariance_wrapper_cov.hpp. They
// follow the same copy, check and export steps as the components above.
void bind_covariance_wrappers(py::module_& module)
{
  // probes [nobs] int32, pair_window [nobs,nnode], projected
  // [4*nbin,4*nbin,nnode], measure [nnode] -> [nobs*nbin,nobs*nbin].
  module.def("covariance_project_connected", [](
          const py::object& probes,
          const py::object& pair_window,
          const py::object& projected,
          const py::object& measure) {
        const arma::Col<int> probes_input =
            notebook_input_cov<arma::Col<int>>(probes, 1);
        const arma::Mat<double> pair_window_input =
            notebook_input_cov<arma::Mat<double>>(pair_window, 2);
        const arma::Cube<double> projected_input =
            notebook_input_cov<arma::Cube<double>>(projected, 3);
        const arma::Col<double> measure_input =
            notebook_input_cov<arma::Col<double>>(measure, 1);
        return covariance_project_connected_cpp(
            probes_input,
            pair_window_input, projected_input, measure_input);
      },
      R"doc(Project a connected matter table through every catalog pair.

probes[observable] is int32: 0 xi+, 1 xi-, 2 gamma_t, 3 w.
pair_window[observable,node] contains W_A*W_B in (c/H0)^-2.
projected[4*nbin,4*nbin,node] contains the already angularly/band-projected
matter trispectrum in (c/H0)^9, with bin inside probe on both axes.
Only its probe/bin upper triangle is consumed and mirrored in the result.
measure[node] supplies dchi/(area*f_K^6), in (c/H0)^-5. These three arrays
are float64 and share their radial nodes. Signed inputs are
retained. The common matter model and its approximations belong to the
caller; this operation does not compute halo physics or add SSC/noise.

Returns an owned symmetric [nobservable*nbin,nobservable*nbin] matrix,
with bin inside observable. Every entry retains increasing radial-node
sum order. No input or cosmology/likelihood state is changed.
)doc",
      py::arg("probes"), py::arg("pair_window"),
      py::arg("projected"), py::arg("measure"));

  // spectra [nell,nfield,nfield], noise [nfield], rows [nobs,3] int32,
  // operators [4,nbin,nell], pair_area_sr2 [nbin], optional b_spectra
  // like spectra -> [nobs*nbin,nobs*nbin]. noise holds white-noise powers
  // in steradians: 1/n and sigma_e^2/n, with n per steradian.
  module.def("covariance_gaussian_real",
      [](
          const py::object& spectra,
          const py::object& noise,
          const py::object& rows,
          const py::object& operators,
          const int ell_min,
          const double area_sr,
          const py::object& pair_area_sr2,
          const py::object& b_spectra) {
        const arma::Cube<double> spectra_input =
            notebook_input_cov<arma::Cube<double>>(spectra, 3);
        const arma::Col<double> noise_input =
            notebook_input_cov<arma::Col<double>>(noise, 1);
        const arma::Mat<int> rows_input =
            notebook_input_cov<arma::Mat<int>>(rows, 2);
        const arma::Cube<double> operators_input =
            notebook_input_cov<arma::Cube<double>>(operators, 3);
        const arma::Col<double> pair_area_sr2_input =
            notebook_input_cov<arma::Col<double>>(pair_area_sr2, 1);
        // None leaves b_input empty, which the wrapper reads as E only.
        arma::Cube<double> b_input;
        if (!b_spectra.is_none()) {
          b_input = notebook_input_cov<arma::Cube<double>>(b_spectra, 3);
        }
        return covariance_gaussian_real_cpp(
            spectra_input,
            noise_input, rows_input, operators_input, ell_min, area_sr,
            pair_area_sr2_input, b_input);
      },
      R"doc(Compute a real-space Gaussian matrix from supplied field spectra.

spectra[ell,field,field] contains signal in the observed-shear convention;
noise[field] gives independent white shot/shape powers per steradian.
Optional b_spectra has the same axes and units as spectra and contains
source-source BB signals, with galaxy rows/columns zero. EB and gB vanish
by parity. The xi+/xi- BB signs are applied in C; pure noise is not doubled.
rows[observable,3] contains (probe,A,B), with probe=0 xi+, 1 xi-,
2 gamma_t, 3 w. operators[4,bin,ell] covers the same consecutive ell
values as spectra, starting at ell_min>=2. Supply angular-bin-averaged
kernels and positive ordered-pair areas pair_area_sr2[bin]. area_sr is
survey area in steradians. All numeric arrays are float64;
rows is int32. Every internal crossed field pair is required.

Returns an owned symmetric [nobservable*nbin,nobservable*nbin] matrix,
with bins inside each observable. Pure noise is added analytically;
the multipole sum contains signal and mixed noise only. No likelihood
state is changed and no input is modified. This is a Gaussian fsky
calculation, not exact cut-sky mode coupling or non-Gaussian covariance.
)doc",
      py::arg("spectra"), py::arg("noise"),
      py::arg("rows"), py::arg("operators"),
      py::arg("ell_min"), py::arg("area_sr"),
      py::arg("pair_area_sr2"), py::arg("b_spectra") = py::none());

  // spectra [nell,nfield,nfield], noise [nfield] in steradians, pairs
  // [nobs,2] int32, operators [nband,nell] -> [nobs*nband,nobs*nband].
  module.def("covariance_gaussian_fourier",
      [](
          const py::object& spectra,
          const py::object& noise,
          const py::object& pairs,
          const py::object& operators,
          const int ell_min,
          const double area_sr) {
        const arma::Cube<double> spectra_input =
            notebook_input_cov<arma::Cube<double>>(spectra, 3);
        const arma::Col<double> noise_input =
            notebook_input_cov<arma::Col<double>>(noise, 1);
        const arma::Mat<int> pairs_input =
            notebook_input_cov<arma::Mat<int>>(pairs, 2);
        const arma::Mat<double> operators_input =
            notebook_input_cov<arma::Mat<double>>(operators, 2);
        return covariance_gaussian_fourier_cpp(
            spectra_input,
            noise_input, pairs_input, operators_input, ell_min, area_sr);
      },
      R"doc(Compute a Gaussian bandpower matrix from supplied field spectra.

spectra[ell,field,field] contains observed signal on consecutive integer
multipoles starting at ell_min>=0. noise[field] is independent white
shot/shape power. pairs[observable,2] contains field IDs (A,B).
operators[band,ell] supplies normalized band weights; the supplied bands
may overlap. area_sr is the common survey area in steradians.
Use float64 arrays and int32 pairs. Internal crossed spectra
are required even when excluded from the measured bandpower vector.

Returns an owned symmetric [nobservable*nband,nobservable*nband] matrix,
with bands inside each observable. The harmonic Wick sum includes pure
noise. No state or input is changed. This fsky Gaussian calculation does
not include SSC, cNG or exact cut-sky mode coupling.
)doc",
      py::arg("spectra"), py::arg("noise"),
      py::arg("pairs"), py::arg("operators"),
      py::arg("ell_min"), py::arg("area_sr"));
}
}
