// ----------------------------------------------------------------------------
// EXPERIMENTAL (future_port_unfinished/fisher): the pybind blocks a project
// adds to its interface/interface.cpp to expose cosmo2d_fisher. Fragment,
// not a compilable file.
//
// 1. Next to the other cosmolike wrapper includes at the top:
//
//      #include "cosmolike_core/future_port_unfinished/fisher/cosmo2d_fisher_wrapper.hpp"
//
// 2. Inside PYBIND11_MODULE(cosmolike_<project>_interface, m):
// ----------------------------------------------------------------------------

  m.def("set_fisher_response",
      &cosmolike_interface::set_fisher_response_cpp,
      "Load the response of the set_cosmology tables to one parameter",
      py::arg("ip").none(false).noconvert(),
      py::arg("dlnOm_dX").none(false),
      py::arg("z_chi").none(false),
      py::arg("dchi_dX").none(false),
      py::arg("z_G").none(false),
      py::arg("dlnG_dX").none(false),
      py::arg("log10k").none(false),
      py::arg("z_P").none(false),
      py::arg("dlnPNL_dX").none(false)
    );

  m.def("reset_fisher_response",
      &cosmolike_interface::reset_fisher_response_cpp,
      "Forget every loaded parameter response"
    );

  m.def("dC_ss_dX_tomo_limber",
      &cosmolike_interface::dC_ss_dX_tomo_limber_cpp,
      "C_ss (NLA) and its analytic derivative wrt every loaded parameter",
      py::arg("l").none(false),
      py::return_value_policy::move
    );

  m.def("dxi_pm_dX_tomo",
      &cosmolike_interface::dxi_pm_dX_tomo_cpp,
      "xi_pm (NLA) and its analytic derivative wrt every loaded parameter",
      py::return_value_policy::move
    );
