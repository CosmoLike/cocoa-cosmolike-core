#ifndef COSMOLIKE_PRODUCTION_INTERFACE_COV_HPP
#define COSMOLIKE_PRODUCTION_INTERFACE_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Production conversions borrow contiguous NumPy inputs and allocate owned
// outputs. Numerical work stays in C, shared with the Armadillo wrappers.
void bind_covariance_production(pybind11::module_& parent);
void bind_production_components_cov(pybind11::module_& module);
void bind_production_matrices_cov(pybind11::module_& module);
void bind_production_cluster_cov(pybind11::module_& parent);
}
#endif
