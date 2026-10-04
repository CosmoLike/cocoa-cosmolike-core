#ifndef COSMOLIKE_GENERIC_INTERFACE_COV_HPP
#define COSMOLIKE_GENERIC_INTERFACE_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
void bind_covariance(pybind11::module_& module);
void bind_covariance_components(pybind11::module_& module);
}
#endif
