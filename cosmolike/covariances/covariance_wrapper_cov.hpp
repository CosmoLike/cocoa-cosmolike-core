#ifndef COSMOLIKE_COVARIANCE_WRAPPER_COV_HPP
#define COSMOLIKE_COVARIANCE_WRAPPER_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Whole-matrix notebook calls; component calls remain separately available.
void bind_covariance_wrappers(pybind11::module_& module);
}
#endif
