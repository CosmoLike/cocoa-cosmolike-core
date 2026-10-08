#ifndef COSMOLIKE_GENERIC_INTERFACE_COV_HPP
#define COSMOLIKE_GENERIC_INTERFACE_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Register every galaxy/shear covariance binding on a project module:
// the production NumPy bindings in the submodule module.covariance, and
// the Armadillo notebook bindings with the same names on module itself.
// Each project interface calls this once, unless built with
// COSMOLIKE_NO_COVARIANCE.
void bind_covariance(pybind11::module_& module);

// Notebook bindings of the individual components (python_components_cov.cpp).
// Each copies its Python inputs into owning Armadillo containers.
void bind_covariance_components(pybind11::module_& module);
}
#endif
