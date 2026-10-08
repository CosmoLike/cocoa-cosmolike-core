#ifndef COSMOLIKE_GENERIC_INTERFACE_CLUSTER_COV_HPP
#define COSMOLIKE_GENERIC_INTERFACE_CLUSTER_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Register the cluster covariance bindings: production functions in the
// existing submodule module.covariance, notebook functions on module.
// Call bind_covariance(module) first; it creates that submodule.
void bind_covariance_cluster(pybind11::module_& module);
}
#endif
