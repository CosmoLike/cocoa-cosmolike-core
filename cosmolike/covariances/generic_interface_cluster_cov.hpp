#ifndef COSMOLIKE_GENERIC_INTERFACE_CLUSTER_COV_HPP
#define COSMOLIKE_GENERIC_INTERFACE_CLUSTER_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Cluster covariance bindings are linked only by a cluster-capable project.
void bind_covariance_cluster(pybind11::module_& module);
}
#endif
