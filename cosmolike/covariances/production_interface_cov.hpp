#ifndef COSMOLIKE_PRODUCTION_INTERFACE_COV_HPP
#define COSMOLIKE_PRODUCTION_INTERFACE_COV_HPP

#include <pybind11/pybind11.h>

namespace cosmolike_interface {
// Production bindings for command-line covariance runs, registered in the
// submodule `covariance` of a project's Python module.
//
// Array contract shared by every binding:
// - Inputs are borrowed. Each array argument must already be C-contiguous
//   float64 (int32 for integer maps); noconvert() makes pybind11 reject
//   any other dtype or layout instead of silently copying it. The C code
//   reads the caller's memory and never writes to it.
// - Outputs are new NumPy arrays that own their memory. A later call or a
//   cosmology change cannot modify an earlier result.
// - Each binding checks the shapes, values and initialized state listed
//   in its comment before any C call, so those mistakes raise a Python
//   exception. The C routines keep their own fatal checks, which stop the
//   process (for example flat geometry, or a computed pair area <= 0).
//
// Numerical work stays in the shared C routines, also called by the
// Armadillo notebook wrappers. The conversions call no BLAS routine; the
// C routines run their own OpenMP loops, with the team size chosen by the
// module's set_omp_threads (which also keeps OpenBLAS at one thread).

// Create parent.covariance; register the components, the matrix
// assemblers and covariance_limber_spectra (alias covariance_spectra).
void bind_covariance_production(pybind11::module_& parent);

// Component bindings of components_interface_cov.cpp: quadrature rule,
// Wick and projection sums, angular and band operators, pair noise, mask
// and SSC terms, halo moments, power, tree averages, halo trispectrum
// and halo response.
void bind_production_components_cov(pybind11::module_& module);

// Whole Gaussian and connected matrices, matrix_interface_cov.cpp.
void bind_production_matrices_cov(pybind11::module_& module);

// Cluster bindings of cluster_interface_cov.cpp, added to the existing
// parent.covariance: call bind_covariance_production(parent) first.
void bind_production_cluster_cov(pybind11::module_& parent);
}
#endif
