#ifndef COSMOLIKE_NOTEBOOK_BINDINGS_COV_HPP
#define COSMOLIKE_NOTEBOOK_BINDINGS_COV_HPP

#include <algorithm>
#include <carma.h>
#include <armadillo>
#include <pybind11/numpy.h>

namespace cosmolike_interface {
// Conversion only: notebook calculations never see Python buffer details.
// Take a private Fortran-order copy, matching Armadillo's column order.
// In particular, do not let CARMA's borrowing caster rearrange user arrays
// or keep a pointer into a temporary. Its small-cube move path also assumes
// matrix-sized inline storage, which differs from Armadillo cube storage.
// A plain copy into an owning Armadillo container avoids both problems.
template <typename Array>
Array notebook_input_cov(const pybind11::object& input, const int rank)
{
  if (pybind11::cast<int>(input.attr("ndim")) != rank) {
    throw pybind11::value_error("covariance input has the wrong array rank");
  }
  const pybind11::object owned = input.attr("astype")(
      pybind11::dtype::of<typename Array::elem_type>(),
      pybind11::arg("order") = "F", pybind11::arg("copy") = true);
  const pybind11::buffer_info buffer =
      pybind11::reinterpret_borrow<pybind11::buffer>(owned).request();

  // Keep the physical axes unchanged. Storage order alone changes: a
  // matrix entry (i,j) or cube entry (i,j,k) retains its original meaning.
  Array output;
  if constexpr (arma::is_Cube<Array>::value) {
    output.set_size(buffer.shape[0], buffer.shape[1], buffer.shape[2]);
  } else if constexpr (arma::is_Col<Array>::value) {
    output.set_size(buffer.shape[0]);
  } else {
    output.set_size(buffer.shape[0], buffer.shape[1]);
  }
  std::copy_n(static_cast<const typename Array::elem_type*>(buffer.ptr),
              buffer.size, output.memptr());
  return output;
}
}
#endif
