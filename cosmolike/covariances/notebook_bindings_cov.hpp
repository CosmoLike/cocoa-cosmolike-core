#ifndef COSMOLIKE_NOTEBOOK_BINDINGS_COV_HPP
#define COSMOLIKE_NOTEBOOK_BINDINGS_COV_HPP

#include <algorithm>
#include <carma.h>
#include <armadillo>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <spdlog/spdlog.h>

// Abort on invalid input like the data-vector layer: print through
// the shared logger, then end the process. No C++ exceptions.
using spdlog::critical;
using std::exit;

namespace cosmolike_interface {
// ---------------------------------------------------------------------------
// Copy one NumPy argument into an owning Armadillo container.
//
// Every notebook binding passes its array arguments through this function
// before it calls an Armadillo *_cpp wrapper. Array selects the container:
// rank 1 gives arma::Col, rank 2 arma::Mat and rank 3 arma::Cube. This is
// conversion only: notebook calculations never see Python buffer details.
//
// 1. rank is the number of axes (NumPy ndim) that the wrapper expects;
//    another rank raises ValueError. The argument must be a NumPy array:
//    a Python list has neither ndim nor astype.
// 2. astype(dtype, order="F", copy=True) makes a new, private NumPy array
//    with the container's element type: float64 for double, int32 for int.
//    NumPy's default unsafe casting applies, so an integer argument given
//    as 2.7 silently becomes 2: pass catalog and probe IDs as integers.
//    Fortran order stores the first index fastest, exactly as Armadillo
//    does: matrix entry (i,j) sits at offset i + n_rows*j and cube entry
//    (i,j,k) at i + n_rows*j + n_rows*n_cols*k. copy=True always copies,
//    so the caller's array is never written or aliased, whether it is
//    C-order, Fortran-order, a sliced (strided) view or read-only.
// 3. reinterpret_borrow views that same Python object as a pybind11
//    buffer, without copying. request() then returns its description
//    (data pointer, shape, element count) through the Python buffer
//    protocol. The copy is contiguous, so its elements are adjacent in
//    storage order; owned keeps it alive until this function returns.
// 4. std::copy_n copies those elements into an owning Armadillo object.
//
// CARMA's own input caster is deliberately not used. It can rearrange
// user arrays or keep a pointer into a temporary. Armadillo also stores
// small objects inside the object itself: up to 16 elements for a matrix
// and 64 for a cube. The caster applies the matrix limit to cubes too, so
// a cube of 17 to 64 elements would get inconsistent memory ownership.
// A plain copy into an owning container avoids both problems; the extra
// copy is an accepted notebook cost.
// ---------------------------------------------------------------------------
template <typename Array>
Array notebook_input_cov(const pybind11::object& input, const int rank)
{
  if (pybind11::cast<int>(input.attr("ndim")) != rank) {
    spdlog::critical("{}: covariance input has the wrong array rank",
      "notebook_input_cov");
    std::exit(1);
  }
  const pybind11::object owned = input.attr("astype")(
      pybind11::dtype::of<typename Array::elem_type>(),
      pybind11::arg("order") = "F", pybind11::arg("copy") = true);
  const pybind11::buffer_info buffer =
      pybind11::reinterpret_borrow<pybind11::buffer>(owned).request();

  // Keep the physical axes unchanged. Storage order alone changes: a
  // matrix entry (i,j) or cube entry (i,j,k) retains its original meaning.
  // if constexpr chooses the set_size call at compile time: each template
  // instantiation keeps only the branch valid for its container type.
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
