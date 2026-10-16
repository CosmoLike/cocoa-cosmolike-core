"""Shared n(z) selection cache round-trip checks for every project.

Live n(z) selection changes must behave like any other cache
invalidation. Flipping the z-column reading (Z_LOW left edges, Z_MID
sample points), the reconstruction (midpoint curve, histogram plus
Akima), the preparation factor or the stage-1 interpolant must take
effect on the very next query, move the fiducial means together with
the curves, and restore every value bit for bit when the original
selection returns.

These are cache-behavior checks, not accuracy checks: the compiled
production readers and the stored fiducial means are compared bitwise
across selection round trips on synthetic histograms, with no
cosmology calculation involved. The means matter because the lens
means are the pivot of the photo-z stretch: a selection change that
rebuilds the curves without recomputing the means is exactly the
staleness these checks exist to refuse.

Each project's tests/data_vector/test_nz_cache_roundtrip.py is a thin
stub: it resolves its own compiled interface module and binds the
three test callables that build_nz_cache_tests returns. The checks are
identical in every project because the machinery under test lives in
the shared core, not in the project.
"""
import ctypes
from pathlib import Path

import numpy as np
import pytest

# How many trailing entries of _state() hold fiducial means (two lens
# bins, then two source bins); the rest are dense reader samples.
NMEANS = 4


def _compiled_readers(ci):
    """The production C readers and mean accessors inside the module.

    ci is the project's compiled interface module; the shared C
    symbols are reached through the same loaded library with ctypes,
    exactly as the projects' reader regressions do.
    """
    library = ctypes.CDLL(str(Path(ci.__file__).resolve()))
    for name in ("nz_source_photoz", "nz_lens_photoz"):
        getattr(library, name).argtypes = [ctypes.c_double, ctypes.c_int]
        getattr(library, name).restype = ctypes.c_double
    for name in ("zmean", "zmean_source"):
        getattr(library, name).argtypes = [ctypes.c_int]
        getattr(library, name).restype = ctypes.c_double
    return library


def _write_histogram(tmp_path, irregular):
    """A two-bin synthetic histogram file, uniform or irregular in z."""
    z = np.arange(9, dtype=float) * .1 + .05
    if irregular:
        z = np.array([.05, .11, .18, .24, .39, .42, .61, .76, .85])
    heights = np.array([0., .2, .7, 1., .4, .6, .1, 0., 0.])
    table = np.column_stack((z, heights, heights[::-1]))
    filename = tmp_path / "histogram.nz"
    np.savetxt(fname=filename, X=table)
    return filename


def _load_sample(ci, filename):
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=str(filename), lens_ntomo=2,
        source_multihisto_file=str(filename), source_ntomo=2)


def _state(library):
    """Everything one n(z) selection owns: curves plus fiducial means.

    The dense samples watch the reconstructed curves; the means watch
    the derived state only the sample loaders compute.
    """
    query = np.linspace(start=.025, stop=1.0, num=2001)
    values = []
    for sample in ("source", "lens"):
        reader = getattr(library, "nz_" + sample + "_photoz")
        for tomo in range(2):
            values.append(np.fromiter(
                (reader(float(x), tomo) for x in query),
                dtype=float, count=query.size))
    means = [library.zmean(tomo) for tomo in range(2)]
    means += [library.zmean_source(tomo) for tomo in range(2)]
    return np.concatenate([np.asarray(values).ravel(), np.asarray(means)])


def build_nz_cache_tests(ci):
    """Build the three round-trip checks against one compiled interface.

    Returns a dict of pytest test callables. The project stub assigns
    each to a module-level name, which is how pytest collects them:

        _suite = build_nz_cache_tests(ci)
        test_zmid_convention_round_trip = _suite["zmid_convention"]
        test_reconstruction_round_trip = _suite["reconstruction"]
        test_interpolant_round_trip = _suite["interpolant"]
    """

    @pytest.mark.parametrize("mode,factor", ((0, 3), (1, 3), (1, 7)))
    @pytest.mark.parametrize("irregular", (False, True))
    def test_zmid_convention_round_trip(irregular, mode, factor, tmp_path):
        """Z_LOW <-> Z_MID flips rebuild curves and means, and restore bitwise."""
        library = _compiled_readers(ci)
        filename = _write_histogram(tmp_path, irregular)
        try:
            ci.initial_setup()
            ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)
            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
            ci.init_photoz_histogram_akima(enabled=mode,
                                           preparation_factor=factor)
            _load_sample(ci, filename)
            base = _state(library)

            # Repeating the same selection must not move a single bit.
            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
            np.testing.assert_array_equal(_state(library), base)

            # The flip reads the same z column as sample points instead
            # of left edges, which moves every histogram edge by half a
            # cell: the curves and the fiducial means must change
            # together, on the very next query.
            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=1)
            flipped = _state(library)
            assert np.all(np.isfinite(flipped))
            assert not np.array_equal(flipped[:-NMEANS], base[:-NMEANS])
            assert not np.array_equal(flipped[-NMEANS:], base[-NMEANS:])

            # Returning to the original reading restores the original
            # state exactly; no residue of the visited selection may
            # survive.
            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
            np.testing.assert_array_equal(_state(library), base)
        finally:
            ci.initial_setup()

    @pytest.mark.parametrize("first,second", (
        ((0, 3), (1, 3)),   # midpoint curve <-> Akima reconstruction
        ((1, 3), (1, 7)),   # Akima preparation factor
        ((1, 7), (0, 3)),   # Akima back to the midpoint curve
    ))
    def test_reconstruction_round_trip(first, second, tmp_path):
        """Reconstruction and factor switches take effect and restore bitwise."""
        library = _compiled_readers(ci)
        filename = _write_histogram(tmp_path, irregular=True)
        try:
            ci.initial_setup()
            ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)
            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
            ci.init_photoz_histogram_akima(enabled=first[0],
                                           preparation_factor=first[1])
            _load_sample(ci, filename)
            base = _state(library)

            ci.init_photoz_histogram_akima(enabled=second[0],
                                           preparation_factor=second[1])
            switched = _state(library)
            assert np.all(np.isfinite(switched))
            assert not np.array_equal(switched, base)

            ci.init_photoz_histogram_akima(enabled=first[0],
                                           preparation_factor=first[1])
            np.testing.assert_array_equal(_state(library), base)
        finally:
            ci.initial_setup()

    def test_interpolant_round_trip(tmp_path):
        """The midpoint curve's stage-1 interpolant switch restores bitwise."""
        library = _compiled_readers(ci)
        filename = _write_histogram(tmp_path, irregular=True)
        try:
            ci.initial_setup()
            ci.init_accuracy_boost(accuracy_boost=1, integration_accuracy=0)
            ci.init_photoz_histogram_akima(enabled=0, preparation_factor=3)
            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
            _load_sample(ci, filename)
            base = _state(library)

            # cspline (0) to linear (1) changes the midpoint curve.
            ci.init_photoz_conventions(interpolation_type=1, zmid_convention=0)
            switched = _state(library)
            assert np.all(np.isfinite(switched))
            assert not np.array_equal(switched, base)

            ci.init_photoz_conventions(interpolation_type=0, zmid_convention=0)
            np.testing.assert_array_equal(_state(library), base)
        finally:
            ci.initial_setup()

    return {
        "zmid_convention": test_zmid_convention_round_trip,
        "reconstruction": test_reconstruction_round_trip,
        "interpolant": test_interpolant_round_trip,
    }
