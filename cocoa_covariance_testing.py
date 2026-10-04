"""Project-boundary checks for the shared galaxy/shear covariance forecast.

Each project supplies its compiled interface and thin survey adapter. These
checks exercise the real catalog files with a small measured subset. They
check assembly and state/thread consistency, not integration convergence.
Independent component references live in the LSST covariance tests.
"""

import json

import numpy as np
from threadpoolctl import threadpool_limits

from cosmolike_notebook_utils import covariance as cov
from cosmolike_notebook_utils.covariance.forecast import save_forecast


def check_project_forecast(interface, survey, expected_sizes, directory):
    """Check a project's inputs, two measurement spaces and saved metadata.

    Arguments:
        interface = compiled project module with covariance bindings.
        survey = project adapter with configuration, initialize and compute.
        expected_sizes = (real, Fourier) full measured vector lengths.
        directory = temporary output directory supplied by the calling test.
    Returns:
        Nothing; assertions report a violated contract to pytest.
    Side effects:
        Runs CAMB once, replaces this process's cosmology/nuisance state,
        exercises one and eight OpenMP threads and writes temporary archives.
        No likelihood matrix, mask or reference snapshot is read or changed.
    """
    settings = survey.configuration(accuracy_boost=1)
    nlens = len(settings["lens_density_arcmin2"])
    nsource = len(settings["source_density_arcmin2"])
    full_rows = cov.observable_rows(
        nlens=nlens, nsource=nsource,
        excluded_gammat=settings["excluded_gammat"],
    )
    assert len(full_rows)*(len(settings["theta_edges_arcmin"])-1) == expected_sizes[0]
    nfourier = np.count_nonzero(full_rows[:, 0] != 1)
    assert nfourier*len(settings["band_first"]) == expected_sizes[1]
    assert len(settings["sigma_e_component"]) == nsource

    # Scientific bins must stay fixed when only numerical accuracy changes.
    # Otherwise a comparison would mix quadrature error with a new observable.
    refined = survey.configuration(accuracy_boost=2)
    for key in ("band_first", "band_last", "theta_edges_arcmin"):
        np.testing.assert_array_equal(settings[key], refined[key])
    # Table refinement must leave the GSL rule fixed. Only the independent
    # integration level advances radial, mass and angular quadrature.
    integrated = survey.configuration(integration_accuracy=1)
    for key in ("radial_nquad", "angle_nquad", "halo_mass_nquad", "tree_nquad"):
        assert refined[key] == settings[key]
        assert settings[key] == 96
        assert integrated[key] == 128
    assert len(refined["ng_ell"]) > len(settings["ng_ell"])

    # Use each real redshift distribution and the project's noise inputs,
    # while reducing the quadrature size for this assembly check. These
    # settings are deliberately not an inference-accuracy prescription.
    settings.update({
        "ell_max": 160,
        "ng_ell": np.geomspace(2.5, 160.5, 12)-0.5,
        "mask_ell_max": 128,
        "radial_nquad": 64,
        "angle_nquad": 64,
        "nwindow": 1025,
        "halo_mass_nquad": 64,
        "tree_nquad": 64,
        "tree_npanel": 16,
        "response_step": 0.005,
        "theta_edges_arcmin": np.array([15.0, 45.0, 120.0]),
        "band_first": np.array([20, 60], dtype=np.int32),
        "band_last": np.array([59, 160], dtype=np.int32),
    })
    # Select one retained row of each probe. A project's measured-pair cuts
    # can exclude the first lens paired with the first source, as in Roman KL.
    selected_rows = []
    for probe in (0, 2, 3):
        candidates = full_rows[full_rows[:, 0] == probe]
        selected_rows.append(candidates[0])
    rows = np.ascontiguousarray(selected_rows, dtype=np.int32)

    # Serial BLAS prevents matrix diagnostics from nesting another thread
    # team inside the explicitly controlled CosmoLike OpenMP calculation.
    with threadpool_limits(limits=1, user_api="blas"):
        survey.initialize(interface=interface, settings=settings)
        for space in ("real", "fourier"):
            previous = None
            for threads in (1, 8):
                interface.set_omp_threads(n=threads)
                result = survey.compute(
                    interface=interface, settings=settings,
                    space=space, rows=rows,
                )
                for component in ("gaussian", "ssc", "cng", "total"):
                    matrix = result[component]
                    assert matrix.shape == (6, 6)
                    assert np.all(np.isfinite(matrix))
                    np.testing.assert_array_equal(matrix, matrix.T)
                    if previous is not None:
                        np.testing.assert_array_equal(
                            matrix.view(np.uint64),
                            previous[component].view(np.uint64),
                        )
                np.testing.assert_allclose(
                    result["total"],
                    result["gaussian"]+result["ssc"]+result["cng"],
                    rtol=2.e-15, atol=0.0,
                )
                assert cov.covariance_modes(result["total"])["positive_definite"]
                previous = result

            filename = directory/f"{space}.npz"
            save_forecast(result=result, filename=filename)
            with np.load(file=filename, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["total"], result["total"])
                np.testing.assert_array_equal(saved["rows"], rows)
                metadata = json.loads(str(saved["settings_json"]))
                assert metadata["space"] == space
                assert metadata["lens_file"] == settings["lens_file"]
                assert metadata["source_file"] == settings["source_file"]
                assert metadata["area_deg2"] == settings["area_deg2"]
                np.testing.assert_array_equal(metadata["ng_ell"], settings["ng_ell"])
