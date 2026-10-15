"""Compare the two public galaxy n(z) reconstructions on frozen survey data.

The comparison uses the same cosmology, nuisance parameters, mask and
covariance. It measures a distance between model vectors, not a change in
chi2 to the observed data. No frozen inputs or references are rewritten.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def compare_akima_preparation(harness, native, project, example,
                              integration_accuracy, preparation_factor=5):
    """Compare fixed 3x with 5x or 7x preparation at one frozen survey point.

    The first factor changes the smooth model. Integration accuracy only
    changes its numerical evaluation. Keep the cosmology, signed histogram
    cells, mask and covariance fixed; test a live 3 -> factor -> 3 round trip
    and compare the live alternative vector with a separately initialized
    YAML model.

    Parameters
    ----------
    harness, native : module
        Project test harness and compiled interface.
    project, example : str
        Survey name and its frozen NLA example.
    integration_accuracy : int
        Numerical level for both reconstructions, independently of the
        preparation factor.
    preparation_factor : int
        Alternative to compare with three: five or seven.

    Returns
    -------
    tuple
        Measurement dictionary, a dictionary of full-length model vectors,
        and the inverse covariance with masked rows and columns zeroed.
        DES Y6's identity metric has no statistical interpretation.
        COCOA_NZ_REPORT_DIR optionally saves the record and vectors, using
        separate filenames for each preparation factor.
    """
    if preparation_factor not in (5, 7):
        raise ValueError("Comparison factor must be 5 or 7; baseline is 3")
    name = "five" if preparation_factor == 5 else "seven"
    harness.require_cocoa_environment()
    harness.verify_frozen()
    likelihood_name = harness.EXAMPLES[example]["likelihood"]
    info = harness.load_frozen_info(example=example, tatt=False)
    options = info["likelihood"][likelihood_name]
    options["integration_accuracy"] = integration_accuracy
    options["print_datavector"] = False
    options["photoz_histogram_akima"] = 1
    options["photoz_histogram_akima_factor"] = 3

    model = harness.make_model(info=info)
    try:
        point = harness.build_point(model=model, example=example, tatt=False)
        three, inverse, _ = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        likelihood = model.likelihood[likelihood_name]
        boost = float(likelihood.accuracyboost)
        inputs = {}
        for key in ("source_file", "lens_file", "cov_file", "mask_file",
                    "data_vector_file"):
            filename = getattr(likelihood, key, None)
            if filename is not None:
                path = Path(filename).resolve()
                inputs[key] = {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

        # This changes neither the histogram values nor the mode flag.
        # It therefore specifically tests the preparation-factor cache key.
        native.init_photoz_histogram_akima(
            enabled=1, preparation_factor=preparation_factor)
        alternative_live, inverse_live, _ = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        np.testing.assert_array_equal(x=inverse_live, y=inverse)

        native.init_photoz_histogram_akima(enabled=1, preparation_factor=3)
        returned, inverse_returned, _ = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        np.testing.assert_array_equal(x=returned, y=three)
        np.testing.assert_array_equal(x=inverse_returned, y=inverse)
    finally:
        model.close()

    # A fresh YAML model must agree with the live switch, including its
    # integration bounds and lens means used by the photo-z stretch.
    options["photoz_histogram_akima_factor"] = preparation_factor
    model = harness.make_model(info=info)
    try:
        repeated_point = harness.build_point(
            model=model, example=example, tatt=False)
        assert repeated_point == point
        alternative, inverse_yaml, _ = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        np.testing.assert_array_equal(x=alternative, y=alternative_live)
        np.testing.assert_array_equal(x=inverse_yaml, y=inverse)
    finally:
        model.close()

    selected = np.diag(inverse) > 0.0
    np.linalg.cholesky(inverse[np.ix_(selected, selected)])
    np.testing.assert_array_equal(x=inverse[~selected, :], y=0.0)
    difference = alternative - three
    distance = float(difference @ inverse @ difference)
    assert np.isfinite(distance) and distance > 0.0
    record = {
        "project": project,
        "example": example,
        "point": point,
        "input_files": inputs,
        "integration_accuracy": integration_accuracy,
        "accuracyboost": boost,
        "preparation_factors": [3, preparation_factor],
        "delta_chi2": distance,
        "physical_covariance": project != "des_y6",
        "roundtrip_bitwise_equal": True,
        "yaml_live_bitwise_equal": True,
        "inverse_covariance_unchanged": True,
    }
    label = "Delta chi2"
    if project == "des_y6":
        label = "Identity-covariance distance (not survey Delta chi2)"
    print(f"{project}: 3x -> {preparation_factor}x {label} = {distance:.12g}")
    vectors = {"three": three, name: alternative}

    # Retain the vectors when studying integration refinement. A factor
    # suffix keeps the five- and seven-point runs from overwriting each other.
    output_name = os.environ.get("COCOA_NZ_REPORT_DIR")
    if output_name:
        output = Path(output_name).resolve()
        output.mkdir(parents=True, exist_ok=True)
        stem = output / f"{project}_factor{preparation_factor}"
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        np.savez_compressed(file=stem.with_suffix(".npz"), **vectors)

    return record, vectors, inverse


def _snapshot(harness, native, model, point, likelihood_name):
    """Evaluate one state and copy its vector and masked inverse covariance."""
    chi2 = harness.evaluate_chi2(model=model, point=point)
    likelihood = model.likelihood[likelihood_name]
    if getattr(likelihood, "has_clusters", False):
        vector = np.array(native.compute_data_vector_cluster_masked(), copy=True)
        inverse = np.array(native.get_inv_cov_masked_cluster(), copy=True)
    else:
        vector = np.array(native.compute_data_vector_masked(), copy=True)
        inverse = np.array(native.get_inv_cov_masked(), copy=True)
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(inverse)):
        raise AssertionError("The n(z) comparison produced a nonfinite vector or covariance")
    return vector, inverse, float(chi2)


def compare_histogram_reconstructions(harness, native, project, example,
                                      integration_accuracy=6):
    """Test the YAML selection and live 0 -> 1 -> 0 cache round trip.

    Parameters
    ----------
    harness : module
        The project's cocoa_test_utils, which supplies its frozen case.
    native : module
        That project's compiled Cosmolike interface.
    project, example : str
        Project name and frozen NLA example to compare.
    integration_accuracy : int
        The project test supplies its measured practical level. Level 6
        (4096 points per physical Limber panel) is available to reproduce
        the refined reference. The first histogram factor stays three
        at every integration level.

    Returns
    -------
    dict
        Model distance, inputs and settings. If COCOA_NZ_REPORT_DIR is set,
        also save full precision vectors and this record there. DES Y6's
        identity-covariance distance is explicitly a software diagnostic.
    """
    harness.require_cocoa_environment()
    harness.verify_frozen()
    likelihood_name = harness.EXAMPLES[example]["likelihood"]
    info = harness.load_frozen_info(example=example, tatt=False)
    options = info["likelihood"][likelihood_name]
    options["integration_accuracy"] = integration_accuracy
    options["print_datavector"] = False
    # Omission exercises the public YAML default rather than injecting zero.
    options.pop("photoz_histogram_akima", None)
    model = harness.make_model(info=info)
    point = harness.build_point(model=model, example=example, tatt=False)
    try:
        original, inverse, original_chi2 = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        likelihood = model.likelihood[likelihood_name]
        assert likelihood.photoz_histogram_akima == 0
        inputs = {}
        for key in ("source_file", "lens_file", "cov_file", "mask_file", "data_vector_file"):
            filename = getattr(likelihood, key, None)
            if filename is not None:
                path = Path(filename).resolve()
                inputs[key] = {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        boost = float(likelihood.accuracyboost)
        source_file = str(likelihood.source_file)
        lens_file = str(likelihood.lens_file)

        # Changing only the native selector must update the n(z), integration
        # limits, lens means and every dependent projection on this same model.
        native.init_photoz_histogram_akima(enabled=1)
        akima_live, inverse_live, _ = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        np.testing.assert_array_equal(x=inverse_live, y=inverse)
        native.init_photoz_histogram_akima(enabled=0)
        returned, inverse_returned, _ = _snapshot(
            harness=harness, native=native, model=model, point=point,
            likelihood_name=likelihood_name)
        np.testing.assert_array_equal(x=returned, y=original,
                                      err_msg="0 -> 1 -> 0 changed the original vector")
        np.testing.assert_array_equal(x=inverse_returned, y=inverse)
    finally:
        model.close()

    # A separately initialized YAML opt-in must reproduce the live switch.
    # This catches a missing YAML -> likelihood -> C binding even when the
    # native setter itself works correctly.
    options["photoz_histogram_akima"] = 1
    model = harness.make_model(info=info)
    try:
        point_again = harness.build_point(model=model, example=example, tatt=False)
        assert point_again == point
        akima, inverse_yaml, akima_chi2 = _snapshot(
            harness=harness, native=native, model=model, point=point_again,
            likelihood_name=likelihood_name)
        np.testing.assert_array_equal(x=inverse_yaml, y=inverse)
        np.testing.assert_array_equal(x=akima, y=akima_live,
                                      err_msg="YAML Akima differs from the live selector")
    finally:
        model.close()

    # Masked entries have zero rows/columns in this interface. Check the
    # selected covariance metric before using it for the model distance.
    selected = np.diag(inverse) > 0.0
    metric = inverse[np.ix_(selected, selected)]
    np.linalg.cholesky(metric)
    np.testing.assert_array_equal(x=inverse[~selected, :], y=0.0)
    delta = akima - original
    distance = float(delta @ inverse @ delta)
    assert np.isfinite(distance) and distance > 0.0, "The reconstruction selector had no effect"
    record = {
        "project": project,
        "example": example,
        "alignment": "NLA",
        "integration_accuracy": integration_accuracy,
        "accuracyboost": boost,
        "histogram_preparation_factor": 3,
        "input_control": "unchanged frozen histograms; no clipping or resampling in the test",
        "input_files": inputs,
        "point": point,
        "delta_chi2": distance,
        "chi2_to_data_original": original_chi2,
        "chi2_to_data_akima": akima_chi2,
        "full_vector_size": int(original.size),
        "selected_vector_size": int(np.count_nonzero(selected)),
        "physical_covariance": project != "des_y6",
        "default_roundtrip_bitwise_equal": True,
        "yaml_and_live_selection_bitwise_equal": True,
        "inverse_covariance_unchanged": True,
    }
    label = "Delta chi2"
    if project == "des_y6":
        label = "Identity-covariance software distance (not survey Delta chi2)"
    print(f"\n{project}: {label} = {distance:.12g}")
    print(f"  integration_accuracy={integration_accuracy}, accuracyboost={boost:g}")
    print("  YAML selection and 0 -> 1 -> 0 round trip: bitwise agreement")
    areas, area_arrays = histogram_probability_audit(
        native=native, source_file=source_file, lens_file=lens_file)
    record["probability_audit"] = areas
    output_name = os.environ.get("COCOA_NZ_REPORT_DIR")
    if output_name:
        output = Path(output_name).resolve()
        output.mkdir(parents=True, exist_ok=True)
        stem = output / project
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        np.savez_compressed(file=stem.with_suffix(".npz"), original=original,
                            akima=akima, inverse_covariance=inverse, mask=selected,
                            **area_arrays)
    return record


def histogram_probability_audit(native, source_file, lens_file,
                                preparation_factors=(3,), accuracyboost=1.0):
    """Integrate the compiled curves over every original Z_LOW histogram cell.

    Reset shifts/stretch, so this measures reconstruction alone. The optional
    preparation_factors select fixed histogram models; accuracyboost sets only
    their final lookup density (and the default cubic's second fit). Split at
    every reader knot; a
    two-point Gauss rule integrates each cubic exactly. A four-point rule
    independently checks the partition. No Limber or cosmology is involved.
    Returns per-distribution metrics and arrays suitable for area plots.
    """
    import ctypes

    library = ctypes.CDLL(str(Path(native.__file__).resolve()))
    tables = {}
    for sample, filename in (("source", source_file), ("lens", lens_file)):
        tables[sample] = np.loadtxt(fname=filename)
    arrays = {}
    rows = []
    if not preparation_factors or any(factor not in (3, 5, 7)
                                     for factor in preparation_factors):
        raise ValueError("Choose preparation factors from 3, 5 and 7")
    if accuracyboost < 1:
        raise ValueError("The probability audit requires accuracyboost >= 1")
    arms = [(0, "original")]
    names = {3: "three", 5: "five", 7: "seven"}
    for factor in preparation_factors:
        label = "akima" if len(preparation_factors) == 1 else names[factor]
        arms.append((factor, label))
    final_factor = int(np.ceil(5 * accuracyboost))
    for factor, arm in arms:
        enabled = int(factor != 0)
        native.initial_setup()
        native.init_accuracy_boost(accuracy_boost=accuracyboost, integration_accuracy=0)
        native.init_photoz_histogram_akima(
            enabled=enabled, preparation_factor=max(factor, 3))
        native.init_redshift_distributions_from_files(
            lens_multihisto_file=str(lens_file), lens_ntomo=tables["lens"].shape[1] - 1,
            source_multihisto_file=str(source_file), source_ntomo=tables["source"].shape[1] - 1)
        for sample in ("source", "lens"):
            table = tables[sample]
            z = table[:, 0]
            count = z.size
            edges = np.append(z, z[-1] + z[-1] - z[-2])
            if enabled == 0:
                lower = max(z[0], 1.e-5)
                upper = z[-1] + (z[-1] - z[0]) / (count - 1)
                width = (upper - lower) / count
                knots = np.linspace(start=lower + .5 * width + 1.e-15,
                                    stop=lower + (count - .5) * width - 1.e-15,
                                    num=final_factor * count + 1)
            else:
                prepared = factor * count
                width = (edges[-1] - edges[0]) / prepared
                refinement = int(np.ceil(max(4, final_factor) / factor))
                node_count = refinement * (prepared + 1) + 1
                knots = edges[0] - .5 * width + np.arange(node_count) * width / refinement
            inside = (knots > edges[0]) & (knots < edges[-1])
            partition = np.unique(np.concatenate((edges, knots[inside])))
            centres = .5 * (partition[:-1] + partition[1:])
            half_widths = .5 * np.diff(partition)
            cells = np.searchsorted(edges, centres, side="right") - 1
            cells = np.minimum(cells, count - 1)
            target = table[:, 1:].T * np.diff(edges)[None, :]
            target /= np.sum(target, axis=1)[:, None]
            reader = getattr(library, "nz_" + sample + "_photoz")
            reader.argtypes = [ctypes.c_double, ctypes.c_int]
            reader.restype = ctypes.c_double
            masses = []
            for order in (2, 4):
                nodes, weights = np.polynomial.legendre.leggauss(deg=order)
                positions = centres[:, None] + half_widths[:, None] * nodes[None, :]
                bins = []
                for tomo in range(table.shape[1] - 1):
                    # This diagnostic uses the exported production reader,
                    # avoiding a second implementation of its spline rules.
                    values = np.fromiter((reader(float(x), tomo) for x in positions.flat),
                                         dtype=float, count=positions.size)
                    values = values.reshape(positions.shape)
                    areas = half_widths * (values @ weights)
                    bins.append(np.bincount(cells, weights=areas, minlength=count))
                masses.append(np.asarray(bins))
            np.testing.assert_allclose(actual=masses[0], desired=masses[1], atol=2.e-13, rtol=0.0)
            actual = masses[1]
            total = np.sum(actual, axis=1)
            normalized = actual / total[:, None]
            prefix = sample + "_" + arm
            arrays[prefix + "_edges"] = edges
            arrays[prefix + "_target"] = target
            arrays[prefix + "_areas"] = actual
            query = np.linspace(start=edges[0], stop=edges[-1], num=4001)
            curves = []
            for tomo in range(table.shape[1] - 1):
                curves.append(np.fromiter((reader(float(x), tomo) for x in query),
                                          dtype=float, count=query.size))
                rows.append({
                    "sample": sample,
                    "arm": arm,
                    "tomographic_bin": tomo + 1,
                    "total_area": float(total[tomo]),
                    "total_area_error_pct": float(100 * (total[tomo] - 1)),
                    "bin_redistribution_pct": float(50 * np.sum(np.abs(normalized[tomo] - target[tomo]))),
                    "negative_input_cells": int(np.count_nonzero(table[:, tomo + 1] < 0.0)),
                    "two_vs_four_point_max_difference": float(np.max(np.abs(masses[0][tomo] - actual[tomo]))),
                })
            arrays[prefix + "_z"] = query
            arrays[prefix + "_density"] = np.asarray(curves)
    _sample_histogram_edge(native=native, source_file=source_file, lens_file=lens_file,
                           rows=rows, arrays=arrays, arms=arms,
                           accuracyboost=accuracyboost)
    return {"final_grid_factor": final_factor, "shift": 0,
            "lens_stretch": 1, "rows": rows}, arrays


def _sample_histogram_edge(native, source_file, lens_file, rows, arrays,
                           arms, accuracyboost):
    """Add dense C-reader samples of one interior transition for the figure.

    Choose the distribution with the largest original-bin redistribution,
    then its largest interior density jump. The full-domain samples alone
    can undersample a close-up. No spline is fitted to those plotted points.
    """
    import ctypes

    originals = []
    for row in rows:
        if row["arm"] == "original":
            originals.append(row)
    selected = max(originals, key=lambda row: row["bin_redistribution_pct"])
    sample = selected["sample"]
    tomo = selected["tomographic_bin"] - 1
    prefix = sample + "_original"
    edges = arrays[prefix + "_edges"]
    heights = arrays[prefix + "_target"][tomo] / np.diff(edges)
    cell = int(np.argmax(np.abs(np.diff(heights)))) + 1
    left = max(0, cell - 2)
    right = min(heights.size, cell + 3)
    query = np.linspace(start=edges[left], stop=edges[right], num=2001)
    library = ctypes.CDLL(str(Path(native.__file__).resolve()))
    reader = getattr(library, "nz_" + sample + "_photoz")
    reader.argtypes = [ctypes.c_double, ctypes.c_int]
    reader.restype = ctypes.c_double
    source_count = np.loadtxt(fname=source_file).shape[1] - 1
    lens_count = np.loadtxt(fname=lens_file).shape[1] - 1
    for factor, arm in arms:
        native.initial_setup()
        native.init_accuracy_boost(accuracy_boost=accuracyboost, integration_accuracy=0)
        native.init_photoz_histogram_akima(
            enabled=int(factor != 0), preparation_factor=max(factor, 3))
        native.init_redshift_distributions_from_files(
            lens_multihisto_file=str(lens_file), lens_ntomo=lens_count,
            source_multihisto_file=str(source_file), source_ntomo=source_count)
        prefix = sample + "_" + arm
        # Reinitializing for the close-up must recover exactly the curve
        # used for the area audit, not a differently configured reader.
        previous = arrays[prefix + "_z"]
        check = np.fromiter((reader(float(z), tomo) for z in previous),
                            dtype=float, count=previous.size)
        np.testing.assert_array_equal(x=check, y=arrays[prefix + "_density"][tomo])
        arrays[prefix + "_zoom_z"] = query
        arrays[prefix + "_zoom_density"] = np.fromiter(
            (reader(float(z), tomo) for z in query), dtype=float, count=query.size)
