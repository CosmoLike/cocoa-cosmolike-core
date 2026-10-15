"""Run an opt-in study of n(z) interpretation and numerical refinement.

The physical inputs come from the project's stored NLA example2 point.
Only accuracyboost and integration_accuracy are taken from today's likelihood
YAML for the default arm. The refined arms compare the original midpoint
curve with fixed 3/5/7-point histogram preparation followed by Akima.

This is a model sensitivity study, not a frozen-output regression. Structural
checks require finite vectors, an unchanged covariance, and a reproducible
switch back to the original curve; no expected Delta chi2 is asserted.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from cobaya.yaml import yaml_load_file

from cocoa_nz_testing import _snapshot


def calculate(harness, native, project, output, part, boost=5.0, level=7):
    """Save one default or refined calculation without overwriting a result.

    Arguments:
        harness, native: Project test utilities and compiled interface.
        project: Survey name; DES Y6 has no physical covariance for this study.
        output: Destination directory, outside the stored test inputs.
        part: 'defaults' evaluates original only; 'refined' evaluates four curves.
        boost, level: Refined interpolation and integration settings.

    Returns:
        The measurement dictionary. The adjacent NPZ holds model vectors;
        inverse_covariance.npz stores the common masked inverse covariance once.
    """
    if project == "des_y6":
        raise ValueError("DES Y6 has no physical covariance for a survey Delta chi2")
    if part not in ("defaults", "refined") or level not in range(8) or boost < 1:
        raise ValueError("Use defaults/refined, boost >= 1 and level between 0 and 7")
    output = Path(output).resolve()
    stem = output / part
    if stem.with_suffix(".json").exists() or stem.with_suffix(".npz").exists():
        raise FileExistsError(f"Preserve existing results: choose a new output for {stem}")

    harness.require_cocoa_environment()
    harness.verify_frozen()
    name = harness.EXAMPLES["example2"]["likelihood"]
    project_root = Path(harness.__file__).resolve().parents[1]
    defaults_path = project_root / "likelihood" / (name.split(".")[-1] + ".yaml")
    defaults = yaml_load_file(file_name=str(defaults_path))
    if part == "defaults":
        boost = float(defaults["accuracyboost"])
        level = int(defaults["integration_accuracy"])

    info = harness.load_frozen_info(example="example2", tatt=False)
    options = info["likelihood"][name]
    options.update(
        photoz_histogram_akima=0,
        accuracyboost=boost,
        integration_accuracy=level,
        print_datavector=False,
    )
    arms = (("original", 0),)
    if part == "refined":
        arms = (("original", 0), ("three", 3), ("five", 5), ("seven", 7))

    started = time.monotonic()
    model = harness.make_model(info=info)
    vectors = {}
    try:
        point = harness.build_point(model=model, example="example2", tatt=False)
        likelihood = model.likelihood[name]
        inputs = {}
        for key in ("source_file", "lens_file", "cov_file", "mask_file", "data_vector_file"):
            filename = getattr(likelihood, key, None)
            if filename is not None:
                path = Path(filename).resolve()
                inputs[key] = {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }

        inverse = None
        for label, factor in arms:
            native.init_photoz_histogram_akima(
                enabled=int(factor != 0), preparation_factor=max(factor, 3))
            vector, current_inverse, _ = _snapshot(
                harness=harness, native=native, model=model, point=point,
                likelihood_name=name)
            if inverse is None:
                inverse = current_inverse
            else:
                np.testing.assert_array_equal(x=current_inverse, y=inverse)
            vectors[label] = vector
            print(f"{project}: {part}, {label} complete", flush=True)

        if part == "refined":
            # A repeated physical model must recover the same vector after
            # changing reconstruction; otherwise the comparison mixes cache states.
            native.init_photoz_histogram_akima(enabled=0)
            returned, current_inverse, _ = _snapshot(
                harness=harness, native=native, model=model, point=point,
                likelihood_name=name)
            np.testing.assert_array_equal(x=returned, y=vectors["original"])
            np.testing.assert_array_equal(x=current_inverse, y=inverse)
    finally:
        model.close()

    selected = np.diag(inverse) > 0
    np.linalg.cholesky(inverse[np.ix_(selected, selected)])
    distances = {}
    if part == "refined":
        for left, right in (("original", "three"), ("three", "five"),
                            ("three", "seven"), ("five", "seven")):
            delta = vectors[right] - vectors[left]
            distances[left + "_to_" + right] = float(delta @ inverse @ delta)

    record = {
        "project": project,
        "part": part,
        "point": point,
        "input_files": inputs,
        "accuracyboost": boost,
        "integration_accuracy": level,
        "defaults_yaml_sha256": hashlib.sha256(defaults_path.read_bytes()).hexdigest(),
        "interface_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "study_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "delta_chi2": distances,
        "elapsed_seconds": time.monotonic() - started,
        "thread_environment": {},
    }
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        record["thread_environment"][key] = os.environ.get(key)
    output.mkdir(parents=True, exist_ok=True)
    covariance_path = output / "inverse_covariance.npz"
    if covariance_path.exists():
        with np.load(file=covariance_path) as saved:
            np.testing.assert_array_equal(x=inverse, y=saved["inverse_covariance"])
    else:
        np.savez_compressed(file=covariance_path, inverse_covariance=inverse)
    record["inverse_covariance_sha256"] = hashlib.sha256(
        covariance_path.read_bytes()).hexdigest()
    np.savez_compressed(file=stem.with_suffix(".npz"), **vectors)
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def compare_saved(output):
    """Measure original default versus original refined with identical inputs.

    Both files must be produced with the same binary and stored physical point.
    The distance uses the supplied survey covariance, not a difference in fits.
    """
    output = Path(output).resolve()
    default = json.loads((output / "defaults.json").read_text())
    refined = json.loads((output / "refined.json").read_text())
    for key in ("project", "point", "input_files", "interface_sha256",
                "inverse_covariance_sha256"):
        if default[key] != refined[key]:
            raise ValueError(f"Default and refined records disagree in {key}")
    covariance_path = output / "inverse_covariance.npz"
    if hashlib.sha256(covariance_path.read_bytes()).hexdigest() != default["inverse_covariance_sha256"]:
        raise ValueError("The saved inverse covariance fingerprint changed")
    with np.load(file=output / "defaults.npz") as first:
        with np.load(file=output / "refined.npz") as second:
            with np.load(file=output / "inverse_covariance.npz") as covariance:
                inverse = covariance["inverse_covariance"]
            delta = second["original"] - first["original"]
            distance = float(delta @ inverse @ delta)
    report = {
        "project": default["project"],
        "default_accuracyboost": default["accuracyboost"],
        "default_integration_accuracy": default["integration_accuracy"],
        "refined_accuracyboost": refined["accuracyboost"],
        "refined_integration_accuracy": refined["integration_accuracy"],
        "original_default_to_refined": distance,
        "reconstruction_distances": refined["delta_chi2"],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def audit_and_plot(harness, native, project, output):
    """Integrate all four compiled reconstructions over the original input cells.

    This uses unshifted histograms, not the nuisance-shifted fiducial vectors.
    Every piecewise-cubic segment is integrated independently, so narrow edge
    transitions cannot be missed by a quadrature rule spanning many cells.
    """
    from cocoa_nz_testing import histogram_probability_audit
    from cosmolike_notebook_utils.plot_nz_reconstruction import plot_nz_four_reconstructions

    output = Path(output).resolve()
    record = json.loads((output / "refined.json").read_text())
    frozen_data = Path(harness.__file__).resolve().parent / "frozen/data"
    paths = {}
    for sample in ("source", "lens"):
        metadata = record["input_files"][sample + "_file"]
        path = frozen_data / metadata["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"The histogram changed: {path}")
        paths[sample] = path
    areas, arrays = histogram_probability_audit(
        native=native, source_file=paths["source"], lens_file=paths["lens"],
        preparation_factors=(3, 5, 7), accuracyboost=record["accuracyboost"])
    report = {
        "project": project,
        "project_label": project.replace("_", " ").upper(),
        "input_files": record["input_files"],
        "probability_audit": areas,
    }
    report_path = output / "probability.json"
    arrays_path = output / "probability.npz"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(file=arrays_path, **arrays)
    plot_nz_four_reconstructions(report_file=report_path, arrays_file=arrays_path,
                                output_file=output / "reconstructions.png")


def main(harness, native, project):
    """Parse the explicitly requested study; ordinary pytest never calls this."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", choices=("all", "defaults", "refined", "compare", "plot"),
                        default="all")
    parser.add_argument("--boost", type=float, default=5.0)
    parser.add_argument("--integration-accuracy", type=int, default=7)
    args = parser.parse_args()
    parts = ("defaults", "refined") if args.part == "all" else (args.part,)
    for part in parts:
        if part in ("defaults", "refined"):
            calculate(harness=harness, native=native, project=project,
                      output=args.output, part=part, boost=args.boost,
                      level=args.integration_accuracy)
    if args.part in ("all", "compare"):
        compare_saved(output=args.output)
    if args.part in ("all", "plot"):
        audit_and_plot(harness=harness, native=native, project=project, output=args.output)
