"""Evaluate a survey covariance from a Cobaya-style YAML file.

Cobaya's YAML reader and parameterization resolve the theory/params/evaluate
language already used by the likelihood examples. This driver evaluates one
specified cosmology; it neither runs an MCMC sampler nor draws from a prior.
A project adapter supplies the survey layout and its default accuracy YAML.

The shared survey assembly calls the production C++ interface, which borrows
contiguous NumPy inputs and bypasses the Armadillo notebook wrappers. The
same C kernels serve both entry points. The saved archive contains G, SSC,
cNG, their sum, measurement coordinates and fully resolved settings.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import time

from cobaya.log import logger_setup
from cobaya.parameterization import Parameterization
from cobaya.yaml import yaml_load_file

from .forecast import save_forecast
from .forecast_cluster import save_forecast as save_cluster_forecast


def load_run_configuration(filename, survey, default_space="real", joint=False):
    """Read one explicitly specified cosmology with Cobaya's YAML machinery.

    Arguments:
        filename = input YAML path. Paths inside it are relative to the
            working directory, as in Cocoa's likelihood example commands.
        survey = project adapter exposing configuration().
        default_space = native "real" or "fourier" measurement space.
        joint = True restricts the selected-cluster forecast to real space.
    Returns:
        (settings, run): resolved survey/cosmology settings, with the parsed
        YAML in settings["run_yaml"], and run options containing space,
        threads from OMP_NUM_THREADS, output, timing and optional CAMB path.
    Raises:
        ValueError for unsupported blocks or keys, missing or unphysical
        cosmology values, nonzero mnu, sampling requests, a missing CAMB
        directory, a covariance.threads key, an unset or nonpositive
        OMP_NUM_THREADS (the OpenMP thread count) or a missing output name.
        Fixed params and Cobaya input expressions are accepted. A param with
        a prior needs an explicit evaluate.override; the covariance
        generator never chooses a random reference cosmology.
    Side effects:
        Configures Cobaya's logging from the optional debug key.
    """
    # --- 1. TOP-LEVEL BLOCKS ---

    # Any other top-level key, likelihood included, is rejected: this
    # driver evaluates no likelihood and must not ignore a block silently.
    # set(info)-allowed holds the unlisted keys; a nonempty set is true.
    info = yaml_load_file(file_name=str(filename))
    allowed = {
        "theory",
        "params",
        "sampler",
        "covariance",
        "output",
        "timing",
        "debug",
    }
    if not isinstance(info, dict) or set(info)-allowed:
        raise ValueError("covariance YAML needs theory, params, sampler, "
                         "covariance and output blocks; likelihood is not used")
    logger_setup(debug=info.get("debug", False))

    # --- 2. THE SAMPLER BLOCK: evaluate WITH N: 1 ONLY ---

    # One explicitly selected cosmology produces one covariance. Reusing
    # evaluate's syntax does not ask Cobaya to run a sampler here.
    sampler = info.get("sampler", {})
    if set(sampler) != {"evaluate"}:
        raise ValueError("use sampler: evaluate with N: 1 for one covariance")
    # YAML reads a bare "evaluate:" line as None; "or {}" turns that None
    # into an empty mapping, so the key checks below still apply.
    evaluate = sampler["evaluate"] or {}
    if set(evaluate)-{"N", "override"} or evaluate.get("N", 1) != 1:
        raise ValueError("evaluate accepts N: 1 and an optional override mapping")

    # --- 3. COSMOLOGICAL PARAMETERS ---

    # Cobaya handles fixed numbers, value expressions and parameter names.
    # Require every prior parameter in override, rather than sampling its
    # prior/ref distribution and quietly changing the requested cosmology.
    params = info.get("params", {})
    names = (
        "omegam",
        "omegab",
        "H0",
        "ns",
        "As_1e9",
        "w",
        "w0pwa",
        "mnu",
    )
    if set(params) != set(names):
        raise ValueError("params must specify omegam, omegab, H0, ns, "
                         "As_1e9, w, w0pwa and mnu")

    # sampled_params() lists the parameters that carry a prior. to_input
    # fills the fixed values and evaluates the input expressions at the
    # override values, giving one number per name.
    parameterization = Parameterization(info_params=params)
    overrides = evaluate.get("override") or {}
    if set(overrides) != set(parameterization.sampled_params()):
        raise ValueError("evaluate.override must supply each parameter with "
                         "a prior, and only those parameters")
    values = parameterization.to_input(sampled_params_values=overrides)

    # bool is a subclass of int in Python, so True would pass as 1 without
    # the explicit bool test. Every value is stored as a plain float.
    cosmology = {}
    for name in names:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"params.{name} must resolve to a number")
        if not math.isfinite(value):
            raise ValueError(f"params.{name} must be finite")
        cosmology[name] = float(value)

    # --- 4. PHYSICAL RANGES ---

    # Python chains comparisons: a < b < c < d means a < b and b < c and
    # c < d, so all three inequalities must hold.
    if not 0.0 < cosmology["omegab"] < cosmology["omegam"] < 1.0:
        raise ValueError("need 0 < omegab < omegam < 1")
    if cosmology["H0"] <= 0.0 or cosmology["As_1e9"] <= 0.0:
        raise ValueError("H0 and As_1e9 must be positive")
    # The halo-model trispectrum and response combine cb halo moments with
    # total-matter formulas, which is consistent only for mnu = 0.
    if cosmology["mnu"] != 0.0:
        raise ValueError("the supported non-Gaussian covariance requires mnu: 0")

    # --- 5. THE CAMB THEORY BLOCK ---

    theory = info.get("theory", {})
    if set(theory) != {"camb"} or not isinstance(theory["camb"], dict):
        raise ValueError("theory must contain a camb mapping")
    camb = theory["camb"]
    if set(camb)-{"path", "extra_args"}:
        raise ValueError("theory.camb accepts path and extra_args")

    # AccuracyBoost in this block is CAMB's own boost, matching the
    # likelihood theory block. Covariance table accuracy stays independent.
    # camb_options maps each accepted YAML key to its name in the survey
    # cosmology mapping; only AccuracyBoost is renamed.
    camb_options = {
        "AccuracyBoost": "CAMBAccuracyBoost",
        "kmax": "kmax",
        "k_per_logint": "k_per_logint",
        "lens_potential_accuracy": "lens_potential_accuracy",
        "halofit_version": "halofit_version",
    }
    extra = camb.get("extra_args") or {}
    unknown = set(extra)-set(camb_options)
    if unknown:
        raise ValueError(f"unsupported CAMB options: {sorted(unknown)}; "
                         f"supported options: {list(camb_options)}")
    for name, value in extra.items():
        cosmology[camb_options[name]] = value

    # Only checked here; run_covariance puts the directory on sys.path.
    camb_path = camb.get("path")
    if camb_path is not None and not Path(camb_path).is_dir():
        raise ValueError(f"CAMB directory does not exist: {camb_path}")

    # --- 6. COVARIANCE CONTROLS AND MEASUREMENT SPACE ---

    # dict(...) copies the YAML block, so the pop below leaves the run
    # record in settings["run_yaml"] intact. After the pop, every remaining
    # key is an accuracy control for survey.configuration.
    controls = dict(info.get("covariance") or {})
    if "threads" in controls:
        raise ValueError("remove covariance.threads; set OMP_NUM_THREADS "
                         "in the environment instead")
    space = controls.pop("space", default_space)
    spaces = ["real"] if joint else ["real", "fourier"]
    if space not in spaces:
        raise ValueError(f"covariance.space must be one of {spaces}")

    # --- 7. OPENMP THREADS FROM THE ENVIRONMENT ---

    # The shell or HPC job defines the available OpenMP team. Keep that
    # resource choice out of the scientific YAML and record it in outputs.
    # An unset variable reads as "" and fails int() like any non-integer;
    # "from None" drops the chained int() traceback from the error report.
    try:
        threads = int(os.environ.get("OMP_NUM_THREADS", ""))
    except ValueError:
        raise ValueError("set OMP_NUM_THREADS to a positive integer "
                         "before running the covariance command") from None
    if threads < 1:
        raise ValueError("OMP_NUM_THREADS must be a positive integer")

    # --- 8. RESOLVED SETTINGS AND RUN OPTIONS ---

    # Unknown accuracy keys fail in the shared resolver. Fine controls
    # refine the project's baseline, under the same global boost as Jupyter.
    # **controls passes each remaining YAML key as a keyword argument.
    settings = survey.configuration(**controls)
    settings["cosmology"].update(cosmology)
    settings["run_yaml"] = info

    # run_covariance may replace this path with --output, then checks the
    # .npz suffix, an existing file and the parent directory.
    output = info.get("output")
    if not isinstance(output, str) or not output:
        raise ValueError("output must name the .npz archive to write")
    run = {
        "space": space,
        "threads": threads,
        "output": Path(output),
        "timing": info.get("timing", True),
        "camb_path": camb_path,
    }
    return settings, run


def run_covariance(interface, survey, default_space="real", joint=False,
                   argv=None):
    """Evaluate and save one YAML-configured covariance through production C.

    Arguments:
        interface = compiled project module; covariance must be enabled.
        survey = adapter exposing configuration, initialize and compute.
        default_space = native "real" or "fourier" measurement space.
        joint = True for the angular cluster 6x2pt+N adapter.
        argv = explicit command arguments; None reads the command line.
    Returns:
        Path of the written .npz archive. --help prints usage and exits.
    Side effects:
        Resolves the YAML, initializes CAMB and the project, applies
        OMP_NUM_THREADS and saves G/SSC/cNG/total. Prepends theory.camb.path
        to sys.path when given and prints the saved size, plus timings
        unless the YAML sets timing to false. No plots or eigenproblems run.
        A disabled build or invalid output options exit through argparse
        (status 2), and a rejected YAML setting raises ValueError, all before
        numerical setup.
    """
    # --- 1. COMMAND-LINE ARGUMENTS ---

    # parse_args(args=None) reads sys.argv; an explicit argv list replaces it.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="covariance evaluate YAML")
    parser.add_argument("--output", type=Path, help="override YAML output path")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output archive")
    args = parser.parse_args(args=argv)

    # --- 2. BUILD, YAML AND OUTPUT CHECKS BEFORE ANY NUMERICAL SETUP ---

    # getattr returns False when the compiled module lacks the
    # has_covariance flag. parser.error prints the usage and message, then
    # exits with status 2 without a Python traceback; a rejected YAML
    # setting raises ValueError inside load_run_configuration instead.
    if not getattr(interface, "has_covariance", False):
        parser.error("enable covariance generation and recompile; "
                     "see this project's README")
    settings, run = load_run_configuration(
        filename=args.input, survey=survey, default_space=default_space,
        joint=joint,
    )

    # --output replaces the YAML output name; the checks apply to the path
    # actually written.
    if args.output is not None:
        run["output"] = args.output
    output = run["output"]
    if output.suffix != ".npz":
        parser.error("output must name a .npz archive")
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; use a new path or --overwrite")
    if not output.parent.is_dir():
        parser.error(f"output directory does not exist: {output.parent}")

    # Position 0 makes this CAMB directory take precedence over any other
    # camb package already on the import path.
    if run["camb_path"] is not None:
        sys.path.insert(0, str(Path(run["camb_path"]).resolve()))

    # --- 3. RUN RECORD AND THE OPENMP TEAM ---

    # Keep the run record consistent with command-line overrides. The saved
    # cosmology and accuracy mapping already contain every resolved value.
    settings["execution"] = {
        "backend": "production",
        "threads": run["threads"],
        "input": str(args.input),
        "output": str(output),
    }
    interface.set_omp_threads(n=run["threads"])

    # --- 4. INITIALIZATION: CAMB AND THE PROJECT ---

    started = time.perf_counter()
    survey.initialize(interface=interface, settings=settings)
    setup_seconds = time.perf_counter()-started
    # A library loaded during initialization can reset the process-wide
    # OpenMP thread count; apply the requested team again before assembly.
    interface.set_omp_threads(n=run["threads"])
    if run["timing"]:
        print(f"Initialization including CAMB: {setup_seconds:.2f} s", flush=True)

    # --- 5. COVARIANCE ASSEMBLY AND ARCHIVE ---

    # progress is a nested function (a closure): it reads run["timing"]
    # from this call, so survey.compute can report stages without seeing
    # the YAML.
    def progress(stage, elapsed_seconds):
        """Report elapsed matrix-construction time, excluding initial setup.

        Arguments:
            stage = label printed before the time.
            elapsed_seconds = seconds since matrix construction began.
        """
        if run["timing"]:
            print(f"{stage}: {elapsed_seconds:.2f} s", flush=True)

    # Only the numerical backend differs from a notebook invocation. Model
    # choices and survey assembly remain in the shared Python routines.
    if joint:
        result = survey.compute(
            interface=interface, settings=settings, progress=progress,
            backend=interface.covariance,
        )
        save_cluster_forecast(result=result, filename=output)
    else:
        result = survey.compute(
            interface=interface, settings=settings, space=run["space"],
            progress=progress, backend=interface.covariance,
        )
        save_forecast(result=result, filename=output)

    # --- 6. SUMMARY ---

    size = result["total"].shape[0]
    print(f"Saved {size} x {size} G, SSC, cNG and total to {output}")
    if run["timing"]:
        seconds = result["stages_s"]["total"]
        print(f"Covariance construction: {seconds:.2f} s "
              f"({run['threads']} threads)")
    return output
