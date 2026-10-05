"""Notebook preparation for galaxy and shear covariance forecasts.

A project supplies its redshift files, cosmology, catalog densities and
measurement bins. This module installs that forecast through the ordinary
project setters and binds those inputs to the shared G/SSC/cNG calculation.
It imports no project interface and reads no supplied likelihood covariance.
The forecast uses massless neutrinos, linear bias, zero magnification,
photo-z shifts and shear calibration. Gaussian IA/non-Limber choices are
explicit; SSC/cNG keep their original zero-IA Limber model. CMB and cluster
fields need separate models; increasing the galaxy-bin count is insufficient.
"""

import json
from pathlib import Path

import numpy as np

from ..camb_cosmology import get_camb_cosmology
from .geometry import noise_powers
from .survey import observable_rows, realspace_covariance, fourier_covariance


def gaussian_model(gaussian, nsource):
    """Resolve Gaussian-only spectra choices before changing core state.

    gaussian = optional mapping with nonlimber, ia (none/NLA/TATT), A1,
        A2 and B_TA. Amplitudes are constants per source bin: a scalar
        applies to all bins; an array needs one value per bin. The core
        supplies the standard growth dependence. No redshift power law is
        inferred from a two-element list. SSC/cNG retain zero IA and Limber.
    nsource = number of source bins.
    Returns a new mapping with explicit per-bin amplitude lists.
    """
    choices = {} if gaussian is None else dict(gaussian)
    allowed = {"nonlimber", "ia", "A1", "A2", "B_TA"}
    unknown = set(choices)-allowed
    if unknown:
        raise ValueError(f"unknown Gaussian model choices: {sorted(unknown)}")
    nonlimber = choices.get("nonlimber", True)
    model = choices.get("ia", "none")
    if not isinstance(nonlimber, bool):
        raise ValueError("gaussian.nonlimber must be true or false")
    if model not in ("none", "NLA", "TATT"):
        raise ValueError("gaussian.ia must be none, NLA or TATT")
    result = {"nonlimber": nonlimber, "ia": model}
    for name in ("A1", "A2", "B_TA"):
        values = np.asarray(choices.get(name, 0.0), dtype=float)
        if values.ndim == 0:
            values = np.full(nsource, float(values))
        if values.shape != (nsource,) or not np.all(np.isfinite(values)):
            raise ValueError(f"gaussian.{name} needs a finite scalar or {nsource} values")
        if model == "none" and np.any(values != 0):
            raise ValueError("nonzero IA amplitudes require gaussian.ia=NLA or TATT")
        if model == "NLA" and name != "A1" and np.any(values != 0):
            raise ValueError("NLA permits A1 only; A2 and B_TA require TATT")
        result[name] = values.tolist()
    return result


def set_gaussian_model(interface, settings):
    """Install explicitly resolved per-bin IA for the Gaussian calculation."""
    nsource = len(settings["source_density_arcmin2"])
    # Legacy low-level settings retain their original zero-IA state.
    requested = settings.get("gaussian", {"nonlimber": False, "ia": "none"})
    model = gaussian_model(gaussian=requested, nsource=nsource)
    interface.init_IA(
        ia_model=1 if model["ia"] == "TATT" else 0,
        ia_redshift_evolution=2, ia_code=0,
    )
    interface.set_nuisance_ia(A1=model["A1"], A2=model["A2"], B_TA=model["B_TA"])


def initialize_forecast(interface, settings, project):
    """Install the specified galaxy/shear forecast and return its CAMB tables.

    Arguments:
        interface = caller's compiled project interface, with covariance
            generation enabled at build time.
        settings = resolved project mapping. cosmology supplies CAMB inputs;
            lens_file/source_file are paths relative to project;
            lens_density_arcmin2/source_density_arcmin2 specify bin counts;
            bias contains one linear bias per lens bin; photoz_interpolation
            and photoz_zmid select the project's redshift-file convention.
            lens_photoz_stretch, when present, supplies the per-bin width
            factors required by the DESxPlanck and DES cluster setters.
        project = project directory, a Path or string.
    Returns:
        Dict of CAMB arrays in the set_cosmology interchange format.
    Side effects:
        Replaces the interface's cosmology and galaxy/source nuisance state.
        No likelihood data, mask or covariance is read or overwritten.
    """
    if not getattr(interface, "has_covariance", False):
        raise RuntimeError(
            "Covariance generation is not enabled in this interface. "
            "Follow the project's README covariance build steps, then "
            "restart the notebook kernel."
        )
    project = Path(project)
    lens_file = project/settings["lens_file"]
    source_file = project/settings["source_file"]
    if not lens_file.is_file() or not source_file.is_file():
        raise FileNotFoundError(
            f"forecast needs redshift files {lens_file} and {source_file}"
        )
    nlens = len(settings["lens_density_arcmin2"])
    nsource = len(settings["source_density_arcmin2"])
    if nlens < 1 or nsource < 1 or len(settings["bias"]) != nlens:
        raise ValueError("forecast needs lens/source bins and one bias per lens")
    cosmology = settings["cosmology"]
    if cosmology["mnu"] != 0.0:
        raise ValueError("the full halo forecast currently requires mnu=0")

    # These catalog powers are also validated before changing C state.
    # n(z) fixes each bin's radial shape, not its number of observed objects.
    noise_powers(
        lens_density=settings["lens_density_arcmin2"],
        source_density=settings["source_density_arcmin2"],
        sigma_component=settings["sigma_e_component"],
    )

    # The ordered tuple is the documented CAMB/set_cosmology interchange.
    # Keep the tables so a notebook can archive the inputs of its forecast.
    arrays = get_camb_cosmology(**cosmology)
    names = (
        "log10k_2D",
        "z_2D",
        "lnP_linear",
        "lnP_nonlinear",
        "G",
        "z_G",
        "z_1D",
        "chi",
        "omegan2",
        "lnP_linear_cb",
    )
    tables = dict(zip(names, arrays))

    interface.initial_setup()
    interface.init_accuracy_boost(
        accuracy_boost=settings["core_accuracyboost"],
        integration_accuracy=settings["integration_accuracy"],
    )
    interface.init_probes(possible_probes="3x2pt")
    interface.init_IA(ia_model=0, ia_redshift_evolution=2, ia_code=0)
    interface.init_bias(bias_model=[0, 0, 0, 0, 0])
    interface.init_photoz_conventions(
        interpolation_type=settings["photoz_interpolation"],
        zmid_convention=settings["photoz_zmid"],
    )
    interface.init_cosmo_runmode(is_linear=False)
    interface.init_redshift_distributions_from_files(
        lens_multihisto_file=str(lens_file), lens_ntomo=nlens,
        source_multihisto_file=str(source_file), source_ntomo=nsource,
    )
    interface.set_cosmology(
        omegam=cosmology["omegam"], omegab=cosmology["omegab"],
        H0=cosmology["H0"], **tables,
    )
    lens_zero = [0.0]*nlens
    source_zero = [0.0]*nsource
    interface.set_nuisance_bias(
        B1=settings["bias"], B2=lens_zero, B_MAG=lens_zero,
        B3nl=lens_zero, BK=lens_zero,
    )
    set_gaussian_model(interface=interface, settings=settings)
    interface.set_nuisance_shear_photoz(bias=source_zero)
    if "lens_photoz_stretch" in settings:
        interface.set_nuisance_clustering_photoz(
            bias=lens_zero, stretch=settings["lens_photoz_stretch"],
        )
    else:
        interface.set_nuisance_clustering_photoz(bias=lens_zero)
    interface.set_nuisance_shear_calib(M=source_zero)
    return tables


def compute_forecast(interface, settings, space="real", rows=None,
                     progress=None, backend=None):
    """Compute all 3x2pt rows, or a supplied subset, with separate components.

    Arguments:
        interface = already initialized project interface.
        settings = resolved project mapping with covariance_accuracy values,
            cosmology, densities, area_deg2, theta_edges_arcmin, a_edges,
            lnm_edges, excluded_gammat, band_first and band_last.
        space = "real" for angular bins, "fourier" for E-mode bandpowers.
        backend = None uses notebook wrappers; interface.covariance uses
            the direct production bindings to the same C calculations.
        rows = optional int32 [nobservable,3] (type,A,B) table. None selects
            the project's full galaxy/shear forecast layout. A subset only
            reduces measured rows; all internal crossed spectra remain.
        progress = optional callable taking (stage, elapsed_seconds).
    Returns:
        Dict with G/SSC/cNG/total, mean signals and diagnostic arrays from
        the survey assembler, plus coordinate (arcminutes or multipoles),
        coordinate_label and the fully resolved integration settings.
        No files are written and no eigenvalues are repaired.
    """
    if space not in ("real", "fourier"):
        raise ValueError("space must be 'real' or 'fourier'")
    # A notebook may refine settings while retaining its CAMB inputs.
    # Apply the reader-table boost on every calculation; the independent
    # quadrature level remains exactly the value selected in the YAML.
    interface.init_accuracy_boost(
        accuracy_boost=settings["core_accuracyboost"],
        integration_accuracy=settings["integration_accuracy"],
    )
    set_gaussian_model(interface=interface, settings=settings)
    if backend is not None:
        interface = backend
    nlens = len(settings["lens_density_arcmin2"])
    nsource = len(settings["source_density_arcmin2"])
    if rows is None:
        rows = observable_rows(
            nlens=nlens, nsource=nsource,
            excluded_gammat=settings["excluded_gammat"],
        )
        if space == "fourier":
            rows = rows[rows[:, 0] != 1]
    rows = np.asarray(rows)
    if rows.dtype.kind not in "iu":
        raise ValueError("observable rows must contain integer probe and field IDs")
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    noise = noise_powers(
        lens_density=settings["lens_density_arcmin2"],
        source_density=settings["source_density_arcmin2"],
        sigma_component=settings["sigma_e_component"],
    )
    resolved = dict(settings)
    resolved["space"] = space
    resolved["mnu"] = settings["cosmology"]["mnu"]
    resolved["area_sr"] = settings["area_deg2"]*(np.pi/180.0)**2
    resolved["edges_rad"] = settings["theta_edges_arcmin"]*np.pi/(180.0*60.0)

    if space == "real":
        result = realspace_covariance(
            interface=interface, settings=resolved, rows=rows, noise=noise,
            progress=progress,
        )
        edges = settings["theta_edges_arcmin"]
        result["coordinate"] = np.sqrt(edges[:-1]*edges[1:])
        result["coordinate_label"] = r"$\theta\;[\mathrm{arcmin}]$"
    else:
        result = fourier_covariance(
            interface=interface, settings=resolved, rows=rows, noise=noise,
            progress=progress,
        )
        first = settings["band_first"]
        last = settings["band_last"]
        result["coordinate"] = np.sqrt(first*last)
        result["coordinate_label"] = r"$\ell$"
    result["settings"] = resolved
    return result


def _json_array(value):
    """Represent numerical settings in JSON without losing their resolved values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot save setting of type {type(value).__name__}")


def save_forecast(result, filename):
    """Write a computed forecast and its resolved settings to a NumPy archive.

    Arguments:
        result = dict returned by compute_forecast.
        filename = output .npz path, outside the likelihood's data directory.
    Returns:
        Nothing. Replaces the named output if it already exists.
        Arrays load with numpy.load(..., allow_pickle=False). settings_json
        and stages_json are text scalars read with json.loads(str(...)).
        Archive CAMB input tables separately when exact input reuse is needed.
    """
    settings = json.dumps(result["settings"], default=_json_array, allow_nan=False)
    # Before Gaussian-only model choices, both means were the same. Keep
    # older notebook results saveable while recording the separate SSC mean
    # whenever the Gaussian spectrum now contains non-Limber or IA terms.
    ssc_signal = result.get("ssc_normalization_signal", result["signal"])
    np.savez(
        file=filename,
        gaussian=result["gaussian"],
        ssc=result["ssc"],
        cng=result["cng"],
        total=result["total"],
        signal=result["signal"],
        ssc_normalization_signal=ssc_signal,
        rows=result["rows"],
        coordinate=result["coordinate"],
        coordinate_label=result["coordinate_label"],
        geometry=result["geometry"],
        coarse_ell=result["coarse_ell"],
        pair_area_sr2=result["pair_area_sr2"],
        settings_json=settings,
        stages_json=json.dumps(result["stages_s"], allow_nan=False),
    )
