"""Notebook preparation for galaxy and shear covariance forecasts.

A project supplies its redshift files, cosmology, catalog densities and
measurement bins. This module installs that forecast through the ordinary
project setters and binds those inputs to the shared Gaussian (G),
super-sample (SSC) and connected non-Gaussian (cNG) calculation.
It imports no project interface and reads no supplied likelihood covariance.
The forecast uses massless neutrinos and linear bias, with zero
magnification, photo-z shifts and shear calibration. Gaussian IA and
non-Limber choices are explicit; SSC/cNG always use the zero-IA Limber
model. CMB and cluster fields need separate models; increasing the
galaxy-bin count is insufficient.
"""

import json
from pathlib import Path

import numpy as np

from ..camb_cosmology import get_camb_cosmology
from .geometry import noise_powers
from .power import refine_power_tables
from .survey import observable_rows, realspace_covariance, fourier_covariance


def gaussian_model(gaussian, nsource):
    """Resolve Gaussian-only spectra choices before changing core state.

    Arguments:
        gaussian = optional mapping with nonlimber (bool, default True), ia
            (none/NLA/TATT, default none), A1, A2 and B_TA (default 0).
            Amplitudes are constants per source bin: a scalar applies to
            all bins; an array needs one value per bin. The core supplies
            the standard growth dependence. No redshift power law is
            inferred from a two-element list. SSC/cNG retain zero IA and
            Limber.
        nsource = number of source bins.
    Returns:
        A new mapping with nonlimber, ia and explicit per-bin amplitude
        lists A1, A2 and B_TA. The supplied mapping is not modified.
    Raises:
        ValueError for an unknown key, a non-boolean nonlimber, another IA
        model, nonfinite amplitudes or a wrong amplitude count, nonzero
        amplitudes with ia=none, or nonzero A2/B_TA with NLA.
    """
    # --- 1. THE SUPPLIED MAPPING: ONLY THE FIVE KNOWN KEYS ---

    # dict(gaussian) copies the caller's mapping, so nothing below writes
    # into it; gaussian=None selects every default. set(choices)-allowed is
    # the set of supplied keys outside the five allowed ones, so a misspelled
    # key such as nonLimber or a1 raises instead of being ignored.
    choices = {} if gaussian is None else dict(gaussian)
    allowed = {"nonlimber", "ia", "A1", "A2", "B_TA"}
    unknown = set(choices)-allowed
    if unknown:
        raise ValueError(f"unknown Gaussian model choices: {sorted(unknown)}")

    # --- 2. THE TWO SWITCHES: NON-LIMBER SPECTRA AND THE IA MODEL ---

    # choices.get(key, default) returns the default when the key is absent:
    # non-Limber spectra on, no IA. isinstance(nonlimber, bool) accepts only
    # True and False, so an integer such as nonlimber: 1 in a YAML file raises.
    nonlimber = choices.get("nonlimber", True)
    model = choices.get("ia", "none")
    if not isinstance(nonlimber, bool):
        raise ValueError("gaussian.nonlimber must be true or false")
    if model not in ("none", "NLA", "TATT"):
        raise ValueError("gaussian.ia must be none, NLA or TATT")

    # --- 3. IA AMPLITUDES: ONE FINITE CONSTANT PER SOURCE BIN ---

    # The loop adds A1, A2 and B_TA as plain lists of nsource floats, zeros
    # included, so set_gaussian_model always passes three explicit lists.
    result = {"nonlimber": nonlimber, "ia": model}
    for name in ("A1", "A2", "B_TA"):
        # A scalar arrives as a 0-d array (ndim 0), and np.full copies it into
        # every source bin. An array must already hold one value per bin; a
        # two-element list is never read as a redshift power law.
        values = np.asarray(choices.get(name, 0.0), dtype=float)
        if values.ndim == 0:
            values = np.full(nsource, float(values))
        if values.shape != (nsource,) or not np.all(np.isfinite(values)):
            raise ValueError(f"gaussian.{name} needs a finite scalar or {nsource} values")

        # Each model reads only some amplitudes. ia=none is installed as NLA
        # with zero amplitudes (set_gaussian_model), so a nonzero value would
        # switch IA on. NLA has no A2 or B_TA term, so a nonzero value there
        # would be dropped without notice.
        if model == "none" and np.any(values != 0):
            raise ValueError("nonzero IA amplitudes require gaussian.ia=NLA or TATT")
        if model == "NLA" and name != "A1" and np.any(values != 0):
            raise ValueError("NLA permits A1 only; A2 and B_TA require TATT")
        result[name] = values.tolist()
    return result


def set_gaussian_model(interface, settings):
    """Install explicitly resolved per-bin IA for the Gaussian calculation.

    Arguments:
        interface = compiled project interface.
        settings = mapping with source_density_arcmin2 (one entry per
            source bin) and an optional gaussian mapping (see
            gaussian_model).
    Side effects:
        Replaces the interface's IA model and amplitudes: TATT when
        requested, otherwise NLA, with constant per-bin amplitudes
        (ia_redshift_evolution=2); ia=none installs zero amplitudes.
    """
    nsource = len(settings["source_density_arcmin2"])
    # Settings without a gaussian entry select zero IA and Limber spectra.
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
            lens_density_arcmin2/source_density_arcmin2 give one density
            per bin, in objects per arcmin^2; sigma_e_component gives one
            per-component shape dispersion per source bin;
            bias contains one linear bias per lens bin; photoz_interpolation
            and photoz_zmid select the project's redshift-file convention.
            power_refinement, core_accuracyboost and integration_accuracy
            set the power tables and core accuracy; gaussian is the
            optional Gaussian IA model (see gaussian_model).
            lens_photoz_stretch, when present, supplies the per-bin width
            factors required by the DESxPlanck and DES cluster setters.
        project = project directory, a Path or string.
    Returns:
        Installed power and background arrays in the set_cosmology format.
        Power tables include the resolved power_refinement. Reinitialize
        after changing that refinement, including a global accuracy boost.
    Side effects:
        Replaces the interface's cosmology and galaxy/source nuisance state.
        No likelihood data, mask or covariance is read or overwritten.
    Raises:
        RuntimeError when the build lacks covariance support;
        FileNotFoundError for a missing redshift file; ValueError for
        missing bins, a bias count that differs from the lens count,
        nonzero mnu, or invalid densities or dispersions. These checks run
        before the interface state changes.
    """
    # --- 1. INPUT CHECKS: THE INTERFACE IS NOT TOUCHED UNTIL ALL PASS ---

    # getattr with a default reads False when the build does not define the
    # flag at all. Without covariance support the C entry points used by
    # compute_forecast are missing, so fail here and name the remedy.
    if not getattr(interface, "has_covariance", False):
        raise RuntimeError(
            "Covariance generation is not enabled in this interface. "
            "Follow the project's README covariance build steps, then "
            "restart the notebook kernel."
        )

    # Redshift file names are relative to the project directory; the Path
    # "/" operator joins the two pieces into one path.
    project = Path(project)
    lens_file = project/settings["lens_file"]
    source_file = project/settings["source_file"]
    if not lens_file.is_file() or not source_file.is_file():
        raise FileNotFoundError(
            f"forecast needs redshift files {lens_file} and {source_file}"
        )

    # One density per tomographic bin fixes the bin counts, and the linear
    # bias list must match the lens count. The halo-model terms assume
    # massless neutrinos.
    nlens = len(settings["lens_density_arcmin2"])
    nsource = len(settings["source_density_arcmin2"])
    if nlens < 1 or nsource < 1 or len(settings["bias"]) != nlens:
        raise ValueError("forecast needs lens/source bins and one bias per lens")
    cosmology = settings["cosmology"]
    if cosmology["mnu"] != 0.0:
        raise ValueError("the full halo forecast currently requires mnu=0")

    # Validate the catalog densities and dispersions before changing C
    # state; compute_forecast recomputes these powers, so the result is
    # discarded here. n(z) fixes each bin's radial shape, not its number
    # of observed objects.
    noise_powers(
        lens_density=settings["lens_density_arcmin2"],
        source_density=settings["source_density_arcmin2"],
        sigma_component=settings["sigma_e_component"],
    )

    # --- 2. CAMB TABLES, REFINED IN log k FOR THE CORE READERS ---

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
    # zip pairs each name with the array in the same position, and dict
    # turns those pairs into a mapping. A cubic spline in log10(k) splits
    # each CAMB k interval into power_refinement intervals; z nodes stay.
    tables = refine_power_tables(
        tables=dict(zip(names, arrays)),
        refinement=settings["power_refinement"],
    )

    # --- 3. INSTALL THE FORECAST IN THE INTERFACE ---

    # initial_setup resets every core struct, so it precedes all other init_
    # and set_ calls. IA starts as NLA with zero amplitudes and is replaced
    # by set_gaussian_model below. Bias model code 0 in each of the five
    # slots (b1, b2, bs2, b3, bmag) keeps one constant amplitude per lens bin.
    interface.initial_setup()
    interface.init_accuracy_boost(
        accuracy_boost=settings["core_accuracyboost"],
        integration_accuracy=settings["integration_accuracy"],
    )
    interface.init_probes(possible_probes="3x2pt")
    interface.init_IA(ia_model=0, ia_redshift_evolution=2, ia_code=0)
    interface.init_bias(bias_model=[0, 0, 0, 0, 0])

    # The project's n(z) file convention: the interpolant of the histogram,
    # and whether the file's z column holds left bin edges or sample points.
    # is_linear=False selects the nonlinear matter power.
    interface.init_photoz_conventions(
        interpolation_type=settings["photoz_interpolation"],
        zmid_convention=settings["photoz_zmid"],
    )
    interface.init_cosmo_runmode(is_linear=False)
    interface.init_redshift_distributions_from_files(
        lens_multihisto_file=str(lens_file), lens_ntomo=nlens,
        source_multihisto_file=str(source_file), source_ntomo=nsource,
    )

    # **tables passes every refined table as a keyword argument named by
    # its key (log10k_2D=..., z_2D=..., and so on).
    interface.set_cosmology(
        omegam=cosmology["omegam"], omegab=cosmology["omegab"],
        H0=cosmology["H0"], **tables,
    )

    # Nuisance state of the forecast: linear bias B1 per lens bin, and zero
    # for every other bias term, photo-z shift and shear calibration.
    # [0.0]*n is a list of n zeros.
    lens_zero = [0.0]*nlens
    source_zero = [0.0]*nsource
    interface.set_nuisance_bias(
        B1=settings["bias"], B2=lens_zero, B_MAG=lens_zero,
        B3nl=lens_zero, BK=lens_zero,
    )
    set_gaussian_model(interface=interface, settings=settings)
    interface.set_nuisance_shear_photoz(bias=source_zero)

    # The DESxPlanck and DES cluster setters require a stretch argument;
    # the other projects' setters accept only the photo-z shift.
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
        the survey assembler, plus coordinate (geometric bin or band
        centers, in arcminutes or multipoles), coordinate_label and the
        fully resolved integration settings. No files are written and no
        eigenvalues are repaired.
    Side effects:
        Reapplies the core accuracy boost and the Gaussian IA model to
        interface before the calculation.
    """
    # --- 1. THE CORE STATE, REAPPLIED ON EVERY CALL ---

    # The space is checked first, so a misspelled space leaves the core alone.
    if space not in ("real", "fourier"):
        raise ValueError("space must be 'real' or 'fourier'")

    # Reapply the core accuracy boost (the resolution of the C table
    # readers) and the Gaussian IA model on every calculation. The caller
    # must reinitialize when changing power_refinement; those input tables
    # are prepared before this assembly step. Covariance quadrature does
    # not follow the core boost.
    interface.init_accuracy_boost(
        accuracy_boost=settings["core_accuracyboost"],
        integration_accuracy=settings["integration_accuracy"],
    )
    set_gaussian_model(interface=interface, settings=settings)

    # The production bindings belong to the same compiled module, so they
    # read the core state configured above.
    if backend is not None:
        interface = backend

    # --- 2. MEASURED ROWS AS ONE CONTIGUOUS int32 [nrow,3] TABLE ---

    # Each row is (probe, field A, field B). Without a caller subset the
    # project's full galaxy/shear layout is measured.
    nlens = len(settings["lens_density_arcmin2"])
    nsource = len(settings["source_density_arcmin2"])
    if rows is None:
        rows = observable_rows(
            nlens=nlens, nsource=nsource,
            excluded_gammat=settings["excluded_gammat"],
        )
        # One E-mode spectrum supplies both shear correlations in Fourier
        # space, so the xi- rows (probe 1) are dropped.
        if space == "fourier":
            rows = rows[rows[:, 0] != 1]

    # Caller rows may be a list or an int64 array. dtype.kind is "i" for
    # signed and "u" for unsigned integers; anything else, floats included,
    # raises, because the int32 cast below would truncate it without notice.
    rows = np.asarray(rows)
    if rows.dtype.kind not in "iu":
        raise ValueError("observable rows must contain integer probe and field IDs")
    rows = np.ascontiguousarray(rows, dtype=np.int32)

    # --- 3. WHITE NOISE AND SETTINGS IN THE UNITS THE C CODE READS ---

    # One white-noise power per field, lenses first (see noise_powers).
    noise = noise_powers(
        lens_density=settings["lens_density_arcmin2"],
        source_density=settings["source_density_arcmin2"],
        sigma_component=settings["sigma_e_component"],
    )

    # resolved is a shallow copy of settings plus derived values: (pi/180)^2
    # converts deg^2 to sr and pi/(180*60) converts arcmin to rad. It is
    # returned with the result, so a saved forecast records its inputs.
    resolved = dict(settings)
    resolved["space"] = space
    resolved["mnu"] = settings["cosmology"]["mnu"]
    resolved["area_sr"] = settings["area_deg2"]*(np.pi/180.0)**2
    resolved["edges_rad"] = settings["theta_edges_arcmin"]*np.pi/(180.0*60.0)

    # --- 4. G, SSC AND cNG IN THE REQUESTED SPACE ---

    if space == "real":
        result = realspace_covariance(
            interface=interface, settings=resolved, rows=rows, noise=noise,
            progress=progress,
        )
        # The plotted coordinate is each bin's geometric center
        # sqrt(lower*upper) in arcmin: edges[:-1] holds the lower edges and
        # edges[1:] the upper ones.
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
    """Represent numerical settings in JSON without losing their resolved values.

    json.dumps calls this function for each value it cannot encode itself.

    Arguments:
        value = the value json.dumps could not encode.
    Returns:
        A list for a numpy array, or the Python scalar of a numpy scalar.
    Raises:
        TypeError for any other type, so no setting is written in a
        silently changed form.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot save setting of type {type(value).__name__}")


def save_forecast(result, filename):
    """Write a computed forecast and its resolved settings to a NumPy archive.

    Arguments:
        result = dict returned by compute_forecast.
        filename = output .npz path, outside the likelihood's data directory;
            numpy.savez appends .npz when the name lacks it.
    Returns:
        Nothing. Replaces the named output if it already exists.
        Arrays load with numpy.load(..., allow_pickle=False). settings_json
        and stages_json are text scalars read with json.loads(str(...)).
        Archive CAMB input tables separately when exact input reuse is needed.
    """
    settings = json.dumps(result["settings"], default=_json_array, allow_nan=False)
    # compute_forecast always returns the zero-IA Limber mean used by SSC.
    # A result without it is saved with its Gaussian signal in that slot,
    # which equals the SSC mean only for zero-IA Limber Gaussian spectra.
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
