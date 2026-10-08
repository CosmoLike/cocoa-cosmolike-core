"""Resolve each project's covariance accuracy baseline and nested refinements.

A project YAML defines its base accuracy. The public boost refines that
baseline's interpolation and cutoffs. An independent integration level
selects precomputed GSL rules: Gauss-Legendre nodes and weights that GSL
stores for fixed sizes. These controls change only numerical accuracy.
They do not change CAMB inputs, the survey footprint or the measured bins.
"""

from pathlib import Path

import numpy as np
import yaml


def covariance_accuracy(
    accuracy_boost=1, *, ell_max=100000, mask_ell_max=32768,
    ng_ell_intervals=127, non_gaussian_accuracyboost=1,
    window_accuracyboost=1, response_step=0.00005,
    core_accuracyboost=1, power_accuracyboost=8, integration_accuracy=0,
    nonlimber_lmax=1000, nonlimber_accuracyboost=1,
):
    """Resolve project-specific base controls and one overall refinement.

    Arguments:
        accuracy_boost = 1, 2, 4 or 8; refines tables and multipole cutoffs.
        ell_max = base real-space signal cutoff. Fourier band edges stay fixed.
        mask_ell_max = base cutoff of the survey-footprint spectrum.
        ng_ell_intervals = base intervals in ln(ell+1/2), from 2 to ell_max.
        non_gaussian_accuracyboost = refines only that interpolation grid.
        window_accuracyboost = multiplies 16384 lensing-window intervals.
        core_accuracyboost = multiplies the shared core reader table boost.
        power_accuracyboost = subdivisions of each CAMB log-k interval.
            Natural cubic preparation fills linear, nonlinear and cb tables;
            C readers then interpolate linearly. The default eight yields
            11,993 nodes from 1,500. The global boost multiplies this factor.
        nonlimber_lmax = base gg/gs correction cutoff, multiplied by the global boost.
        nonlimber_accuracyboost = 1, 2, 4 or 8; with the global boost, it
            multiplies the 4096 log-distance intervals. Padding scales with
            the interval count, preserving every old radial sample and
            Fourier period.
        integration_accuracy = independent quadrature level, 0 through 4.
            Selects 96, 128, 256, 512 or 1024 precomputed GSL nodes per
            radial, mass or angular panel, and adds one tree-angle panel
            per level to the base 20. It is also passed unchanged to the
            shared core. accuracy_boost never changes these rules.
        response_step = base half-width of the centered ln(k) derivative.
            The overall boost divides this width. Input-power interpolation
            can limit derivative convergence even with a very small step.

    Internal table boosts are positive integers multiplying the public boost.
    integration_accuracy=0 must already resolve the integrals. Angular
    kernels split wide bins into panels to resolve their fastest oscillations.
    Low-level tests may also use the precomputed 64-node rule. The internal
    Wynn mass tail alone uses 32/64/128/256/512 nodes at these levels;
    arbitrary smaller or generated rules are unsupported. Refinement never
    changes measured bins.

    Returns:
        dict of resolved settings. accuracy_boost holds the boost as an int
        and accuracy_parameters the unboosted inputs, for reproducibility.
        Boosted entries: ell_max, mask_ell_max and nonlimber_lmax (multipole
        cutoffs); ng_ell (float array of table multipoles, from 2 through
        the first node at or above ell_max*boost); nwindow and
        nonlimber_nchi (sample counts); response_step (half-width in ln k);
        core_accuracyboost and power_refinement (table factors). Quadrature
        entries ignore the boost: integration_accuracy, tree_npanel and one
        node count shared by radial_nquad, angle_nquad, halo_mass_nquad and
        tree_nquad. No data-vector or physical survey settings change.
    Raises:
        ValueError for accuracy_boost or nonlimber_accuracyboost outside
        1, 2, 4, 8, another factor or count that is not a positive integer,
        nonlimber_lmax < 2, an integration level outside 0..4, ell_max < 3
        or a nonpositive response_step.

    Interpolation refinement divides intervals, retaining every old sample.
    The boosted cutoff ell_max*boost appends cells to the boost-1
    logarithmic grid without moving its anchors at ell=2 and ell=ell_max;
    a different base ell_max defines a different grid. Gaussian quadrature
    nodes and weights refine together; they are not interpolation nodes and
    need not retain their old positions. High resolution alone is not proof
    of covariance/Fisher convergence.
    """
    if isinstance(accuracy_boost, (bool, np.bool_)) or not isinstance(
        accuracy_boost, (int, float, np.integer, np.floating)
    ) or accuracy_boost not in (1, 2, 4, 8):
        raise ValueError("accuracy_boost must be one of the integers 1, 2, 4, 8")
    internal = {
        "non_gaussian_accuracyboost": non_gaussian_accuracyboost,
        "window_accuracyboost": window_accuracyboost,
        "core_accuracyboost": core_accuracyboost,
        "power_accuracyboost": power_accuracyboost,
        "nonlimber_accuracyboost": nonlimber_accuracyboost,
    }
    for name, value in internal.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ) or not np.isfinite(value) or value < 1 or value != int(value):
            raise ValueError(f"{name} must be a positive integer")
        internal[name] = int(value)
    if nonlimber_accuracyboost not in (1, 2, 4, 8):
        raise ValueError("nonlimber_accuracyboost must be 1, 2, 4 or 8")
    if (isinstance(nonlimber_lmax, (bool, np.bool_))
            or not isinstance(nonlimber_lmax, (int, np.integer))
            or nonlimber_lmax < 2):
        raise ValueError("nonlimber_lmax must be an integer >= 2")
    if isinstance(integration_accuracy, (bool, np.bool_)) or not isinstance(
        integration_accuracy, (int, np.integer)
    ) or integration_accuracy not in (0, 1, 2, 3, 4):
        raise ValueError("integration_accuracy must be an integer from 0 to 4")
    for name, value in (
        ("ell_max", ell_max), ("mask_ell_max", mask_ell_max),
        ("ng_ell_intervals", ng_ell_intervals),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if ell_max < 3 or not np.isfinite(response_step) or response_step <= 0:
        raise ValueError("ell_max must exceed two and response_step must be positive")
    boost = int(accuracy_boost)
    intervals = int(ng_ell_intervals*non_gaussian_accuracyboost)

    # Use one anchored grid: a doubled boost inserts midpoint samples, and
    # the boosted cutoff ell_max*boost appends complete cells to the same
    # grid. Both anchors, ell=2 and the base ell_max, are assigned exactly
    # so exp/log rounding cannot move them.
    span = np.log(ell_max+0.5)-np.log(2.5)
    required_span = np.log(ell_max*boost+0.5)-np.log(2.5)
    base_intervals = int(np.ceil(intervals*required_span/span))
    position = np.arange(base_intervals*boost+1)/(intervals*boost)
    ng_ell = np.exp(np.log(2.5)+span*position)-0.5
    ng_ell[0] = 2.0
    ng_ell[intervals*boost] = float(ell_max)

    parameters = dict(internal)
    parameters.update({
        "ell_max": int(ell_max),
        "mask_ell_max": int(mask_ell_max),
        "ng_ell_intervals": int(ng_ell_intervals),
        "response_step": float(response_step),
        "nonlimber_lmax": int(nonlimber_lmax),
        "integration_accuracy": int(integration_accuracy),
    })
    # With twenty graded tree-angle panels, the panel next to theta=pi has
    # width pi/2^19 (about 6e-6 rad); each integration level adds one panel
    # and halves that width. The window and non-Limber sample counts are
    # intervals plus one, so a doubled boost keeps every old sample.
    result = {
        "accuracy_boost": boost,
        "accuracy_parameters": parameters,
        "ell_max": int(ell_max*boost),
        "nonlimber_lmax": int(nonlimber_lmax*boost),
        "nonlimber_nchi": int(4096*nonlimber_accuracyboost*boost+1),
        "mask_ell_max": int(mask_ell_max*boost),
        "ng_ell": ng_ell,
        "nwindow": int(16384*window_accuracyboost*boost+1),
        "tree_npanel": 20+int(integration_accuracy),
        "response_step": response_step/boost,
        "core_accuracyboost": int(core_accuracyboost*boost),
        "power_refinement": int(power_accuracyboost*boost),
        "integration_accuracy": int(integration_accuracy),
    }
    # Resolve the integration ladder once, before calling any C kernel.
    # Each size is a precomputed GSL rule, not an arbitrary requested count;
    # from level 1 on, each level doubles the nodes of the level below.
    # Keep this independent of every interpolation-table refinement.
    nodes = (96, 128, 256, 512, 1024)[integration_accuracy]
    for name in ("radial_nquad", "angle_nquad", "halo_mass_nquad", "tree_nquad"):
        result[name] = nodes
    return result


def load_covariance_accuracy(filename, accuracy_boost=None, **overrides):
    """Read a project's YAML baseline and resolve optional explicit refinements.

    Arguments:
        filename = path to covariance/default.yaml, containing only the
            covariance_accuracy keyword arguments.
        accuracy_boost = optional overall boost; None uses the YAML value.
        overrides = optional internal controls overriding that file.
    Returns:
        The resolved covariance_accuracy mapping. Unknown keys raise
        TypeError at the covariance_accuracy call, so misspelled accuracy
        controls are never ignored. A file that does not hold a mapping
        raises ValueError.
    """
    settings = yaml.safe_load(Path(filename).read_text())
    if not isinstance(settings, dict):
        raise ValueError(f"{filename} must contain a covariance accuracy mapping")
    settings.update(overrides)
    if accuracy_boost is not None:
        settings["accuracy_boost"] = accuracy_boost
    return covariance_accuracy(**settings)


def non_gaussian_multipoles(samples, ell_max):
    """Retain only the supplied table nodes needed to bracket the signal.

    Arguments:
        samples = finite 1D array, uniform in ln(ell+1/2), starting at two
            and covering ell_max. covariance_accuracy supplies nested nodes.
        ell_max = last measured/integrated signal multipole, at least two.
    Returns:
        Owned float array containing the original nodes through the first
        node at or above ell_max. The endpoint is never moved to ell_max.
        Fourier bands can end below the real-space cutoff, so unused upper
        samples are removed before the expensive halo calculations.
    Raises:
        ValueError unless samples are finite, increasing, uniform in
        ln(ell+1/2), start at ell=2 and reach ell_max, with ell_max >= 2.
    """
    grid = np.asarray(samples, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
            or grid[0] != 2.0 or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("ng_ell needs increasing finite samples starting at ell=2")
    if not np.isfinite(ell_max) or ell_max < 2 or grid[-1] < ell_max:
        raise ValueError("ng_ell must cover the signal range 2 <= ell <= ell_max")
    steps = np.diff(np.log(grid+0.5))
    if not np.allclose(steps, steps[0], rtol=1.e-12, atol=0.0):
        raise ValueError("ng_ell samples must be uniform in ln(ell+1/2)")
    count = max(2, np.count_nonzero(grid < ell_max)+1)
    return grid[:count].copy()
