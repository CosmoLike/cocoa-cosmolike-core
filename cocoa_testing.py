"""Shared harness machinery for the Cocoa project unit tests.

Every Cocoa project repository (roman_real, desy1xplanck, des_y3,
roman_fourier, roman_kl, ...) carries the same test architecture:
frozen configurations pinned by a SHA-256 manifest, reference chi2
values recorded at freeze time, race checks that share one model
instance, accuracy advisory scans, baryonic-feedback checks, and the
CFASTPT-vs-FASTPT comparison. The machinery is identical from
project to project; only the project DATA differs (the examples
table, the TATT point, the accuracy knobs, the dataset names, the
comparison contract). This module holds the machinery once; each
project's tests/cocoa_test_utils.py holds its data, builds ONE
CocoaTestHarness from it, and re-exports the harness's bound methods
under the historical names, so the test modules and the generator
keep importing everything from their project's cocoa_test_utils
exactly as before. A WORKED EXAMPLE below shows the complete shim a
new project writes.

THE FROZEN-STATE DOCTRINE

Everything a test evaluates is FROZEN: stored under the project's
tests/frozen/ and pinned by a SHA-256 hash (a 64-character
fingerprint that changes when any byte of the file changes) in
tests/manifest_sha256.json. The live project configuration is never
read, so a user can edit the examples, the likelihood default yaml
files, or ../data without touching the tests. The frozen state has
three parts:

  - frozen/frozen_config_*.py: one auto-generated module per
    configuration holding (a) the complete cobaya configuration as a
    yaml string, with every likelihood option and every parameter
    written out, including the ones that normally come from the
    likelihood default files, and (b) the exact sampled-parameter
    point the reference chi2 was evaluated at. Because every default
    is materialized in the frozen copy, a later edit to a live
    default file is shadowed and cannot reach the test.
  - frozen/data/: the tests' own copy of the data vectors,
    covariances, n(z), and masks.
  - frozen/reference_chi2.json: the chi2 values recorded at freeze
    time; each reference test compares its freshly computed chi2
    against one of them. (frozen/EXAMPLE_*.yaml snapshots sit next
    to them as human-readable provenance; no test reads them.)

Every test first verifies the manifest (verify_frozen) and refuses
to run when any frozen file changed - a tampered frozen state must
not produce a plausible-looking chi2. Refreshing the frozen state is
a deliberate maintainer action: generate_frozen_reference.py
--overwrite.

WORKER ISOLATION

Every model build runs in its own worker subprocess: the public test
quantities (single_model_chi2, ten_in_a_row_chi2, the baryon checks)
spawn a fresh python that imports the project's cocoa_test_utils by
file path, evaluates in-process, and hands the numbers back through
a temporary json file, while its progress lines stream to the same
terminal. The reason is the cosmolike C layer: it keeps the
data-vector dimensions in C globals and aborts the whole process
when a second configuration with different dimensions initializes
after the first ("IP::set_mask: inconsistent mask"). In a project
whose examples share one data set the isolation is preventive rather
than required; every project keeps the one architecture so they all
behave identically. The race check is the deliberate exception to
one-model-per-worker: its 11 evaluations share one model instance
inside one worker, because leaked state inside that instance is
exactly what it hunts.

THE ZERO-BASED COMPARISONS

Two families of checks refuse to compare chi2 values taken against
shipped data far from its minimum, where the chi2 responds LINEARLY
to tiny numerical changes and a harmless rounding-level shift reads
as an alarming difference:

  - the baryon accuracy checks write the default-settings theory
    vector at the fiducial point, make it the data of a temporary
    dataset descriptor, and evaluate the pushed-settings model
    against it: the default chi2 against its own vector is zero by
    construction, so the pushed model's chi2 IS the delta, a pure
    numerics response at a minimum (_baryon_accuracy_delta_impl);
  - the CFASTPT-vs-FASTPT comparison prints each implementation's
    theory vector at the same 30 IA points and computes
    delta^T C^-1 delta between the vectors, with C^-1 the masked
    inverse covariance from the compiled interface: the chi2 of the
    implementation DIFFERENCE, zero when the vectors agree. Each of
    the three blocks (cfastpt; python FAST-PT at the pinned low
    settings; python FAST-PT at the doubled grids) runs in its own
    fresh subprocess, so no cobaya cache, CAMB state, or C global
    survives from one block into the next
    (_fastpt_comparison_block).

A WORKED EXAMPLE: A PROJECT SHIM

A new project "demo" whose compiled interface imports as
cosmolike_demo_interface writes tests/cocoa_test_utils.py as below
(data values abbreviated; every name shown is part of the public
surface the test modules and the generator import):

    import os
    import sys

    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
    MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
    REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

    # path-based import so it works before start_cocoa.sh's
    # python-path setup
    _CORE_DIR = os.path.abspath(os.path.join(
        TESTS_DIR, "..", "..", "..", "external_modules", "code",
        "cosmolike_core"))
    if _CORE_DIR not in sys.path:
        sys.path.insert(0, _CORE_DIR)
    import cocoa_testing as _cct

    # ---- the project data --------------------------------------
    TATT_POINT = {"demo_A2_1": 0.05, ...}
    TATT_GENERATORS = {"tatt_demo.dataset": "example2"}
    EXAMPLES = {"example1": {...}, "example2": {...}, ...}
    HIGH_ACCURACY_LIKELIHOOD = {...}
    ACCURACY_KNOBS = [...]
    FASTPT_COMPARISON_TOLERANCE = 0.2
    FASTPT_LOW_SETTINGS = {...}
    FASTPT_HIGH_SETTINGS = {...}

    # ---- project-independent constants, re-exported ------------
    REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
    CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
    RACE_TOLERANCE = _cct.RACE_TOLERANCE
    RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
    HIGH_ACCURACY_CAMB_EXTRA_ARGS = (
        _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS)
    BARYON_METHODS = _cct.BARYON_METHODS
    BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES
    FASTPT_COMPARISON_POINTS = (
        _cct.fastpt_comparison_points("demo"))

    # ---- the harness -------------------------------------------
    _H = _cct.CocoaTestHarness(
        worker_file=__file__,
        interface_module="cosmolike_demo_interface",
        examples=EXAMPLES,
        tatt_point=TATT_POINT,
        accuracy_knobs=ACCURACY_KNOBS,
        high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
        fastpt_low_settings=FASTPT_LOW_SETTINGS,
        fastpt_high_settings=FASTPT_HIGH_SETTINGS,
        fastpt_points=FASTPT_COMPARISON_POINTS,
    )

    # ---- module functions under the historical names -----------
    require_cocoa_environment = _cct.require_cocoa_environment
    assert_omp_threads = _cct.assert_omp_threads
    sha256_of = _cct.sha256_of
    make_model = _cct.make_model
    evaluate_chi2 = _cct.evaluate_chi2
    _evaluate_cached = _cct._evaluate_cached
    _load_datavector = _cct._load_datavector
    _baryon_method = _cct._baryon_method
    _baryon_dataset = _cct._baryon_dataset
    report_chi2_test = _cct.report_chi2_test
    report_race_test = _cct.report_race_test
    report_accuracy = _cct.report_accuracy
    report_knob = _cct.report_knob
    report_fastpt_comparison = _cct.report_fastpt_comparison

    # ---- bound harness methods under the historical names ------
    compute_manifest = _H.compute_manifest
    verify_frozen = _H.verify_frozen
    load_reference = _H.load_reference
    _frozen_module = _H._frozen_module
    load_frozen_info = _H.load_frozen_info
    load_frozen_point = _H.load_frozen_point
    build_point = _H.build_point
    _single_model_chi2_impl = _H._single_model_chi2_impl
    _ten_in_a_row_impl = _H._ten_in_a_row_impl
    single_model_chi2 = _H.single_model_chi2
    ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
    baryon_accuracy_delta = _H.baryon_accuracy_delta
    baryon_drift_chi2 = _H.baryon_drift_chi2
    _worker = _H._worker
    _run_isolated = _H._run_isolated
    _fastpt_comparison_info = _H._fastpt_comparison_info
    _fastpt_comparison_block = _H._fastpt_comparison_block
    _run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
    cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s

A project with one project-wide synthetic NLA vector also passes
nla_dataset="synthetic_demo.dataset" to the constructor; a project
whose EXAMPLES entries carry their own "nla_dataset" keys needs
nothing extra (the per-example key wins); a project that evaluates
NLA against its shipped data_file passes neither. A project with
extra data blocks the generator reads (SYNTHETIC_VECTORS, a
NLA_DATASET constant) or project-only report printers keeps them in
the shim next to the data.

MAP OF THIS FILE

  Section 1: MODULE CONSTANTS (identical in every project)
    REQUIRED_OMP_THREADS, CHI2_TOLERANCE, RACE_TOLERANCE
    RACE_PERTURBATIONS, HIGH_ACCURACY_CAMB_EXTRA_ARGS
    BARYON_METHODS, BARYON_POINT_OVERRIDES
    fastpt_comparison_points  the 30 comparison points, built for a
                              project's parameter prefix

  Section 2: MODULE FUNCTIONS (no project state)
    require_cocoa_environment, assert_omp_threads, sha256_of
    make_model, evaluate_chi2, _evaluate_cached, _load_datavector
    _baryon_method, _baryon_dataset
    report_chi2_test, report_race_test, report_accuracy, report_knob
    report_fastpt_comparison

  Section 3: WORKER-SUBPROCESS PLUMBING (shared text)
    _WORKER_FLAG, _WORKER_DRIVER

  Section 4: class CocoaTestHarness (bound to one project's spec)
    frozen-state integrity   compute_manifest, verify_frozen,
                             load_reference
    the chi2 pipeline        _frozen_module, load_frozen_info,
                             load_frozen_point, build_point,
                             _single_model_chi2_impl,
                             _ten_in_a_row_impl
    baryonic feedback        _baryon_accuracy_delta_impl,
                             _baryon_drift_chi2_impl
    worker isolation         _worker, _run_isolated
    test quantities          single_model_chi2, ten_in_a_row_chi2,
                             baryon_accuracy_delta, baryon_drift_chi2
    CFASTPT vs FASTPT        _fastpt_comparison_info,
                             _fastpt_comparison_block,
                             _run_fastpt_comparison_worker,
                             cfastpt_vs_fastpt_chi2s
"""

import hashlib
import json
import os
import shutil
import tempfile


# =============================================================================
# SECTION 1: MODULE CONSTANTS (identical in every project)
# =============================================================================

# The race tests must run multi-threaded: with one thread there is no
# thread scheduling, so an OpenMP race could never show up. The value
# is a string, not a number: environment variables only carry text.
REQUIRED_OMP_THREADS = "4"

# Reference tests: |chi2(now) - chi2(frozen reference)| must stay
# below this. The bound tolerates compiler and library-version noise
# but catches a real physics change.
CHI2_TOLERANCE = 0.2

# Race tests: |chi2(10th of a row) - chi2(fresh model)|. The two
# numbers come from the same code on the same inputs, so only float
# noise is allowed; a state leak produces a much larger shift.
RACE_TOLERANCE = 1.0e-4

# The nine cosmologies evaluated before the fiducial point in a race
# test. Each entry replaces the named parameters in the frozen point.
# They stay inside the priors of the frozen configurations (an
# out-of-prior point would evaluate to -inf and abort the test), and
# they change the chi2 by orders of magnitude, so state leaked from
# any of them would visibly move the final fiducial evaluation.
RACE_PERTURBATIONS = [
    {"As_1e9": 1.95},
    {"As_1e9": 2.25},
    {"omegam": 0.28},
    {"omegam": 0.33},
    {"H0": 64.0},
    {"H0": 71.0},
    {"ns": 0.95},
    {"w": -1.1, "w0pwa": -1.1},
    {"omegab": 0.052, "mnu": 0.15},
]

# High-accuracy CAMB settings for the accuracy advisory checks: the
# same physics evaluated with the numerical knobs pushed far beyond
# the defaults. The LIKELIHOOD-side table is project data (real-space
# projects push lmax; Fourier-space projects have no lmax option), so
# it lives in each project's spec.
HIGH_ACCURACY_CAMB_EXTRA_ARGS = {
    "halofit_version": "takahashi",
    "AccuracyBoost": 2.0,       # default 1.05
    "dark_energy_model": "ppf",
    "accurate_massive_neutrino_transfers": False,
    "k_per_logint": 50,         # default 10
    "kmax": 50.0,               # default 5.0-7.5
}

# ---- baryonic feedback methods (bfmt theory block) --------------------------

# One entry per feedback method the bfmt theory block implements:
# (label, theory-block options selecting it, fixed evaluation point
# for its sampled parameters). The SP(k) points are pyspk's
# documented examples; the emulator points are the fiducial values
# quoted in the example yamls. test_accuracy_baryons.py evaluates
# each method at the default and the pushed numerical settings.
BARYON_METHODS = [
    ("spk power law", {"baryon_model": 1, "spk_fb_model": 1},
     {"fb_a_spk": 0.4, "fb_pow_spk": 0.3}),
    ("spk akino", {"baryon_model": 1, "spk_fb_model": 2},
     {"alpha_spk": 4.189, "beta_spk": 1.273, "gamma_spk": 0.298}),
    # pyspk's documented double-power-law example (epsilon 0.3,
    # alpha 1.1, beta 0.2, gamma 0.5) pushes fb outside SP(k)'s
    # calibrated band at z >~ 1.4 (pyspk then returns NaN and the
    # block falls back to unity per redshift): a check with that
    # point would test the fallback, not the method. This point
    # matches the Akino relation's amplitude and mass slope at the
    # pivot and stays inside the band over the full z grid.
    ("spk double power law", {"baryon_model": 1, "spk_fb_model": 3},
     {"epsilon_spk": 0.66, "alpha_spk": 0.35, "beta_spk": 0.2,
      "gamma_spk": 0.3}),
    ("bcemu", {"baryon_model": 2},
     {"log10Mc_bcemu": 13.32, "mu_bcemu": 0.93, "thej_bcemu": 4.235,
      "gamma_bcemu": 2.25, "delta_bcemu": 6.40, "eta_bcemu": 0.15,
      "deta_bcemu": 0.14}),
    ("flamingo", {"baryon_model": 3},
     {"fgas_sigma_flamingo": 0.0, "mstar_sigma_flamingo": 0.0,
      "jet_frac_flamingo": 0.0}),
    ("baccoemu", {"baryon_model": 4},
     {"M_c_baccoemu": 14.0, "eta_baccoemu": -0.3,
      "beta_baccoemu": -0.22, "M1_z0_cen_baccoemu": 10.5,
      "theta_inn_baccoemu": -0.86}),
    ("bcemu2025", {"baryon_model": 5},
     {"Theta_co_bcemu25": 0.3, "log10Mc_bcemu25": 13.1,
      "mu_bcemu25": 1.0, "delta_bcemu25": 6.0, "eta_bcemu25": 0.10,
      "deta_bcemu25": 0.22, "Nstar_bcemu25": 0.028}),
]

# Cosmology shifts a method needs so its OWN training box contains
# the evaluation point. BACCOemu's omega_baryon floor is 0.04001,
# exactly above the fiducial omegab = 0.04, so its checks (and its
# generated data vector) evaluate at omegab = 0.049 - inside the box
# and inside the yaml prior. Generator and checks apply the SAME
# override, so the chi2 still sits at the minimum by construction.
BARYON_POINT_OVERRIDES = {
    "baccoemu": {"omegab": 0.049},
}


def _baryon_method(label):
    """Look one BARYON_METHODS entry up by its label.

    Arguments:
      label = the first field of one BARYON_METHODS entry.

    Returns:
      the (label, theory options, parameter point) tuple.

    Raises:
      ValueError when label names no entry.
    """
    matches = [b for b in BARYON_METHODS if b[0] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown baryon method {label!r}")
    return matches[0]


def _baryon_dataset(label):
    """Dataset descriptor name for one feedback method's own vector.

    Arguments:
      label = a BARYON_METHODS label.

    Returns:
      the frozen/data descriptor file name, e.g.
      baryon_spk_akino.dataset for "spk akino".

    Raises:
      nothing.
    """
    return "baryon_" + label.replace(" ", "_") + ".dataset"


# ---- the CFASTPT-vs-FASTPT comparison points --------------------------------

# The comparison evaluates the SAME intrinsic-alignment points in
# every project (each project's five TATT parameters carry the same
# [-5, 5] and [0, 2] prior boxes, so a per-point number is comparable
# project to project). The first 20 points are hard-coded draws:
# drawn ONCE, uniformly across the prior boxes, with
# numpy.random.default_rng(20250922), and written out here so every
# run evaluates exactly these points. Points 21-30 are the
# one-parameter-at-a-time family that names the TATT parameter
# driving a divergence: an all-zero null, each amplitude alone at
# both signs, and the redshift powers and BTA riding on the amplitude
# that activates them (alone they are exact nulls). Each tuple is
# (A1_1, A1_2, A2_1, A2_2, BTA_1); fastpt_comparison_points attaches
# a project's parameter prefix.
_FASTPT_COMPARISON_VALUES = [
    (-2.347878, -2.631529, -4.548776, 4.089803, 1.299591),
    (-4.5757, 1.842436, -2.778512, 2.349448, 1.771619),
    (-1.885483, -4.988485, -1.485879, -4.495962, 0.884889),
    (4.467901, -3.113044, 0.788438, 4.321531, 0.595105),
    (2.925418, 2.106288, 3.514003, -4.307538, 0.106922),
    (0.737236, -2.601612, -3.525379, -2.532121, 1.551288),
    (-2.208689, 1.166122, -3.89846, -3.625951, 1.458385),
    (3.577859, -2.150368, 3.68132, -0.800727, 1.338299),
    (-3.301099, -1.899991, 2.247241, -2.085669, 1.21194),
    (-1.957003, 2.044701, 1.357407, 3.861245, 0.112379),
    (1.56864, -1.191915, 2.709754, -3.689616, 0.581821),
    (0.092758, 4.358014, -0.665741, -1.574787, 0.158704),
    (1.205006, 1.544994, -2.161392, -0.047689, 1.065058),
    (2.623534, -4.583449, 4.401911, 3.545763, 1.463577),
    (-4.459096, 3.114698, -0.526033, 2.268589, 1.739542),
    (0.206121, -4.811619, 4.734078, -2.90544, 1.16909),
    (3.56276, -0.662237, -0.174979, 4.150983, 1.275072),
    (3.593366, -4.875329, 2.680328, -3.169156, 0.538465),
    (1.290369, 0.975332, -0.422358, -2.637129, 0.641767),
    (1.835056, -1.493955, 1.138184, 4.009877, 1.228736),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 0.0),
    (-4.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 4.0, 0.0, 0.0),
    (0.0, 0.0, -4.0, 0.0, 0.0),
    (4.0, 4.0, 0.0, 0.0, 0.0),
    (4.0, -4.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 4.0, 4.0, 0.0),
    (0.0, 0.0, 4.0, -4.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 2.0),
]


def fastpt_comparison_points(prefix):
    """The 30 comparison points under one project's parameter prefix.

    Arguments:
      prefix = the sampled-parameter prefix of the project's five
               TATT parameters, e.g. "roman" (roman_A1_1, ...),
               "DES", or "ROMAN_KL".

    Returns:
      a list of 30 {parameter name: value} dictionaries, one per
      comparison point, index-aligned with _FASTPT_COMPARISON_VALUES.

    Raises:
      nothing.
    """
    names = [f"{prefix}_A1_1", f"{prefix}_A1_2", f"{prefix}_A2_1",
             f"{prefix}_A2_2", f"{prefix}_BTA_1"]
    return [dict(zip(names, values))
            for values in _FASTPT_COMPARISON_VALUES]


# =============================================================================
# SECTION 2: MODULE FUNCTIONS (no project state)
# =============================================================================
def require_cocoa_environment():
    """Refuse to run outside a started Cocoa shell, then move to ROOTDIR.

    start_cocoa.sh exports ROOTDIR (the absolute path of the Cocoa/
    folder) and prepares the library paths the compiled cosmolike
    interface needs. Without it, importing the likelihood would fail
    with a confusing linker error, so this check turns that failure
    into an instruction. The chdir matters because component paths in
    the frozen configuration (for example CAMB's
    ./external_modules/code/CAMB) are relative to ROOTDIR.

    Arguments:
      none.

    Returns:
      nothing; on success the process working directory is ROOTDIR.

    Raises:
      RuntimeError telling the user to activate the cocoa environment
      and source start_cocoa.sh when ROOTDIR is not exported.
    """
    if "ROOTDIR" not in os.environ:
        raise RuntimeError(
            "ROOTDIR is not set. Activate the cocoa conda environment and run "
            "`source start_cocoa.sh` from the Cocoa/ folder before running "
            "these tests."
        )
    os.chdir(os.environ["ROOTDIR"])


def assert_omp_threads():
    """Refuse a race test that would not actually run multi-threaded.

    OpenMP reads OMP_NUM_THREADS once, when the compiled library is
    first loaded, so the value must be in the environment before any
    cobaya or cosmolike import. The test modules set it at their first
    line; this check catches a run that imported the stack some other
    way first (for example from an interactive session).

    Arguments:
      none.

    Returns:
      nothing when OMP_NUM_THREADS equals REQUIRED_OMP_THREADS.

    Raises:
      RuntimeError naming the observed value and the required one.
    """
    # .get returns None when the variable is unset, so the error can
    # show "None" rather than crash on a missing key
    observed = os.environ.get("OMP_NUM_THREADS")
    if observed != REQUIRED_OMP_THREADS:
        # !r prints the value in its python literal form: None and
        # the text '4' stay distinguishable in the message
        raise RuntimeError(
            f"OMP_NUM_THREADS={observed!r}; the race-condition tests require "
            f"OMP_NUM_THREADS={REQUIRED_OMP_THREADS} and it must be set "
            "before cobaya/cosmolike are imported."
        )


def sha256_of(path):
    """Fingerprint one file with SHA-256.

    Arguments:
      path = absolute path of the file to hash.

    Returns:
      the 64-character lowercase hexadecimal SHA-256 digest of the
      file's bytes. Reading happens in 1 MiB blocks so the 80 MB
      covariance never sits in memory at once.

    Raises:
      OSError when the file cannot be opened or read.
    """
    hasher = hashlib.sha256()
    # "rb" reads raw bytes (hashing is byte-level); the with block
    # closes the file on every exit, an exception included
    with open(path, "rb") as f:
        # the lambda is an unnamed one-line function wrapping f.read;
        # two-argument iter calls it again and again until it returns
        # b"" (end of file); 1 << 20 is 2**20 bytes = 1 MiB per read
        for block in iter(lambda: f.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_model(info):
    """Build a cobaya model (theory plus likelihood, ready to evaluate).

    Arguments:
      info = a cobaya input dictionary from load_frozen_info.

    Returns:
      the cobaya Model. Building one loads CAMB and the compiled
      cosmolike interface and reads the frozen data files, which takes
      a few seconds; the callers print a progress line first.

    Raises:
      whatever cobaya's get_model raises on an invalid configuration
      (the frozen strings were valid at freeze time, so a failure
      here signals an environment or code change, not a data change).
    """
    from cobaya.model import get_model

    return get_model(info)


def evaluate_chi2(model, point):
    """Evaluate one point and return the likelihood chi2.

    cached=False forces a full recomputation: the race tests evaluate
    the same point twice on one model, and letting cobaya return a
    cached value would compare a number with itself.

    Arguments:
      model = the cobaya Model to evaluate on.
      point = {parameter name: value} covering the sampled parameters.

    Returns:
      chi2 = -2 ln L of the single likelihood, as a plain float.

    Raises:
      RuntimeError when the model holds more than one likelihood
      (the -2*loglikes[0] extraction would then be ambiguous);
      AssertionError when the chi2 is not finite, which is how an
      out-of-prior or rejected point shows up.
    """
    import numpy as np

    # logposterior runs the full pipeline (theory + likelihood) at the
    # point; cached=False forces recomputation (see docstring)
    posterior = model.logposterior(point, cached=False)
    # loglikes = one ln L per likelihood component, in model order
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    # float() converts the numpy scalar into a plain python float,
    # which json can store and print cleanly
    return float(chi2)


def _evaluate_cached(model, point):
    """One evaluation with cobaya's caching left on.

    The CFASTPT-vs-FASTPT comparison points share one cosmology and
    differ only in the five IA parameters, so letting cobaya skip
    components whose inputs did not change (CAMB, and the fastpt
    theory block) makes each point cost only the likelihood's
    data-vector rebuild. evaluate_chi2 forces cached=False because
    the race tests need full recomputation; this comparison does not.

    Arguments:
      model = the cobaya Model to evaluate on.
      point = {parameter name: value} covering the sampled
              parameters.

    Returns:
      chi2 = -2 ln L of the single likelihood, as a plain float.

    Raises:
      RuntimeError on more than one likelihood; AssertionError on a
      non-finite chi2 (an out-of-prior or rejected point).
    """
    import numpy as np

    posterior = model.logposterior(point, cached=True)
    if len(posterior.loglikes) != 1:
        raise RuntimeError(
            f"expected exactly one likelihood: {posterior.loglikes}")
    chi2 = -2.0 * posterior.loglikes[0]
    if not np.isfinite(chi2):
        raise AssertionError(f"non-finite chi2 at point {point}")
    return float(chi2)


def _load_datavector(path):
    """Read one printed theory data vector as a 1D numpy array.

    print_datavector writes one line per data point; cosmolike's
    format carries the entry index in the first column and the value
    in the second, so a two-column file is reduced to its value
    column. A one-column file is used as it is.

    Arguments:
      path = the .modelvector file a likelihood evaluation printed.

    Returns:
      the vector values as a 1D float array.

    Raises:
      RuntimeError when the file has more than two columns.
    """
    import numpy as np

    table = np.loadtxt(path)
    # ndim == 1 means one number per line: the values themselves
    if table.ndim == 1:
        return table
    if table.shape[1] == 2:
        # [:, 1] selects the second column (all rows): the values
        return table[:, 1]
    raise RuntimeError(
        f"{path}: expected 1 or 2 columns, found {table.shape[1]}")


# ---- terminal reports -------------------------------------------------------
# Each printer returns the difference it printed; the assertion on
# that difference lives in the calling test method, not here.
def report_chi2_test(number, label, chi2, ref, tol):
    """Print one reference-comparison test as a readable block.

    A bare pytest PASSED does not say what was compared, so each test
    prints its own numbers: the freshly computed chi2, the frozen
    reference, their absolute difference, and the limit the assertion
    uses. flush=True makes the block appear immediately (pytest runs
    with -s, so nothing buffers it).

    Arguments:
      number = the test number shown in the header.
      label  = one line naming the example, probe, and IA model.
      chi2   = the chi2 computed in this run.
      ref    = the frozen reference chi2.
      tol    = the pass limit on |chi2 - ref| (CHI2_TOLERANCE).

    Returns:
      |chi2 - ref|, the printed difference.

    Raises:
      nothing.
    """
    delta = abs(chi2 - ref)
    # one multi-line f-string: '-' * 66 repeats the dash 66 times (a
    # rule), :.6f prints a fixed six decimals, and the a-if-else
    # inside the last braces picks the verdict word
    print(f"""
{'-' * 66}
TEST {number}: {label}
  chi2 (this run)     = {chi2:.6f}
  frozen reference    = {ref:.6f}
  |delta chi2|        = {delta:.6f}   (limit: < {tol})
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_race_test(number, label, fresh, tenth, tol):
    """Print one race-condition test as a readable block.

    Arguments:
      number = the test number shown in the header.
      label  = one line naming the example, probe, and IA model.
      fresh  = chi2 of the fiducial point evaluated first on the model.
      tenth  = chi2 of the same point evaluated as the 10th of a row.
      tol    = the pass limit on |tenth - fresh| (RACE_TOLERANCE).

    Returns:
      |tenth - fresh|, the printed difference.

    Raises:
      nothing.
    """
    delta = abs(tenth - fresh)
    # the same construct as report_chi2_test's block, at :.8f (a
    # fixed eight decimals: the race limit is 1e-4)
    print(f"""
{'-' * 66}
TEST {number}: {label}
  fresh-model chi2    = {fresh:.8f}
  10th of 10 in a row = {tenth:.8f}
  |delta chi2|        = {delta:.8f}   (limit: < {tol})
  OMP_NUM_THREADS     = {os.environ.get('OMP_NUM_THREADS')}
  -> {'OK' if delta < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return delta


def report_accuracy(label, chi2_high, default_ref,
                    default_name="default, frozen"):
    """Print one default-vs-high-accuracy check. Advisory only.

    The default-settings chi2 is the frozen reference (recorded at
    freeze time); the high-accuracy chi2 is computed in this run. The
    difference is the numerical error of the default settings at this
    point: there is no pass/fail because how much numerical error an
    analysis tolerates is a judgment call, not a fixed bound.

    Arguments:
      label       = one line naming the probe and IA model.
      chi2_high   = chi2 with HIGH_ACCURACY settings, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2_high - default_ref, the printed difference.

    Raises:
      nothing.
    """
    delta = chi2_high - default_ref
    # ljust(28) pads each name to the same column so the = signs line
    # up whatever the default is called; :+.6f prints six decimals
    # with the sign ALWAYS shown, so a shift reads as +0.01 at a
    # glance
    line_high = "  chi2 (high accuracy)".ljust(28)
    line_default = f"  chi2 ({default_name})".ljust(28)
    line_delta = "  delta chi2 (high-default) "
    print(f"""
{'-' * 66}
ACCURACY: {label}
{line_high}= {chi2_high:.6f}
{line_default}= {default_ref:.6f}
{line_delta}= {delta:+.6f}
{'-' * 66}""", flush=True)
    return delta


def report_knob(label, chi2, default_ref):
    """Print one entry of the one-knob-at-a-time scan. Advisory only.

    Arguments:
      label       = the ACCURACY_KNOBS entry evaluated.
      chi2        = chi2 with only that knob changed, this run.
      default_ref = the frozen default-settings reference chi2.

    Returns:
      chi2 - default_ref, the printed difference.

    Raises:
      nothing.
    """
    delta = chi2 - default_ref
    # :30s pads the label to 30 characters so the columns line up;
    # :12.6f = width 12 with six decimals; the + forces the sign
    print(f"  KNOB {label:30s} chi2 = {chi2:12.6f}  "
          f"delta = {delta:+12.6f}", flush=True)
    return delta


def report_fastpt_comparison(number, label, chi2_cfastpt,
                             chi2_fastpt_low, chi2_fastpt_high,
                             dchi2_low, dchi2_high, tol):
    """Print the CFASTPT-vs-FASTPT sweep as a readable block.

    Two lines per comparison point. The first carries the raw chi2
    of each configuration against the shipped data; those values are
    large across the IA prior and are printed only as information,
    because their differences ride the local chi2 slope. The second
    line carries the tested quantities: the chi2 of each FASTPT
    vector against the CFASTPT vector at the same point. The
    fastpt-low entry is the pass/fail quantity; the doubled-grid
    entry is advisory.

    Arguments:
      number           = the test number shown in the header.
      label            = one line naming the example, probe, and the
                         camb/cosmolike settings.
      chi2_cfastpt     = the per-point chi2 list under IA_code 0.
      chi2_fastpt_low  = the list under IA_code 1 at
                         FASTPT_LOW_SETTINGS.
      chi2_fastpt_high = the list under IA_code 1 at
                         FASTPT_HIGH_SETTINGS.
      dchi2_low        = per point, the chi2 of the FASTPT(low)
                         vector against the CFASTPT vector.
      dchi2_high       = the same for FASTPT(high).
      tol              = the pass limit on max of dchi2_low.

    Returns:
      max of dchi2_low over the points, the quantity the calling
      test asserts on.

    Raises:
      nothing.
    """
    import numpy as np

    largest = float(np.max(dchi2_low))
    middle = float(np.median(dchi2_low))
    high_largest = float(np.max(dchi2_high))
    print(f"""
{'-' * 66}
TEST {number}: {label}""", flush=True)
    # zip walks the five lists in step, handing every per-point
    # quantity at once; start=1 makes the rows read 1..30
    for i, (c, fl, fh, dl, dh) in enumerate(
            zip(chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
                dchi2_low, dchi2_high), start=1):
        # :14.6f = six decimals in a 14-wide field so the columns
        # line up
        print(f"  point {i:2d}:  CFASTPT = {c:14.6f}   "
              f"FASTPT(low) = {fl:14.6f}   FASTPT(high) = {fh:14.6f}",
              flush=True)
        print(f"             dchi2 vs the CFASTPT vector:  "
              f"low = {dl:.6f}   high = {dh:.6f}", flush=True)
    print(f"""  max dchi2(FASTPT low vs CFASTPT)    = {largest:.6f}   \
(limit: < {tol})
  median dchi2(FASTPT low vs CFASTPT) = {middle:.6f}
  max dchi2(FASTPT high vs CFASTPT)   = {high_largest:.6f}   (advisory)
  -> {'OK' if largest < tol else 'EXCEEDS LIMIT'}
{'-' * 66}""", flush=True)
    return largest


# =============================================================================
# SECTION 3: WORKER-SUBPROCESS PLUMBING (shared text)
# =============================================================================
# One flag separates the two roles: the parent process (pytest or the
# generator) spawns workers; a process carrying this environment
# variable IS a worker and runs the physics in-process.
_WORKER_FLAG = "COCOA_TESTS_WORKER"

# The driver handed to `python -c` inside the worker: load the
# project's cocoa_test_utils from its file path (which builds its
# harness and re-exports _worker) and call _worker with the seven
# command-line arguments. The * in _worker(*sys.argv[2:9]) spreads
# the seven-element slice into seven separate arguments.
_WORKER_DRIVER = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location("
    "'cocoa_test_utils_worker', sys.argv[1])\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "module._worker(*sys.argv[2:9])\n"
)


# =============================================================================
# SECTION 4: THE PROJECT-BOUND HARNESS
# =============================================================================
class CocoaTestHarness:
    """The shared test machinery, bound to one project's data.

    A project's tests/cocoa_test_utils.py builds ONE instance and
    re-exports its bound methods under the names the test modules
    already import, so the project file reduces to its data plus the
    binding.

    Constructor arguments (the project spec):
      worker_file      = the project's cocoa_test_utils.py path
                         (__file__ in that module); the tests/ paths
                         and the worker-subprocess target derive from
                         it.
      interface_module = the compiled interface's import name, e.g.
                         "cosmolike_roman_real_interface" (the
                         CFASTPT-vs-FASTPT comparison reads the
                         masked inverse covariance from it).
      examples         = the project's EXAMPLES table.
      tatt_point       = the project's TATT_POINT replacements.
      accuracy_knobs   = the project's ACCURACY_KNOBS table.
      high_accuracy_likelihood = the project's likelihood-side
                         high-accuracy settings.
      fastpt_low_settings, fastpt_high_settings = the comparison's
                         pinned fastpt extra_args tables.
      fastpt_points    = the comparison points under the project's
                         parameter prefix (fastpt_comparison_points).
      nla_dataset      = None, or one synthetic-NLA dataset name used
                         by EVERY example (a project whose examples
                         name their own carries "nla_dataset" keys in
                         EXAMPLES instead, which take precedence).
      fastpt_masks     = the scale-cut masks the CFASTPT-vs-FASTPT
                         comparison can run under (the --mask option
                         of the tests). "frozen" (always first) keeps
                         each example's own tatt_dataset; every other
                         name selects the frozen dataset variant
                         "<tatt_dataset stem>_<name>.dataset", a copy
                         of that descriptor whose mask_file line
                         names the chosen mask (e.g. "ones" for the
                         no-scale-cuts all-ones mask). The default
                         offers "frozen" alone.
    """

    def __init__(self, worker_file, interface_module, examples,
                 tatt_point, accuracy_knobs, high_accuracy_likelihood,
                 fastpt_low_settings, fastpt_high_settings,
                 fastpt_points, nla_dataset=None,
                 fastpt_masks=("frozen",)):
        """Bind the shared machinery to one project's data.

        Arguments:
          see the class docstring, which documents every constructor
          argument once, next to the project spec it belongs to.

        Returns:
          nothing; the instance carries the spec plus the tests/
          paths derived from worker_file.

        Raises:
          nothing.
        """
        # Everything the tests read or write lives relative to the
        # project's tests folder, so the suite works no matter which
        # directory pytest is launched from.
        self.worker_file = os.path.abspath(worker_file)
        self.tests_dir = os.path.dirname(self.worker_file)
        self.frozen_dir = os.path.join(self.tests_dir, "frozen")
        self.manifest_file = os.path.join(self.tests_dir,
                                          "manifest_sha256.json")
        self.reference_file = os.path.join(self.frozen_dir,
                                           "reference_chi2.json")
        self.interface_module = interface_module
        self.examples = examples
        self.tatt_point = tatt_point
        self.accuracy_knobs = accuracy_knobs
        self.high_accuracy_likelihood = high_accuracy_likelihood
        self.fastpt_low_settings = fastpt_low_settings
        self.fastpt_high_settings = fastpt_high_settings
        self.fastpt_points = fastpt_points
        self.nla_dataset = nla_dataset
        self.fastpt_masks = tuple(fastpt_masks)

    # ---- frozen-state integrity ---------------------------------------------
    def compute_manifest(self):
        """Hash every file currently under tests/frozen/.

        __pycache__ folders and .pyc files are skipped: Python writes
        them as a side effect of importing the frozen modules, so
        hashing them would make the manifest fail after the first
        run. .DS_Store files (macOS Finder metadata) are skipped for
        the same reason.

        Arguments:
          none.

        Returns:
          a dictionary {relative path: sha256 digest}, with paths
          relative to the tests/ folder using "/" separators, sorted
          by path so the manifest file is stable across platforms.

        Raises:
          OSError when a frozen file cannot be read.
        """
        files = {}
        # os.walk visits every folder under frozen/, handing back the
        # folder path, its subfolder names, and its file names
        for base, dirs, names in os.walk(self.frozen_dir):
            # the comprehension keeps every subfolder name except
            # __pycache__; assigning through dirs[:] rewrites
            # os.walk's own list in place, which stops the walk from
            # entering the dropped folders (a plain dirs = ... would
            # not)
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in sorted(names):
                if name == ".DS_Store" or name.endswith(".pyc"):
                    continue
                full = os.path.join(base, name)
                rel = os.path.relpath(full, self.tests_dir).replace(
                    os.sep, "/")
                files[rel] = sha256_of(full)
        # sorted(files.items()) orders the (path, digest) pairs by
        # path; dict() rebuilds the table in that order, so the
        # manifest json written from it never reshuffles between runs
        return dict(sorted(files.items()))

    def verify_frozen(self):
        """Fail every test up front when the frozen state was edited.

        Compares the stored manifest with a fresh hash of
        tests/frozen/ in both directions, so an edited file
        (CHANGED), a deleted file (MISSING), and a new file (EXTRA)
        are all reported. This runs before any model is built: a
        tampered frozen state must not produce a plausible-looking
        chi2.

        Arguments:
          none.

        Returns:
          nothing when every frozen file matches the manifest.

        Raises:
          AssertionError listing every mismatched path and pointing
          to generate_frozen_reference.py --overwrite for a
          deliberate refresh; AssertionError also when the manifest
          file itself is absent (the frozen state was never
          generated).
        """
        if not os.path.isfile(self.manifest_file):
            raise AssertionError(
                "tests/manifest_sha256.json is missing; run "
                "generate_frozen_reference.py --overwrite to create the "
                "frozen test state."
            )
        # expected = the {relative path: sha256 digest} table written
        # at freeze time; it is the definition of "untouched"
        with open(self.manifest_file) as f:
            expected = json.load(f)["files"]
        # actual = the same table computed from the files on disk now
        actual = self.compute_manifest()
        # collect every discrepancy before raising: a report naming
        # all problem files at once beats failing on the first one
        problems = []
        for rel, digest in expected.items():
            if rel not in actual:
                # the manifest lists it but the file is gone from disk
                problems.append(f"MISSING  {rel}")
            elif actual[rel] != digest:
                # the file exists but at least one byte differs
                problems.append(f"CHANGED  {rel}")
        # both directions matter: a file ADDED to frozen/ is as
        # suspicious as an edited one, so the reverse scan runs too
        for rel in actual:
            if rel not in expected:
                problems.append(f"EXTRA    {rel}")
        if problems:
            raise AssertionError(
                "Frozen test data does not match tests/manifest_sha256.json "
                "(someone edited the frozen copies; the tests refuse to "
                "run):\n  "
                + "\n  ".join(problems)
                + "\nIf the change is deliberate, regenerate with "
                "generate_frozen_reference.py --overwrite."
            )

    def load_reference(self):
        """Read the frozen reference chi2 values.

        Arguments:
          none.

        Returns:
          the dictionary stored in frozen/reference_chi2.json: one
          entry per configuration plus a "_meta" entry recording when
          and how the references were generated. The file sits inside
          frozen/, so verify_frozen() also protects it from editing.

        Raises:
          OSError when the reference file is absent (the frozen state
          was never generated).
        """
        with open(self.reference_file) as f:
            return json.load(f)

    # ---- the chi2 pipeline --------------------------------------------------
    # cobaya and numpy are imported inside the functions, not at the
    # top of this module. The reason is OpenMP: OMP_NUM_THREADS must
    # be in the environment before the compiled libraries load, and
    # it is the TEST modules that set it, on their first line, before
    # importing this module's callers.
    def _frozen_module(self, example):
        """Load one frozen configuration module from its file path.

        importlib is used instead of a plain import statement because
        the frozen modules live inside frozen/, which is data, not a
        package: it has no __init__.py and is never on sys.path.
        Loading by path also guarantees the file that verify_frozen()
        hashed is exactly the file being executed.

        Arguments:
          example = a key of the project's EXAMPLES table.

        Returns:
          the loaded module, carrying the attributes `yaml_string`
          (the complete configuration) and `point` (the frozen
          evaluation point).

        Raises:
          KeyError when example names no EXAMPLES entry; OSError
          when the frozen module file is absent.
        """
        import importlib.util

        path = os.path.join(self.frozen_dir,
                            self.examples[example]["frozen_module"])
        # the importlib three-step: describe the file (spec), create
        # an empty module object from the description, then run the
        # file's code inside that object to fill in its attributes
        spec = importlib.util.spec_from_file_location(
            f"frozen_{example}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_frozen_info(self, example, tatt, high_accuracy=False,
                         overrides=None, baryon=None):
        """Build the cobaya input dictionary for one frozen configuration.

        Starts from the frozen module's yaml string and applies the
        run-time adjustments the tests need: the likelihood `path` is
        pointed at the absolute location of frozen/data, IA_model
        selects the intrinsic-alignment model, the data_file switches
        to the generated TATT or synthetic NLA vector when the
        project carries one, and cobaya's log level is raised so the
        component-loading chatter does not bury the test reports.

        An EXAMPLES entry carrying the "emulator" flag gets two more
        adjustments, both serving the accuracy measurement of
        test_emul2.py: its data_file becomes the exact counterpart's
        synthetic vector (so |emulator - exact| is the emulator error
        and nothing else), and the counterpart's ggl_exclude is
        copied in, because cosmolike only accepts a mask whose length
        equals the data-vector layout it computes from ggl_exclude.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True selects the TATT IA model, False keeps NLA.
          high_accuracy = True applies the project's high-accuracy
                    likelihood settings and
                    HIGH_ACCURACY_CAMB_EXTRA_ARGS on top of the
                    frozen configuration (accuracy advisory checks
                    only; not available for emulator configurations,
                    which have no camb block).
          overrides = None, or a pair (likelihood overrides, camb
                    extra_args overrides) applied on top of the
                    frozen configuration; _single_model_chi2_impl
                    builds this pair from one ACCURACY_KNOBS entry.
          baryon  = None, or a BARYON_METHODS label: the bfmt theory
                    block is added with that method selected, the
                    likelihood's external_baryon_suppression switch
                    is turned on, and the method's parameter point
                    enters the params block as fixed values.

        Returns:
          the input dictionary ready for cobaya's get_model.

        Raises:
          ValueError when tatt or high_accuracy is requested for an
          emulator entry (the emulators were trained for the shipped
          IA settings and carry no camb accuracy knobs to push).
        """
        from cobaya.yaml import yaml_load

        cfg = self.examples[example]
        # cfg.get("emulator") is None (falsy) for the exact-physics
        # entries, which do not carry the flag at all
        if cfg.get("emulator") and (tatt or high_accuracy):
            raise ValueError(
                f"{example}: the emulator configurations run only at "
                "their shipped IA settings (no TATT variant) and carry "
                "no camb accuracy knobs (no high_accuracy variant)")
        # _frozen_module loads frozen/<frozen_module>.py by path and
        # hands back its yaml_string attribute: the complete
        # configuration with every option and parameter written out
        # at freeze time
        info = yaml_load(self._frozen_module(example).yaml_string)
        # the tests drive the model directly, so a sampler or output
        # block left in the info would only confuse cobaya; pop's
        # second argument makes the removal a no-op instead of an
        # error when the key is absent
        info.pop("sampler", None)
        info.pop("output", None)
        # log level WARNING (30): component-loading chatter would
        # bury the test reports
        info["debug"] = 30
        info["timing"] = False
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        # the frozen string stores the ROOTDIR-relative data path;
        # the absolute path is independent of the working directory
        likelihood_block["path"] = os.path.join(self.frozen_dir, "data")
        # tests must not write files as a side effect: an example can
        # ship print_datavector: True aimed at chains/, which does
        # not exist in a fresh clone (the vector generators re-enable
        # printing deliberately, into their own directories)
        likelihood_block["print_datavector"] = False
        # intrinsic-alignment model selection: 0 = NLA, 1 = TATT (the
        # `1 if tatt else 0` form yields 1 when tatt is True, else 0)
        likelihood_block["IA_model"] = 1 if tatt else 0
        if tatt:
            # TATT evaluates against its own generated data vector so
            # the chi2 sits at a minimum (see the TATT_GENERATORS
            # comment in the project file)
            likelihood_block["data_file"] = cfg["tatt_dataset"]
        else:
            # a project with synthetic NLA vectors does the same for
            # NLA: the shipped data_file is real data (or an
            # off-minimum modelvector) and the fiducial point sits
            # away from its minimum. Per-example "nla_dataset" keys
            # win over the project-wide constant; a project with
            # neither keeps the shipped data_file.
            nla = cfg.get("nla_dataset") or self.nla_dataset
            if nla is not None:
                likelihood_block["data_file"] = nla
        if cfg.get("emulator"):
            # the dataset just selected belongs to the exact
            # counterpart, and cosmolike only accepts a mask whose
            # length equals the data-vector layout it computes from
            # ggl_exclude (see the docstring); copy the counterpart's
            # frozen layout so the two configurations describe the
            # same data vector
            exact_cfg = self.examples[cfg["exact_example"]]
            exact_module = self._frozen_module(cfg["exact_example"])
            exact_info = yaml_load(exact_module.yaml_string)
            exact_block = exact_info["likelihood"][exact_cfg["likelihood"]]
            likelihood_block["ggl_exclude"] = exact_block["ggl_exclude"]
        if high_accuracy:
            # dict.update merges the pushed settings into the block
            # in place, overwriting any key both sides carry
            likelihood_block.update(self.high_accuracy_likelihood)
            info["theory"]["camb"]["extra_args"].update(
                HIGH_ACCURACY_CAMB_EXTRA_ARGS)
        if overrides is not None:
            # one knob at a time (an ACCURACY_KNOBS entry): the same
            # mechanism as high_accuracy, restricted to a single
            # setting; the overrides pair unpacks into its two
            # dictionaries
            like_over, camb_over = overrides
            likelihood_block.update(like_over)
            info["theory"]["camb"]["extra_args"].update(camb_over)
        if tatt:
            # a TATT parameter can be SAMPLED in the frozen
            # configuration (it has a prior; build_point then sets
            # its value in the evaluation point) or FIXED (a plain
            # value; it must be replaced here, before the model is
            # built, because a fixed parameter cannot change per
            # evaluation)
            for name, value in self.tatt_point.items():
                # .get returns None instead of raising when the name
                # is missing, so the error below can name the
                # parameter
                block = info["params"].get(name)
                if block is None:
                    raise ValueError(
                        f"TATT parameter {name} is not in the frozen "
                        "configuration")
                if isinstance(block, dict) and "prior" not in block:
                    block["value"] = value
        if baryon is not None:
            _, theory_options, baryon_point = _baryon_method(baryon)
            # the likelihood requests the suppression product only
            # when this switch is on (see external_baryon_suppression
            # in likelihood/_cosmolike_prototype_base.py)
            likelihood_block["external_baryon_suppression"] = True
            # dict(a, **b) builds a new dictionary with a's entries
            # plus b's: python_path tells cobaya where the bfmt class
            # lives
            info["theory"]["bfmt"] = dict(
                {"python_path": os.path.join(
                    os.environ["ROOTDIR"], "external_modules", "code",
                    "baryon_suppression")},
                **theory_options)
            for name, value in baryon_point.items():
                info["params"][name] = value
        return info

    def load_frozen_point(self, example):
        """Read the frozen evaluation point of one example.

        Arguments:
          example = a key of the project's EXAMPLES table.

        Returns:
          a fresh {parameter name: value} dictionary (copied, so a
          caller may modify it without affecting later calls).

        Raises:
          KeyError when example names no EXAMPLES entry (through
          _frozen_module).
        """
        # dict(...) builds a COPY of the module's point table: the
        # edits a caller makes stay in its copy, never in the module
        return dict(self._frozen_module(example).point)

    def build_point(self, model, example, tatt):
        """Assemble the exact point a test evaluates, with a safety check.

        The frozen point must cover the model's sampled parameters
        one to one. When likelihood or theory code changes its
        parameter set (a new nuisance parameter appears, or one is
        removed), evaluating would either fail cryptically or
        silently pick up a new default, so the mismatch is reported
        here by name instead.

        Arguments:
          model   = the cobaya Model the point will be evaluated on.
          example = a key of the project's EXAMPLES table.
          tatt    = True replaces the TATT_POINT values (nonzero
                    A2/BTA) in the frozen point; False evaluates it
                    unchanged.

        Returns:
          a {parameter name: value} dictionary covering every sampled
          parameter of the model.

        Raises:
          AssertionError listing the parameters that appeared or
          vanished when the model's sampled set differs from the
          frozen point.
        """
        # load_frozen_point returns a copy of the frozen module's
        # point: the exact {parameter: value} table the references
        # were computed at
        point = self.load_frozen_point(example)
        # sampled = the parameters THIS model, built from today's
        # code, expects to receive; the frozen point must cover them
        # exactly. set() turns the name list into a set so the two
        # sides can be compared and subtracted; set(point) takes the
        # dictionary's KEYS
        sampled = set(model.parameterization.sampled_params())
        if sampled != set(point):
            # sampled - set(point) = names only the model has; the
            # reversed difference = names only the point has;
            # sorted() fixes the order so the message is reproducible
            raise AssertionError(
                "sampled-parameter set differs from the frozen point (the "
                "likelihood/theory code changed its parameters):\n"
                f"  new since freeze: {sorted(sampled - set(point))}\n"
                f"  gone since freeze: {sorted(set(point) - sampled)}"
            )
        if tatt:
            # only SAMPLED TATT parameters appear in the point; the
            # fixed ones were already replaced inside the
            # configuration by load_frozen_info (which also catches
            # unknown names)
            for name, value in self.tatt_point.items():
                if name in point:
                    point[name] = value
        return point

    def _single_model_chi2_impl(self, example, tatt,
                                high_accuracy=False, knob=None,
                                baryon=None):
        """In-process body of single_model_chi2 (worker side).

        Runs inside the worker subprocess only: building a model
        here, next to a model of different dimensions, would abort
        the process (see _run_isolated). It chains the pipeline:
        load_frozen_info, make_model, build_point, evaluate_chi2.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True evaluates the TATT variant, False the NLA
                    one.
          high_accuracy = True evaluates with the pushed numerical
                    settings (see load_frozen_info).
          knob    = None, or the label of one ACCURACY_KNOBS entry;
                    that knob's overrides are applied alone (the
                    one-at-a-time scan of test_accuracy.py).
          baryon  = None, or a BARYON_METHODS label: the bfmt theory
                    block computes the baryonic suppression of the
                    nonlinear power spectrum
                    (test_accuracy_baryons.py).

        Returns:
          the chi2 as a float, or None when a baryon method rejected
          the frozen fiducial (a training-box violation).

        Raises:
          ValueError when knob names no ACCURACY_KNOBS entry.
        """
        # `"TATT" if tatt else "NLA"` picks the first name when tatt
        # is True, the second otherwise
        ia_label = "TATT" if tatt else "NLA"
        if high_accuracy:
            ia_label += ", high accuracy"
        overrides = None
        if knob is not None:
            # knob = a label from ACCURACY_KNOBS; the comprehension
            # keeps only the entries whose first field k[0] equals
            # it, so matches is a list with one member (or none:
            # unknown label)
            matches = [k for k in self.accuracy_knobs if k[0] == knob]
            if len(matches) != 1:
                raise ValueError(f"unknown accuracy knob {knob!r}")
            overrides = (matches[0][1], matches[0][2])
            ia_label += f", knob: {knob}"
        if baryon is not None:
            ia_label += f", baryons: {baryon}"
        print(f"  building model ({example}, {ia_label}) ...", flush=True)
        # load_frozen_info returns the frozen configuration
        # dictionary with the run-time adjustments applied;
        # make_model turns it into an evaluable cobaya Model (loads
        # CAMB and the cosmolike interface)
        info = self.load_frozen_info(example, tatt,
                                     high_accuracy=high_accuracy,
                                     overrides=overrides, baryon=baryon)
        model = make_model(info)
        # build_point returns the frozen evaluation point after
        # checking that the point and the model name the same
        # sampled parameters: if the likelihood or theory code gained
        # or lost a sampled parameter since the freeze, the mismatch
        # is reported by name instead of failing deep inside cobaya
        point = self.build_point(model, example, tatt)
        if baryon is not None:
            # dict(point) copies before the in-place update below
            point = dict(point)
            point.update(BARYON_POINT_OVERRIDES.get(baryon, {}))
        print("  evaluating the fiducial point ...", flush=True)
        if baryon is not None:
            # With the feedback on, a non-finite chi2 means the
            # method REJECTED the frozen fiducial (a training-box
            # violation; the warning above names the offending
            # parameter). The B-checks report that as documented
            # behavior, so hand back None instead of letting the
            # assertion kill the worker.
            try:
                return evaluate_chi2(model, point)
            except AssertionError:
                return None
        return evaluate_chi2(model, point)

    def _ten_in_a_row_impl(self, example, tatt):
        """In-process body of ten_in_a_row_chi2 (worker side).

        Race check: the fiducial evaluated fresh and as 10th of a
        row. On ONE model instance, in order: the fiducial point (the
        fresh value), then the nine RACE_PERTURBATIONS cosmologies,
        then the fiducial again as the 10th point of the row. State
        leaked between evaluations, or an OpenMP race under
        REQUIRED_OMP_THREADS threads, shifts the second fiducial
        value away from the first; correct code reproduces it to
        float noise. Each evaluation prints its chi2, so a stuck or
        slow run is visible line by line.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True runs the TATT variant, False the NLA one.

        Returns:
          (fresh, tenth): chi2 of the first fiducial evaluation and
          chi2 of the fiducial as the 10th point of the row, both
          floats.

        Raises:
          AssertionError when the sampled-parameter set differs from
          the frozen point (build_point) or a chi2 is not finite
          (evaluate_chi2).
        """
        # `"TATT" if tatt else "NLA"` picks the first name when tatt
        # is True, the second otherwise
        ia_label = "TATT" if tatt else "NLA"
        print(f"  building model ({example}, {ia_label}) ...", flush=True)
        # one model instance for the whole sequence: sharing the
        # instance is the point, since leaked state lives inside it
        info = self.load_frozen_info(example, tatt)
        model = make_model(info)
        point = self.build_point(model, example, tatt)
        # the fresh value: the fiducial evaluated before anything
        # else touched this model instance
        fresh = evaluate_chi2(model, point)
        # :.8f prints the value with a fixed eight decimals (:.4f
        # below: four); :2d pads the row counter to a width of two
        print(f"  fresh model, fiducial point:  chi2 = {fresh:.8f}",
              flush=True)
        # enumerate yields (counter, entry) pairs; start=1 makes the
        # printed row numbers begin at 1 instead of 0
        for i, perturbation in enumerate(RACE_PERTURBATIONS, start=1):
            # an emulator configuration can sample fewer parameters
            # than the exact ones (mnu is fixed inside the emulator
            # training), so a perturbation key the model does not
            # sample is dropped rather than kept in a separate
            # perturbation table per configuration. The comprehension
            # builds a new dictionary from the (name, value) pairs
            # whose name the point carries.
            applied = {k: v for k, v in perturbation.items()
                       if k in point}
            # {**point, **applied} is a NEW dictionary: point's
            # entries with the applied ones written over them; point
            # itself stays untouched for the final fiducial
            # evaluation
            chi2 = evaluate_chi2(model, {**point, **applied})
            # one "name=value" text per changed parameter, glued with
            # ", " into the progress line
            changed = ", ".join(f"{k}={v}" for k, v in applied.items())
            print(f"  row {i:2d}/10 ({changed}):  chi2 = {chi2:.4f}",
                  flush=True)
        tenth = evaluate_chi2(model, point)
        print(f"  row 10/10 (fiducial again):  chi2 = {tenth:.8f}",
              flush=True)
        return fresh, tenth

    # ---- baryonic feedback --------------------------------------------------
    def _baryon_accuracy_delta_impl(self, baryon, knob=None):
        """Delta chi2 for one feedback method, against its own vector.

        The zero-based mechanism at the frozen fiducial: a
        DEFAULT-settings model with this method's feedback on writes
        its theory vector during evaluation (print_datavector); that
        vector becomes the data of a temporary dataset descriptor, so
        the default chi2 against it is zero by construction; a second
        model - high accuracy, or one accuracy knob alone - evaluates
        at the SAME point against that descriptor, and its chi2 IS

            delta chi2 = chi2(pushed settings) - chi2(default)

        a pure numerics response at the minimum. Nothing is written
        into frozen/ (the manifest pins every byte there); the
        vector, the descriptor, and the symlinked data folder live
        and die inside a temporary directory. The evaluation point is
        the frozen fiducial plus the method's cosmology override
        (BARYON_POINT_OVERRIDES), applied to BOTH evaluations. Both
        models share example1's data-vector dimensions, so building
        them one after another inside one worker process is safe.

        Arguments:
          baryon = a BARYON_METHODS label.
          knob   = None for the all-knobs high-accuracy comparison,
                   or an ACCURACY_KNOBS label evaluated alone.

        Returns:
          the delta chi2 as a float.

        Raises:
          ValueError when knob names no ACCURACY_KNOBS entry;
          RuntimeError when the evaluation printed no vector, the
          vector's length disagrees with the original data, or the
          descriptor's data_file line is not unique.
        """
        cfg = self.examples["example1"]
        frozen_data_dir = os.path.join(self.frozen_dir, "data")
        info = self.load_frozen_info("example1", tatt=False,
                                     baryon=baryon)
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        workdir = tempfile.mkdtemp(prefix="cocoa_baryon_model_")
        try:
            # the likelihood joins path + filename for EVERY file a
            # descriptor names, so the temporary directory must look
            # like a complete data folder: symlink each frozen data
            # file in
            for name in sorted(os.listdir(frozen_data_dir)):
                os.symlink(os.path.join(frozen_data_dir, name),
                           os.path.join(workdir, name))
            slug = baryon.replace(" ", "_")
            vector_name = f"baryon_{slug}.modelvector"
            descriptor_name = f"baryon_{slug}.dataset"
            vector_path = os.path.join(workdir, vector_name)
            # the default model's evaluation writes the theory
            # vector; its chi2 (against the frozen no-feedback data)
            # plays no role
            likelihood_block["print_datavector"] = True
            likelihood_block["print_datavector_file"] = vector_path
            print(f"  building the default model ({baryon}) ...",
                  flush=True)
            model = make_model(info)
            point = dict(self.build_point(model, "example1",
                                          tatt=False))
            point.update(BARYON_POINT_OVERRIDES.get(baryon, {}))
            print("  evaluating (writes the synthetic vector) ...",
                  flush=True)
            evaluate_chi2(model, point)
            if not os.path.isfile(vector_path):
                raise RuntimeError(
                    f"print_datavector wrote no file at {vector_path}")
            # full-length check: the covariance and the masks select
            # entries by position, so a short vector would misalign
            # them
            with open(vector_path) as f:
                generated_lines = sum(1 for _ in f)
            with open(os.path.join(frozen_data_dir,
                                   likelihood_block["data_file"])) as f:
                descriptor = f.read()
            original_vector = None
            for line in descriptor.splitlines():
                if line.strip().startswith("data_file"):
                    original_vector = line.split("=", 1)[1].strip()
            with open(os.path.join(frozen_data_dir,
                                   original_vector)) as f:
                original_lines = sum(1 for _ in f)
            if generated_lines != original_lines:
                raise RuntimeError(
                    f"generated vector has {generated_lines} lines; "
                    f"the original {original_vector} has "
                    f"{original_lines}")
            # the temporary descriptor: the frozen text with only the
            # data_file line renamed, so the same covariance, n(z),
            # and masks are read but the synthetic vector is the data
            replaced = 0
            out_lines = []
            for line in descriptor.splitlines(keepends=True):
                if line.strip().startswith("data_file"):
                    out_lines.append(f"data_file = {vector_name}\n")
                    replaced += 1
                else:
                    out_lines.append(line)
            if replaced != 1:
                raise RuntimeError(
                    "expected exactly one data_file line, found "
                    f"{replaced}")
            with open(os.path.join(workdir, descriptor_name), "w") as f:
                f.write("".join(out_lines))
            # the pushed-settings model, at the same point, against
            # the synthetic vector: its chi2 is the delta by
            # construction
            overrides = None
            high_accuracy = knob is None
            if knob is not None:
                matches = [k for k in self.accuracy_knobs
                           if k[0] == knob]
                if len(matches) != 1:
                    raise ValueError(f"unknown accuracy knob {knob!r}")
                overrides = (matches[0][1], matches[0][2])
            info_high = self.load_frozen_info(
                "example1", tatt=False, baryon=baryon,
                high_accuracy=high_accuracy, overrides=overrides)
            block_high = info_high["likelihood"][cfg["likelihood"]]
            block_high["path"] = workdir
            block_high["data_file"] = descriptor_name
            label = knob if knob is not None else "high accuracy"
            print(f"  building the pushed model ({label}) ...",
                  flush=True)
            model_high = make_model(info_high)
            print("  evaluating against the synthetic vector ...",
                  flush=True)
            return float(evaluate_chi2(model_high, point))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _baryon_drift_chi2_impl(self, baryon):
        """Drift chi2 of one feedback method against its FROZEN vector.

        The frozen vector was written at freeze time by
        generate_frozen_reference.py --baryons: the default-settings
        theory prediction with this method's feedback on, at the
        frozen fiducial plus the method's cosmology override. At
        freeze time the chi2 against it was zero by construction, so
        any chi2 above the tolerance today means cosmolike or the
        bfmt theory block changed its prediction since the freeze -
        the same pinning idea as the reference tests, applied to the
        feedback pipeline.

        Arguments:
          baryon = a BARYON_METHODS label.

        Returns:
          the chi2 as a float.

        Raises:
          AssertionError when the sampled-parameter set differs from
          the frozen point (build_point) or the chi2 is not finite
          (evaluate_chi2).
        """
        cfg = self.examples["example1"]
        info = self.load_frozen_info("example1", tatt=False,
                                     baryon=baryon)
        likelihood_block = info["likelihood"][cfg["likelihood"]]
        # the method's own frozen dataset: same covariance, n(z), and
        # masks, but the freeze-time feedback prediction as the data
        likelihood_block["data_file"] = _baryon_dataset(baryon)
        print(f"  building model ({baryon}, frozen vector) ...",
              flush=True)
        model = make_model(info)
        point = dict(self.build_point(model, "example1", tatt=False))
        point.update(BARYON_POINT_OVERRIDES.get(baryon, {}))
        print("  evaluating the fiducial point ...", flush=True)
        return float(evaluate_chi2(model, point))

    # ---- worker-subprocess isolation ----------------------------------------
    def _worker(self, function, example, tatt, high_accuracy, knob,
                baryon, result_path):
        """Worker-side entry: run one evaluation and save the numbers.

        Arguments:
          function      = "single" (one chi2), "race" (fresh, tenth),
                          "bdelta", or "bdrift" (the baryon checks).
          example       = a key of the project's EXAMPLES table.
          tatt          = "1" for the TATT variant, "0" for NLA.
          high_accuracy = "1" for the pushed numerical settings, "0"
                          not.
          knob          = an ACCURACY_KNOBS label evaluated alone, or
                          the empty string for none.
          baryon        = a BARYON_METHODS label switching the bfmt
                          theory block on, or the empty string for
                          none.
          result_path   = file the result is written into as json;
                          the parent reads it back. Progress prints
                          go to the inherited stdout, so the terminal
                          streams them.

        Returns:
          nothing; the result lands in result_path.

        Raises:
          whatever the selected implementation raises; the parent
          then sees a nonzero exit code instead of a result file
          (_run_isolated turns that into a RuntimeError).
        """
        require_cocoa_environment()
        # the flags arrive as text (command-line arguments are
        # strings); comparing against "1" turns them back into
        # booleans
        tatt = tatt == "1"
        high_accuracy = high_accuracy == "1"
        if function == "single":
            # `knob or None` turns the empty string (no knob) into
            # None: `or` hands back its second operand when the first
            # is empty
            value = self._single_model_chi2_impl(
                example, tatt, high_accuracy=high_accuracy,
                knob=knob or None, baryon=baryon or None)
        elif function == "bdelta":
            value = self._baryon_accuracy_delta_impl(baryon,
                                                     knob=knob or None)
        elif function == "bdrift":
            value = self._baryon_drift_chi2_impl(baryon)
        else:
            value = list(self._ten_in_a_row_impl(example, tatt))
        # json.dump writes the value into the file as json text; the
        # parent reads it back with json.load. The with block closes
        # the file even when the dump fails
        with open(result_path, "w") as f:
            json.dump(value, f)

    def _run_isolated(self, function, example, tatt,
                      high_accuracy=False, knob=None, baryon=None):
        """Spawn one worker subprocess and hand back its result.

        Every model build runs in its own worker: in a project whose
        examples use different data-set dimensions, the cosmolike C
        layer aborts a process that initializes both, and every
        project keeps one architecture so they all behave
        identically.

        Arguments:
          function      = "single", "race", "bdelta", or "bdrift"
                          (see _worker).
          example       = a key of the project's EXAMPLES table.
          tatt          = True for the TATT variant.
          high_accuracy = True for the pushed numerical settings.
          knob          = an ACCURACY_KNOBS label evaluated alone, or
                          None.
          baryon        = a BARYON_METHODS label, or None.

        Returns:
          the json value the worker wrote: a float for "single", a
          two-element list [fresh, tenth] for "race".

        Raises:
          RuntimeError naming the configuration when the worker dies
          without writing a result (the cosmolike C layer aborts the
          process on an internal inconsistency instead of raising).
        """
        import subprocess
        import sys

        # delete=False keeps the file when the with block closes it:
        # only a fresh unique NAME is needed here; the worker writes
        # the file and the finally below removes it
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as tmp:
            result_path = tmp.name
        # dict(os.environ) is a COPY of the environment: the edits
        # below reach only the worker subprocess, never this process
        environment = dict(os.environ)
        environment[_WORKER_FLAG] = "1"
        # OpenMP reads this at library load inside the fresh worker,
        # so the requirement holds for every spawned evaluation
        environment["OMP_NUM_THREADS"] = REQUIRED_OMP_THREADS
        # subprocess.run starts the worker and BLOCKS until it exits;
        # env=environment hands the child the edited environment
        # copy. The booleans travel as "1"/"0" text (command-line
        # arguments are strings; _worker decodes them), and `knob or
        # ""` turns None into the empty string the same way
        completed = subprocess.run(
            [sys.executable, "-c", _WORKER_DRIVER, self.worker_file,
             function, example, "1" if tatt else "0",
             "1" if high_accuracy else "0",
             knob or "", baryon or "", result_path],
            env=environment)
        # the finally below runs on EVERY exit from the try, an
        # exception included, so the temporary file never outlives
        # this call
        try:
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker for ({function}, {example}, tatt={tatt}) "
                    f"exited with code {completed.returncode} before "
                    "writing a result; a cosmolike-level abort prints "
                    "its reason (e.g. IP::set_mask) just above")
            # json.load parses the worker's file back into the value
            # the worker json.dump-ed (a float, or a two-element
            # list)
            with open(result_path) as f:
                return json.load(f)
        finally:
            if os.path.exists(result_path):
                os.unlink(result_path)

    # ---- test quantities ----------------------------------------------------
    def single_model_chi2(self, example, tatt, high_accuracy=False,
                          knob=None, baryon=None):
        """chi2 of the frozen fiducial point, evaluated in a fresh worker.

        This is the quantity the reference tests compare against the
        frozen reference and the quantity the generator stores as
        that reference. The evaluation runs in a subprocess (see
        _run_isolated for why isolation is mandatory).

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True evaluates the TATT variant, False the NLA
                    one.
          high_accuracy = True evaluates with the pushed numerical
                    settings; expect minutes instead of seconds.
          knob    = None, or the label of one ACCURACY_KNOBS entry;
                    forwarded to the worker, which applies that
                    knob's overrides alone (the one-at-a-time scan of
                    test_accuracy.py).
          baryon  = None, or a BARYON_METHODS label.

        Returns:
          the chi2 as a float, or None when a baryon method rejected
          the frozen fiducial.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        # .get returns None when the variable is absent, so a normal
        # (parent) process fails this test and spawns a worker
        # instead
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._single_model_chi2_impl(
                example, tatt, high_accuracy=high_accuracy, knob=knob,
                baryon=baryon)
        value = self._run_isolated("single", example, tatt,
                                   high_accuracy=high_accuracy,
                                   knob=knob, baryon=baryon)
        # a None from the worker means a baryon method rejected the
        # frozen fiducial (see _single_model_chi2_impl); it travels
        # as json null
        return None if value is None else float(value)

    def baryon_accuracy_delta(self, baryon, knob=None):
        """Delta chi2 of one feedback method, in a fresh worker.

        See _baryon_accuracy_delta_impl for the mechanism (the
        synthetic on-the-fly data vector). Both models of the pair
        run inside ONE worker subprocess.

        Arguments:
          baryon = a BARYON_METHODS label.
          knob   = None for the all-knobs high-accuracy comparison,
                   or an ACCURACY_KNOBS label evaluated alone.

        Returns:
          the delta chi2 as a float.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._baryon_accuracy_delta_impl(baryon, knob=knob)
        return float(self._run_isolated("bdelta", "example1", False,
                                        knob=knob, baryon=baryon))

    def baryon_drift_chi2(self, baryon):
        """Drift chi2 of one feedback method, in a fresh worker.

        See _baryon_drift_chi2_impl for the mechanism (the frozen
        feedback vector written at freeze time).

        Arguments:
          baryon = a BARYON_METHODS label.

        Returns:
          the chi2 as a float.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._baryon_drift_chi2_impl(baryon)
        return float(self._run_isolated("bdrift", "example1", False,
                                        baryon=baryon))

    def ten_in_a_row_chi2(self, example, tatt):
        """Race check, evaluated in one fresh worker subprocess.

        The whole 11-evaluation sequence runs inside ONE worker: the
        race check needs the evaluations to share a model instance,
        and the worker boundary only isolates this configuration from
        the other configurations' dimensions.

        Arguments:
          example = a key of the project's EXAMPLES table.
          tatt    = True runs the TATT variant, False the NLA one.

        Returns:
          (fresh, tenth): chi2 of the first fiducial evaluation and
          chi2 of the fiducial as the 10th point of the row, both
          floats.

        Raises:
          RuntimeError when the worker dies without writing a result
          (_run_isolated).
        """
        # .get returns None when the variable is absent, so a normal
        # (parent) process fails this test and spawns a worker
        # instead
        if os.environ.get(_WORKER_FLAG) == "1":
            return self._ten_in_a_row_impl(example, tatt)
        # the worker's two-element json list unpacks into the two
        # names
        fresh, tenth = self._run_isolated("race", example, tatt)
        return float(fresh), float(tenth)

    # ---- the CFASTPT vs FASTPT comparison -----------------------------------
    def _fastpt_comparison_info(self, example, fastpt, high,
                                fastpt_settings, mask="frozen"):
        """The cobaya input of one comparison block.

        Starts from the project's frozen TATT configuration
        (load_frozen_info) and applies the comparison's two
        adjustments: IA_code selects the perturbation-theory
        implementation (0 keeps cfastpt, 1 asks the python FAST-PT
        package through the fastpt theory block, which is added here
        with the path the examples use), and fastpt_settings becomes
        that block's extra_args.

        A project can FIX some TATT second-order parameters
        (value: 0.0) in its configuration instead of sampling them;
        the comparison varies all five IA parameters, so any fixed
        comparison parameter is promoted to a sampled one carrying
        its prior box (a no-op in the projects that sample all five).

        Arguments:
          example = a key of the project's EXAMPLES table; the
                    comparison uses "example1" (cosmic shear).
          fastpt  = False keeps cfastpt (IA_code 0), True selects
                    python FAST-PT (IA_code 1).
          high    = True applies the HIGH_ACCURACY settings on top
                    (the --high=1 option of the tests).
          fastpt_settings = None, or the fastpt block's extra_args
                    table (FASTPT_LOW_SETTINGS or
                    FASTPT_HIGH_SETTINGS).
          mask    = a fastpt_masks entry: "frozen" (the default)
                    keeps the example's own tatt_dataset; any other
                    name swaps the data_file for the frozen variant
                    "<tatt_dataset stem>_<mask>.dataset", identical
                    except for its mask_file line (the --mask option
                    of the tests).

        Returns:
          the input dictionary ready for cobaya's get_model.

        Raises:
          ValueError when mask is not a fastpt_masks entry, or from
          load_frozen_info when example names an emulator entry (the
          comparison never evaluates one).
        """
        if mask not in self.fastpt_masks:
            raise ValueError(
                f"mask {mask!r} is not offered by this project; the "
                f"choices are {self.fastpt_masks}")
        info = self.load_frozen_info(example, tatt=True,
                                     high_accuracy=high)
        likelihood_block = info["likelihood"][
            self.examples[example]["likelihood"]]
        if mask != "frozen":
            # the variant descriptor follows the naming rule the
            # class docstring states; str.removesuffix drops the
            # ".dataset" tail so the mask lands before it
            stem = self.examples[example]["tatt_dataset"].removesuffix(
                ".dataset")
            likelihood_block["data_file"] = f"{stem}_{mask}.dataset"
        if fastpt:
            likelihood_block["IA_code"] = 1
            # setdefault hands back the existing "theory" dictionary,
            # or first inserts the empty {} and hands that back, so
            # the assignment lands inside info either way
            info.setdefault("theory", {})["fastpt"] = {
                "path": "./external_modules/code/FAST-PT",
            }
            if fastpt_settings is not None:
                # dict(...) copies, so a caller's settings table is
                # never shared with (or mutated through) the built
                # model
                info["theory"]["fastpt"]["extra_args"] = dict(
                    fastpt_settings)

        # promote any fixed comparison parameter to a sampled one
        # with its prior box; the evaluation base gains it with a
        # zero default in _fastpt_comparison_block
        for name in self.fastpt_points[0]:
            spec = info["params"].get(name)
            if (isinstance(spec, dict) and "value" in spec
                    and "prior" not in spec):
                lo, hi = ((0.0, 2.0) if name.endswith("BTA_1")
                          else (-5.0, 5.0))
                info["params"][name] = {
                    "prior": {"min": lo, "max": hi},
                    "ref": spec["value"],
                    "proposal": 0.1,
                    "latex": spec.get("latex", name),
                }
        return info

    def _fastpt_comparison_block(self, example, fastpt, high,
                                 fastpt_settings=None, label=None,
                                 vectors_dir=None,
                                 reference_label=None, mask="frozen"):
        """The 30-point sweep on ONE model: the comparison's worker half.

        Builds the frozen TATT configuration with one
        perturbation-theory implementation selected and evaluates
        every comparison point on it. Each evaluation prints its
        theory data vector into vectors_dir, and a block given a
        reference_label measures itself against the vectors a
        previous block left there: at every point it computes

            delta chi2 = delta^T C^-1 delta,
            delta = dv(this block) - dv(reference block),

        with C^-1 the masked inverse covariance from the compiled
        interface. That is exactly the chi2 this block would score
        against a dataset whose data vector IS the reference block's
        prediction at the same point: the reference block's own chi2
        against it is zero, so the number is a pure second-order
        deviation and never rides the slope of the distance to the
        shipped data.

        This function is meant to run inside a FRESH python process
        (see cfastpt_vs_fastpt_chi2s): process isolation is what
        guarantees that nothing computed under the other
        implementation, or under the other accuracy settings,
        survives into this block - cobaya's caches, CAMB's state, and
        the C globals of the compiled cosmolike interface all die
        with their process, so there is no cache to flush by hand.

        Arguments:
          example = a key of the project's EXAMPLES table.
          fastpt  = False evaluates with cfastpt (IA_code 0), True
                    with python FAST-PT (IA_code 1).
          high    = False keeps the frozen default settings; True
                    applies the HIGH_ACCURACY settings on top.
          fastpt_settings = None for the cfastpt block; the fastpt
                    blocks pass FASTPT_LOW_SETTINGS or
                    FASTPT_HIGH_SETTINGS.
          label   = the file-name tag of this block's printed
                    vectors.
          vectors_dir = the directory the per-point vectors are
                    printed into, shared by the three blocks of one
                    sweep.
          reference_label = None to only print vectors, or the label
                    of the block to measure against.
          mask    = a fastpt_masks entry, forwarded to
                    _fastpt_comparison_info (see there): the
                    scale-cut mask this block's dataset carries.

        Returns:
          {"chi2s": the per-point chi2 list against the shipped data
          (informational), "dchi2_vs_reference": the per-point
          delta^T C^-1 delta list or None, "eval_seconds": per-point
          wall-clock seconds (the first entry carries CAMB plus the
          PT tables; the rest are the steady data-vector cost)}.

        Raises:
          AssertionError when the sampled-parameter set differs from
          the frozen point plus the promoted IA parameters;
          RuntimeError when an evaluation did not print its vector.
        """
        import time

        import numpy as np

        require_cocoa_environment()
        code = "FASTPT" if fastpt else "CFASTPT"
        if fastpt_settings is not None:
            # name the fastpt settings by their boost, the one entry
            # the low and high tables differ in
            code += f"(boost {fastpt_settings['accuracyboost']:g})"
        setting = "high accuracy" if high else "default settings"
        print(f"  building model ({example}, TATT, {code}, {setting}, "
              f"mask {mask}) ...", flush=True)
        info = self._fastpt_comparison_info(example, fastpt, high,
                                            fastpt_settings, mask=mask)
        # every evaluation rewrites this one file; the loop below
        # moves it to a per-point name right after each evaluation
        current_path = os.path.join(vectors_dir,
                                    f"{label}_current.modelvector")
        likelihood_block = info["likelihood"][
            self.examples[example]["likelihood"]]
        likelihood_block["print_datavector"] = True
        likelihood_block["print_datavector_file"] = current_path
        model = make_model(info)
        # the frozen point does not carry a promoted parameter (it is
        # fixed in the shipped configuration), so the evaluation base
        # is assembled here: the frozen point plus a zero for each
        # promoted name, checked against the model's sampled set the
        # same way build_point checks the unpromoted configurations
        base = dict(self.load_frozen_point(example))
        for name in self.fastpt_points[0]:
            if name not in base:
                base[name] = 0.0
        sampled = set(model.parameterization.sampled_params())
        if sampled != set(base):
            raise AssertionError(
                "sampled-parameter set differs from the frozen point "
                "plus the promoted IA parameters: model only "
                f"{sorted(sampled - set(base))}; point only "
                f"{sorted(set(base) - sampled)}")
        n_points = len(self.fastpt_points)
        chi2s = []
        eval_seconds = []
        # enumerate pairs each point with a counter; start=1 makes
        # the printed rows read 1..30 instead of 0..29
        for i, ia_values in enumerate(self.fastpt_points, start=1):
            # perf_counter is a monotonic wall clock; the difference
            # of two readings is the elapsed time of the evaluation
            # alone
            started = time.perf_counter()
            # {**base, **ia_values} builds a NEW dictionary: the
            # fiducial entries first, then the five IA values
            # replacing their fiducial counterparts; base itself
            # stays untouched
            chi2 = _evaluate_cached(model, {**base, **ia_values})
            elapsed = time.perf_counter() - started
            chi2s.append(chi2)
            eval_seconds.append(elapsed)
            if not os.path.isfile(current_path):
                raise RuntimeError(
                    f"{code} point {i}: the evaluation printed no "
                    f"data vector at {current_path} (did "
                    "print_datavector's path handling change?)")
            # os.replace moves the freshly printed vector to its
            # per-point name; the next evaluation must write
            # current_path anew, which the isfile check above
            # enforces point by point
            os.replace(current_path,
                       os.path.join(vectors_dir,
                                    f"{label}_point{i:02d}.modelvector"))
            print(f"  {code} point {i:2d}/{n_points}: chi2 = "
                  f"{chi2:.6f}   ({elapsed:.2f} s)", flush=True)
        # the first evaluation carries the one-time work (CAMB plus
        # this implementation's perturbation-theory tables); the mean
        # of the rest is the steady per-point cost. eval_seconds[1:]
        # is the list without its first entry.
        print(f"  {code}: first evaluation {eval_seconds[0]:.2f} s "
              f"(CAMB + PT tables + data vector); later evaluations "
              f"mean {float(np.mean(eval_seconds[1:])):.3f} s",
              flush=True)
        if reference_label is None:
            return {"chi2s": chi2s, "dchi2_vs_reference": None,
                    "eval_seconds": eval_seconds}
        # the compiled interface was initialized by the likelihood
        # build above, so its masked inverse covariance (masked rows
        # and columns zeroed, full data-vector dimensions) is
        # available here; the import resolves because start_cocoa.sh
        # puts the project interface folder on the python path
        import importlib

        ci = importlib.import_module(self.interface_module)

        icov = np.array(ci.get_inv_cov_masked())
        dchi2s = []
        for i in range(1, n_points + 1):
            own = _load_datavector(
                os.path.join(vectors_dir,
                             f"{label}_point{i:02d}.modelvector"))
            ref = _load_datavector(
                os.path.join(vectors_dir,
                             f"{reference_label}_point{i:02d}"
                             ".modelvector"))
            if own.shape != ref.shape or icov.shape[0] != own.shape[0]:
                raise RuntimeError(
                    f"point {i}: vector/covariance shapes disagree "
                    f"({own.shape}, {ref.shape}, {icov.shape})")
            delta = own - ref
            # delta @ icov @ delta is the quadratic form delta^T C^-1
            # delta: the chi2 of the difference vector in units of
            # the statistical error the covariance defines
            dchi2s.append(float(delta @ icov @ delta))
        return {"chi2s": chi2s, "dchi2_vs_reference": dchi2s,
                "eval_seconds": eval_seconds}

    def _run_fastpt_comparison_worker(self, example, fastpt, high,
                                      fastpt_settings, label,
                                      vectors_dir, reference_label,
                                      mask="frozen"):
        """Run one _fastpt_comparison_block in a fresh python subprocess.

        A fresh process is the cache flush: cobaya's component
        caches, CAMB, and the C globals of the compiled cosmolike
        interface start empty in each block, so no block can reuse
        anything another block computed. The child inherits this
        process's environment, sets OMP_NUM_THREADS before its first
        import, adds the project's tests/ to its import path, calls
        _fastpt_comparison_block, and writes the result as json into
        a temporary file the parent reads back; its per-point
        progress lines stream to the same terminal because stdout is
        inherited.

        Arguments:
          example, fastpt, high, fastpt_settings, label, vectors_dir,
          reference_label, mask = forwarded to
          _fastpt_comparison_block.

        Returns:
          the block's result dictionary.

        Raises:
          subprocess.CalledProcessError when the worker fails (its
          traceback already streamed to the terminal); RuntimeError
          when the worker returns the wrong number of chi2 values.
        """
        import subprocess
        import sys

        workdir = tempfile.mkdtemp(prefix="cocoa_fastpt_compare_")
        out_path = os.path.join(workdir, "result.json")
        # the child program, line by line: pin the OpenMP thread
        # count BEFORE any import can load the compiled libraries,
        # make the project's tests/ importable, run the block, dump
        # the result. !r prints each interpolated value as python
        # source, so the generated program is valid python.
        child_code = (
            "import os\n"
            f"os.environ['OMP_NUM_THREADS'] = "
            f"{REQUIRED_OMP_THREADS!r}\n"
            "import json\n"
            "import sys\n"
            f"sys.path.insert(0, {self.tests_dir!r})\n"
            "import cocoa_test_utils as u\n"
            f"result = u._fastpt_comparison_block({example!r}, "
            f"fastpt={fastpt!r}, high={high!r}, "
            f"fastpt_settings={fastpt_settings!r}, "
            f"label={label!r}, vectors_dir={vectors_dir!r}, "
            f"reference_label={reference_label!r}, mask={mask!r})\n"
            f"with open({out_path!r}, 'w') as f:\n"
            "    json.dump(result, f)\n"
        )
        try:
            # sys.executable is this python interpreter; check=True
            # raises when the child exits nonzero (its traceback has
            # already streamed to the inherited stderr)
            subprocess.run([sys.executable, "-c", child_code],
                           check=True)
            with open(out_path) as f:
                result = json.load(f)
        finally:
            # a finally block runs on EVERY exit from the try, so no
            # failure mode leaves the temporary directory behind
            shutil.rmtree(workdir, ignore_errors=True)
        if len(result["chi2s"]) != len(self.fastpt_points):
            raise RuntimeError(
                f"worker returned {len(result['chi2s'])} chi2 values "
                f"for {len(self.fastpt_points)} comparison points")
        return result

    def cfastpt_vs_fastpt_chi2s(self, example, high=False,
                                mask="frozen"):
        """The comparison-point quantities under the three configurations.

        The same 30 hard-coded intrinsic-alignment points evaluated
        three times with everything else identical:

          1. cfastpt (IA_code 0), the reference: its printed data
             vector at each point becomes the fiducial the other
             blocks are measured against (its own chi2 against that
             vector is zero by construction);
          2. python FAST-PT (IA_code 1) at FASTPT_LOW_SETTINGS, the
             pass configuration;
          3. python FAST-PT at FASTPT_HIGH_SETTINGS, the doubled
             boosts.

        Block 2's per-point delta^T C^-1 delta against block 1 is the
        pass/fail quantity; block 3's shows how the deviation
        responds to the FAST-PT grids (advisory). Each configuration
        runs in its own subprocess, one after the other; the
        per-point vectors travel through one shared temporary
        directory that dies with this call.

        Arguments:
          example = a key of the project's EXAMPLES table; the
                    comparison uses "example1".
          high    = False compares at the frozen default
                    camb/cosmolike settings; True repeats all three
                    blocks with the HIGH_ACCURACY settings (the
                    --high=1 command line option of the tests).
          mask    = a fastpt_masks entry, applied to all three
                    blocks: "frozen" (the default) keeps each
                    example's own tatt_dataset; any other name swaps
                    in the matching frozen dataset variant (the
                    --mask command line option of the tests).

        Returns:
          (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
          dchi2_low, dchi2_high): five lists index-aligned with the
          comparison points.

        Raises:
          subprocess.CalledProcessError when a block's worker fails;
          RuntimeError when a worker returns the wrong number of
          chi2 values (_run_fastpt_comparison_worker).
        """
        vectors_dir = tempfile.mkdtemp(prefix="cocoa_fastpt_vectors_")
        try:
            cfastpt = self._run_fastpt_comparison_worker(
                example, fastpt=False, high=high, fastpt_settings=None,
                label="cfastpt", vectors_dir=vectors_dir,
                reference_label=None, mask=mask)
            fastpt_low = self._run_fastpt_comparison_worker(
                example, fastpt=True, high=high,
                fastpt_settings=self.fastpt_low_settings,
                label="fastpt_low", vectors_dir=vectors_dir,
                reference_label="cfastpt", mask=mask)
            fastpt_high = self._run_fastpt_comparison_worker(
                example, fastpt=True, high=high,
                fastpt_settings=self.fastpt_high_settings,
                label="fastpt_high", vectors_dir=vectors_dir,
                reference_label="cfastpt", mask=mask)
        finally:
            # the shared vectors die with the sweep, whether it
            # finished or an exception is on its way out
            shutil.rmtree(vectors_dir, ignore_errors=True)
        return (cfastpt["chi2s"], fastpt_low["chi2s"],
                fastpt_high["chi2s"],
                fastpt_low["dchi2_vs_reference"],
                fastpt_high["dchi2_vs_reference"])
