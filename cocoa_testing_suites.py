"""Factories that build the shared test classes for one project.

Every project's test suite runs the same checks on its own survey:
the test classes are therefore shared code, and this module is where
they live. Each factory takes a project's cocoa_test_utils module
(the shim that binds the shared harness in cocoa_testing.py to that
project's tables) and returns a ready test class closed over it. A
project's test file binds the class to the module-level name pytest
collects:

    import cocoa_test_utils as u
    TestBaryonDrift = u.baryon_drift_suite(u)

The binding changes nothing about collection or test ids: pytest
finds the class through the module attribute exactly as it found the
class statement the factory replaced. The docstrings on the classes
and their methods are the documentation the test reports show, and
keeping them here keeps them identical across the projects.

What each check means physically is documented once in this
repository's README (the "shared data-vector checks" section); the
factories' docstrings describe what runs, not why.
"""

import os
import re
import unittest


def _interface(u):
    """The project's compiled interface module, imported on first use.

    The harness already knows the module's name (the shim passes it to
    CocoaTestHarness); importing through it keeps the factories free
    of project names. The import happens inside the test bodies, so
    collecting the tests never loads the compiled library.

    Arguments:
      u = the project's cocoa_test_utils module.

    Returns:
      the imported interface module.
    """
    import importlib
    return importlib.import_module(u._H.interface_module)


def _chi2_method(u, example, number, label, tatt):
    """Build one chi2-vs-frozen-reference test method.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = the key of u.EXAMPLES the method evaluates.
      number  = the test number as a string, possibly x-prefixed; the
                prefix stays in the method name and drops out of the
                printed report.
      label   = the opening of the printed report line, up to the IA
                model, e.g. "example1 (cosmic shear".
      tatt    = True for the TATT variant of the frozen point.

    Returns:
      the test method, named and documented; example_suite installs
      it on the class it builds.
    """
    ia_key = "tatt" if tatt else "nla"
    ia_label = "TATT" if tatt else "NLA"

    def test(self):
        chi2 = u.single_model_chi2(example, tatt=tatt)
        # reference_chi2.json names each entry <example>_<IA model>
        ref = self.reference[f"{example}_{ia_key}"]
        # report_chi2_test prints chi2, the reference, their
        # difference and the limit; it never asserts (the assertion
        # follows)
        u.report_chi2_test(
            int(number.lstrip("x")),
            f"{label}, {ia_label}) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        # assertLess(a, b) passes when a < b and fails with msg
        # otherwise; in that message, :.6f prints fixed six decimals
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    # the method name carries the test number; pytest and unittest
    # find the method under this name on the built class
    test.__name__ = (f"test_{number}_chi2_matches_frozen_reference"
                     + ("_tatt" if tatt else ""))
    if tatt:
        test.__doc__ = (
            "The chi2 check repeated with the TATT IA model.\n\n"
            "        TATT replaces the NLA point's zero second-order\n"
            "        amplitudes with the nonzero values of TATT_POINT,\n"
            "        so the frozen TATT reference genuinely exercises\n"
            "        the tidal second-order terms.\n        ")
    else:
        test.__doc__ = (
            "chi2 at the frozen NLA point stays within 0.2 of the "
            "reference.\n\n"
            "        u.single_model_chi2 starts one worker, which\n"
            "        builds the cobaya model (CAMB plus the\n"
            "        likelihood) from the frozen configuration, checks\n"
            "        that the frozen point names exactly the\n"
            "        parameters the model samples, and returns\n"
            "        chi2 = -2 ln L at that point. The pass limit\n"
            "        CHI2_TOLERANCE = 0.2 is the project's bound for\n"
            "        every frozen chi2: compiler and library-version\n"
            "        noise stays below it, and no physics change is\n"
            "        detectable below a chi2 shift of 0.2, so a\n"
            "        failure means the code or the frozen inputs\n"
            "        changed the prediction.\n        ")
    return test


def _race_method(u, example, number, label, tatt):
    """Build one ten-in-a-row race-check test method.

    Arguments: as in _chi2_method.

    Returns:
      the test method, named and documented; example_suite installs
      it on the class it builds.
    """
    ia_label = "TATT" if tatt else "NLA"

    def test(self):
        # with one thread no race can occur: a process that did not
        # get the 4 threads set at the top of the test file stops here
        u.assert_omp_threads()
        # the function returns a (fresh, tenth) pair; the assignment
        # unpacks it into the two names
        fresh, tenth = u.ten_in_a_row_chi2(example, tatt=tatt)
        u.report_race_test(
            int(number.lstrip("x")),
            f"{label}, {ia_label}) race check: "
            "10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")

    test.__name__ = (f"test_{number}_no_race_condition_ten_in_a_row"
                     + ("_tatt" if tatt else ""))
    if tatt:
        test.__doc__ = ("The race check repeated with the TATT IA "
                        "model.\n        ")
    else:
        test.__doc__ = (
            "The fiducial as 10th of 10 cosmologies matches a fresh "
            "run.\n\n"
            "        u.ten_in_a_row_chi2 starts one worker that\n"
            "        builds one model and evaluates on it, in order:\n"
            "        the frozen fiducial point (the fresh value), the\n"
            "        nine cosmologies of RACE_PERTURBATIONS, and the\n"
            "        fiducial again, the 10th point of the row.\n"
            "        Correct code gives the same chi2 both times. A\n"
            "        table kept from an earlier cosmology (leaked\n"
            "        state) or two OpenMP threads writing the same\n"
            "        memory (a race) shift the second value.\n"
            "        RACE_TOLERANCE = 1e-4: both values come from the\n"
            "        same code on the same inputs, so only\n"
            "        floating-point noise is allowed, while leaked\n"
            "        state shifts the chi2 by much more.\n        ")
    return test


def example_suite(u, name, example, label, numbers, tatt_numbers=None):
    """Build one frozen-example test class (chi2 + race, maybe TATT).

    Arguments:
      u            = the project's cocoa_test_utils module.
      name         = the class name to build; the caller assigns the
                     class to this same module-level name, which is
                     what pytest collects, so the test ids do not
                     change.
      example      = the key of u.EXAMPLES the class evaluates.
      label        = the opening of every printed report line, up to
                     the IA model, e.g. "example1 (cosmic shear".
      numbers      = the (chi2 test, race test) numbers as strings,
                     e.g. ("1", "2") or ("x11", "x12"); an x prefix
                     only keeps unittest's alphabetical ordering
                     aligned with the numbering (test_11 would sort
                     before test_2) and drops out of the printed
                     reports.
      tatt_numbers = the same pair for the TATT variants, or None for
                     a configuration without them (the cluster
                     combinations use no TATT variant).

    Returns:
      a unittest.TestCase subclass named `name`.
    """
    # each helper returns a ready method; the dict keys must equal
    # the methods' own __name__ for unittest to find and run them
    methods = [
        _chi2_method(u, example, numbers[0], label, tatt=False),
        _race_method(u, example, numbers[1], label, tatt=False),
    ]
    if tatt_numbers is not None:
        methods.append(_chi2_method(u, example, tatt_numbers[0],
                                    label, tatt=True))
        methods.append(_race_method(u, example, tatt_numbers[1],
                                    label, tatt=True))

    @classmethod
    def setUpClass(cls):
        """Move to ROOTDIR, verify the frozen state, load the references."""
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    # the class docstring names the test-number range; int(...) of
    # the x-stripped strings recovers the printed numbers
    spanned = [int(n.lstrip("x"))
               for n in tuple(numbers) + tuple(tatt_numbers or ())]
    doc = f"""Tests {min(spanned)}-{max(spanned)}, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

    # type(name, bases, namespace) builds the class exactly as a
    # class statement would, with the name chosen at run time
    namespace = {"__doc__": doc, "setUpClass": setUpClass}
    for method in methods:
        namespace[method.__name__] = method
    return type(name, (unittest.TestCase,), namespace)


def accuracy_suite(u, name, checks, knob_scan=None, nmodels_example=None):
    """Build the accuracy-advisory class (A-checks, K-scan, N-models).

    Arguments:
      u               = the project's cocoa_test_utils module.
      name            = the class name to build; the caller assigns
                        the class to this same module-level name, so
                        pytest collects unchanged test ids.
      checks          = one (method_name, advisory, example, tatt,
                        label) tuple per A-check: the full collected
                        method name (e.g. "test_a1_cosmic_shear_nla"),
                        the printed advisory label (e.g. "A1"), the
                        key of u.EXAMPLES, the IA-model flag, and the
                        printed report line (e.g. "example1 (cosmic
                        shear, NLA)").
      knob_scan       = the EXAMPLES key the one-knob-at-a-time scan
                        evaluates (the method
                        test_a0_one_knob_at_a_time, reading the
                        "<key>_nla" reference), or None for a project
                        without the scan.
      nmodels_example = an (EXAMPLES key, display label) pair enabling
                        the opt-in N-random-models check
                        test_ax99_nmodels, or None.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestAccuracy(unittest.TestCase):
        """The advisory accuracy checks, one test method each.

        They share one frozen-state verification (setUpClass) and one
        implementation (_accuracy_check), which evaluates a
        configuration in a worker subprocess and prints the report
        block. There is no pass/fail on the size of the delta: how
        much numerical error an analysis tolerates is a judgment
        call, so only a broken evaluation fails.
        """

        # @classmethod is a decorator (a line starting with @ that
        # wraps the function defined below it): it hands the method
        # the class itself (cls), not an instance; unittest calls
        # setUpClass once before the first test of the class
        @classmethod
        def setUpClass(cls):
            """Check the environment and the frozen state once for all tests.

            Runs once, before the first test of the class: it moves
            to ROOTDIR (the Cocoa/ folder; the frozen configurations
            name paths relative to it), verifies every frozen file
            against the SHA-256 manifest (an edited frozen state must
            fail loudly before any physics runs), and loads the
            frozen reference chi2 values.

            Arguments:
              cls = this test class (the classmethod decorator passes
                    the class itself, not an instance).

            Returns:
              nothing; cls.reference holds the dictionary of
              frozen/reference_chi2.json: {"<example>_nla" or
              "<example>_tatt": reference chi2} plus a "_meta" entry.

            Raises:
              RuntimeError when ROOTDIR is not set (start_cocoa.sh
              was not sourced); AssertionError when a frozen file
              differs from the manifest.
            """
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _accuracy_check(self, check, example, tatt, label):
            """Evaluate one configuration at high accuracy and report.

            u.single_model_chi2(..., high_accuracy=True) starts a
            worker subprocess (a fresh python process, so no
            cosmolike state of another model can reach it) that loads
            the frozen configuration, applies the high-accuracy
            settings on top of it, builds the cobaya model, evaluates
            the chi2 at the frozen point, and hands it back as a
            float. The default-settings value is the frozen
            reference, so the printed difference is the numerical
            error the default settings carry.

            Arguments:
              check   = the advisory label (e.g. "A1") for the
                        report.
              example = a key of u.EXAMPLES (exact-physics
                        configurations only).
              tatt    = True evaluates the TATT variant against the
                        TATT-generated data vector, False the NLA
                        one.
              label   = one line naming the probe and IA model.

            Returns:
              nothing; the delta is printed and nothing is asserted.
            """
            chi2_high = u.single_model_chi2(example, tatt,
                                            high_accuracy=True)
            # `"tatt" if tatt else "nla"` picks the reference-key
            # suffix, the naming the frozen reference file uses
            suffix = "tatt" if tatt else "nla"
            default_ref = self.reference[f"{example}_{suffix}"]
            u.report_accuracy(f"{check}: {label}", chi2_high,
                              default_ref)

    # the A-check methods: one small closure per roster row, named
    # by the row's method_name so the collected ids do not change
    def make_check(check_name, advisory, example, tatt, label):
        def test(self):
            self._accuracy_check(advisory, example, tatt, label)
        test.__name__ = check_name
        test.__doc__ = (f"{advisory}: {label}, default vs high "
                        "accuracy.\n        ")
        return test

    for row in checks:
        method = make_check(*row)
        setattr(TestAccuracy, method.__name__, method)

    if knob_scan is not None:
        def test_a0_one_knob_at_a_time(self):
            """K scan: each accuracy knob alone (check A0).

            Advisory: each knob's chi2 and its difference to the
            frozen default reference print as the scan runs, on the
            NLA configuration the project binds for the scan. A knob
            whose delta rivals the all-knobs delta is the driver; a
            knob whose delta explodes (orders of magnitude beyond the
            others) signals an interface breakdown, not a numerics
            improvement.
            """
            default_ref = self.reference[f"{knob_scan}_nla"]
            print("", flush=True)
            # each entry is (label, likelihood overrides, camb
            # overrides); the two _ discard the overrides, only the
            # label is needed here (the worker looks the knob up
            # again by its label)
            for label, _, _ in u.ACCURACY_KNOBS:
                chi2 = u.single_model_chi2(knob_scan, False,
                                           knob=label)
                u.report_knob(label, chi2, default_ref)

        TestAccuracy.test_a0_one_knob_at_a_time = \
            test_a0_one_knob_at_a_time

    if nmodels_example is not None:
        nm_example, nm_label = nmodels_example

        def test_ax99_nmodels(self):
            """N-random-models check across the prior. Opt-in, advisory.

            The A-checks measure the numerical error of the default
            settings at the one frozen fiducial point. This check
            measures it at N reproducible random points across the
            prior instead. For each point a synthetic data vector is
            generated at that point with the default settings, so the
            default chi2 against it is zero by construction and the
            high-accuracy chi2 against it is the delta directly (the
            mechanism lives in u.random_model_accuracy). Nothing
            fails on a large delta; the deltas only have to be
            finite.

            Opt-in because each model costs a default
            build+evaluation plus a high-accuracy build+evaluation,
            minutes per model. Enable it with the
            COCOA_ACCURACY_NMODELS environment variable, for example

                COCOA_ACCURACY_NMODELS=10 python -m pytest \\
                    <this project's test_accuracy.py> -k nmodels

            or with --nmodels N when the file runs as a script. The
            ax99 in the method name sorts this check after the
            A-checks (unittest runs methods in name order), so the
            cheap fiducial checks always report before the expensive
            multi-model loop starts.
            """
            import math
            # .get returns its second argument, "0", when the
            # variable is unset, so an untouched environment means a
            # clean skip
            n_models_text = os.environ.get(
                "COCOA_ACCURACY_NMODELS", "0")
            # int() raises ValueError on non-numeric text; the except
            # turns the stack trace into an instruction
            try:
                n_models = int(n_models_text)
            except ValueError:
                raise AssertionError(
                    f"COCOA_ACCURACY_NMODELS={n_models_text!r} is "
                    "not an integer; set it to the number of random "
                    "models, for example COCOA_ACCURACY_NMODELS=10")
            if n_models <= 0:
                # the default: report how to enable the check and
                # pass without building anything
                print("\n  N-random-models check skipped (set "
                      "COCOA_ACCURACY_NMODELS=<N>, or run this file "
                      "with --nmodels <N>, to enable it).",
                      flush=True)
                return
            print(f"\n  N-random-models check: {n_models} models, "
                  f"{nm_label}, seed base {u.RANDOM_MODEL_SEED}",
                  flush=True)
            deltas = u.random_model_accuracy(nm_example, n_models)
            # enumerate pairs each delta with its position, counted
            # from 0, so a broken model is named by number
            for index, delta in enumerate(deltas):
                # advisory: the size of the delta is a judgment
                # call, but a non-finite one means the evaluation
                # itself broke
                self.assertTrue(
                    math.isfinite(delta),
                    msg=f"model {index}: non-finite delta chi2 "
                        f"{delta!r}")
            u.report_random_model_summary(deltas)

        TestAccuracy.test_ax99_nmodels = test_ax99_nmodels

    # rename the class last; type(name, ...) is not needed because
    # the class statement above already built it, so only the two
    # name attributes change
    TestAccuracy.__name__ = name
    TestAccuracy.__qualname__ = name
    return TestAccuracy


def fastpt_reference_suite(u, name, cases):
    """Build the FASTPT frozen-reference class (TATT via python FAST-PT).

    Arguments:
      u     = the project's cocoa_test_utils module.
      name  = the class name to build (assigned by the caller to the
              same module-level name, so test ids do not change).
      cases = one (method_name, number, example, label) tuple per
              check: the collected method name, the printed test
              number, the key of u.EXAMPLES (one that carries a
              frozen FASTPT reference), and the full report line,
              e.g. "example1 (cosmic shear, TATT+FASTPT) chi2 vs
              frozen reference".

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestFastptReference(unittest.TestCase):
        """The FASTPT TATT checks, sharing one frozen-state verification.

        Each check evaluates the frozen TATT point with the python
        FAST-PT package (IA_code: 1) instead of the built-in cfastpt,
        compares the chi2 with its own frozen FASTPT reference under
        the usual CHI2_TOLERANCE rule, and prints the frozen
        FASTPT-minus-CFASTPT difference next to it, so a numerics
        change in either implementation is visible at a glance.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _run_fastpt_case(self, number, example, label):
            """Evaluate one FASTPT TATT case against its frozen reference.

            Arguments:
              number  = the test number shown in the report header.
              example = a key of u.EXAMPLES.
              label   = one line naming the example and probe.

            Returns:
              nothing; the assertion fails the test when
              |chi2 - ref| >= CHI2_TOLERANCE.
            """
            chi2 = u.single_model_chi2(example, tatt=True, fastpt=True)
            ref = self.reference[f"{example}_tatt_fastpt"]
            cfastpt_ref = self.reference[f"{example}_tatt"]
            u.report_fastpt_test(number, label, chi2, ref,
                                 cfastpt_ref, u.CHI2_TOLERANCE)
            frozen_diff = self.reference[
                f"{example}_fastpt_minus_cfastpt"]
            # the + in :+.6f forces a sign, so the direction of the
            # frozen difference always shows
            print("  frozen FASTPT - CFASTPT difference = "
                  f"{frozen_diff:+.6f}", flush=True)
            self.assertLess(
                abs(chi2 - ref), u.CHI2_TOLERANCE,
                msg=f"FASTPT chi2 = {chi2:.6f} vs frozen reference "
                    f"{ref:.6f} (|delta| >= {u.CHI2_TOLERANCE})")

    def make_case(method_name, number, example, label):
        def test(self):
            self._run_fastpt_case(number, example, label)
        test.__name__ = method_name
        test.__doc__ = (f"{example} TATT with python FAST-PT stays "
                        "on its reference.\n        ")
        return test

    for row in cases:
        method = make_case(*row)
        setattr(TestFastptReference, method.__name__, method)
    TestFastptReference.__name__ = name
    TestFastptReference.__qualname__ = name
    return TestFastptReference


def fastpt_sweep_suite(u, name, cases):
    """Build the CFASTPT-vs-FASTPT sweep class.

    Arguments:
      u     = the project's cocoa_test_utils module.
      name  = the class name to build (assigned by the caller to the
              same module-level name, so test ids do not change).
      cases = one (method_name, number, example, label) tuple per
              sweep: the collected method name, the printed test
              number, the key of u.EXAMPLES, and the opening of the
              report line up to the IA model, e.g. "example2 (3x2pt".

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestFastptSweep(unittest.TestCase):
        """The direct CFASTPT-vs-FASTPT comparison, one sweep per probe.

        Each sweep evaluates the project's hard-coded points across
        the intrinsic-alignment prior (u.FASTPT_COMPARISON_POINTS,
        cosmology fixed at the frozen fiducial) three times: with
        cfastpt, with FASTPT at the converged two-grid defaults, and
        with FASTPT at the doubled boosts. The CFASTPT vector is the
        fiducial of each point, so the pass rule is the chi2 of the
        FASTPT(low) vector against it (delta^T C^-1 delta), under
        u.FASTPT_COMPARISON_TOLERANCE; FASTPT(high)'s deviation
        prints as the advisory grid response. The sweeps read the
        --high and --mask options (conftest.py). setUpClass loads no
        reference chi2: these tests compare the two implementations
        against each other, so the frozen state only supplies the
        configuration and the data files.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

    def make_sweep(method_name, number, example, label):
        def test(self):
            # the conftest copies the --high command line option into
            # this variable; .get with the "0" default keeps a run
            # outside pytest on the default settings unless the
            # variable is exported by hand
            high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
            setting = "high accuracy" if high else "default settings"
            mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
            (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
             dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
                example, high=high, mask=mask)
            largest = u.report_fastpt_comparison(
                number,
                f"{label}, TATT, camb/cosmolike {setting}, "
                f"mask {mask}): CFASTPT vs FASTPT at 30 "
                "hard-coded points",
                chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
                dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
            self.assertLess(
                largest, u.FASTPT_COMPARISON_TOLERANCE,
                msg="max chi2 of the FASTPT(low)-vs-CFASTPT "
                    f"data-vector difference = {largest:.6f} over "
                    f"the comparison points ({setting}); limit "
                    f"{u.FASTPT_COMPARISON_TOLERANCE}")
        test.__name__ = method_name
        test.__doc__ = (f"{example}: cfastpt and FASTPT agree at the "
                        "30 IA points.\n        ")
        return test

    for row in cases:
        method = make_sweep(*row)
        setattr(TestFastptSweep, method.__name__, method)
    TestFastptSweep.__name__ = name
    TestFastptSweep.__qualname__ = name
    return TestFastptSweep


def ee2_suite(u, name, race, original=None):
    """Build the EuclidEmulator2 class (race check, maybe original).

    Arguments:
      u        = the project's cocoa_test_utils module.
      name     = the class name to build (assigned by the caller to
                 the same module-level name, so test ids do not
                 change).
      race     = a (method_name, number, example, label) tuple for
                 the EE2 race check, label up to " race check", e.g.
                 "example1 (cosmic shear, NLA+EE2)".
      original = a (method_name, number, example) tuple enabling the
                 comparison of the installed, modified EE2 against
                 the pre-modification build (u.EE2_ORIGINAL_COMMIT),
                 or None for a project without that check.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestEE2(unittest.TestCase):
        """The EuclidEmulator2 checks, sharing one frozen-state verification.

        setUpClass loads no reference chi2: the original-vs-modified
        check compares the two EE2 builds against each other and the
        race check compares one build against itself, so the frozen
        state only supplies the configuration and the data files.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

    race_name, race_number, race_example, race_label = race

    def race_test(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(race_example, tatt=False,
                                           ee2=True)
        u.report_race_test(
            race_number,
            f"{race_label} race check: 10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")

    race_test.__name__ = race_name
    setattr(TestEE2, race_name, race_test)

    if original is not None:
        orig_name, orig_number, orig_example = original

        def original_test(self):
            """The modified EE2 reproduces the original's data vectors.

            The per-cosmology delta chi2 of the pre-modification
            build against the installed one, at the ten shared
            cosmologies, must stay below CHI2_TOLERANCE (0.2), the
            band of the reference tests.
            """
            try:
                dchi2s = u.ee2_original_vs_cocoa_dchi2s(orig_example)
            except RuntimeError as error:
                # a shallow local clone cannot supply the original
                # commit; that is an installation limitation, not an
                # EE2 regression
                self.skipTest(str(error))
            print(f"\n==== test {orig_number}: modified EE2 vs "
                  f"original EE2 ({orig_example}, NLA, 10 "
                  "cosmologies) ====", flush=True)
            for i, d in enumerate(dchi2s, start=1):
                print(f"  cosmology {i:2d}: dchi2 = {d:.8f}",
                      flush=True)
            largest = max(dchi2s)
            print(f"  max dchi2 = {largest:.8f}   (limit: < "
                  f"{u.CHI2_TOLERANCE})", flush=True)
            self.assertLess(
                largest, u.CHI2_TOLERANCE,
                msg=f"max dchi2 of the original-vs-modified EE2 data "
                    f"vectors = {largest:.8f} over the ten "
                    f"cosmologies (limit {u.CHI2_TOLERANCE})")

        original_test.__name__ = orig_name
        setattr(TestEE2, orig_name, original_test)

    TestEE2.__name__ = name
    TestEE2.__qualname__ = name
    return TestEE2


def nonlinear_suite(u, name, cases):
    """Build the Halofit-vs-EuclidEmulator2 advisory class.

    Arguments:
      u     = the project's cocoa_test_utils module.
      name  = the class name to build (assigned by the caller to the
              same module-level name, so test ids do not change).
      cases = one (method_name, advisory, example, label) tuple per
              check: the collected method name, the printed advisory
              label (e.g. "NL1"), the key of u.EXAMPLES, and the
              opening of the report line up to the IA model, e.g.
              "example1 (cosmic shear".

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestNonlinear(unittest.TestCase):
        """Halofit scored against EE2, one advisory check per probe.

        Each check evaluates the probe's data vector with both
        nonlinear-P(k) sources at the ten fixed cosmologies of
        u.NONLINEAR_COMPARISON_POINTS and reports, per cosmology, the
        chi2 of the Halofit vector against the EE2 vector (the EE2
        vector is that cosmology's fiducial, so the baseline is zero
        by construction). Advisory: the printed report is the
        product, and the only assertion is structural. The checks
        read the --mask option (conftest.py). setUpClass loads no
        reference chi2: the frozen state only supplies the
        configuration and the data files.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

    def make_check(method_name, advisory, example, label):
        def test(self):
            mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
            dchi2s = u.halofit_vs_ee2_dchi2s(example, mask=mask)
            u.report_nonlinear_comparison(
                f"{advisory}: {label}, NLA, mask {mask}): HALOFIT "
                "vs EE2 at 10 fixed cosmologies", dchi2s)
            self.assertEqual(len(dchi2s),
                             len(u.NONLINEAR_COMPARISON_POINTS))
        test.__name__ = method_name
        test.__doc__ = (f"{advisory}: Halofit scored against EE2 at "
                        "ten cosmologies, advisory.\n        ")
        return test

    for row in cases:
        method = make_check(*row)
        setattr(TestNonlinear, method.__name__, method)
    TestNonlinear.__name__ = name
    TestNonlinear.__qualname__ = name
    return TestNonlinear


def baryon_accuracy_suite(u):
    """Build the BF0-BF7 baryonic-feedback accuracy class for a project.

    Arguments:
      u = the project's cocoa_test_utils module. The class reads its
          BARYON_METHODS labels, ACCURACY_KNOBS table and
          baryon_accuracy_delta through it, so the returned class is
          bound to that project's frozen state and tables.

    Returns:
      a unittest.TestCase subclass; the caller assigns it to the name
      TestBaryonAccuracyAdvisory so pytest collects it with the same
      test ids as before the factory existed.
    """

    class TestBaryonAccuracyAdvisory(unittest.TestCase):
        """Advisory checks BF0-BF7: accuracy with baryonic feedback on.

        Each check evaluates this project's example1 configuration
        (cosmic shear, NLA) with the bfmt theory block computing one
        feedback method, and reports

            delta chi2 = chi2(high accuracy) - chi2(default)

        against a vector the default-settings model generated on the
        fly (so the default chi2 is zero by construction and nothing
        is stored in frozen/). The delta is advisory: only a
        non-finite value fails. BF0 additionally runs the
        one-knob-at-a-time scan with the Akino SP(k) method on, so a
        large delta names the knob causing it.

        setUpClass runs once: it moves to ROOTDIR and verifies every
        frozen file against the SHA-256 manifest before any physics
        runs.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _baryon_accuracy_check(self, name, baryon, label):
            """Print the default-vs-high-accuracy delta with one method on.

            Arguments:
              name   = the advisory label (BF1-BF7) for the report.
              baryon = a label of cocoa_test_utils.BARYON_METHODS.
              label  = one line naming the feedback method.

            Returns:
              nothing; only a non-finite delta fails the check.
            """
            # the default chi2 is zero by construction (the default
            # model produced the very vector it is compared with), so
            # the pushed evaluation's chi2 is the delta; only that is
            # printed
            delta = u.baryon_accuracy_delta(baryon)
            # delta == delta is False only for NaN (NaN compares
            # unequal to itself); with the infinity check, the
            # assertion demands a finite delta
            self.assertTrue(
                delta == delta and abs(delta) != float("inf"),
                f"{name}: non-finite delta")
            print(f"""
{'-' * 66}
ACCURACY: {name}: {label}
  delta chi2 (high-default) = {delta:+.6f}
{'-' * 66}""", flush=True)

        def test_bf0_one_knob_at_a_time(self):
            """BF0: each accuracy knob alone, Akino SP(k) feedback on.

            Advisory: the same K-scan as test_accuracy.py, with the
            bfmt block computing the Akino SP(k) suppression and the
            chi2 measured against that method's own generated vector.
            A knob whose delta rivals the all-knobs delta of BF2 is
            the driver of the numerical error under feedback.
            """
            print("", flush=True)
            # each entry is (label, likelihood overrides, camb
            # overrides); the two _ discard the tables, which
            # baryon_accuracy_delta looks up again by label
            for label, _, _ in u.ACCURACY_KNOBS:
                # each knob's chi2 against the on-the-fly vector is
                # its delta (the default against that vector is zero)
                delta = u.baryon_accuracy_delta("spk akino", knob=label)
                print(f"  KNOB {label:30s} delta chi2 = {delta:+12.6f}",
                      flush=True)

        def test_bf1_spk_power_law(self):
            """BF1: SP(k) with the power-law fb relation."""
            self._baryon_accuracy_check("BF1", "spk power law",
                                        "SP(k), power-law fb relation")

        def test_bf2_spk_akino(self):
            """BF2: SP(k) with the Akino et al. 2022 fb relation."""
            self._baryon_accuracy_check("BF2", "spk akino",
                                        "SP(k), Akino et al. 2022")

        def test_bf3_spk_double_power_law(self):
            """BF3: SP(k) with the double power-law fb relation."""
            self._baryon_accuracy_check("BF3", "spk double power law",
                                        "SP(k), double power-law fb relation")

        def test_bf4_bcemu(self):
            """BF4: BCEmu."""
            self._baryon_accuracy_check("BF4", "bcemu", "BCEmu")

        def test_bf5_flamingo(self):
            """BF5: FlamingoBaryonResponseEmulator."""
            self._baryon_accuracy_check("BF5", "flamingo",
                                        "FlamingoBaryonResponseEmulator")

        def test_bf6_baccoemu(self):
            """BF6: BACCOemu."""
            self._baryon_accuracy_check("BF6", "baccoemu", "BACCOemu")

        def test_bf7_bcemu2025(self):
            """BF7: BCemu2025."""
            self._baryon_accuracy_check("BF7", "bcemu2025", "BCemu2025")

    return TestBaryonAccuracyAdvisory


def baryon_drift_suite(u):
    """Build the BD1-BD7 baryonic-feedback drift class for a project.

    Arguments:
      u = the project's cocoa_test_utils module. The class reads its
          BARYON_METHODS labels, CHI2_TOLERANCE and baryon_drift_chi2
          through it, so the returned class is bound to that
          project's frozen state and tables.

    Returns:
      a unittest.TestCase subclass; the caller assigns it to the name
      TestBaryonDrift so pytest collects it with the same test ids as
      before the factory existed.
    """

    class TestBaryonDrift(unittest.TestCase):
        """Drift tests BD1-BD7: the feedback pipeline against its freeze.

        Each test evaluates this project's example1 configuration
        (cosmic shear, NLA) with the bfmt theory block computing one
        feedback method, against that method's frozen data vector
        (written at freeze time by generate_frozen_reference.py
        --baryons, when the chi2 against it was zero by
        construction). The assertion chi2 <= CHI2_TOLERANCE therefore
        pins the whole feedback pipeline: a failure means cosmolike
        or the bfmt theory block changed its prediction since the
        freeze. The accuracy checks (BF) regenerate their vector per
        run and can never see drift; these tests hold the frozen
        vector still, so they measure drift and nothing else.

        setUpClass runs once: it moves to ROOTDIR and verifies every
        frozen file against the SHA-256 manifest before any physics
        runs.
        """

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _baryon_drift_check(self, name, baryon, label):
            """Check one method's chi2 against its frozen feedback vector.

            Arguments:
              name   = the test label (BD1-BD7) for the report.
              baryon = a label of cocoa_test_utils.BARYON_METHODS.
              label  = one line naming the feedback method.

            Returns:
              nothing; the assertion fails when the chi2 exceeds
              CHI2_TOLERANCE.
            """
            chi2 = u.baryon_drift_chi2(baryon)
            print(f"""
{'-' * 66}
DRIFT: {name}: {label}
  chi2 against the frozen feedback vector = {chi2:.6f}
  (zero at freeze time; tolerance {u.CHI2_TOLERANCE})
{'-' * 66}""", flush=True)
            self.assertLessEqual(
                chi2, u.CHI2_TOLERANCE,
                f"{name}: chi2 {chi2:.6f} exceeds the tolerance "
                f"{u.CHI2_TOLERANCE}; cosmolike or the bfmt theory "
                "block changed its prediction since the freeze")

        def test_bd1_spk_power_law(self):
            """BD1: SP(k) with the power-law fb relation."""
            self._baryon_drift_check("BD1", "spk power law",
                                     "SP(k), power-law fb relation")

        def test_bd2_spk_akino(self):
            """BD2: SP(k) with the Akino et al. 2022 fb relation."""
            self._baryon_drift_check("BD2", "spk akino",
                                     "SP(k), Akino et al. 2022")

        def test_bd3_spk_double_power_law(self):
            """BD3: SP(k) with the double power-law fb relation."""
            self._baryon_drift_check("BD3", "spk double power law",
                                     "SP(k), double power-law fb relation")

        def test_bd4_bcemu(self):
            """BD4: BCEmu."""
            self._baryon_drift_check("BD4", "bcemu", "BCEmu")

        def test_bd5_flamingo(self):
            """BD5: FlamingoBaryonResponseEmulator."""
            self._baryon_drift_check("BD5", "flamingo",
                                     "FlamingoBaryonResponseEmulator")

        def test_bd6_baccoemu(self):
            """BD6: BACCOemu."""
            self._baryon_drift_check("BD6", "baccoemu", "BACCOemu")

        def test_bd7_bcemu2025(self):
            """BD7: BCemu2025."""
            self._baryon_drift_check("BD7", "bcemu2025", "BCemu2025")

    return TestBaryonDrift


# ---- the interface-level families (in-process cache checks) -----------------

# The photo-z convention sweep: (report tag,
# photoz_interpolation_type, photoz_zmid_convention). The default, each
# alternative, and the default again.
PHOTOZ_SETTINGS = (
    ("cspline/Z_LOW (default)", 0, 0),
    ("linear", 1, 0),
    ("steffen", 2, 0),
    ("Z_MID", 0, 1),
    ("default again (round trip)", 0, 0),
)

# The alternatives must be seen (a stale n(z) cache would give exactly
# zero); the floors are orders of magnitude below the measured deltas,
# so they only catch a dead flag, never normal numerical drift. Z_MID
# shifts every distribution by dz/2, hence its larger floor.
PHOTOZ_DCHI2_FLOORS = {"linear": 1.0e-8, "steffen": 1.0e-8,
                       "Z_MID": 1.0e-2}


def photoz_conventions_suite(u, name, example="example1"):
    """Build the runtime photo-z convention sweep class.

    The likelihood exposes two runtime knobs that control how the
    n(z) table files are turned into the smooth distributions the
    Limber integrals consume: photoz_interpolation_type (the stage-1
    interpolant) and photoz_zmid_convention (how the z column of the
    n(z) file is read). The class evaluates the frozen cosmic-shear
    fiducial under the five PHOTOZ_SETTINGS in one process, measures
    each alternative's delta chi2 = delta^T C^-1 delta against the
    default vector, and asserts the dead-flag floors, the frozen
    reference, and the bitwise round trip back to the default.
    Running everything in one process is the point: the n(z) caches
    inside the compiled interface must notice a runtime flag change
    and rebuild.

    Arguments:
      u       = the project's cocoa_test_utils module.
      name    = the class name to build (assigned by the caller to
                the same module-level name, so test ids do not
                change).
      example = the key of u.EXAMPLES the sweep evaluates.

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestPhotozConventions(unittest.TestCase):
        """The five-setting sweep, sharing one frozen-state verification."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def test_photoz_conventions(self):
            """Every alternative is seen, and the default round trip is exact."""
            import shutil
            import tempfile

            import numpy as np
            ci = _interface(u)

            vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_")
            # addCleanup registers a call unittest makes after the
            # test, pass or fail: here it removes the temporary folder
            self.addCleanup(shutil.rmtree, vectors_dir,
                            ignore_errors=True)

            results = {}
            icov = None
            for tag, interp, zmid in PHOTOZ_SETTINGS:
                print(f"  building model ({example}, NLA, {tag}) ...",
                      flush=True)
                info = u.load_frozen_info(example, tatt=False)
                likelihood_block = \
                    info["likelihood"][u.EXAMPLES[example]["likelihood"]]
                likelihood_block["photoz_interpolation_type"] = interp
                likelihood_block["photoz_zmid_convention"] = zmid
                vector_path = os.path.join(
                    vectors_dir, f"dv_{interp}_{zmid}.modelvector")
                likelihood_block["print_datavector"] = True
                likelihood_block["print_datavector_file"] = vector_path
                model = u.make_model(info)
                point = u.build_point(model, example, tatt=False)
                chi2 = u.evaluate_chi2(model, point)
                if not os.path.isfile(vector_path):
                    raise RuntimeError(
                        f"print_datavector wrote no file at "
                        f"{vector_path}")
                results[tag] = (chi2, u._load_datavector(vector_path))
                if icov is None:
                    # all five settings share one dataset (mask,
                    # covariance), so the masked inverse covariance of
                    # the first build serves every comparison
                    icov = np.array(ci.get_inv_cov_masked())

            chi2_default, dv_default = results[PHOTOZ_SETTINGS[0][0]]
            reference_key = f"{example}_nla"

            print(f"\n  chi2 report ({example}, NLA):")
            print(f"    default: chi2 = {chi2_default:.6f} "
                  f"(frozen reference "
                  f"{self.reference[reference_key]:.6f})")
            for tag in ("linear", "steffen", "Z_MID"):
                chi2, dv = results[tag]
                delta = dv - dv_default
                dchi2 = float(delta @ icov @ delta)
                print(f"    {tag:8s}: chi2 = {chi2:.6f}, "
                      f"delta^T C^-1 delta vs default = {dchi2:.6e}")
                self.assertGreater(
                    dchi2, PHOTOZ_DCHI2_FLOORS[tag],
                    f"{tag}: the flag change was not seen by the n(z) "
                    "cache (delta chi2 at or below the dead-flag "
                    "floor)")

            # the default must agree with the frozen reference within
            # the band the reference tests demand
            self.assertLess(
                abs(chi2_default - self.reference[reference_key]),
                u.CHI2_TOLERANCE)

            # round trip: after all the flips, the default settings
            # must reproduce the first vector identically: the cache
            # rebuilt back to the same state
            _, dv_return = results[PHOTOZ_SETTINGS[-1][0]]
            self.assertTrue(
                np.array_equal(dv_return, dv_default),
                "returning to the default settings did not reproduce "
                "the default data vector bit for bit; the n(z) cache "
                "did not rebuild cleanly on the way back")

    TestPhotozConventions.__name__ = name
    TestPhotozConventions.__qualname__ = name
    return TestPhotozConventions


def nonlimber_suite(u, name, kind, measured, default=0, rtol=0.05,
                    floor=1.0e-6, example="example2"):
    """Build one Limber-vs-non-Limber class (clustering or ggl).

    The likelihood key adopt_limber_gg (clustering) or adopt_limber_gs
    (galaxy-galaxy lensing) chooses between the exact low-l projection
    and the Limber approximation at every multipole. The class
    evaluates the frozen fiducial three times in one process (the
    default, the other setting, the default again) and asserts: the
    delta is seen (dead-flag floor), only the probe's own block
    changes, the round trip reproduces the first vector bitwise, the
    delta chi2 matches the project's measured value to the relative
    band, and the default reproduces the frozen reference (last, so a
    stale snapshot cannot hide the other checks).

    Arguments:
      u        = the project's cocoa_test_utils module.
      name     = the class name to build (assigned by the caller to
                 the same module-level name, so test ids do not
                 change).
      kind     = "gg" (clustering, adopt_limber_gg, the block after
                 cosmic shear and ggl) or "ggl" (galaxy-galaxy
                 lensing, adopt_limber_gs, the block after cosmic
                 shear).
      measured = the delta chi2 measured for this project's frozen
                 fiducial (survey data).
      default  = the project's adopt_limber_* default (0 non-Limber,
                 1 Limber).
      rtol     = the relative band around `measured`.
      floor    = the dead-flag floor on the delta chi2.
      example  = the key of u.EXAMPLES the sweep evaluates.

    Returns:
      a unittest.TestCase subclass named `name`.
    """
    flag_key = {"gg": "adopt_limber_gg", "ggl": "adopt_limber_gs"}[kind]
    probe = {"gg": "clustering", "ggl": "ggl"}[kind]

    # (report tag, flag value): the default, the other setting, the
    # default again
    names = {0: "non-Limber", 1: "Limber"}
    settings = (
        (f"{names[default]} (default)", default),
        (names[1 - default], 1 - default),
        (f"{names[default]} again (round trip)", default),
    )

    class TestNonLimber(unittest.TestCase):
        """Limber vs non-Limber on the frozen fiducial."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR, verify the frozen state, load the references."""
            u.require_cocoa_environment()
            u.verify_frozen()
            cls.reference = u.load_reference()

        def _run_sweep(self):
            """Evaluate the three settings and assert the five checks."""
            import time

            import numpy as np
            ci = _interface(u)

            reference_key = f"{example}_nla"
            vectors = {}
            chi2s = {}
            icov = None
            sizes = None
            nlen = None
            pairs = None
            for tag, flag in settings:
                print(f"  building model ({example}, NLA, {tag}) ...",
                      flush=True)
                info = u.load_frozen_info(example, tatt=False)
                lk_name = u.EXAMPLES[example]["likelihood"]
                info["likelihood"][lk_name][flag_key] = flag
                model = u.make_model(info)
                point = u.build_point(model, example, tatt=False)
                start = time.perf_counter()
                chi2s[tag] = u.evaluate_chi2(model, point)
                elapsed = time.perf_counter() - start
                print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                      "first evaluation of the model)", flush=True)
                # full-precision model vector (the printed one keeps
                # 9 digits)
                vectors[tag] = np.array(ci.compute_data_vector_masked())
                if icov is None:
                    # all settings share one data set (mask,
                    # covariance), so the first build's masked inverse
                    # covariance serves all
                    icov = np.array(ci.get_inv_cov_masked())
                    like = model.likelihood[lk_name]
                    if hasattr(ci,
                               "compute_data_vector_3x2pt_real_sizes"):
                        sizes = ci.compute_data_vector_3x2pt_real_sizes()
                        nlen = int(like.ntheta)
                    else:
                        sizes = \
                            ci.compute_data_vector_3x2pt_fourier_sizes()
                        nlen = int(like.ncl)
                    if kind == "ggl":
                        # the kept (lens, source) pairs, lens bin
                        # outer, for the per-pair report labels
                        excluded = {(int(zl), int(zs)) for zl, zs in
                                    (getattr(like, "ggl_exclude", None)
                                     or [])}
                        pairs = [(zl, zs)
                                 for zl in range(int(like.lens_ntomo))
                                 for zs in range(int(like.source_ntomo))
                                 if (zl, zs) not in excluded]

            # tags = {flag value: report tag} of the first two
            # settings, so tags[0] names the non-Limber run and
            # tags[1] the Limber one (the delta enters quadratically,
            # so its sign never matters)
            tags = {flag: tag for tag, flag in settings[:2]}
            dv_default = vectors[settings[0][0]]
            delta = vectors[tags[0]] - vectors[tags[1]]
            dchi2 = float(delta @ icov @ delta)

            if kind == "gg":
                # the clustering block follows cosmic shear and
                # galaxy-galaxy lensing in every probe combination:
                # entries [b0, b1), one block of nlen entries per
                # lens bin
                b0 = int(sizes[0]) + int(sizes[1])
                b1 = b0 + int(sizes[2])
                nblocks = int(sizes[2]) // nlen
                block_label = "lens bin"
                per_block_note = "per lens bin (the bin's block alone):"
            else:
                # the ggl block follows cosmic shear: entries
                # [b0, b1), one block of nlen entries per kept
                # (lens, source) pair
                b0 = int(sizes[0])
                b1 = b0 + int(sizes[1])
                nblocks = int(sizes[1]) // nlen
                block_label = "pair"
                per_block_note = ("per lens-source pair (the pair's "
                                  "block alone):")

            print(f"\n  delta chi2 report ({example}, NLA):")
            print(f"    {settings[0][0]}: chi2 = "
                  f"{chi2s[settings[0][0]]:.6f} "
                  f"(frozen reference "
                  f"{self.reference[reference_key]:.6f})")
            print(f"    {settings[1][0]}: chi2 = "
                  f"{chi2s[settings[1][0]]:.6f}")
            print(f"    delta^T C^-1 delta = {dchi2:.4f} "
                  f"(measured {measured:.4f})")
            print(f"    {per_block_note}")
            # per-block contributions, printed largest first; the
            # loop stops at the first one below 0.1% of the total
            rows = []
            for b in range(nblocks):
                block = np.zeros_like(delta)
                sl = slice(b0 + b*nlen, b0 + (b + 1)*nlen)
                block[sl] = delta[sl]
                rows.append((float(block @ icov @ block), b))
            for contribution, b in sorted(rows, reverse=True):
                if contribution < 1.0e-3*max(dchi2, floor):
                    break
                if kind == "ggl" and pairs is not None \
                        and len(pairs) == nblocks:
                    label = f"(lens {pairs[b][0]}, source {pairs[b][1]})"
                    print(f"      {label:24s} {contribution:.4f}")
                else:
                    print(f"      {block_label} {b:<14d} "
                          f"{contribution:.4f}")

            self.assertGreater(
                dchi2, floor,
                f"non-Limber {probe} did not change the data vector: "
                f"the {flag_key} flag did not reach the C code, or "
                f"the {probe} cache did not rebuild")

            outside = np.concatenate((delta[:b0], delta[b1:]))
            self.assertTrue(
                np.all(outside == 0.0),
                f"entries outside the {probe} block changed with "
                f"{flag_key}")

            self.assertTrue(
                np.array_equal(vectors[settings[-1][0]], dv_default),
                "returning to the default did not reproduce the first "
                f"data vector bit for bit; the {probe} cache did not "
                "rebuild cleanly")

            self.assertLess(
                abs(dchi2/measured - 1.0), rtol,
                f"delta chi2 = {dchi2:.4f} differs from the measured "
                f"{measured:.4f} by more than {rtol:.0%}")

            # last, so that a stale frozen snapshot does not hide the
            # four checks above
            self.assertLess(
                abs(chi2s[settings[0][0]]
                    - self.reference[reference_key]),
                u.CHI2_TOLERANCE,
                f"{settings[0][0]}: chi2 = "
                f"{chi2s[settings[0][0]]:.6f} vs frozen reference "
                f"{self.reference[reference_key]:.6f}")

    def test(self):
        self._run_sweep()
    test.__name__ = f"test_nonlimber_{kind}"
    test.__doc__ = (f"Limber vs non-Limber {probe}: the five checks "
                    "of the factory docstring.\n        ")
    setattr(TestNonLimber, test.__name__, test)
    TestNonLimber.__name__ = name
    TestNonLimber.__qualname__ = name
    return TestNonLimber


# ---- the sector-ladder cache checks -----------------------------------------

# Sector membership by sampled-parameter name. re.compile builds a
# regular-expression matcher: ^(a|b)$ matches exactly the names a or
# b, "_A1_|_A2_" matches a name containing either text, and "_M[0-9]+$"
# a name ending in _M plus digits. _ladder_sector_of returns the first
# sector whose pattern matches, so the order matters and "other" (the
# pattern "." matches any name) collects the rest. The bias and other
# sectors move only in the scramble step.
LADDER_SECTORS = (
    ("cosmo",
     re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
# A key of LADDER_DELTAS is either an exact parameter name (a string)
# or a compiled pattern matched against the names. The offsets are
# small moves inside the priors that still change the data vector
# visibly.
LADDER_PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
LADDER_DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}
# LADDER_NSTEP = steps per sector on the ladder; LADDER_SCRAMBLE_STEP
# = the step count of every sector at the scramble point, beyond the
# ladder's end; LADDER_RESCALE_RTOL = relative tolerance of the
# analytic shear-calibration rescale, a few roundings of a product of
# double-precision factors.
LADDER_NSTEP = 3
LADDER_SCRAMBLE_STEP = 4  # every sector at step 4, bias included
LADDER_RESCALE_RTOL = 1.0e-12


def _ladder_sector_of(name):
    """Return the sector of one sampled-parameter name.

    Arguments:
      name = a sampled-parameter name, e.g. a survey's "..._DZ_S1".

    Returns:
      the name of the first LADDER_SECTORS entry whose pattern
      matches (pat.search looks for the pattern anywhere in the
      name).
    """
    for sector, pat in LADDER_SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _ladder_deltas_for(sector, names):
    """Return {parameter: per-step delta} for one sector's sampled names.

    Arguments:
      sector = a LADDER_SECTORS name.
      names  = the sampled-parameter names that belong to the sector.

    Returns:
      a dictionary with one entry per name that a LADDER_DELTAS key
      of the sector matches; names no key matches are left out.
    """
    # .get(sector, {}) returns an empty table for a sector without
    # offsets (bias moves only in the scramble, "other" never)
    table = LADDER_DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # the condition reads: when key is a string, compare it
            # with n; otherwise key is a compiled pattern, so search
            # n for it
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


def cache_ladder_suite(u, name, example="example2"):
    """Build the sector-wise cache-invalidation class (the ladder).

    cosmolike keeps the result of every expensive stage in memory and
    recomputes it only when an input it depends on changed. The class
    walks a deterministic parameter ladder in one process (three
    steps per sector: cosmology, IA, source photo-z, lens photo-z,
    shear calibration; a sector the configuration samples no
    parameters in drops out), evaluating after every step, and
    asserts: every step changes the vector; each M-only step is the
    analytic (1+m_i)(1+m_j) rescale; a no-op update is bitwise
    neutral; returning to the ladder's final point after a scramble
    reproduces vector and chi2 bit for bit; and a second model
    walking the mirrored ladder lands on the same final vector. Both
    IA models run (the TATT ladder exercises the FAST-PT rebuild
    machinery NLA never touches), and every evaluation bypasses
    cobaya's cache, so the checks test cosmolike's own invalidation.

    Arguments:
      u       = the project's cocoa_test_utils module.
      name    = the class name to build (assigned by the caller to
                the same module-level name, so test ids do not
                change).
      example = the key of u.EXAMPLES the ladder evaluates (the
                flagship combination, so every sector enters its
                data vector).

    Returns:
      a unittest.TestCase subclass named `name`.
    """

    class TestCacheLadder(unittest.TestCase):
        """Sector-ladder cache-invalidation check on the frozen fiducial."""

        @classmethod
        def setUpClass(cls):
            """Move to ROOTDIR and verify the frozen state, once per class."""
            u.require_cocoa_environment()
            u.verify_frozen()

        def _point_at(self, fid, steps):
            """Return the ladder point with each sector at its step count.

            Arguments:
              fid   = the fiducial {parameter name: value} point.
              steps = {sector name: step count}; each parameter of
                      the sector moves to fiducial + step * delta.

            Returns:
              a new {parameter name: value} dictionary (fid is not
              changed).
            """
            point = dict(fid)
            for sector, step in steps.items():
                for n, d in self.sector_deltas[sector].items():
                    point[n] = fid[n] + step * d
            return point

        def _mpairs(self, like, np_):
            """Per data-vector entry, the source bins of its (1+m) factors.

            Cosmic shear scales by both bins' factors (1+m_i)(1+m_j),
            gamma_t (and the ks cross where present) by the source
            bin's, clustering, gk and kk by nothing. Fourier data
            vectors use ncl entries per block, real ones ntheta; the
            xi_pm split doubles the shear block in real space.

            Arguments:
              like = the likelihood object (read for its bin counts
                     and, when present, its ggl_exclude pairs).
              np_  = the numpy module, passed in because numpy is
                     imported inside the test methods.

            Returns:
              int array [n_data, 2]: row k = (i, j), the source bins
              whose factors multiply entry k; -1 marks "no factor"
              (for gamma_t the row is (-1, source bin); for
              clustering both are -1).
            """
            ci = _interface(u)
            real = hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
            sizes = [int(x) for x in
                     (ci.compute_data_vector_3x2pt_real_sizes()
                      if real else
                      ci.compute_data_vector_3x2pt_fourier_sizes())]
            nlen = int(like.ntheta) if real else int(like.ncl)
            nsrc = int(like.source_ntomo)
            # sspairs = every source pair (i, j) with i <= j, in
            # data-vector order (the two for clauses nest like two C
            # loops, i outer)
            sspairs = [(i, j) for i in range(nsrc)
                       for j in range(i, nsrc)]
            # excluded = the set of (lens, source) pairs the
            # likelihood drops from gamma_t; getattr returns None
            # when the likelihood has no ggl_exclude attribute, and
            # `or []` turns that into an empty list
            excluded = {(int(a), int(b)) for a, b in
                        (getattr(like, "ggl_exclude", None) or [])}
            # gglpairs = every (lens, source) pair kept in gamma_t,
            # lens bin outer, source bin inner
            gglpairs = [(zl, zs)
                        for zl in range(int(like.lens_ntomo))
                        for zs in range(nsrc)
                        if (zl, zs) not in excluded]
            # start with -1 ("no factor") everywhere; the loops below
            # fill the shear and gamma_t rows block by block
            fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
            k = 0
            # xi_plus + xi_minus in real space
            ssrep = sspairs + sspairs if real else sspairs
            for (i, j) in ssrep:
                for t in range(nlen):
                    fac[k] = (i, j)
                    k += 1
            for (zl, zs) in gglpairs:
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
            # skip clustering
            k = sizes[0] + sizes[1] + sizes[2]
            # 6x2pt: gk (no factor), ks (source), kk (none)
            if len(sizes) > 3:
                k += sizes[3]
                for zs in range(nsrc):
                    for t in range(nlen):
                        fac[k] = (-1, zs)
                        k += 1
            return fac

        def _run_ladder(self, tatt):
            """Walk the forward and mirrored ladders; assert checks 1-5.

            Each order builds its own model in this process (the two
            models share the compiled library's C globals); check 5
            compares their final vectors bit for bit.

            Arguments:
              tatt = True runs the TATT variant, False the NLA one.

            Returns:
              nothing; a failed check raises AssertionError.
            """
            import numpy as np
            ci = _interface(u)

            lk_name = u.EXAMPLES[example]["likelihood"]
            results = {}
            for order in ("forward", "mirrored"):
                info = u.load_frozen_info(example, tatt=tatt)
                model = u.make_model(info)
                fid = dict(u.build_point(model, example, tatt=tatt))
                # sector_deltas = {sector: {parameter: per-step
                # delta}}; the inner comprehension lists the
                # fiducial's names that belong to sector s
                self.sector_deltas = {
                    s: _ladder_deltas_for(
                        s, [n for n in fid
                            if _ladder_sector_of(n) == s])
                    for s, _ in LADDER_SECTORS}
                for s in ("cosmo", "dz_source", "m"):
                    self.assertTrue(
                        self.sector_deltas[s],
                        f"no sampled parameters in sector {s}")
                # a sector nothing samples drops out of the ladder:
                # lenses that are the source sample carry no separate
                # DZ_L shifts, and a configuration may fix every IA
                # amplitude
                active = tuple(s for s in LADDER_PHASES
                               if self.sector_deltas[s])

                phases = (active if order == "forward"
                          else tuple(reversed(active)))
                # every sector starts at step 0 (the fiducial point)
                steps = {s: 0 for s in self.sector_deltas}
                u.evaluate_chi2(model, self._point_at(fid, steps))
                # compute_data_vector_masked returns the theory
                # vector of the state the last evaluation left in
                # cosmolike
                prev = np.array(ci.compute_data_vector_masked())
                mfac = self._mpairs(model.likelihood[lk_name], np)

                for sector in phases:
                    for r in range(1, LADDER_NSTEP + 1):
                        m_prev = {
                            n: fid[n] + steps["m"] * d
                            for n, d in
                            self.sector_deltas["m"].items()}
                        steps[sector] = r
                        point = self._point_at(fid, steps)
                        u.evaluate_chi2(model, point)
                        dv = np.array(ci.compute_data_vector_masked())
                        self.assertFalse(
                            np.array_equal(dv, prev),
                            f"{order}: {sector} step {r} left the "
                            "data vector unchanged (dead sector flag "
                            "or stale cache)")
                        if sector == "m":
                            # check 2: the expected ratio dv/prev of
                            # every entry is the product of
                            # (1+m_new)/(1+m_old) over the source
                            # bins mfac names for it
                            m_now = {n: point[n]
                                     for n in self.sector_deltas["m"]}
                            # M1..M5 in bin order
                            mp = sorted(m_prev)
                            ratio = np.ones(dv.size)
                            for k in range(dv.size):
                                i, j = mfac[k]
                                if j >= 0:
                                    ratio[k] *= ((1 + m_now[mp[j]]) /
                                                 (1 + m_prev[mp[j]]))
                                if i >= 0:
                                    ratio[k] *= ((1 + m_now[mp[i]]) /
                                                 (1 + m_prev[mp[i]]))
                            # nz = boolean mask of the entries where
                            # prev is nonzero (masked entries are 0
                            # and cannot carry a ratio); rel =
                            # relative deviation from the analytic
                            # rescale on those entries
                            nz = prev != 0
                            rel = np.abs(
                                dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                            self.assertLess(
                                rel.max(), LADDER_RESCALE_RTOL,
                                f"{order}: M step {r} is not the "
                                "analytic (1+m_i)(1+m_j) rescale "
                                f"(max {rel.max():.2e})")
                        prev = dv

                final_point = self._point_at(fid, steps)
                final_chi2 = u.evaluate_chi2(model, final_point)
                final_dv = np.array(ci.compute_data_vector_masked())

                # no-op probe: identical point again, bitwise
                u.evaluate_chi2(model, dict(final_point))
                self.assertTrue(
                    np.array_equal(
                        np.array(ci.compute_data_vector_masked()),
                        final_dv),
                    f"{order}: a no-op re-evaluation changed the "
                    "data vector")

                # scramble: every sector at once, bias included
                scr = {s: LADDER_SCRAMBLE_STEP
                       for s in self.sector_deltas}
                u.evaluate_chi2(model, self._point_at(fid, scr))

                # return: the ladder's final point must reproduce
                # bitwise
                back_chi2 = u.evaluate_chi2(model, final_point)
                back_dv = np.array(ci.compute_data_vector_masked())
                self.assertTrue(
                    np.array_equal(back_dv, final_dv),
                    f"{order}: returning after the scramble did not "
                    "reproduce the data vector bit for bit (stale "
                    "sector cache)")
                self.assertEqual(
                    back_chi2, final_chi2,
                    f"{order}: chi2 after the scramble return "
                    "differs")
                results[order] = final_dv
                print(f"  {order} ladder "
                      f"({'TATT' if tatt else 'NLA'}): "
                      f"final chi2 = {final_chi2:.6f}", flush=True)

            self.assertTrue(
                np.array_equal(results["forward"],
                               results["mirrored"]),
                "the mirrored-order ladder landed on a different "
                "data vector: the answer depends on the invalidation "
                "history")

        def test_cache_consistency_nla(self):
            """The ladder checks 1-5 hold for the NLA configuration."""
            self._run_ladder(tatt=False)

        def test_cache_consistency_tatt(self):
            """The ladder checks 1-5 hold for the TATT configuration."""
            self._run_ladder(tatt=True)

    TestCacheLadder.__name__ = name
    TestCacheLadder.__qualname__ = name
    return TestCacheLadder
