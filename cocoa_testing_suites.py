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
import unittest


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
