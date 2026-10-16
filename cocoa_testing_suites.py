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

import unittest


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
