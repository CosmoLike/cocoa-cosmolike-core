# Transferring the test suite to a new survey project

A new Cosmolike project created from a donor (projects/README.md, "How
do we create a new Cosmolike project?") arrives with **no tests/
folder**: the transfer script deletes it, because the suite is pinned
to the donor survey beyond its file names. This reference is the recipe
for recreating the suite. The human provides the new survey's data
files, likelihoods and covariance adapter first, and reviews every
number this recipe generates; nothing here may be invented or copied
from the donor's frozen values.

## Why the suite cannot be renamed

- `tests/frozen/` and `tests/manifest_sha256.json` are SHA-256-pinned
  snapshots of donor data; against new data the pin is simply broken.
- `tests/cocoa_test_utils.py` (the project shim) encodes survey
  judgment: the EXAMPLES table of configurations, scale-cut datasets
  (the donor's `M2`-`M6` masks have no counterpart in a new survey),
  parameter points inside emulator training boxes, accuracy knobs and
  comparison settings.
- The `tests/data_vector/README.md` and `tests/covariance/README.md`
  tables and figures are measurements of donor data.
A sed rename of these produces a suite that looks transferred but
silently checks the wrong survey, manufacturing false confidence.

## Architecture: what is shared, what is project-owned

The suite is a thin project layer over the shared harness
`cosmolike_core/cocoa_testing.py`:

- **Shared harness** (do not copy, do not fork): environment guards
  (`require_cocoa_environment`, thread assertions), the frozen-state
  verifier, cobaya model construction, chi2 evaluation with worker
  subprocesses, the N-random-models mechanism (a default-settings model
  writes its own vector, which becomes a temporary dataset so the
  default chi2 is zero by construction), baryon-method machinery,
  CFASTPT-vs-FASTPT and Halofit-vs-EE2 sweeps, and report printing.
- **Project shim** `tests/cocoa_test_utils.py`: builds one
  `cocoa_testing.CocoaTestHarness` from project data (worker_file,
  interface_module, EXAMPLES, TATT point, accuracy knobs, fastpt
  settings) and re-exports the harness methods as module names the
  test files use. This file carries ALL survey judgment.
- **Thin test modules** under `tests/data_vector/` and
  `tests/covariance/`: mostly identical across projects apart from the
  interface import and the shim contents. Copy them from the donor,
  then adjust only what the per-file notes below name. Never embed
  survey numbers in a test module; they belong in the shim.
- `tests/conftest.py` binds the shim and registers the shared
  `--high` / `--mask` options; `tests/pytest.ini` keeps
  `testpaths = data_vector` (covariance and studies run by explicit
  path). `tests/generate_frozen_reference.py` writes the frozen state.
- `tests/studies/` holds owner-invoked studies (for example the n(z)
  reconstruction study), outside routine pytest.

## The transfer recipe

1. **Preconditions.** The new project compiles; its likelihoods
   evaluate (an `EXAMPLE_EVALUATE*.yaml` runs); the covariance adapter
   exists if the covariance sector is wanted. Work inside the cocoa
   environment with `start_cocoa.sh` sourced.
2. **Shim first.** Write `tests/cocoa_test_utils.py` from the donor's
   as a template: new interface module name; a new EXAMPLES table
   naming this project's configurations, frozen-module names,
   likelihood component names and datasets; parameter points chosen
   inside every emulator or fit's valid range (the donor's points are
   donor choices, not defaults); scale-cut masks that exist for this
   survey; accuracy knobs kept unless the survey needs others. Every
   entry is a decision to flag to the human, not a rename.
3. **Support files.** Copy `conftest.py`, `pytest.ini` and
   `generate_frozen_reference.py`; adjust only project names and the
   EXAMPLES-dependent parts of the generator.
4. **Test modules.** Copy the donor's `data_vector/` and `covariance/`
   modules; fix the interface import; drop modules whose subject the
   new project lacks (no baryon block, no covariance sector, no KL
   shape noise) instead of leaving vacuous tests; port
   `covariance/abort_check.py` as is (the forked-exit pattern is
   generic). The n(z) cache round-trip test resolves its interface
   folder itself and ports unchanged.
5. **Freeze.** With the human's survey data in place:
   `python ./projects/<new>/tests/generate_frozen_reference.py
   --overwrite`, then `--baryons` if the baryon tests were kept. The
   printed chi2 values become the references **only after the human
   reviews them**; record them to four decimals. The frozen chi2
   tolerance is 0.2 and is never tightened or loosened.
6. **Readmes.** Write `tests/README.md` and the sector READMEs for the
   new survey: document what each test checks and the measured values
   of THIS survey. Do not carry donor tables, figures or timings;
   re-measure or omit. Numbers name their quantity in place.
7. **Acceptance gates.** The full data-vector sector passes; the
   covariance sector passes if present; `grep -rni "lsst" tests/`
   (or the donor's name) returns nothing; no test was weakened,
   skipped by default, or pointed at a stored oracle (reference
   outputs live in frozen/ snapshots, never as maintained Python
   oracles in git).

## Deviations that are allowed

Project differences are legitimate when the physics differs: a project
without a covariance sector has no `tests/covariance/`; DES Y6's
supplied identity metric has no statistical chi2 arm; cluster projects
add cluster sectors. State such deviations in `tests/README.md`
instead of faking the missing piece.
