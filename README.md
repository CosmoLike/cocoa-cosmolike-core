# cocoa-cosmolike-core

This repository holds the Cosmolike core shared by the Cocoa
Cosmolike projects:

- `cosmolike/`: the C core (angular power spectra, real-space
  correlation functions, likelihood assembly).
- `cfastpt/`: the C implementation of FAST-PT used by the TATT
  intrinsic-alignment model (`IA_code: 0`, the default; each
  project's tests compare it against the python FAST-PT package in
  the CFASTPT vs FASTPT comparison of its `tests/README.md`).
- `cosmolike_notebook_utils/`: python support functions shared by
  the projects' example notebooks: the CAMB run packaged for
  `set_cosmology`, the data-vector plots, and the Fisher-forecast
  helpers. The package never imports a project's compiled
  interface (anything that needs cosmolike receives the notebook's
  own data-vector function as an argument); its design is
  documented in `cosmolike_notebook_utils/__init__.py`.
- `cocoa_testing.py`: the machinery of the projects' unit tests:
  the integrity checks of the stored test state under
  `tests/frozen/`, the worker-subprocess isolation, the race
  checks, the baryonic feedback checks, and the CFASTPT vs FASTPT
  comparison. Each project's `tests/cocoa_test_utils.py` keeps
  only its data (examples, TATT point, accuracy settings,
  comparison contract) and binds it to one `CocoaTestHarness`
  instance; the module docstring carries a complete worked example
  of that binding.

A notebook imports the shared functions through Cocoa:

    sys.path.insert(0, os.environ['ROOTDIR'] + '/external_modules/code/cosmolike_core')
    import cosmolike_notebook_utils as cnu

A project's tests import the harness the same way, except that
`tests/cocoa_test_utils.py` computes this repository's path from
its own location, so the import also works before `start_cocoa.sh`
exports `ROOTDIR`:

    import cocoa_testing as _cct
    _H = _cct.CocoaTestHarness(worker_file=__file__, ...)
    verify_frozen = _H.verify_frozen
    single_model_chi2 = _H.single_model_chi2
