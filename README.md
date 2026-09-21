# cocoa-cosmolike-core

This repository holds the Cosmolike core shared by the Cocoa
Cosmolike projects:

- `cosmolike/`: the C core (angular power spectra, real-space
  correlation functions, likelihood assembly).
- `cfastpt/`: the C implementation of FAST-PT used by the TATT
  intrinsic-alignment model.
- `cosmolike_notebook_utils/`: python support functions shared by
  the projects' example notebooks: the CAMB run packaged for
  `set_cosmology`, the data-vector plots, and the Fisher-forecast
  helpers. The package never imports a project's compiled
  interface (anything that needs cosmolike receives the notebook's
  own data-vector function as an argument); its design is
  documented in `cosmolike_notebook_utils/__init__.py`.

A notebook imports the shared functions through Cocoa:

    sys.path.insert(0, os.environ['ROOTDIR'] + '/external_modules/code/cosmolike_core')
    import cosmolike_notebook_utils as cnu
