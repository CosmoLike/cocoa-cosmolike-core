# cocoa-cosmolike-core

This repository holds the Cosmolike core shared by the Cocoa
Cosmolike projects:

- `cosmolike/`: the C core (angular power spectra, real-space
  correlation functions, likelihood assembly), its C++ wrapper
  layer, and the cluster code (both described below).
- `cfastpt/`: the C implementation of FAST-PT used by the TATT
  intrinsic-alignment model (`IA_code: 0`, the default; each
  project's tests compare it against the python FAST-PT package in
  the CFASTPT vs FASTPT comparison of its `tests/README.md`).
- `cosmolike_notebook_utils/`: python support functions shared by
  the projects' example notebooks: `camb_cosmology.py` (the CAMB
  run packaged for `set_cosmology`), `plot_datavectors.py`,
  `plot_datavectors_cluster.py`, and `plot_response.py` (the
  data-vector plots, their cluster counterpart, and the
  response-function plots), and `fisher.py` (the Fisher-forecast
  helpers). The package never imports a project's compiled
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
- `log.c/`: the C99 logging library that the C core includes.
- `dev_scripts/`: checks of the comments of the C and C++ sources:
  the function header blocks (`audit_headers.py`), and that an
  edit changed comments only (`check_comment_only.py`).
- `future_port_unfinished/`: code that no project compiles: the
  unfinished Compton-y port (`cosmo2d_tmp.c`) and an experimental
  draft of analytic Fisher derivatives (`fisher/`).

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

## The C++ wrapper layer

Three files of `cosmolike/` are the C++ batch evaluators of the C
core: each project's `interface/interface.cpp` binds them with
pybind11, and the notebooks call them from Python.

- `cosmo2D_wrapper.cpp/.hpp`: the 2D projections of `cosmo2D.c`,
  returned as whole numpy arrays.
- `halo_wrapper.cpp/.hpp`: the halo model of `halo.c`.
- `cosmo2D_scuts_wrapper.cpp/.hpp`: the scale-cut diagnostics of
  `cosmo2D_scuts.c`.

## Massive neutrinos in the halo model

Every project's `set_cosmology` binding takes, besides the power
spectra, growth and distances, two neutrino inputs:

- `omegan2`: $\Omega_\nu h^2$ of the massive neutrinos (CAMB's
  `omnuh2`; default 0), stored as `cosmology.Omega_nu`, part of
  $\Omega_m$;
- `lnP_linear_cb`: $\ln P_{cb}$, the linear power spectrum of cold
  dark matter plus baryons (CAMB's `delta_nonu`) on the grid and in
  the layout of `lnP_linear` (default empty: no table).

They matter only when the halo model counts halos of cold dark
matter plus baryons, the DES Y1 cluster model (arXiv:2010.01138):
`init_halo_matter_field(1)`, the likelihood yaml key
`halo_matter_field: 1` (`like.halo_model[4]`, `halo.h`). Then
$\sigma^2(M)$ integrates $P_{cb}$, and the Lagrangian radius and the
$\rho/M$ factor of $dn/d\ln M$ use
$\rho_{crit}(\Omega_m - \Omega_\nu)$; the NFW truncation radius, the
matter window $M/\rho_m$, the lensing kernels and the two-halo spectra
stay total matter. The default 0 is the total-matter
halo model, and nothing reads the two inputs. The likelihoods always
send $\Omega_\nu h^2$ and send $P_{cb}$ under 1; the notebook helper
`cosmolike_notebook_utils.get_camb_cosmology` returns both.

## The cluster code

`cosmolike/` also implements the DES Y6-style cluster analysis of
arXiv 2503.13631: cluster counts, cluster lensing, cluster
clustering, and cluster x galaxy clustering, combined with galaxy
clustering as "4x2pt + N" and with the whole 3x2pt as "6x2pt + N".
The project that uses it is `des_cluster` (repository
`cocoa_des_cluster`).

Cluster code lives only in the files whose names end in
`_cluster`. The other files carry no cluster fields or functions,
and when the cluster code needs an internal of the core, it keeps
its own copy instead of exporting it from the core header.

- `structs_cluster.h/.c`: the global `cluster`, which holds every
  piece of cluster state, and its defaults.
- `redshift_spline_cluster.c/.h`: selection kernels, redshift
  distributions, and tomographic pair maps of the cluster sample.
- `radial_weights_cluster.c/.h`: the radial weights of the
  clusters in the Limber integrals (density and magnification).
- `halo_cluster.c/.h`: the mass-observable relation, and the
  number density, bias, and one-halo spectrum of a richness bin.
- `cosmo2D_cluster.c/.h`: the Limber spectra and real-space
  statistics of the cluster two-point functions, and the counts.
- `generic_interface_cluster.cpp/.hpp`: the setters of `cluster`,
  the joint data vector, and `IPCluster` (mask, covariance, chi2).
- `cosmo2D_wrapper_cluster.cpp/.hpp`, `halo_wrapper_cluster.cpp/.hpp`:
  the cluster analogs of `cosmo2D_wrapper` and `halo_wrapper`.
