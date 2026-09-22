"""Shared helpers for the EXAMPLE_EVALUATE notebooks of every project.

Each Cosmolike project ships Jupyter notebooks that drive the compiled
interface directly (their own CAMB run, set_cosmology, the nuisance
setters, compute_data_vector_masked). The notebooks used to carry
private copies of the same support functions; those copies drifted
apart and every interface upgrade had to be repeated in each one. The
functions that do NOT depend on a project live here instead, and each
notebook imports them once:

    sys.path.insert(0, os.environ['ROOTDIR']
                    + '/external_modules/code/cosmolike_core')
    import cosmolike_notebook_utils as cnu

Design rule for this package: it never imports a project's compiled
interface (cosmolike_<project>_interface). Each project compiles its
own interface module, so a function that needs cosmolike receives the
notebook's interface-bound callable as an argument instead (the `dv`
and `ddv` parameters of the Fisher helpers). Three groups:

  camb_cosmology     get_camb_cosmology: one CAMB run packaged into
                     the tuple set_cosmology consumes, on the nested
                     dyadic grids shared with the likelihoods.
  plot_datavectors   tomographic plots of the data vectors: cosmic
                     shear (plot_C_ss_tomo_limber, plot_xi),
                     galaxy-galaxy lensing (plot_C_gs_tomo_limber,
                     plot_gammat_tomo_limber), galaxy clustering
                     (plot_C_gg_tomo, plot_wtheta_tomo), and the
                     bfmt parameter sweeps (plot_baryon_suppression).
  plot_response      plot_response_function: curves of a data
                     vector's response to the matter power spectrum,
                     both d ln DV / d ln k and cumulative R(k_max).
  fisher             finite-difference and derivkit derivatives of an
                     injected data-vector function, Fisher-matrix
                     assembly, figures of merit, and Fisher contour
                     plots via getdist.

The notebooks keep everything project-specific: fiducial parameter
values, tomographic-bin layouts, the interface init sequence, and
thin wrappers that bind those to the functions here.
"""

from .camb_cosmology import get_camb_cosmology
from .plot_datavectors import (plot_C_gg_tomo, plot_C_gs_tomo_limber,
                               plot_C_ss_tomo_limber,
                               plot_baryon_suppression,
                               plot_gammat_tomo_limber, plot_wtheta_tomo,
                               plot_xi)
from .fisher import (add_gaussian_priors, get_Fisher, get_Fisher2,
                     get_FoM, get_ddv, get_ddv_dkit, plot_Fisher)
from .plot_response import plot_response_function
