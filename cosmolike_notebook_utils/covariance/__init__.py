"""Notebook tools for covariance components and survey assembly.

A survey covariance has three components: the Gaussian part G, built from
signal-plus-noise power spectra; the super-sample covariance SSC, from the
response of the measured spectra to density modes larger than the survey
footprint; and the connected non-Gaussian part cNG, built from the
halo-model matter trispectrum.

The caller supplies an initialized project interface. This package never
imports a project's compiled module. Reading a supplied likelihood covariance
requires an explicit call to the comparison tools in likelihood.py.
Survey numbers, redshift files and modeling choices belong to the notebook
or project example. The underlying C components run with OpenMP threads and
SIMDe vector instructions; Python prepares common inputs and assembles the
requested matrix blocks.

Modules: geometry (noise, cap mask, angle rule), accuracy (numerical
controls), power (dense power tables), sampling (dense response tables),
halo (halo-model cNG and SSC inputs), gaussian (G blocks), survey (3x2pt
G/SSC/cNG matrices), counts_cluster (cluster count terms), diagnostics
(positivity and refinement checks), likelihood (comparison with a supplied
matrix), forecast (project set-up and archives) and command_line (YAML
command). The other *_cluster modules are imported explicitly where needed.
"""

from .geometry import angular_rule, cap_mask, noise_powers
from .diagnostics import covariance_modes, compare_covariances

from .gaussian import gaussian_block, observed_spectra, realspace_block, shear_gaussian
from .sampling import DenseLogTable
from .halo import halo_mass_edges, halo_power_response, halo_trispectrum
from .accuracy import covariance_accuracy, load_covariance_accuracy
from .power import refine_power_tables

from .survey import realspace_covariance, fourier_covariance, observable_rows
from .counts_cluster import count_statistics, count_matter_cross
