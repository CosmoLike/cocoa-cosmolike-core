"""Notebook tools for covariance components and survey assembly.

The caller supplies an initialized project interface. This package never
imports a project's compiled module and never loads a likelihood covariance.
Survey numbers, redshift files and modeling choices belong to the notebook
or project example. The underlying C components retain OpenMP and SIMDe;
Python prepares common inputs and assembles the requested matrix blocks.
"""

from .geometry import angular_rule, cap_mask, noise_powers
from .diagnostics import covariance_modes, compare_covariances

from .gaussian import gaussian_block, observed_spectra, realspace_block, shear_gaussian
from .sampling import DenseLogTable
from .halo import halo_power_response, halo_trispectrum
from .accuracy import covariance_accuracy

from .survey import realspace_covariance, fourier_covariance, observable_rows
