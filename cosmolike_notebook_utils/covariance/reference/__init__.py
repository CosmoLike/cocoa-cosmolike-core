"""Independent numerical oracles for covariance component tests.

An oracle is a separate, usually slower implementation of one covariance
ingredient. A test gives the oracle and the production C function the
same inputs and compares their outputs. Production workflows never import
these modules. NumPy, SciPy and mpmath implement separate quadratures or
algebra so comparisons do not merely repeat the C algorithm.

Shared physical samples are identified explicitly. When a test feeds the
same CAMB tables, halo abundances and profiles, or field windows to both
sides, the module docstring says so: agreement then tests the algorithm,
not those inputs.

Each module, and the C file in cosmolike/covariances that it checks:

    gaussian_reference      Gaussian (Wick) covariance, its projection and
                            annulus kernels; gaussian_cov.c
    operators_reference     bin-averaged Legendre and spin-2 angular
                            kernels; operators_cov.c
    mask_reference          ordered pair area of a spherical-cap footprint;
                            mask_cov.c
    ssc_reference           super-sample covariance (SSC): cap mask
                            spectrum, shell weight, shell response and
                            projection; ssc_cov.c
    spectra_reference       Limber field spectra and lensing efficiency;
                            spectra_cov.c
    halo_reference          halo mass moments I11 and the five pair
                            moments; halo_cov.c
    cng_reference           tree-level kernels, Wick-diagram sums and
                            planar angle averages for the connected
                            non-Gaussian (cNG) term; perturbation_cov.c
    non_gaussian_reference  halo partitions of the cNG trispectrum;
                            non_gaussian_cov.c
    projected_ng_reference  line-of-sight projection of supplied cNG, SSC
                            and Gaussian inputs at selected multipoles
"""
