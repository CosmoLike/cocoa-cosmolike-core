# Bounded multipole storage for covariance spectra

## Why the high-boost test needed batching

The LSST single-source Gaussian stress test still requires the initialized
catalog's full internal field matrix. At boost 8, its 7 radial panels have
3584 nodes, the signal has 79999 multipoles, and there are ten fields.
`limber_spectra_cov` allocates two power tables and complete field windows,
about 8*(2+10)*3584*79999 = 27.52 GB of logical double storage. A process
sample found all eight workers inside its Limber-spectrum loop. The
initial unbatched boost-8 run was stopped before completion; its elapsed
time is not a completed-runtime measurement.

`gaussian.limber_spectra` calls the same C routine on contiguous multipole
batches. Every batch retains all radial nodes, weights and field pairs.
No quadrature or summation order changes. It concatenates the spectra and
retains one owned radial snapshot. Batches of 512 modes bound those C
arrays to 176.16 MB in this configuration, excluding returned arrays and
other process memory. This is a storage estimate, not a peak-RSS claim.

The full galaxy/shear, cluster and single-source assemblers share this
helper. The low-level C++ interface still accepts arbitrary supplied
multipoles. No C implementation, SIMD instruction or data-vector path was
changed. Python does not introduce an MPI layer or nested worker pool.

## Measurements (2026-10-04)

Apple M2 Pro, BLAS one, actual five-lens/five-source LSST inputs, massless
Limber model with no IA/RSD, ell=2..4097, seven 128-node radial panels and
4097 lensing-window nodes. Each case has one excluded warm-up and three
timed calls. CAMB/initialization and array comparisons are excluded; radial
preparation, C allocation, integration and Python output assembly are
included. No other numerical job or build ran. Preflight/postflight process
inspection found ordinary desktop background activity.

At eight threads, mean spectrum-preparation times were 0.17456 s for the
unbatched call, 0.08057 s for batches of 512, 0.09616 s for 1024, and
0.15148 s for 2048. The selected 512 size was then measured at all four
worker counts:

| Threads | Unbatched mean (s) | 512-batch mean (s) | Batch sample std (s) |
|---:|---:|---:|---:|
| 1 | 0.48460 | 0.28689 | 0.00192 |
| 2 | 0.27690 | 0.16123 | 0.00029 |
| 4 | 0.20277 | 0.09347 | 0.00012 |
| 8 | 0.17456 | 0.07900 | 0.00058 |

All measured spectra, geometry and windows agree bitwise with the unbatched
reference across sizes and thread counts. This is an all-pairs spectrum
component benchmark, not a full covariance timing or an x86 result. Scaling
from four to eight workers remains weaker than ideal; batching does not
establish the general 8--10-core scaling goal.

External reproducibility files: covariance_reference/benchmark_limber_batches.py
(the --selected option runs the chosen-size thread sequence), and
results/limber_batches.json / results/limber_batches_512.json. The interrupted
stress-test sample is /tmp/gaussian-boost8-sample.txt.

## Numerical and didactic checks

The project check compares batches of one, four and more-than-all modes
with the direct C++ call at one/eight workers, for linear/nonlinear power
and RSD off/on with NLA and magnification present. Every spectrum, base
window and geometry value is bitwise identical, including uneven final
batches. The subsequent all-project covariance run passes 116 checks.

The manual didactic review checked why multipoles are independent while
radial nodes cannot be dropped, preservation of unmeasured crossed pairs,
the first snapshot's ownership, the dictionary copy before replacing its
spectrum array, explicit storage-only batch semantics and physical units.
The loop overview explains the reason for batching; no new intrinsic or
C loop needs a SIMD explanation. Full high-boost Gaussian convergence is
recorded in covariance_accuracy.md separately from performance evidence.
