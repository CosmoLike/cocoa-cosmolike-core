# sigma^2(R, z): FFTLog vs direct integration

Study for PLAN.md Phase 2 (owner request, 2026-10-02). Python only; no
repository was edited. Accuracy only: no speed benchmark was run (the
machine was busy). `timing.py` was checked on a tiny case and is ready for
a run on a quiet machine (section 6).

    sigma^2(R) = 1/(2 pi^2) int dk k^2 P(k) W(kR)^2 = int dln k Delta^2(k) W(kR)^2
    W(x) = 3 j1(x)/x,  Delta^2 = k^3 P/(2 pi^2),  M = (4 pi/3) rho_crit Omega_field R^3

| file | content |
|---|---|
| `sigma_methods.py` | the methods: `quad` (cosmolike sigma2), `Truth`, `FFTLogSigma2`, `QuadWeights` (B1), `FFTLogMatrix` (B2) |
| `accuracy.py` | Part 1: tables T1-T7 (`results/accuracy_tables.txt`), figures (`figures/`), truth cache `results/truth.npz` |
| `timing.py` | Part 2: timing of A, B1, B2, C for N_a = 1, 16, 64, 256 at a fixed thread count; `--check-determinism` |

Units are cosmolike's: k in h/Mpc, P in (Mpc/h)^3, R in Mpc/h, M in Msun/h
(rho_crit = 7.4775e21/coverH0^3 = 2.7752e11 Msun/h/(Mpc/h)^3, structs.c).
Omega_field = Omega_m - Omega_nu for P_cb (0.29858 at mnu = 0.06, 0.29289 at
0.3), Omega_m = 0.3 for P_m (cosmo3D.c `omega_halo_field`). R = 0.0142 Mpc/h
at M = 1e6 and 65.9 Mpc/h at 1e17. The table covers k = 1.49e-5..297 h/Mpc
(1200 nodes, dlnk = 0.01402).

**Extrapolation (mirrors cosmo3D.c `p_lin`/`p_lin_cb`).** A query outside the
table keeps the edge bracket, so ln P continues linearly in ln k with the
slope of the first (last) table interval: P ~ k^n_lo below k_min with
n_lo = 0.9660 (= n_s), P ~ k^n_hi above k_max with n_hi = -2.968, -2.962,
-2.956 at z = 0, 3, 6 (mnu = 0.06; -2.973..-2.960 at mnu = 0.3). Every
method here uses exactly this extension.

## 1. Truth and its checks

`Truth`: integral in ln k split at every table node and every zero of
j1(kR) up to x = kR = 3000, 16-node Gauss-Legendre per piece (every piece is
smooth), analytic piece below k_min (power law times the Taylor series of
W^2), analytic tail beyond x = 3000 (asymptotic series of
int x^(s-1) W^2). d sigma^2/d ln R uses the kernel x dW^2/dx = -18 j1 j2/x.
Two definitions of P between nodes: `plin` (ln P linear in ln k, cosmolike's
p_lin) and `spline` (not-a-knot cubic spline of ln P).

| check (T1, T2) | result |
|---|---|
| Mellin transform of W^2, closed form vs direct integration, real s = 0.03..3.5 | <= 4.4e-15 |
| same, complex s (numeric truncated at x = 1.3e4) | 2.5e-10 .. 4.3e-8 (truncation of the numeric) |
| Truth vs exact result for P ~ k^n, n = -2.5, -1, 0.5, R = 0.0142..66 | s2 <= 1.2e-13, d s2/d ln R <= 1.4e-13 |
| Truth, 16 -> 24 GL nodes per piece | 4.4e-16 |
| Truth, x_end 3000 -> 10000 | 4.8e-14 |

**p_lin's interpolation bias (T3).** The two truths differ: the p_lin truth
is low by up to **1.3e-5** in sigma^2 (mean -5.8e-6; 1.2e-5 in
1e12-1e16) and 1.8e-5 in d ln sigma/d ln M, the same for all 12 cases.
Halving the sampling multiplies it by 4.00 (a dlnk^2 effect: ln P is
concave between nodes). cosmolike's own table (dlnk = 0.0107) would carry
0.58 of it. The spline truth is the integral of the smooth CAMB spectrum;
the FFTLog targets below are stated against it, and against the p_lin truth
for reference.

## 2. Method quad = cosmolike sigma2 (cosmo3D.c:1983)

Re-implemented faithfully: head [1e-5, z_1] with 256 GL nodes in ln x,
512 lobes of 8 GL nodes, folded weights 9 j1^2, stopping rule s r/(1-r) <
1e-7 total, 192 coarse ln M nodes, natural cubic spline of ln sigma^2 to the
1024 dense nodes; the slope as halo.c `dlognudlogm` (symmetric difference,
half step 0.05 in ln M, linear reads, one-sided at the edges). 9.5e4 p_lin
reads per a node (14-89 segments per mass), as the header states.

Max errors over the 12 cases (T4); s2 = |sigma^2/truth - 1|, dlns = |relative error of d ln sigma/d ln M|:

| quad setting | truth | s2 1e6-1e17 | s2 1e12-1e16 | dlns 1e6-1e17 | dlns 1e12-1e16 |
|---|---|---|---|---|---|
| cosmolike default (hdi 0, 192 coarse, natural spline) | p_lin | 4.5e-5 | 1.2e-5 | 2.7e-3 | 2.1e-4 |
| | spline | 5.5e-5 | 2.3e-5 | 2.7e-3 | 2.1e-4 |
| hdi 0, 192 coarse, not-a-knot spline | p_lin | 1.2e-5 | 1.2e-5 | 1.6e-3 | 2.1e-4 |
| hdi 0, exact path (1024 nodes) | p_lin | 1.5e-5 | 1.5e-5 | 1.6e-3 | 4.5e-4 |
| hdi 3, exact path | p_lin | 8.7e-6 | 8.7e-6 | 1.6e-3 | 4.1e-4 |
| same nodes, smooth (spline) P, hdi 0 / hdi 3 | spline | 2.5e-6 / 8e-8 | | | |

Findings (not reproducing the header's "6.2e-6 maximum, slope within 1.3e-4"):

1. The 4.5e-5 maximum sits in the last coarse interval, M = 10^16.95..10^17:
   the natural spline's S'' = 0 end condition (not-a-knot: 1.2e-5).
   `figures/quad_error.png`.
2. Raising hdi does not cure the rest: the floor (8.7e-6 at hdi 3) is
   Gauss-Legendre integrating across the kinks of p_lin's piecewise-linear
   ln P. With a smooth P the same nodes give 2.5e-6 (hdi 0) and 8e-8 (hdi 3).
   The stopping rule costs ~1e-7 (EPS 1e-7 vs 1e-12).
3. The slope's error is the finite difference (O(0.05^2) plus node-to-node
   scatter / 0.1), and O(0.05) one-sided at the two range edges.

## 3. FFTLog (Part 1)

With u = ln k, g(u) = Delta^2 k^-b and the Mellin transform
M(s) = int x^(s-1) W^2 dx, a trigonometric expansion
g = sum_m c_m e^(i eta_m (u - u_0)), eta_m = 2 pi m/(N dlnk) gives

    sigma^2(R) R^b      = sum_m c_m M(b + i eta_m) e^(-i eta_m (u_0 + ln R))
    d sigma^2/d ln R R^b = sum_m c_m [-(b + i eta_m) M(b + i eta_m)] e^(...)
    M(s) = (9 pi/2) Gamma(4-s) Gamma(s/2) / (2^(4-s) Gamma((5-s)/2)^2 Gamma(4-s/2)),  0 < Re s < 4

(DLMF 10.22.57 with j1 = sqrt(pi/2x) J_3/2; M(1) = 3 pi/5, M(2) = 9/4,
M(s) -> 1/s at s -> 0). One rfft of g, two products, two irfft give
sigma^2 and d sigma^2/d ln R on the grid ln R_j = v_0 + j dlnk for all R at
once; a cubic spline in ln R takes ln sigma^2 and d ln sigma^2/d ln R to the
1024 ln M nodes; d ln sigma/d ln M = (1/6) d ln sigma^2/d ln R.

**Recommended settings** (`FFTLogSigma2` defaults): b = 1.5; input
k = 1e-7..1e5 h/Mpc, table nodes inside, p_lin edge power laws outside;
table spacing (N_in = 1973, N_fft = 2000); c_window 0.25 (cfftlog
convention, harmless); no p_window; no zero padding; any kr (low-ringing
not needed); cubic spline to the ln M nodes; derivative from the second
kernel. **A cheaper setting that still meets the target ("fast")**: every
3rd table node, k = 1e-6..1e4 (N_fft = 560).

**Result (T6, `figures/fftlog_recommended_error.png`)**, every one of the 12
cases (Pcb, Pm) x (mnu 0.06, 0.3) x (z = 0, 1, 3), all M = 1e6..1e17:

| FFTLog setting | truth | max s2 | max dlns |
|---|---|---|---|
| recommended | spline | 1.9e-9 | 1.2e-8 |
| recommended | p_lin | 1.3e-5 (= p_lin's own bias, T3) | 1.8e-5 |
| fast (N_fft = 560) | spline | 1.3e-6 | 2.5e-6 |

The target (1e-5, 1e-4) is met with margin 5000 against the smooth truth,
over the whole 1e6..1e17. Against the p_lin truth no smooth-P method can do
better than p_lin's 1.3e-5 bias (cosmolike's own quad: 1.5e-5..4.5e-5).

**What each ingredient does** (T5, max over the 12 cases, others as recommended, spline truth):

| study | setting | N_fft | s2 1e6-1e17 | s2 1e8-1e17 | s2 1e12-1e16 | dlns 1e6-1e17 |
|---|---|---|---|---|---|---|
| bias b | 0.5 | 2000 | 1.7e-2 | 1.7e-2 | 3.0e-3 | 1.7e-2 |
| | 1 | 2000 | 1.4e-8 | 1.4e-8 | 2.8e-9 | 1.9e-8 |
| | **1.5** / 2 | 2000 | 1.9e-9 | 1.9e-9 | 1.4e-9 | 1.2e-8 |
| | 2.5 | 2000 | 3.2e-6 | 1.3e-8 | 1.4e-9 | 3.1e-5 |
| | 3 | 2000 | 3.9 | 1.6e-2 | 5.4e-7 | 7.8 |
| high-k end | table end (297 h/Mpc) | 1568 | 1.0e-3 | 5.6e-6 | 1.4e-9 | 8.6e-3 |
| | 1e3 | 1680 | 1.3e-5 | 4.8e-8 | 1.4e-9 | 1.8e-4 |
| | 3e3 | 1728 | 1.6e-7 | 1.9e-9 | 1.4e-9 | 3.0e-6 |
| | 1e4 / **1e5** / 1e6 | 1875-2160 | 1.9e-9 | 1.9e-9 | 1.4e-9 | 1.2e-8..1.6e-8 |
| low-k end | table start (1.49e-5) | 1620 | 1.8e-9 | 1.8e-9 | 1.4e-9 | 1.2e-8 |
| | 1e-6 / **1e-7** / 1e-9 | 1875-2304 | 1.9e-9 | 1.9e-9 | 1.4e-9 | 1.2e-8 |
| table only, no extrapolation | no pad | 1200 | 1.0e-3 | 5.6e-6 | 1.4e-8 | 8.6e-3 |
| | N_pad 500 or 2000 | 2205, 5250 | 1.0e-3 | 5.6e-6 | 1.4e-9 | 8.6e-3 |
| | N_pad 500 + c_window 0.25 + p_window 0.5 dex | 2205 | 3.3e-2 | 8.4e-5 | 2.5e-9 | 3.1e-1 |
| | N_pad 500 + low-ringing kr | 2205 | 1.0e-3 | 5.6e-6 | 1.4e-9 | 8.6e-3 |
| windows (extrapolated input) | none / c_window 0.1, **0.25**, 0.5 | 2000 | 1.9e-9 | 1.9e-9 | 1.4e-9 | 1.2e-8 |
| | b = 2 without / with c_window 0.25 | 2000 | 1.9e-9 | 1.9e-9 | 1.4e-9 | 1.2e-7 / 1.2e-8 |
| | p_window 1 dex; low-ringing kr; N_pad 500 | 2000-3000 | 1.8e-9..1.9e-9 | | 1.4e-9 | 1.2e-8..1.4e-8 |
| sampling dlnk | 0.0070 (spline-resampled) | 3969 | 1.1e-10 | 1.1e-10 | 8.7e-11 | 9.1e-10 |
| | **0.0140 (table)** | 2000 | 1.9e-9 | 1.9e-9 | 1.4e-9 | 1.2e-8 |
| | 0.0280 / 0.0421 / 0.0561 | 1000 / 672 / 500 | 2.5e-7 / 1.3e-6 / 1.3e-6 | | | 4.5e-7 / 2.5e-6 / 1.0e-5 |
| | 0.1122 (every 8th) | 250 | 3.1e-5 | 3.1e-5 | 2.1e-5 | 2.2e-4 |
| interpolation to ln M | **cubic spline** / 6-point Lagrange | 2000 | 1.9e-9 | | | 1.2e-8 / 1.3e-8 |
| | 4-point Lagrange | 2000 | 2.8e-8 | | | 2.2e-7 |
| derivative | **second kernel** / derivative of the spline | 2000 | | | | 1.2e-8 / 3.1e-7 |
| | derivative of a 4-point Lagrange | 2000 | | | | 3.2e-6 |

Reading of the table:

1. **No ringing to remove.** Once the input is extended with p_lin's power
   laws to where g has fallen by 1e-7 (high k) and 1e-13 (low k), the
   periodic copy of g is smooth and no window, padding or kr choice moves
   the result. The c_m spectrum (`figures/fftlog_kernel_and_spectrum.png`)
   flattens at ~1e-7 of c_0 from eta ~ 100 to the Nyquist eta = 224: the BAO
   chirp (local frequency eta ~ k r_s), not an edge (moving k_lo, k_hi or b
   leaves it; smoothing the BAO lowers it 10x). |M(b + i eta)| ~ eta^(b-4.5)
   = eta^-3 brings the products to 1e-13.
2. **The extension matters, the padding does not.** Without the high-k
   power law the low masses lose their k > 297 h/Mpc part (1e-3 at 1e6,
   5.6e-6 at 1e8); zero padding cannot supply it, and a FAST-PT p_window on
   the table makes it worse (it tapers real signal). For 1e12-1e16 alone the
   bare table is enough (1.4e-9).
3. **b in [1, 2].** g must decay at both ends: low k as k^(3.97-b), high k
   as k^(0.03-b). b = 0.5 leaves a step at k_hi, b >= 2.5 a step at k_lo.
   b = 1.5 makes the dynamic range of g even at both ends and keeps the
   slope at 1.2e-8 without a window (b = 2 needs the c_window for that).
4. **Sampling is the only real cost knob**: the error at coarser dlnk is the
   band-limited reconstruction of P (BAO), 1.3e-6 at every 3rd node.
5. The FFTLog output grid (dlnk in ln R = 0.042 in ln M) is interpolated;
   the spline costs 2e-9, the derivative kernel is 25x better than
   differentiating the spline.

## 4. Part 2: sigma^2(M, a) on N_a = 256 nodes x 1024 masses

| | A quad per a | B1 quad-node weight matrix | B2 FFTLog matrix | C FFTLog |
|---|---|---|---|---|
| what | cosmolike sigma2 at every a | W, D (192 x 1857) on the extended table grid; g by 4-pt Lagrange | G, GD (620 x 1322) = C written as a matrix | rfft + 2 irfft per a (N = 2000) |
| s2 / dlns vs spline truth (T7) | 5.5e-5 / 2.7e-3 | 2.8e-6 / 1.2e-5 | 2.8e-9 / 1.3e-8 | 1.9e-9 / 1.2e-8 |
| same vs p_lin truth | 4.5e-5 / 2.7e-3 | 1.4e-5 / 2.8e-5 | 1.3e-5 / 1.8e-5 | 1.3e-5 / 1.8e-5 |
| cheaper variant | - | hdi 3, 1024 rows: 2.0e-7 / 2.1e-7 (more cost) | fast grid 218 x 443: 1.3e-6 / 2.5e-6 | fast N = 560: 1.3e-6 / 2.5e-6 |
| work per a node | 9.5e4 p_lin reads (log10 + exp + bilinear each) | 1857 exp + 2 x 192 x 1857 = 7.1e5 MAC | 1322 exp + 2 x 620 x 1322 = 1.6e6 MAC (fast: 443 exp + 1.9e5 MAC) | 1973 exp + 3 FFTs ~ 3 x 2.5 N log2 N = 1.6e5 flop + 2 x 1001 complex mult + spline (fast: 550 exp + 4e4 flop) |
| work, N_a = 256 | 2.4e7 p_lin reads | 4.8e5 exp + 1.8e8 MAC | 3.4e5 exp + 4.2e8 MAC (fast 4.9e7) | 5.1e5 exp + ~5e7 flop (fast 1.4e5 exp + ~1.5e7) |
| per-cosmology setup | none (node cache once) | rebuild W, D: 192 x 1280 x 4 x 2 scatter-adds, 2.5e5 exp | none if grids fixed in Mpc; else rebuild (N_in FFT pairs) | kernel phase e^(-i eta (u_0+v_0)): 1001 sincos; M(b + i eta) once |
| memory | node cache 104 kB | W + D 5.7 MB, g 3.8 MB | G + GD 13 MB (fast 1.5 MB) | kernels 32 kB; per thread 2000 + 2 x 1001 complex (48 kB); batched 12 MB |
| threading | OpenMP over masses (today) or over a | BLAS, or OpenMP over (a, row) blocks with serial dot products | same as B1 | OpenMP over a, one single-threaded FFTW plan per transform (cosmo2D model) |
| bit-identical 1 vs 4 threads (measured, `--check-determinism`) | yes | **no** with OpenBLAS (1.8e-15) | **no** with OpenBLAS (1.8e-15); fast grid: yes | yes (batched and per-a) |

**When B1's weights must be rebuilt.** W_ij = sum_q wf_q k_q^(b-3)
L_j(ln k_q)/R_i^3 with k_q = x_q/R_i: the x-nodes and wf_q never change, but
R_i depends on rho_crit Omega_field (Omega_m, and Omega_nu for the cb
field) and the column grid is cosmolike's h/Mpc grid, which shifts with h
(the likelihood fixes log10 k in 1/Mpc). So W, D change at every MCMC step.
The fix is to work in Mpc units on a fixed ln R grid (rows) and interpolate
to R(M_i): then the matrix is built once per run, and the matrix is then
exactly B2 (a log-grid convolution is a Toeplitz matrix, which FFTLog
applies in N log N). p_lin's ln P interpolation is not linear in P, so no
fixed-weight method reproduces quad bit for bit; B1 uses 4-point Lagrange
on g, which removes quad's kink error (2.8e-6 instead of 2.4e-5).

**OpenBLAS is not deterministic across thread counts here.** The same
DGEMM differs by 1e-14 relative between OPENBLAS_NUM_THREADS = 1 and 4,
even with the thread count set to 1 at run time and fixed column blocks
(the library initializes differently). A deterministic B needs a
hand-written loop (OpenMP over output blocks, serial dot products), which
runs well below BLAS speed.

**cosmo2D.c's non-Limber FFTLog threading** (`cfftlog_ells_p1/p2`,
cosmo2D.c:8180-8870): plans are static, made once with `FFTW_ESTIMATE`
(deterministic algorithm choice) on the first buffers, rebuilt only when
the sizes or the thread count change. Phase 1 runs `omp parallel for
collapse(2) schedule(static)` over (radial bin, component), each iteration
one complete r2c through the new-array interface `fftw_execute_dft_r2c`
plus the c_window. Phase 2 runs, per bin, `omp parallel for collapse(2)`
over (component, multipole); each thread uses its own buffers
`outfwd[id]`, `outbcw[id]` and `fftw_execute_dft_c2r`. Parallelism is over
whole transforms, never inside one, so results do not depend on the thread
count. The same model for sigma^2: OpenMP over a nodes, one r2c and two c2r
plans, per-thread buffers; K and KD precomputed (fixed when the k and R grids
are fixed in Mpc).

**Expectation for N_a = 256 (to be confirmed by `timing.py`).** C wins.
Per a node it does ~1.6e5 flop of FFT plus 2000 exp, against 7.1e5 MAC
(B1) or 1.6e6 MAC (B2) of matrix product and 9.5e4 table reads with a log
and an exp each (A). Estimate in C, 4 threads: A ~ 180 ms (0.71 ms per a
from the cosmo3D.c header x 256), B1 ~ 10-20 ms with threaded BLAS (not
deterministic) and ~30-60 ms as a deterministic loop plus the per-cosmology
rebuild, C ~ 3-6 ms (fast: ~1-2 ms), dominated by the exp of the input,
which every method but A pays. C is also the most accurate (2e-9) and
deterministic by construction. A matrix only competes at small sizes
(B2 fast grid: 4.9e7 MAC) where BLAS outruns an FFT of 560 points, and it
then has the same accuracy as C fast.

## 4b. Measured timings (2026-10-02, quiet machine)

Apple M2 Pro laptop, macOS; Python 3.12, NumPy 2.4.3, SciPy 1.17.1, OpenBLAS
(the `ccl` conda environment); nothing else of the session running (load
average 2.5 to 3.5 from the system). Each entry is the median over 30 calls
after 3 warm-up calls; "setup" is the per-cosmology build. Times in ms for
N_a = 256 a nodes x 1024 ln M nodes, from the ln P(k, a) rows to
ln sigma^2 and d ln sigma/d ln M (`results/timing_T1.json`,
`results/timing_T4.json`, `results/determinism.log`).

| method | time, 1 thread (ms) | time, 4 threads (ms) | setup per cosmology (ms) | 1 vs 4 threads | error in sigma^2 (Sec. 4) |
|---|---|---|---|---|---|
| A quad (cosmolike's algorithm, per a node) | 1010 | 563 | 0 | bit-identical | 5.5e-5 |
| B1 quad-node weight matrix (OpenBLAS) | 14.3 | 10.3 | 24.8 (rebuilt every step) | differs by 1.8e-15 | 2.8e-6 |
| B2 FFTLog as a matrix (OpenBLAS) | 27.3 | 17.6 | once | differs by 1.8e-15 | 2.8e-9 |
| B2f same, fast grid | 6.7 | 6.0 | once | bit-identical | fast setting |
| C FFTLog batched over a (scipy.fft) | 18.7 | 19.2 | 0.5 | bit-identical | 1.9e-9 |
| Cf same, fast setting (N_fft = 560) | 6.6 | 6.5 | 0.1 | bit-identical | 1.3e-6 |
| Ct FFTLog threaded per a node (cosmo2D.c model) | 19.9 | 13.6 | 0.5 | bit-identical | 1.9e-9 |

These are Python timings: they rank the methods, they do not predict the C
cost. In Python, FFTLog (C, Ct) is 50 to 54 times faster than the quad at
one thread and 41 times at four (Ct), at 1e-9 instead of 5.5e-5, and bit-
identical across thread counts. B1 pays its 25 ms weight build every step
(R(M) moves with Omega_m and Omega_nu, the k grid with h).

## 5. Figures

| file | content |
|---|---|
| `figures/fftlog_recommended_error.png` | FFTLog (recommended) vs spline truth, sigma^2 (top) and d ln sigma/d ln M (bottom), z = 0, 1, 3, four spectra |
| `figures/quad_error.png` | quad (cosmolike default) vs both truths, and p_lin truth vs spline truth, P_cb mnu = 0.06 |
| `figures/fftlog_settings.png` | |error| vs M for recommended, fast, table only + padding, k_hi = 1e3, b = 3 |
| `figures/fftlog_kernel_and_spectrum.png` | |M(b + i eta)| for b = 1, 1.5, 3; input spectrum |c_m| and product |c_m M_m| |

## 6. How to run

Python: `/Users/vivianmiranda/miniforge/envs/ccl/bin/python` (numpy 2.4,
scipy 1.17, OpenBLAS). Do not source start_cocoa.sh.

    cd /Users/vivianmiranda/data/COCOA/september2026/test/neutrino_growth_study/sigma_fftlog
    PY=/Users/vivianmiranda/miniforge/envs/ccl/bin/python
    $PY accuracy.py                      # ~50 s first time (truth), ~10 s with the cache

Timing (quiet machine only; nothing else running):

    $PY timing.py --threads 1            # N_a = 1 16 64 256, 30 reps, 3 warm-ups excluded
    $PY timing.py --threads 4
    $PY timing.py --threads 8            # optional, the laptop's P cores
    $PY timing.py --check-determinism    # bitwise comparison 1 vs 4 threads (seconds)

Options: `--na 1 16 64 256`, `--reps 30`, `--warmup 3`,
`--methods A,B1,B2,B2f,C,Cf,Ct`, `--spectrum Pcb_mnu0.06`. The thread
count sets OMP/OpenBLAS/vecLib threads before numpy loads, the scipy.fft
workers (C, Cf), and the Python thread pool that runs A and Ct over
contiguous chunks of a. Output: a table (median [min, max] ms per call, and
the per-cosmology setup time) and `results/timing_T<threads>.json` with
every repetition. A full run at one thread count should take a few minutes,
mostly A. Caveats for reading it: the numpy quad evaluates 128 segments for
every mass (C stops at 14-89), the Python thread pool only overlaps the
numpy array operations, and B1's setup uses `np.add.at` (slow in Python,
~2e6 scatter-adds in C).
