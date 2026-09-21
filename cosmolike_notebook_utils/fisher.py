"""Fisher forecasting helpers shared by the EXAMPLE_EVALUATE notebooks.

A Fisher matrix forecasts parameter uncertainties from the
derivatives of the theory data vector: F = D^T C^-1 D, with D the
matrix whose column p is the derivative of the data vector with
respect to parameter p, and C the data covariance. Everything in
this module is pure numpy/getdist mathematics EXCEPT the data vector
itself, which needs a project's compiled cosmolike interface. That
dependency is injected: the functions here take the notebook's
data-vector function as their first argument (`dv`, called as
dv(param=<1D parameter array>, AccuracyBoost=<float>) and returning
the theory vector) or a derivative function built from it (`ddv`).
This module therefore never imports a cosmolike interface, and every
project shares it; each notebook keeps thin wrappers that bind its
own dv, fiducial values, priors, and covariance.

Contents:
  get_ddv             five-point finite-difference derivative of dv
  get_ddv_dkit        the same derivative via the derivkit package
  get_Fisher          F = D^T C^-1 D from a derivative function
  get_Fisher2         the same, with derivkit derivatives
  add_gaussian_priors add 1/sigma^2 to the diagonal for Gaussian
                      priors
  get_FoM             figure of merit of one parameter pair
  plot_Fisher         getdist triangle plot of Fisher contours,
                      optionally against MCMC chains
"""

import itertools

import numpy as np
from getdist import plots
from getdist.mcsamples import MCSamples, loadMCSamples


def get_ddv(dv, index=0, h=0.02, CV=None, AccuracyBoost=1.0):
    """Derivative of the data vector along one parameter (5-point rule).

    Four evaluations at +-h and +-2h around the fiducial, combined
    with the standard five-point-stencil weights (the center point
    has weight zero, so it is never evaluated). The step is RELATIVE
    (p0 * (1 + s)) so one h suits parameters of any magnitude; a
    parameter whose fiducial is zero falls back to an absolute step.

    Arguments:
      dv    = the data-vector function, called as
              dv(param=<1D array>, AccuracyBoost=AccuracyBoost).
      index = position of the differentiated parameter inside CV.
      h     = relative step size (0.02 = two percent).
      CV    = 1D array of fiducial parameter values.
      AccuracyBoost = forwarded to every dv evaluation.

    Returns:
      1D array: d(data vector)/d(parameter) at the fiducial.
    """
    assert h > 0.0, "h must be > 0"
    # the five-point-stencil weights (-1, 8, -8, 1)/12 paired with
    # the steps (2h, h, -h, -2h); zip walks the two arrays in step,
    # handing one (step, weight) pair per loop turn
    coeffs = np.array([-1.0/12.0, 8.0/12.0, -8.0/12.0, +1.0/12.0], dtype=np.float64)
    steps  = np.array([2*h, h, -h, -2*h], dtype=np.float64)
    p0 = float(CV[index])
    result = None
    for s, c in zip(steps, coeffs):
        # copy=True: p is an independent copy, so writing p[index]
        # below cannot touch the caller's CV array
        p = np.array(CV, dtype = np.float64, copy=True)
        # ternary a if cond else b: absolute step when the fiducial
        # is (numerically) zero, relative step otherwise
        p[index] = p0 + s if np.isclose(p0, 0.0, atol=1e-12) else p0 * (1.0 + s)
        vec = dv(param=p, AccuracyBoost=AccuracyBoost)
        # accumulate c * vec; the first turn initializes result
        result = c*vec if result is None else result + c*vec
    # divide by the actual step used: h for the absolute branch,
    # h * p0 for the relative one
    return result / (h if p0 == 0.0 else h * p0)


def get_ddv_dkit(dv, index=0, CV=None, AccuracyBoost=1.0,
                 min_samples=7, fallback_mode="poly_at_floor"):
    """Derivative of the data vector via the derivkit package.

    derivkit fits polynomials through adaptively chosen sample
    points and differentiates the fit, which is more robust than a
    fixed stencil when the data vector is noisy in the parameter.

    Arguments:
      dv    = the data-vector function (see get_ddv).
      index = position of the differentiated parameter inside CV.
      CV    = 1D array of fiducial parameter values.
      AccuracyBoost = forwarded to every dv evaluation.
      min_samples   = minimum number of sample points for the fit.
      fallback_mode = derivkit's strategy when the fit rejects the
              samples (see the derivkit documentation).

    Returns:
      1D array: d(data vector)/d(parameter) at the fiducial.
    """
    # imported here, not at module top: derivkit is optional and only
    # this function and get_Fisher2 need it
    from derivkit import adaptive_fit

    p0 = float(CV[index])
    # func is a one-argument closure over CV and dv: derivkit varies
    # x, and func rebuilds the full parameter array around it
    def func(x):
        p = np.array(CV, dtype = np.float64, copy=True)
        p[index] = x
        return dv(param=p, AccuracyBoost=AccuracyBoost)
    dk = adaptive_fit.AdaptiveFitDerivative(function=func, central_value=p0)
    return dk.compute(diagnostics=False, min_samples=min_samples, fallback_mode=fallback_mode)


def add_gaussian_priors(F, priors):
    """Add Gaussian priors to a Fisher matrix.

    A Gaussian prior of width sigma on parameter i adds 1/sigma^2 to
    F[i, i] and nothing else.

    Arguments:
      F      = Fisher matrix (2D square array), not modified.
      priors = {parameter index: (mean, sigma)}; only sigma enters a
               Fisher matrix (the mean shifts no curvature).

    Returns:
      a new Fisher matrix with the priors added.
    """
    Fp = F.copy()
    # np.fromiter builds an array straight from an iterator:
    # priors.keys() yields the parameter indices, and the generator
    # (v[1] for v in priors.values()) yields each (mean, sigma)
    # tuple's second entry, the sigma
    idxs = np.fromiter(priors.keys(), dtype=int)
    sigmas = np.fromiter((v[1] for v in priors.values()), dtype=float)
    # np.add.at adds in place at the listed positions; the index pair
    # (idxs, idxs) targets the diagonal entries (i, i) only
    np.add.at(Fp, (idxs, idxs), 1.0 / (sigmas ** 2))
    return Fp


def get_Fisher(ddv, CV=None, h=0.02, AccuracyBoost=3.1, priors=None,
               invcov=None):
    """Assemble F = D^T C^-1 D from a derivative function.

    Arguments:
      ddv   = derivative function, called as
              ddv(index=p, h=h, CV=CV, AccuracyBoost=AccuracyBoost)
              for every parameter p (the notebooks pass their get_ddv
              wrapper, which already carries dv).
      CV    = 1D array of fiducial parameter values; its length sets
              the number of columns of D.
      h     = relative step forwarded to ddv.
      AccuracyBoost = forwarded to ddv (derivatives want accurate
              vectors: differences amplify numerical noise).
      priors  = None, or {index: (mean, sigma)} Gaussian priors
              added on the diagonal.
      invcov  = inverse covariance of the data vector (2D array).

    Returns:
      the Fisher matrix (n_params x n_params array).
    """
    # the comprehension evaluates ddv once per parameter (each call
    # is len(steps) data-vector evaluations); np.column_stack glues
    # the returned 1D arrays side by side as the columns of D
    D = np.column_stack([ddv(index=p, h=h, CV=CV, AccuracyBoost=AccuracyBoost) for p in list(range(len(CV)))]).astype(np.float64)
    # @ is numpy matrix multiplication: D^T (C^-1 D)
    F = D.T @ (invcov @ D)
    if priors is not None:
        F = add_gaussian_priors(F=F, priors=priors)
    return F


def get_Fisher2(dv, CV=None, AccuracyBoost=3.0, priors=None,
                invcov=None, min_samples=7,
                fallback_mode="poly_at_floor"):
    """Assemble F = D^T C^-1 D with derivkit derivatives.

    Arguments:
      dv    = the data-vector function (see get_ddv); the derivkit
              derivative is built from it per parameter.
      CV    = 1D array of fiducial parameter values.
      AccuracyBoost = forwarded to every dv evaluation.
      priors  = None, or {index: (mean, sigma)} Gaussian priors.
      invcov  = inverse covariance of the data vector.
      min_samples, fallback_mode = forwarded to get_ddv_dkit.

    Returns:
      the Fisher matrix (n_params x n_params array).
    """
    D = np.column_stack([get_ddv_dkit(dv, index=p, CV=CV, AccuracyBoost=AccuracyBoost, min_samples=min_samples, fallback_mode=fallback_mode) for p in list(range(len(CV)))]).astype(np.float64)
    F = D.T @ (invcov @ D)
    if priors is not None:
        F = add_gaussian_priors(F=F, priors=priors)
    return F


def get_FoM(i, j, F):
    """Figure of merit of the parameter pair (i, j).

    FoM = 1/sqrt(det C_ij), with C_ij the 2x2 block of the parameter
    covariance (the inverse Fisher matrix) for parameters i and j: a
    larger FoM means a smaller error ellipse.

    Arguments:
      i, j = the two parameter indices.
      F    = Fisher matrix.

    Returns:
      the figure of merit as a float.
    """
    # 0.5 * (F + F^T) symmetrizes away numerical asymmetry before
    # inverting; np.ix_([i, j], [i, j]) selects the 2x2 block with
    # rows AND columns (i, j)
    C = np.linalg.inv(0.5 * (F + F.T))
    C = C[np.ix_([i, j], [i, j])]
    detC = np.linalg.det(C)
    if detC <= 0:
        # a numerically indefinite block: clip negative eigenvalues
        # to zero (eigvalsh = eigenvalues of a symmetric matrix) and
        # use the product of the clipped ones as the determinant
        w = np.clip(np.linalg.eigvalsh(C), 0, None)
        detC = w.prod()
    return float(1.0/np.sqrt(detC))


def plot_Fisher(F, mu, F2=None, root=None, select=None, labels=None,
                names=None, filled=True, flat_priors=None,
                chain_names=None, rng=None):
    """getdist triangle plot of Fisher contours, optionally vs MCMC.

    Each Fisher matrix is turned into a cloud of samples from the
    Gaussian it defines (mean mu, covariance F^-1), truncated to the
    flat-prior boxes, and handed to getdist, which draws the familiar
    triangle of 1D and 2D marginals. An MCMC chain loaded from root
    joins the same figure, so forecast and chain overlay directly.

    Arguments:
      F      = one Fisher matrix or a list of them (one contour set
               each).
      mu     = 1D array of fiducial parameter values (the Gaussian
               mean), full length; select cuts it down.
      F2     = a second matrix or list drawn after F (extra styles).
      root   = getdist chain root to load and overlay, or None.
      select = list of parameter indices to show, or None for all.
      labels = full-length list of LaTeX parameter labels.
      names  = full-length list of parameter names (getdist ids).
      filled = True fills the 2D contours.
      flat_priors = {parameter index: (min, max)} truncation boxes,
               or None for no truncation.
      chain_names = legend labels, one per drawn set, or None.
      rng    = numpy random Generator for the sample clouds; None
               creates a fresh seeded one (reproducible, but then
               repeated calls reuse identical sample noise).

    Returns:
      the getdist subplot plotter holding the figure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if flat_priors is None:
        flat_priors = {}
    if select is None:
        select = list(range(len(mu)))
    idx = list(select)

    # cut the mean and the label lists down to the selected
    # parameters; the comprehensions keep the order of idx
    mu = np.asarray(mu, float)[idx]
    names  = [names[k]  for k in idx]
    labels = [labels[k] for k in idx]

    # accept one matrix or a list of them: a bare matrix is wrapped
    # into a one-element list so the loop below covers both cases
    F = F if isinstance(F, (list, tuple)) else [F]

    samples = []
    for Fi in F:
        Fi = 0.5 * (Fi + Fi.T)
        C = np.linalg.inv(Fi)
        # np.ix_ selects the (idx x idx) block: marginalizing a
        # Gaussian over the dropped parameters IS taking the
        # covariance sub-block
        C = C[np.ix_(idx, idx)]
        # 40000 draws from the Gaussian the Fisher matrix defines
        y = rng.multivariate_normal(mu, C, size=40000)
        # one boolean column per flat prior: True where the draw sits
        # inside that parameter's box
        conds = [(y[:,j] >= lo) & (y[:,j] <= hi) for j, (lo, hi) in flat_priors.items()]
        # np.all(conds, axis=0) ands the columns together: a draw
        # survives only when EVERY box contains it; with no boxes,
        # np.ones(..., dtype=bool) keeps every draw
        valid = np.all(conds, axis=0) if conds else np.ones(y.shape[0], dtype=bool)
        # {name: [lo, hi]} tells getdist where hard prior edges sit,
        # so its density estimate does not smooth across them
        ranges = {names[j]: [lo, hi] for j, (lo, hi) in flat_priors.items()}
        x = MCSamples(samples=y[valid], names=names, ranges=ranges, labels=labels)
        samples.append(x)

    analysissettings={'smooth_scale_1D':0.1,
                      'smooth_scale_2D':0.1,
                      'ignore_rows': u'0.0',
                      'range_confidence' : u'0.005',
                      'fine_bins_2D': 2048,
                      'bins_2D': 1024,
                      'fine_bins_1D': 512,
                      'bins_1D': 256}

    if F2 is not None:
        F2 = F2 if isinstance(F2, (list, tuple)) else [F2]
        for Fi in F2:
            Fi = 0.5 * (Fi + Fi.T)
            C = np.linalg.inv(Fi)
            C = C[np.ix_(idx, idx)]
            y = rng.multivariate_normal(mu, C, size=40000)
            conds = [(y[:,j] >= lo) & (y[:,j] <= hi) for j, (lo, hi) in flat_priors.items()]
            valid = np.all(conds, axis=0) if conds else np.ones(y.shape[0], dtype=bool)
            ranges = {names[j]: [lo, hi] for j, (lo, hi) in flat_priors.items()}
            x = MCSamples(samples=y[valid], names=names, ranges=ranges, labels=labels)
            samples.append(x)

    if root:
        samples.append(loadMCSamples(root,settings={'ignore_rows': u'0.0'}))

    g = plots.get_subplot_plotter(width_inch=10.5,analysis_settings=analysissettings)
    g.settings.axis_tick_x_rotation=65
    g.settings.lw_contour=1.0
    g.settings.legend_rect_border = False
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 16.0
    g.settings.legend_fontsize = 16.5
    g.settings.alpha_filled_add = 0.85
    g.settings.lab_fontsize=15.5
    g.legend_labels=False

    # itertools.cycle repeats each style list forever: every drawn
    # set pulls the next color/dash/width, wrapping at the end
    cc  = itertools.cycle(["cornflowerblue", "darkorange", "seagreen", "firebrick", "purple", "gold"])
    lsc = itertools.cycle(["solid", "dashed", "dotted", "dashdot"])
    lwc = itertools.cycle([1.2, 1.5])

    line_args = []
    contour_cols = []
    contour_ls = []
    contour_lws = []

    for _ in samples:
        c  = next(cc)
        ls = next(lsc)
        lw = next(lwc)
        line_args.append({'lw': lw, 'ls': ls, 'color': c})
        contour_cols.append(c)
        contour_ls.append(ls)
        contour_lws.append(lw)

    g.triangle_plot(samples,
                    params=names,
                    filled=filled,
                    line_args=line_args,
                    contour_colors=contour_cols,
                    contour_ls=contour_ls,
                    contour_lws=contour_lws,
                   legend_labels=chain_names)
    return g
