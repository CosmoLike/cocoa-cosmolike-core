"""The shared machinery of the photo-z convention figure scripts.

Every project's tests/data_vector/generate_photoz_convention_figure.py
evaluates the frozen cosmic-shear fiducial under the four runtime
photo-z settings (cspline/Z_LOW default, linear, Steffen, Z_MID; see
the core README's shared data-vector checks for what they mean) and
plots the fractional data-vector differences against the default, per
tomographic pair and per angular bin. The machinery is the same in
every project, so it lives here; the thin script keeps the survey
data (binning, mask file, panel grid) and calls
photoz_convention_figures with it.

This module imports matplotlib, so the test shims never import it:
only the figure scripts do, directly, after their own
OMP_NUM_THREADS line.
"""

import os
import shutil
import tempfile

# "Agg" is matplotlib's file-only backend: figures are written to PNG
# files without opening a window, so the scripts also run on a machine
# without a display. It must be chosen before pyplot is imported.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (report tag, photoz_interpolation_type, photoz_zmid_convention); the
# first entry is the default every other setting is compared with
SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))

# The fixed y half-ranges of the two figures: fractional differences
# blow up where the signal crosses zero, and a fixed range keeps the
# flat offsets, which are the physics, readable (the spikes run
# off-panel). The Z_MID shift is percent-level; the interpolant
# differences are 1e-4-level.
ZMID_YLIM = 4.0
INTERP_YLIM = 8.0


def _datavectors(u, example):
    """Return the printed theory vector under each setting, keyed by tag.

    Builds one model per SETTINGS entry on the frozen fiducial, with
    print_datavector writing the theory vector into a temporary folder
    that is removed afterwards, also after an error.

    Arguments:
      u       = the project's cocoa_test_utils module.
      example = the key of u.EXAMPLES to evaluate.

    Returns:
      {tag: 1D float array of the full data-vector length}.
    """
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({example}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(example, tatt=False)
            block = info["likelihood"][u.EXAMPLES[example]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir,
                                f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, example, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def _plot(curves, path, title, grid, x, x_label, y_label, real,
          panel_fontsize, scale, ylim):
    """Draw one grid of glued panels, one per source pair, and save it.

    The layout of the notebook plotters: the source pairs (i <= j)
    fill the grid row by row, panels share both axes, and the bin
    labels count from 1. A panel the pair count does not reach is
    switched off, and the panel above a switched-off one carries the
    x label of its column.

    Arguments:
      curves         = {label: per-pair curve data}; in a real-space
                       figure each value is a (dxi_plus, dxi_minus)
                       pair of (npair, nlen) arrays, in a Fourier one
                       a single (npair, nlen) array. NaN marks masked
                       bins.
      path           = the PNG path to write.
      title          = figure title.
      grid           = (nrows, ncols) of the panel grid.
      x              = the angular-bin or band centers.
      x_label        = the x-axis label of the bottom row.
      y_label        = the y-axis label of the first column.
      real           = True for xi_pm curves (solid xi_+, dashed
                       xi_-), False for one C_l curve per label.
      panel_fontsize = the font size of the per-panel pair label.
      scale          = factor applied to the fractions before
                       plotting (100 for percent).
      ylim           = None, or the half-width of a fixed symmetric
                       y range in plotted units.

    Returns:
      nothing; the PNG file is written and the figure closed.
    """
    nrows, ncols = grid
    # the number of pairs comes from any curve's first dimension
    first = next(iter(curves.values()))
    npair = (first[0] if real else first).shape[0]
    # ntomo from npair = ntomo (ntomo + 1) / 2, solved by search
    ntomo = 1
    while ntomo * (ntomo + 1) // 2 < npair:
        ntomo += 1
    # every source pair (i, j) with i <= j, in data-vector order
    pairs = [(i, j) for i in range(ntomo) for j in range(i, ntomo)]
    # the figure height follows the row count (the widths the
    # projects always used: 7 for two rows, 9 for three, 16 beyond)
    height = {2: 7, 3: 9}.get(nrows, 16)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(20, height),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    # axes.ravel() lists the panels row by row; panel p shows pair p.
    # Only panel 0 labels its curves, so the legend lists each once.
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        for q, (label, data) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            if real:
                dp, dm = data
                ax.semilogx(x, scale * dp[p], color=color, lw=1.6,
                            label=rf"$\xi_+$: {label}"
                            if p == 0 else None)
                ax.semilogx(x, scale * dm[p], color=color, lw=1.6,
                            ls="--",
                            label=rf"$\xi_-$: {label}"
                            if p == 0 else None)
            else:
                ax.semilogx(x, scale * data[p], color=color, lw=1.6,
                            label=label if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$",
                transform=ax.transAxes, fontsize=panel_fontsize)
        # the bottom row carries the x labels; p + ncols >= npair
        # also catches a panel whose below-neighbor is switched off
        if p + ncols >= len(pairs):
            ax.set_xlabel(x_label, fontsize=16)
        if p % ncols == 0:
            ax.set_ylabel(y_label, fontsize=16 if real else 14)
    # a grid larger than the pair count leaves blank panels at the
    # end; switching them off removes their frames
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    if ylim is not None:
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.basename(path)}")


def photoz_convention_figures(u, out_dir, kind, ntomo, nlen,
                              x_min, x_max, mask_file,
                              grid, example="example1"):
    """Evaluate the four settings and write the two figures.

    Arguments:
      u         = the project's cocoa_test_utils module.
      out_dir   = the folder the PNG files go into (the figure
                  script's own folder, tests/data_vector/).
      kind      = "real" (xi_pm over theta) or "fourier" (C_l over
                  the bands).
      ntomo     = the number of source tomographic bins.
      nlen      = angular bins (real) or bands (fourier) per block.
      x_min     = the lower edge of the angular or band range.
      x_max     = the upper edge.
      mask_file = the scale-cut mask of the frozen contract (two
                  columns, entry index and 0/1; one column accepted).
      grid      = (nrows, ncols) of the panel grid; the pairs fill
                  it row by row.
      example   = the key of u.EXAMPLES to evaluate.

    Returns:
      nothing; photoz_zmid_*.png and photoz_interp_*.png are written
      into out_dir.

    Raises:
      RuntimeError when Cocoa is not activated; AssertionError when
      the frozen state does not match the manifest.
    """
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = _datavectors(u, example)

    # the mask file has two columns (entry index, 0/1); keep the
    # second
    mask = np.loadtxt(mask_file)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = ntomo * (ntomo + 1) // 2
    # entries of one block: xi_+ or xi_- in real space, the leading
    # cosmic-shear block in Fourier space
    nblock = npair * nlen

    real = kind == "real"
    edges = np.geomspace(x_min, x_max, nlen + 1)
    if real:
        # the area-weighted centers of the log-spaced bins,
        # (2/3)(t_max^3 - t_min^3)/(t_max^2 - t_min^2) per bin, the
        # convention of cosmolike's angular bins
        x = (2.0 / 3.0) * (edges[1:]**3 - edges[:-1]**3) \
                        / (edges[1:]**2 - edges[:-1]**2)
    else:
        # geometric band centers
        x = np.sqrt(edges[1:] * edges[:-1])

    def frac(tag):
        """One setting's fractional difference against the default.

        Real space returns the (dxi_plus, dxi_minus) pair, Fourier
        space the single C_l array, each (npair, nlen) with NaN at
        masked bins. errstate silences numpy's division warnings on
        masked entries (0 in both vectors), which np.where replaces
        by NaN anyway.
        """
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        out = []
        for s in range(2 if real else 1):
            sl = slice(s * nblock, (s + 1) * nblock)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(mask[sl] > 0, cur[sl] / ref[sl] - 1.0,
                             np.nan)
            out.append(d.reshape(npair, nlen))
        return tuple(out) if real else out[0]

    # the figure files and labels of the two spaces
    project = os.path.basename(os.path.dirname(u.TESTS_DIR))
    suffix = "dxi" if real else "dcl"
    x_label = r"$\theta$ [arcmin]" if real else r"$\ell$"
    y_unit_pct = "%"
    y_unit_1em4 = r"$10^{-4}$"
    if real:
        y_label_pct = rf"$\Delta\xi_\pm/\xi_\pm$ [{y_unit_pct}]"
        y_label_1em4 = rf"$\Delta\xi_\pm/\xi_\pm$ [{y_unit_1em4}]"
    else:
        y_label_pct = rf"$\Delta C_\ell/C_\ell$ [{y_unit_pct}]"
        y_label_1em4 = rf"$\Delta C_\ell/C_\ell$ [{y_unit_1em4}]"
    panel_fontsize = 15 if real else 13

    _plot({"Z_MID": frac("Z_MID")},
          os.path.join(out_dir, f"photoz_zmid_{suffix}.png"),
          f"{project} cosmic shear: n(z) z-column read as Z_MID "
          "instead of Z_LOW (frozen fiducial)",
          grid, x, x_label, y_label_pct, real, panel_fontsize,
          scale=100.0, ylim=ZMID_YLIM)
    _plot({"linear": frac("linear"), "steffen": frac("steffen")},
          os.path.join(out_dir, f"photoz_interp_{suffix}.png"),
          f"{project} cosmic shear: linear and Steffen n(z) "
          "interpolation vs cubic spline (frozen fiducial)",
          grid, x, x_label, y_label_1em4, real, panel_fontsize,
          scale=1.0e4, ylim=INTERP_YLIM)
