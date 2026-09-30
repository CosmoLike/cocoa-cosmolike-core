"""Cluster data-vector plots shared by the EXAMPLE_EVALUATE notebooks.

The cluster counterpart of plot_datavectors: one function per block
of the cluster data vectors (4x2pt + N, 6x2pt + N). Cluster counts
as one panel per cluster redshift bin (plot_N_cluster), cluster
lensing as a cluster z x source grid (plot_gammat_cluster_tomo for
gamma_t, plot_sigma_cluster_tomo for Sigma = Y gamma_t,
plot_C_cs_tomo_limber in harmonic space), cluster clustering as one
row of cluster z bins (plot_wcc_tomo, plot_C_cc_tomo_limber), and
cluster x galaxy clustering as one row of (cluster z, lens) pairs
(plot_wcg_tomo, plot_C_cg_tomo_limber).

The module is not part of the package namespace (cnu.*): import it
explicitly,

    from cosmolike_notebook_utils import plot_datavectors_cluster as pdc

The conventions of plot_datavectors hold here unchanged (value, or
value/reference - 1 with a *_ref argument, on glued panels; rescale
= 1 glues the absolute panels with a per-panel power of ten alpha;
tick labels near an interior panel boundary are hidden; an
identically zero panel is drawn "excluded"), and the glued-panel
helpers are imported from there, so cluster and galaxy figures look
alike. What the cluster blocks add is one more bin index, the
richness bin:

- A panel is a tomographic bin (or bin pair), as in the galaxy
  plots: (cluster z bin, source bin) for cluster lensing, the
  cluster z bin for cluster clustering and the counts, the
  (cluster z bin, lens bin) pair for cluster x galaxy. The richness
  bins (richness pairs for cluster clustering) are curves inside
  the panel. One panel per richness bin as well would need 48
  panels for 4 richness x 3 cluster z x 4 source bins; as curves,
  the same grid stays at 12 panels and the dependence on richness
  is read inside each one. The counts are the exception: richness
  is their x axis.
- Two things vary inside a panel, the list entry (the sweep) and
  the richness bin, and two line properties tell them apart. The
  color follows the list entry (the param value, as in the galaxy
  plots) and the line style follows the richness bin. With a single
  list entry there is no sweep to color, so the color follows the
  richness bin too; with a single richness bin drawn, the line
  styles (linestyle, linewidth, marker) follow the list entries,
  exactly as in the galaxy plots.
- richness (or pairs) selects which richness bins a figure draws,
  e.g. richness = 0 for the lowest bin only: a sweep then shows one
  curve per param value in each panel.
- data overlays measurements with error bars on the real-space
  blocks and on the counts; NaN entries (the scale cuts) are
  skipped.
- The entries of a list may sit on different theta grids (a change
  of the angular binning), as in the galaxy plots; a ratio needs
  every entry, and the data, on the reference's grid.

Bin arguments (richness, pairs) count from 0, as the arrays do; the
bin labels printed inside the panels count from 1, as in the galaxy
plots.

Pure matplotlib and numpy: nothing here touches CAMB or the compiled
cosmolike interface. Figure styling (fonts, usetex, rcParams) stays
in the notebooks; these functions only build the figures.
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from .plot_datavectors import (_align_log_ticklabels, _glued_supylabel,
                               _hide_glued_edge_ticklabels)

# one line style per richness curve of a panel, cycled when a panel
# holds more curves than styles
_RICHNESS_LINESTYLES = ['solid', 'dashed', 'dashdot', 'dotted']


def _select_bins(nbins, which):
    """The list of bin indices a richness selection stands for.

    Arguments:
      nbins = number of bins on the axis.
      which = None (every bin), one index, or a sequence of indices.

    Returns:
      list of int, or None when an index falls outside the axis.
    """
    if which is None:
        return list(range(nbins))
    # np.atleast_1d turns a bare index into a one-element array, so
    # one index and a list of indices take the same path
    sel = [int(x) for x in np.atleast_1d(which)]
    if len(sel) == 0 or min(sel) < 0 or max(sel) >= nbins:
        return None
    return sel


def _entry_colors(nentry, param, cmap, bar = None):
    """One color per list entry.

    With param the color is read at the entry's own position on the
    colorbar axis, so line and bar colors agree exactly (a sweep of
    five values would otherwise sit visibly off the bar); without it
    the entries are spread evenly over the colormap. In a figure
    with a colorbar the colors are read from the bar itself:
    matplotlib widens the range of a bar whose param values are all
    equal (a list of one entry), and only the bar holds the range it
    ended up with.

    Arguments:
      nentry = number of list entries (curves of the sweep).
      param  = the parameter values, or None.
      cmap   = colormap name.
      bar    = the colorbar _sweep_colorbar returned, or None for a
               figure without one.

    Returns:
      list of RGBA tuples, one per entry.
    """
    cm = plt.get_cmap(cmap)
    if param is None or len(param) != nentry:
        return [cm(x/nentry) for x in range(nentry)]
    if not (bar is None):
        # to_rgba applies the bar's own normalization and colormap
        return [bar.mappable.to_rgba(p) for p in param]
    norm = matplotlib.colors.Normalize(np.min(param), np.max(param))
    return [cm(norm(p)) for p in param]


def _sweep_colorbar(fig, axes, param, colorbarlabel, cmap, colorbarshrink):
    """The colorbar of a parameter sweep, next to the panels.

    Arguments:
      fig, axes = the figure and the array of its panels.
      param     = the parameter values coloring the curves.
      colorbarlabel = its label (LaTeX string), or None.
      cmap, colorbarshrink = colormap name and bar length as a
                  fraction of the panels' height.

    Returns:
      the matplotlib Colorbar, from which _entry_colors reads the
      colors of the list entries.
    """
    # the colorbar is not read off the plotted lines: it is drawn
    # from a ScalarMappable, a bare description of "this colormap
    # spans these values", with Normalize mapping the parameter
    # range onto the colormap's 0..1 axis (see _entry_colors)
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(np.min(param), np.max(param)), cmap = cmap),
        ax = np.ravel(axes).tolist(),
        orientation = 'vertical',
        aspect = 50,
        pad = 0.03,
        shrink = colorbarshrink)
    if not (colorbarlabel is None):
        cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
    return cb


def _proxy_legend(fig, axes, handles, labels, legendloc, legendfontsize):
    """One figure legend built from proxy handles.

    Arguments:
      fig, axes = the figure and the array of its panels.
      handles, labels = the legend keys and their texts.
      legendloc = None lays the entries in rows right above the
                  panels, centered on them; an (x, y) pair in figure
                  fractions places the lower-left corner anywhere.
      legendfontsize = entry font size in points, or None.
    """
    # legendloc None (the default) centers the legend on the panels'
    # measured span: np.ravel flattens the axes array, get_position
    # returns each panel's box in figure fractions, and
    # bbox_to_anchor pins the legend's lower-center point
    if legendloc is None:
        pos = [a.get_position() for a in np.ravel(axes)]
        cx = 0.5*(min(q.x0 for q in pos) + max(q.x1 for q in pos))
        ty = max(q.y1 for q in pos)
        legendloc = "lower center"
        legendanchor = (cx, ty + 0.008)
        # up to six keys share one row: four richness bins plus a
        # data key (or two list entries) still fit above the panels
        ncols = min(len(labels), 6)
    else:
        legendanchor = None
        ncols = 1
    fig.legend(
        handles,
        labels,
        loc=legendloc,
        bbox_to_anchor=legendanchor,
        ncols=ncols,
        fontsize=legendfontsize,
        borderpad=0.1,
        handletextpad=0.4,
        handlelength=2.2,
        columnspacing=1.0,
        frameon=False)


def _plot_cluster_panels(X, Y, X_ref, Y_ref, data, bintext, curvelabel, xlabel, ylabel,
                         ylabelglued, xlim, param, colorbarlabel, marker,
                         linestyle, linewidth, ylim, cmap, legend, legendloc,
                         richnesslegend, datalabel, yaxislabelsize,
                         yaxisticklabelsize, xaxisticklabelsize, xaxislabelsize,
                         bintextpos, bintextsize, figsize, show, colorbar,
                         colorbarshrink, markersize, rescale, alphatextpos,
                         ydecades, legendfontsize):
    """The panel grid every theta / ell cluster plotter draws.

    The public functions only slice their arrays into the layout
    below and pick the labels; the figure logic (glued panels,
    rescale, colorbar, styles, legends) lives here once.

    Arguments:
      X     = list of 1D arrays, one per list entry: the x axis of
              that entry (theta in arcmin, or ell). Entries may
              differ in their x grid (a change of binning) as long
              as no reference is given.
      Y     = list of 4D arrays (n_x, n_curve, n_col, n_row), one
              per list entry: n_curve richness curves inside each
              of the n_col x n_row panels.
      X_ref, Y_ref = None, or the x axis and one such array used as
              the ratio reference; every entry must then sit on the
              reference's x grid.
      data  = None, or (x, values, errors): the x axis of the
              measurements and arrays in the layout of one Y entry
              (errors may be None); NaN entries are skipped.
      bintext = n_col x n_row nested list of the panel labels.
      curvelabel = one legend label per richness curve.
      xlabel, ylabel, ylabelglued = axis labels: ylabel on the
              first column of the per-panel layout, ylabelglued
              once for the rescaled grid.
      xlim  = x-axis range.
      the remaining arguments = the options of the public
              functions, passed through unchanged.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      1D for a single row of panels, 2D (row, column) otherwise.
    """
    ncurve, ncol, nrow = Y[0].shape[1:]
    nentry = len(Y)

    if any(y.shape[0] != len(x) for (x, y) in zip(X, Y)):
        print("Bad Input (number of theta / ell)")
        return 0
    # shape[1:] drops the x axis: the bins must agree between the
    # entries, the x grid may differ
    if any(y.shape[1:] != Y[0].shape[1:] for y in Y):
        print("Bad Input (list entries with different bins)")
        return 0
    if not (Y_ref is None):
        if Y_ref.shape[1:] != Y[0].shape[1:] or Y_ref.shape[0] != len(X_ref):
            print("Bad Input (reference shape)")
            print(f"shape = {Y[0].shape}, shape_REF = {Y_ref.shape}")
            return 0
        if any(not np.array_equal(x, X_ref) for x in X):
            print("inconsistent theta / ell bins")
            return 0
    if not (param is None or colorbar is None) and len(param) != nentry:
        print("Bad Input (number of param values)")
        return 0
    if not (legend is None) and len(legend) != nentry:
        print("Bad Input (number of legend labels)")
        return 0
    if not (data is None):
        if (np.shape(data[1])[1:] != Y[0].shape[1:]
                or np.shape(data[1])[0] != len(data[0])):
            print("Bad Input (data shape)")
            return 0
        if not (Y_ref is None) and not np.array_equal(data[0], X_ref):
            print("inconsistent theta / ell bins (data)")
            return 0

    # rescale=1: alpha[i,j] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued grid. Panels
    # that are identically zero keep alpha = 0 and are skipped.
    rescale = None if not (Y_ref is None) else rescale
    alpha = np.zeros((ncol, nrow))
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(ncol):
            for j in range(nrow):
                v = np.abs(np.concatenate([y[:,:,i,j].ravel() for y in Y]))
                pmax = np.max(v)
                if pmax == 0:
                    continue
                alpha[i,j] = -np.floor(np.log10(pmax))
                # the glued floor only counts positive values: a curve
                # touching zero (the last angular bin of Sigma) cannot
                # set a log-axis lower limit
                panlo.append(np.min(v[v > 0])*10.0**alpha[i,j])
                panhi.append(pmax*10.0**alpha[i,j])
        if not panhi:
            print("Bad Input (every panel is identically zero)")
            return 0
        yglued = [ylim[0]*np.min(panlo), ylim[1]*np.max(panhi)]
        if not (ydecades is None):
            yglued[0] = max(yglued[0], yglued[1]/10.0**ydecades)

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge. squeeze=False keeps axes 2D
    # even for a single row, so one loop serves rows and grids.
    glued = not (Y_ref is None and rescale is None)
    if not glued:
        fig, axes = plt.subplots(
            nrows = nrow,
            ncols = ncol,
            figsize = figsize,
            sharex = True,
            sharey = False,
            squeeze = False,
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = nrow,
            ncols = ncol,
            figsize = figsize,
            sharex = True,
            sharey = True,
            squeeze = False,
            gridspec_kw = {'wspace': 0, 'hspace': 0})

    bar = None
    if not (param is None or colorbar is None):
        bar = _sweep_colorbar(fig, axes, param, colorbarlabel, cmap, colorbarshrink)

    # color follows the list entry, line style the richness curve;
    # with one entry the color follows the richness curve, with one
    # richness curve the styles follow the entries (module docstring)
    cm = plt.get_cmap(cmap)
    entrycolor = _entry_colors(nentry, param, cmap, bar)
    curvecolor = [cm(c/ncurve) for c in range(ncurve)]
    if linestyle is None:
        linestyle = _RICHNESS_LINESTYLES if ncurve > 1 else ['solid']
    if linewidth is None:
        linewidth = [1.0]

    def style(e, c):
        # k is the index the line styles follow; % wraps it around,
        # so a list shorter than the curves is cycled
        k = c if ncurve > 1 else e
        color = entrycolor[e] if nentry > 1 else curvecolor[c]
        if marker is None:
            return dict(color = color,
                        linewidth = linewidth[k % len(linewidth)],
                        linestyle = linestyle[k % len(linestyle)])
        return dict(color = color,
                    markerfacecolor = 'None',
                    marker = marker[k % len(marker)],
                    markeredgecolor = color,
                    linestyle = 'None',
                    markersize = markersize)

    # axes[j,i] is the panel at row j and column i
    for i in range(ncol):
        for j in range(nrow):
            ax = axes[j,i]
            ax.set_xlim(xlim)

            # a panel whose curves (or reference) are identically zero
            # holds a pair the dataset does not have: it gets an
            # "excluded" placeholder, since zeros can be neither log
            # scaled nor used as a ratio reference
            excluded = all(not np.any(y[:,:,i,j]) for y in Y)
            if not (Y_ref is None):
                excluded = excluded or not np.any(Y_ref[:,:,i,j])

            if Y_ref is None:
                if not (rescale is None):
                    ax.set_ylim(yglued)
                    ax.set_yscale('log')
                elif excluded:
                    ax.set_yticks([])
                else:
                    # per-panel range [ylim[0]*min, ylim[1]*max] over
                    # the positive values of every curve in the panel
                    v = np.abs(np.concatenate([y[:,:,i,j].ravel() for y in Y]))
                    ax.set_ylim([ylim[0]*np.min(v[v > 0]), ylim[1]*np.max(v)])
                    ax.set_yscale('log')
            else:
                # with a reference every curve is value/reference - 1, so ylim
                # (multipliers around 1) is drawn as the band ylim - 1 around 0
                ax.set_ylim([ylim[0] - 1.0, ylim[1] - 1.0])
                ax.set_yscale('linear')

            ax.set_xscale('log')

            if i == 0:
                if Y_ref is None:
                    # with rescale the y label is global: one fig.supylabel
                    if rescale is None:
                        ax.set_ylabel(ylabel, fontsize=yaxislabelsize)
                else:
                    ax.set_ylabel("frac. diff.", fontsize=yaxislabelsize)
            # which='both' also sizes the minor tick numbers a log
            # axis prints when a panel spans less than a decade
            ax.tick_params(axis='y', which='both', labelsize=yaxisticklabelsize)
            ax.tick_params(axis='x', which='both', labelsize=xaxisticklabelsize)

            if j == nrow-1:
                ax.set_xlabel(xlabel, fontsize=xaxislabelsize)

            # transform=transAxes puts the text in panel fractions: (0, 0)
            # is the panel's lower-left corner, (1, 1) its upper-right
            ax.text(bintextpos[0], bintextpos[1],
                bintext[i][j],
                horizontalalignment = 'center',
                verticalalignment = 'center',
                fontsize = bintextsize,
                usetex = True,
                transform = ax.transAxes)

            if excluded:
                ax.text(0.5, 0.5, "excluded",
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    transform = ax.transAxes)
                continue

            if not (rescale is None):
                expo = int(alpha[i,j])
                ax.text(alphatextpos[0], alphatextpos[1],
                    "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                    horizontalalignment = 'left',
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    usetex = True,
                    transform = ax.transAxes)

            # 10**alpha = 1 unless rescale is on for this panel
            fac = 10.0**alpha[i,j]
            for c in range(ncurve):
                if not (Y_ref is None):
                    ref = Y_ref[:,c,i,j]
                    # a zero of the reference (the last angular bin of
                    # Sigma) has no ratio: NaN breaks the line there
                    ref = np.where(ref != 0, ref, np.nan)
                for e, y in enumerate(Y):
                    if Y_ref is None:
                        tmp = np.abs(y[:,c,i,j])*fac
                        # zeros cannot sit on a log axis: NaN skips them
                        tmp = np.where(tmp > 0, tmp, np.nan)
                    else:
                        tmp = y[:,c,i,j]/ref - 1
                    ax.plot(X[e], tmp, **style(e, c))

                if not (data is None):
                    d = np.asarray(data[1])[:,c,i,j]
                    err = None if data[2] is None else np.asarray(data[2])[:,c,i,j]
                    if Y_ref is None:
                        d = np.abs(d)*fac
                        err = None if err is None else err*fac
                    else:
                        d = d/ref - 1
                        err = None if err is None else err/np.abs(ref)
                    # the scale cuts arrive as NaN: only finite points
                    # are drawn. With a sweep the data are black, else
                    # they take the color of their richness curve.
                    keep = np.isfinite(d)
                    ax.errorbar(np.asarray(data[0])[keep], d[keep],
                                yerr = None if err is None else err[keep],
                                fmt = 'o',
                                color = 'black' if nentry > 1 else curvecolor[c],
                                markersize = markersize,
                                capsize = 2,
                                elinewidth = 0.8)

    if not (rescale is None):
        # the minus sign otherwise staggers the stacked y numbers
        _align_log_ticklabels(axes[:,0])
        _glued_supylabel(fig, axes[:,0], ylabelglued, yaxislabelsize)

    if glued:
        # glued panels put a neighbor's edge tick number on the same
        # spot: prune the y labels at interior row boundaries and the
        # x labels at interior column boundaries. matplotlib only
        # computes tick positions when it needs them (at draw time,
        # or when the labels are asked for), and the pruning reads
        # them, so the labels are asked for first.
        for ax in axes[:,0]:
            ax.get_yticklabels()
        for ax in axes[nrow-1,:]:
            ax.get_xticklabels()
        if nrow > 1:
            if Y_ref is None:
                _hide_glued_edge_ticklabels(
                    [(axes[j,0], j == nrow-1, j == 0) for j in range(nrow)],
                    yglued[0], yglued[1])
            else:
                _hide_glued_edge_ticklabels(
                    [(axes[j,0], j == nrow-1, j == 0) for j in range(nrow)],
                    ylim[0]-1.0, ylim[1]-1.0, log = False)
        if ncol > 1:
            _hide_glued_edge_ticklabels(
                [(axes[nrow-1,i], i == 0, i == ncol-1) for i in range(ncol)],
                xlim[0], xlim[1], axis = "x")

    # the legend keys what the line properties mean, not the plotted
    # lines: Line2D([], []) is a line with no data points, it never
    # draws inside a panel and exists only as a legend key (a "proxy
    # handle" in matplotlib terms). One key per list entry when
    # legend is given, then one per richness curve.
    handles, labels = [], []
    if not (legend is None):
        for e in range(nentry):
            handles.append(matplotlib.lines.Line2D([], [], **style(e, 0)))
            labels.append(legend[e])
    if not (richnesslegend is None) and ncurve > 1:
        for c in range(ncurve):
            key = style(0, c)
            if nentry > 1:
                # the color belongs to the sweep: a black key shows
                # the line style alone
                key['color'] = 'black'
                if not (marker is None):
                    key['markeredgecolor'] = 'black'
            handles.append(matplotlib.lines.Line2D([], [], **key))
            labels.append(curvelabel[c])
    if not (data is None or datalabel is None):
        handles.append(matplotlib.lines.Line2D([], [], color = 'black',
                           marker = 'o', linestyle = 'None', markersize = markersize))
        labels.append(datalabel)
    if handles:
        _proxy_legend(fig, axes, handles, labels, legendloc, legendfontsize)

    # a single row is returned 1D, as the galaxy row plotters do
    axes = axes[0,:] if nrow == 1 else axes
    # warn=False: outside a notebook, showing a figure on a
    # non-interactive backend would otherwise warn
    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def plot_N_cluster(N, N_ref = None, param = None, colorbarlabel = None, richness_edges = None,
                   data = None, marker = None, linestyle = None, linewidth = None,
                   ylim = [0.75,1.25], cmap = 'gist_rainbow', legend = None, legendloc = None,
                   datalabel = None, yaxislabelsize = 16, yaxisticklabelsize = 10,
                   xaxisticklabelsize = 14, bintextpos = [0.8, 0.85], bintextsize = 15,
                   figsize = (16, 5), show = 1, colorbar = 1, colorbarshrink = 1.0,
                   markersize = 4, ylabel = r"$N$", legendfontsize = None, xaxislabelsize = 16):
    """One panel per cluster redshift bin of the cluster counts.

    The x axis is the richness: each curve is a staircase with one
    step per richness bin. Without N_ref each curve is the count N
    on a log scale; with N_ref each curve is the fractional
    difference N / N_ref - 1. The counts of the redshift bins are
    of the same order, so the row is glued in both cases and shares
    one y-range (no per-panel rescale).

    Arguments:
      N        = list of 2D arrays (n_richness, n_cluster_z), one
                 per curve, as the notebook N_cluster wrapper
                 returns them.
      N_ref    = None, or one 2D array used as the ratio reference.
      param    = list of parameter values (one per curve) coloring
                 the curves and the colorbar, or None.
      colorbarlabel = colorbar label (LaTeX string), or None.
      richness_edges = the n_richness + 1 edges of the richness
                 bins: the steps then span the bins on a log axis
                 with one tick per edge. None (default) draws the
                 steps against the bin number.
      data     = None, or (values, errors): 2D arrays in the layout
                 of one N entry, drawn as points with error bars at
                 the bin centers (errors may be None); NaN entries
                 are skipped. datalabel = its legend label, or None
                 for no legend entry.
      marker   = list of matplotlib markers cycled across curves
                 (points at the bin centers instead of steps), or
                 None for steps.
      linestyle, linewidth = lists cycled across curves, or None.
      ylim     = without N_ref, multipliers on the min/max of the
                 counts; with it, the band around 1 (drawn as
                 ylim - 1).
      legend   = one label per curve, or None. legendloc = None (the
                 default) lays the legend right above the panels,
                 centered on them; an (x, y) pair in figure
                 fractions places its lower-left corner anywhere.
                 legendfontsize = entry font size in points; None
                 follows matplotlib's legend.fontsize rcParam.
      ylabel   = y-axis label without N_ref.
      yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize = axis-label and tick-number font sizes in
                 points.
      cmap, colorbarshrink, markersize, bintextpos, bintextsize,
      figsize = matplotlib layout knobs.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """

    N = [np.asarray(n) for n in N]
    nrichness, ncluster = N[0].shape
    nentry = len(N)

    if any(n.shape != N[0].shape for n in N):
        print("Bad Input (list entries of different shapes)")
        return 0
    if not (N_ref is None):
        N_ref = np.asarray(N_ref)
        if N_ref.shape != N[0].shape:
            print("Bad Input")
            print(f"Nrichness = {nrichness}, Nrichness_REF = {N_ref.shape[0]}")
            print(f"Ncluster = {ncluster}, Ncluster_REF = {N_ref.shape[-1]}")
            return 0
    if not (richness_edges is None) and len(richness_edges) != nrichness + 1:
        print("Bad Input (number of richness edges)")
        return 0
    if not (param is None or colorbar is None) and len(param) != nentry:
        print("Bad Input (number of param values)")
        return 0
    if not (legend is None) and len(legend) != nentry:
        print("Bad Input (number of legend labels)")
        return 0
    if not (data is None) and np.shape(data[0]) != N[0].shape:
        print("Bad Input (data shape)")
        return 0

    # the step edges and the points' x positions: the richness edges
    # with geometric bin centers (log axis), or the bin numbers
    if richness_edges is None:
        edges = np.arange(nrichness + 1) + 0.5
        center = np.arange(nrichness) + 1.0
    else:
        edges = np.asarray(richness_edges, dtype = float)
        center = np.sqrt(edges[:-1]*edges[1:])

    # the row is glued (zero wspace) and shares its y axis in both
    # modes: without a reference the range covers every redshift bin
    fig, axes = plt.subplots(
        nrows = 1,
        ncols = ncluster,
        figsize = figsize,
        sharex = True,
        sharey = True,
        squeeze = False,
        gridspec_kw = {'wspace': 0, 'hspace': 0})
    axes = axes[0,:]

    bar = None
    if not (param is None or colorbar is None):
        bar = _sweep_colorbar(fig, axes, param, colorbarlabel, cmap, colorbarshrink)

    color = _entry_colors(nentry, param, cmap, bar)
    if linestyle is None:
        linestyle = ['solid']
    if linewidth is None:
        linewidth = [1.5]

    if N_ref is None:
        yrange = [ylim[0]*min(np.min(n) for n in N), ylim[1]*max(np.max(n) for n in N)]
    else:
        # with a reference every curve is value/reference - 1, so ylim
        # (multipliers around 1) is drawn as the band ylim - 1 around 0
        yrange = [ylim[0] - 1.0, ylim[1] - 1.0]

    # one panel per cluster redshift bin in a single row: axes[i] is
    # column i, and the [:,i] slices read the counts of that bin
    for i in range(ncluster):
        ax = axes[i]
        ax.set_xlim([edges[0], edges[-1]])
        ax.set_ylim(yrange)
        ax.set_yscale('log' if N_ref is None else 'linear')
        if richness_edges is None:
            ax.set_xticks(center)
        else:
            # one plain number per richness edge instead of the powers
            # of ten of a log axis (the edges span about one decade)
            ax.set_xscale('log')
            ax.set_xticks(edges)
            ax.set_xticklabels(["%g" % x for x in edges])
            ax.minorticks_off()

        if i == 0:
            ax.set_ylabel(ylabel if N_ref is None else "frac. diff.", fontsize=yaxislabelsize)
        ax.tick_params(axis='y', which='both', labelsize=yaxisticklabelsize)
        ax.tick_params(axis='x', which='both', labelsize=xaxisticklabelsize)
        ax.set_xlabel("richness bin" if richness_edges is None else r"$\lambda$", fontsize=xaxislabelsize)

        # transform=transAxes puts the text in panel fractions: (0, 0)
        # is the panel's lower-left corner, (1, 1) its upper-right
        ax.text(bintextpos[0], bintextpos[1],
            "$(" + str(i+1) + ")$",
            horizontalalignment = 'center',
            verticalalignment = 'center',
            fontsize = bintextsize,
            usetex = True,
            transform = ax.transAxes)

        for e, n in enumerate(N):
            tmp = n[:,i] if N_ref is None else n[:,i]/N_ref[:,i] - 1
            if marker is None:
                # stairs draws one horizontal step per bin between
                # consecutive edges; baseline=None leaves out the
                # vertical drops to zero at both ends
                ax.stairs(tmp, edges,
                          baseline = None,
                          color = color[e],
                          linewidth = linewidth[e % len(linewidth)],
                          linestyle = linestyle[e % len(linestyle)])
            else:
                ax.plot(center, tmp,
                        color = color[e],
                        markerfacecolor = 'None',
                        marker = marker[e % len(marker)],
                        markeredgecolor = color[e],
                        linestyle = 'None',
                        markersize = markersize)

        if not (data is None):
            d = np.asarray(data[0])[:,i]
            err = None if data[1] is None else np.asarray(data[1])[:,i]
            if not (N_ref is None):
                d = d/N_ref[:,i] - 1
                err = None if err is None else err/np.abs(N_ref[:,i])
            keep = np.isfinite(d)
            ax.errorbar(center[keep], d[keep],
                        yerr = None if err is None else err[keep],
                        fmt = 'o',
                        color = 'black',
                        markersize = markersize,
                        capsize = 3,
                        elinewidth = 0.8)

    if N_ref is None and np.log10(yrange[1]/yrange[0]) <= 3.0:
        # counts read best as plain numbers: ticks at 1, 2 and 5
        # times each power of ten, written out in full (a log axis
        # would otherwise label only the powers of ten, often a
        # single one over the range of the counts). The panels share
        # their y axis, so setting the first one sets them all.
        axes[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = (1.0, 2.0, 5.0)))
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, pos: "%g" % v))
        axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # the row is glued horizontally, so the clash is between the x
    # tick labels at interior panel boundaries. matplotlib only
    # computes tick positions when it needs them, and the pruning
    # reads them, so the labels are asked for first.
    for ax in axes:
        ax.get_xticklabels()
    _hide_glued_edge_ticklabels(
        [(axes[i], i == 0, i == ncluster-1) for i in range(ncluster)],
        edges[0], edges[-1], axis = "x", log = not (richness_edges is None))

    # proxy handles (lines without data points) key the curves and
    # the data, since a staircase has no single line to point at
    handles, labels = [], []
    if not (legend is None):
        for e in range(nentry):
            if marker is None:
                handles.append(matplotlib.lines.Line2D([], [], color = color[e],
                                   linewidth = linewidth[e % len(linewidth)],
                                   linestyle = linestyle[e % len(linestyle)]))
            else:
                handles.append(matplotlib.lines.Line2D([], [], color = color[e],
                                   markerfacecolor = 'None', marker = marker[e % len(marker)],
                                   linestyle = 'None', markersize = markersize))
            labels.append(legend[e])
    if not (data is None or datalabel is None):
        handles.append(matplotlib.lines.Line2D([], [], color = 'black',
                           marker = 'o', linestyle = 'None', markersize = markersize))
        labels.append(datalabel)
    if handles:
        _proxy_legend(fig, axes, handles, labels, legendloc, legendfontsize)

    # warn=False: outside a notebook, showing a figure on a
    # non-interactive backend would otherwise warn
    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def _lensing_panels(theta_y, y_ref, data, richness, richnesslabel, thetashow,
                    ylabel, ylabelglued, **options):
    """Slices cluster lensing arrays into panels and draws them.

    Shared by plot_gammat_cluster_tomo and plot_sigma_cluster_tomo,
    which differ only in their y labels.

    Arguments:
      theta_y = list of (theta, y) pairs, y a 4D array (n_theta,
                n_richness, n_cluster_z, n_source).
      y_ref   = None, or one such pair used as the ratio reference.
      data    = None, or (theta, values, errors) in the same layout.
      richness, richnesslabel, thetashow = as in the public
                functions.
      ylabel, ylabelglued = the y labels of the two absolute
                layouts.
      options = the remaining options of _plot_cluster_panels.

    Returns:
      what _plot_cluster_panels returns.
    """
    (theta, y) = theta_y[0]
    ntheta, nrichness, ncluster, nsource = np.shape(y)

    if ntheta != len(theta):
        print("Bad Input (theta)")
        print(theta)
        print(ntheta, len(theta))
        return 0
    sel = _select_bins(nrichness, richness)
    if sel is None:
        print("Bad Input (richness bin outside the array)")
        return 0
    if not (richnesslabel is None) and len(richnesslabel) != nrichness:
        print("Bad Input (number of richness labels)")
        return 0

    if thetashow is None:
        thetashow = [np.min(theta), np.max(theta)]

    # the panel layout (n_x, n_curve, n_col, n_row) is the array's
    # own axis order: richness curves, cluster z columns, source
    # rows; only the richness selection is applied
    Y = [np.asarray(g)[:,sel,:,:] for (t, g) in theta_y]
    Y_ref = None if y_ref is None else np.asarray(y_ref[1])[:,sel,:,:]
    if not (data is None):
        data = (data[0], np.asarray(data[1])[:,sel,:,:],
                None if data[2] is None else np.asarray(data[2])[:,sel,:,:])
    if richnesslabel is None:
        richnesslabel = [r"$\lambda$ bin %d" % (nl+1) for nl in range(nrichness)]

    return _plot_cluster_panels(
        X = [np.asarray(t) for (t, g) in theta_y], Y = Y,
        X_ref = None if y_ref is None else np.asarray(y_ref[0]), Y_ref = Y_ref,
        data = data,
        bintext = [["$(" + str(i+1) + "," + str(j+1) + ")$" for j in range(nsource)]
                   for i in range(ncluster)],
        curvelabel = [richnesslabel[nl] for nl in sel],
        xlabel = r"$\theta$ [arcmin]", ylabel = ylabel, ylabelglued = ylabelglued,
        xlim = thetashow, **options)


def plot_gammat_cluster_tomo(theta_gammat, gammat_ref = None, param = None, colorbarlabel = None,
                             richness = None, data = None, marker = None,
                             linestyle = None, linewidth = None, ylim = [0.75,1.25],
                             cmap = 'gist_rainbow', legend = None, legendloc = None,
                             richnesslabel = None, richnesslegend = 1, datalabel = None,
                             yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                             bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 13),
                             show = 1, colorbar = 1, colorbarshrink = 0.5, markersize = 3,
                             thetashow = None, rescale = None, alphatextpos = [0.05, 0.12],
                             ydecades = 4, ylabel = r"$\alpha\,|\gamma_{t}(\theta)|$",
                             legendfontsize = None, xaxislabelsize = 16):
    """Panel grid of the cluster tangential shear gamma_t(theta).

    One panel per (cluster z bin, source bin) pair: columns are
    cluster redshift bins, rows are source bins, and the label
    inside each panel reads (cluster z bin, source bin). The
    richness bins are curves inside each panel, told apart by their
    line style (see the module docstring for why they are curves
    and not panels). Without gammat_ref each curve is |gamma_t| on
    a log scale; with gammat_ref each curve is the fractional
    difference gamma_t / ref - 1.

    Arguments:
      theta_gammat = list of (theta, gammat) pairs, one per list
                 entry, as the notebook gamma_t_cluster wrapper
                 returns them: theta in arcmin, gammat a 4D array
                 (n_theta, n_richness, n_cluster_z, n_source). The
                 entries may sit on different theta grids when no
                 reference is given.
      gammat_ref = None, or one (theta, gammat) pair used as the
                 ratio reference; every entry (and the data) must
                 then sit on its theta grid.
      param    = list of parameter values (one per list entry)
                 coloring the curves and the colorbar, or None.
      colorbarlabel = colorbar label (LaTeX string), or None.
      richness = which richness bins are drawn, counted from 0:
                 None (default) draws every bin, an index draws one
                 bin, a list draws those bins.
      data     = None, or (theta, values, errors): measurements in
                 the layout of one gammat array, drawn as points
                 with error bars (errors may be None); NaN entries
                 (the scale cuts) are skipped. datalabel = its
                 legend label, or None for no legend entry.
      marker   = list of matplotlib markers (points instead of
                 lines), or None for lines.
      linestyle, linewidth = lists cycled across the richness
                 curves of a panel, or None: the default line style
                 is solid, dashed, dashdot, dotted from the lowest
                 richness bin up. With a single richness bin drawn,
                 these lists and marker are cycled across the list
                 entries instead, as in the galaxy plots.
      ylim     = without gammat_ref, multipliers on each panel's
                 min/max; with it, the band around 1 (drawn as
                 ylim - 1).
      thetashow = x-axis range in arcmin; None (default) spans the
                 theta array itself.
      legend   = one label per list entry, or None.
      richnesslabel = one label per richness bin of the array for
                 the legend of the line styles; None (default)
                 writes "lambda bin n". richnesslegend = None
                 suppresses that legend.
      legendloc = None (the default) lays the legend right above
                 the panels, centered on them; an (x, y) pair in
                 figure fractions places its lower-left corner
                 anywhere. legendfontsize = entry font size in
                 points; None follows matplotlib's legend.fontsize
                 rcParam.
      yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize = axis-label and tick-number font sizes in
                 points.
      cmap, colorbarshrink, markersize, bintextpos, bintextsize,
      figsize = matplotlib layout knobs.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 every panel then shares one y-range, interior y
                 axes disappear and the panels are glued together,
                 with the factor alpha annotated inside each panel
                 and the y-axis label reading alpha |gamma_t|.
                 Ignored with gammat_ref (already dimensionless and
                 shared). None (default) keeps per-panel y-ranges.
      alphatextpos = axes-fraction (x, y) anchoring the left edge
                 of the alpha annotation.
      ydecades = with rescale, cap on how many decades the shared
                 y-range extends below its ceiling (default 4).
                 None keeps the full union of the panel ranges.
      ylabel   = with rescale, the single global y-axis label of
                 the glued grid (drawn once with fig.supylabel).

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 2D array of panels, [source bin, cluster z bin].
    """
    return _lensing_panels(
        theta_gammat, gammat_ref, data, richness, richnesslabel, thetashow,
        ylabel = r"$|\gamma_{t}(\theta)|$", ylabelglued = ylabel,
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = datalabel, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)


def plot_sigma_cluster_tomo(theta_sigma, sigma_ref = None, param = None, colorbarlabel = None,
                            richness = None, data = None, marker = None,
                            linestyle = None, linewidth = None, ylim = [0.75,1.25],
                            cmap = 'gist_rainbow', legend = None, legendloc = None,
                            richnesslabel = None, richnesslegend = 1, datalabel = None,
                            yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                            bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 13),
                            show = 1, colorbar = 1, colorbarshrink = 0.5, markersize = 3,
                            thetashow = None, rescale = None, alphatextpos = [0.05, 0.12],
                            ydecades = 4, ylabel = r"$\alpha\,|\Sigma(\theta)|$",
                            legendfontsize = None, xaxislabelsize = 16):
    """Panel grid of the cluster lensing statistic Sigma(theta).

    Sigma = Y gamma_t is cluster lensing as the data vector holds
    it: the Y transform of gamma_t, times the selection bias and
    the shear calibration. The layout is the one of
    plot_gammat_cluster_tomo: one panel per (cluster z bin, source
    bin) pair (columns are cluster redshift bins, rows are source
    bins), the richness bins as curves told apart by their line
    style. Without sigma_ref each curve is |Sigma| on a log scale;
    with sigma_ref each curve is the fractional difference
    Sigma / ref - 1. The last angular bin of Sigma is identically
    zero (the Y transform has no value there): it is left out of
    every curve and of the ratio.

    Arguments:
      theta_sigma = list of (theta, sigma) pairs, one per list
                 entry, as the notebook sigma_cluster wrapper
                 returns them: theta in arcmin, sigma a 4D array
                 (n_theta, n_richness, n_cluster_z, n_source).
      sigma_ref = None, or one (theta, sigma) pair used as the
                 ratio reference.
      data     = None, or (theta, values, errors): the cluster
                 lensing block of the data vector in the layout of
                 one sigma array, drawn as points with error bars;
                 NaN entries (the scale cuts) are skipped.
      rescale  = 1 glues the absolute panels with a per-panel power
                 of ten alpha, the y-axis label reading
                 alpha |Sigma| (the ylabel argument).
      param, colorbarlabel, richness, datalabel, marker, linestyle,
      linewidth, ylim, thetashow, legend, richnesslabel,
      richnesslegend, legendloc, legendfontsize, the *size
      arguments (yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize), cmap, colorbarshrink, markersize,
      bintextpos, bintextsize, figsize, show, colorbar,
      alphatextpos, ydecades, ylabel = as in
      plot_gammat_cluster_tomo.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 2D array of panels, [source bin, cluster z bin].
    """
    return _lensing_panels(
        theta_sigma, sigma_ref, data, richness, richnesslabel, thetashow,
        ylabel = r"$|\Sigma(\theta)|$", ylabelglued = ylabel,
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = datalabel, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)


def _richness_pairs(nrichness, pairs):
    """The richness pairs a cluster-clustering figure draws.

    Arguments:
      nrichness = number of richness bins.
      pairs     = None (the auto pairs (n, n) of every bin), or a
                  sequence of (bin 1, bin 2) index pairs.

    Returns:
      list of (int, int), or None when a pair falls outside the
      array.
    """
    if pairs is None:
        return [(nl, nl) for nl in range(nrichness)]
    sel = [(int(p[0]), int(p[1])) for p in pairs]
    if len(sel) == 0 or min(min(p) for p in sel) < 0 or max(max(p) for p in sel) >= nrichness:
        return None
    return sel


def _cc_panels(X, y, X_ref, y_ref, data, pairs, richnesslabel, **options):
    """Slices cluster-clustering arrays into one row of panels.

    Shared by plot_wcc_tomo and plot_C_cc_tomo_limber.

    Arguments:
      X     = list of 1D arrays, the x axis (theta or ell) of each
              list entry.
      y     = list of 4D arrays (n_x, n_richness, n_richness,
              n_cluster_z), one per list entry.
      X_ref, y_ref = None, or the x axis and one such array used as
              the ratio reference.
      data  = None, or (x, values, errors) in the same layout.
      pairs, richnesslabel = as in the public functions.
      options = the remaining options of _plot_cluster_panels.

    Returns:
      what _plot_cluster_panels returns.
    """
    nx, nrichness, nrichness2, ncluster = np.shape(y[0])
    if nrichness != nrichness2:
        print("Bad Input (number of richness bins 1/2)")
        return 0
    sel = _richness_pairs(nrichness, pairs)
    if sel is None:
        print("Bad Input (richness pair outside the array)")
        return 0
    if not (richnesslabel is None) and len(richnesslabel) != len(sel):
        print("Bad Input (number of richness labels)")
        return 0

    def panels(a):
        # (n_x, n_richness, n_richness, n_cluster_z) to the panel
        # layout (n_x, n_curve, n_col, n_row): np.stack lines the
        # selected richness pairs up on a new axis 1 (the curves),
        # the cluster z bins stay as the columns, and the trailing
        # None adds the single row
        a = np.asarray(a)
        return np.stack([a[:,p,q,:] for (p, q) in sel], axis = 1)[:,:,:,None]

    if not (data is None):
        data = (data[0], panels(data[1]), None if data[2] is None else panels(data[2]))
    if richnesslabel is None:
        richnesslabel = [r"$\lambda$ bins $(%d,%d)$" % (p+1, q+1) for (p, q) in sel]

    return _plot_cluster_panels(
        X = X, Y = [panels(a) for a in y],
        X_ref = X_ref, Y_ref = None if y_ref is None else panels(y_ref), data = data,
        bintext = [["$(" + str(i+1) + ")$"] for i in range(ncluster)],
        curvelabel = richnesslabel, **options)


def plot_wcc_tomo(theta_wcc, theta_wcc_ref = None, param = None, colorbarlabel = None,
                  pairs = None, data = None, marker = None,
                  linestyle = None, linewidth = None, ylim = [0.75,1.25],
                  cmap = 'gist_rainbow', legend = None, legendloc = None,
                  richnesslabel = None, richnesslegend = 1, datalabel = None,
                  yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                  bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 5),
                  show = 1, colorbar = 1, colorbarshrink = 1.0, markersize = 3,
                  thetashow = None, rescale = None, alphatextpos = [0.05, 0.12],
                  ydecades = 4, ylabel = r"$\alpha\,|w_{cc}(\theta)|$",
                  legendfontsize = None, xaxislabelsize = 16):
    """One panel per cluster z bin of the cluster clustering w_cc.

    The panels show the auto-correlation of each cluster redshift
    bin; the richness pairs are curves inside each panel, told
    apart by their line style. Without theta_wcc_ref each curve is
    |w_cc| on a log scale; with it each curve is the fractional
    difference w_cc / ref - 1.

    Arguments:
      theta_wcc = list of (theta, wcc) pairs, one per list entry,
                 as the notebook w_cc wrapper returns them: theta
                 in arcmin, wcc a 4D array (n_theta, n_richness,
                 n_richness, n_cluster_z).
      theta_wcc_ref = None, or one (theta, wcc) pair used as the
                 ratio reference.
      pairs    = which richness pairs are drawn, as (bin 1, bin 2)
                 index pairs counted from 0: None (default) draws
                 the auto pair (n, n) of every richness bin.
      data     = None, or (theta, values, errors): the cluster
                 clustering block of the data vector in the layout
                 of one wcc array, drawn as points with error bars
                 (errors may be None); NaN entries (the scale cuts)
                 are skipped. datalabel = its legend label, or None
                 for no legend entry.
      linestyle, linewidth, marker = lists cycled across the
                 richness pairs of a panel, or None; with a single
                 pair drawn they are cycled across the list entries
                 instead.
      richnesslabel = one label per drawn pair for the legend of
                 the line styles; None (default) writes
                 "lambda bins (n, m)". richnesslegend = None
                 suppresses that legend.
      thetashow = x-axis range in arcmin; None (default) spans the
                 theta array itself.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 the row then shares one y-range and is glued, with
                 the factor alpha annotated inside each panel and
                 one global y label (the ylabel argument). Ignored
                 with theta_wcc_ref. None (default) keeps per-panel
                 y-ranges.
      param, colorbarlabel, ylim, legend, legendloc,
      legendfontsize, the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize), cmap,
      colorbarshrink, markersize, bintextpos, bintextsize, figsize,
      show, colorbar, alphatextpos, ydecades, ylabel = as in
      plot_gammat_cluster_tomo.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 1D array of panels, one per cluster z bin.
    """
    (theta, wcc) = theta_wcc[0]
    if np.shape(wcc)[0] != len(theta):
        print("Bad Input (theta)")
        print(theta)
        print(np.shape(wcc)[0], len(theta))
        return 0
    if thetashow is None:
        thetashow = [np.min(theta), np.max(theta)]

    return _cc_panels(
        [np.asarray(t) for (t, w) in theta_wcc], [w for (t, w) in theta_wcc],
        None if theta_wcc_ref is None else np.asarray(theta_wcc_ref[0]),
        None if theta_wcc_ref is None else theta_wcc_ref[1],
        data, pairs, richnesslabel,
        xlabel = r"$\theta$ [arcmin]", ylabel = r"$|w_{cc}(\theta)|$",
        ylabelglued = ylabel, xlim = thetashow,
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = datalabel, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)


def _cg_pairs(y, pairs):
    """The (cluster z bin, lens bin) pairs a cluster x galaxy figure draws.

    Arguments:
      y     = list of 4D arrays (n_x, n_richness, n_cluster_z,
              n_lens), one per list entry.
      pairs = None (every pair that is not identically zero in
              every entry), or a sequence of (cluster z bin, lens
              bin) index pairs.

    Returns:
      list of (int, int), or None when no pair is left or a pair
      falls outside the array.
    """
    nx, nrichness, ncluster, nlens = np.shape(y[0])
    if pairs is None:
        # the wrappers fill only the pairs of the dataset (its
        # cg_lens_bins) and leave the others identically zero
        sel = [(i, g) for i in range(ncluster) for g in range(nlens)
               if any(np.any(np.asarray(a)[:,:,i,g]) for a in y)]
    else:
        sel = [(int(p[0]), int(p[1])) for p in pairs]
        if any(i < 0 or i >= ncluster or g < 0 or g >= nlens for (i, g) in sel):
            return None
    return sel if len(sel) > 0 else None


def _cg_panels(X, y, X_ref, y_ref, data, richness, pairs, richnesslabel, **options):
    """Slices cluster x galaxy arrays into one row of panels.

    Shared by plot_wcg_tomo and plot_C_cg_tomo_limber.

    Arguments:
      X     = list of 1D arrays, the x axis (theta or ell) of each
              list entry.
      y     = list of 4D arrays (n_x, n_richness, n_cluster_z,
              n_lens), one per list entry.
      X_ref, y_ref = None, or the x axis and one such array used as
              the ratio reference.
      data  = None, or (x, values, errors) in the same layout.
      richness, pairs, richnesslabel = as in the public functions.
      options = the remaining options of _plot_cluster_panels.

    Returns:
      what _plot_cluster_panels returns.
    """
    nx, nrichness, ncluster, nlens = np.shape(y[0])
    sel = _select_bins(nrichness, richness)
    if sel is None:
        print("Bad Input (richness bin outside the array)")
        return 0
    if not (richnesslabel is None) and len(richnesslabel) != nrichness:
        print("Bad Input (number of richness labels)")
        return 0
    cg = _cg_pairs(y, pairs)
    if cg is None:
        print("Bad Input (no (cluster z bin, lens bin) pair to draw)")
        return 0

    def panels(a):
        # (n_x, n_richness, n_cluster_z, n_lens) to the panel layout
        # (n_x, n_curve, n_col, n_row): np.stack lines the selected
        # (cluster z, lens) pairs up on a new last axis (the
        # columns), and the trailing None adds the single row
        a = np.asarray(a)[:,sel,:,:]
        return np.stack([a[:,:,i,g] for (i, g) in cg], axis = 2)[:,:,:,None]

    if not (data is None):
        data = (data[0], panels(data[1]), None if data[2] is None else panels(data[2]))
    if richnesslabel is None:
        richnesslabel = [r"$\lambda$ bin %d" % (nl+1) for nl in range(nrichness)]

    return _plot_cluster_panels(
        X = X, Y = [panels(a) for a in y],
        X_ref = X_ref, Y_ref = None if y_ref is None else panels(y_ref), data = data,
        bintext = [["$(" + str(i+1) + "," + str(g+1) + ")$"] for (i, g) in cg],
        curvelabel = [richnesslabel[nl] for nl in sel], **options)


def plot_wcg_tomo(theta_wcg, theta_wcg_ref = None, param = None, colorbarlabel = None,
                  richness = None, pairs = None, data = None, marker = None,
                  linestyle = None, linewidth = None, ylim = [0.75,1.25],
                  cmap = 'gist_rainbow', legend = None, legendloc = None,
                  richnesslabel = None, richnesslegend = 1, datalabel = None,
                  yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                  bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 5),
                  show = 1, colorbar = 1, colorbarshrink = 1.0, markersize = 3,
                  thetashow = None, rescale = None, alphatextpos = [0.05, 0.12],
                  ydecades = 4, ylabel = r"$\alpha\,|w_{cg}(\theta)|$",
                  legendfontsize = None, xaxislabelsize = 16):
    """One panel per (cluster z, lens) pair of cluster x galaxy w_cg.

    Only the pairs the dataset holds get a panel (one row), and the
    label inside each panel reads (cluster z bin, lens bin). The
    richness bins are curves inside each panel, told apart by their
    line style. Without theta_wcg_ref each curve is |w_cg| on a log
    scale; with it each curve is the fractional difference
    w_cg / ref - 1.

    Arguments:
      theta_wcg = list of (theta, wcg) pairs, one per list entry,
                 as the notebook w_cg wrapper returns them: theta
                 in arcmin, wcg a 4D array (n_theta, n_richness,
                 n_cluster_z, n_lens).
      theta_wcg_ref = None, or one (theta, wcg) pair used as the
                 ratio reference.
      richness = which richness bins are drawn, counted from 0:
                 None (default) draws every bin, an index draws one
                 bin, a list draws those bins.
      pairs    = which (cluster z bin, lens bin) pairs get a panel,
                 as index pairs counted from 0 (the rows of the
                 interface's get_cg_redshift_bins): None (default)
                 takes every pair that is not identically zero.
      data     = None, or (theta, values, errors): the cluster x
                 galaxy block of the data vector in the layout of
                 one wcg array, drawn as points with error bars
                 (errors may be None); NaN entries (the scale cuts)
                 are skipped. datalabel = its legend label, or None
                 for no legend entry.
      thetashow = x-axis range in arcmin; None (default) spans the
                 theta array itself.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 the row then shares one y-range and is glued, with
                 the factor alpha annotated inside each panel and
                 one global y label (the ylabel argument). Ignored
                 with theta_wcg_ref. None (default) keeps per-panel
                 y-ranges.
      param, colorbarlabel, marker, linestyle, linewidth, ylim,
      legend, richnesslabel, richnesslegend, legendloc,
      legendfontsize, the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize), cmap,
      colorbarshrink, markersize, bintextpos, bintextsize, figsize,
      show, colorbar, alphatextpos, ydecades, ylabel = as in
      plot_gammat_cluster_tomo.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 1D array of panels, one per (cluster z, lens) pair.
    """
    (theta, wcg) = theta_wcg[0]
    if np.shape(wcg)[0] != len(theta):
        print("Bad Input (theta)")
        print(theta)
        print(np.shape(wcg)[0], len(theta))
        return 0
    if thetashow is None:
        thetashow = [np.min(theta), np.max(theta)]

    return _cg_panels(
        [np.asarray(t) for (t, w) in theta_wcg], [w for (t, w) in theta_wcg],
        None if theta_wcg_ref is None else np.asarray(theta_wcg_ref[0]),
        None if theta_wcg_ref is None else theta_wcg_ref[1],
        data, richness, pairs, richnesslabel,
        xlabel = r"$\theta$ [arcmin]", ylabel = r"$|w_{cg}(\theta)|$",
        ylabelglued = ylabel, xlim = thetashow,
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = datalabel, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)


def plot_C_cs_tomo_limber(ell, C_cs, C_cs_ref = None, param = None, colorbarlabel = None,
                          lmin = 30, lmax = 1500, richness = None, marker = None,
                          linestyle = None, linewidth = None, ylim = [0.75,1.25],
                          cmap = 'gist_rainbow', legend = None, legendloc = None,
                          richnesslabel = None, richnesslegend = 1,
                          yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                          bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 13),
                          show = 1, colorbar = 1, colorbarshrink = 0.5, markersize = 3,
                          rescale = None, alphatextpos = [0.05, 0.12], ydecades = 4,
                          ylabel = r"$\alpha\,|C_{\ell}^{cs}|$", legendfontsize = None,
                          xaxislabelsize = 16):
    """Panel grid of the cluster-lensing angular power spectra C_cs.

    The harmonic-space counterpart of plot_gammat_cluster_tomo, on
    the same grid: one panel per (cluster z bin, source bin) pair
    (columns are cluster redshift bins, rows are source bins), the
    richness bins as curves told apart by their line style. Without
    C_cs_ref each curve is |C_ell| on a log scale; with C_cs_ref
    each curve is the fractional difference C / C_ref - 1.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_cs entry.
      C_cs     = list of 4D arrays (n_ell, n_richness, n_cluster_z,
                 n_source), one per list entry, as the notebook
                 C_cs_tomo_limber wrapper returns them.
      C_cs_ref = None, or one such array used as the ratio
                 reference.
      lmin, lmax = x-axis range in ell.
      rescale  = 1 glues the absolute panels with a per-panel power
                 of ten alpha, the y-axis label reading
                 alpha |C^cs| (the ylabel argument).
      param, colorbarlabel, richness, marker, linestyle, linewidth,
      ylim, legend, richnesslabel, richnesslegend, legendloc,
      legendfontsize, the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize), cmap,
      colorbarshrink, markersize, bintextpos, bintextsize, figsize,
      show, colorbar, alphatextpos, ydecades, ylabel = as in
      plot_gammat_cluster_tomo.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 2D array of panels, [source bin, cluster z bin].
    """
    nell, nrichness, ncluster, nsource = np.shape(C_cs[0])
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    sel = _select_bins(nrichness, richness)
    if sel is None:
        print("Bad Input (richness bin outside the array)")
        return 0
    if not (richnesslabel is None) and len(richnesslabel) != nrichness:
        print("Bad Input (number of richness labels)")
        return 0
    if richnesslabel is None:
        richnesslabel = [r"$\lambda$ bin %d" % (nl+1) for nl in range(nrichness)]

    # the panel layout (n_x, n_curve, n_col, n_row) is the array's
    # own axis order: richness curves, cluster z columns, source
    # rows; only the richness selection is applied
    ell = np.asarray(ell)
    return _plot_cluster_panels(
        X = [ell for Cl in C_cs], Y = [np.asarray(Cl)[:,sel,:,:] for Cl in C_cs],
        X_ref = None if C_cs_ref is None else ell,
        Y_ref = None if C_cs_ref is None else np.asarray(C_cs_ref)[:,sel,:,:],
        data = None,
        bintext = [["$(" + str(i+1) + "," + str(j+1) + ")$" for j in range(nsource)]
                   for i in range(ncluster)],
        curvelabel = [richnesslabel[nl] for nl in sel],
        xlabel = r"$\ell$", ylabel = r"$|C_{\ell}^{cs}|$", ylabelglued = ylabel,
        xlim = [lmin, lmax],
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = None, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)


def plot_C_cc_tomo_limber(ell, C_cc, C_cc_ref = None, param = None, colorbarlabel = None,
                          lmin = 30, lmax = 1500, pairs = None, marker = None,
                          linestyle = None, linewidth = None, ylim = [0.75,1.25],
                          cmap = 'gist_rainbow', legend = None, legendloc = None,
                          richnesslabel = None, richnesslegend = 1,
                          yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                          bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 5),
                          show = 1, colorbar = 1, colorbarshrink = 1.0, markersize = 3,
                          rescale = None, alphatextpos = [0.05, 0.12], ydecades = 4,
                          ylabel = r"$\alpha\,|C_{\ell}^{cc}|$", legendfontsize = None,
                          xaxislabelsize = 16):
    """One panel per cluster z bin of the cluster-clustering spectra.

    The harmonic-space counterpart of plot_wcc_tomo: the panels
    show the auto-correlation C_cc of each cluster redshift bin,
    the richness pairs as curves told apart by their line style.
    Without C_cc_ref each curve is |C_ell| on a log scale; with
    C_cc_ref each curve is the fractional difference C / C_ref - 1.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_cc entry.
      C_cc     = list of 4D arrays (n_ell, n_richness, n_richness,
                 n_cluster_z), one per list entry, as the notebook
                 C_cc_tomo_limber wrapper returns them.
      C_cc_ref = None, or one such array used as the ratio
                 reference.
      lmin, lmax = x-axis range in ell.
      pairs    = which richness pairs are drawn, as (bin 1, bin 2)
                 index pairs counted from 0: None (default) draws
                 the auto pair (n, n) of every richness bin.
      richnesslabel = one label per drawn pair for the legend of
                 the line styles; None (default) writes
                 "lambda bins (n, m)".
      rescale  = 1 glues the absolute panels with a per-panel power
                 of ten alpha and one global y label (the ylabel
                 argument).
      param, colorbarlabel, marker, linestyle, linewidth, ylim,
      legend, richnesslegend, legendloc, legendfontsize, the *size
      arguments (yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize), cmap, colorbarshrink, markersize,
      bintextpos, bintextsize, figsize, show, colorbar,
      alphatextpos, ydecades, ylabel = as in plot_wcc_tomo.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 1D array of panels, one per cluster z bin.
    """
    if np.shape(C_cc[0])[0] != len(ell):
        print("Bad Input (number of ell)")
        return 0

    ell = np.asarray(ell)
    return _cc_panels(
        [ell for Cl in C_cc], C_cc, None if C_cc_ref is None else ell, C_cc_ref,
        None, pairs, richnesslabel,
        xlabel = r"$\ell$", ylabel = r"$|C_{\ell}^{cc}|$",
        ylabelglued = ylabel, xlim = [lmin, lmax],
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = None, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)


def plot_C_cg_tomo_limber(ell, C_cg, C_cg_ref = None, param = None, colorbarlabel = None,
                          lmin = 30, lmax = 1500, richness = None, pairs = None, marker = None,
                          linestyle = None, linewidth = None, ylim = [0.75,1.25],
                          cmap = 'gist_rainbow', legend = None, legendloc = None,
                          richnesslabel = None, richnesslegend = 1,
                          yaxislabelsize = 16, yaxisticklabelsize = 10, xaxisticklabelsize = 20,
                          bintextpos = [0.85, 0.85], bintextsize = 15, figsize = (16, 5),
                          show = 1, colorbar = 1, colorbarshrink = 1.0, markersize = 3,
                          rescale = None, alphatextpos = [0.05, 0.12], ydecades = 4,
                          ylabel = r"$\alpha\,|C_{\ell}^{cg}|$", legendfontsize = None,
                          xaxislabelsize = 16):
    """One panel per (cluster z, lens) pair of the cluster x galaxy spectra.

    The harmonic-space counterpart of plot_wcg_tomo: only the pairs
    the dataset holds get a panel (one row), the label inside each
    panel reads (cluster z bin, lens bin), and the richness bins
    are curves told apart by their line style. Without C_cg_ref
    each curve is |C_ell| on a log scale; with C_cg_ref each curve
    is the fractional difference C / C_ref - 1.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_cg entry.
      C_cg     = list of 4D arrays (n_ell, n_richness, n_cluster_z,
                 n_lens), one per list entry, as the notebook
                 C_cg_tomo_limber wrapper returns them.
      C_cg_ref = None, or one such array used as the ratio
                 reference.
      lmin, lmax = x-axis range in ell.
      richness = which richness bins are drawn, counted from 0:
                 None (default) draws every bin, an index draws one
                 bin, a list draws those bins.
      pairs    = which (cluster z bin, lens bin) pairs get a panel,
                 as index pairs counted from 0: None (default)
                 takes every pair that is not identically zero.
      rescale  = 1 glues the absolute panels with a per-panel power
                 of ten alpha and one global y label (the ylabel
                 argument).
      param, colorbarlabel, marker, linestyle, linewidth, ylim,
      legend, richnesslabel, richnesslegend, legendloc,
      legendfontsize, the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize), cmap,
      colorbarshrink, markersize, bintextpos, bintextsize, figsize,
      show, colorbar, alphatextpos, ydecades, ylabel = as in
      plot_wcg_tomo.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None; axes is
      the 1D array of panels, one per (cluster z, lens) pair.
    """
    if np.shape(C_cg[0])[0] != len(ell):
        print("Bad Input (number of ell)")
        return 0

    ell = np.asarray(ell)
    return _cg_panels(
        [ell for Cl in C_cg], C_cg, None if C_cg_ref is None else ell, C_cg_ref,
        None, richness, pairs, richnesslabel,
        xlabel = r"$\ell$", ylabel = r"$|C_{\ell}^{cg}|$",
        ylabelglued = ylabel, xlim = [lmin, lmax],
        param = param, colorbarlabel = colorbarlabel, marker = marker,
        linestyle = linestyle, linewidth = linewidth, ylim = ylim, cmap = cmap,
        legend = legend, legendloc = legendloc, richnesslegend = richnesslegend,
        datalabel = None, yaxislabelsize = yaxislabelsize,
        yaxisticklabelsize = yaxisticklabelsize,
        xaxisticklabelsize = xaxisticklabelsize, xaxislabelsize = xaxislabelsize,
        bintextpos = bintextpos, bintextsize = bintextsize, figsize = figsize,
        show = show, colorbar = colorbar, colorbarshrink = colorbarshrink,
        markersize = markersize, rescale = rescale, alphatextpos = alphatextpos,
        ydecades = ydecades, legendfontsize = legendfontsize)
