"""Data-vector plots shared by the EXAMPLE_EVALUATE notebooks.

One function per probe, each drawing one panel per tomographic bin
(or bin pair): cosmic shear as a lower triangle
(plot_C_ss_tomo_limber in harmonic space, plot_xi for xi_plus/minus
in real space), galaxy-galaxy lensing as a full lens x source grid
(plot_C_gs_tomo_limber, plot_gammat_tomo_limber), galaxy clustering
as one row of lens-bin auto-correlations (plot_C_gg_tomo,
plot_wtheta_tomo), and one flat panel of baryonic feedback
suppression curves for bfmt parameter sweeps
(plot_baryon_suppression).

Conventions every panel plotter shares:

- Without a reference, each panel shows the spectrum itself with its
  own y-range; with a *_ref argument it shows the fractional
  difference (value/reference - 1) on one shared linear band, and
  the panels are glued edge to edge (zero subplot spacing).
- rescale = 1 glues the absolute panels too: each panel is
  multiplied by its own power of ten, chosen so its maximum lands in
  [1, 10), one y-range serves the whole grid, the factor is
  annotated as alpha inside the panel, and one global y label
  replaces the per-row labels. ydecades caps how far the shared
  range extends below its ceiling.
- Glued panels put a neighbor's edge tick number on the same spot,
  so tick labels within 10% of an interior panel boundary are
  hidden (y boundaries on grids and triangles, x boundaries on the
  one-row plots).
- A (lens, source) pair dropped via cosmolike's init_ggl_exclude
  arrives as an identically zero spectrum; its panel is drawn empty
  with an "excluded" placeholder instead of log scaling zeros.

Pure matplotlib and numpy: nothing here touches CAMB or the compiled
cosmolike interface, so every project shares these functions
unchanged. Figure styling (fonts, usetex, rcParams) stays in the
notebooks; these functions only build the figures.
"""

import math
import itertools

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def _hide_glued_edge_ticklabels(panels, lo, hi, axis = "y", log = True):
    """Hides tick labels near interior boundaries of a glued grid.

    Glued panels put a neighbor's edge tick label on the same spot,
    so labels within 10% of an interior boundary disappear.

    Arguments:
      panels = sequence of (axes, free_lo, free_hi): free_lo and
               free_hi say whether that panel's low / high edge sits
               on the outer figure boundary (labels there stay)
               instead of on a neighbor.
      lo, hi = the shared axis limits the fractions are measured in.
      axis   = "y" (default) or "x": which axis the glue direction
               runs along.
      log    = True measures the 10% fraction in log10 of the tick
               value (log-scaled axes); False measures it linearly.
    """
    span = (np.log10(hi) - np.log10(lo)) if log else (hi - lo)
    for ax, free_lo, free_hi in panels:
        axis_obj = ax.yaxis if axis == "y" else ax.xaxis
        for t in axis_obj.get_major_ticks():
            v = t.get_loc()
            if log:
                if v <= 0:
                    continue
                frac = (np.log10(v) - np.log10(lo))/span
            else:
                frac = (v - lo)/span
            # the tolerance keeps ticks sitting exactly on a limit,
            # where float dust puts them a hair outside [lo, hi]
            if frac < -1e-6 or frac > 1 + 1e-6:
                continue
            if (frac < 0.1 and not free_lo) or (frac > 0.9 and not free_hi):
                t.label1.set_visible(False)


def _glued_supylabel(fig, leftcol, ylabel, yaxislabelsize):
    """One global y label placed against the measured tick-label edge.

    A draw realizes the tick labels of the leftmost panels so their
    extent can be measured; supylabel anchors the rotated text's
    LEFT edge (ha='left'), so the label is measured too and shifted
    until its right edge clears the ticks by a small pad.

    Arguments:
      fig     = the figure the label is drawn on.
      leftcol = the panels whose tick labels set the clearance (the
                leftmost column, the only one that shows y numbers
                on a shared axis).
      ylabel  = the label text.
      yaxislabelsize = its font size in points.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xmin = min(a.get_tightbbox(renderer).x0 for a in leftcol)
    xmin = fig.transFigure.inverted().transform((xmin, 0))[0]
    lab = fig.supylabel(ylabel, fontsize=yaxislabelsize)
    labw = lab.get_window_extent(renderer).width/fig.bbox.width
    lab.set_x(max(xmin - 0.004 - labw, 0.001))


def _align_log_ticklabels(axs):
    """Left-aligns y tick labels so the "10" bases stack vertically.

    matplotlib right-aligns y tick labels, so "10^-1" (whose box is
    wider by the minus sign) pushes its "10" base left of its
    neighbors' and stacked labels zig-zag. Every label is anchored
    instead by its LEFT edge on one common column: the tick pad
    grows to the widest label's width (plus a small gap), and the
    labels' text then runs from that column toward the axis. The
    bases align exactly, and the leftover width shows up as ragged
    space next to the axis, where it reads naturally.
    """
    for ax in axs:
        fig = ax.figure
        # a draw realizes the labels so their widths can be measured
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [t for t in ax.get_yticklabels()
                  if t.get_text() and t.get_visible()]
        if not labels:
            continue
        widest = max(t.get_window_extent(renderer).width for t in labels)
        # the tick pad is in points; window extents are in pixels,
        # and 72 points make an inch, so pixels * 72 / dpi converts
        pad_points = widest*72.0/fig.dpi + 3.5
        ax.yaxis.set_tick_params(pad=pad_points)
        for t in ax.get_yticklabels():
            t.set_horizontalalignment('left')


def plot_C_ss_tomo_limber(ell, C_ss, C_ss_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, colorbarshrink=0.3,
                          cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                          legend = None, legendloc = None, yaxislabelsize = 16, yaxisticklabelsize = 10, 
                          xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], bintextsize = 13, figsize = (18, 18),
                          show = 1, colorbar=1, wspace=0.25, hspace=0.05, 
                          marker = None, markersize = 3, rescale = None,
                          alphatextpos = [0.05, 0.12], ydecades = 4,
                          ylabel = r"$\alpha\,\ell (\ell+1) C_{\ell}^{EE}/(2 \pi)$",
                          legendfontsize = None, xaxislabelsize = 16):
    """Triangle plot of tomographic angular power spectra.

    One panel per tomographic bin pair (i, j), lower triangle only.
    Without C_ss_ref each curve is ell(ell+1) C_ell / 2 pi on a log
    scale; with C_ss_ref each curve is the fractional difference
    C / C_ref - 1 on a linear scale.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_ss entry.
      C_ss     = list of 3D arrays (n_ell, n_bins, n_bins), one per
                 curve.
      C_ss_ref = None, or one 3D array used as the ratio reference.
      param    = list of parameter values (one per curve) coloring
                 the curves and the colorbar, or None.
      colorbarlabel = colorbar label (LaTeX string), or None.
      lmin, lmax    = x-axis range in ell.
      ylim     = without C_ss_ref, multipliers on each panel's
                 min/max; with it, the band around 1 (drawn as
                 ylim - 1).
      linestyle, linewidth = lists cycled across curves, or None.
      marker, markersize = point markers instead of lines (a list
                 cycled across curves, drawn at markersize points).
      legend   = one label per curve, or None. legendloc = None (the
                 default) puts the legend inside the empty upper
                 triangle; an (x, y) pair in figure fractions places
                 its lower-left corner anywhere. legendfontsize =
                 entry font size in points; None follows
                 matplotlib's legend.fontsize rcParam.
      yaxislabelsize, xaxislabelsize = font sizes in points of the
                 y and x axis labels; yaxisticklabelsize and
                 xaxisticklabelsize size the tick numbers the same
                 way.
      cmap, colorbarshrink, bintextpos, bintextsize, figsize,
      wspace, hspace = matplotlib layout knobs.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 every panel then shares one y-range, interior y
                 axes disappear and the panels are glued together,
                 with the factor alpha annotated inside each panel
                 and one global y label (the ylabel argument).
                 Ignored with C_ss_ref (already dimensionless and
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
      None after drawing, or (fig, axes) when show is None.
    """

    # each C_ss entry is a 3D array; unpacking .shape reads its three
    # sizes (number of ell values, bins, bins) in one line
    nell, ntomo, ntomo2 = C_ss[0].shape
    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
      
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    if not (C_ss_ref is None):
        nell2, ntomo3, ntomo4 = C_ss_ref.shape
        if (ntomo3 != ntomo4) or (nell != nell2):
            print(f"notomo = {ntomo}, ntomo_REF = {ntomo3}")
            print(f"Nell = {nell}, Nell_REF = {nell2}")
            return 0   
        
    # rescale=1: alpha[i,j] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued grid. The
    # plotted quantity is ell(ell+1) C / 2 pi, as in the panels.
    rescale = None if not (C_ss_ref is None) else rescale
    alpha = np.zeros((ntomo, ntomo))
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(ntomo):
            for j in range(i, ntomo):
                D = [ell*(ell + 1)*Cl[:,i,j]/(2*math.pi) for Cl in C_ss]
                pmax = max(np.max(np.abs(d)) for d in D)
                if pmax == 0:
                    continue
                alpha[i,j] = -np.floor(np.log10(pmax))
                # the glued floor only counts positive minima: a curve
                # touching zero cannot set a log-axis lower limit
                lo = min(np.min(np.abs(d)) for d in D)*10.0**alpha[i,j]
                if lo > 0:
                    panlo.append(lo)
                panhi.append(pmax*10.0**alpha[i,j])
        yglued = [ylim[0]*np.min(panlo if panlo else panhi), ylim[1]*np.max(panhi)]
        if not (ydecades is None):
            yglued[0] = max(yglued[0], yglued[1]/10.0**ydecades)

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge (see the module docstring)
    if C_ss_ref is None and rescale is None:
        fig, axes = plt.subplots(
            nrows = ntomo,
            ncols = ntomo,
            figsize = figsize,
            sharex = True,
            sharey = False,
            gridspec_kw = {'wspace': wspace, 'hspace': hspace})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 18, weight = 'bold', labelpad = 2)
        if len(param) != len(C_ss):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)

    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following entry, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])

    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
    
    # axes is the 2D array of panels plt.subplots returned; [j, i]
    # is row j, column i, and the upper triangle is switched off
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                # collect each curve's extremes in this panel: without rescale
                # the per-panel y-range is [ylim[0]*min, ylim[1]*max]
                clmin = []
                clmax = []
                for Cl in C_ss:  
                    tmp = ell * (ell + 1) * Cl[:,i,j] / (2 * math.pi)
                    clmin.append(np.min(tmp))
                    clmax.append(np.max(tmp))
     
                axes[j,i].set_xlim([lmin, lmax])
                
                if C_ss_ref is None:
                    if not (rescale is None):
                        axes[j,i].set_ylim(yglued)
                        axes[j,i].set_yscale('log')
                    else:
                        axes[j,i].set_ylim([np.min(ylim[0]*np.array(clmin)), np.max(ylim[1]*np.array(clmax))])
                        axes[j,i].set_yscale('log')
                else:
                    # with a reference every curve is value/reference - 1, so ylim
                    # (multipliers around 1) is drawn as the band ylim - 1 around 0
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                    axes[j,i].set_yscale('linear')

                axes[j,i].set_xscale('log')

                if i == 0:
                    if C_ss_ref is None:
                        # with rescale the y label is global: one fig.supylabel
                        if rescale is None:
                            axes[j,i].set_ylabel(r"$\ell (\ell+1) C_{\ell}^{EE}/(2 \pi)$", fontsize=yaxislabelsize)
                    else:
                        axes[j,i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)
                
                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\ell$", fontsize=xaxislabelsize)
                
                # transform=transAxes puts the text in panel fractions: (0, 0)
                # is the panel's lower-left corner, (1, 1) its upper-right
                axes[j,i].text(bintextpos[0], bintextpos[1], 
                    "$(" +  str(i) + "," +  str(j) + ")$", 
                    horizontalalignment = 'center', 
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    usetex = True,
                    transform = axes[j,i].transAxes)
                
                if not (rescale is None):
                    expo = int(alpha[i,j])
                    axes[j,i].text(alphatextpos[0], alphatextpos[1],
                        "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                        horizontalalignment = 'left',
                        verticalalignment = 'center',
                        fontsize = bintextsize,
                        usetex = True,
                        transform = axes[j,i].transAxes)

                for x, Cl in enumerate(C_ss):
                    if C_ss_ref is None:
                        # 10**alpha = 1 unless rescale is on
                        tmp = ell * (ell + 1) * Cl[:,i,j] / (2 * math.pi) * 10.0**alpha[i,j]
                    else:
                        tmp = Cl[:,i,j] / C_ss_ref[:,i,j] - 1
                    if marker is None:
                        axes[j,i].plot(ell, tmp, 
                                       color=cm(x/len(C_ss)), 
                                       linewidth=next(linewidthcycler), 
                                       linestyle=next(linestylecycler))
                    else:
                        axes[j,i].plot(ell, tmp, 
                                       color=cm(x/len(C_ss)), 
                                       markerfacecolor='None', 
                                       marker=next(markercycler), 
                                       markeredgecolor=cm(x/len(C_ss)), 
                                       linestyle='None', 
                                       markersize=markersize)
    
    if not (rescale is None):
        # the minus sign otherwise staggers the stacked y numbers
        _align_log_ticklabels(axes[:,0])
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == ntomo-1, j == 0) for j in range(ntomo)],
            yglued[0], yglued[1])
        _glued_supylabel(fig, axes[:,0], ylabel, yaxislabelsize)

    if not (C_ss_ref is None):
        # the ratio triangle is glued too: same boundary-label
        # pruning, on the shared linear range ylim - 1
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == ntomo-1, j == 0) for j in range(ntomo)],
            ylim[0]-1.0, ylim[1]-1.0, log = False)

    if not (legend is None):
        if len(legend) != len(C_ss):
            print("Bad Input")
            return 0
        # legendloc None (the default) puts the legend inside the
        # empty upper triangle; an (x, y) pair places it anywhere
        fig.legend(
            legend,
            loc=(0.6, 0.78) if legendloc is None else legendloc,
            ncols=(2 if len(legend) > 4 else 1) if legendloc is None else 1,
            fontsize=legendfontsize,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)

    # warn=False: outside a notebook, showing a figure on a
    # non-interactive backend would otherwise warn
    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)

def plot_xi(pm, xi, xi_ref = None, param = None, colorbarlabel = None, marker = None, colorbarshrink=0.3,
                linestyle = None, linewidth = None, ylim = [0.88,1.12], 
                cmap = 'gist_rainbow', legend = None, legendloc = None, yaxislabelsize = 16, 
                yaxisticklabelsize = 10, xaxisticklabelsize = 20, bintextpos = [[0.8, 0.875],[0.2,0.875]],
                bintextsize = 15, figsize = (18, 18), show = 1, thetashow=[3,250], colorbar=1, wspace=0.25,hspace=0.05,
                markersize = 3, rescale = None, alphatextpos = [0.05, 0.12], ydecades = None, ylabel = None, legendfontsize = None, xaxislabelsize = 16):
    """Triangle plot of the real-space shear correlation functions.

    One panel per tomographic bin pair (i, j), lower triangle only.
    pm > 0 plots xi_plus, pm <= 0 plots xi_minus. Without xi_ref each
    curve is theta * xi * 10^4; with xi_ref each curve is the
    fractional difference xi / xi_ref - 1.

    Arguments:
      pm     = +1 for xi_plus, -1 for xi_minus.
      xi     = list of (theta, xi_plus, xi_minus) tuples, one per
               curve, as the notebook xi wrapper returns them; the
               xi arrays are 3D (n_theta, n_bins, n_bins).
      xi_ref = None, or one such tuple used as the ratio reference.
      param  = list of parameter values (one per curve) coloring the
               curves and the colorbar, or None.
      colorbarlabel = colorbar label (LaTeX string), or None.
      marker = list of matplotlib markers cycled across curves
               (points instead of lines), or None for lines.
      linestyle, linewidth = lists cycled across curves, or None.
      ylim   = without xi_ref, multipliers on each panel's min/max;
               with it, the band around 1 (drawn as ylim - 1).
      thetashow = x-axis range in arcmin.
      legend = one label per curve, or None. legendloc = None (the
                 default) puts the legend inside the empty upper
                 triangle; an (x, y) pair in figure fractions places
                 its lower-left corner anywhere. legendfontsize =
                 entry font size in points; None follows
                 matplotlib's legend.fontsize rcParam.
      yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize = axis-label and tick-number font sizes in
                 points, as in plot_C_ss_tomo_limber.
      cmap, colorbarshrink, bintextpos, bintextsize, figsize,
      wspace, hspace = matplotlib layout knobs.
      show   = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 every panel then shares one y-range and the panels
                 are glued together, with the factor alpha annotated
                 inside each panel and one global y label. Ignored
                 with xi_ref. None (default) keeps per-panel ranges.
      alphatextpos = axes-fraction (x, y) anchoring the left edge
                 of the alpha annotation.
      ydecades = accepted for a uniform signature; the xi panels are
                 linear, so it has no effect here.
      ylabel   = with rescale, the single global y-axis label; None
                 (default) picks the theta xi_pm 10^4 label with an
                 alpha prefix, following pm.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """
    
    # unpack the first curve's tuple to size the panels; every other
    # entry must share the same theta grid and bin count
    (theta, xip, xim) = xi[0]
    (ntheta, ntomo, ntomo2) = xip.shape    

    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
            
    if ntheta != len(theta):
        print("Bad Input (theta)")
        return 0

    # rescale=1: alpha[i,j] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued grid. The
    # plotted quantity is theta xi 10^4 on a LINEAR scale, so
    # ydecades does not apply here (kept for a uniform signature).
    rescale = None if not (xi_ref is None) else rescale
    alpha = np.zeros((ntomo, ntomo))
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(ntomo):
            for j in range(i, ntomo):
                D = [theta*(xip if pm > 0 else xim)[:,i,j]*10**4
                     for (theta, xip, xim) in xi]
                pmax = max(np.max(np.abs(d)) for d in D)
                if pmax == 0:
                    continue
                alpha[i,j] = -np.floor(np.log10(pmax))
                panlo.append(min(np.min(d) for d in D)*10.0**alpha[i,j])
                panhi.append(pmax*10.0**alpha[i,j])
        yglued = [ylim[0]*np.min(panlo), ylim[1]*np.max(panhi)]

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge (see the module docstring)
    if xi_ref is None and rescale is None:
        fig, axes = plt.subplots(
            nrows = ntomo,
            ncols = ntomo,
            figsize = figsize,
            sharex = True,
            sharey = False,
            gridspec_kw = {'wspace': wspace, 'hspace': hspace}
        )
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0.0, 'hspace': 0.0}
        )    

    cm = plt.get_cmap(cmap)

    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = 'gist_rainbow'), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = colorbarshrink
        )
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(xi):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
    
    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following entry, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
        
    # axes is the 2D array of panels plt.subplots returned; [j, i]
    # is row j, column i, and the upper triangle is switched off
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                # collect each curve's extremes in this panel: without rescale
                # the per-panel y-range is [ylim[0]*min, ylim[1]*max]
                ximin = []
                ximax = []
                for (theta, xip, xim) in xi:
                    if pm > 0:
                        ximin.append(np.min(theta*xip[:,i,j]*10**4))
                        ximax.append(np.max(theta*xip[:,i,j]*10**4))
                    else:
                        ximin.append(np.min(theta*xim[:,i,j]*10**4))
                        ximax.append(np.max(theta*xim[:,i,j]*10**4))
                        
                axes[j,i].set_xlim(thetashow)
                
                if xi_ref is None:
                    if not (rescale is None):
                        axes[j,i].set_ylim(yglued)
                    else:
                        axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)), np.max(ylim[1]*np.array(ximax))])
                else:
                    # with a reference every curve is value/reference - 1, so ylim
                    # (multipliers around 1) is drawn as the band ylim - 1 around 0
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('linear')

                if i == 0:
                    if xi_ref is None:
                        # with rescale the y label is global: one fig.supylabel
                        if not (rescale is None):
                            pass
                        elif pm > 0:
                            axes[j,i].set_ylabel(r"$\theta \xi_{+} \times 10^4$", fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"$\theta \xi_{-} \times 10^4$", fontsize=yaxislabelsize)
                    else:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{+})$", fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{-})$", fontsize=yaxislabelsize)

                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\theta$ [arcmin]", fontsize=xaxislabelsize)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)

                # transform=transAxes puts the text in panel fractions:
                # (0, 0) is the panel's lower-left corner, (1, 1) its
                # upper-right; xi_plus and xi_minus carry their own
                # positions in the nested bintextpos list
                if pm > 0:
                    axes[j,i].text(bintextpos[0][0], 
                                   bintextpos[0][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=bintextsize,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)
                else:
                    axes[j,i].text(bintextpos[1][0], 
                                   bintextpos[1][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=15,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)

                if not (rescale is None):
                    expo = int(alpha[i,j])
                    axes[j,i].text(alphatextpos[0], alphatextpos[1],
                        "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                        horizontalalignment = 'left',
                        verticalalignment = 'center',
                        fontsize = bintextsize,
                        usetex = True,
                        transform = axes[j,i].transAxes)

                # 10**alpha = 1 unless rescale is on for this panel
                fac = 10.0**alpha[i,j]

                if xi_ref is None:
                    for x, (theta, xip, xim) in enumerate(xi):
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, theta*xip[:,i,j]*10**4*fac, color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, theta*xip[:,i,j]*10**4*fac, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), linestyle='None', markersize=markersize)
                        else:
                            if marker is None:   
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4*fac, color=cm(x/len(xi)), 
                                    linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4*fac, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), linestyle='None', markersize=markersize)
                else:
                    (theta_ref, xip_ref, xim_ref) = xi_ref
                    for x, (theta, xip, xim) in enumerate(xi):
                        if not np.array_equal(theta, theta_ref):
                            print("inconsistent theta bins")
                            return 0
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), markerfacecolor='None',
                                               marker=next(markercycler),  markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=markersize)
                        else:
                            if marker is None:   
                                lines = axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                                       linewidth=next(linewidthcycler), linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, color=cm(x/len(xi)), 
                                               markerfacecolor='None', marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=markersize)    
    if not (rescale is None):
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == ntomo-1, j == 0) for j in range(ntomo)],
            yglued[0], yglued[1], log = False)
        if ylabel is None:
            ylabel = (r"$\alpha\,\theta \xi_{+} \times 10^4$" if pm > 0
                      else r"$\alpha\,\theta \xi_{-} \times 10^4$")
        _glued_supylabel(fig, axes[:,0], ylabel, yaxislabelsize)

    if not (xi_ref is None):
        # the ratio triangle is glued too: same boundary-label
        # pruning, on the shared linear range ylim - 1
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == ntomo-1, j == 0) for j in range(ntomo)],
            ylim[0]-1.0, ylim[1]-1.0, log = False)

    if not (legend is None):
        if len(legend) != len(xi):
            print("Bad Input")
            return 0
        # legendloc None (the default) puts the legend inside the
        # empty upper triangle; an (x, y) pair places it anywhere
        fig.legend(legend,
                   loc=(0.6, 0.78) if legendloc is None else legendloc,
                   ncols=(2 if len(legend) > 4 else 1) if legendloc is None else 1,
                   fontsize=legendfontsize,
                   borderpad=0.1,
                   handletextpad=0.4,
                   handlelength=1.5,
                   columnspacing=0.35,
                   scatteryoffsets=[0],
                   frameon=False)  
    # warn=False: outside a notebook, showing a figure on a
    # non-interactive backend would otherwise warn
    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def plot_C_gs_tomo_limber(ell, C_gs, C_gs_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, 
                          cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                          legend = None, legendloc = None, yaxislabelsize = 16, yaxisticklabelsize = 10, 
                          xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], bintextsize = 15, figsize = (20, 12),
                          show = 1, colorbar=1, colorbarshrink = 0.5, marker = None, 
                          markersize = 3, rescale = None, alphatextpos = [0.05, 0.12],
                          ydecades = 4, ylabel = r"$\alpha\,|C_{\ell}^{gs}|$", legendfontsize = None, xaxislabelsize = 16):
    """Panel grid of galaxy-galaxy lensing angular power spectra.

    One panel per (lens, source) bin pair: rows are lens bins,
    columns are source bins. Without C_gs_ref each curve is
    ell(ell+1) C_ell / 2 pi on a log scale; with C_gs_ref each curve
    is the fractional difference C / C_ref - 1.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_gs entry.
      C_gs     = list of 3D arrays (n_ell, n_lens, n_source), one
                 per curve.
      C_gs_ref = None, or one 3D array used as the ratio reference.
      param, colorbarlabel, lmin, lmax, cmap, colorbarshrink, ylim,
      linestyle, linewidth, marker, markersize, legend,
      legendfontsize, the *size arguments
      (yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize), bintextpos, bintextsize, figsize =
      layout knobs as in plot_C_ss_tomo_limber.
      legendloc = None (the default) lays one legend row right
                 above the panels, centered on them; an (x, y) pair
                 in figure fractions places its lower-left corner
                 anywhere.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 every panel then shares one y-range, interior y
                 axes disappear and the panels are glued together,
                 with the factor alpha annotated inside each panel
                 and the y-axis label reading alpha |C^gs|. Ignored
                 with C_gs_ref (already dimensionless and shared).
                 None (default) keeps per-panel y-ranges.
      alphatextpos = axes-fraction (x, y) anchoring the left edge
                 of the alpha annotation.
      ydecades = with rescale, cap on how many decades the shared
                 y-range extends below its ceiling (default 4):
                 deep |C| dips at sign crossings otherwise drag the
                 common floor down and compress every panel. None
                 keeps the full union of the panel ranges.
      ylabel   = with rescale, the single global y-axis label of
                 the glued grid (drawn once with fig.supylabel).

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """

    nell, nlens, nsource = C_gs[0].shape
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    
    if not (C_gs_ref is None):
        nell2, nlens2, nsource2 = C_gs_ref.shape
        if (nlens != nlens2) or (nell != nell2) or (nsource != nsource2):
            print("Bad Input")
            print(f"Nlens = {nlens}, Nlens_REF = {nlens2}")
            print(f"Nsource = {nsource}, Nsource_REF = {nsource2}")
            print(f"Nell = {nell}, Nell_REF = {nell2}")
            return 0   

    # rescale=1: alpha[i,j] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued grid. Panels
    # zeroed by init_ggl_exclude keep alpha = 0 and are skipped.
    rescale = None if not (C_gs_ref is None) else rescale
    alpha = np.zeros((nlens, nsource))
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(nlens):
            for j in range(nsource):
                pmax = max(np.max(np.abs(Cl[:,i,j])) for Cl in C_gs)
                if pmax == 0:
                    continue
                alpha[i,j] = -np.floor(np.log10(pmax))
                # the glued floor only counts positive minima: a curve
                # touching zero cannot set a log-axis lower limit
                lo = min(np.min(np.abs(Cl[:,i,j])) for Cl in C_gs)*10.0**alpha[i,j]
                if lo > 0:
                    panlo.append(lo)
                panhi.append(pmax*10.0**alpha[i,j])
        yglued = [ylim[0]*np.min(panlo if panlo else panhi), ylim[1]*np.max(panhi)]
        if not (ydecades is None):
            yglued[0] = max(yglued[0], yglued[1]/10.0**ydecades)

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge (see the module docstring)
    if C_gs_ref is None and rescale is None:
        fig, axes = plt.subplots(
            nrows = nsource,
            ncols = nlens,
            figsize = figsize,
            sharex = True,
            sharey = False,
            gridspec_kw = {'wspace': 0.275, 'hspace': 0.135})
    else:
        fig, axes = plt.subplots(
            nrows = nsource,
            ncols = nlens,
            figsize = figsize,
            sharex = True,
            sharey = True,
            gridspec_kw = {'wspace': 0, 'hspace': 0})

    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = cmap), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = 0.03, 
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(C_gs):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)

    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following entry, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])

    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
    
    # axes[j,i] is the panel at row j (source bin) and column i
    # (lens bin): every (lens, source) pair gets one panel
    for i in range(nlens):
        for j in range(nsource):
            # collect each curve's extremes in this panel: without rescale
            # the per-panel y-range is [ylim[0]*min, ylim[1]*max]
            clmin = []
            clmax = []
            for Cl in C_gs:  
                tmp = Cl[:,i,j]
                clmin.append(np.min(tmp))
                clmax.append(np.max(tmp))
 
            axes[j,i].set_xlim([lmin, lmax])
            
            # (lens, source) pairs dropped via init_ggl_exclude come back as
            # identically zero: such panels get an "excluded" placeholder,
            # since zeros can be neither log scaled nor used as a ratio ref.
            # (all(...) over a generator is True only when every curve
            # in the panel is identically zero)
            excluded = all(not np.any(Cl[:,i,j]) for Cl in C_gs)
            if not (C_gs_ref is None):
                excluded = excluded or not np.any(C_gs_ref[:,i,j])

            if C_gs_ref is None:
                if not (rescale is None):
                    axes[j,i].set_ylim(yglued)
                    axes[j,i].set_yscale('log')
                elif excluded:
                    axes[j,i].set_yticks([])
                else:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(clmin)), np.max(ylim[1]*np.array(clmax))])
                    axes[j,i].set_yscale('log')
            else:
                # with a reference every curve is value/reference - 1, so ylim
                # (multipliers around 1) is drawn as the band ylim - 1 around 0
                tmp = np.array(ylim) - 1
                axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_yscale('linear')

            axes[j,i].set_xscale('log')

            if i == 0:
                if C_gs_ref is None:
                    # with rescale the y label is global: one fig.supylabel
                    if rescale is None:
                        axes[j,i].set_ylabel(r"$|C_{\ell}^{gs}|$", fontsize=yaxislabelsize)
                else:
                    axes[j,i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
            for item in (axes[j,i].get_yticklabels()):
                item.set_fontsize(yaxisticklabelsize)
            for item in (axes[j,i].get_xticklabels()):
                item.set_fontsize(xaxisticklabelsize)
            
            if j == nsource-1:
                axes[j,i].set_xlabel(r"$\ell$", fontsize=xaxislabelsize)
            
            # transform=transAxes puts the text in panel fractions: (0, 0)
            # is the panel's lower-left corner, (1, 1) its upper-right
            axes[j,i].text(bintextpos[0], bintextpos[1], 
                "$(" +  str(i+1) + "," +  str(j+1) + ")$", 
                horizontalalignment = 'center', 
                verticalalignment = 'center',
                fontsize = bintextsize,
                usetex = True,
                transform = axes[j,i].transAxes)
            
            if excluded:
                axes[j,i].text(0.5, 0.5, "excluded",
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    transform = axes[j,i].transAxes)
                continue

            if not (rescale is None):
                expo = int(alpha[i,j])
                axes[j,i].text(alphatextpos[0], alphatextpos[1],
                    "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                    horizontalalignment = 'left',
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    usetex = True,
                    transform = axes[j,i].transAxes)

            for x, Cl in enumerate(C_gs):
                if C_gs_ref is None:
                    # 10**alpha = 1 unless rescale is on for this panel
                    tmp = Cl[:,i,j] * 10.0**alpha[i,j]
                else:
                    tmp = Cl[:,i,j] / C_gs_ref[:,i,j] - 1
                if marker is None:
                    axes[j,i].plot(ell,
                                   tmp,
                                   color=cm(x/len(C_gs)),
                                   linewidth=next(linewidthcycler),
                                   linestyle=next(linestylecycler))
                else:
                    axes[j,i].plot(ell,
                                   tmp,
                                   color=cm(x/len(C_gs)),
                                   markerfacecolor='None',
                                   marker=next(markercycler),
                                   markeredgecolor=cm(x/len(C_gs)),
                                   linestyle='None',
                                   markersize=markersize)

    if not (rescale is None):
        # the minus sign otherwise staggers the stacked y numbers
        _align_log_ticklabels(axes[:,0])
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == nsource-1, j == 0) for j in range(nsource)],
            yglued[0], yglued[1])
        _glued_supylabel(fig, axes[:,0], ylabel, yaxislabelsize)

    if not (C_gs_ref is None):
        # the ratio grid is glued too: same boundary-label pruning,
        # on the shared linear range ylim - 1
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == nsource-1, j == 0) for j in range(nsource)],
            ylim[0]-1.0, ylim[1]-1.0, log = False)

    if not (legend is None):
        if len(legend) != len(C_gs):
            print("Bad Input")
            return 0
        # legendloc None (the default) lays the entries in one row
        # right above the panels, centered on their measured span:
        # np.ravel flattens the axes array into one flat list,
        # get_position returns each panel's box in figure fractions,
        # and bbox_to_anchor pins the legend's lower-center point
        if legendloc is None:
            pos = [a.get_position() for a in np.ravel(axes)]
            cx = 0.5*(min(q.x0 for q in pos) + max(q.x1 for q in pos))
            ty = max(q.y1 for q in pos)
            legendloc = "lower center"
            legendanchor = (cx, ty + 0.008)
            ncols = min(len(legend), 4)
        else:
            legendanchor = None
            ncols = 1
        fig.legend(
            legend, 
            loc=legendloc,
            bbox_to_anchor=legendanchor,
            ncols=ncols,
            fontsize=legendfontsize,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)

    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def plot_C_gg_tomo(ell, C_gg, C_gg_ref = None, param = None, colorbarlabel = None, lmin = 30, lmax = 1500, 
                   cmap = 'gist_rainbow', ylim = [0.75,1.25], linestyle = None, linewidth = None,
                   legend = None, legendloc = None, yaxislabelsize = 16, yaxisticklabelsize = 10, 
                   xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], bintextsize = 15, figsize = (20, 12), 
                   show = 1, forcelinearyscale=False, overwriteylabel=None, forcelinearxscale=False,
                   marker = None, markersize=3, colorbar=1, colorbarshrink = 1.0, rescale = None,
                   alphatextpos = [0.05, 0.12], ydecades = 4, ylabel = r"$\alpha\,C_{\ell}^{gg}$", legendfontsize = None, xaxislabelsize = 16):
    """One panel per lens bin of galaxy-clustering angular spectra.

    The panels show the auto-correlation C_gg of each lens bin.
    Without C_gg_ref each curve is ell(ell+1) C_ell / 2 pi; with
    C_gg_ref each curve is the fractional difference C / C_ref - 1.

    Arguments:
      ell      = 1D array of multipoles, shared by every C_gg entry.
      C_gg     = list of 3D arrays (n_ell, n_lens, n_lens), one per
                 curve; the panels read the diagonal.
      C_gg_ref = None, or one 3D array used as the ratio reference.
      forcelinearyscale, forcelinearxscale = True switches that axis
                 to linear even without a reference.
      overwriteylabel = y-axis label replacing the default.
      marker, markersize = point markers instead of lines.
      param, colorbarlabel, lmin, lmax, cmap, colorbarshrink, ylim,
      linestyle, linewidth, marker, markersize, legend,
      legendfontsize, the *size arguments
      (yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize), bintextpos, bintextsize, figsize =
      layout knobs as in plot_C_ss_tomo_limber.
      legendloc = None (the default) lays one legend row right
                 above the panels, centered on them; an (x, y) pair
                 in figure fractions places its lower-left corner
                 anywhere.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 the row then shares one y-range and is glued, with
                 the factor alpha annotated inside each panel and
                 one global y label (ylabel, or overwriteylabel when
                 given). Ignored with C_gg_ref. None (default)
                 keeps per-panel y-ranges.
      alphatextpos = axes-fraction (x, y) anchoring the left edge
                 of the alpha annotation.
      ydecades = with rescale, cap on how many decades the shared
                 y-range extends below its ceiling (default 4).
                 None keeps the full union of the panel ranges.
      ylabel   = with rescale, the single global y-axis label of
                 the glued row (drawn once with fig.supylabel).

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """

    nell, nlens1, nlens2 = C_gg[0].shape   
    if nlens1 != nlens2:
        print("Bad Input (number of nlens1/nlens2)")
        return 0
    if nell != len(ell):
        print("Bad Input (number of ell)")
        return 0
    if not (C_gg_ref is None):
        nell2, nlens3, nlens4 = C_gg_ref.shape
        if nlens3 != nlens4:
            print("Bad Input (number of nlens3/nlens4)")
            return 0
        if (nlens1 != nlens3) or (nell != nell2):
            print("Bad Input")
            print(f"Nlens  = {nlens1}, Nlens_REF = {nlens3}")
            print(f"Nell = {nell}, Nell_REF = {nell2}")
            return 0   
    
    # rescale=1: alpha[i] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued row.
    rescale = None if not (C_gg_ref is None) else rescale
    alpha = np.zeros(nlens1)
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(nlens1):
            pmax = max(np.max(np.abs(Cl[:,i,i])) for Cl in C_gg)
            if pmax == 0:
                continue
            alpha[i] = -np.floor(np.log10(pmax))
            # the glued floor only counts positive minima: a curve
            # touching zero cannot set a log-axis lower limit
            lo = min(np.min(np.abs(Cl[:,i,i])) for Cl in C_gg)*10.0**alpha[i]
            if lo > 0:
                panlo.append(lo)
            panhi.append(pmax*10.0**alpha[i])
        yglued = [ylim[0]*np.min(panlo if panlo else panhi), ylim[1]*np.max(panhi)]
        if not (ydecades is None) and forcelinearyscale != True:
            yglued[0] = max(yglued[0], yglued[1]/10.0**ydecades)

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge (see the module docstring)
    if C_gg_ref is None and rescale is None:
        fig, axes = plt.subplots(
            nrows = 1,
            ncols = nlens1,
            figsize = figsize,
            sharex = True,
            sharey = False,
            gridspec_kw = {'wspace': 0.275, 'hspace': 0.135})
    else:
        fig, axes = plt.subplots(
            nrows = 1, 
            ncols = nlens1, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = cmap), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = 0.03, 
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(C_gg):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)     
    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following entry, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
    
    # one panel per lens bin in a single row: axes[i] is column i,
    # and the [:,i,i] slices below read the (i, i) diagonal, the
    # auto-correlation of lens bin i with itself
    for i in range(nlens1):
        # collect each curve's extremes in this panel: without rescale
        # the per-panel y-range is [ylim[0]*min, ylim[1]*max]
        clmin = []
        clmax = []
        for Cl in C_gg:  
            tmp = Cl[:,i,i]
            clmin.append(np.min(tmp))
            clmax.append(np.max(tmp))

        axes[i].set_xlim([lmin, lmax])
        
        if C_gg_ref is None:
            if not (rescale is None):
                axes[i].set_ylim(yglued)
                axes[i].set_yscale('log')
            else:
                axes[i].set_ylim([np.min(ylim[0]*np.array(clmin)), np.max(ylim[1]*np.array(clmax))])
                axes[i].set_yscale('log')
            if forcelinearyscale == True:
                axes[i].set_yscale('linear')
        else:
            # with a reference every curve is value/reference - 1, so ylim
            # (multipliers around 1) is drawn as the band ylim - 1 around 0
            tmp = np.array(ylim) - 1
            axes[i].set_ylim(tmp.tolist())
            axes[i].set_yscale('linear')

        axes[i].set_xscale('log')
        if forcelinearxscale == True:
            axes[i].set_xscale('linear')

        if i == 0:
            if C_gg_ref is None:
                # with rescale the y label is global: one fig.supylabel
                if rescale is None:
                    axes[i].set_ylabel(r"$C_{\ell}^{gg}$",fontsize=yaxislabelsize)
                    if not (overwriteylabel is None):
                        axes[i].set_ylabel(overwriteylabel,fontsize=yaxislabelsize)
            else:
                axes[i].set_ylabel("frac. diff.",fontsize=yaxislabelsize)
                if not (overwriteylabel is None):
                    axes[i].set_ylabel(overwriteylabel,fontsize=yaxislabelsize)
        
        for item in (axes[i].get_yticklabels()):
            item.set_fontsize(yaxisticklabelsize)
        for item in (axes[i].get_xticklabels()):
            item.set_fontsize(xaxisticklabelsize)
        
        axes[i].set_xlabel(r"$\ell$", fontsize=xaxislabelsize)
        
        # transform=transAxes puts the text in panel fractions: (0, 0)
        # is the panel's lower-left corner, (1, 1) its upper-right
        axes[i].text(bintextpos[0], bintextpos[1], 
            "$(" +  str(i+1) + ")$", 
            horizontalalignment = 'center', 
            verticalalignment = 'center',
            fontsize = bintextsize,
            usetex = True,
            transform = axes[i].transAxes)
        
        if not (rescale is None):
            expo = int(alpha[i])
            axes[i].text(alphatextpos[0], alphatextpos[1],
                "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                horizontalalignment = 'left',
                verticalalignment = 'center',
                fontsize = bintextsize,
                usetex = True,
                transform = axes[i].transAxes)

        for x, Cl in enumerate(C_gg):
            if C_gg_ref is None:
                # 10**alpha = 1 unless rescale is on for this panel
                tmp = Cl[:,i,i] * 10.0**alpha[i]
            else:
                tmp = Cl[:,i,i] / C_gg_ref[:,i,i] - 1
            
            if marker is None:
                axes[i].plot(ell, 
                             tmp, 
                             color=cm(x/len(C_gg)), 
                             linewidth=next(linewidthcycler), 
                             linestyle=next(linestylecycler))
            else:
                axes[i].plot(ell, 
                             tmp, 
                             color=cm(x/len(C_gg)), 
                             markerfacecolor='None',
                             marker=next(markercycler),
                             markeredgecolor=cm(x/len(C_gg)),
                             linestyle='None',
                             markersize=markersize)
    
    if not (rescale is None):
        if forcelinearyscale != True:
            # the minus sign otherwise staggers the stacked y numbers
            _align_log_ticklabels([axes[0]])
        # the row is glued horizontally, so the clash is between the
        # x tick labels at interior panel boundaries
        _hide_glued_edge_ticklabels(
            [(axes[i], i == 0, i == nlens1-1) for i in range(nlens1)],
            lmin, lmax, axis = "x", log = (forcelinearxscale != True))
        _glued_supylabel(fig, [axes[0]],
            ylabel if overwriteylabel is None else overwriteylabel,
            yaxislabelsize)

    if not (C_gg_ref is None):
        # the ratio row is glued too: prune the x tick labels at
        # interior panel boundaries
        _hide_glued_edge_ticklabels(
            [(axes[i], i == 0, i == nlens1-1) for i in range(nlens1)],
            lmin, lmax, axis = "x", log = (forcelinearxscale != True))

    if not (legend is None):
        if len(legend) != len(C_gg):
            print("Bad Input")
            return 0
        # legendloc None (the default) lays the entries in one row
        # right above the panels, centered on their measured span:
        # np.ravel flattens the axes array into one flat list,
        # get_position returns each panel's box in figure fractions,
        # and bbox_to_anchor pins the legend's lower-center point
        if legendloc is None:
            pos = [a.get_position() for a in np.ravel(axes)]
            cx = 0.5*(min(q.x0 for q in pos) + max(q.x1 for q in pos))
            ty = max(q.y1 for q in pos)
            legendloc = "lower center"
            legendanchor = (cx, ty + 0.008)
            ncols = min(len(legend), 4)
        else:
            legendanchor = None
            ncols = 1
        fig.legend(
            legend, 
            loc=legendloc,
            bbox_to_anchor=legendanchor,
            ncols=ncols,
            fontsize=legendfontsize,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)

    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def plot_gammat_tomo_limber(theta_gammat, gammat_ref = None, param = None, colorbarlabel = None, marker = None,
                            linestyle = None, linewidth = None, ylim = [0.75,1.25],
                            cmap = 'gist_rainbow', legend = None, legendloc = None, yaxislabelsize = 16,
                            yaxisticklabelsize = 10,  xaxisticklabelsize = 20, bintextpos = [0.2, 0.85],
                            bintextsize = 15, figsize = (12, 12), show = 1, colorbar=1,
                     colorbarshrink = 0.5, markersize = 3, 
                     thetashow = None, rescale = None, alphatextpos = [0.05, 0.12],
                     ydecades = 4, ylabel = r"$\alpha\,|\gamma_{t}(\theta)|$", legendfontsize = None, xaxislabelsize = 16):
    """Panel grid of the real-space tangential shear gamma_t(theta).

    One panel per (lens, source) bin pair: rows are lens bins,
    columns are source bins. Without gammat_ref each curve is
    theta * gamma_t * 10^4; with gammat_ref each curve is the
    fractional difference gamma_t / ref - 1.

    Arguments:
      theta_gammat = list of (theta, gammat) pairs, one per curve:
                 theta in arcmin, gammat a 3D array
                 (n_theta, n_lens, n_source).
      gammat_ref = None, or one (theta, gammat) pair used as the
                 ratio reference.
      thetashow = x-axis range in arcmin; None (default) spans the
                 theta arrays themselves.
      marker   = list of matplotlib markers cycled across curves
                 (points instead of lines), or None for lines.
      param, colorbarlabel, cmap, colorbarshrink, ylim, linestyle,
      linewidth, markersize, legend, legendfontsize,
      the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize),
      bintextpos, bintextsize, figsize = layout knobs as in
      plot_C_ss_tomo_limber.
      legendloc = None (the default) lays one legend row right
                 above the panels, centered on them; an (x, y) pair
                 in figure fractions places its lower-left corner
                 anywhere.
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
                 y-range extends below its ceiling (default 4):
                 deep |gamma_t| dips at sign crossings otherwise
                 drag the common floor down and compress every
                 panel. None keeps the full union of panel ranges.
      ylabel   = with rescale, the single global y-axis label of
                 the glued grid (drawn once with fig.supylabel).

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """

    (theta, gammat) = theta_gammat[0]
    ntheta, nlens, nsource = gammat.shape

    if ntheta != len(theta):
        print("Bad Input (theta)")
        print(theta)
        print(ntheta, len(theta))
        return 0

    if thetashow is None:
        thetashow = [np.min(theta), np.max(theta)]

    if not (gammat_ref is None):
        (theta1, gammat2) = gammat_ref
        ntheta2, nlens2, nsource2 = gammat2.shape
        if (nlens != nlens2) or (ntheta != ntheta2) or (nsource != nsource2):
            print("Bad Input")
            print(f"Nlens = {nlens}, Nlens_REF = {ntheta2}")
            print(f"Nsource = {nsource}, Nsource_REF = {nsource2}")
            print(f"Ntheta = {ntheta}, Ntheta_REF = {ntheta2}")
            return 0   
        
    # rescale=1: alpha[i,j] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued grid. Panels
    # zeroed by init_ggl_exclude keep alpha = 0 and are skipped.
    rescale = None if not (gammat_ref is None) else rescale
    alpha = np.zeros((nlens, nsource))
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(nlens):
            for j in range(nsource):
                pmax = max(np.max(np.abs(g[:,i,j])) for (t, g) in theta_gammat)
                if pmax == 0:
                    continue
                alpha[i,j] = -np.floor(np.log10(pmax))
                # the glued floor only counts positive minima: a curve
                # touching zero cannot set a log-axis lower limit
                lo = min(np.min(np.abs(g[:,i,j])) for (t, g) in theta_gammat)*10.0**alpha[i,j]
                if lo > 0:
                    panlo.append(lo)
                panhi.append(pmax*10.0**alpha[i,j])
        yglued = [ylim[0]*np.min(panlo if panlo else panhi), ylim[1]*np.max(panhi)]
        if not (ydecades is None):
            yglued[0] = max(yglued[0], yglued[1]/10.0**ydecades)

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge (see the module docstring)
    if gammat_ref is None and rescale is None:
        fig, axes = plt.subplots(
            nrows = nsource,
            ncols = nlens,
            figsize = figsize,
            sharex = True,
            sharey = False,
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = nsource, 
            ncols = nlens, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = cmap), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = 0.03, 
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(theta_gammat):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
        
    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following entry, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])

    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])

    # axes[j,i] is the panel at row j (source bin) and column i
    # (lens bin): every (lens, source) pair gets one panel
    for i in range(nlens):
        for j in range(nsource):
            # collect each curve's extremes in this panel: without rescale
            # the per-panel y-range is [ylim[0]*min, ylim[1]*max]
            ximin = []
            ximax = []
            for (theta, gammat) in theta_gammat:  
                ximin.append(np.min(np.abs(gammat[:,i,j])))
                ximax.append(np.max(np.abs(gammat[:,i,j])))
 
            axes[j,i].set_xlim(thetashow)
            
            # (lens, source) pairs dropped via init_ggl_exclude come back as
            # identically zero: such panels get an "excluded" placeholder,
            # since zeros can be neither log scaled nor used as a ratio ref.
            # (all(...) over a generator is True only when every curve
            # in the panel is identically zero)
            excluded = all(not np.any(g[:,i,j]) for (t, g) in theta_gammat)
            if not (gammat_ref is None):
                excluded = excluded or not np.any(gammat_ref[1][:,i,j])

            if gammat_ref is None:
                if not (rescale is None):
                    axes[j,i].set_ylim(yglued)
                    axes[j,i].set_yscale('log')
                elif excluded:
                    axes[j,i].set_yticks([])
                else:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)),np.max(ylim[1]*np.array(ximax))])
                    axes[j,i].set_yscale('log')
            else:
                # with a reference every curve is value/reference - 1, so ylim
                # (multipliers around 1) is drawn as the band ylim - 1 around 0
                tmp = np.array(ylim) - 1
                axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_yscale('linear')
                
            axes[j,i].set_xscale('log')
            
            if i == 0:
                if gammat_ref is None:
                    # with rescale the y label is global: one fig.supylabel
                    if rescale is None:
                        axes[j,i].set_ylabel(r"$|\gamma_{t}(\theta)|$", fontsize=yaxislabelsize)
                else:
                    axes[j,i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
            for item in (axes[j,i].get_yticklabels()):
                item.set_fontsize(yaxisticklabelsize)
            for item in (axes[j,i].get_xticklabels()):
                item.set_fontsize(xaxisticklabelsize)
            
            if j == nsource-1:
                axes[j,i].set_xlabel(r"$\theta$", fontsize=xaxislabelsize)
            
            # transform=transAxes puts the text in panel fractions: (0, 0)
            # is the panel's lower-left corner, (1, 1) its upper-right
            axes[j,i].text(bintextpos[0], bintextpos[1], 
                "$(" +  str(i+1) + "," +  str(j+1) + ")$", 
                horizontalalignment = 'center', 
                verticalalignment = 'center',
                fontsize = bintextsize,
                usetex = True,
                transform = axes[j,i].transAxes)

            if excluded:
                axes[j,i].text(0.5, 0.5, "excluded",
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    transform = axes[j,i].transAxes)
                continue

            if not (rescale is None):
                expo = int(alpha[i,j])
                axes[j,i].text(alphatextpos[0], alphatextpos[1],
                    "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                    horizontalalignment = 'left',
                    verticalalignment = 'center',
                    fontsize = bintextsize,
                    usetex = True,
                    transform = axes[j,i].transAxes)

            for x, (theta, gammat) in enumerate(theta_gammat):
                if gammat_ref is None:
                    # 10**alpha = 1 unless rescale is on for this panel
                    tmp = np.abs(gammat[:,i,j]) * 10.0**alpha[i,j]
                else:
                    (theta1, gammat2) = gammat_ref
                    tmp = gammat[:,i,j]/gammat2[:,i,j] - 1
                
                if marker is None:
                    axes[j,i].plot(theta, 
                                   tmp, 
                                   color=cm(x/len(theta_gammat)), 
                                   linewidth=next(linewidthcycler), 
                                   linestyle=next(linestylecycler))
                else:
                    axes[j,i].plot(theta, 
                                   tmp, 
                                   color=cm(x/len(theta_gammat)), 
                                   markerfacecolor='None', 
                                   marker=next(markercycler),
                                   markeredgecolor=cm(x/len(theta_gammat)), 
                                   linestyle='None', 
                                   markersize=markersize)                    
    
    if not (rescale is None):
        # the minus sign otherwise staggers the stacked y numbers
        _align_log_ticklabels(axes[:,0])
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == nsource-1, j == 0) for j in range(nsource)],
            yglued[0], yglued[1])
        _glued_supylabel(fig, axes[:,0], ylabel, yaxislabelsize)

    if not (gammat_ref is None):
        # the ratio grid is glued too: same boundary-label pruning,
        # on the shared linear range ylim - 1
        _hide_glued_edge_ticklabels(
            [(axes[j,0], j == nsource-1, j == 0) for j in range(nsource)],
            ylim[0]-1.0, ylim[1]-1.0, log = False)

    if not (legend is None):
        if len(legend) != len(theta_gammat):
            print("Bad Input")
            return 0
        # legendloc None (the default) lays the entries in one row
        # right above the panels, centered on their measured span:
        # np.ravel flattens the axes array into one flat list,
        # get_position returns each panel's box in figure fractions,
        # and bbox_to_anchor pins the legend's lower-center point
        if legendloc is None:
            pos = [a.get_position() for a in np.ravel(axes)]
            cx = 0.5*(min(q.x0 for q in pos) + max(q.x1 for q in pos))
            ty = max(q.y1 for q in pos)
            legendloc = "lower center"
            legendanchor = (cx, ty + 0.008)
            ncols = min(len(legend), 4)
        else:
            legendanchor = None
            ncols = 1
        fig.legend(
            legend, 
            loc=legendloc,
            bbox_to_anchor=legendanchor,
            ncols=ncols,
            fontsize=legendfontsize,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)

    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def plot_wtheta_tomo(theta_wtheta, theta_wtheta_ref = None, param = None, colorbarlabel = None, marker = None, 
                     linestyle = None, linewidth = None, ylim = [0.75,1.25],
                     cmap = 'gist_rainbow', legend = None, legendloc = None, yaxislabelsize = 16, 
                     yaxisticklabelsize = 10,  xaxisticklabelsize = 20, bintextpos = [0.2, 0.85], 
                     bintextsize = 15, figsize = (12, 12), show = 1, colorbar=1,
                     colorbarshrink = 1.0, markersize = 3, 
                     thetashow = None, rescale = None, alphatextpos = [0.05, 0.12],
                     ydecades = 4, ylabel = r"$\alpha\,|w_{t}(\theta)|$", legendfontsize = None, xaxislabelsize = 16):
    """One panel per lens bin of the clustering correlation w(theta).

    Without theta_wtheta_ref each curve is theta * w(theta) * 10^4;
    with it each curve is the fractional difference w / ref - 1.

    Arguments:
      theta_wtheta = list of (theta, wtheta) pairs, one per curve:
                 theta in arcmin, wtheta a 3D array
                 (n_theta, n_lens, n_lens); the panels read the
                 diagonal.
      theta_wtheta_ref = None, or one (theta, wtheta) pair used as
                 the ratio reference.
      thetashow = x-axis range in arcmin; None (default) spans the
                 theta arrays themselves.
      marker   = list of matplotlib markers cycled across curves
                 (points instead of lines), or None for lines.
      param, colorbarlabel, cmap, colorbarshrink, ylim, linestyle,
      linewidth, markersize, legend, legendfontsize,
      the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize),
      bintextpos, bintextsize, figsize = layout knobs as in
      plot_C_ss_tomo_limber.
      legendloc = None (the default) lays one legend row right
                 above the panels, centered on them; an (x, y) pair
                 in figure fractions places its lower-left corner
                 anywhere.
      show     = 1 draws the figure; None returns (fig, axes).
      colorbar = None suppresses the colorbar even with param set.
      rescale  = 1 multiplies each panel by its own power of ten,
                 chosen so the rescaled maximum lands in [1, 10):
                 the row then shares one y-range and is glued, with
                 the factor alpha annotated inside each panel and
                 one global y label (the ylabel argument). Ignored
                 with theta_wtheta_ref. None (default) keeps
                 per-panel y-ranges.
      alphatextpos = axes-fraction (x, y) anchoring the left edge
                 of the alpha annotation.
      ydecades = with rescale, cap on how many decades the shared
                 y-range extends below its ceiling (default 4).
                 None keeps the full union of the panel ranges.
      ylabel   = with rescale, the single global y-axis label of
                 the glued row (drawn once with fig.supylabel).

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, axes) when show is None.
    """

    (theta, wtheta) = theta_wtheta[0]
    ntheta, nlens1, nlens2 = wtheta.shape
    if nlens1 != nlens2:
        print("Bad Input (number of nlens1/nlens2)")
        return 0

    if ntheta != len(theta):
        print("Bad Input (theta)")
        print(theta)
        print(ntheta, len(theta))
        return 0

    if thetashow is None:
        thetashow = [np.min(theta), np.max(theta)]

    if not (theta_wtheta_ref is None):
        (theta1, wtheta2) = theta_wtheta_ref
        ntheta2, nlens3, nlens4 = wtheta2.shape
        if nlens3 != nlens4:
            print("Bad Input (number of nlens3/nlens4)")
            return 0
        if (nlens1 != nlens3) or (ntheta != ntheta2):
            print("Bad Input")
            print(f"Nlens = {nlens1}, Nlens_REF = {nlens3}")
            print(f"Ntheta = {ntheta}, Ntheta_REF = {ntheta2}")
            return 0   
        
    # rescale=1: alpha[i] holds the log10 of the per-panel factor;
    # multiplied in, every panel's maximum lands in [1, 10), so one
    # common y-range (yglued) serves the whole glued row.
    rescale = None if not (theta_wtheta_ref is None) else rescale
    alpha = np.zeros(nlens1)
    if not (rescale is None):
        panlo, panhi = [], []
        for i in range(nlens1):
            pmax = max(np.max(np.abs(w[:,i,i])) for (t, w) in theta_wtheta)
            if pmax == 0:
                continue
            alpha[i] = -np.floor(np.log10(pmax))
            # the glued floor only counts positive minima: a curve
            # touching zero cannot set a log-axis lower limit
            lo = min(np.min(np.abs(w[:,i,i])) for (t, w) in theta_wtheta)*10.0**alpha[i]
            if lo > 0:
                panlo.append(lo)
            panhi.append(pmax*10.0**alpha[i])
        yglued = [ylim[0]*np.min(panlo if panlo else panhi), ylim[1]*np.max(panhi)]
        if not (ydecades is None):
            yglued[0] = max(yglued[0], yglued[1]/10.0**ydecades)

    # sharex/sharey tie the panels' axis limits together; the glued
    # branch also zeroes wspace and hspace, the gaps between panels,
    # so neighbors touch edge to edge (see the module docstring)
    if theta_wtheta_ref is None and rescale is None:
        fig, axes = plt.subplots(
            nrows = 1,
            ncols = nlens1,
            figsize = figsize,
            sharex = True,
            sharey = False,
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = 1, 
            ncols = nlens1, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0, 'hspace': 0})
    
    cm = plt.get_cmap(cmap)
    
    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = cmap), 
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = 0.03, 
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 20, weight = 'bold', labelpad = 2)
        if len(param) != len(theta_wtheta):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)    
    # itertools.cycle repeats a list forever: each next(...) in the
    # panel loop pulls the following entry, wrapping at the end
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])

    # one panel per lens bin in a single row: axes[i] is column i,
    # and the [:,i,i] slices below read the (i, i) diagonal, the
    # auto-correlation of lens bin i with itself
    for i in range(nlens1):
        # collect each curve's extremes in this panel: without rescale
        # the per-panel y-range is [ylim[0]*min, ylim[1]*max]
        ximin = []
        ximax = []
        for (theta, wtheta) in theta_wtheta:  
            ximin.append(np.min(np.abs(wtheta[:,i,i])))
            ximax.append(np.max(np.abs(wtheta[:,i,i])))

        axes[i].set_xlim(thetashow)
        
        if theta_wtheta_ref is None:
            if not (rescale is None):
                axes[i].set_ylim(yglued)
                axes[i].set_yscale('log')
            else:
                axes[i].set_ylim([np.min(ylim[0]*np.array(ximin)),np.max(ylim[1]*np.array(ximax))])
                axes[i].set_yscale('log')
        else:
            # with a reference every curve is value/reference - 1, so ylim
            # (multipliers around 1) is drawn as the band ylim - 1 around 0
            tmp = np.array(ylim) - 1
            axes[i].set_ylim(tmp.tolist())
            axes[i].set_yscale('linear')

        axes[i].set_xscale('log')

        if i == 0:
            if theta_wtheta_ref is None:
                # with rescale the y label is global: one fig.supylabel
                if rescale is None:
                    axes[i].set_ylabel(r"$|w_{t}(\theta)|$", fontsize=yaxislabelsize)
            else:
                axes[i].set_ylabel("frac. diff.", fontsize=yaxislabelsize)
        for item in (axes[i].get_yticklabels()):
            item.set_fontsize(yaxisticklabelsize)
        for item in (axes[i].get_xticklabels()):
            item.set_fontsize(xaxisticklabelsize)

        axes[i].set_xlabel(r"$\theta$", fontsize=xaxislabelsize)
        
        # transform=transAxes puts the text in panel fractions: (0, 0)
        # is the panel's lower-left corner, (1, 1) its upper-right
        axes[i].text(bintextpos[0], bintextpos[1], 
            "$(" +  str(i+1) + ")$", 
            horizontalalignment = 'center', 
            verticalalignment = 'center',
            fontsize = bintextsize,
            usetex = True,
            transform = axes[i].transAxes)

        if not (rescale is None):
            expo = int(alpha[i])
            axes[i].text(alphatextpos[0], alphatextpos[1],
                "$\\alpha=1$" if expo == 0 else f"$\\alpha=10^{{{expo}}}$",
                horizontalalignment = 'left',
                verticalalignment = 'center',
                fontsize = bintextsize,
                usetex = True,
                transform = axes[i].transAxes)

        for x, (theta, wtheta) in enumerate(theta_wtheta):
            if theta_wtheta_ref is None:
                # 10**alpha = 1 unless rescale is on for this panel
                tmp = np.abs(wtheta[:,i,i]) * 10.0**alpha[i]
            else:
                (theta1, wtheta2) = theta_wtheta_ref
                tmp = wtheta[:,i,i]/wtheta2[:,i,i] - 1
            
            if marker is None:
                axes[i].plot(theta, 
                               tmp, 
                               color=cm(x/len(theta_wtheta)), 
                               linewidth=next(linewidthcycler), 
                               linestyle=next(linestylecycler))
            else:
                axes[i].plot(theta, 
                               tmp, 
                               color=cm(x/len(theta_wtheta)), 
                               markerfacecolor='None', 
                               marker=next(markercycler),
                               markeredgecolor=cm(x/len(theta_wtheta)), 
                               linestyle='None', 
                               markersize=markersize)                    
    
    if not (rescale is None):
        # the minus sign otherwise staggers the stacked y numbers
        _align_log_ticklabels([axes[0]])
        # the row is glued horizontally, so the clash is between the
        # x tick labels at interior panel boundaries
        _hide_glued_edge_ticklabels(
            [(axes[i], i == 0, i == nlens1-1) for i in range(nlens1)],
            thetashow[0], thetashow[1], axis = "x")
        _glued_supylabel(fig, [axes[0]], ylabel, yaxislabelsize)

    if not (theta_wtheta_ref is None):
        # the ratio row is glued too: prune the x tick labels at
        # interior panel boundaries
        _hide_glued_edge_ticklabels(
            [(axes[i], i == 0, i == nlens1-1) for i in range(nlens1)],
            thetashow[0], thetashow[1], axis = "x")

    if not (legend is None):
        if len(legend) != len(theta_wtheta):
            print("Bad Input")
            return 0
        # legendloc None (the default) lays the entries in one row
        # right above the panels, centered on their measured span:
        # np.ravel flattens the axes array into one flat list,
        # get_position returns each panel's box in figure fractions,
        # and bbox_to_anchor pins the legend's lower-center point
        if legendloc is None:
            pos = [a.get_position() for a in np.ravel(axes)]
            cx = 0.5*(min(q.x0 for q in pos) + max(q.x1 for q in pos))
            ty = max(q.y1 for q in pos)
            legendloc = "lower center"
            legendanchor = (cx, ty + 0.008)
            ncols = min(len(legend), 4)
        else:
            legendanchor = None
            ncols = 1
        fig.legend(
            legend, 
            loc=legendloc,
            bbox_to_anchor=legendanchor,
            ncols=ncols,
            fontsize=legendfontsize,
            borderpad=0.1,
            handletextpad=0.4,
            handlelength=1.5,
            columnspacing=0.35,
            scatteryoffsets=[0],
            frameon=False)
    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, axes)


def plot_baryon_suppression(log10k, sup, param = None, colorbarlabel = None, 
                            zlabels = None, cmap = 'gist_rainbow', ylim = None, 
                            linewidth = None, legendloc = None, title = None, 
                            titlesize = 16, yaxislabelsize = 16, yaxisticklabelsize = 14, 
                            xaxisticklabelsize = 14, xaxislabelsize = 16, 
                            legendfontsize = None, figsize = (10, 6), show = 1, 
                            colorbar = 1, colorbarshrink = 1.0):
    """One panel of baryonic feedback suppression curves S(k).

    Made for parameter sweeps of the bfmt theory block: each entry of
    sup is one curve set (one parameter value), colored by param, and
    each of its rows is one redshift slice, drawn with its own
    linestyle (solid, dashed, dashdot, dotted, cycling).

    Arguments:
      log10k   = 1D array, log10 of k (the block's own unit, 1/Mpc).
      sup      = list of 2D arrays (n_z, n_k), one per parameter
                 value.
      param    = list of parameter values (one per curve) coloring
                 the curves and the colorbar, or None.
      colorbarlabel = colorbar label, or None.
      zlabels  = one label per redshift slice for the linestyle
                 legend, or None for no legend; legendloc places it
                 (None picks matplotlib's best corner) and
                 legendfontsize sizes its entries in points.
      ylim     = explicit (lo, hi), or None for matplotlib's choice.
      title    = panel title (e.g. "method: vary parameter"), or
                 None; titlesize sizes it.
      linewidth = list cycled across curves, or None.
      cmap, colorbarshrink, the *size arguments (yaxislabelsize,
      xaxislabelsize, yaxisticklabelsize, xaxisticklabelsize),
      figsize = layout knobs as in plot_C_ss_tomo_limber.
      show     = 1 draws the figure; None returns (fig, ax).
      colorbar = None suppresses the colorbar even with param set.

    Returns:
      0 on malformed input (a printed message names the problem),
      None after drawing, or (fig, ax) when show is None.
    """

    # sup arrives as a python list whose entries may themselves be
    # nested lists; np.asarray converts the first entry to a numpy
    # array so .shape can be read, and the (rows, columns) tuple
    # unpacks into the number of redshift slices and of k points
    nz, nk = np.asarray(sup[0]).shape
    # validate before building anything: a size mismatch caught here
    # names the problem, instead of surfacing later as a matplotlib
    # broadcast error deep inside ax.plot
    if nk != len(log10k):
        print("Bad Input (number of k)")
        return 0
    if not (zlabels is None) and len(zlabels) != nz:
        print("Bad Input (number of zlabels)")
        return 0

    # plt.subplots builds one figure and one drawing area (the axes)
    # in a single call and returns them as a pair
    fig, ax = plt.subplots(figsize = figsize)
    # a colormap is a function: cm(0.0) is the first color of the
    # map, cm(1.0) the last, with a continuous blend in between
    cm = plt.get_cmap(cmap)

    if not (param is None or colorbar is None):
        # the colorbar is not read off the plotted lines: it is drawn
        # from a ScalarMappable, a bare description of "this colormap
        # spans these values", with Normalize mapping the parameter
        # range onto the colormap's 0..1 axis; the curves use the same
        # colormap, so bar and line colors agree
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(param[0], param[-1]), cmap = cmap),
            ax = ax,
            orientation = 'vertical',
            aspect = 30,   # bar length over bar width: larger = thinner
            pad = 0.02,    # gap to the panel, as a fraction of the axes
            shrink = colorbarshrink)
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, size = 18, weight = 'bold', labelpad = 2)
        if len(param) != len(sup):
            print("Bad Input")
            return 0

    # itertools.cycle repeats a list forever: each next(...) in the
    # curve loop pulls the following width, wrapping at the end
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.5])

    # np.power works elementwise: one call turns the whole log10 grid
    # back into k values, no loop needed
    k = np.power(10.0, log10k)
    # one linestyle per redshift slice; iz % len(zstyles) is the
    # remainder of the division, so a fifth slice wraps around and
    # reuses the solid style
    zstyles = ['solid', 'dashed', 'dashdot', 'dotted']
    # enumerate yields (position, entry) pairs, so x counts the
    # curves; x/len(sup) is this curve's position in 0..1, the
    # coordinate the colormap expects
    for x, S in enumerate(sup):
        S = np.asarray(S)
        lw = next(linewidthcycler)
        for iz in range(nz):
            # S[iz] is row iz of the 2D array: this parameter value's
            # suppression at one redshift, over all k
            ax.plot(k,
                    S[iz],
                    color = cm(x/len(sup)),
                    linewidth = lw,
                    linestyle = zstyles[iz % len(zstyles)])

    # feedback acts over decades in k, so the x axis is logarithmic;
    # S is of order one, so the y axis stays linear
    ax.set_xscale('log')
    ax.set_xlim([k[0], k[-1]])
    if not (ylim is None):
        # set_ylim accepts any two-element sequence; list(...) turns
        # the (lo, hi) tuple into one
        ax.set_ylim(list(ylim))
    ax.set_xlabel(r"$k$ [1/Mpc]", fontsize = xaxislabelsize)
    ax.set_ylabel(r"$S(k) = P_{\rm feedback}/P_{\rm DM}$", fontsize = yaxislabelsize)
    # the tick numbers are Text objects; matplotlib has no single
    # "tick font size" setter on the axes, so each label is resized
    # on its own
    for item in ax.get_yticklabels():
        item.set_fontsize(yaxisticklabelsize)
    for item in ax.get_xticklabels():
        item.set_fontsize(xaxisticklabelsize)
    if not (title is None):
        ax.set_title(title, fontsize = titlesize)

    if not (zlabels is None):
        # the legend must key the linestyles, not the plotted lines:
        # every curve of one redshift shares a style but has its own
        # color. Line2D([], []) is a line with no data points - it
        # never draws inside the panel and exists only as a black
        # legend key for one style (a "proxy handle" in matplotlib
        # terms). The comprehension builds one such key per redshift
        # slice, in slice order.
        handles = [matplotlib.lines.Line2D([], [], color = 'black',
                       linestyle = zstyles[iz % len(zstyles)]) for iz in range(nz)]
        # ax.legend pairs handles with labels position by position;
        # the conditional expression falls back to matplotlib's
        # 'best' corner search when no location was passed
        ax.legend(handles, zlabels,
                  loc = 'best' if legendloc is None else legendloc,
                  fontsize = legendfontsize,
                  frameon = False)

    # warn=False: outside a notebook, showing a figure on a
    # non-interactive backend would otherwise warn
    if not (show is None):
        fig.show(warn=False)
    else:
        return (fig, ax)
