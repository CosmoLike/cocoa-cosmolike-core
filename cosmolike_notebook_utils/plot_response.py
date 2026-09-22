"""Response-function plots shared by the EXAMPLE_EVALUATE notebooks.

Curves of a data vector's response to the matter power spectrum as a
function of wavenumber k: either the local logarithmic derivative
d ln DV / d ln k or its cumulative integral R(k_max). Pure matplotlib
and numpy: nothing here touches CAMB or the compiled cosmolike
interface, so every project shares this function unchanged. The
notebooks compute the response arrays with their own interface-bound
functions and hand the finished arrays here.
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

# One dash pattern per tomographic bin. A matplotlib line style is
# either the string "-" (solid) or a tuple (offset, (on, off, on,
# off, ...)) giving the dash pattern in points: (0, (6, 2)) draws 6
# points of ink, then a 2-point gap, and repeats.
_LINESTYLES = [
    "-",                  # solid
    (0, (6, 2)),          # long dash
    (0, (1, 1)),          # dotted
    (0, (3, 1, 1, 1)),    # dense dash-dot
    (0, (10, 3)),         # very long dash
    (0, (5, 1, 1, 1, 1, 1)),             # dash-dot-dot
    (0, (6, 2, 1, 2, 1, 2, 1, 2)),       # dash-dot-dot-dot
    (0, (6, 2, 1, 2, 1, 2, 1, 2, 1, 2))  # dash-dot-dot-dot-dot
]


def _luminance(rgb):
    """Perceived brightness of an RGB color, between 0 and 1.

    Uses the ITU-R BT.709 weights; green dominates because the eye
    is most sensitive to it. plot_response_function draws brighter,
    harder-to-see colors with thicker lines, and this is its
    brightness measure.

    Arguments:
      rgb = color as (r, g, b) or (r, g, b, a), each channel in
            [0, 1]; an alpha channel is ignored.

    Returns:
      the weighted channel sum, a float in [0, 1].
    """
    r, g, b = rgb[:3]
    return 0.2126*r + 0.7152*g + 0.0722*b


def _style_factor(ls):
    """Extra line thickness compensating a dashed style's ink loss.

    A pattern with short "on" segments puts less ink on the page
    than a solid line of the same width and looks thinner than it
    is; the factor grows as the mean "on" segment shrinks.

    Arguments:
      ls = a _LINESTYLES entry: the string "-" (solid) or an
           (offset, dash pattern) tuple.

    Returns:
      a float in [1, 2]: 1 for solid, up to 2 for the shortest
      dashes, so no style gets thinner than solid or more than
      twice as thick.
    """
    if isinstance(ls, str):
        return 1.0  # solid line: no extra thickness
    _, dashes = ls
    on_lengths = dashes[0::2]   # only the "on" (ink) segments
    mean_on = sum(on_lengths) / len(on_lengths)
    return float(np.clip(6.0 / mean_on, 1.0, 2.0))


def plot_response_function(k, resp, labels, ylabel, idx = None, normalize = True,
                           ncolors = None, ntomo = 8, xtickformat = "2g",
                           cmap = 'berlin', figsize = (20, 4), fontsize = 16,
                           show = 1, yaxislabelsize = None, xaxislabelsize = None, 
                           yaxisticklabelsize = None, xaxisticklabelsize = None, 
                           legendfontsize = None):
    """Response of a data vector to the matter power spectrum vs k.

    One curve per (label, tomographic bin) pair, all in one panel.
    The label picks a slice of the response array (a multipole ell,
    or an angular bin theta) and sets the color; the diagonal
    tomographic bin (i, i) sets the dash pattern. The same figure
    serves the local response d ln DV / d ln k (normalize=True
    divides each curve by its own maximum) and the cumulative
    response R(k_max) (normalize=False plots the array as given).

    Arguments:
      k      = wavenumbers in h/Mpc (the x axis).
      resp   = 4D response array (n_k, n_slice, n_bin, n_bin), as
               the notebook response functions return it; only the
               diagonal bins (i, i) are plotted.
      labels = one LaTeX legend label per plotted slice.
      ylabel = y-axis label (LaTeX string).
      idx    = position of each label's slice on the array's second
               axis, or None when label j is slice j.
      normalize = divide each curve by its own maximum.
      ncolors = how many colors the colormap is split into, or None
               for one per label. Passing more than len(labels)
               reproduces figures whose color list was built for a
               longer slice axis.
      ntomo  = number of tomographic bins; bin (i, i) takes dash
               pattern i.
      xtickformat = "2g" for two-significant-digit x tick labels
               (0.01, 0.1, 1, 10), or "scalar" for matplotlib's
               plain number formatter.
      cmap, figsize = matplotlib layout knobs.
      fontsize = one size for every text element; the family
               knobs below override it one by one.
      yaxislabelsize, xaxislabelsize, yaxisticklabelsize,
      xaxisticklabelsize, legendfontsize = the same size names
               every plotter of plot_datavectors takes; None
               (default) falls back to fontsize.
      show   = call plt.show() at the end.
    """
    # the family size knobs override the shared fontsize one by
    # one; a None keeps the shared value, so fontsize alone still
    # sizes everything at once
    if yaxislabelsize is None:
        yaxislabelsize = fontsize
    if xaxislabelsize is None:
        xaxislabelsize = fontsize
    if yaxisticklabelsize is None:
        yaxisticklabelsize = fontsize
    if xaxisticklabelsize is None:
        xaxisticklabelsize = fontsize
    if legendfontsize is None:
        legendfontsize = fontsize
    if ncolors is None:
        ncolors = len(labels)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, ncolors))
    plt.figure(figsize=figsize)
    for i in range(0, ntomo):
        ls = _LINESTYLES[i % len(_LINESTYLES)]
        for j in range(0, len(labels)):
            c = colors[j]
            if idx is None:
                sel = j
            else:
                sel = idx[j]
            y = resp[:, sel, i, i]
            if normalize:
                y = y / np.max(y)
            lw = 0.9 + 1.2*_luminance(c)*_style_factor(ls)
            if i == 0:
                # label only the first dash pattern, so the legend
                # shows one entry per slice instead of ntomo copies
                plt.plot(k, y, color=c, label=labels[j], ls=ls, linewidth=lw)
            else:
                plt.plot(k, y, color=c, ls=ls, linewidth=lw)
    plt.xlabel("k [h/Mpc]", fontsize=xaxislabelsize)
    ax = plt.gca()
    ax.set_xscale("log")
    if xtickformat == "scalar":
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=xaxisticklabelsize)
    ax.tick_params(axis="y", which="major", labelsize=yaxisticklabelsize)
    plt.xlim(k[0], k[-1])
    plt.ylabel(ylabel, fontsize=yaxislabelsize)
    plt.legend(fontsize=legendfontsize)
    if show:
        plt.show()
