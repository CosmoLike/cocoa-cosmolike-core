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
    # Perceived brightness of an RGB color (the ITU-R BT.709 weights;
    # green dominates because the eye is most sensitive to it). Used
    # below to draw brighter, harder-to-see colors with thicker lines.
    r, g, b = rgb[:3]
    return 0.2126*r + 0.7152*g + 0.0722*b


def _style_factor(ls):
    # Extra thickness for dashed styles. A pattern with short "on"
    # segments puts less ink on the page than a solid line and looks
    # thinner than it is; compensate by up to a factor of 2.
    if isinstance(ls, str):
        return 1.0  # solid line: no extra thickness
    _, dashes = ls
    on_lengths = dashes[0::2]   # only the "on" (ink) segments
    mean_on = sum(on_lengths) / len(on_lengths)
    # shorter dashes => larger factor; clamp to keep it sane
    return float(np.clip(6.0 / mean_on, 1.0, 2.0))


def plot_response_function(k, resp, labels, ylabel, idx = None, normalize = True,
                           ncolors = None, ntomo = 8, xtickformat = "2g",
                           cmap = 'berlin', figsize = (20, 4), fontsize = 16,
                           show = 1):
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
               axis, or None when label j simply is slice j.
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
      cmap, figsize, fontsize = matplotlib layout knobs.
      show   = call plt.show() at the end.
    """
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
    plt.xlabel("k [h/Mpc]", fontsize=fontsize)
    ax = plt.gca()
    ax.set_xscale("log")
    if xtickformat == "scalar":
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.xlim(k[0], k[-1])
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if show:
        plt.show()
