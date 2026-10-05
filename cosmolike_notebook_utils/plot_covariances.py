"""Covariance figures shared by survey notebooks.

The split-triangle comparison follows the visual idea of Friedrich et al.
(2021), Fig. 6, arXiv:2012.08568. Component maps and their distributions
follow Barreira, Krause & Schmidt (2018), Fig. 1, arXiv:1807.04266.
The functions plot the supplied calculation, not those papers' data.

Only NumPy and Matplotlib are used. Labels, ordering and survey choices
belong to the caller. No covariance is rescaled, repaired or saved here;
fonts and global Matplotlib settings stay in the notebook. show=1 draws
the figure; show=None returns (figure, axes) for further annotation/saving.
"""

import numpy as np
from matplotlib import pyplot as plt


def _matrix(values, name):
    """Check the matrix before creating a figure; preserve signed entries."""
    matrix = np.asarray(a=values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values")
    scale = np.max(np.abs(matrix))
    if np.max(np.abs(matrix-matrix.T)) > 1.e-12*scale:
        raise ValueError(f"{name} must be symmetric; check the matrix ordering")
    return matrix


def _blocks(size, block_sizes, block_labels):
    """Resolve contiguous matrix groups; labels sit at group centers."""
    if block_sizes is None:
        if block_labels is not None:
            raise ValueError("block_labels requires block_sizes")
        return None, None, None
    counts = np.asarray(a=block_sizes)
    if counts.ndim != 1 or counts.dtype.kind not in "iu" or np.any(counts <= 0):
        raise ValueError("block_sizes must contain positive integer counts")
    if np.sum(counts) != size:
        raise ValueError("block_sizes must sum to the covariance dimension")
    if block_labels is None or len(block_labels) != len(counts):
        raise ValueError("provide one block label for each block size")
    edges = np.concatenate((np.array([0]), np.cumsum(counts)))-0.5
    centers = (edges[1:]+edges[:-1])/2.0
    return edges, centers, block_labels


def _decorate(axis, blocks):
    """Mark estimator/tomographic boundaries without altering matrix order."""
    edges, centers, labels = blocks
    if edges is None:
        axis.set_xlabel(xlabel="Data-vector index")
        axis.set_ylabel(ylabel="Data-vector index")
        return
    for boundary in edges[1:-1]:
        axis.axvline(x=boundary, color="0.25", linewidth=0.6)
        axis.axhline(y=boundary, color="0.25", linewidth=0.6)
    axis.set_xticks(ticks=centers, labels=labels, rotation=45, ha="right")
    axis.set_yticks(ticks=centers, labels=labels)


def _finish(figure, axes, show):
    """Keep the same display/return convention as the data-vector plotters."""
    if show is not None:
        plt.show()
        return None
    return figure, axes


def plot_trispectrum_terms(k_hmpc, components, redshift, linthresh=1.0,
                          figsize=(9, 6), show=1):
    """Show signed halo contributions to the matter trispectrum diagonal.

    Arguments:
        k_hmpc = positive increasing [nk] wavenumbers in h/Mpc.
        components = ordered mapping of labels to finite [nk] arrays,
            each containing one contribution to T(k,k) in (Mpc/h)^9.
            Supply disjoint terms, e.g. 1h, combined 2h, 3h and 4h;
            their sum is drawn as Total. Do not also supply the total.
        redshift = finite nonnegative redshift shared by these arrays.
        linthresh = positive threshold in (Mpc/h)^9 for the y axis:
            linear between -linthresh and +linthresh, logarithmic outside.
        figsize = figure size in inches.
        show = 1 to display; None to return the figure and axis.
    Returns:
        None or (figure, axis). Invalid inputs raise ValueError before
        creating a figure. Inputs are neither modified nor saved.

    The supplied terms are averaged over the relative wavevector angle,
    before survey projection. They exclude SSC. A signed logarithmic
    scale keeps negative terms and zero crossings visible; no absolute
    value, clipping or covariance normalization changes the data.
    """
    wave = np.asarray(a=k_hmpc, dtype=float)
    if wave.ndim != 1 or len(wave) == 0 or not np.all(np.isfinite(wave)):
        raise ValueError("k_hmpc must be nonempty, finite and 1D")
    if np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError("k_hmpc must be positive and increasing")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    if not np.isfinite(linthresh) or linthresh <= 0:
        raise ValueError("linthresh must be finite and positive")
    if not components:
        raise ValueError("components must contain at least one named halo term")

    # Check every term before drawing. Adding the disjoint contributions
    # reconstructs cNG's matter trispectrum, not the full G+SSC+cNG covariance.
    curves = {}
    total = np.zeros_like(a=wave)
    for name, values in components.items():
        term = np.asarray(a=values, dtype=float)
        if term.shape != wave.shape or not np.all(np.isfinite(term)):
            raise ValueError(f"{name} must be finite with shape {wave.shape}")
        curves[name] = term
        total += term

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    # Set the scales before plotting so margins are measured in log space,
    # rather than adding a large linear margin to a many-decade range.
    axis.set_xscale(value="log")
    axis.set_yscale(value="symlog", linthresh=linthresh)
    styles = ["solid", "dashed", "dashdot", "dotted"]
    for index, (name, term) in enumerate(curves.items()):
        # Matplotlib takes plotting coordinates as positional x,y arguments.
        axis.plot(wave, term, label=name, linewidth=1.8,
                  linestyle=styles[index % len(styles)])
    axis.plot(wave, total, label="Total", color="black", linewidth=2.0)
    axis.set_xlabel(xlabel=r"$k\;[h/\mathrm{Mpc}]$", fontsize=17)
    axis.set_ylabel(ylabel=r"$\overline{T}(k,k)\;[(\mathrm{Mpc}/h)^9]$",
                    fontsize=17)
    axis.set_title(label=f"Matter trispectrum diagonal, z = {redshift:g}")
    axis.tick_params(axis="both", labelsize=14)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc="outside upper center",
                  ncol=len(curves)+1, frameon=False)
    return _finish(figure=figure, axes=axis, show=show)


def plot_correlation(covariance, covariance_ref=None, block_sizes=None,
                     block_labels=None, labels=("Calculation", "Reference"),
                     cmap="RdBu_r", figsize=(7, 6), show=1):
    """Plot correlations, optionally comparing two matrix triangles.

    Arguments:
        covariance, covariance_ref = symmetric [ndata,ndata] matrices.
            Every diagonal must be positive. The optional reference fills
            the upper triangle; the calculation fills the lower triangle.
        block_sizes, block_labels = contiguous group counts and labels,
            or both None for index axes. Their order is never sorted.
        labels = two display names, calculation then reference.
        cmap, figsize, show = Matplotlib colormap, inches, display control.
    Returns:
        None for show=1, or (figure, axis) for show=None.

    Each matrix uses its own diagonal: R_ij=C_ij/sqrt(C_ii C_jj).
    Negative correlations are physical and remain visible. Values outside
    [-1,1] expand the scale instead of being clipped; such values violate
    the covariance Cauchy--Schwarz bound. A well-behaved picture alone does
    not establish positive definiteness: inspect the eigenvalues too.
    """
    values = _matrix(values=covariance, name="covariance")
    blocks = _blocks(size=len(values), block_sizes=block_sizes,
                     block_labels=block_labels)
    if len(labels) != 2:
        raise ValueError("labels needs calculation and reference names")
    diagonal = np.diag(v=values)
    if np.any(diagonal <= 0):
        raise ValueError("correlation normalization needs positive diagonal variances")
    result = values/np.sqrt(diagonal[:, None]*diagonal[None, :])
    if covariance_ref is not None:
        reference = _matrix(values=covariance_ref, name="covariance_ref")
        if reference.shape != values.shape or np.any(np.diag(reference) <= 0):
            raise ValueError("reference needs matching shape and positive diagonal")
        ref_diagonal = np.diag(v=reference)
        ref_correlation = reference/np.sqrt(
            ref_diagonal[:, None]*ref_diagonal[None, :]
        )
        upper = np.triu_indices(n=len(values), k=1)
        result[upper] = ref_correlation[upper]

    limit = max(1.0, float(np.max(np.abs(result))))
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    artist = axis.imshow(X=result, origin="lower", cmap=cmap,
                         vmin=-limit, vmax=limit, interpolation="nearest")
    _decorate(axis=axis, blocks=blocks)
    title = labels[0]
    if covariance_ref is not None:
        title = f"Lower: {labels[0]}   |   Upper: {labels[1]}"
    axis.set_title(label=title)
    figure.colorbar(mappable=artist, ax=axis, label=r"$C_{ij}/\sqrt{C_{ii}C_{jj}}$")
    return _finish(figure=figure, axes=axis, show=show)


def plot_covariance_components(total, components, block_sizes=None,
                               block_labels=None, normalization="diagonal",
                               denominator_floor=1.e-12, bins=50,
                               percent=False, figsize=None, show=1):
    """Compare covariance components with maps and a shared histogram.

    Arguments:
        total = finite symmetric [ndata,ndata] matrix with positive diagonal.
        components = ordered mapping of display name to component matrix.
        block_sizes, block_labels = groups as in plot_correlation.
        normalization = 'diagonal' gives component_ij/sqrt(total_ii total_jj);
            'element' gives component_ij/total_ij, as in the paper's Fig. 1.
        denominator_floor = for element ratios, mask abs(total_ij) <= this
            fraction of sqrt(total_ii total_jj). Must be nonnegative.
        bins = histogram bin count; percent = multiply ratios by 100.
        figsize = inches or None to size by component count.
        show = 1 to display, None to return figure and map/histogram axes.
    Returns:
        None or (figure, axes dict), with keys maps and histogram.

    A component may be signed. Maps share a diverging scale and the
    histogram retains those signs. Undefined element ratios are grey and
    excluded from the histogram, with their counts stated in each title.
    Element ratios can be large when total entries nearly cancel; they
    do not measure the effect on cosmological parameter constraints.
    """
    values = _matrix(values=total, name="total")
    if np.any(np.diag(values) <= 0):
        raise ValueError("total needs positive diagonal variances")
    if normalization not in ("diagonal", "element"):
        raise ValueError("normalization must be 'diagonal' or 'element'")
    if not np.isfinite(denominator_floor) or denominator_floor < 0:
        raise ValueError("denominator_floor must be finite and nonnegative")
    if not components:
        raise ValueError("components must contain at least one named matrix")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    blocks = _blocks(size=len(values), block_sizes=block_sizes,
                     block_labels=block_labels)
    diagonal = np.diag(v=values)
    scale = np.sqrt(diagonal[:, None]*diagonal[None, :])
    denominator = scale
    valid = np.ones(shape=values.shape, dtype=bool)
    if normalization == "element":
        denominator = values
        valid = np.abs(values) > denominator_floor*scale
    if not np.any(valid):
        raise ValueError("no ratios survive denominator_floor; inspect total")
    ratios = {}
    limit = 0.0
    for name, component in components.items():
        matrix = _matrix(values=component, name=name)
        if matrix.shape != values.shape:
            raise ValueError(f"{name} does not match the total covariance shape")
        ratio = np.full(shape=values.shape, fill_value=np.nan)
        np.divide(matrix, denominator, out=ratio, where=valid)
        if percent:
            ratio *= 100.0
        ratios[name] = np.ma.masked_invalid(a=ratio)
        limit = max(limit, float(np.max(np.abs(ratio[valid]))))
    if limit == 0:
        limit = 1.0
    if figsize is None:
        figsize = (5*len(components), 8)

    figure = plt.figure(figsize=figsize, constrained_layout=True)
    grid = figure.add_gridspec(nrows=2, ncols=len(components),
                               height_ratios=[1.0, 0.55])
    histogram = figure.add_subplot(grid[1, :])
    panels = []
    colormap = plt.get_cmap(name="RdBu_r").copy()
    colormap.set_bad(color="0.8")
    edges = np.linspace(start=-limit, stop=limit, num=bins+1)
    masked_count = int(np.count_nonzero(~valid))
    for index, (name, ratio) in enumerate(ratios.items()):
        axis = figure.add_subplot(grid[0, index])
        artist = axis.imshow(X=ratio, origin="lower", cmap=colormap,
                             vmin=-limit, vmax=limit, interpolation="nearest")
        _decorate(axis=axis, blocks=blocks)
        axis.set_title(label=f"{name} ({masked_count} masked)")
        panels.append(axis)
        histogram.hist(x=ratio.compressed(), bins=edges, histtype="step",
                       linewidth=1.5, label=name)
    label = r"$C^{\rm component}_{ij}/\sqrt{C^{\rm total}_{ii}C^{\rm total}_{jj}}$"
    if normalization == "element":
        label = r"$C^{\rm component}_{ij}/C^{\rm total}_{ij}$"
    if percent:
        label = "100 × "+label+" [%]"
    figure.colorbar(mappable=artist, ax=panels, label=label, shrink=0.8)
    histogram.set_xlabel(xlabel=label)
    histogram.set_ylabel(ylabel="Matrix entries")
    histogram.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                     ncol=len(components), frameon=False)
    return _finish(figure=figure, axes={"maps": panels, "histogram": histogram},
                   show=show)


def plot_covariance_diagonal(theta_arcmin, covariances, panel_labels,
                             covariance_ref=None, figsize=None, show=1,
                             coordinate_label=r"$\theta$ [arcmin]"):
    """Plot standard deviations or their fractional changes by estimator.

    Arguments:
        theta_arcmin = positive increasing [ntheta] bin centers. For Fourier
            plots, supply multipole centers and a multipole coordinate_label.
            The historical argument name remains valid for angular callers.
        coordinate_label = horizontal axis label, including the supplied units.
        covariances = ordered mapping of name to [ndata,ndata] covariance.
        panel_labels = estimator/tomographic labels; ndata=ntheta*len(labels).
        covariance_ref = optional matching reference matrix. If supplied,
            plot 100*(sqrt(diag(C)/diag(C_ref))-1), in percent, rather than
            standard deviation.
        figsize, show = inches (None for automatic size), display control.
    Returns:
        None or (figure, axes [npanel]). No component with negative diagonal
        can be square rooted; use the signed component map for that case.
    """
    theta = np.asarray(a=theta_arcmin, dtype=float)
    if theta.ndim != 1 or len(theta) == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("theta_arcmin must be nonempty, finite and 1D")
    if np.any(theta <= 0) or np.any(np.diff(theta) <= 0):
        raise ValueError("theta_arcmin must be positive and increasing")
    if not panel_labels or not covariances:
        raise ValueError("provide panel labels and at least one covariance")
    size = len(theta)*len(panel_labels)
    diagonals = {}
    for name, matrix in covariances.items():
        values = _matrix(values=matrix, name=name)
        if values.shape != (size, size) or np.any(np.diag(values) <= 0):
            raise ValueError(f"{name} needs matching shape and positive diagonal")
        diagonals[name] = np.sqrt(np.diag(values)).reshape(
            (len(panel_labels), len(theta))
        )
    if covariance_ref is not None:
        reference = _matrix(values=covariance_ref, name="covariance_ref")
        if reference.shape != (size, size) or np.any(np.diag(reference) <= 0):
            raise ValueError("reference needs matching shape and positive diagonal")
        baseline = np.sqrt(np.diag(reference)).reshape(
            (len(panel_labels), len(theta))
        )
        for name in diagonals:
            diagonals[name] = 100.0*(diagonals[name]/baseline-1.0)
    if figsize is None:
        figsize = (5*len(panel_labels), 4)
    figure, axes = plt.subplots(nrows=1, ncols=len(panel_labels),
                                figsize=figsize, squeeze=False,
                                constrained_layout=True)
    panels = axes[0]
    for index, label in enumerate(panel_labels):
        axis = panels[index]
        styles = ["solid", "dashed", "dashdot", "dotted"]
        for curve, (name, diagonal) in enumerate(diagonals.items()):
            # Plot coordinates use Matplotlib's positional x,y convention.
            # Distinct dashes keep nearly coincident boost curves visible.
            axis.plot(theta, diagonal[index], label=name,
                      linestyle=styles[curve % len(styles)], linewidth=1.6)
        axis.set_xscale(value="log")
        axis.set_title(label=label)
        axis.set_xlabel(xlabel=coordinate_label)
        if covariance_ref is None:
            axis.set_yscale(value="log")
            axis.set_ylabel(ylabel=r"$\sqrt{C_{ii}}$")
        else:
            axis.axhline(y=0, color="0.4", linewidth=0.7)
            axis.set_ylabel(ylabel=r"$100(\sigma_i/\sigma_i^{\rm ref}-1)$ [%]")
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles=handles, labels=labels, loc="outside upper center",
                  ncol=len(covariances), frameon=False)
    return _finish(figure=figure, axes=panels, show=show)
