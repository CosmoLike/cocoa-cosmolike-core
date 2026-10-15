"""Illustrate natural-cubic and 3-, 5-, 7-point Akima histogram curves.

The same normalized uniform histogram is reconstructed in four columns.
The lower row zooms onto one edge. Both odd-factor centres and the narrower
transitions are visible. This is an illustration, not survey input data.
GSL values/slopes and exact lookup subdivision are checked before saving.

Run in Cocoa's Python environment:
    python dev_scripts/plot_nz_akima_preparation.py --output-directory figures

The output directory receives PNG/PDF figures and a JSON area/check record.
"""
import argparse
import ctypes
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicHermiteSpline, CubicSpline


def check_gsl_akima(knots, values, query, curve):
    """Check the plotted Akima values and slopes against native GSL.

    All arguments use the dimensionless coordinate of the illustration.
    The function returns maximum absolute value and derivative differences.
    """
    library = Path(sys.prefix) / 'lib/libgsl.dylib'
    if sys.platform != 'darwin':
        library = Path(sys.prefix) / 'lib/libgsl.so'
    gsl = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    gsl.gsl_spline_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    gsl.gsl_spline_alloc.restype = ctypes.c_void_p
    gsl.gsl_spline_init.argtypes = [ctypes.c_void_p, pointer, pointer,
                                  ctypes.c_size_t]
    gsl.gsl_spline_init.restype = ctypes.c_int
    gsl.gsl_spline_eval.argtypes = [ctypes.c_void_p, ctypes.c_double,
                                  ctypes.c_void_p]
    gsl.gsl_spline_eval.restype = ctypes.c_double
    gsl.gsl_spline_eval_deriv.argtypes = gsl.gsl_spline_eval.argtypes
    gsl.gsl_spline_eval_deriv.restype = ctypes.c_double
    gsl.gsl_spline_free.argtypes = [ctypes.c_void_p]
    kind = ctypes.c_void_p.in_dll(gsl, 'gsl_interp_akima')
    spline = gsl.gsl_spline_alloc(kind, knots.size)
    if not spline:
        raise MemoryError('GSL could not allocate the illustrative Akima spline.')
    native_values = np.empty_like(query)
    native_slopes = np.empty_like(query)
    try:
        status = gsl.gsl_spline_init(spline, knots.ctypes.data_as(pointer),
                                     values.ctypes.data_as(pointer), knots.size)
        if status != 0:
            raise RuntimeError(f'GSL Akima initialization returned status {status}.')
        for index, coordinate in enumerate(query):
            native_values[index] = gsl.gsl_spline_eval(spline, coordinate, None)
            native_slopes[index] = gsl.gsl_spline_eval_deriv(spline, coordinate, None)
    finally:
        gsl.gsl_spline_free(spline)
    np.testing.assert_allclose(actual=curve(query), desired=native_values,
                               rtol=2.e-13, atol=2.e-14)
    np.testing.assert_allclose(actual=curve(query, nu=1), desired=native_slopes,
                               rtol=2.e-13, atol=2.e-14)
    return {
        'library': str(library),
        'max_value_difference': float(np.max(np.abs(curve(query)-native_values))),
        'max_derivative_difference': float(np.max(
            np.abs(curve(query, nu=1)-native_slopes))),
    }


def curve_metrics(curve, edges, heights):
    """Measure exact spline areas and extrema inside the histogram support."""
    roots = curve.derivative().roots(extrapolate=False)
    roots = roots[np.isfinite(roots)]
    candidates = np.concatenate((edges, curve.x, roots))
    inside = (candidates >= edges[0]) & (candidates <= edges[-1])
    extrema = curve(candidates[inside])
    probabilities = []
    for left, right in zip(edges[:-1], edges[1:]):
        probabilities.append(curve.integrate(a=left, b=right))
    probabilities = np.asarray(probabilities)
    expected = heights * np.diff(edges)
    return {
        'minimum_density': float(np.min(extrema)),
        'maximum_density': float(np.max(extrema)),
        'total_probability': float(np.sum(probabilities)),
        'original_bin_probabilities': expected.tolist(),
        'spline_bin_probabilities': probabilities.tolist(),
        'max_original_bin_probability_change': float(np.max(
            np.abs(probabilities-expected))),
    }


def main():
    """Draw matched full-range and edge panels, then save their checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    edges = np.arange(8.)
    heights = np.array([0., .3, 1., 3., 1., .1, 0.])
    heights /= np.sum(heights * np.diff(edges))
    midpoints = .5 * (edges[:-1] + edges[1:])
    knots = []
    values = []
    curves = []
    for factor in (1, 3, 5, 7):
        repeated = np.repeat(heights, repeats=factor)
        centres = (np.arange(repeated.size + 2) - .5) / factor
        padded = np.concatenate((repeated[:1], repeated, repeated[-1:]))
        knots.append(centres)
        values.append(padded)
        curves.append(CubicSpline(x=centres, y=padded, bc_type='natural'))
    coordinate = np.linspace(0., 7., 7001)
    native_checks = {}
    subdivision_checks = {}
    for index, factor in enumerate((3, 5, 7), start=1):
        akima = Akima1DInterpolator(x=knots[index], y=values[index])
        # Each lookup interval lies inside one Akima polynomial. Preserving
        # values AND slopes reproduces that polynomial after subdivision.
        refine = int(np.ceil(5 / factor))
        fine_knots = np.linspace(
            knots[index][0], knots[index][-1],
            refine * (knots[index].size - 1) + 1)
        final_akima = CubicHermiteSpline(
            x=fine_knots, y=akima(fine_knots), dydx=akima(fine_knots, nu=1))
        np.testing.assert_allclose(
            actual=final_akima(coordinate), desired=akima(coordinate),
            atol=2.e-14, rtol=2.e-13)
        np.testing.assert_allclose(
            actual=final_akima(midpoints), desired=heights,
            atol=2.e-14, rtol=2.e-13)
        native_checks[str(factor)] = check_gsl_akima(
            knots=knots[index], values=values[index],
            query=coordinate, curve=akima)
        subdivision_checks[str(factor)] = float(np.max(
            np.abs(final_akima(coordinate) - akima(coordinate))))
        curves[index] = final_akima
    titles = [
        'Natural cubic through original centres',
        '3× zero order → Akima',
        '5× zero order → Akima',
        '7× zero order → Akima',
    ]
    colours = ['#bd5124', '#168568', '#3763ad', '#8756a3']
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelsize': 11})
    fig, axes = plt.subplots(2, 4, figsize=(24, 8.5), sharey='row')
    for column, curve_index in enumerate((0, 1, 2, 3)):
        for row in range(2):
            ax = axes[row, column]
            ax.stairs(values=heights, edges=edges, color='0.28', linewidth=1.6,
                      fill=True, alpha=.13, label='Original histogram')
            ax.stairs(values=heights, edges=edges, color='0.28', linewidth=1.5)
            ax.plot(coordinate, curves[curve_index](coordinate),
                    color=colours[curve_index],
                    linewidth=2.4, label='Smooth reconstruction')
            ax.scatter(knots[curve_index][1:-1], values[curve_index][1:-1],
                       color=colours[curve_index], s=26, zorder=5,
                       label='Points defining the spline')
            ax.scatter(midpoints, heights, s=78, facecolors='none',
                       edgecolors='black', linewidths=1.2, zorder=6,
                       label='Original bin centres')
            ax.axhline(0., color='0.55', linewidth=.8)
            ax.grid(alpha=.13)
            ax.set_xlabel('Coordinate / original bin width')
            if column == 0:
                ax.set_ylabel('Normalized density')
        axes[0, column].set_title(titles[curve_index])
        axes[0, column].set_xlim(0., 7.)
        axes[0, column].set_ylim(-.025, .62)
        axes[0, column].legend(loc='upper left', fontsize=8.4)
        axes[1, column].set_xlim(2.45, 3.55)
        axes[1, column].set_ylim(.12, .62)
        axes[1, column].axvline(3., color='0.4', linestyle=':', linewidth=1)
        axes[1, column].set_title('Close-up of the same bin edge', fontsize=11)
        if column == 3:
            positions = [2.5, 41/14, 3., 43/14, 3.5]
            labels = ['2½', '2.929', '3\n(bin edge)', '3.071', '3½']
        elif column == 2:
            positions = [2.5, 2.9, 3., 3.1, 3.5]
            labels = ['2½', '2.9', '3\n(bin edge)', '3.1', '3½']
        else:
            positions = [2.5, 17/6, 3., 19/6, 3.5]
            labels = ['2½', '2⅚', '3\n(bin edge)', '3⅙', '3½']
        axes[1, column].set_xticks(positions, labels=labels)
    for column, factor in enumerate((3, 5, 7), start=1):
        ax = axes[1, column]
        colour = colours[column]
        ax.annotate(
            'Flat plateau', xy=(2.68, heights[2]), xytext=(2.49, .28),
            color=colour, fontsize=11,
            arrowprops={'arrowstyle': '->', 'color': colour})
        half_width = .5 / factor
        ax.annotate(
            '', xy=(3. - half_width, .575), xytext=(3. + half_width, .575),
            arrowprops={'arrowstyle': '<->', 'color': colour})
        ax.text(3., .584, f'Transition width: 1/{factor} bin',
                ha='center', color=colour, fontsize=9)
    fig.suptitle('Original cubic versus 3-, 5- and 7-point Akima preparation',
                 fontsize=20, y=.98)
    fig.text(.5, .055,
             'All three odd factors retain the original bin centre. '
             '3 points: 1/6, 1/2, 5/6.  '
             '5 points: 1/10, 3/10, 1/2, 7/10, 9/10.  '
             '7 points: 1/14, 3/14, …, 13/14 of each bin.',
             ha='center', fontsize=11)
    fig.text(.5, .027,
             'Same illustrative uniform histogram and axis scales. '
             'More points narrow the smoothed edge; '
             'all three Akima curves still redistribute probability between bins.',
             ha='center', fontsize=10.5)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(output / 'histogram_original_cubic_vs_three_five_seven_akima.png', dpi=180)
    fig.savefig(output / 'histogram_original_cubic_vs_three_five_seven_akima.pdf')
    plt.close(fig)
    metrics = {}
    for title, curve in zip(titles, curves):
        metrics[title] = curve_metrics(curve=curve, edges=edges, heights=heights)
    metrics['gsl_akima_checks'] = native_checks
    metrics['lookup_subdivision_max_value_change'] = subdivision_checks
    path = output / 'histogram_three_five_seven_checks.json'
    path.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
