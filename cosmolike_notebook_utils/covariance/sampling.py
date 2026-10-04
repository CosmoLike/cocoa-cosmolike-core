"""Construct dense linear lookup tables from sparse exact calculations.

Expensive halo responses can be sampled at a small number of fixed physical
wavenumbers. A cubic spline fills a uniform log-k table once; subsequent
queries only read two adjacent nodes. The query does not evaluate a cubic
spline and does not search the table. This separation matters when a
Limber projection requests many changing k=(ell+1/2)/f_K values.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class DenseLogTable:
    """Cubic construction and linear queries on one fixed physical k interval.

    Arguments:
        k = strictly increasing positive coarse wavenumbers [ncoarse].
        values = finite [...,ncoarse] samples; the final axis is k.
        ndense = integer number of uniform log-k nodes, at least ncoarse.
        logarithmic = interpolate ln(values), requiring positive samples;
            use False for signed responses. The default is False.
    Returns:
        A table object; calling it with k returns [...,*k.shape] samples.
        Units are unchanged. Queries outside the specified interval raise
        ValueError rather than silently extrapolating a covariance response.

    Both ncoarse and ndense require refinement checks at off-grid queries.
    More dense nodes cannot recover a feature missing from the coarse data.
    """

    def __init__(self, k, values, ndense, logarithmic=False):
        """Validate the supplied samples before constructing a cubic spline."""
        wave = np.asarray(a=k, dtype=float)
        samples = np.asarray(a=values, dtype=float)
        if wave.ndim != 1 or len(wave) < 4:
            raise ValueError("k needs at least four coarse samples")
        if not np.all(np.isfinite(wave)) or np.any(wave <= 0):
            raise ValueError("k must be finite and positive")
        if np.any(np.diff(wave) <= 0):
            raise ValueError("k must increase strictly")
        if samples.ndim < 1 or samples.shape[-1] != len(wave):
            raise ValueError("the final values axis must match k")
        if not np.all(np.isfinite(samples)):
            raise ValueError("values must be finite")
        if not isinstance(ndense, (int, np.integer)) or ndense < len(wave):
            raise ValueError("ndense must be an integer at least len(k)")
        if logarithmic and np.any(samples <= 0):
            raise ValueError("logarithmic interpolation needs positive values")

        coordinate = np.log(wave)
        self.minimum_k = float(wave[0])
        self.maximum_k = float(wave[-1])
        self.minimum = float(coordinate[0])
        self.step = float((coordinate[-1]-coordinate[0])/(ndense-1))
        self.logarithmic = logarithmic
        if logarithmic:
            samples = np.log(samples)

        # The cubic is evaluated only during construction. Keeping the dense
        # table in the same representation preserves signed quantities, or
        # positivity when logarithmic interpolation was explicitly selected.
        spline = CubicSpline(x=coordinate, y=samples, axis=-1)
        dense_coordinate = np.linspace(
            start=coordinate[0], stop=coordinate[-1], num=ndense
        )
        self.values = np.ascontiguousarray(spline(x=dense_coordinate))

    def __call__(self, k):
        """Read adjacent dense nodes by arithmetic indexing; do not extrapolate."""
        wave = np.asarray(a=k, dtype=float)
        if not np.all(np.isfinite(wave)):
            raise ValueError("query k must be finite")
        if np.any(wave < self.minimum_k) or np.any(wave > self.maximum_k):
            raise ValueError("query k lies outside the sampled physical interval")

        position = (np.log(wave)-self.minimum)/self.step
        node = np.minimum(position.astype(np.intp), self.values.shape[-1]-2)
        fraction = position-node
        lower = self.values[..., node]
        result = lower+fraction*(self.values[..., node+1]-lower)
        if self.logarithmic:
            result = np.exp(result)
        return result
