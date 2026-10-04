"""Independent numerical oracles for covariance component tests.

Production workflows never import these modules. NumPy, SciPy and mpmath
implement separate quadratures or algebra so comparisons do not merely
repeat the C algorithm. Shared physical samples are identified explicitly.
"""
