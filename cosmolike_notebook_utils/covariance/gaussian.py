"""Gaussian blocks from shared angular spectra and spherical bin operators.

A covariance between AB and CD needs AC, BD, AD and BC even when those
cross-bin spectra are absent from the measured data vector. The routines
here always receive the complete field matrix. They never apply a survey's
pair-selection mask to those internal spectra.

Only the Gaussian fsky approximation is assembled here. A supplied mask
corrects the pure pair-noise term; it does not turn the signal part into
an exact cut-sky estimator covariance. The C components perform the sums.
"""

import numpy as np

from .geometry import cap_mask


def observed_spectra(spectra, ell, nlens):
    """Convert core source legs into the shear convention of spin operators.

    Arguments:
        spectra = float [nell,nfield,nfield] core spectra; lenses first.
        ell = float [nell] multipoles >= 2.
        nlens = count of leading scalar galaxy fields.
    Returns:
        owned float64 array of the same shape, with one spin-conversion
        factor per source leg. Input spectra are not modified.

    A spin-2 shear leg differs from the core convention by
    sqrt((ell-1)(ell+2)/(ell(ell+1))). White shape-noise powers already
    describe the measured shear and must not receive this factor.
    """
    values = np.asarray(a=spectra, dtype=float)
    modes = np.asarray(a=ell, dtype=float)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("spectra must have shape [nell,nfield,nfield]")
    if modes.shape != (values.shape[0],) or np.any(modes < 2):
        raise ValueError("ell must match the spectrum rows and be >= 2")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(modes)):
        raise ValueError("spectra and ell must be finite")
    if not isinstance(nlens, (int, np.integer)) or not 0 <= nlens <= values.shape[1]:
        raise ValueError("nlens must be an integer within the field count")

    factors = np.ones(shape=(len(modes), values.shape[1]), dtype=float)
    spin = np.sqrt((modes-1.0)*(modes+2.0)/(modes*(modes+1.0)))
    factors[:, nlens:] = spin[:, None]
    return np.ascontiguousarray(values*factors[:, :, None]*factors[:, None, :])


def gaussian_block(interface, spectra, noise, fields, left, right,
                   ell_min, area_sr, include_noise_noise=True):
    """Project one AB-by-CD Gaussian block with caller-supplied operators.

    Arguments:
        interface = initialized project's compiled interface.
        spectra = signal-only float [nell,nfield,nfield], observed convention.
        noise = float [nfield] independent white-noise powers, per steradian.
        fields = four integer IDs (A,B,C,D), zero based.
        left, right = float [nbin,nell] operators on consecutive integer ell.
        ell_min = first integer multipole, >= 0.
        area_sr = common survey area in steradians, in (0,4*pi].
        include_noise_noise = keep pure noise products for bandpowers;
            real-space callers instead add the analytic pair term below.
    Returns:
        owned [nleft,nright] covariance block. The C projection can compute
        arbitrary rectangular blocks without MPI calls or global work queues.
    """
    values = np.asarray(a=spectra, dtype=float)
    white = np.asarray(a=noise, dtype=float)
    identifiers = np.asarray(a=fields)
    if white.ndim != 1 or not np.all(np.isfinite(white)) or np.any(white < 0):
        raise ValueError("noise must be a finite nonnegative 1D array")
    if values.ndim != 3 or values.shape[1:] != (len(white), len(white)):
        raise ValueError("spectra must have shape [nell,len(noise),len(noise)]")
    if identifiers.shape != (4,) or identifiers.dtype.kind not in "iu":
        raise ValueError("fields must contain four integer IDs: A,B,C,D")
    if np.any(identifiers < 0) or np.any(identifiers >= len(white)):
        raise ValueError("fields contains an index outside the spectrum matrix")
    if not np.isfinite(area_sr) or not 0.0 < area_sr <= 4.0*np.pi:
        raise ValueError("area_sr must lie in (0,4*pi]")

    field_a, field_b, field_c, field_d = identifiers
    crossings = [
        (field_a, field_c),
        (field_b, field_d),
        (field_a, field_d),
        (field_b, field_c),
    ]
    cross_signal = np.empty(shape=(4, values.shape[0]), dtype=float)
    cross_noise = np.zeros(shape=4, dtype=float)

    # The two Wick contractions pair each left field with a right field.
    # Independent catalogs share noise only when their field IDs coincide.
    for row, (first, second) in enumerate(crossings):
        cross_signal[row] = values[:, first, second]
        if first == second:
            cross_noise[row] = white[first]

    harmonic = interface.covariance_gaussian_wick(
        cross_spectra=cross_signal,
        cross_noise=cross_noise,
        ell_min=ell_min,
        fsky=area_sr/(4.0*np.pi),
        include_noise_noise=include_noise_noise,
    )
    return interface.covariance_project(
        left=np.ascontiguousarray(left, dtype=float),
        right=np.ascontiguousarray(right, dtype=float),
        weight=harmonic,
    )


def realspace_block(interface, spectra, noise, fields, operators,
                    probe_left, probe_right, pair_area_sr2, area_sr):
    """Add exact pair noise to a Gaussian block on the same angular bins.

    Arguments:
        interface, spectra, noise, fields, area_sr = as in gaussian_block;
            spectra starts at ell=2 and ends at the operators' maximum ell.
        operators = [4,nbin,ell_max+1], from covariance_realspace_operator.
        probe_left, probe_right = 0 xi+, 1 xi-, 2 gamma_t, or 3 w(theta).
        pair_area_sr2 = positive [nbin] ordered-pair areas from the mask.
    Returns:
        [nbin,nbin] block for one common set of disjoint angular bins.

    A white-noise spectrum has an infinite multipole tail. Truncating it
    can underestimate small-angle variance, so the harmonic sum contains
    signal-signal and signal-noise only. The C pair-count formula supplies
    the full noise-noise contribution on matching bins. The two pieces
    therefore do not double count shape or shot noise.
    """
    kernels = np.asarray(a=operators, dtype=float)
    areas = np.asarray(a=pair_area_sr2, dtype=float)
    if kernels.ndim != 3 or kernels.shape[0] != 4:
        raise ValueError("operators must have shape [4,nbin,ell_max+1]")
    if (areas.shape != (kernels.shape[1],)
            or not np.all(np.isfinite(areas))
            or np.any(areas <= 0)):
        raise ValueError("pair_area_sr2 must contain one positive area per bin")
    for probe in (probe_left, probe_right):
        if not isinstance(probe, (int, np.integer)) or not 0 <= probe <= 3:
            raise ValueError("probe IDs must be integers from 0 to 3")

    result = gaussian_block(
        interface=interface,
        spectra=spectra,
        noise=noise,
        fields=fields,
        left=kernels[probe_left, :, 2:],
        right=kernels[probe_right, :, 2:],
        ell_min=2,
        area_sr=area_sr,
        include_noise_noise=False,
    )
    identifiers = np.ascontiguousarray(fields, dtype=np.int32)
    noise_ab = np.ascontiguousarray(
        [noise[identifiers[0]], noise[identifiers[1]]], dtype=float
    )
    for bin_index, pair_area in enumerate(areas):
        result[bin_index, bin_index] += interface.covariance_noise_pair(
            probe_left=probe_left,
            probe_right=probe_right,
            fields=identifiers,
            noise_ab=noise_ab,
            pair_area_sr2=float(pair_area),
        )
    return result


def shear_gaussian(interface, source, ell_max, area, edges_rad, a_edges,
                   radial_nquad, nwindow, angle_nquad, mask_ell_max, noise):
    """Assemble xi+/xi- Gaussian covariance for one source bin.

    Arguments:
        interface = initialized project interface (all survey bins installed).
        source = zero-based source bin index; ell_max = multipole cutoff.
        area = common footprint area in sr, modeled as a spherical cap.
        edges_rad = common disjoint angular bins, in radians.
        a_edges, radial_nquad, nwindow = covariance-owned radial controls.
        angle_nquad = Gaussian nodes per angular bin.
        mask_ell_max = separate cap-mask multipole cutoff.
        noise = [nlens+nsource] catalog noise powers, lenses first.
    Returns:
        dict with gaussian, signal_mixed, pure_noise [2*ntheta,2*ntheta],
        theta_rad centers, pair_area_sr2 and spectrum snapshot. Rows are
        xi+ bins followed by xi- bins; the cross block is retained.

    This example uses Limber spectra, zero IA and no RSD; it includes
    neither SSC nor cNG. It never changes data-vector accuracy controls.
    Refine radial, angular, signal-ell and mask-ell choices independently.
    """
    if not isinstance(source, (int, np.integer)) or source < 0:
        raise ValueError("source must be a nonnegative integer")
    ell = np.arange(start=2, stop=ell_max+1, dtype=float)
    edges = np.ascontiguousarray(edges_rad, dtype=float)
    snapshot = interface.covariance_limber_spectra(
        ell=ell,
        a_edges=np.ascontiguousarray(a_edges),
        nquad=radial_nquad,
        nwindow=nwindow,
        include_ia=False,
        include_rsd=False,
        linear=False,
    )
    if source >= snapshot["nsource"]:
        raise ValueError("source index exceeds the initialized source count")
    spectra = observed_spectra(
        spectra=snapshot["spectra"], ell=ell, nlens=snapshot["nlens"]
    )
    operators = interface.covariance_realspace_operator(
        edges_rad=edges, ell_max=ell_max, nquad=angle_nquad
    )
    mask = cap_mask(area_sr=area, ell_max=mask_ell_max)
    mask_operator = interface.covariance_realspace_operator(
        edges_rad=edges,
        ell_max=mask_ell_max,
        nquad=angle_nquad,
    )
    pair_area = interface.covariance_mask_pair_area(
        edges_rad=edges,
        mask_cl=mask,
        area_sr=area,
        scalar_kernel=np.ascontiguousarray(mask_operator[3]),
    )
    ntheta = len(edges)-1
    gaussian = np.empty(shape=(2*ntheta, 2*ntheta), dtype=float)
    mixed = np.empty_like(prototype=gaussian)
    field = snapshot["nlens"]+source
    fields = [field]*4

    # Both estimators observe the same shear field. Their operators differ,
    # so the xi+--xi- block must be computed as well as the two auto blocks.
    # Compute each unordered block once and copy its transpose exactly.
    for first in range(2):
        rows = slice(first*ntheta, (first+1)*ntheta)
        for second in range(first, 2):
            columns = slice(second*ntheta, (second+1)*ntheta)
            block = realspace_block(
                interface=interface, spectra=spectra, noise=noise,
                fields=fields, operators=operators,
                probe_left=first, probe_right=second,
                pair_area_sr2=pair_area, area_sr=area,
            )
            signal_mixed = gaussian_block(
                interface=interface, spectra=spectra, noise=noise,
                fields=fields,
                left=operators[first, :, 2:],
                right=operators[second, :, 2:],
                ell_min=2, area_sr=area, include_noise_noise=False,
            )
            gaussian[rows, columns] = block
            gaussian[columns, rows] = block.T
            mixed[rows, columns] = signal_mixed
            mixed[columns, rows] = signal_mixed.T
    return {
        "gaussian": gaussian,
        "signal_mixed": mixed,
        "pure_noise": gaussian-mixed,
        "theta_rad": np.sqrt(
            edges[:-1]*edges[1:]
        ),
        "pair_area_sr2": pair_area,
        "snapshot": snapshot,
    }
