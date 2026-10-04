"""Assemble a real-space cluster 6x2pt+N forecast under a limited halo model.

All measured families and every Gaussian/SSC cross block are retained.
The connected two-point term treats galaxies and clusters as linearly
biased tracers of the matter trispectrum. Count cross covariance contains
SSC only. Thus the output is a complete matrix under this approximation,
not a complete discrete-halo covariance: selected-cluster one-halo cNG and
non-SSC count-spectrum terms are omitted. These limits are returned with
the result and must accompany any exported forecast.

SSC additionally includes the abundance response of the selected halo's
own lensing profile, at fixed selection and profile. Catalog normalization
is subtracted once using the full projected signal. The same long-mode
shells correlate counts and all two-point functions.

See Krause & Eifler (2017), Appendix A, arXiv:1601.05779;
Takada & Hu (2013), corrected Eq. 44, arXiv:1302.6994; and the DES
covariance and localization conventions in arXiv:2503.13631, Sec. II.4.
The linearly biased cluster cNG is the approximation implemented in the
public DES lighthouse covariance, not a claim that all its terms are exact.
"""

import json
import time

import numpy as np

from .accuracy import non_gaussian_multipoles
from .counts_cluster import count_statistics
from .forecast import _json_array
from .gaussian import observed_spectra, limber_spectra
from .geometry import cap_mask, noise_powers
from .survey import compress_operators, project_connected, _matter_covariance_tables
from .survey_cluster import observable_layout, selected_windows, all_pairs_spectra
from .transform_cluster import localize_covariance


def _own_profile_response(interface, settings, geometry, catalogs, coarse_ell,
                          transform):
    """Project J11/n for the selected halo's fixed-profile abundance response.

    J11 integrates dn*S*b*(M/rho)*u. Divide by the reference selected
    abundance n, holding that denominator fixed here. The response of the
    observed catalog denominator is a separate angular subtraction in
    compute_forecast. Including it here as well would count it twice.

    Arguments:
        interface = initialized fixed-selection, massless cluster model.
        settings = cluster_lnm_bounds in ln(M/[Msun/h]), halo_mass_nquad.
        geometry = [4,nstate] a,chi,f_K,dchi in the core length unit c/H0.
        catalogs = selected_windows result on these same states.
        coarse_ell = [nk] angular modes; transform = [nbin,nk] compressed
            tangential-shear operators, with their source-leg factor.
    Returns:
        [nrichness,nbin,nstate] angularly transformed J11/n in (c/H0)^3.
        It contains no cluster radial window or observed-mean subtraction.
    """
    lower, upper = settings['cluster_lnm_bounds']
    # Two mass panels resolve the selected interval without requesting a
    # generated GSL rule. Both use the precomputed rule selected by the
    # independent integration level, including its 64-node minimum.
    node, measure = interface.covariance_integration_rule(
        nquad=settings['halo_mass_nquad'],
    )
    midpoint = 0.5*(lower+upper)
    half_width = 0.25*(upper-lower)
    centers = np.array([0.5*(lower+midpoint), 0.5*(midpoint+upper)])
    lnm = (centers[:, None]+half_width*node).ravel()
    dlnm = np.tile(half_width*measure, reps=2)
    nstate = geometry.shape[1]
    nrichness = catalogs['nrichness']
    response = np.zeros((nrichness, nstate, len(coarse_ell)))
    active = np.flatnonzero(np.any(catalogs['density'] > 0.0, axis=0))

    # The selected population exists in only part of the source interval.
    # Process 16 active shells together: C gets many independent mass sums,
    # while high boosts do not allocate profiles for every shell at once.
    # Batch size affects storage, not quadrature or floating-point sum order.
    for start in range(0, len(active), 16):
        states = active[start:start+16]
        wave = (coarse_ell[None, :]+0.5)/geometry[2, states, None]
        samples = interface.covariance_cluster_halo_samples(
            a=np.ascontiguousarray(geometry[0, states]),
            k=np.ascontiguousarray(wave), lnm=lnm, dlnm=dlnm,
        )
        moments = interface.covariance_cluster_moments(**samples)
        number = moments['density'][0]
        if np.any(number <= 0.0):
            raise ValueError(
                "selected mass rule has an empty bin; check cluster_lnm_bounds"
            )
        own = moments['single'][1]/number[:, :, None]
        response[:, states] = own.transpose(1, 0, 2)

    # All richness bins share the same angular operator and k samples.
    # One C contraction integrates every (richness,state) profile row.
    right = np.ascontiguousarray(response.reshape(nrichness*nstate, -1))
    projected = interface.covariance_project(
        left=transform, right=right, weight=np.ones(len(coarse_ell)),
    )
    return projected.reshape(len(transform), nrichness, nstate).transpose(1, 0, 2)


def compute_forecast(interface, settings, progress=None):
    """Return the joint angular forecast with separate G, SSC and cNG.

    Arguments:
        interface = caller's initialized galaxy/source/cluster interface.
            Use massless neutrinos, linear galaxy bias, zero IA, RSD and
            magnification, lognormal richness selection, selection_model=0
            and abundance-weighted cluster windows (kernel_mode=1).
        settings = resolved galaxy forecast configuration plus cg_lens_bin
            [ncluster_z], cluster_lnm_bounds [2], and cluster_ytransform
            bool. The latter applies the mean model's exact Y operator to
            every cluster-lensing row and both covariance axes. Numerical
            controls come from the common covariance accuracy boost.
        progress = optional callable receiving stage and elapsed seconds.
    Returns:
        Dict with owned gaussian, ssc, cng, total [ndata,ndata]; signal
        [nrow,nbin]; mean_counts [ncount]; joint_signal [ndata]; layout
        positions; geometry; coarse_ell; pair_area_sr2; coordinates;
        resolved settings and elapsed stages. ndata=nrow*nbin+ncount.
        G includes count Poisson noise. cNG has zero count rows under the
        stated approximation. valid_indices excludes only the known zero
        last Y bin, not the project's physical scale cuts. No eigenvalue
        correction or file write is performed. Initialization is separate.
    """
    resolved = dict(settings)
    resolved['mnu'] = settings['cosmology']['mnu']
    resolved['area_sr'] = settings['area_deg2']*(np.pi/180.0)**2
    resolved['edges_rad'] = np.asarray(settings['theta_edges_arcmin'])*np.pi/10800.0
    resolved['space'] = 'real'
    resolved['cluster_cng_model'] = 'linear tracer biases times matter trispectrum'
    resolved['count_cross_model'] = 'SSC only'
    resolved['omitted_terms'] = [
        'selected-cluster one-halo cNG corrections',
        'non-SSC count-spectrum cross covariance',
        'all-pairs non-Limber corrections',
        'tidal, nonlinear-bias and environmental-selection responses',
    ]
    if resolved['mnu'] != 0.0:
        raise ValueError("cluster halo forecast requires mnu=0")
    # Counts and selected profiles also read shared core tables. Refine
    # those tables with the global boost, retaining the independent core
    # quadrature level selected by integration_accuracy.
    interface.init_accuracy_boost(
        accuracy_boost=settings['core_accuracyboost'],
        integration_accuracy=settings['integration_accuracy'],
    )
    mass_bounds = np.asarray(settings['cluster_lnm_bounds'], dtype=float)
    if (mass_bounds.shape != (2,) or not np.all(np.isfinite(mass_bounds))
            or mass_bounds[1] <= mass_bounds[0]):
        raise ValueError("cluster_lnm_bounds needs two increasing finite log masses")
    coarse_ell = non_gaussian_multipoles(samples=settings['ng_ell'],
                                        ell_max=settings['ell_max'])
    if not isinstance(settings['cluster_ytransform'], (bool, np.bool_)):
        raise ValueError("cluster_ytransform must be True or False")
    operator = None
    if settings['cluster_ytransform']:
        operator = np.ascontiguousarray(interface.get_cluster_ytransform_matrix())
        nbin = len(resolved['edges_rad'])-1
        if operator.shape != (nbin, nbin):
            raise ValueError(
                "Y operator shape differs from bins; initialize matching binning"
            )
    started = time.perf_counter()
    stages = {}

    def checkpoint(name, since):
        """Record stages without treating an ordinary run as a benchmark."""
        stages[name] = time.perf_counter()-since
        if progress is not None:
            progress(name, time.perf_counter()-started)

    # --- 1. One radial rule for every catalog and every internal spectrum ---
    # The covariance of AB and CD needs AC, BD, AD and BC, even when those
    # spectra are excluded from the measured vector. Keep all field pairs.
    ell = np.arange(2, settings['ell_max']+1, dtype=float)
    snapshot = limber_spectra(
        interface=interface,
        ell=ell, a_edges=settings['a_edges'], nquad=settings['radial_nquad'],
        nwindow=settings['nwindow'], include_ia=False, include_rsd=False,
        linear=False,
    )
    geometry = snapshot['geometry']
    base = snapshot['windows']
    nlens = snapshot['nlens']
    nsource = base.shape[1]-nlens
    if np.any(base[1, :nlens] != 0.0):
        raise ValueError("cluster forecast requires zero galaxy magnification")
    catalogs = selected_windows(interface=interface, geometry=geometry)
    ncount = len(catalogs['number_per_sr'])
    layout = observable_layout(
        nlens=nlens, nsource=nsource, ncluster_z=catalogs['ncluster_z'],
        nrichness=catalogs['nrichness'], cg_lens_bin=settings['cg_lens_bin'],
        nbin=len(resolved['edges_rad'])-1,
        excluded_gammat=settings['excluded_gammat'],
    )
    rows = layout['rows']
    spectra = all_pairs_spectra(interface=interface, ell=ell, snapshot=snapshot,
                                catalogs=catalogs)
    signal = observed_spectra(spectra=spectra, ell=ell, nlens=nlens+ncount)
    ordinary_noise = noise_powers(
        lens_density=settings['lens_density_arcmin2'],
        source_density=settings['source_density_arcmin2'],
        sigma_component=settings['sigma_e_component'],
    )
    noise = np.concatenate((ordinary_noise[:nlens], 1.0/catalogs['number_per_sr'],
                             ordinary_noise[nlens:]))
    checkpoint('all_pairs_limber_spectra', started)

    # --- 2. Bin-averaged full-sky angular transforms and Gaussian covariance ---
    # Pure white noise is integrated analytically through mask pair areas.
    # Signal terms retain every integer multipole below the chosen cutoff.
    tick = time.perf_counter()
    kernels = interface.covariance_realspace_operator(
        edges_rad=resolved['edges_rad'], ell_max=settings['ell_max'],
        nquad=settings['angle_nquad'],
    )[:, :, 2:]
    kernels = np.ascontiguousarray(kernels)
    nbin = kernels.shape[1]
    mask = cap_mask(area_sr=resolved['area_sr'], ell_max=settings['mask_ell_max'])
    mask_operator = interface.covariance_realspace_operator(
        edges_rad=resolved['edges_rad'], ell_max=settings['mask_ell_max'],
        nquad=settings['angle_nquad'],
    )
    pair_area = interface.covariance_mask_pair_area(
        mask_cl=mask, area_sr=resolved['area_sr'], edges_rad=resolved['edges_rad'],
        scalar_kernel=np.ascontiguousarray(mask_operator[3]),
    )
    gaussian = interface.covariance_gaussian_real(
        spectra=signal, noise=noise, rows=rows, operators=kernels,
        ell_min=2, area_sr=resolved['area_sr'], pair_area_sr2=pair_area,
    )
    means = np.empty((len(rows), nbin))
    for probe in range(4):
        selected = np.flatnonzero(rows[:, 0] == probe)
        fields = rows[selected, 1:]
        data = np.ascontiguousarray(signal[:, fields[:, 0], fields[:, 1]].T)
        means[selected] = interface.covariance_project(
            left=data, right=kernels[probe], weight=np.ones(len(ell)),
        )
    del spectra, signal, snapshot, mask_operator
    checkpoint('gaussian_and_mean', tick)

    # --- 3. Shared matter cNG and SSC, plus the selected own-profile response ---
    # The expensive matter calculation is independent of catalog labels.
    # Reuse the exact galaxy/shear pipeline, then attach cluster windows.
    tick = time.perf_counter()
    compressed = compress_operators(operators=kernels, ell=ell, coarse_ell=coarse_ell)

    def report_matter(completed, total):
        """Report the common halo-table progress in whole-forecast seconds."""
        if progress is not None:
            progress(f'Matter shell {completed}/{total}', time.perf_counter()-started)

    matter = _matter_covariance_tables(
        interface=interface, settings=resolved, geometry=geometry,
        coarse_ell=coarse_ell, transform=compressed.reshape(4*nbin, -1),
        mask_nell=len(mask), progress=report_matter,
    )
    own = _own_profile_response(
        interface=interface, settings=resolved, geometry=geometry, catalogs=catalogs,
        coarse_ell=coarse_ell, transform=np.ascontiguousarray(compressed[2]),
    )
    checkpoint('shared_halo_tables', tick)

    # --- 4. Normalize measured catalogs and correlate all responses together ---
    # A long fluctuation changes both the local spectrum and the observed
    # number used to normalize a density contrast. Subtract the latter as
    # (U_A+U_B)*C_AB, where C_AB is the full angular mean, once per density
    # leg. Counts are absolute numbers and receive no such subtraction.
    tick = time.perf_counter()
    distance = geometry[2]
    dchi = geometry[3]
    nstate = len(distance)
    windows = np.concatenate((base[0, :nlens], catalogs['window']*catalogs['bias'],
                               base[1, nlens:]))
    mean_window = windows.copy()
    mean_window[nlens+ncount:] = 0.0
    pair = windows[rows[:, 1]]*windows[rows[:, 2]]
    projected_response = matter['response'].reshape(4, nbin, nstate)
    shell = pair[:, None, :]*projected_response[rows[:, 0]]/distance**2
    for row, (probe, first, second) in enumerate(rows):
        if probe == 2 and nlens <= first < nlens+ncount:
            category = first-nlens
            source = second-ncount
            local_window = catalogs['window'][category]*base[1, source]/distance**2
            shell[row] += local_window*own[category % catalogs['nrichness']]
    shell -= ((mean_window[rows[:, 1]]+mean_window[rows[:, 2]])[:, None, :]
              *means[:, :, None])
    shell = np.ascontiguousarray(shell.reshape(len(rows)*nbin, nstate))
    variance = interface.covariance_ssc_mask_variance(
        mask_cl=mask, area_sr=resolved['area_sr'],
        distance=np.ascontiguousarray(distance), power=matter['long_power'],
    )
    counts = count_statistics(
        interface=interface, distance=distance, dchi=dchi,
        density=catalogs['density'], derivative=catalogs['derivative'],
        area_sr=resolved['area_sr'], background_variance=variance,
    )
    two_point = layout['two_point_positions']
    count_positions = layout['count_positions']
    ndata = len(two_point)+len(count_positions)
    response = np.empty((ndata, nstate))
    response[two_point] = shell
    response[count_positions] = counts['shell_response']
    ssc = interface.covariance_project(
        left=response, right=response, weight=dchi*variance,
    )
    ssc = np.triu(ssc)+np.triu(ssc, k=1).T

    # In the biased-tracer cNG approximation the matter trispectrum has
    # one linear bias per density leg. Keep every crossed radial window;
    # count rows are zero ONLY for this separately stated cNG approximation.
    connected = project_connected(
        interface=interface, rows=rows, pair_window=pair,
        projected=matter['projected'], measure=dchi/(resolved['area_sr']*distance**6),
    )
    joint_gaussian = np.zeros((ndata, ndata))
    joint_gaussian[np.ix_(two_point, two_point)] = gaussian
    joint_gaussian[np.ix_(count_positions, count_positions)] = counts['poisson']
    joint_cng = np.zeros((ndata, ndata))
    joint_cng[np.ix_(two_point, two_point)] = connected
    components = {
        'gaussian': joint_gaussian,
        'ssc': ssc,
        'cng': joint_cng,
    }
    joint_signal = np.empty(ndata)
    joint_signal[two_point] = means.ravel()
    joint_signal[count_positions] = counts['mean']
    checkpoint('joint_ssc_and_connected', tick)

    # --- 5. Apply the same localization as the mean before scale selection ---
    # Y needs neighboring unmasked angular bins. Its final bin is exactly
    # zero by definition. Keep that null row in the returned full layout;
    # report its complement for positivity checks without clipping modes.
    tick = time.perf_counter()
    valid = np.arange(ndata)
    if settings['cluster_ytransform']:
        positions = layout['cluster_lensing_positions']
        for name, matrix in components.items():
            transformed = localize_covariance(
                interface=interface, covariance=matrix, indices=positions,
                operator=operator,
            )
            # A*C*A^T is symmetric. Its two multiplication orders can
            # differ by roundoff across the diagonal; retain one computed
            # triangle, as in the ordinary SSC/cNG assembly. This neither
            # changes a diagonal variance nor repairs a negative mode.
            components[name] = np.triu(transformed)+np.triu(transformed, k=1).T
        localized_mean = interface.covariance_project(
            left=np.ascontiguousarray(joint_signal[positions]), right=operator,
            weight=np.ones(nbin),
        )
        joint_signal[positions] = localized_mean
        means = joint_signal[two_point].reshape(len(rows), nbin)
        valid = np.delete(arr=valid, obj=positions[:, -1])
    components['total'] = components['gaussian']+components['ssc']+components['cng']
    checkpoint('localization', tick)
    checkpoint('total', started)
    edges = settings['theta_edges_arcmin']
    components.update(layout)
    components.update({
        'signal': means,
        'mean_counts': counts['mean'],
        'joint_signal': joint_signal,
        'valid_indices': valid,
        'geometry': geometry,
        'coarse_ell': coarse_ell,
        'pair_area_sr2': pair_area,
        'coordinate': np.sqrt(edges[:-1]*edges[1:]),
        'coordinate_label': r'$\theta\;[\mathrm{arcmin}]$',
        'settings': resolved,
        'stages_s': stages,
    })
    return components


def save_forecast(result, filename):
    """Save joint arrays, row positions and explicit model limits together.

    Arguments:
        result = compute_forecast output, including its resolved settings.
        filename = destination .npz path outside the likelihood data folder.
    Returns:
        Nothing. Replaces the named file if present. Numerical arrays,
        including count positions and Y null-mode selection, load without
        pickle. settings_json and stages_json are JSON text scalars.
        Archive the initialization's CAMB tables separately for exact reuse.
    """
    arrays = {}
    for name, value in result.items():
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError(f"{name} contains Python objects; save numerical arrays")
            arrays[name] = value
        elif isinstance(value, str):
            arrays[name] = value
    arrays['settings_json'] = json.dumps(
        result['settings'], default=_json_array, allow_nan=False,
    )
    arrays['stages_json'] = json.dumps(result['stages_s'], allow_nan=False)
    np.savez(file=filename, **arrays)
