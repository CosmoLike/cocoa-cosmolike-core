"""Full real-space 3x2pt matrices under an explicit Limber halo model.

The Gaussian, super-sample and connected terms follow the decomposition
in Krause & Eifler (2017), Appendix A, arXiv:1601.05779. SSC uses the
isotropic halo response of Takada & Hu (2013), corrected Eq. 44 and
Appendix A, arXiv:1302.6994. The mask is a spherical cap; long and short
modes both use Limber. This is not a calibrated nonlinear tidal response.

All cross-bin blocks are retained. The first survey assembly supports
massless neutrinos, linear galaxy bias, zero magnification and zero IA.
It returns a forecast, not a replacement for a project's supplied matrix.
Numerical refinement and Fisher convergence remain separate checks.
"""

import time

import numpy as np

from .gaussian import observed_spectra
from .geometry import angular_rule, cap_mask


def observable_rows(nlens, nsource, excluded_gammat=()):
    """Return (probe,A,B) rows in Cocoa's xi+,xi-,gamma_t,w order.

    Field IDs put lenses before sources. Exclusions select measured
    gamma_t rows only; internal crossed spectra keep every field pair.
    Each returned row contains all angular bins consecutively.

    Arguments:
        nlens, nsource = positive counts of lens/source redshift bins.
        excluded_gammat = zero-based (lens,source) pairs absent from the
            measured vector. These do not exclude internal field spectra.
    Returns:
        int32 [nobservable,3] array, columns (probe,A,B). Probe IDs are
        0 xi+, 1 xi-, 2 gamma_t and 3 w(theta).
    """
    if nlens < 1 or nsource < 1:
        raise ValueError("the 3x2pt layout requires lens and source bins")
    excluded = set()
    for lens, source in excluded_gammat:
        if not 0 <= lens < nlens or not 0 <= source < nsource:
            raise ValueError("excluded gamma_t pair is outside the bin range")
        excluded.add((lens, source))
    rows = []
    for probe in (0, 1):
        for first in range(nsource):
            for second in range(first, nsource):
                rows.append((probe, nlens+first, nlens+second))
    for lens in range(nlens):
        for source in range(nsource):
            if (lens, source) not in excluded:
                rows.append((2, lens, nlens+source))
    for lens in range(nlens):
        rows.append((3, lens, lens))
    return np.array(rows, dtype=np.int32)


def compress_operators(operators, ell, coarse_ell):
    """Project linear interpolation weights instead of a dense trispectrum.

    If T(l,l') is bilinear in x=ln(l+1/2), write
    T(l,l') = sum_ij h_i(x) T_ij h_j(x'). Then K T K^T equals B T B^T,
    where B_i=sum_l K_l h_i(x_l). The two nonzero h values are the
    ordinary lower/upper linear weights. This works because interpolation
    and projection are both linear sums: their order can be exchanged.
    No oscillatory theta kernel is
    sampled sparsely: every integer multipole contributes to B.

    operators is [4,ntheta,nell] and contains the observed spin kernels.
    Attach the Limber shear factor (l-1)(l+2)/(l+1/2)^2 per source leg.
    The coarse trispectrum stays signed. Accuracy depends on refining
    coarse_ell; this algebra does not claim that a particular grid suffices.

    Arguments:
        operators = float [4,ntheta,nell] bin-averaged angular kernels.
        ell = increasing [nell] integer multipoles, beginning at two.
        coarse_ell = [ncoarse] samples uniform in ln(ell+1/2), covering ell.
    Returns:
        float [4,ntheta,ncoarse] compressed operators. Multipoles, kernels
        and these operators are dimensionless.
    """
    grid = np.log(np.asarray(coarse_ell)+0.5)
    query = np.log(np.asarray(ell)+0.5)
    step = (grid[-1]-grid[0])/(len(grid)-1)
    if not np.allclose(np.diff(grid), step, rtol=1.e-12, atol=0):
        raise ValueError("coarse_ell must be uniform in ln(ell+1/2)")
    if query[0] < grid[0] or query[-1] > grid[-1]:
        raise ValueError("the coarse grid must cover every integer multipole")
    position = (query-grid[0])/step
    left = np.minimum(position.astype(np.intp), len(grid)-2)
    fraction = position-left
    spin = (ell-1.0)*(ell+2.0)/(ell+0.5)**2
    result = np.empty((4, operators.shape[1], len(grid)))

    # A source leg carries one shear factor; galaxy density carries none.
    # Accumulate both interpolation endpoints with their signed kernels.
    for probe, source_legs in enumerate((2, 2, 1, 0)):
        for angular_bin, kernel in enumerate(operators[probe]):
            values = kernel*spin**source_legs
            lower = np.bincount(left, weights=values*(1.0-fraction),
                                minlength=len(grid))
            upper = np.bincount(left+1, weights=values*fraction,
                                minlength=len(grid))
            result[probe, angular_bin] = lower+upper
    return result


def project_connected(interface, rows, pair_window, projected, measure):
    """Integrate all tomographic pairs after the shared angular transform.

    projected is [4*ntheta,4*ntheta,nradial], the angularly transformed
    matter trispectrum. Each covariance block weights it by W_A W_B W_C
    W_D dchi/(area*f_K^6). The C contraction integrates all bin-pair
    combinations for one pair of angular bins together, including crossed
    lens families. Symmetric entries are copied from one computation.

    Arguments:
        interface = compiled project interface exposing covariance_project.
        rows = int [nobservable,3] from observable_rows.
        pair_window = float [nobservable,nradial], each W_A W_B, in
            (c/H0)^-2; this already includes the linear lens biases.
        projected = float [4*ntheta,4*ntheta,nradial], in (c/H0)^9.
        measure = float [nradial], dchi/(area*f_K^6), in (c/H0)^-5.
    Returns:
        Dimensionless symmetric [nobservable*ntheta,nobservable*ntheta]
        cNG covariance, retaining the supplied signed contributions.
    """
    ntheta = projected.shape[0]//4
    result = np.empty((len(rows)*ntheta, len(rows)*ntheta))
    groups = []
    for probe in range(4):
        groups.append(np.flatnonzero(rows[:, 0] == probe))
    # At a fixed pair of angular bins, every catalog pair uses the same
    # projected matter trispectrum. Only its two radial windows differ.
    # Grouping catalog pairs lets C integrate the entire rectangular block
    # with one shared radial weight and enough outputs for eight workers.
    for left_probe in range(4):
        left_rows = groups[left_probe]
        left = np.ascontiguousarray(pair_window[left_rows])
        for right_probe in range(left_probe, 4):
            right_rows = groups[right_probe]
            right = np.ascontiguousarray(pair_window[right_rows])
            for first in range(ntheta):
                start = first if left_probe == right_probe else 0
                for second in range(start, ntheta):
                    weight = measure*projected[
                        left_probe*ntheta+first, right_probe*ntheta+second
                    ]
                    block = interface.covariance_project(
                        left=left, right=right, weight=weight
                    )
                    if left_probe == right_probe and first == second:
                        block = np.triu(block)+np.triu(block, k=1).T
                    i = left_rows*ntheta+first
                    j = right_rows*ntheta+second
                    result[np.ix_(i, j)] = block
                    result[np.ix_(j, i)] = block.T
    return result


def realspace_covariance(interface, settings, rows, noise, progress=None):
    """Compute separate G, SSC and cNG matrices and return elapsed stages.

    Arguments:
        interface = project interface with cosmology and catalogs initialized.
        settings = resolved integration and physical choices:
            ell_max, mask_ell_max = signal and footprint multipole cutoffs;
            ng_ell_nodes = count of samples uniform in ln(ell+1/2);
            a_edges, radial_nquad = scale-factor panels and GL nodes/panel;
            nwindow = uniform-a samples for cumulative lensing efficiencies;
            angle_nquad = GL nodes within each observed angular bin;
            lnm_edges, halo_mass_nquad = ln(M/[Msun/h]) panels and GL rule;
            tree_nquad, tree_npanel = relative-wavevector angular rule;
            response_step = centered finite-difference step in ln(k);
            area_sr, edges_rad = common footprint area and angular-bin edges;
            mnu = initialized neutrino mass in eV, currently required zero.
        rows = int32 [nobservable,3] from observable_rows(...).
        noise = float [nlens+nsource] white powers from noise_powers(...).
        progress = optional callable receiving a stage name and elapsed seconds.

    Returns:
        dict with gaussian, ssc, cng and total [ndata,ndata] dimensionless
        matrices; signal [nobservable,ntheta]; rows; coarse_ell; radial
        geometry [4,nradial] (a,chi,f_K,dchi); pair_area_sr2 [ntheta]; and
        stages_s elapsed times. Here ndata=nobservable*ntheta; distances
        have units c/H0. The function writes no files and repairs no modes.

    Returned arrays use row-major observable ordering with theta innermost.
    Times include Python preparation, core workspaces and cold table builds;
    CAMB/initialization, diagnostic eigenproblems and file writing are outside
    this function. No shared table is recomputed inside a tomographic block.
    """
    if settings["mnu"] != 0.0:
        raise ValueError("the full halo matter model currently requires mnu=0")
    if settings["ng_ell_nodes"] < 2 or settings["ell_max"] < 2:
        raise ValueError("need at least two NG samples and ell_max >= 2")
    if not np.isfinite(settings["response_step"]) or settings["response_step"] <= 0:
        raise ValueError("response_step must be finite and positive")
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype.kind not in "iu":
        raise ValueError("rows must contain integer (probe,A,B) triplets")
    if np.any(rows < 0) or np.any(rows[:, 0] > 3):
        raise ValueError("field IDs must be nonnegative and probe IDs in 0..3")
    started = time.perf_counter()
    stages = {}

    def checkpoint(name, since):
        """Record a completed stage and report time since assembly began."""
        stages[name] = time.perf_counter()-since
        if progress is not None:
            progress(name, time.perf_counter()-started)

    # --- 1. Common radial windows and all crossed angular spectra ---
    # Gaussian Wick products need field pairs that never appear as measured
    # observables. Building one complete snapshot keeps these correlations
    # consistent across every covariance block.
    ell = np.arange(2, settings["ell_max"]+1, dtype=float)
    snapshot = interface.covariance_limber_spectra(
        ell=ell, a_edges=settings["a_edges"],
        nquad=settings["radial_nquad"], nwindow=settings["nwindow"],
        include_ia=False, include_rsd=False, linear=False,
    )
    geometry = snapshot["geometry"]
    base = snapshot["windows"]
    nlens = snapshot["nlens"]
    if np.any(base[1, :nlens] != 0.0):
        raise ValueError("this separable survey assembly needs zero magnification")
    if len(noise) != base.shape[1]:
        raise ValueError("noise count must match the initialized field count")
    if np.any(rows[:, 1:] >= base.shape[1]):
        raise ValueError("an observable field ID exceeds the initialized count")
    signal = observed_spectra(snapshot["spectra"], ell=ell, nlens=nlens)
    checkpoint("all_pairs_limber_spectra", started)

    # --- 2. Full-sky angular bins and the footprint's available pairs ---
    # White noise extends to arbitrarily high ell. Its exact pair-count
    # term is added later; only signal and mixed terms use the finite sum.
    tick = time.perf_counter()
    operators = interface.covariance_realspace_operator(
        edges_rad=settings["edges_rad"], ell_max=settings["ell_max"],
        nquad=settings["angle_nquad"],
    )
    mask = cap_mask(area_sr=settings["area_sr"],
                    ell_max=settings["mask_ell_max"])
    mask_operator = interface.covariance_realspace_operator(
        edges_rad=settings["edges_rad"], ell_max=settings["mask_ell_max"],
        nquad=settings["angle_nquad"],
    )
    pair_area = interface.covariance_mask_pair_area(
        edges_rad=settings["edges_rad"], mask_cl=mask,
        area_sr=settings["area_sr"],
        scalar_kernel=np.ascontiguousarray(mask_operator[3]),
    )
    kernels = np.ascontiguousarray(operators[:, :, 2:])
    ntheta = kernels.shape[1]
    ndata = len(rows)*ntheta
    checkpoint("angular_and_mask_geometry", tick)

    # --- 3. Gaussian covariance and the observable mean signals ---
    # A block covariance needs crossed spectra, including lens cross bins
    # absent from the data vector. The snapshot retains that full matrix.
    tick = time.perf_counter()
    gaussian = interface.covariance_gaussian_real(
        spectra=signal, noise=np.ascontiguousarray(noise),
        rows=np.ascontiguousarray(rows, dtype=np.int32), operators=kernels,
        ell_min=2, area_sr=settings["area_sr"], pair_area_sr2=pair_area,
    )
    checkpoint("gaussian_blocks", tick)

    # The SSC mean subtraction uses the full angular signal of each
    # measured observable, rather than its local radial integrand.
    tick = time.perf_counter()
    observable_signal = np.empty((len(rows), ntheta))
    for probe in range(4):
        selected = np.flatnonzero(rows[:, 0] == probe)
        fields = rows[selected, 1:]
        spectra = np.ascontiguousarray(signal[:, fields[:, 0], fields[:, 1]].T)
        observable_signal[selected] = interface.covariance_project(
            left=spectra, right=kernels[probe], weight=np.ones(len(ell))
        )

    checkpoint("observable_signal", tick)

    # --- 4. Shared matter trispectrum and response at each radial shell ---
    # The matter calculation is independent of the observed catalog pair.
    # Project its angular dependence once per shell; catalog windows enter
    # only in the final radial sums. This avoids repeating expensive halo
    # integrations for thousands of observable pairs.
    tick = time.perf_counter()
    coarse_ell = np.exp(np.linspace(np.log(2.5), np.log(ell[-1]+0.5),
                                   settings["ng_ell_nodes"]))-0.5
    coarse_ell[0] = ell[0]
    coarse_ell[-1] = ell[-1]
    compressed = compress_operators(kernels, ell=ell, coarse_ell=coarse_ell)
    transform = compressed.reshape(4*ntheta, -1)
    nnode = geometry.shape[1]
    projected = np.empty((4*ntheta, 4*ntheta, nnode))
    response = np.empty((4*ntheta, nnode))
    long_power = np.empty((nnode, len(mask)))
    unused, angle_weight, corner = angular_rule(
        nquad=settings["tree_nquad"], npanel=settings["tree_npanel"]
    )
    first, second = np.triu_indices(settings["ng_ell_nodes"])
    # K,Q both scale as 1/chi along a Limber shell. Build the angular
    # geometry once in multipole units, then rescale at each radial node.
    modes = coarse_ell+0.5
    magnitude = np.sqrt((modes[first, None]-modes[second, None])**2
                        +2*modes[first, None]*modes[second, None]*corner)
    unit_weight = np.ones(len(coarse_ell))
    shift = np.exp(np.array([-settings["response_step"], 0.0,
                             settings["response_step"]]))

    # A given multipole probes a different physical k at each distance.
    # Each shell therefore evaluates its own halo moments and tree terms.
    # The two angular projections contract T(k,k') with the pre-summed
    # interpolation weights; the final stored object no longer has k axes.
    for node, a in enumerate(geometry[0]):
        distance = geometry[2, node]
        k = modes/distance
        single, moments = interface.covariance_halo_moments(
            a=np.array([a]), k=k[None, :], lnm_edges=settings["lnm_edges"],
            nquad=settings["halo_mass_nquad"],
        )

        # Tree-level terms couple the two external wavevectors through
        # P(|K+Q|). Reuse the common relative-angle geometry from above,
        # changing only its physical length scale at this distance.
        linear = interface.covariance_power(a=a, k=k, linear=True)
        pk = np.array([linear[first], linear[second]])
        internal = interface.covariance_power(
            a=a, k=magnitude/distance, linear=True
        )
        angular = interface.covariance_tree_averages(
            k=np.array([k[first], k[second]]), pk=pk, corner=corner,
            weight=angle_weight, ps=internal,
        )
        terms = interface.covariance_halo_trispectrum(
            pk=pk, i11=np.array([single[0, first], single[0, second]]),
            moments=np.ascontiguousarray(moments[:, 0]), tree=angular,
        )

        # Both axes use the same matter field, so T(K,Q)=T(Q,K).
        # Expand the triangular table, then contract one axis at a time.
        trispectrum = np.empty((len(k), len(k)))
        total = np.sum(terms, axis=0)
        trispectrum[first, second] = total
        trispectrum[second, first] = total
        half = interface.covariance_project(
            left=transform, right=trispectrum, weight=unit_weight
        )
        projected[:, :, node] = interface.covariance_project(
            left=half, right=transform, weight=unit_weight
        )

        # Only I11 needs shifted k for the logarithmic two-halo slope.
        # Keep the independently integrated central moments for I02/I12.
        shifted_k = k[:, None]*shift
        shifted_single, unused = interface.covariance_halo_moments(
            a=np.full(len(k), a), k=shifted_k,
            lnm_edges=settings["lnm_edges"],
            nquad=settings["halo_mass_nquad"],
        )
        shifted_power = interface.covariance_power(a=a, k=shifted_k, linear=True)
        two_halo = shifted_single**2*shifted_power
        slope = np.log(two_halo[:, 2]/two_halo[:, 0])/(2*settings["response_step"])
        diagonal = np.flatnonzero(first == second)

        # The halo model predicts the fractional response to background
        # density. Transfer that fraction to the chosen nonlinear power;
        # this defines the stated SSC approximation, not a tidal response.
        target = interface.covariance_power(a=a, k=k, linear=False)
        dimensional = interface.covariance_halo_response(
            inputs=np.array([linear, target, single[0], moments[0, 0, diagonal],
                             moments[1, 0, diagonal], slope]),
            growth_coefficient=47.0/21.0, dilation_coefficient=1.0/3.0,
            fractional=True,
        )[1]
        response[:, node] = interface.covariance_project(
            left=transform, right=dimensional[None, :], weight=unit_weight
        )[:, 0]

        # The footprint weights much longer wavelengths than the measured
        # angular bins. Its background variance uses linear matter power.
        long_power[node] = interface.covariance_power(
            a=a, k=(np.arange(len(mask))+0.5)/distance, linear=True
        )
        if progress is not None and node % 32 == 0:
            progress(f"Matter shell {node+1}/{nnode}", time.perf_counter()-started)
    checkpoint("shared_halo_response_and_trispectrum", tick)

    # --- 5. Catalog weights, survey-mean response, and all radial blocks ---
    # Lenses contribute b*n and sources contribute lensing efficiency.
    # Normalizing galaxy density by its observed survey mean subtracts
    # (U_A+U_B)*C_AB from its response to a background density fluctuation.
    # Shear has no galaxy-count normalization, so its U is zero.
    tick = time.perf_counter()
    windows = np.concatenate((base[0, :nlens], base[1, nlens:]))
    pair_window = windows[rows[:, 1]]*windows[rows[:, 2]]
    mean = np.zeros_like(windows)
    mean[:nlens] = base[0, :nlens]
    pair_mean = mean[rows[:, 1]]+mean[rows[:, 2]]
    response = response.reshape(4, ntheta, nnode)
    shell = pair_window[:, None, :]*response[rows[:, 0]]/geometry[2]**2
    shell -= pair_mean[:, None, :]*observable_signal[:, :, None]
    shell = shell.reshape(ndata, nnode)
    variance = interface.covariance_ssc_mask_variance(
        mask_cl=mask, area_sr=settings["area_sr"],
        distance=np.ascontiguousarray(geometry[2]), power=long_power,
    )

    # Shared responses on both sides with positive radial weights form a
    # weighted outer product. Keeping every cross-bin entry preserves that
    # nonnegative SSC construction before the separate cNG term is added.
    ssc = interface.covariance_project(
        left=shell, right=shell, weight=geometry[3]*variance
    )
    ssc = np.triu(ssc)+np.triu(ssc, k=1).T
    cng = project_connected(
        interface=interface, rows=rows, pair_window=pair_window,
        projected=projected,
        measure=geometry[3]/(geometry[2]**6*settings["area_sr"]),
    )
    checkpoint("ssc_and_connected_projection", tick)
    total = gaussian+ssc+cng
    checkpoint("total", started)
    return {
        "gaussian": gaussian,
        "ssc": ssc,
        "cng": cng,
        "total": total,
        "rows": rows,
        "geometry": geometry,
        "coarse_ell": coarse_ell,
        "pair_area_sr2": pair_area,
        "signal": observable_signal,
        "stages_s": stages,
    }
