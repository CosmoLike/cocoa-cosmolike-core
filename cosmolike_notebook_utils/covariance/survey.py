"""Real-space and Fourier 3x2pt matrices under an explicit Limber halo model.

The Gaussian, super-sample and connected terms follow the decomposition
in Krause & Eifler (2017), Appendix A, arXiv:1601.05779. SSC uses the
isotropic halo response of Takada & Hu (2013), corrected Eq. 44 and
Appendix A, arXiv:1302.6994. The mask is a spherical cap; long and short
modes in SSC/cNG use Limber. Gaussian gg/gs can retain radial mode coupling. This is not a calibrated nonlinear tidal response.

All cross-bin blocks are retained. The first survey assembly supports
massless neutrinos, linear galaxy bias and zero magnification. Gaussian
IA is selectable; SSC/cNG retain zero IA and their original mean signal.
It returns a forecast, not a replacement for a project's supplied matrix.
Numerical refinement and Fisher convergence remain separate checks.
"""

import time

import numpy as np

from .gaussian import observed_spectra, limber_spectra
from .geometry import angular_rule, cap_mask
from .accuracy import non_gaussian_multipoles


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


def compress_operators(operators, ell, coarse_ell, source_factor=None):
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
        source_factor = optional [nell] transfer per source leg. The default
            reproduces Cocoa's real-space transformation. Fourier callers
            supply sqrt[(ell-1)ell(ell+1)(ell+2)]/(ell+1/2)^2, matching
            the C spectra directly without an angular-kernel conversion.
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
    if source_factor is not None:
        spin = np.asarray(source_factor, dtype=float)
        if spin.shape != np.shape(ell) or not np.all(np.isfinite(spin)):
            raise ValueError("source_factor must be finite and match ell")
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
    lens families. The wrapper distributes complete angular blocks over
    one thread team, retaining each radial sum's order. Symmetric entries
    are copied from one computation.

    Arguments:
        interface = compiled project exposing covariance_project_connected.
        rows = int [nobservable,3] from observable_rows.
        pair_window = float [nobservable,nradial], each W_A W_B, in
            (c/H0)^-2; this already includes the linear lens biases.
        projected = float [4*ntheta,4*ntheta,nradial], in (c/H0)^9.
        measure = float [nradial], dchi/(area*f_K^6), in (c/H0)^-5.
    Returns:
        Dimensionless symmetric [nobservable*ntheta,nobservable*ntheta]
        cNG covariance, retaining the supplied signed contributions.
    """
    indices = np.asarray(rows)
    if (indices.ndim != 2 or indices.shape[1] != 3
            or not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("rows must be an integer [nobservable,3] array")
    probes = indices[:, 0]
    if np.any(probes < 0) or np.any(probes > 3):
        raise ValueError("row probe IDs must lie in 0..3")
    return interface.covariance_project_connected(
        probes=np.ascontiguousarray(probes, dtype=np.int32),
        pair_window=np.ascontiguousarray(pair_window),
        projected=np.ascontiguousarray(projected),
        measure=np.ascontiguousarray(measure),
    )


def _matter_covariance_tables(interface, settings, geometry, coarse_ell,
                             transform, mask_nell, progress=None):
    """Project shared matter trispectra and responses before catalog weights.

    A foreground shell has one matter trispectrum regardless of which
    galaxies, clusters or sources observe it. Contract its two multipole
    axes with the supplied angular/band operators here; catalog windows
    are applied afterwards. Reusing these tables avoids repeating the
    costly halo and perturbation-theory integrals for each observable pair.

    Arguments:
        interface = initialized project exposing covariance halo components.
        settings = resolved halo_mass_nquad, lnm_edges, tree_nquad,
            tree_npanel and response_step integration controls; mnu must
            be zero for this combined matter prescription.
        geometry = float [4,nstate], rows a, chi, f_K, dchi; lengths in c/H0.
        coarse_ell = positive float [nk] samples of the angular modes.
        transform = float [ntransform,nk] compressed measurement operators,
            including the intended source-leg transfer factors.
        mask_nell = number of integer mask multipoles, starting at zero.
        progress = optional callable receiving completed and total shells.
    Returns:
        Dict with projected [ntransform,ntransform,nstate] trispectra in
        (c/H0)^9; response [ntransform,nstate] in (c/H0)^3; and
        long_power [nstate,mask_nell] in (c/H0)^3. Arrays are owned. No
        file, catalog normalization or matrix-positivity repair is applied.
    """
    if settings["mnu"] != 0.0:
        raise ValueError("matter covariance tables require mnu=0")
    if not np.isfinite(settings["response_step"]) or settings["response_step"] <= 0:
        raise ValueError("response_step must be finite and positive")
    nnode = geometry.shape[1]
    projected = np.empty((len(transform), len(transform), nnode))
    response = np.empty((len(transform), nnode))
    long_power = np.empty((nnode, mask_nell))
    unused, angle_weight, corner = angular_rule(
        nquad=settings["tree_nquad"], npanel=settings["tree_npanel"],
        interface=interface,
    )
    first, second = np.triu_indices(n=len(coarse_ell))
    # K,Q both scale as 1/chi along a Limber shell. Build the angular
    # geometry once in multipole units, then rescale at each radial node.
    modes = coarse_ell+0.5
    magnitude = np.sqrt((modes[first, None]-modes[second, None])**2
                        +2*modes[first, None]*modes[second, None]*corner)
    unit_weight = np.ones(len(coarse_ell))
    shift = np.exp(np.array([-settings["response_step"], 0.0,
                             settings["response_step"]]))
    shift = shift[[0, 2]]
    diagonal = np.flatnonzero(first == second)
    mask_modes = np.arange(mask_nell)+0.5

    # Halos at different distances have independent mass integrals. A
    # small group of shells shares the mass rule and gives OpenMP enough
    # independent (a,k) rows to occupy eight workers. Only eight shells
    # are held at once, rather than the entire refined radial grid.
    # This grouping changes scheduling only, never the integration nodes.
    batch_size = 8
    for begin in range(0, nnode, batch_size):
        end = min(begin+batch_size, nnode)
        scale_factor = geometry[0, begin:end].copy()
        wave = modes[None, :]/geometry[2, begin:end, None]
        single, moments = interface.covariance_halo_moments(
            a=scale_factor, k=wave, lnm_edges=settings["lnm_edges"],
            nquad=settings["halo_mass_nquad"],
        )

        # The logarithmic slope of I11^2 P needs two displaced k values.
        # Keep all of them on their parent shell's row so the mass function,
        # bias and concentration are evaluated once per shell. Pair moments
        # belong to the central k grid above; they are not used here.
        shifted_k = wave[:, :, None]*shift
        shifted_single, unused = interface.covariance_halo_moments(
            a=scale_factor, k=shifted_k.reshape(end-begin, -1),
            lnm_edges=settings["lnm_edges"],
            nquad=settings["halo_mass_nquad"], pair_moments=False,
        )
        shifted_single = shifted_single.reshape(shifted_k.shape)

        # A fixed multipole samples k=(ell+1/2)/chi, so each shell still
        # needs its own power spectrum and tree-level angular terms. Keep
        # those physical k values and both final projections unchanged.
        for row, node in enumerate(range(begin, end)):
            a = scale_factor[row]
            distance = geometry[2, node]
            k = wave[row]

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
                pk=pk, i11=np.array([single[row, first], single[row, second]]),
                moments=np.ascontiguousarray(moments[:, row]), tree=angular,
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

            # Evaluate the same centered difference at k*exp(+/-step).
            # Omitting the unused central sample does not change either
            # endpoint or their logarithmic separation of twice the step.
            shifted_power = interface.covariance_power(
                a=a, k=shifted_k[row], linear=True
            )
            two_halo = shifted_single[row]**2*shifted_power
            slope = np.log(two_halo[:, 1]/two_halo[:, 0])
            slope /= 2*settings["response_step"]

            # The halo model predicts the fractional response to background
            # density. Transfer that fraction to the chosen nonlinear power;
            # this defines the stated SSC approximation, not a tidal response.
            target = interface.covariance_power(a=a, k=k, linear=False)
            dimensional = interface.covariance_halo_response(
                inputs=np.array([
                    linear,
                    target,
                    single[row],
                    moments[0, row, diagonal],
                    moments[1, row, diagonal],
                    slope,
                ]),
                growth_coefficient=47.0/21.0, dilation_coefficient=1.0/3.0,
                fractional=True,
            )[1]
            response[:, node] = interface.covariance_project(
                left=transform, right=dimensional[None, :], weight=unit_weight
            )[:, 0]

            # The footprint weights much longer wavelengths than the measured
            # angular bins. Its background variance uses linear matter power.
            long_power[node] = interface.covariance_power(
                a=a, k=mask_modes/distance, linear=True
            )
            if progress is not None and node % 32 == 0:
                progress(node+1, nnode)
    return {
        "projected": projected,
        "response": response,
        "long_power": long_power,
    }


def realspace_covariance(interface, settings, rows, noise, progress=None):
    """Compute angular-bin G, SSC, cNG and total covariances.

    Arguments:
        interface = initialized project interface.
        settings = resolved survey/integration mapping, documented in
            _survey_covariance; edges_rad contains common angular edges.
        rows = int32 [nobservable,3], (probe,A,B) from observable_rows.
        noise = float64 [nfield], independent observed white-noise powers.
        progress = optional callable taking (stage, elapsed_seconds).
    Returns:
        Owned component/total matrices with angular bins inside each row,
        mean signals, geometry and stage times; see _survey_covariance.
        No files or likelihood covariance are changed.
    """
    return _survey_covariance(
        interface=interface, settings=settings, rows=rows, noise=noise,
        progress=progress, space="real",
    )


def fourier_covariance(interface, settings, rows, noise, progress=None):
    """Compute E-mode bandpower G, SSC, cNG and total covariances.

    The supplied mean spectra use the core Fourier shear normalization.
    The extra factor that matches Cocoa's real-space transform is absent;
    the same Fourier normalization enters G, SSC and cNG, once per leg.

    The same matter trispectrum and background response used in angular
    space are averaged over integer multipoles in each band. Each weight
    is proportional to 2*ell+1, the number of harmonic modes. White noise
    belongs to the finite band sum; no angular pair-area term is added.

    Arguments:
        interface = initialized project interface.
        settings = common resolved mapping plus int32 band_first and
            band_last arrays with inclusive integer band endpoints >= 2.
            Bands may overlap. Their endpoints are scientific choices,
            held fixed under interpolation and quadrature refinements.
        rows = int32 [nobservable,3], (type,A,B). Type 0 is shear E-E,
            2 galaxy-E and 3 galaxy-galaxy. Do not include xi- rows: an
            E-mode spectrum supplies both real-space shear correlations.
        noise = float64 [nfield], independent observed white-noise powers.
        progress = optional callable taking (stage, elapsed_seconds).
    Returns:
        The same dict as realspace_covariance, with bands innermost.
        pair_area_sr2 is an empty array because it has no Fourier role.
        No files or likelihood covariance are changed.
    """
    return _survey_covariance(
        interface=interface, settings=settings, rows=rows, noise=noise,
        progress=progress, space="fourier",
    )


def _survey_covariance(interface, settings, rows, noise, progress, space):
    """Compute separate G, SSC and cNG matrices and return elapsed stages.

    Arguments:
        interface = project interface with cosmology and catalogs initialized.
        settings = resolved integration and physical choices:
            ell_max, mask_ell_max = signal and footprint multipole cutoffs;
            ng_ell = supplied samples uniform in ln(ell+1/2);
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
        space = "real" for spin angular bins, "fourier" for integer bands.

    Returns:
        dict with gaussian, ssc, cng and total [ndata,ndata] dimensionless
        matrices; signal [nobservable,nbin]; rows; coarse_ell; radial
        geometry [4,nradial] (a,chi,f_K,dchi); pair_area_sr2 [nbin]; and
        stages_s elapsed times. Here ndata=nobservable*nbin; distances
        have units c/H0. The function writes no files and repairs no modes.

    Returned arrays order observables first, then angular bins or bands.
    Times include Python preparation, core workspaces and cold table builds;
    CAMB/initialization, diagnostic eigenproblems and file writing are outside
    this function. No shared table is recomputed inside a tomographic block.
    """
    if settings["mnu"] != 0.0:
        raise ValueError("the full halo matter model currently requires mnu=0")
    if not np.isfinite(settings["response_step"]) or settings["response_step"] <= 0:
        raise ValueError("response_step must be finite and positive")
    if rows.ndim != 2 or rows.shape[1] != 3 or rows.dtype.kind not in "iu":
        raise ValueError("rows must contain integer (probe,A,B) triplets")
    if space == "fourier" and np.any(rows[:, 0] == 1):
        raise ValueError("Fourier rows must omit xi-: retain E-E only once")
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
    ell_max = settings["ell_max"]
    if space == "fourier":
        # Unlike a real-space transform, a measured band ends at a fixed
        # multipole. Refinement must not change which modes it measures.
        first_band = np.asarray(settings["band_first"])
        last_band = np.asarray(settings["band_last"])
        if (first_band.ndim != 1 or len(first_band) == 0
                or last_band.shape != first_band.shape
                or first_band.dtype.kind not in "iu"
                or last_band.dtype.kind not in "iu"
                or np.any(first_band < 2)
                or np.any(last_band < first_band)):
            raise ValueError("need matching integer bands with 2<=first<=last")
        ell_max = int(np.max(last_band))
    coarse_ell = non_gaussian_multipoles(samples=settings["ng_ell"], ell_max=ell_max)
    ell = np.arange(2, ell_max+1, dtype=float)
    snapshot = limber_spectra(
        interface=interface,
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
    # SSC/cNG keep their established lensing-only, Limber normalization.
    # Gaussian spectra below are a separate snapshot; changing their IA or
    # radial-mode treatment must not alter the SSC mean-subtraction signal.
    lensing_spectra = snapshot["spectra"]
    model = settings.get("gaussian", {"nonlimber": False, "ia": "none"})
    if model["ia"] != "none":
        snapshot = limber_spectra(
            interface=interface, ell=ell, a_edges=settings["a_edges"],
            nquad=settings["radial_nquad"], nwindow=settings["nwindow"],
            include_ia=True, include_rsd=False, linear=False,
        )
    if model["nonlimber"]:
        cutoff = min(settings["nonlimber_lmax"], ell_max)
        selected = ell <= cutoff
        low = interface.covariance_spectra(
            ell=ell[selected], a_edges=np.ascontiguousarray(settings["a_edges"]),
            nquad=settings["radial_nquad"], nwindow=settings["nwindow"],
            include_ia=model["ia"] != "none", include_rsd=False, linear=False,
            nonlimber_lmax=cutoff, nonlimber_nchi=settings["nonlimber_nchi"],
        )
        # Copy when Gaussian and SSC still reference the same no-IA array.
        snapshot["spectra"] = snapshot["spectra"].copy()
        snapshot["spectra"][selected] = low["spectra"]
    b_signal = snapshot.get("b_spectra")
    source_factor = None
    if space == "real":
        # The core angular transforms carry an additional source-leg
        # factor relative to unit-normalized Wigner kernels. Preserve that
        # real-space convention in the mean, G, SSC and cNG together.
        signal = observed_spectra(snapshot["spectra"], ell=ell, nlens=nlens)
        ssc_signal = observed_spectra(lensing_spectra, ell=ell, nlens=nlens)
        if b_signal is not None:
            b_signal = observed_spectra(b_signal, ell=ell, nlens=nlens)
    else:
        # A Fourier band averages the core C_ell directly. Do not apply
        # the conversion used only to match the real-space transforms.
        signal = np.ascontiguousarray(snapshot["spectra"])
        ssc_signal = np.ascontiguousarray(lensing_spectra)
        source_factor = np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+0.5)**2
    checkpoint("all_pairs_gaussian_spectra", started)

    # --- 2. Measurement operators and the common survey footprint ---
    # Real-space white noise extends above any finite ell cutoff, so its
    # pair-count term is analytic. A Fourier band instead includes exactly
    # the modes between its endpoints; its noise uses that finite sum.
    tick = time.perf_counter()
    mask = cap_mask(area_sr=settings["area_sr"],
                    ell_max=settings["mask_ell_max"])
    pair_area = np.empty(0)
    if space == "real":
        operators = interface.covariance_realspace_operator(
            edges_rad=settings["edges_rad"], ell_max=ell_max,
            nquad=settings["angle_nquad"],
        )
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
    else:
        bands = interface.covariance_bandpower_operator(
            first=np.ascontiguousarray(first_band, dtype=np.int32),
            last=np.ascontiguousarray(last_band, dtype=np.int32),
            ell_min=2, nell=len(ell),
        )
        # All Fourier fields use the same band average. Keep four operator
        # slots so the common SSC/cNG projection can attach two, one or no
        # shear transfer factors to E-E, galaxy-E and galaxy density.
        kernels = np.repeat(bands[None, :, :], repeats=4, axis=0)
    nbin = kernels.shape[1]
    ndata = len(rows)*nbin
    checkpoint("angular_and_mask_geometry", tick)

    # --- 3. Gaussian covariance and the observable mean signals ---
    # A block covariance needs crossed spectra, including lens cross bins
    # absent from the data vector. The snapshot retains that full matrix.
    tick = time.perf_counter()
    if space == "real":
        b_options = {}
        if b_signal is not None:
            b_options["b_spectra"] = np.ascontiguousarray(b_signal)
        gaussian = interface.covariance_gaussian_real(
            spectra=signal, noise=np.ascontiguousarray(noise),
            rows=np.ascontiguousarray(rows, dtype=np.int32), operators=kernels,
            ell_min=2, area_sr=settings["area_sr"], pair_area_sr2=pair_area,
            **b_options,
        )
    else:
        gaussian = interface.covariance_gaussian_fourier(
            spectra=signal, noise=np.ascontiguousarray(noise),
            pairs=np.ascontiguousarray(rows[:, 1:], dtype=np.int32),
            operators=bands, ell_min=2, area_sr=settings["area_sr"],
        )
    checkpoint("gaussian_blocks", tick)

    # The SSC mean subtraction uses the full projected signal of each
    # measured observable, rather than its local radial integrand.
    tick = time.perf_counter()
    observable_signal = np.empty((len(rows), nbin))
    gaussian_signal = np.empty_like(observable_signal)
    for probe in range(4):
        selected = np.flatnonzero(rows[:, 0] == probe)
        if len(selected) == 0:
            continue
        fields = rows[selected, 1:]
        spectra = np.ascontiguousarray(ssc_signal[:, fields[:, 0], fields[:, 1]].T)
        gaussian_power = signal[:, fields[:, 0], fields[:, 1]].T.copy()
        if b_signal is not None and space == "real" and probe in (0, 1):
            sign = 1.0 if probe == 0 else -1.0
            gaussian_power += sign*b_signal[:, fields[:, 0], fields[:, 1]].T
        gaussian_signal[selected] = interface.covariance_project(
            left=np.ascontiguousarray(gaussian_power), right=kernels[probe],
            weight=np.ones(len(ell)),
        )
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
    compressed = compress_operators(
        operators=kernels, ell=ell, coarse_ell=coarse_ell,
        source_factor=source_factor,
    )
    transform = compressed.reshape(4*nbin, -1)
    def report_matter(completed, total):
        """Translate shared-table progress into the survey's elapsed time."""
        if progress is not None:
            progress(f"Matter shell {completed}/{total}", time.perf_counter()-started)

    matter = _matter_covariance_tables(
        interface=interface, settings=settings, geometry=geometry,
        coarse_ell=coarse_ell, transform=transform, mask_nell=len(mask),
        progress=report_matter,
    )
    projected = matter["projected"]
    response = matter["response"]
    long_power = matter["long_power"]
    nnode = geometry.shape[1]
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
    response = response.reshape(4, nbin, nnode)
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
        "signal": gaussian_signal,
        "ssc_normalization_signal": observable_signal,
        "stages_s": stages,
    }
