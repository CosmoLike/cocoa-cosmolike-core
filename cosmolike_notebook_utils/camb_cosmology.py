"""One CAMB run packaged into the tuple set_cosmology consumes.

Every EXAMPLE_EVALUATE notebook needs the same thing before it can
call the compiled cosmolike interface: linear and nonlinear matter
power spectra on a (z, k) grid, the growth factor, and comoving
distances, all in the h units cosmolike expects. This module holds
the one function that produces them, together with the two
massive-neutrino inputs of the halo model: omega_nu h^2 and the linear
power spectrum of cold dark matter + baryons (P_cb, CAMB's delta_nonu),
which cosmolike reads when the halo field is cb. It talks only to
CAMB (and, optionally, to the EuclidEmulator2 boost); it never
imports a project's compiled cosmolike interface, so every project
shares it.

The interpolation grids follow the likelihoods
(likelihood/_cosmolike_prototype_base.py of each project). The 2D z
grid is the nested dyadic grid: raising the accuracy boost refines
it by an integer factor with the same block endpoints, so a boost
increase is a true refinement of the previous grid instead of a
re-phasing of the linear-interpolation error. The 1D z grid has the
likelihoods' segments, but its node count grows with the CAMB-side
boost here and with the cosmolike accuracy boost there: the two
grids coincide at boost 1.

The tables live on three redshift grids. The power spectra use the
2D grid z_interp_2D (140 nodes at boost 1, dz ~ 0.03 below z = 3).
The distances use the dense 1D grid z_interp_1D (dz = 0.003 below
z = 3 at boost 1). The growth table uses the same dense grid, cut
at the last 2D node (z_growth). cosmolike reads the growth table
linearly in z, and on the 2D grid that linear read would miss D(z)
by up to 9e-5. As in the likelihoods, the growth table is divided
by its value at the last 2D node, z_interp_2D[-1], and goes to
set_cosmology as G together with its z grid as z_G.

Units handed back (cosmolike conventions):
  k          in h/Mpc (log10 grid)
  P(k)       in (Mpc/h)^3 (natural log of it)
  distances  in Mpc/h
"""

import numpy as np
from scipy.interpolate import interp1d


def get_camb_cosmology(omegam,
                       omegab,
                       H0,
                       ns,
                       As_1e9,
                       w,
                       w0pwa,
                       mnu=0.06,
                       AccuracyBoost=1.0,
                       kmax=10.0,
                       k_per_logint=10,
                       CAMBAccuracyBoost=1.0,
                       CLAccuracyBoost=1.0,
                       non_linear_emul=2,
                       lens_potential_accuracy=1.0,
                       halofit_version='takahashi'):
    """Run CAMB once and package the cosmology for ci.set_cosmology.

    The notebooks call this through thin wrappers that add their
    project's nuisance parameters; the cosmology itself is generic.
    AccuracyBoost is the overall knob: a third of its excess over 1
    goes to the CAMB side (CAMB boost = CAMBAccuracyBoost*(1 +
    (AccuracyBoost - 1)/3)), and it multiplies CLAccuracyBoost for
    the cosmolike grids, so one number raises the whole pipeline.
    CAMBAccuracyBoost and CLAccuracyBoost move only their own side.

    Arguments:
      omegam  = total matter density parameter Omega_m (cold dark
              matter + baryons + massive neutrinos).
      omegab  = baryon density parameter Omega_b.
      H0      = Hubble constant in km/s/Mpc.
      ns      = scalar spectral index.
      As_1e9  = 10^9 x As (primordial amplitude, so ~2.1).
      w       = dark-energy equation of state today.
      w0pwa   = w0 + wa; wa is reconstructed as w0pwa - w.
      mnu     = sum of neutrino masses in eV (one massive state).
      AccuracyBoost = overall accuracy knob (see above).
      kmax    = CAMB maximum k in 1/Mpc before the boost scaling.
      k_per_logint  = CAMB k samples per log interval before boost.
      CAMBAccuracyBoost = CAMB-only boost multiplier.
      CLAccuracyBoost   = cosmolike-only boost multiplier; with
              AccuracyBoost it sets the dyadic z-grid factor m and
              the number of k nodes.
      non_linear_emul   = 1 replaces the z < 10 nonlinear spectrum
              with the EuclidEmulator2 boost on top of the linear
              one; 2 keeps CAMB's halofit everywhere. Any other value
              leaves lnPNL unset, and the function fails at its
              return.
      lens_potential_accuracy = CAMB lensing accuracy (scaled by the
              CAMB boost like the other CAMB settings).
      halofit_version = CAMB halofit flavor for the nonlinear P(k).

    Returns:
      (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_growth,
       z_interp_1D, chi, omegan2, lnPL_cb): what ci.set_cosmology
      consumes, as the keywords log10k_2D, z_2D, lnP_linear,
      lnP_nonlinear, G, z_G, z_1D, chi, omegan2 and lnP_linear_cb.
      omegan2 is a float, every other entry a 1D float64 array:
        log10k_interp_2D = log10 of k in h/Mpc, n_k nodes.
        z_interp_2D      = z nodes of the power spectra, n_z2D nodes.
        lnPL, lnPNL      = ln of the linear and nonlinear P(k) in
                           (Mpc/h)^3: the (n_z2D, n_k) tables flattened
                           in Fortran (column) order, the layout the
                           compiled interface expects.
        G_growth         = G(z) = D(z) (1 + z) on z_growth, divided by
                           its value at z_interp_2D[-1]; n_zG entries.
        z_growth         = z nodes of G_growth: the z_interp_1D nodes
                           up to z_interp_2D[-1]; n_zG nodes.
        z_interp_1D      = z nodes of chi, n_z1D nodes.
        chi              = comoving distance in Mpc/h, n_z1D entries.
        omegan2          = omega_nu h^2 of the massive neutrinos, as
                           CAMB computed it (pars.omnuh2).
        lnPL_cb          = ln of the linear P_cb (cold dark matter +
                           baryons) in (Mpc/h)^3, laid out as lnPL.
      Halo statistics require both of the last two quantities.
      G_growth is not on the z_2D nodes: it must go to set_cosmology
      together with z_G = z_growth (set_cosmology without z_G pairs G
      with z_2D, and the size mismatch aborts the process).
    """
    # camb is imported at the first call, not at module top: the
    # package __init__ imports this module, so a top-level import would
    # make every import of cosmolike_notebook_utils (plots and
    # covariance tools included) need CAMB, and a caller importing the
    # package before putting its CAMB build on sys.path would get
    # whichever camb the default path holds
    import camb
    from camb import model

    # each lambda is a one-line function: called below with explicit
    # arguments, they keep the unit conversions readable and in one
    # place (omegach2 subtracts the massive-neutrino share,
    # omega_nu h^2 = mnu (3.046/3)^(3/4) / 94.0708 eV for N_eff =
    # 3.046, from the cold component)
    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    # the overall boost hands one third of its excess over 1 to the
    # CAMB side (AccuracyBoost = 4 doubles it); the CAMB-side settings
    # then scale together
    CAMBAccuracyBoost = CAMBAccuracyBoost*(1.0 + (AccuracyBoost-1.0)/3.0)
    lens_potential_accuracy = lens_potential_accuracy*CAMBAccuracyBoost
    kmax = kmax*(1.0 + 3*(CAMBAccuracyBoost-1))
    k_per_logint = int(k_per_logint) + int(30*(CAMBAccuracyBoost-1))
    # the interpolators extrapolate P(k) beyond kmax up to extrap_kmax
    # (1/Mpc), past the end of the k table at 10^2 = 100/Mpc
    extrap_kmax=2.5e2*CAMBAccuracyBoost

    # cosmolike aborts on non-monotone chi(z) grids (duplicate segment
    # endpoints), so interior segments drop their endpoint
    # (endpoint=False leaves z = 3 out of the first segment; the
    # second segment starts there). The [1070, 1100] segment brackets
    # last scattering, z ~ 1090, the source plane of CMB lensing.
    # Same segments as likelihood/_cosmolike_prototype_base.py, but
    # tmp grows here with the CAMB boost and there with the cosmolike
    # accuracyboost: the two grids coincide at boost 1.
    tmp=int(1000 + 250*CAMBAccuracyBoost)
    z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                  np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                  np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    len_z_interp_1D = len(z_interp_1D)

    # z nodes of the 2D power-spectrum tables handed to cosmolike,
    # which interpolates linearly in z between exactly these nodes.
    # The dyadic factor m = 2^ceil(log2(boost)), capped at 16, with
    # boost = CLAccuracyBoost*AccuracyBoost, refines each uniform
    # block with the same endpoints, so a boost increase is a true
    # refinement (the O(dz^2) interpolation residual falls like 1/m^2
    # instead of re-phasing); same grid as
    # likelihood/_cosmolike_prototype_base.py at its default
    # pk_z_refinement = 1.
    m = int(min(2**np.ceil(np.log2(max(1.0, CLAccuracyBoost*AccuracyBoost))), 16))
    z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False),
                                  np.linspace(3.0,49.99,34*m + 1)),axis=0)
    len_z_interp_2D = len(z_interp_2D)

    # CAMB caps requested transfer redshifts at 256, so the request
    # grid handed to set_matter_power stays at the boost-independent
    # 140-node (m = 1) grid; the denser nodes above only re-evaluate
    # the smooth z-spline CAMB builds from these transfer redshifts.
    z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False),
                                       np.linspace(3.0,49.99,35)),axis=0)

    # log10 of k in 1/Mpc (CAMB's units while the tables are filled);
    # shifted to h/Mpc once the tables are evaluated
    log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*CLAccuracyBoost*AccuracyBoost))
    len_log10k_interp_2D = len(log10k_interp_2D)

    # only P(k), distances and the growth factor are read from this
    # run: lmax = 10 keeps the CMB part minimal and tau (reionization)
    # enters none of them. nnu = 3.046 with one massive state is the
    # neutrino content omegach2 assumes; CAMB converts mnu with its
    # own constants, so pars.omnuh2 differs slightly from that share
    pars = camb.set_params(H0=H0,
                           ombh2=omegabh2(omegab, H0),
                           omch2=omegach2(omegam, omegab, mnu, H0),
                           mnu=mnu,
                           omk=0,
                           tau=0.06,
                           As=As(As_1e9),
                           ns=ns,
                           halofit_version=halofit_version,
                           lmax=10,
                           AccuracyBoost=CAMBAccuracyBoost,
                           lens_potential_accuracy=lens_potential_accuracy,
                           num_massive_neutrinos=1,
                           nnu=3.046,
                           accurate_massive_neutrino_transfers=False,
                           k_per_logint=k_per_logint,
                           kmax = kmax);
    # PPF dark-energy perturbations stay stable when w(a) crosses -1
    pars.set_dark_energy(w=w, wa=wa(w0pwa, w), dark_energy_model='ppf');
    pars.NonLinear = model.NonLinear_both
    pars.set_matter_power(redshifts=z_interp_2D_camb, kmax=kmax, silent=True);

    results = camb.get_results(pars)

    # hubble_units=False and k_hunit=False: the interpolators take k
    # in 1/Mpc and return P in Mpc^3; the h conversions happen when
    # the tables are filled
    PKL  = results.get_matter_power_interpolator(var1="delta_tot",
                                                 var2="delta_tot",
                                                 nonlinear=False,
                                                 extrap_kmax=extrap_kmax,
                                                 hubble_units = False,
                                                 k_hunit = False);
    # the matter without the massive neutrinos, which free-stream out
    # of halos: sigma(M) of the cb halo field integrates this spectrum
    PKL_cb = results.get_matter_power_interpolator(var1="delta_nonu",
                                                   var2="delta_nonu",
                                                   nonlinear=False,
                                                   extrap_kmax=extrap_kmax,
                                                   hubble_units=False,
                                                   k_hunit=False);
    PKNL = results.get_matter_power_interpolator(var1="delta_tot",
                                                 var2="delta_tot",
                                                 nonlinear=True,
                                                 extrap_kmax=extrap_kmax,
                                                 hubble_units=False,
                                                 k_hunit=False);

    # PKL.P(z, k) evaluates the interpolator on the full z x k grid at
    # once and returns a 2D table; flatten(order='F') serializes it
    # column by column (Fortran order), the memory layout the compiled
    # interface expects. The added log((H0/100)^3) converts P(k) from
    # Mpc^3 to (Mpc/h)^3.
    lnPL = np.log(PKL.P(z_interp_2D,np.power(10.0, log10k_interp_2D)).flatten(order='F')) + np.log((H0/100.0)**3)
    # the same grid, layout and units for P_cb
    k_grid = np.power(10.0, log10k_interp_2D)
    lnPL_cb = np.log(PKL_cb.P(z_interp_2D, k_grid).flatten(order='F'))
    lnPL_cb = lnPL_cb + np.log((H0/100.0)**3)

    if non_linear_emul == 1:
        # imported only on this branch: EuclidEmulator2 prints a
        # warning banner at import time, and the halofit-only branch
        # should not pay it
        import euclidemu2
        params = { 'Omm'  : omegam,
                   'As'   : As(As_1e9),
                   'Omb'  : omegab,
                   'ns'   : ns,
                   'h'    : H0/100.,
                   'mnu'  : mnu,
                   'w'    : w,
                   'wa'   : wa(w0pwa, w)
                 }
        # z_interp_2D[z_interp_2D < 10.0] keeps only the entries below
        # z = 10 (a boolean mask used as an index): EE2 is trained on
        # z < 10 and rejects anything above. The boost is requested on
        # as many k nodes as the table, from 10^-2.0589 = 8.73e-3 to
        # 10^0.973 = 9.40 h/Mpc
        kbt, tmp_bt = euclidemu2.get_boost(params,z_interp_2D[z_interp_2D < 10.0],10**np.linspace(-2.0589,0.973,len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # interp1d builds a function that linearly interpolates
        # ln(boost) in log10(k), extrapolating linearly beyond the EE2
        # nodes; calling it on the shifted grid resamples ln(boost)
        # onto the table's k nodes in h/Mpc
        tmp = interp1d(np.log10(kbt),
                        np.log(bt),
                        axis=1,
                        kind='linear',
                        fill_value='extrapolate',
                        assume_sorted=True)(log10k_interp_2D-np.log10(H0/100.)) #h/Mpc
        # boolean-mask assignment: every column whose k (in h/Mpc)
        # sits below EE2's trained range gets ln(boost) = 0, a boost
        # of 1, so the nonlinear spectrum equals the linear one there
        tmp[:,10**(log10k_interp_2D-np.log10(H0/100)) < 8.73e-3] = 0.0
        lnbt = np.zeros((len_z_interp_2D, len_log10k_interp_2D))
        lnbt[z_interp_2D < 10.0, :] = tmp
        # halofit on every redshift first
        lnPNL = np.log(PKNL.P(z_interp_2D, np.power(10.0, log10k_interp_2D)).flatten(order='F')) + np.log((H0/100.0)**3)
        # on z < 10.0, replace it with the linear spectrum times the
        # EE2 boost. np.where(cond, a, b) picks a where cond is True
        # and b where it is False, row by row: the (z < 10)[:, None]
        # adds a length-1 axis so the z condition broadcasts across
        # every k column of the reshaped tables
        lnPNL = np.where((z_interp_2D<10)[:,None], lnPL.reshape(len_z_interp_2D, len_log10k_interp_2D, order='F') + lnbt,
                                                   lnPNL.reshape(len_z_interp_2D, len_log10k_interp_2D, order='F')).ravel(order='F')
    elif non_linear_emul == 2:
        lnPNL = np.log(PKNL.P(z_interp_2D, np.power(10.0, log10k_interp_2D)).flatten(order='F')) + np.log((H0/100.0)**3)
    # only after the tables were evaluated (k in 1/Mpc): shift the k
    # grid to h/Mpc
    log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)

    # growth factor G(z) = D(z) (1 + z) from the linear P(k) at one
    # sub-horizon scale (k = 0.05/Mpc; at 5e-4/Mpc, about 2 H0/c, CAMB's
    # dark-energy perturbations change the growth at w != -1, see the
    # likelihoods' comment), where P grows as D^2. It is sampled
    # on the dense 1D grid, cut where the 2D grid (the z range of PKL)
    # ends: cosmolike reads G linearly in z, and on the 2D grid
    # (dz ~ 0.03) that read misses D by up to 9e-5 and the growth rate
    # (the slope of the table) by 1%. PKL is a spline in z through
    # CAMB's transfer redshifts, so the extra nodes ask CAMB for no
    # extra redshifts. Same expression as the likelihoods
    # (likelihood/_cosmolike_prototype_base.py).
    z_growth = z_interp_1D[z_interp_1D <= z_interp_2D[-1]]
    power_ratio = PKL.P(z_growth, 0.05)/PKL.P(0, 0.05)
    G_growth = np.sqrt(power_ratio)*(1 + z_growth)
    # the table is divided by G at the last 2D node (z = 49.99), just
    # above the last z_growth node; cosmolike divides by G(z = 0) on
    # its side, so D(z = 0) = 1 whatever this constant is
    z_norm = z_interp_2D[-1]
    power_ratio_norm = PKL.P(z_norm, 0.05)/PKL.P(0, 0.05)
    G_growth = G_growth/(np.sqrt(power_ratio_norm)*(1 + z_norm))

    # CAMB returns Mpc; multiplying by h gives Mpc/h
    chi = results.comoving_radial_distance(z_interp_1D) * (H0/100.)

    # omega_nu h^2 as CAMB set it from mnu (one massive state)
    omegan2 = float(pars.omnuh2)

    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_growth,
            z_interp_1D, chi, omegan2, lnPL_cb)
