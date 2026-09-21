"""One CAMB run packaged into the tuple set_cosmology consumes.

Every EXAMPLE_EVALUATE notebook needs the same thing before it can
call the compiled cosmolike interface: linear and nonlinear matter
power spectra on a (z, k) grid, the growth factor, and comoving
distances, all in the h units cosmolike expects. This module holds
the one function that produces them. It talks only to CAMB (and,
optionally, to the EuclidEmulator2 boost); it never imports a
project's compiled cosmolike interface, so every project shares it.

The interpolation grids built here are the same nested dyadic grids
the likelihoods use (likelihood/_cosmolike_prototype_base.py of each
project): raising the accuracy boost refines the z grid by an
integer factor with the same block endpoints, so a boost increase is
a true refinement of the previous grid instead of a re-phasing of
the linear-interpolation error.

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
    AccuracyBoost is the overall knob: it feeds a third of itself to
    the CAMB boost and multiplies CLAccuracyBoost for the grids, so
    one number raises the whole pipeline. CAMBAccuracyBoost and
    CLAccuracyBoost move only their own side.

    Arguments:
      omegam  = total matter density parameter Omega_m.
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
              AccuracyBoost it sets the dyadic grid factor m.
      non_linear_emul   = 1 replaces the z < 10 nonlinear spectrum
              with the EuclidEmulator2 boost on top of the linear
              one; 2 keeps CAMB's halofit everywhere.
      lens_potential_accuracy = CAMB lensing accuracy (scaled by the
              CAMB boost like the other CAMB settings).
      halofit_version = CAMB halofit flavor for the nonlinear P(k).

    Returns:
      (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth,
       z_interp_1D, chi): the exact tuple ci.set_cosmology consumes.
      lnPL/lnPNL are the (z, k) tables flattened in Fortran (column)
      order, the layout the compiled interface expects.
    """
    # camb is imported here, not at module top: the notebooks adjust
    # sys.path before importing it, and a module-top import would fix
    # the choice before the notebook had its say
    import camb
    from camb import model

    # each lambda is a one-line function: called below with explicit
    # arguments, they keep the unit conversions readable and in one
    # place (omegach2 subtracts the massive-neutrino share from the
    # cold component)
    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    # the overall boost hands one third of itself to the CAMB side;
    # the CAMB-side settings then scale together
    CAMBAccuracyBoost = CAMBAccuracyBoost*(1.0 + (AccuracyBoost-1.0)/3.0)
    lens_potential_accuracy = lens_potential_accuracy*CAMBAccuracyBoost
    kmax = kmax*(1.0 + 3*(CAMBAccuracyBoost-1))
    k_per_logint = int(k_per_logint) + int(3*(CAMBAccuracyBoost-1))
    extrap_kmax=2.5e2*CAMBAccuracyBoost

    # cosmolike aborts on non-monotone chi(z) grids (duplicate segment
    # endpoints), so interior segments drop their endpoint
    # (endpoint=False: the last point of the segment is left out and
    # the next segment supplies it); same grid as
    # likelihood/_cosmolike_prototype_base.py
    tmp=int(1000 + 250*CAMBAccuracyBoost)
    z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                  np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                  np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    len_z_interp_1D = len(z_interp_1D)

    # z nodes of the 2D power-spectrum tables handed to cosmolike,
    # which interpolates LINEARLY in z between exactly these nodes.
    # The dyadic factor m = 2^ceil(log2(boost)) refines each uniform
    # block with the same endpoints, so a boost increase is a true
    # refinement (the O(dz^2) interpolation residual falls like 1/m^2
    # instead of re-phasing); same grid as
    # likelihood/_cosmolike_prototype_base.py.
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

    log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*CLAccuracyBoost*AccuracyBoost))
    len_log10k_interp_2D = len(log10k_interp_2D)

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
    pars.set_dark_energy(w=w, wa=wa(w0pwa, w), dark_energy_model='ppf');
    pars.NonLinear = model.NonLinear_both
    pars.set_matter_power(redshifts=z_interp_2D_camb, kmax=kmax, silent=True);

    results = camb.get_results(pars)

    PKL  = results.get_matter_power_interpolator(var1="delta_tot",
                                                 var2="delta_tot",
                                                 nonlinear=False,
                                                 extrap_kmax=extrap_kmax,
                                                 hubble_units = False,
                                                 k_hunit = False);
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
        # z < 10 and rejects anything above
        kbt, tmp_bt = euclidemu2.get_boost(params,z_interp_2D[z_interp_2D < 10.0],10**np.linspace(-2.0589,0.973,len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # interp1d builds a function that linearly interpolates the
        # boost in log10(k); calling it on the shifted grid resamples
        # the boost onto the table's k nodes in h/Mpc
        tmp = interp1d(np.log10(kbt),
                        np.log(bt),
                        axis=1,
                        kind='linear',
                        fill_value='extrapolate',
                        assume_sorted=True)(log10k_interp_2D-np.log10(H0/100.)) #h/Mpc
        # boolean-mask assignment: every column whose k (in h/Mpc)
        # sits below EE2's trained range is set to zero boost
        tmp[:,10**(log10k_interp_2D-np.log10(H0/100)) < 8.73e-3] = 0.0
        lnbt = np.zeros((len_z_interp_2D, len_log10k_interp_2D))
        lnbt[z_interp_2D < 10.0, :] = tmp
        # Use Halofit first that works on all redshifts
        lnPNL = np.log(PKNL.P(z_interp_2D, np.power(10.0, log10k_interp_2D)).flatten(order='F')) + np.log((H0/100.0)**3)
        # on z < 10.0, replace it with EE2. np.where(cond, a, b) picks
        # a where cond is True and b where it is False, row by row:
        # the (z < 10)[:, None] adds a length-1 axis so the z condition
        # broadcasts across every k column of the reshaped tables
        lnPNL = np.where((z_interp_2D<10)[:,None], lnPL.reshape(len_z_interp_2D, len_log10k_interp_2D, order='F') + lnbt,
                                                   lnPNL.reshape(len_z_interp_2D, len_log10k_interp_2D, order='F')).ravel(order='F')
    elif non_linear_emul == 2:
        lnPNL = np.log(PKNL.P(z_interp_2D, np.power(10.0, log10k_interp_2D)).flatten(order='F')) + np.log((H0/100.0)**3)
    # only AFTER the tables were evaluated: shift the k grid to h/Mpc
    log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)

    # growth factor from the linear P(k) at one large scale
    # (k = 5e-4/Mpc), normalized to 1 at the last (highest-z) node
    G_growth = np.sqrt(PKL.P(z_interp_2D, 0.0005)/PKL.P(0, 0.0005))*(1 + z_interp_2D)
    G_growth = G_growth/G_growth[len(G_growth)-1]

    chi = results.comoving_radial_distance(z_interp_1D) * (H0/100.)

    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi)
