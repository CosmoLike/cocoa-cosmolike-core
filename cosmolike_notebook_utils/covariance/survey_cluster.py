"""Prepare the catalog geometry of a cluster 6x2pt plus counts forecast.

Clusters have two distinct normalizations. A count adds every selected
object, whereas a cluster density contrast divides by the selected
catalog's mean number per steradian. These helpers obtain both from the
same radial samples. The six two-point families are shear-shear,
galaxy-shear, galaxy-galaxy, cluster-galaxy, cluster-cluster and
cluster-shear. Counts are inserted between cluster-galaxy and
cluster-cluster, as in the joint DES likelihood.

The initialized model must have massless neutrinos, lognormal richness
selection, no environmental selection correction and no cluster
magnification. All-pairs spectra follow the hybrid DES mean model:
linearly biased nonlinear matter for cluster clustering and cluster-galaxy;
cluster lensing also contains the selected halo's own mass profile.
These preparation helpers alone do not return a covariance matrix.
"""

import numpy as np

from .survey import observable_rows


def observable_layout(nlens, nsource, ncluster_z, nrichness, cg_lens_bin,
                      nbin, excluded_gammat=()):
    """Describe all measured rows and the insertion positions of counts.

    Internal fields put galaxies first, then clusters, then sources. The
    cluster category index is redshift*nrichness+richness; its field ID
    adds nlens. Different richness categories at the same redshift have
    measured cross clustering; cross-redshift cluster spectra remain
    internal Gaussian inputs even when absent from the measured vector.

    Arguments:
        nlens, nsource, ncluster_z, nrichness = positive integer bin counts.
        cg_lens_bin = integer [ncluster_z], the galaxy bin paired with each
            cluster redshift bin in its nrichness cluster-galaxy rows,
            between zero and nlens-1.
        nbin = positive integer number of angular bins per two-point row.
        excluded_gammat = measured galaxy-source exclusions, as in
            survey.observable_rows; these do not cut internal spectra.
    Returns:
        Dict with rows int32 [nrow,3] (probe,field_A,field_B),
        two_point_positions int [nrow*nbin], count_positions int [ncount],
        and cluster_lensing_positions int [ncluster_z*nsource*nrichness,nbin],
        whose rows run over cluster redshift, then source, then richness.
        Positions refer to the full vector, including the
        ncount=ncluster_z*nrichness counts. Probe codes are 0 xi+, 1 xi-,
        2 tangential shear and 3 clustering.
    Raises:
        ValueError for invalid dimensions or galaxy-bin assignments.
    """
    for name, value in (("nlens", nlens), ("nsource", nsource),
                        ("ncluster_z", ncluster_z), ("nrichness", nrichness),
                        ("nbin", nbin)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name}={value} must be a positive integer")
    matched = np.asarray(cg_lens_bin)
    if (matched.shape != (ncluster_z,) or matched.dtype.kind not in "iu"
            or np.any(matched < 0) or np.any(matched >= nlens)):
        raise ValueError(
            "cg_lens_bin needs one valid integer galaxy bin per cluster z bin"
        )
    ncount = ncluster_z*nrichness
    rows = observable_rows(nlens=nlens, nsource=nsource,
                           excluded_gammat=excluded_gammat)

    # Inserting cluster fields shifts every source ID. It does not change
    # the galaxy/shear measurement order or remove crossed Gaussian inputs.
    fields = rows[:, 1:]
    fields[fields >= nlens] += ncount
    rows = rows.tolist()
    for redshift, galaxy in enumerate(matched):
        for richness in range(nrichness):
            cluster = nlens+redshift*nrichness+richness
            rows.append([3, cluster, int(galaxy)])
    # Counts follow the cluster-galaxy rows, as in the joint DES order
    # ss, gs, gg, cg, N (counts), cc, cs.
    insertion = len(rows)*nbin

    for redshift in range(ncluster_z):
        for first in range(nrichness):
            for second in range(first, nrichness):
                cluster_first = nlens+redshift*nrichness+first
                cluster_second = nlens+redshift*nrichness+second
                rows.append([3, cluster_first, cluster_second])

    lensing_start = len(rows)
    for redshift in range(ncluster_z):
        for source in range(nsource):
            for richness in range(nrichness):
                cluster = nlens+redshift*nrichness+richness
                rows.append([2, cluster, nlens+ncount+source])
    rows = np.array(rows, dtype=np.int32)
    two_point = np.arange(len(rows)*nbin)
    two_point[two_point >= insertion] += ncount
    lensing = two_point[lensing_start*nbin:].reshape(-1, nbin)
    return {
        "rows": rows,
        "two_point_positions": two_point,
        "count_positions": np.arange(insertion, insertion+ncount),
        "cluster_lensing_positions": lensing,
    }


def selected_windows(interface, geometry):
    """Return absolute cluster densities and normalized projected windows.

    At each distance chi, n_i=phi_i(z)*n_richness(a) is the number per
    comoving volume assigned to observed category i. Its integral over
    f_K^2 dchi is nbar_i per steradian; f_K is the transverse comoving
    distance, equal to chi only without spatial curvature. The
    density-contrast window is q_i=f_K^2 n_i/nbar_i, so integral
    q_i dchi=1. The fixed-selection abundance response is B_i=n_i*b_i.
    Counts retain B_i itself; the observed-mean response of a density
    contrast uses f_K^2 B_i/nbar_i.

    Arguments:
        interface = initialized cluster project exposing phi_cluster,
            ncl_richness, bcl_richness and covariance_project.
        geometry = float [4,nstate] (a,chi,f_K,dchi), lengths in c/H0.
    Returns:
        Dict with density, derivative [ncount,nstate] in (c/H0)^-3;
        window [ncount,nstate] in (c/H0)^-1; dimensionless bias of the same
        shape; number_per_sr [ncount] in sr^-1; and ncluster_z, nrichness
        integers. Category order is redshift then richness. No model
        setting changes; the C readers may build their cached tables.
    Raises:
        ValueError for nonfinite/invalid geometry, a nonfinite or negative
        selected abundance, a nonfinite response or an empty selected bin.
    """
    geometry = np.asarray(geometry, dtype=float)
    if (geometry.ndim != 2 or geometry.shape[0] != 4
            or geometry.shape[1] == 0 or not np.all(np.isfinite(geometry))
            or np.any(geometry[0] <= 0.0) or np.any(geometry[0] >= 1.0)
            or np.any(geometry[2:] <= 0.0)):
        raise ValueError("geometry needs finite [4,nstate], 0<a<1 and positive f_K,dchi")
    a = np.ascontiguousarray(geometry[0])
    phi = np.asarray(interface.phi_cluster(z=1.0/a-1.0)).T
    abundance = np.asarray(interface.ncl_richness(a=a)).T
    bias = np.asarray(interface.bcl_richness(a=a)).T
    ncluster_z = len(phi)
    nrichness = len(abundance)

    # Radial assignment and richness selection enter once, before any
    # catalog normalization. Multiplying two normalized q windows would
    # instead compute a two-point projection, not an absolute count.
    density = phi[:, None, :]*abundance[None, :, :]
    derivative = density*bias[None, :, :]
    density = np.ascontiguousarray(density.reshape(-1, len(a)))
    derivative = np.ascontiguousarray(derivative.reshape(-1, len(a)))
    if (not np.all(np.isfinite(density)) or np.any(density < 0.0)
            or not np.all(np.isfinite(derivative))):
        raise ValueError(
            "selected abundance must be finite/nonnegative, response finite"
        )
    unity = np.ones(shape=(1, len(a)))
    per_sr = interface.covariance_project(
        left=density, right=unity, weight=geometry[2]**2*geometry[3],
    )[:, 0]
    if np.any(per_sr <= 0.0):
        raise ValueError("a selected catalog is empty: check its bins and radial coverage")
    window = density*geometry[2]**2/per_sr[:, None]
    return {
        "density": density,
        "derivative": derivative,
        "window": np.ascontiguousarray(window),
        "bias": np.tile(bias, (ncluster_z, 1)),
        "number_per_sr": per_sr,
        "ncluster_z": ncluster_z,
        "nrichness": nrichness,
    }


def all_pairs_spectra(interface, ell, snapshot, catalogs):
    """Add every cluster cross spectrum to a galaxy/shear Limber snapshot.

    The hybrid model uses b_c b_g P_nonlinear and b_c b_c' P_nonlinear
    for cluster-galaxy and cluster-cluster spectra. Cluster-shear also
    includes the cluster's own selected NFW profile, P_cm^1h. The C
    projection attaches one harmonic shear factor per source leg.
    No white noise or real-space convention conversion is added here.

    Arguments:
        interface = initialized cluster project exposing covariance_power,
            pcm_1h_richness and covariance_cluster_spectra.
        ell = float [nell] multipoles >= 2, the values used to construct
            snapshot.
        snapshot = covariance_limber_spectra output on these exact multipoles.
        catalogs = selected_windows output on the snapshot's radial geometry.
    Returns:
        Owned float [nell,nfield,nfield] dimensionless angular spectra,
        with nfield=nlens+ncount+nsource. Field order is galaxy, cluster
        (redshift then richness), source. All cross-redshift and
        cross-richness spectra are retained.
    """
    geometry = snapshot['geometry']
    a, unused, distance, dchi = geometry
    nlens = snapshot['nlens']
    base = snapshot['windows']
    base_window = np.concatenate((base[0, :nlens], base[1, nlens:]))
    nbase = len(base_window)
    ncount = len(catalogs['window'])
    nrichness = catalogs['nrichness']
    richness = np.tile(np.arange(nrichness, dtype=np.int32),
                        catalogs['ncluster_z'])
    result = np.empty((len(ell), nbase+ncount, nbase+ncount))
    base_positions = np.concatenate((np.arange(nlens),
                                      np.arange(nlens+ncount, nbase+ncount)))
    cluster_positions = np.arange(nlens, nlens+ncount)
    result[:, base_positions[:, None], base_positions] = snapshot['spectra']

    # Only the projection needs the full k-by-distance table. Feed it
    # 1024 multipoles at a time so a large accuracy boost does not retain
    # all profile samples at once. This block size changes memory use,
    # not integration nodes, sum order or the physical approximation.
    for start in range(0, len(ell), 1024):
        modes = np.ascontiguousarray(ell[start:start+1024])
        power = np.empty((len(modes), len(a)))
        profile = np.empty((nrichness, len(modes), len(a)))
        for node, scale in enumerate(a):
            # Limber: each multipole samples k=(ell+1/2)/f_K on this shell.
            wave = np.ascontiguousarray((modes+0.5)/distance[node])
            power[:, node] = interface.covariance_power(a=scale, k=wave, linear=False)
            # pcm_1h_richness returns [k,1,nrichness]; store [nrichness,k].
            own = interface.pcm_1h_richness(k=wave, a=np.array([scale]))
            profile[:, :, node] = np.asarray(own)[:, 0, :].T
        projected = interface.covariance_cluster_spectra(
            ell=modes, distance=np.ascontiguousarray(distance),
            dchi=np.ascontiguousarray(dchi), base=base_window,
            window=catalogs['window'], bias=catalogs['bias'], power=power,
            profile=profile, richness=richness, nlens=nlens,
        )
        block = result[start:start+len(modes)]
        block[:, cluster_positions[:, None], base_positions] = projected['cluster_base']
        block[:, base_positions[:, None], cluster_positions] = (
            projected['cluster_base'].transpose(0, 2, 1)
        )
        block[:, cluster_positions[:, None], cluster_positions] = (
            projected['cluster_cluster']
        )
    return result
