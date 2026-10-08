"""Hybrid-emulator minimization, profile and Nautilus runs for one project.

In a hybrid example (use_emulator: 2 in every likelihood), trained emulators
replace the Boltzmann code for the background expansion and the matter power
spectra; CosmoLike still computes the survey projections. Each entry point
reads its project's evaluate YAML. Its sampler.evaluate.override block is
the fiducial point: every mode evaluates it first, --check stops there, and
a minimization starts from it.

The score every mode works with is -2 log posterior: the chi2-like quantity
-2 (log prior + log likelihood). Smaller is better, and differences in it
are the units of the chi2 comparisons used across these projects.

MODES

minimize   Search for the smallest -2 log posterior with emcee, an ensemble
           sampler: a set of points called walkers moves through parameter
           space, and each proposed move is built from the positions of
           other walkers, so steps automatically follow the lengths and
           directions of the cloud (no hand-tuned step size). Two move
           types are mixed: a differential-evolution (DE) move steps along
           the difference of two other walkers (80%), and a DE snooker move
           uses three other walkers to build the step (20%), which handles
           long, curved degeneracies better.

           The search is annealed. Sampling the posterior raised to the
           power 1/T (temperature T < 1) concentrates the walkers near the
           highest-posterior region; the smaller T is, the more the
           ensemble behaves like a hill climber. Each stage of the ladder
           T = 1.0, 0.25, 0.1, 0.005, 0.001 restarts the walkers around
           the best point found so far, with a proposal cloud of
           covariance (parameter covariance)*T/3, and keeps the best point
           ever seen: wide, warm stages can still cross between nearby
           basins, and cold stages polish the minimum.

profile    Fix one sampled parameter at each value of a grid and rerun the
           same annealed search over the remaining parameters. The curve
           of minimized scores against the fixed value is a profile of the
           posterior; priors are included, so it is not a pure likelihood
           profile. The grid is centered on a saved minimization result
           (--minfile), which this mode requires and verifies.

nautilus   Nested sampling with the Nautilus package: a set of live points
           shrinks from the full prior toward high likelihood, estimating
           the Bayesian evidence and weighted posterior samples; neural
           networks learn the likelihood boundary to propose new points.
           Cobaya's independent one-parameter priors are sampled directly
           by Nautilus; any external (joint) prior factor enters the
           likelihood exactly once instead.

OPTIONS (all modes)

--input    Evaluate YAML to read (default: the project's
           EXAMPLE_EMUL2_EVALUATE<n>.yaml).
--root     Folder whose chains/ subfolder receives every output file
           (default: the project folder).
--outroot  Basename of the output files (default: EXAMPLE_EMUL2_<MODE><n>).
           The run refuses to overwrite an existing record.
--check    Evaluate the YAML fiducial, print the sampled-parameter order
           and both scores, then stop without sampling.
--seed     Random seed for the walker draws and for Nautilus (default 42).

OPTIONS (minimize and profile)

--nstw     emcee steps per walker in each temperature stage (default 200).
--cov      Text file with a sampled-parameter covariance; its header must
           list the parameter names in the printed sampled order. Without
           it, the prior covariance is used.

OPTIONS (profile)

--profile  Sampled parameter to fix, by name or zero-based index
           (default "1", the second sampled parameter).
--factor   Half-width of the profile grid, in units of the parameter's
           proposal standard deviation (default 1).
--numpts   Number of grid points before the saved minimum's own value is
           merged in (default 11).
--minfile  JSON record written by the matching minimization run; required
           outside --check.

OPTIONS (nautilus)

--nlive    Live points (default 1000). --maxfeval caps likelihood calls
           (default 100000). --neff is the target effective posterior
           sample count (default 10000). --flive is the stopping fraction
           of evidence in the live set (default 0.01). --nnetworks is the
           number of boundary networks (default 4).

RUNNING

Serial:    python EXAMPLE_EMUL2_MINIMIZE1.py --check
MPI:       mpirun -n 4 python EXAMPLE_EMUL2_MINIMIZE1.py
Outputs land in <root>/chains/<outroot>.json plus the mode's own files
(.txt table, .<name>.txt profile rows, or Nautilus chain files).

MPI ranks own separate Cobaya models. Only parameter arrays and scalar
scores cross the worker pool; no model or compiled-library state is
pickled. Projects wrap this module in thin EXAMPLE_EMUL2_*.py shims that
call run(mode, project folder, example number).
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

# Set before importing numerical libraries or Cobaya on every MPI rank, since
# they read these variables when they load or first initialize. One BLAS thread
# per rank leaves the cores to the MPI ranks and CosmoLike's OpenMP threads; an
# empty CUDA_VISIBLE_DEVICES keeps the emulators on the CPU; COBAYA_NOMPI makes
# each rank's Cobaya act as one process, because every rank owns its own model.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["COBAYA_NOMPI"] = "1"

import numpy as np

# Each MPI process installs its own model before entering the worker loop.
_MODEL = None


def evaluate(model, values):
    """Return log prior and log likelihood for one ordered sampled point."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("sampled parameters must be finite")
    point = dict(zip(model.parameterization.sampled_params(), values, strict=True))
    prior = float(model.logprior(params_values=point))
    if prior == -np.inf:
        return prior, -np.inf
    # Hybrid background emulators need the derived-parameter store enabled.
    likelihood, _ = model.loglike(
        params_values=point, cached=False, return_derived=True)
    if not np.isfinite(prior) or np.isnan(likelihood) or likelihood == np.inf:
        raise ValueError("the hybrid model returned an invalid score")
    return prior, float(likelihood)


def log_probability(values, fixed, fixed_value, temperature):
    """Evaluate the tempered posterior, restoring a fixed profile coordinate."""
    if fixed >= 0:
        values = np.insert(values, fixed, fixed_value)
    prior, likelihood = evaluate(model=_MODEL, values=values)
    return (prior+likelihood)/temperature


def nested_likelihood(values):
    """Exclude the independent prior already represented by Nautilus.

    Returns log likelihood plus any external prior factor: Cobaya's full log
    prior minus its independent one-parameter part, which Nautilus samples.
    """
    prior, likelihood = evaluate(model=_MODEL, values=values)
    if prior == -np.inf:
        return -np.inf
    independent = float(_MODEL.prior.logps_internal(np.asarray(values)))
    return likelihood+prior-independent


def minimize(start, covariance, steps, pool, rng, fixed=-1):
    """Minimize -2 log posterior by annealed emcee sampling.

    Each temperature stage draws walkers around the best point so far with
    covariance covariance*temperature/3, samples the posterior raised to
    1/temperature and keeps the best point ever seen. Dividing by 3 keeps
    the starting cloud well inside the region the stage should refine,
    while the tempered posterior still lets walkers climb outward. A
    profile search starts its ladder at 0.3 rather than 1.0 because the
    outer minimization already located the basin; the remaining stages
    only need to re-polish after one coordinate moved. These ladders, the
    /3 width and the 80/20 mixture of differential-evolution (DE) and DE
    snooker moves are those of Cocoa's
    projects/example/EXAMPLE_PROFILE1.py.

    Arguments:
        start = full sampled point [n_sampled], in the model's sampled order.
        covariance = [n_sampled, n_sampled] covariance in that order; the fixed
            row and column are removed before walkers are drawn.
        steps = emcee steps per walker in each temperature stage (--nstw).
        pool = schwimmbad MPIPool on the master rank, or None for a serial run.
        rng = numpy Generator that draws the starting walkers.
        fixed = index of the parameter held at start[fixed], or -1 for none.
    Returns:
        (best point [n_sampled] including any fixed value, its -2 log posterior)
    """
    import emcee

    fixed_value = 0.0
    center = np.array(start, dtype=float)
    if fixed >= 0:
        fixed_value = center[fixed]
        center = np.delete(center, fixed)
        covariance = np.delete(np.delete(covariance, fixed, axis=0), fixed, axis=1)
    dimension = len(center)
    if dimension == 0:
        return np.array(start, dtype=float), -2*sum(evaluate(_MODEL, start))
    # emcee's snooker move splits the walkers into four groups and draws one
    # walker from each of the other three, so even a one-parameter profile needs
    # several; eight leave two per group. The usual 3*dimension remains for
    # large models.
    walkers = max(8, 3*dimension)
    # At least one walker per MPI rank; pool.size counts only the worker ranks.
    if pool is not None:
        walkers = max(walkers, pool.size+1)
    best = center.copy()
    best_score = log_probability(best, fixed, fixed_value, 1.0)
    temperatures = (1.0, 0.25, 0.1, 0.005, 0.001)
    if fixed >= 0:
        temperatures = (0.3, 0.1, 0.005, 0.001)
    for temperature in temperatures:
        initial = rng.multivariate_normal(
            mean=best, cov=covariance*temperature/3.0, size=walkers)
        # Draw within the prior so an invalid starting ensemble fails clearly.
        # A for loop's else branch runs only if the loop ends without break:
        # here, after 1000 non-finite draws for one walker.
        for row in range(walkers):
            for attempt in range(1000):
                if np.isfinite(log_probability(initial[row], fixed, fixed_value, 1.0)):
                    break
                initial[row] = rng.multivariate_normal(
                    mean=best, cov=covariance*temperature/3.0)
            else:
                raise ValueError("cannot initialize walkers; reduce the proposal covariance")
        sampler = emcee.EnsembleSampler(
            nwalkers=walkers, ndim=dimension, log_prob_fn=log_probability,
            args=(fixed, fixed_value, temperature), pool=pool,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
        # A cold stage draws nearly identical walkers, which emcee's linear
        # independence check would reject even though the DE moves re-spread
        # the ensemble; the explicit finite-score loop above already ran.
        sampler.run_mcmc(initial_state=initial, nsteps=steps,
                         skip_initial_state_check=True)
        # emcee stores (log prior + log likelihood)/temperature; multiplying by
        # the temperature restores the log posterior, so all stages share one scale.
        scores = sampler.get_log_prob(flat=True)*temperature
        index = int(np.argmax(scores))
        if scores[index] > best_score:
            best = sampler.get_chain(flat=True)[index].copy()
            best_score = float(scores[index])
        print(f"temperature={temperature:g}, -2 log posterior={-2*best_score:.8g}",
              flush=True)
    if fixed >= 0:
        best = np.insert(best, fixed, fixed_value)
    return best, -2*best_score


def parser_for(mode, project, example):
    """Use the same public command options in every survey project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=project/f"EXAMPLE_EMUL2_EVALUATE{example}.yaml")
    parser.add_argument("--root", type=Path, default=project)
    parser.add_argument("--outroot", default=f"EXAMPLE_EMUL2_{mode.upper()}{example}")
    parser.add_argument("--check", action="store_true",
                        help="evaluate the YAML fiducial and print parameter order; no sampling")
    parser.add_argument("--seed", type=int, default=42)
    if mode in ("minimize", "profile"):
        parser.add_argument("--nstw", type=int, default=200)
        parser.add_argument("--cov", type=Path,
                            help="sampled-parameter covariance with ordered names in its header")
    if mode == "profile":
        parser.add_argument("--profile", default="1", help="sampled parameter name or index")
        parser.add_argument("--factor", type=float, default=1.0)
        parser.add_argument("--numpts", type=int, default=11)
        parser.add_argument("--minfile", type=Path, required=False,
                            help="JSON record written by the matching minimization example")
    if mode == "nautilus":
        parser.add_argument("--nlive", type=int, default=1000)
        parser.add_argument("--maxfeval", type=int, default=100000)
        parser.add_argument("--neff", type=int, default=10000)
        parser.add_argument("--flive", type=float, default=0.01)
        parser.add_argument("--nnetworks", type=int, default=4)
    return parser


def load_model(filename):
    """Load the shared evaluate configuration on CPU without writing its outputs.

    Refuses (ValueError) a likelihood without use_emulator: 2, an mnu other
    than the fixed 0.06 eV the emulator networks assume, and a
    sampler.evaluate.override block that misses a sampled parameter.

    Arguments:
        filename = Path of the project's evaluate YAML.
    Returns:
        (Cobaya model, sampled parameter names in sampled order,
         start point [n_sampled] from sampler.evaluate.override,
         SHA-256 hex digest of the YAML bytes, which ties a profile to the
         minimization it starts from)
    """
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(str(filename))
    point = info.get("sampler", {}).get("evaluate", {}).get("override", {})
    info.pop("sampler", None)
    info.pop("output", None)
    for options in info["likelihood"].values():
        if options.get("use_emulator") != 2:
            raise ValueError("hybrid examples require use_emulator: 2 in every likelihood")
        options["print_datavector"] = False
    mass = info["params"]["mnu"]
    if isinstance(mass, dict):
        mass = mass.get("value")
    if mass != 0.06:
        raise ValueError("these hybrid networks require fixed mnu=0.06 eV")
    if "emulbaosn" in info["theory"]:
        info["theory"]["emulbaosn"]["extra_args"]["device"] = "cpu"
    model = get_model(info)
    names = list(model.parameterization.sampled_params())
    missing = [name for name in names if name not in point]
    if missing:
        raise ValueError(f"sampler.evaluate.override must specify {missing}")
    start = np.array([point[name] for name in names], dtype=float)
    fingerprint = hashlib.sha256(filename.read_bytes()).hexdigest()
    return model, names, start, fingerprint


def run_nautilus(model, pool, args, prefix, names):
    """Sample with each exact independent prior and save weighted GetDist rows.

    Arguments:
        model = Cobaya model; its prior.pdf gives the one-parameter priors.
        pool = schwimmbad MPIPool on the master rank, or None.
        args = parsed options: nlive, nnetworks, seed, flive, neff, maxfeval.
        prefix = output path without extension.
        names = sampled parameter names in sampled order.
    Returns:
        dict for the JSON run record: converged, log_evidence, evaluations,
        posterior_samples and effective_samples.
    Side effects:
        Writes <prefix>_checkpoint.hdf5. When posterior samples exist, also
        writes <prefix>.1.txt (weight, -log likelihood including any external
        prior factor, then the sampled parameters) and <prefix>.paramnames.
    """
    from nautilus import Prior, Sampler

    prior = Prior()
    for name, distribution in zip(names, model.prior.pdf, strict=True):
        prior.add_parameter(key=name, dist=distribution)
    # pool=(pool, None): the MPI workers evaluate likelihoods; Nautilus's own
    # calculations, such as network training, stay on this rank.
    sampler = Sampler(
        prior=prior, likelihood=nested_likelihood, pass_dict=False,
        pool=(pool, None), n_live=args.nlive, n_networks=args.nnetworks,
        filepath=str(prefix)+"_checkpoint.hdf5", resume=False, seed=args.seed)
    converged = sampler.run(
        f_live=args.flive, n_eff=args.neff, n_like_max=args.maxfeval,
        verbose=True, discard_exploration=True)
    if sampler.n_eff == 0:
        print("Budget exhausted before retaining posterior samples; "
              "the checkpoint is saved, but there is no posterior chain.", flush=True)
        return {"converged": False, "log_evidence": None,
                "evaluations": int(sampler.n_like), "posterior_samples": 0,
                "effective_samples": 0.0}
    points, log_weight, log_like = sampler.posterior()
    np.savetxt(str(prefix)+".1.txt",
               np.column_stack((np.exp(log_weight), -log_like, points)),
               header="weight neg_log_likelihood "+" ".join(names))
    params = model.info()["params"]
    with open(str(prefix)+".paramnames", "w") as output:
        for name in names:
            label = params[name].get("latex", name)
            output.write(f"{name} {label}\n")
    return {"converged": bool(converged), "log_evidence": float(sampler.log_z),
            "evaluations": int(sampler.n_like), "posterior_samples": len(points),
            "effective_samples": float(sampler.n_eff)}


def run(mode, project, example=1):
    """Run a serial or MPI example, with one initialized model per process.

    Arguments:
        mode = "minimize", "profile" or "nautilus".
        project = project directory, a Path or string.
        example = number in the default input EXAMPLE_EMUL2_EVALUATE<example>.yaml.
    Side effects:
        Parses the command line and sets the module's _MODEL. Except with
        --check, writes <root>/chains/<outroot>.json and the mode's result
        files, refusing to start when that JSON or a Nautilus checkpoint exists.
    """
    global _MODEL
    parser = parser_for(mode=mode, project=Path(project), example=example)
    args = parser.parse_args()
    # getattr gives 1 for an option this mode does not define, so the check passes it.
    for key in ("nstw", "numpts", "factor", "nlive", "maxfeval", "neff", "nnetworks"):
        value = getattr(args, key, 1)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{key} must be positive")
    if mode == "profile" and not args.check and args.minfile is None:
        parser.error("--minfile is required: first run the matching minimization")
    if mode == "nautilus" and not 0 < args.flive < 1:
        parser.error("--flive must lie between zero and one")
    # Each emcee sampler copies NumPy's global random state when it is built; rng
    # draws the starting walkers; Nautilus receives the seed directly.
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)
    _MODEL, names, start, fingerprint = load_model(filename=args.input)
    # Open MPI, the MPI of the Cocoa environment, exports OMPI_COMM_WORLD_SIZE to
    # every rank. A plain python run lacks it and stays serial: nullcontext then
    # yields pool = None.
    ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    context = nullcontext(None)
    if ranks > 1:
        from schwimmbad import MPIPool
        context = MPIPool()
    with context as pool:
        # Worker ranks serve evaluations here until the master closes the pool.
        if pool is not None and not pool.is_master():
            pool.wait()
            return
        prior, likelihood = evaluate(model=_MODEL, values=start)
        if not np.isfinite(prior+likelihood):
            raise ValueError("the YAML fiducial has a non-finite prior or likelihood")
        print("Sampled parameter order:", dict(enumerate(names)), flush=True)
        print(f"Fiducial log prior={prior:.10g}, log likelihood={likelihood:.10g}", flush=True)
        if args.check:
            _MODEL.close()
            return
        prefix = args.root/"chains"/args.outroot
        if Path(str(prefix)+".json").exists() or Path(str(prefix)+"_checkpoint.hdf5").exists():
            raise FileExistsError(f"{prefix} already exists; select a new --outroot")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        record = {"input": str(args.input.resolve()), "input_sha256": fingerprint,
                  "parameters": names, "mode": mode, "seed": args.seed,
                  "omp_threads": os.environ.get("OMP_NUM_THREADS"), "mpi_ranks": ranks}
        if mode == "nautilus":
            record.update(run_nautilus(
                model=_MODEL, pool=pool, args=args, prefix=prefix, names=names))
        else:
            covariance = _MODEL.prior.covmat(ignore_external=True)
            if args.cov is not None:
                header = args.cov.read_text().splitlines()[0].lstrip("# ").split()
                if header != names:
                    raise ValueError("--cov header must match the printed parameter order")
                covariance = np.loadtxt(args.cov)
            if covariance.shape != (len(names), len(names)):
                raise ValueError("proposal covariance has the wrong shape")
            if not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T):
                raise ValueError("proposal covariance must be finite and symmetric")
            # Cholesky raises LinAlgError unless the covariance is positive
            # definite; the factor itself is not used.
            np.linalg.cholesky(covariance)
            if mode == "minimize":
                best, score = minimize(start=start, covariance=covariance,
                                       steps=args.nstw, pool=pool, rng=rng)
                record.update({"point": best.tolist(), "minus2_log_posterior": score,
                               "nstw": args.nstw})
                np.savetxt(str(prefix)+".txt", [np.append(best, score)],
                           header=" ".join(names)+" minus2_log_posterior")
            else:
                minimum = json.loads(args.minfile.read_text())
                if minimum["input_sha256"] != fingerprint or minimum["parameters"] != names:
                    raise ValueError("minimum was produced with a different YAML or parameter order")
                start = np.array(minimum["point"], dtype=float)
                fixed = int(args.profile) if args.profile.isdigit() else names.index(args.profile)
                if not 0 <= fixed < len(names):
                    raise ValueError("profile index is outside the sampled parameter order")
                prior, likelihood = evaluate(model=_MODEL, values=start)
                # Re-evaluating the saved minimum must reproduce its saved score:
                # 0.02 in -2 log posterior is far below the 0.2 physics floor of
                # the chi2 tests, so this only trips when the YAML, emulator
                # networks or sampled-parameter order actually changed.
                if abs(-2*(prior+likelihood)-minimum["minus2_log_posterior"]) > 0.02:
                    raise ValueError("the saved minimum score no longer matches this model")
                # 0.999999 confidence turns an unbounded or very wide prior into
                # finite grid limits spanning essentially its full support.
                bounds = _MODEL.prior.bounds(confidence=0.999999)
                # The grid half-width is --factor proposal standard deviations
                # of the fixed parameter, clipped to the finite prior bounds.
                width = args.factor*np.sqrt(covariance[fixed, fixed])
                lower = max(bounds[fixed, 0], start[fixed]-width)
                upper = min(bounds[fixed, 1], start[fixed]+width)
                # np.unique sorts the grid, merging in the minimum's own value once.
                grid = np.unique(np.append(np.linspace(lower, upper, args.numpts), start[fixed]))
                rows = []
                for value in grid:
                    initial = start.copy()
                    initial[fixed] = value
                    best, score = minimize(start=initial, covariance=covariance,
                                           steps=args.nstw, pool=pool, rng=rng, fixed=fixed)
                    rows.append(np.concatenate(([value, score], best)))
                np.savetxt(str(prefix)+f".{names[fixed]}.txt", rows,
                           header=names[fixed]+" minus2_log_posterior "+" ".join(names))
                record.update({"profile": names[fixed], "grid": grid.tolist(),
                               "minfile": str(args.minfile), "nstw": args.nstw})
        Path(str(prefix)+".json").write_text(json.dumps(record, indent=2)+"\n")
    _MODEL.close()
