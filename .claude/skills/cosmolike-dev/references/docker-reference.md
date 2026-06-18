# Docker for ML/Scientific Computing — Reference

Methodology for building Docker images for ML and scientific-computing workloads,
distilled from real debugging sessions. Covers GPU passthrough, dependency
resolution, image size, and the writing conventions Vivian wants Claude to follow
when editing Dockerfiles.

The discipline is the same as for code review: theory + experiment + numbers,
not pattern-matching to a remembered pattern. When a `docker build` fails,
read the actual error before proposing a fix.

---

## Editing conventions Claude must follow

These are non-negotiable formatting rules for any Dockerfile fix or modification.

### 1. Always output the entire RUN block, never a snippet

When fixing or modifying a `RUN` block (or any multi-line shell instruction),
output the **complete block** as a single drop-in replacement. Never give a
patched fragment that the user has to splice into the rest by hand.

Same rule applies to multi-line `COPY` / `ENV` / `ARG` chains, and any instance
where partial output would force the user to reconstruct context themselves.

The cost of doing this is repeating ~20 lines per fix. The benefit is the
user can apply the fix in one paste, no thinking required.

### 2. Don't pattern-match to "common Dockerfile bugs"

Specific failure mode to avoid: claiming that comments between
backslash-continued lines break a `RUN`. They don't. BuildKit strips comment
lines (lines starting with `#`) before passing the joined command to the
shell. The following is valid and parses cleanly:

```dockerfile
RUN apt-get update \
# --- some commentary ---
  && apt-get install -y curl \
# --- more commentary ---
  && rm -rf /var/lib/apt/lists/*
```

The only related rule that *does* apply: a comment line itself cannot end with
`\` to continue. The continuation must be on a code line.

When in doubt, check Docker's parsing rules (or actually `docker build` the
file) before declaring a bug.

### 3. Honest retractions

If a previous claim turns out to be wrong, state the correction directly,
explain what was missed, and move on. No grovelling, no excessive
self-flagellation. Vivian wants accountability without the noise.

---

## GPU on Linux — the full stack

### Layer model

```
Host kernel + NVIDIA driver           ← determines max CUDA version reachable
└── NVIDIA Container Toolkit          ← passes GPU through to containers
    └── Docker daemon + runtime
        └── Container base image (CUDA runtime/devel)
            └── Pip-installed PyTorch/TF/JAX wheels  ← bundle their own CUDA libs
```

Each layer pins a CUDA version, and they must be **compatible top-to-bottom**.

- **Driver caps everything.** `nvidia-smi` shows "CUDA Version: X.Y" top-right —
  that's the *maximum* the driver supports, not what's installed.
- **Container CUDA ≤ Driver CUDA.** A `nvidia/cuda:13.x` base will not run on a
  driver that only supports CUDA 12.2. The NVIDIA Container Toolkit's hook
  enforces this at container-start time with a clear error.
- **Pip wheel CUDA ≤ Driver CUDA** by minor compat. PyTorch's `cu126` wheels
  work down to CUDA 12.0+ via CUDA forward compatibility at the runtime layer.

### Installing NVIDIA Container Toolkit

On Linux hosts where the toolkit isn't installed (`dpkg -l | grep
nvidia-container-toolkit` returns empty), the official path is to add NVIDIA's
apt repo, install, then optionally remove the repo to keep `apt update` clean:

```bash
# Add temporarily
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Remove the apt source so future apt-get update doesn't poll NVIDIA
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo rm /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt-get update
```

Do not direct-download the `.deb` from `github.com/NVIDIA/nvidia-container-toolkit/releases/latest/download/...`
— NVIDIA does not host the unified `.deb` as a GitHub release asset; the
"latest" shortcut 404s. The packages live in the libnvidia-container apt repo.

**Verify before launching real workloads:**

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

If this prints the host's `nvidia-smi` output, GPU passthrough is working.

### Shared / production hosts

For workstations or clusters that other users depend on, **never recommend
driver updates or kernel-level changes** without explicit user authorization.
Driver upgrades require reboots and can disrupt running jobs or break the host
outright. The right move is to align everything *above* the driver to its
capability ceiling.

For a host with old driver `D` supporting CUDA `C` max:
- Container base: `nvidia/cuda:C.x-...-ubuntu22.04`
- PyTorch wheel index: `cu<C-minor-as-int>` (e.g. cu121 for CUDA 12.1+)
- TensorFlow / JAX: pick a version whose bundled NVIDIA wheels target ≤ C
- Accept the resulting feature gap explicitly

---

## Base image selection

Choose the smallest tag that contains what the *image needs*, not what the
build host has.

| Tag suffix | Size | Contains | Use when |
|---|---|---|---|
| `-base` | ~150 MB | Driver shim only | Almost never useful for ML |
| `-runtime` | ~1 GB | CUDA runtime libs | Running prebuilt wheels (PyTorch ships bundled CUDA) |
| `-cudnn8-runtime` | ~2 GB | + cuDNN runtime | TF/JAX that look for system cuDNN before bundled |
| `-cudnn8-devel` | ~4 GB | + nvcc + headers | Compiling CUDA C/C++ in the container |

For a pure-Python ML notebook image, `-runtime` is almost always correct.
The PyTorch wheel doesn't look at `/usr/local/cuda` at import time — it loads
from `site-packages/torch/lib/`. Choosing `-devel` over `-runtime` wastes
~3 GB for no benefit unless you're compiling kernels.

The trade-off: `-runtime` removes `nvcc`, so custom CUDA extensions can't be
built in the container. For a research image that compiles native code
(CosmoLike, Cocoa), `-devel` may be right. For teaching / pure-Python
work, `-runtime` is right.

---

## Dependency resolution failures (TF + PyTorch)

The NVIDIA wheel ecosystem fractured around 2024. Each major ML framework now
bundles a specific CUDA minor version, and the pip resolver cannot unify them
when bundles disagree.

Real failure shape:

```
tensorflow[and-cuda] 2.18.0 depends on nvidia-cusolver-cu12==11.6.3.83
torch 2.5.1+cu121 depends on nvidia-cusolver-cu12==11.4.5.107
ERROR: ResolutionImpossible
```

This is **not** a transient resolver issue — both packages exact-pin and there
is no satisfying version. Three responses, in order of preference:

1. **Find a (TF, PyTorch) pair whose bundles align.** Sometimes a one-minor TF
   bump resolves it (e.g. TF 2.18 → 2.17). Try this first because both
   frameworks stay GPU-capable.

2. **Drop `[and-cuda]` from TF.** TF runs CPU-only, PyTorch keeps GPU. Pick
   this when GPU PyTorch matters more than GPU TF (almost always true for
   research that uses both).

3. **Bump the host driver / base / wheel index together.** Only an option on
   hosts where driver upgrades are safe.

The general rule: **on driver `D` constrained below the latest, you usually
cannot have both TF-GPU and PyTorch-GPU in the same env.** Accept this and
move on; don't burn hours trying every TF version.

---

## Image size — what's actually eating GB

Typical breakdown for a CUDA + PyTorch + TF + JAX scientific image:

| Component | Size | Notes |
|---|---|---|
| NVIDIA CUDA libraries | ~10–12 GB | cuDNN 700 MB, cuBLAS 400 MB, etc. — duplicated across torch + jax |
| Base image (-devel) | ~4 GB | -runtime saves ~3 GB if applicable |
| ML framework Python code | ~2 GB | torch wheel alone is 900 MB |
| Scientific Python stack | ~1.5 GB | numpy/scipy/pandas/sklearn/numba/matplotlib |
| System packages | ~3 GB | build-essential, cmake, LaTeX tools, etc. |
| Layer "ghost" weight | ~4–5 GB | apt caches from prior `RUN` blocks |

Total often lands at 25–30 GB for a "kitchen-sink" image.

### Leverage points

1. **Base image: `-runtime` not `-devel`** → ~3 GB
2. **Consolidate `RUN apt-get install` blocks**, end with `apt-get clean && rm
   -rf /var/lib/apt/lists/*` in the same block → ~1–2 GB
3. **Drop unused C dev headers** (libhdf5-dev, libgflags-dev, libleveldb-dev,
   libsnappy-dev, libgoogle-glog-dev, autoconf, automake, libtool, etc.) when
   the image only runs Python wheels → ~1.5 GB
4. **Drop one of {emacs, vim, nano}** — Jupyter is your editor
5. **LaTeX packages for matplotlib** — see the dedicated section below. The
   short version: KEEP them by default when matplotlib is in the image.
   `text.usetex=True` is the user's default and the build must support it.
6. **Drop `[and-cuda]` from TF if PyTorch already brings CUDA** → ~3 GB of
   duplicated NVIDIA stack
7. **Use `pip install --no-cache-dir`** inside the RUN

Realistic floor for "PyTorch GPU + TF (CPU) + JAX (CPU) + full scientific
stack" is ~14–15 GB. Going lower requires giving something up structurally.

For comparison: official `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
(PyTorch only, nothing else) is ~6 GB.

### Layer ghost weight, explained

Every `RUN` creates an image layer. Even if a later `RUN` deletes files, the
earlier layer still holds them. `apt-get clean` in a *separate* `RUN` is
useless — it cleans the current layer's cache, not the previous layer's.

Correct:

```dockerfile
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
```

Wrong:

```dockerfile
RUN apt-get update
RUN apt-get install -y --no-install-recommends curl wget
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# ← previous two layers still contain the cache
```

This is the single biggest source of bloat in real-world Dockerfiles.

---

## LaTeX for matplotlib (`text.usetex=True`)

> ⚠️ **DEFAULT ASSUMPTION**: if matplotlib is in the image, the build MUST
> support `rcParams['text.usetex'] = True`. This is the user's default
> matplotlib configuration. The right LaTeX packages must be installed and
> NOT cut for image size. **Only drop them if the user explicitly says
> usetex is off.**
>
> Whenever a Dockerfile that includes matplotlib is being built, edited, or
> trimmed, flag this:
>
> *"I'll assume `text.usetex=True` is on (default unless you tell me
> otherwise). That means cm-super + the TeX Live packages stay in. Confirm
> if you want them dropped."*
>
> Do not silently remove `cm-super`, `dvipng`, `ghostscript`, or any
> `texlive-*` package. These are non-negotiable defaults.

### Required apt packages for matplotlib usetex

The minimum set for matplotlib to render math expressions with LaTeX:

```dockerfile
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       cm-super \
       dvipng \
       ghostscript \
       texlive-latex-base \
       texlive-latex-extra \
       texlive-latex-recommended \
       texlive-fonts-recommended \
       texlive-fonts-extra \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
```

Size impact: ~1.5–2 GB for the full set. Cannot be meaningfully reduced
without breaking common LaTeX expressions matplotlib emits.

## UID handling for bind mounts

Docker Desktop on macOS silently translates UIDs at the bind-mount layer, so
images "just work" regardless of what UID the container user has. **On Linux,
this is not true.** Bind mounts pass straight through to the kernel; file
ownership is whatever `stat()` reports, and UID mismatches cause read-only
behavior or permission errors.

Symptom: Jupyter shows "Notebook is read-only" or `docker logs` shows
permission-denied writes to `/home/<user>/host/`.

Diagnosis:

```bash
docker exec -it <container> bash -c 'id; ls -la /home/<user>/host/'
```

If the container user UID (e.g. 1005) doesn't match the file owner UID
(e.g. 1000), you've found it.

Two valid fixes:

1. **Match container UID to host UID at build time.** Keep `ARG NB_UID=1000`
   as a default, override via `--build-arg` when building on hosts where the
   user UID differs:
   ```bash
   docker build --build-arg NB_UID=$(id -u) --build-arg NB_GID=$(id -g) -t myimage .
   ```

2. **Hard-code the right UID** if the image is single-user. Simpler but less
   portable.

Do NOT `chown` host files to match the container UID — that breaks other host
users' access and confuses anyone SSH'ing into the host.

### Common UID-related gotcha

A line like `groupmod -g 1005 users` inside the Dockerfile hard-codes a GID
that overrides whatever `ARG NB_GID` you set. Remove these stale hard-codes
when changing the ARG defaults.

---

## Other Linux-vs-Mac Docker differences worth knowing

These exist because Docker Desktop on Mac runs a VM with a translation layer
that masks several POSIX-strict behaviors:

| Behavior | macOS Docker Desktop | Linux Docker |
|---|---|---|
| Bind-mount UID handling | Translates transparently | POSIX-strict |
| Privileged port binding from non-root | Often allowed via VM | Blocked |
| `/var/run/docker.sock` access | Permissive | Requires `docker` group membership |
| `ulimit` defaults | VM-controlled | systemd-controlled |
| Filesystem performance on bind mounts | Slow (VirtioFS / gRPC-FUSE) | Native |

The Linux behavior is the POSIX reference. If something works only on Mac,
expect it to break the first time you deploy on a Linux host or HPC.

---

## Useful inspection commands

```bash
# What's the host driver / max CUDA?
nvidia-smi | head -3

# Is the toolkit installed?
dpkg -l | grep nvidia-container-toolkit

# Verify GPU passthrough at minimum CUDA level
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# What's actually in a built image (by layer)?
docker history myimage --no-trunc --format "table {{.Size}}\t{{.CreatedBy}}"

# What's the total size, and how much is recoverable?
docker system df

# Inside a running container: where is PyTorch loading CUDA from?
python -c "import torch; print(torch.cuda.is_available()); \
           import os; print(os.path.dirname(torch.__file__))"
ls /opt/venv/lib/python3.10/site-packages/torch/lib/   # bundled CUDA libs live here
```

---

## Verification at end of build

For any GPU ML image, the smoke test after build should be:

```python
import sys, platform
print("python   :", sys.version.split()[0])
print("arch     :", platform.machine())   # x86_64 on Linux

import torch
print("torch    :", torch.__version__)
print("torch CUDA avail:", torch.cuda.is_available())
print("torch CUDA ver  :", torch.version.cuda)
if torch.cuda.is_available():
  print("device           :", torch.cuda.get_device_name(0))

try:
  import tensorflow as tf
  print("TF GPUs  :", tf.config.list_physical_devices('GPU'))
except ImportError:
  pass

try:
  import jax
  print("JAX backend:", jax.default_backend())
except ImportError:
  pass

try:
  import h5py
  print("h5py     :", h5py.__version__, "with HDF5", h5py.version.hdf5_version)
except ImportError:
  pass
```

If `torch CUDA avail` is `False` despite `--gpus all`, the most common causes
in order of likelihood are:

1. Driver < base image CUDA — base image is too new for the host
2. Wheel installed without CUDA (e.g. `--index-url https://download.pytorch.org/whl/cpu`)
3. NVIDIA Container Toolkit not configured / Docker not restarted after configure
4. Wrong `--gpus` flag in the `docker run` invocation

---

## Patterns to avoid

- **Multiple `RUN apt-get install` separated by other RUNs.** Consolidate into
  one apt block when possible.
- **Hard-coding `groupmod -g <N> users`** when `ARG NB_GID` is supposed to be
  configurable.
- **`apt-get clean` and `rm -rf /var/lib/apt/lists/*` in their own RUN.** Must
  be in the same RUN as the install to actually reduce image size.
- **Mixing pip CUDA index URLs across RUN lines.** Pin everything to one
  index in a single pip install call.
- **Hosting the same package via both PyPI and a wheel index.** Pip may pick
  the CPU-only PyPI build over the CUDA wheel index build if version pin
  allows both. Use `--extra-index-url` carefully and check `torch.__version__`
  ends in `+cuXXX`.
- **Direct-downloading NVIDIA `.deb`s from GitHub releases.** They aren't
  there — packages live in the libnvidia-container apt repo only.

---

## When asked to fix a Dockerfile

1. Read the actual error first. Don't pattern-match.
2. Output the **entire** RUN block being modified (per memory rule).
3. Explain the minimum diff vs the previous version.
4. Estimate the size / time impact when it's interesting.
5. Honestly retract any prior claim that turned out to be wrong.
