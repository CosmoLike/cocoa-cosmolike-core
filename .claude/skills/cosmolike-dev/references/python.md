# Python in the Cosmolike repositories: the style contract

Python style is a condition for accepting a change, not a preference. This
contract applies to every Python file committed to a Cosmolike repository:

- `cosmolike_core`: `cosmolike_notebook_utils/` (the CAMB run, the
  data-vector plots, the response plots, the Fisher helpers) and
  `cocoa_testing.py` (the test machinery);
- every project (`projects/<name>/`): `likelihood/*.py`, the notebook
  wrappers in `interface/`, `tests/*.py`, `scripts/*.py`, the `EXAMPLE_*.py`
  sampler scripts, and the code cells of the `EXAMPLE_*.ipynb` notebooks.

It covers the code and every piece of text the code carries: comments,
docstrings, command help, printed diagnostics and error messages.

A correct result written in an unnecessarily difficult form is rejected.
Passing tests do not override a rule marked NO-GO below.

The C side has its own rules (SKILL.md, `patterns.md`, `pitfalls.md`). One
difference to remember: C files use 2-space indentation; Python files keep
the indentation of the file being edited (4 spaces in the existing Python).


## 1. The reader

The intended reader is a library user or a physics student who understands
C-like control flow but may not know advanced Python idioms. Code must be
easy to trace one operation at a time. Text must define a software term at
the point where the term first matters.

Words used below:

- A **cold path** is code that runs once or rarely: configuration,
  validation, set-up, file handling, command parsing, reporting, object
  construction, figure layout.
- A **hot path** is code that repeats many times or works on whole arrays:
  a vectorized numpy kernel, the per-evaluation body of a likelihood
  (`set_cosmo_related`, the data-vector assembly), a loop over thousands of
  table nodes.
- A **monkey patch** replaces existing executable behavior while Python is
  running: replacing an imported function or a method, changing
  `sys.modules`, `__defaults__`, `__code__` or `__class__`, and using
  `patch`, `patch.object`, `patch.dict` or pytest's `monkeypatch` fixture.
- **Module-global state** is a value defined at the top level of a Python
  file and shared by calls that do not receive it explicitly.


## 2. Before writing

1. Read every function or class the change touches in full, plus its nearby
   callers and its tests. A diff fragment is not enough.
2. Mark each changed path as cold or hot (definitions above).
3. Decide the code structure, the intermediate names, the argument style,
   the validation order, the error behavior and the docstrings before
   typing. Do not leave a choice between a compact idiom and an explicit
   form open: the explicit form is the answer on cold paths.
4. Match the file: its indentation, its naming, its docstring format, its
   comment density. A rename is its own commit.


## 3. Cold paths: explicit, C-like control flow

Each consequential operation is visible on its own line.

| Condition | GO | NO-GO |
|---|---|---|
| Loops | An explicit loop shows state changes and failure points. | A comprehension hides branching, mutation, validation, or more than one transformation. |
| Conditions | A normal `if` block, or one simple ternary that reads at a glance. | Nested ternaries, chained expression tricks, a condition buried in a long call. |
| Function calls | Named parameters identify meanings whenever the callee permits names. | Several unexplained positional arguments the reader must map by memory. |
| Staging | Named intermediate variables separate selection, conversion and validation. | One nested expression doing several operations with different failure modes. |
| Containers | A mapping or structured sequence with three or more items puts one item per line. | A packed literal that hides keys, values, shapes or units. |
| Names | Names state the scientific quantity, its representation, its unit or its role. | One-letter names outside conventional local mathematics; names that collide with physics symbols. |
| Errors | Validation happens before mutation, expensive set-up, or any file write. | A late check that permits partial state, or a silent fallback. |

Use an explicit loop instead of a comprehension when the loop validates,
mutates, logs, handles an exception, or does more than one transformation. A
short comprehension is acceptable only when the mapping is direct and the
result is clearer than the loop.

Stage operations with different meanings into named variables.

NO-GO:

```python
G_growth = np.sqrt(PKL.P(z[z <= z2D[-1]], 0.0005)/PKL.P(0, 0.0005))*(1 + z[z <= z2D[-1]])
```

GO (each step can fail on its own line, and an error can name it):

```python
z_growth = z_interp_1D[z_interp_1D <= z_interp_2D[-1]]
power_ratio = PKL.P(z_growth, 0.0005)/PKL.P(0, 0.0005)
G_growth = np.sqrt(power_ratio)*(1 + z_growth)
```

Forbidden when a plainer function, loop or temporary variable exists:

- assignment expressions using `:=`;
- nested comprehensions;
- several ternaries in one expression;
- a `lambda` that would read more clearly as a named function;
- starred unpacking that hides the source or the order of values;
- metaprogramming for ordinary configuration or validation;
- chained calls that combine selection, conversion and mutation;
- silent reads from mutable module-global state;
- names shortened, or logic collapsed, to make a change look smaller.


## 4. Hot paths: keep them vectorized

Readability work must not replace an array operation with slower
element-by-element Python. Vectorized numpy stays vectorized.

Dense numerical syntax is allowed only where it is needed for speed or
states the mathematics more directly. It still needs:

- descriptive names at the stable boundaries (inputs, outputs);
- a comment giving the mathematical reason or the shape invariant;
- the shapes, dtypes and units in the owning docstring;
- a timing before and after when the execution shape changes;
- a regression check of the numerical result.

The exception does not cover the set-up, validation and error handling
around the kernel: those are cold paths.


## 5. No monkey patches

New monkey patches are prohibited in library code, tests, scripts and
notebooks. To make code behave differently for a test or a check, use one
of:

- an explicit argument that receives the replacement (the pattern of
  `cosmolike_notebook_utils/fisher.py`, whose functions receive the
  notebook's data-vector function as `dv`);
- a subclass defined before use;
- a temporary file or directory;
- a separate process whose files, arguments or environment are chosen
  before Python imports the code (the worker-subprocess pattern of
  `cocoa_testing.py`).

Importing a module or binding an alias is not a monkey patch; replacing
behavior through that alias is. Replacing a method on even one local
instance is a monkey patch.

An existing monkey patch met during other work is reported, not fixed in
passing (Section 9).


## 6. Interfaces and failures

- Use named parameters whenever the called function supports names.
- Document unavoidable positional conventions near the call or in the
  docstring: plotting coordinates, `einsum` operands, the tuple a wrapper
  returns.
- Return shapes and units are part of the interface. Changing a returned
  shape, the order of a returned tuple, or a unit is an interface change:
  every caller (each project's wrappers, notebooks and tests) changes in
  the same piece of work.
- Validate types, finite values, shapes, ranges and cross-field consistency
  before using a value.
- A value saved to a file for later use (a frozen test configuration, a
  chain header, a cached array) is saved fully resolved, including the
  defaults the code applied. A reader of that file never substitutes
  today's code default for a missing key: it names the missing key and
  stops.

A failure message states:

1. what failed;
2. the observed value, or the conflicting fields, when safe to print;
3. the required condition;
4. the corrective action, when the user can correct the input.

`invalid configuration` is not enough. `the mask has 2800 entries but the
data vector has 2812: regenerate the mask with scripts/make_cluster_mask.py`
names what failed, both values, the condition and the repair.

Silent coercion, silent fallback, and a warning where the result is already
wrong are NO-GO.

Do not add compatibility branches for a dependency version outside the
declared Cocoa environment (the conda yml files of the Cocoa repository).
Detect the unsupported version at the first shared boundary and stop with
one clear error.


## 7. Text inside Python

### Docstrings

Every module, function, method and class has one.

- The module docstring teaches the domain first: what the file computes,
  every non-obvious term it relies on, how the pieces relate, and how to
  run it when it is runnable.
- A function docstring has, as applicable:
  - a first sentence with a subject and a verb;
  - the mechanism and the reason it is built this way;
  - an `Arguments:` block naming every parameter, in the file's format
    (`name = what it is, units / shape / valid range`);
  - a `Returns:` block with type, shape and units;
  - a `Raises:` block, or the refusal conditions;
  - side effects: files written, figures drawn, state changed;
  - for an array pipeline, the shape flow and a legend for every shape
    symbol:

```text
C_cs [n_ell, n_richness, n_cluster_z, n_source]
    -> select richness bins [n_ell, n_drawn, n_cluster_z, n_source]
    -> one panel per (cluster z, source) [n_ell, n_drawn]

legend: n_ell = multipoles, n_richness = richness bins,
        n_cluster_z = cluster redshift bins, n_source = source bins
```

### Comments

A comment gives a reason, an invariant, a scientific convention, a shape, a
unit, or a non-obvious failure boundary. It does not narrate the next line.

NO-GO:

```python
# Add one to the counter.
counter += 1
```

GO:

```python
# Count accepted rows only; rejected rows must not shift checkpoint indices.
accepted_rows += 1
```

When a line uses a Python construct a C programmer would not read at sight
(a comprehension with a condition, a generator inside `sum`,
`functools.partial`, star-unpacking, `str.format` field syntax), the comment
at that line says in plain words what the construct produces. Prefer the
plainer construct when it reads better in C terms.

Every constant carries the meaning of its value: why 0.2, why 4 threads, why
these nine cosmologies.

### Explain the current code, not its history

Comments, docstrings, help and error text say what the code does now and
why. They do not record the requests or reviews that led to it.

- When behavior changes, replace the old explanation in place. No dated
  correction, no `now does X`, no review round, no ticket number, no model
  name.
- No person's name, no personal pronouns, no attributed quotations.
- Audience nouns: **the user** (who runs or configures the code) and
  **the reader** (who reads the code or its documentation).

NO-GO:

```python
# Rule from the latest review: now reject a masked last theta bin.
```

GO:

```python
# The last theta bin of each cluster-lensing row is masked: the Y transform
# there needs gamma_t beyond the last bin.
```

A date stays when the program reads or computes it, or when it identifies a
data release or a publication. `previous`, `history` and `phase` stay when
they name run-time data or an algorithm step, not the way the code came to
be.


## 8. Formatting and naming

- Lines within 90 columns, unless a URL or another indivisible value makes
  the limit harmful. Existing files with longer signature lines are not
  reflowed in passing.
- Parentheses for continuation, not backslashes.
- One logical operation per line on cold paths.
- Scientific names with units where ambiguity is possible:
  `theta_arcmin`, `radius_mpc_over_h`, `redshift_grid`, `covariance_cholesky`.
- A local name keeps one meaning within a function.
- Logarithms are `ln<quantity>` or `log10<quantity>`, as in the C code.


## 9. Scope: keep the repair proportional

- A narrow bug gets a narrow change. No registry, policy layer, general
  validation system or other large abstraction where a short direct check
  fixes the named problem.
- As a guide, a fix for one bug that adds plus deletes more than about
  4000 characters outside the tests needs a stated reason why the smaller
  direct repair is unsafe, or it is split.
- Tests may be longer than the fix they cover, because they show valid and
  invalid examples. They follow every rule above.
- Do not turn one task into a repository-wide cleanup. Note the other
  problem sites and report them.
- Add a protective check when it is simple, cheap and sits where the value
  enters. Do not build a framework to anticipate every way a user could
  express an equivalent scientific choice; document the limit and leave the
  choice with the user. A best-effort check says what it actually compares.
- A bounded repair may leave a harmless exceptional case uncovered. Say so
  exactly; do not claim complete coverage.


## 10. `cosmolike_notebook_utils`: the shared notebook package

Design rules of the package (its `__init__.py` documents them):

- **It never imports a project's compiled interface**
  (`cosmolike_<project>_interface`). A function that needs cosmolike
  receives the notebook's own callable as an argument.
- **Pure numpy and matplotlib in the plotting modules.** Nothing in them
  touches CAMB or the compiled interface, so every project shares them
  unchanged.
- **Project-specific facts stay in the project**: fiducial values, bin
  layouts, the interface init sequence, and the thin wrappers in
  `projects/<name>/interface/cosmolike_<name>_notebook_wrappers.py` that
  bind those to the shared functions.
- **Cluster code lives in files whose names end in `_cluster`**
  (`plot_datavectors_cluster.py`), as in the C core. It does not edit the
  galaxy modules or the package `__init__.py`; notebooks import it
  explicitly:
  `from cosmolike_notebook_utils import plot_datavectors_cluster as pdc`.
  Shared private helpers are imported from the galaxy module, not copied.

### Plotting functions (the `plot_datavectors` family)

A new plotting function follows the conventions of the existing ones, so
figures of different probes look alike and notebooks can swap them.

- **One function per probe and space**, named `plot_<quantity>`
  (`plot_xi`, `plot_gammat_tomo_limber`, `plot_C_gg_tomo`,
  `plot_wcc_tomo`).
- **One panel per tomographic bin or bin pair.** Shear is a lower triangle,
  galaxy-galaxy lensing a lens x source grid, clustering one row. An extra
  bin index (the richness bin of the cluster blocks) becomes curves inside
  the panel, not more panels, when the panel count would otherwise
  multiply.
- **First arguments:** the list of curves (each entry what the notebook
  wrapper returns, e.g. `(theta, array)`), then the optional reference
  `<quantity>_ref`, then `param` and `colorbarlabel` for a parameter sweep.
- **Without a reference** each panel shows the quantity with its own
  y-range; **with a reference** it shows `value/reference - 1` on one
  shared linear band with the panels glued edge to edge, or, when `ylim`
  is a list of one `[lo, hi]` per row (`plot_xi`,
  `plot_gammat_tomo_limber`), on one band per row (`sharey="row"`).
- **`rescale = 1`** glues the absolute panels: each panel is multiplied by
  its own power of ten, annotated inside the panel as alpha.
- **Shared option set**, with the same names and defaults style: `marker`,
  `linestyle`, `linewidth`, `ylim`, `cmap`, `legend`, `legendloc`,
  `legendfontsize`, the axis-label and tick-label size arguments,
  `bintextpos`, `bintextsize`, `figsize`, `show`, `colorbar`,
  `colorbarshrink`, `markersize`, and `thetashow` or `lmin`/`lmax`. An
  option that has no effect for one probe is still accepted, and its
  docstring line says so.
- **`show = 1` draws the figure; `show = None` returns `(fig, axes)`.**
  The functions do not save files and do not set fonts or rcParams: figure
  styling belongs to the notebook.
- **Malformed input** prints one message that names the problem
  (`Bad Input (theta)`) and returns 0, before any figure is created.
- **Tick labels within 10% of an interior boundary of glued panels are
  hidden**, and a bin pair that arrives identically zero is drawn as an
  empty panel marked "excluded". Use the helpers of `plot_datavectors`
  (`_hide_glued_edge_ticklabels`, `_glued_supylabel`,
  `_align_log_ticklabels`).
- **Bin arguments count from 0**, as the arrays do; the bin labels printed
  inside panels count from 1.
- **The color of a swept curve is read from the same normalization as the
  colorbar**, so a curve and the bar agree for a sweep of any length.
- **The docstring** lists every argument in the `name = description` form
  of the file and ends with the three return cases (0, None, `(fig, axes)`).
- **Check the figure, not only the code.** Render every mode on real or
  saved arrays (absolute, rescaled, ratio, with data, with markers, with an
  excluded panel) and look at the images: overlapping tick labels, a legend
  that wraps, and a bin label sitting on a curve are found by looking.
  Array dependences invented for a dry run prove the call signature, not
  the y-ranges.

### Notebooks and notebook wrappers

- A notebook cell calls the project's wrappers and the shared plotting
  functions; it does not reimplement them inline.
- A wrapper initializes cosmolike exactly as the project's likelihood does
  (the init chain of `likelihood/_cosmolike_prototype_base.py` is the
  reference) and returns numpy arrays whose axis order its docstring
  states.
- A wrapper copies an array view before handing it to the compiled
  interface when the interface requires owned, contiguous arrays.
- Notebook figure resolution is chosen so a committed notebook with outputs
  stays within a few megabytes.
- The commit message says whether a notebook's outputs come from a
  headless run of the committed cells
  (`jupyter nbconvert --to notebook --execute`).


## 11. What to report with a Python change

- the changed `path::symbol` list, each marked cold or hot;
- the tests and checks that were run: exact commands, return codes and the
  important output lines;
- for a hot path or a changed allocation pattern: the timing before and
  after, and the numerical regression result;
- for a plotting function: which modes were rendered and looked at;
- what was not run or not verified, stated plainly;
- other problem sites noticed and left alone.

A checkbox without a command or an inspected result is not evidence.


## 12. Review checklist

Reject or push back unless all of these hold:

- [ ] Every changed function was read in full, with its callers.
- [ ] Cold paths use explicit control flow, named stages and named
      arguments; no form of the forbidden list in Section 3.
- [ ] Hot paths stayed vectorized, with a numerical check and a timing when
      their shape changed.
- [ ] No new monkey patch.
- [ ] Inputs are validated before mutation, set-up or a file write; every
      failure message has the four parts of Section 6; no silent fallback.
- [ ] A changed return shape, tuple order or unit was carried through
      every caller.
- [ ] Docstrings give arguments, returns (type, shape, units) and side
      effects; comments give reasons, not narration; no names, dates or
      development history in the text.
- [ ] The package rules of Section 10 hold: no compiled-interface import in
      `cosmolike_notebook_utils`, cluster code only in `_cluster` files,
      plotting conventions followed and figures looked at.
- [ ] The change is proportional to the problem; unrelated sites were
      reported, not edited.
- [ ] The report states what was run and what was not.
