# Notes on implementation of PR

Companion to [bounce_Notes.md](../bounce/bounce_Notes.md).

## The algorithm in one paragraph

Probabilistic reparameterization (PR) turns the discrete/mixed acquisition-optimization problem into
a **smooth, unconstrained** one. 

Instead of optimizing `α(round(x))`, continuous relaxation - which is piecewise constant
and has no useful gradient, PR treats the relaxed point `x ∈ [0,1]^d` as the **parameters of a
distribution** over discrete configurations `z`:
- each integer coordinate defines a Bernoulli over `floor(x)` / `ceil(x)`
- each one-hot block defines a Categorical. 

The objective becomes `E_{z ~ p(z|x)}[α(z)]`, which *is* differentiable in `x`, and its maximizer over `x` has the same
maximizer in `z` as the original problem. The BO loop therefore runs entirely in the continuous unit
cube; the discreteness only shows up inside the acquisition wrapper and at evaluation time.

Contrast with Bounce, which keeps the problem discrete and searches it by Hamming-ball local search inside a growing random embedding.

## Repo map

| file | what lives there |
|---|---|
| [experiments/main.py](experiments/main.py) | entry point: `python main.py <exp_dir> <label> <seed>`; reads `config.json`, owns the output path |
| [run_one_replication.py](discrete_mixed_bo/run_one_replication.py) | the BO loop; one replication, no filesystem access |
| [experiment_utils.py](discrete_mixed_bo/experiment_utils.py) | `get_problem` registry, `initialize_model`, `get_acqf`, `eval_problem`, `get_exact_rounding_func` |
| [probabilistic_reparameterization.py](discrete_mixed_bo/probabilistic_reparameterization.py) | the PR acquisition wrappers (analytic + MC, with the custom autograd `Function`) |
| [input.py](discrete_mixed_bo/input.py) | `Round`, `OneHotToNumeric`, and the two PR input transforms that build the distribution |
| [kernels.py](discrete_mixed_bo/kernels.py) | `get_kernel` — the mixed product kernel and the combinatorial kernel |
| [optimize.py](discrete_mixed_bo/optimize.py) | vendored BoTorch `optimize_acqf` with a `stochastic` option (Adam) and a `callback` hook |
| [trust_region.py](discrete_mixed_bo/trust_region.py) | `TurboState` / `update_state`, only used when `use_trust_region=True` |
| [problems/](discrete_mixed_bo/problems/) | benchmarks; [base.py](discrete_mixed_bo/problems/base.py) has the `DiscreteTestProblem` machinery |

Configuration is per-experiment JSON: `experiments/<exp_dir>/config.json` holds the kwargs of
`run_one_replication`, and the second CLI argument (`label`) picks the method.

## The `label` axis

Naming is `<discrete strategy>__<acqf>`, a list of supported labels can be found in `supported_labels`
([run_one_replication.py:50-78](discrete_mixed_bo/run_one_replication.py#L50-L78)).

The prefix selects **how the discrete part is handled**: 

| prefix | what it does |
|---|---|
| `pr__` | probabilistic reparameterization — the paper's method |
| `cont_optim__round_after__` | optimize via continuous relaxation, round the winner back to discrete input space (the naive baseline) |
| `exact_round__fin_diff__` | round before evaluating the acqf, estimate gradients by finite differences to mitigate flat gradients |
| `exact_round__ste__` | round before evaluating the acqf, use a straight-through estimator ([ste.py](discrete_mixed_bo/ste.py)) |
| `enumerate__` | enumerate all discrete configurations; `optimize_acqf_mixed` over them if there are continuous dims |
| `sobol` / `nevergrad_portfolio` | non-BO baselines |

Suffix is the **acquisition function**: `ei`, `ts`, `ucb`, `ehvi`, `nehvi-1`.

My auto-generated `study_*` configs use `pr__ei` everywhere except for purely continuous problems
(e.g. `study_cont_ackley6`, `study_cont_ackley_hd`), which use `cont_optim__round_after__ei`. With no
discrete dimensions the two are the same algorithm, but `pr__` would build a degenerate PR wrapper.

---

# Part 1 — Spaces and encodings (all fixed before the loop starts)

Everything in this part is set up once, inside `get_problem` / `_setup`, and never changes during
optimization. Read it before [the BO loop](#the-bo-loop): the loop assumes all of it already exists.

## Input Order Remapping


`_setup` reorders the columns so the layout is always `[continuous | integer | categorical]`,
regardless of how the benchmark itself ordered them
([base.py:64-75](discrete_mixed_bo/problems/base.py#L64-L75)):

```python
for i, orig_idx in enumerate(cont_indices + integer_indices + categorical_indices):
    remapper[i] = orig_idx
    reverse_mapper[orig_idx] = i
self.bounds = self.bounds[:, remapper]
```

`self.problem`
- is a wrapped BoTorch Function like `Ackley`. 
- `self.problem.bounds` have the benchmarks native column order 
- Values encompass the function's real domain.

`self.bounds`
- is **permuted** (this is what "post-remapping" means below).
- That is why `forward` undoes the permutation with `X = X[..., self._reverse_mapper]` before
  evaluating ([base.py:108](discrete_mixed_bo/problems/base.py#L108),
  [base.py:230](discrete_mixed_bo/problems/base.py#L230)).

**Practical consequence when reading result files.** 

E.g. for ackley53, `_remapper[:6] = [50, 51, 52, 0, 1, 2]` — columns 0-2 of the saved `X` are the three *continuous* variables and the 50 binaries are columns 3-52, the opposite of the benchmark's own ordering.  
??? Same story for `discrete_ackley`: `_remapper[:6] = [15, 16, 17, 18, 19, 0]`, so the 5 continuous dims come first.

## Categorical variables -> One-hot expansion

Categorical variables are one-hot encoded in `_setup`, when `effective_dim` is computed
([base.py:80-84](discrete_mixed_bo/problems/base.py#L80-L84)):

```python
self.effective_dim = (
    self.cont_indices.shape[0]
    + self.integer_indices.shape[0]
    + int(sum(self.categorical_features.values()))
)
```

`categorical_features` ([base.py:154-165](discrete_mixed_bo/problems/base.py#L154-L165)) is an
`OrderedDict` that maps **variable index → variable cardinality**.  
It derives this from the bounds as `upper-bound - lower-bound + 1`.

```python
@property
def categorical_features(self) -> Optional[Dict[int, int]]:
    # Return dictionary mapping indices to cardinalities
    if self.categorical_indices is not None:
        categ_bounds = self.categorical_bounds
        return OrderedDict(
            zip(
                self.categorical_indices.tolist(),
                (categ_bounds[1] - categ_bounds[0] + 1).long().tolist(),
            )
        )
    return None
```

One-hot encoding changes the indices of the categorical variables, the keys in `categorical_features` refer to the indices in the numeric space. The indices of the one-hot-encoded variables I will call *block* here.

numeric:  [c c | i i i | k₅ k₆ k₇ k₈ ]                      width  9  
one-hot:  [c c | i i i | ─5─ ─5─ ─5─ ─5─ ]                  width 25  
indices:  0 1   2 3 4   5-9 10-14 15-19 20-24

The block offsets are *not* the dict keys. Both `eval_problem` and `Round` recompute them by starting
at `min(keys)` and accumulating cardinalities
([experiment_utils.py:96-103](discrete_mixed_bo/experiment_utils.py#L96-L103)):

```python
start_idx = None
for i, cardinality in base_function.categorical_features.items():
    if start_idx is None:
        start_idx = i
    end_idx = start_idx + cardinality
    ...
    start_idx = end_idx
```

**Example — `PestControl`**  

- declares `_bounds = [(0, 4)] × 25` and `categorical_indices = list(range(25))` ([pest.py:110-116](discrete_mixed_bo/problems/pest.py#L110-L116)).
- So `categorical_features = {0: 5, 1: 5, …, 24: 5}` and `effective_dim = 0 + 0 + 25·5 = 125`.
- Categorical variable `i` occupies columns `5i … 5i+4`; variable #3 is columns **15-19**. For a mixed
problem the blocks instead start after the continuous and integer columns.

## Input normalization

The BO loop in bo_pr runs **entirely in the unit cube** `[0,1]^effective_dim` (with one-hot encoded
categoricals). 

Nothing in the acquisition function, the GP, or the trust region ever sees the
benchmark's native scale. The mapping out to the benchmark happens only at evaluation time, and it's
done in **two stages**, which is why `normalize` is immediately followed by `unnormalize` in
`get_orig_X`.

### The three coordinate systems

**1. Optimization space** - `standard_bounds` in [run_one_replication.py:207-208](discrete_mixed_bo/run_one_replication.py#L207-L208):

```python
standard_bounds = torch.ones(2, base_function.effective_dim, **tkwargs)
standard_bounds[0] = 0
```

- Creates `[0,1]^d`
- Were categorical values were one-hot expanded to account for every possible value (`effective_dim` = #cont + #int + Σ cardinalities,
[base.py:80-84](discrete_mixed_bo/problems/base.py#L80-L84)). 
- Sobol init draws here
- The GP is fit on `train_x` here
- `optimize_acqf` searches here
- The TR bounds are clipped to it ([run_one_replication.py:437-438](discrete_mixed_bo/run_one_replication.py#L437-L438))
- And the `X` written to the output file is in this space (see [Where the output is written](#where-the-output-is-written))

**2. Search space** — `base_function.bounds` (post-remapping) 
- integers at their true levels (`0..9`)
- continuous at their declared range (`[-1,1]` for ackley53)
- categoricals as indices `0..C-1`. `one_hot_bounds` ([base.py:86-92](discrete_mixed_bo/problems/base.py#L86-L92)) is the
one-hot-expanded version of this.

**3. Native benchmark space** — `self.problem.bounds`, whatever the wrapped botorch function was
defined on (e.g. Ackley's default `[-32.768, 32.768]^d`).

### Concrete values

`base_function.bounds` for **ackley53**, shape `(2, 53)`:

| columns | 0-2 | 3-52 |
|---|---|---|
| what | continuous (orig `x_50..x_52`) | binary (orig `x_0..x_49`) |
| lower | -1.0 | 0.0 |
| upper | 1.0 | 1.0 |

`base_function.bounds` for **discrete_ackley**, shape `(2, 20)`:

| columns | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| what | continuous (orig 15-19) | integer, 3 levels | integer, 5 levels | integer, 10 levels |
| lower | -32.768 | 0.0 | 0.0 | 0.0 |
| upper | 32.768 | 2.0 | 4.0 | 9.0 |

Note the mixed units in the second table: the continuous columns carry native Ackley bounds, the
integer columns carry level counts.

The corresponding `self.problem.bounds` — the bounds of the wrapped botorch `Ackley`, unpermuted:

- **discrete_ackley** — untouched botorch default, `[-32.768, 32.768]` on all 20 dims.
- **ackley53** — overwritten in
  [experiment_utils.py:616-619](discrete_mixed_bo/experiment_utils.py#L616-L619) to `[0,1]` on dims
  0-49 and `[-1,1]` on dims 50-52.

## The transitions

**1 → 2** happens in `eval_problem`
([experiment_utils.py:82-112](discrete_mixed_bo/experiment_utils.py#L82-L112)): one-hot blocks are
`argmax`'d into category indices, those indices are `normalize`d back into unit scale so the whole
vector is uniformly in `[0,1]`, and then a single `unnormalize(X_numeric, base_function.bounds)`
lifts everything into the search space.

This is the **only** place a categorical is decoded, and it runs inside the loop (step 7). For
categoricals the `normalize`/`unnormalize` pair is a round trip with no net effect (`k → k/(C-1) → k`)
— it exists purely so one `unnormalize` call can handle continuous, integer and categorical columns
uniformly.

For ackley53's continuous dims that call *is literally* `-1 + 2u` — the same arithmetic as
[benchmarks.py:1736](../bounce/bounce/benchmarks.py#L1736), just expressed as bounds instead of a
hardcoded affine map.

**2 → 3** is `get_orig_X` ([base.py:208-212](discrete_mixed_bo/problems/base.py#L208-L212)), reached
via `DiscretizedBotorchTestProblem.forward`. `_orig_bounds` is the snapshot of the discretized bounds
taken before index remapping ([base.py:61](discrete_mixed_bo/problems/base.py#L61)), so this maps
"integer level *k* out of 0..9" to "the corresponding point in Ackley's native domain":

- `discrete_ackley` — `integer_bounds = [0,2]` but `problem.bounds = [-32.768, 32.768]`, so level 1 →
  0.5 → `0.0`. This stage does the real work; it's what makes "discretizing a continuous test
  function" mean placing a grid over its native domain.
- `ackley53` — `problem.bounds` was already overwritten to `[0,1]`/`[-1,1]` at
  [experiment_utils.py:617-619](discrete_mixed_bo/experiment_utils.py#L617-L619), and those match
  `_orig_bounds`, so `get_orig_X` is an identity map. All the real rescaling already happened in
  stage 1.

### A traced example

One point pushed all the way through `discrete_ackley`, starting from a unit-cube input of all
`0.5`:

```
unit cube after rounding : 0.5   0.5 ...        (all dims)
search space             : 0.0   … 1.0 … 2.0 …  (cont=0.0, 3-level→1, 5-level→2)
search space, orig order : 1.0 … 2.0 … 4.0 …
native space (get_orig_X): 0.0 … 0.0 … -3.6409 …
```

Reading the last row: a 10-level integer at level 4 becomes `-32.768 + (4/9)·65.536 = -3.6409`. The
3-level dim at level 1 maps to `-32.768 + (1/2)·65.536 = 0.0`, and the continuous dims at `0.5` also
map to `0.0`.

### So what *is* the search space?

Everything the optimizer touches is in `[0,1]`. The search space is the intermediate layer, and it
differs from the native benchmark space in **two independent ways**:

**(a) A box restriction.** `get_orig_X` maps the `_orig_bounds` box affinely onto the
`problem.bounds` box, so the reachable region is exactly the `problem.bounds` box — but
`problem.bounds` is itself mutable, and `ackley53` mutates it. Native Ackley lives on
`[-32.768, 32.768]^53`; the experiment redefines it to `[0,1]^50 × [-1,1]^3`. That is a genuine
sub-box: a deliberately shrunk, much easier version of Ackley, chosen to match the
CASMOPOLITAN/Bounce benchmark definition. `discrete_ackley` does no such shrinking and keeps the full
native box.

**(b) A grid restriction.** Within whatever box `problem.bounds` defines, the integer dims can only
land on `k/(K-1)` fractions of it. For `discrete_ackley`, a 3-level dim reaches only
`{-32.768, 0, 32.768}` — the two endpoints and the midpoint. The level count comes from
`integer_bounds` and is completely decoupled from *where* those levels land, which comes from
`problem.bounds`.

So: not a subset in the loose sense, but an affine reparametrization of a (possibly shrunk) sub-box,
with the discrete dimensions collapsed to an evenly spaced grid inside it. Continuous dims pass
through as a pure rescale, with no restriction at all.

## Where rounding fits

Integrality is also enforced in the unit cube, by round-tripping: `get_exact_rounding_func`
([experiment_utils.py:123-199](discrete_mixed_bo/experiment_utils.py#L123-L199)) chains
`Normalize(reverse=True)` → `Round` → `Normalize(reverse=False)`, with `indices=integer_indices`. So
a unit-cube value is expanded to `0..9`, snapped, and pushed back to the unit cube — the candidate
that leaves is still normalized, just restricted to the grid `{0, 1/9, ..., 1}`.

Two things follow from the `indices=integer_indices` argument: continuous dims are never touched by
the rounding chain (they stay free in `[0,1]`), and the `cont_bounds` written into `one_hot_bounds`
at [base.py:91](discrete_mixed_bo/problems/base.py#L91) is effectively dead for those dims —
`one_hot_bounds` is only ever consumed by `get_exact_rounding_func`. Categoricals are handled inside
`Round` on the one-hot block, which is natively `[0,1]` anyway.

Concretely, `Round.transform` ([input.py:229-241](discrete_mixed_bo/input.py#L229-L241)) does
`one_hot(X[..., start:end].argmax(-1), num_classes=cardinality)` per block — so an arbitrary vector of
`C` values in `[0,1]` becomes a valid one-hot vector.

The `initialization=True` variant widens integer bounds by ±0.4999
([experiment_utils.py:150-152](discrete_mixed_bo/experiment_utils.py#L150-L152)) so the endpoints get
the same probability mass as interior levels under Sobol sampling — otherwise `0` and `9` would each
be half as likely as the rest. **Categoricals need no such correction**: the argmax of `C` i.i.d.
uniforms is already a uniform category, which is why the widening is applied only to
`integer_indices`.

---

# Part 2 — The optimization loop

## The BO loop

**Before iteration 0** ([run_one_replication.py:135-219](discrete_mixed_bo/run_one_replication.py#L135-L219))
the following already exist and never change again:

- `base_function = get_problem(...)` — runs `_setup`, fixing the column order
  ([Remapping](#remapping)), the [one-hot expansion](#one-hot-expansion), `effective_dim` and
  `one_hot_bounds`.
- `standard_bounds = [0,1]^effective_dim` — the only space anything below operates in
  ([Input normalization](#input-normalization)).
- `exact_rounding_func` / `init_exact_rounding_func` — built once
  ([Where rounding fits](#where-rounding-fits)), applied inside the loop.
- `X, Y = generate_initial_data(...)` — Sobol over `standard_bounds`, then `init_exact_rounding_func`,
  so the initial `train_x` is already valid: integers on-grid, one-hot blocks exactly one-hot.

So there is **no encoding step inside the loop**. One-hot is the shape of the space, not a transform;
the only conversion is the *decode* inside `eval_problem` at step 7.

[`run_one_replication`](discrete_mixed_bo/run_one_replication.py#L81), one iteration:

1. **Fit the model**
    - `initialize_model` + `fit_gpytorch_model`, refit from scratch every iteration.
    - Default is a `FixedNoiseGP` with noise `1e-7 * var(Y)` (essentially noise-free) on `Standardize`d `Y`.
2. **Build the acqf** (`get_acqf`)
    - then wrap it in a PR object if the label says so.
3. **Optionally shrink to a trust region**
    - only if `use_trust_region=True`.
    - The TuRBO box is applied to continuous dims only: binary and categorical dims are reset to their full range
   ([run_one_replication.py:434-441](discrete_mixed_bo/run_one_replication.py#L434-L441)). 
4. **Optimize the acqf** 
    - over `standard_bounds` with `optimize_acqf(..., return_best_only=False)`
    - so all `num_restarts` (default 20) candidates come back.
    - Mid-search the iterate is **fractional everywhere**, including inside one-hot blocks — which is
      why the GP has to be defined on the whole relaxed cube.
5. **Discretize**
    - for PR, `sample_candidates` draws `z ~ p(z|x)` from the optimized distribution (`sample_candidates=True` by default); 
   - otherwise `exact_rounding_func` snaps to the grid.
   - either way, one-hot blocks leave this step valid again.
6. **Select across restarts** 
    - by evaluating the *unwrapped* acquisition function on the rounded candidates ([run_one_replication.py:526](discrete_mixed_bo/run_one_replication.py#L526)). 
7. **Evaluate** 
    - via `eval_problem` — the only decode step, see [The transitions](#the-transitions) — append, save.

Note `batch_size > 1` raises `NotImplementedError` at step 6 — the repo is effectively q=1 only.

## How PR is actually computed

`get_acqf` wraps the base acquisition function in either
- [`AnalyticProbabilisticReparameterization`](discrete_mixed_bo/probabilistic_reparameterization.py#L241) or 
- [`MCProbabilisticReparameterization`](discrete_mixed_bo/probabilistic_reparameterization.py#L323).

Both delegate the distribution to an input transform built by [`get_probabilistic_reparameterization_input_transform`](discrete_mixed_bo/probabilistic_reparameterization.py#L36).

This is the same `unnormalize → (PR) → normalize` sandwich as the exact rounding chain described in
[Where rounding fits](#where-rounding-fits) — only the middle stage differs.

**The probabilities** ([input.py:385-408](discrete_mixed_bo/input.py#L385-L408)): 

- for an integer coordinate, `p = sigmoid((|x| - floor(|x|) - 0.5) / tau)`;
- for a one-hot block, `p = softmax((x - 0.5) / tau)`. 
- `tau = 0.1` by default — small `tau` sharpens the distribution toward a hard rounding, `tau → None` falls back to the raw fractional part / normalized weights.

**Analytic** 

- enumerates the full Cartesian product of discrete configurations
- (`all_discrete_options`, built once in the transform's `__init__`)
- returns the exact expectation `Σ_z p(z|x) α(z)`. 
- Exact gradients, but the option count is `Π cardinalities` — fine for ackley13 (2¹⁰), hopeless for ackley53 (2⁵⁰) or pest (5²⁵).

**MC** 

- is the default for everything 
- It draws `pr_mc_samples = 128` configurations per forward pass
- uses a **custom autograd `Function`**
([probabilistic_reparameterization.py:399](discrete_mixed_bo/probabilistic_reparameterization.py#L399))
whose backward implements a REINFORCE / score-function estimator with a moving-average baseline
(`pr_grad_estimator = "reinforce_ma"`, the `ma_*` buffers on the wrapper). 
- Because the samples are redrawn every call (`pr_resample=True`), the objective is stochastic, and `optimization_kwargs["stochastic"] = True` switches `optimize_acqf` from L-BFGS to **Adam** ([run_one_replication.py:177-184](discrete_mixed_bo/run_one_replication.py#L177-L184)). 
- These four defaults are set automatically whenever `"pr"` is in the label; these are not in my config files.

## The model

[`initialize_model`](discrete_mixed_bo/experiment_utils.py#L237) → [`get_kernel`](discrete_mixed_bo/kernels.py#L18).

`kernel_type="mixed"` (the default) builds **at most two** Matérn kernels and multiplies them, wrapped
in a single `ScaleKernel`. Which ones exist is problem-dependent, because each is appended only if its
index list is non-empty ([kernels.py:39-63](discrete_mixed_bo/kernels.py#L39-L63)):

```python
cont_dims = list(set(list(range(dim))) - set(binary_dims))
if ("latent" in kernel_type) or ("categorical" in kernel_type):
    cont_dims = list(set(cont_dims) - set(categorical_dims))
```

- `cont_dims` is defined by subtracting **`binary_dims` only**. Under plain `mixed` the second line
  does not fire, so **one-hot categorical columns stay in `cont_dims`** and are modelled by the ARD
  Matérn. There is no separate categorical kernel unless you switch `kernel_type`.
- `binary_dims` is not read off the problem directly — it is `integer_indices` filtered to those whose
  range is exactly 1 ([run_one_replication.py:137-148](discrete_mixed_bo/run_one_replication.py#L137-L148)).

| problem | `binary_dims` | `cont_dims` | kernels actually built |
|---|---|---|---|
| pest | `[]` | all 125 one-hot columns | **one** Matérn, ARD, 125 lengthscales |
| ackley53 / svm | the 50 binary columns | the 3 continuous columns | two: ARD Matérn over 3, isotropic Matérn over 50 |
| contamination / labs | all 25 resp. 50 columns | `[]` | **one** Matérn, isotropic |
| cont_ackley6 / _hd | `[]` | all columns | one Matérn, ARD |

So the product form is never actually exercised by my configs except on ackley53 and svm, and no
benchmark in the study set has categoricals *and* binaries at the same time.

Both kernels use `nu=2.5` and `lengthscale_constraint=Interval(0.1, 20.0)`. `use_ard_binary=False` is
the default, so the binary block shares one lengthscale — the same isotropic choice Bounce makes for
its discrete bins.

Alternatives (none selected by my configs)
- `mixed_categorical` — replaces the one-hot Matérn with BoTorch's `CategoricalKernel`, which operates
  on the **numeric** representation, so `initialize_model` applies `OneHotToNumeric` to `train_x` and
  `run_one_replication` sets `apply_numeric=True`. Returns `ScaleKernel(prod) + ScaleKernel(sum)` —
  two separately-scaled terms. `CategoricalKernel` is non-differentiable w.r.t. inputs, which is why
  [run_one_replication.py:156-162](discrete_mixed_bo/run_one_replication.py#L156-L162) silently
  downgrades `cont_optim__round_after` + this kernel back to plain `mixed`.
  ⚠️ `prod_kernel *= k` runs in *both* loops ([kernels.py:85-91](discrete_mixed_bo/kernels.py#L85-L91)),
  so the product is `k₀·k₁²·k₂²` rather than `k₀·k₁·k₂`.
- `mixed_latent` — a trainable `C × latent_dim` embedding table per categorical
  ([input.py:873-879](discrete_mixed_bo/input.py#L873-L879)), fit jointly with the GP hyperparameters;
  an isotropic Matérn per embedding block. `latent_dim` is 1 for `C ≤ 3`, else 2.
- `ard_combo`/`iso_combo` — `CombinatorialCovarModule`
  ([kernels.py:104](discrete_mixed_bo/kernels.py#L104)) for pure-binary problems. Its priors and
  constraints are applied only when `ard_num_dims > 1`, so `iso_combo` silently falls back to gpytorch
  defaults.

Currently none of my auto-generated configs set `model_kwargs`, so they all run `mixed` on `FixedNoiseGP`.

## Where the output is written

`run_one_replication` never touches the filesystem — it only calls `save_callback`, an injected
argument ([run_one_replication.py:96](discrete_mixed_bo/run_one_replication.py#L96)). The actual
write lives in the launcher, [experiments/main.py:45-55](experiments/main.py#L45-L55):

```python
output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")
...
save_callback = lambda data: torch.save(data, output_path)
```

So it's `torch.save` to `experiments/<exp_dir>/<label>/<seed>_<label>.pt`, not a raw pickle — load it
with `torch.load`. `experiments/main.py` is the only caller of `run_one_replication` in the repo.

Inside the loop the callback fires twice, with the same `X`:

- [run_one_replication.py:591](discrete_mixed_bo/run_one_replication.py#L591) — periodic save, gated
  on `save_frequency` (set to 5 in `main.py`)
- [run_one_replication.py:602](discrete_mixed_bo/run_one_replication.py#L602) — final save, which
  additionally carries `all_loss_trajs`, `all_xs_trajs`, `all_true_af_trajs`

`X` is accumulated at [run_one_replication.py:557](discrete_mixed_bo/run_one_replication.py#L557)
(`X = torch.cat([X, candidates], dim=0)`), and `candidates` there is still the unit-cube, one-hot
tensor the acqf produced. Nothing un-normalizes it before saving, so **what lands on disk is
optimization-space**. To recover benchmark-space inputs from a results file you have to replay
`eval_problem`'s transform (and undo the [remapping](#remapping)). For a categorical problem that
means an `argmax` per one-hot block — a saved pest `X` is 125 columns wide, not 25.

## For the bounce comparison

The relevant consequence: both codebases hand the objective continuous values in `[-1,1]` and
binaries in `{0,1}`, and both optimize in `[0,1]`. The difference is only bookkeeping — Bounce writes
the affine map by hand inside `__call__`, bo_pr declares it as bounds and lets `unnormalize` apply
it. GP lengthscales are on the same unit scale in both, so kernel hyperparameters are comparable
across the two.

Three things that are *not* just bookkeeping, and that bite when merging result files:

- **Sign.** bo_pr negates every objective (`negate=True` throughout `get_problem`), so the saved `Y`
  is already "higher is better". Bounce saves the raw minimization value.
- **Budget.** `n_initial_points + iterations` in bo_pr is the exact number of evaluations. Bounce's
  loop condition is `<=`, so it performs `maximum_number_evaluations + batch_size` evaluations, and
  a full-dimensional restart injects extra `sample_init` points mid-run.
- **Reproducibility.** bo_pr seeds `torch` and `numpy` from the `seed` argument; Bounce seeds
  nothing.

The side-by-side table of both methods is at the end of
[bounce_Notes.md](../bounce/bounce_Notes.md#for-the-bo_pr-comparison); benchmark-level differences
are in [Benchmarks_new.md](../Thesis_Repo/analysis/docs/Benchmarks_new.md).
