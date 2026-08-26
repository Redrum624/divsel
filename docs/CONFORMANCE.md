# Conformance: how a port proves it implements GIST the way divsel does

divsel is the **reference implementation** of GIST (arXiv:2405.18754v3): it was
written directly from the paper and is backed by a brute-force oracle over 500
instances proving the `(1/2 − ε)` and `(2/3 − ε)` guarantees. A port — Aura's
`gist_select` (Python), limbic's `gistSelect` (TypeScript), or any other —
proves conformance against **`test-assets/golden-selection.json`** (schema 1),
not against divsel's source.

## Independent verification (2026-08-26)

divsel's strongest independent check to date is a **90,000-instance
differential** between divsel's Rust core and Aura's pure-Python port of GIST —
a port written from **this document alone**, never from divsel's source.

| | |
|---|---|
| instances | 90,000 random |
| parameters | `n` 1–40, `dim` 2–48, `k` 1–14, `lam ∈ {0 … 64}`, `eps ∈ {0.01 … 1.0}`, six vector families, six weight families |
| **algorithmic disagreements** | **0** |
| raw `selected` differences | 1,164 |
| of those, exact `f` ties (two sets, same `f`) | 1,032 |
| of those, the amplification in [Degenerate geometry](#degenerate-geometry-divsels-f32-distances-are-not-the-exact-ones) | 132 |
| largest distance disagreement it drew | `4.77e-07` — 4 ulp of `1.0`, 2 ulp of `2.0`, in f32 |

Every one of the 1,164 differences was re-run by handing the port's Algorithm 1
**divsel's own f32 distance matrix**; on 1,164 of 1,164 it then reproduced
divsel's answer exactly, tie rules included. That is the claim, and it is
narrower than "the two always agree": it says that where they differed on the
instances drawn, the difference was the width of the distance arithmetic and not
the algorithm.

Two defects in **this document** came out of it, and both are fixed here:

1. The float tolerance was stated on `f` rather than on the primitives `f` is
   built from, so it was not `lam`-independent — 147 of the 90,000 instances
   that otherwise agreed exceeded it purely because `lam` multiplied a distance
   ulp. See [Tolerance rules](#tolerance-rules).
2. Nothing warned a port author that divsel's f32 kernel is the *less* accurate
   side on degenerate geometry, so a correct float64 port disagrees there. See
   [Degenerate geometry](#degenerate-geometry-divsels-f32-distances-are-not-the-exact-ones).

## The procedure

1. Load `test-assets/golden-selection.json`.
2. For each of the 22 `cases`, rebuild the instance from `vectors`,
   `utilities`, `metric`, `utility`, `k`, `lam`, `eps`,
   `exhaustive_thresholds`, `diameter`, `diameter_sweeps` and run your GIST.
3. Compare against the `expected_*` fields under the tolerance rules below.

All inputs are **dyadic rationals** (multiples of 1/64 in [−4, 4]), so every
language parses them into exactly the same f32/f64 values — there is no input
rounding to disagree about.

### Tolerance rules

Write **`tol(x) = f_rel * max(1, abs(x))`**, with `f_rel = 1e-6` read from the
file's `tolerance` block.

| Field | Rule |
|---|---|
| `expected_selected` | **Exact** list equality, order included (selection order; ascending for a diameter pair). |
| `expected_stage` | Exact string equality (`"greedy"` / `"diameter_pair"` / `"sweep"`). |
| `expected_g`, `expected_div`, `expected_threshold`, `expected_d_max` | `abs(actual − expected) <= tol(expected)`. |
| `expected_f` | `abs(actual − expected_f) <= tol(expected_g) + lam * tol(expected_div)`, with `lam` the case's own `lam` field. |

The `tolerance` block itself is unchanged: `f_rel` is still the single knob, and
the `lam`-weighting is derived per case from that case's own `lam`.

#### Why `f`'s bound is derived instead of relative to `f`

`f = g + lam * div`, so `f` is **not a primitive**. `g` and `div` are; `f` is
the number they combine, and an error in `div` reaches `f` multiplied by `lam`
— a caller's parameter this contract puts no ceiling on. A bound stated
directly on `f` has to guess how much of `f` arrived through `lam`, and the
previous rule (`tol(expected_f)`) guessed that the answer scales with
`abs(f)`. That is right everywhere except one regime, and the regime is
reachable: when `g` dominates `f` and `div` is small, `abs(f)` is small while
`lam * (last ulp of div)` is not.

Cosine distance is `clamp(1 − a·b, 0, 2)` (rule 14), and on near-duplicate rows
that subtraction is a cancellation — the result is tiny but its **absolute**
error stays at the `ulp(1) = 1.1920929e-7` scale. A worked case, dyadic and
reproducible:

```
vectors = [[-0.140625, -0.921875, -0.609375],
           [-0.15625,  -0.9375,   -0.609375]]     # -9/64,-59/64,-39/64 and -10/64,-60/64,-39/64
metric = "cosine", utility = "linear", utilities = null, k = 2, lam = 64.0, eps = 0.1

divsel (f32 kernel)   selected [0, 1], stage "sweep",
                      g = 2.0, div = 1.0418891906738281e-4, f = 2.0066680908203125
a float64 port        same selection, same stage,
                      div = 1.0442041095148902e-4, f = 2.0066829063008953

distance gap  2.31e-07   (~2 ulp at magnitude 1 — the cancellation scale)
f gap         1.48e-05   = lam * distance gap
old bound     tol(f)                        = 2.007e-06   -> REJECTED (7.4x over)
new bound     tol(g) + lam * tol(div)       = 6.600e-05   -> accepted (22% of budget)
```

A port that is *more* accurate than divsel here is failed by the old rule. That
is a defect in the contract, not in the port. The derived bound propagates each
primitive's own budget instead of guessing: `div`'s budget multiplied by the
same `lam` the objective multiplies it by, plus `g`'s budget.

Two properties worth stating explicitly:

* `tol(div)` keeps the `max(1, ·)` **absolute floor** of `1e-6`, and the floor
  — not the relative half — is what carries this case: a cosine distance near
  zero has no relative accuracy left, but its absolute error is bounded by a few
  `ulp(1)`. Measured here over 40,000 random instances (six vector families,
  both metrics, `n <= 40`, `dim <= 48`, `lam` up to 64), the largest `div` error
  a float64 recomputation produced against divsel used **23.7%** of `tol(div)`
  and the largest `d_max` error used **24.1%**.
* At `lam == 0` the bound collapses to `tol(expected_g)`, which is exactly
  right: rule 18 makes `f` equal `g` there, `div` included when `div` is `+inf`.

#### It does not loosen the committed contract, and it cannot accept a wrong answer

On the 22 cases the derived bound is **identical** to the old one on 17 of them
and never more than **1.17x** larger (case 21, `lam = 1`, `div = 0`); the
largest bound anywhere in the file is `1.906e-05` (case 5).

The separation between "ulp noise" and "detectable error" is three orders of
magnitude, measured rather than asserted:

| | measured |
|---|---|
| noise ceiling — largest share of the new `f` budget used by a genuine f32-vs-float64 difference, over 40,000 instances (`lam` up to 64, both metrics, six vector families, linear and facility-location `g`) | **0.21** of the budget; **0** exceedances. The old rule was exceeded **69** times, by up to **8.1x**. |
| detection floor — distance from `expected_f` to the nearest *different* `f` value attainable by any subset of size `<= k`, over the 22 cases | **>= 631x** the new bound (case 16, the tightest; the median case is ~1e5x) |
| the generator's own robustness margin (`MARGIN = 1e-4` relative, `python/tools/gen_golden.py`) | the new bound consumes at most **1.17%** of it, on every one of the 22 cases |
| smallest discrete step the objective can take — `1` for coverage and uniform linear, `1/64` for the dyadic-weight linear cases | **>= 819x** the largest bound in the file |

So an error big enough to be a bug — a broken tie-break, a strict `>` where the
fold is `>=`, the wrong FacilityLocation scale — moves `f` by at least ~600
times what the rule tolerates. Perturbing every `expected_f` by `1e-4` relative
(the generator's margin floor) or by the case's smallest possible `Δg` is
rejected on all 22 cases.

**The derived bound is never the sole detector, which is why widening it is
cheap.** `expected_g` and `expected_div` keep their own, unchanged, tight
bounds and are checked on every case. A port whose `g` is wrong fails on `g`;
a port whose `div` is wrong fails on `div`; neither outcome depends on what the
`f` check does. What the `f` check adds is a consistency check on the
*combination* — the wrong `lam`, the wrong sign, `0 * inf` producing `NaN`
(rule 18) — and every one of those is gross, not marginal. Nor does the
tolerance gate `expected_selected` or `expected_stage`: those are exact-equality
fields, and a wrong answer that happens to tie on `f` — which is what the five
margin-exempt cases are built from — is caught there.

That matters most in the regime the fix exists for. At `lam = 64` with a
near-zero `div`, the bound is `~6.6e-05` while a `1e-4`-relative change in `f`
is `~2.0e-04`: about **3x** of headroom on that yardstick, not the 85x the 22
committed cases enjoy (they run at `lam <= 4`). Stated honestly, the yardstick
is the weak part there, not the bound: with `f` dominated by `g`, a `1e-4`
*relative* change in `f` is a `1e-4` relative change in `g`, and `g`'s own
`1e-6` bound catches that a hundred times sooner. Against the yardstick that
actually applies to a wrong *answer* — the smallest discrete step the objective
can take, `1/64` for dyadic weights — the headroom at `lam = 64` is **237x**.

What each field *means* is fixed by the contract below: `expected_f`,
`expected_g`, `expected_div` are `f(S)`, `g(S)` and `div(S)` of the reported
selection (rules 4, 8, 16, 17); `expected_threshold` is the per-stage value of
rule 3; `expected_d_max` is the line-3 diameter — exact, or the estimate
`d_hat` under `diameter == "approx"` (rules 9, 10).

Every case must pass. The only optional case is listed under
[Optional cases](#optional-cases).

### Minimal reader (TypeScript-flavoured pseudocode)

```ts
const g = JSON.parse(readFileSync("test-assets/golden-selection.json", "utf8"));
const tol = (x: number) => g.tolerance.f_rel * Math.max(1, Math.abs(x));
for (const c of g.cases) {
  const r = gistSelect(c.vectors, c.utilities, c); // k, lam, eps, metric, utility, ...
  assert.deepEqual(r.selected, c.expected_selected);
  assert.equal(r.stage, c.expected_stage);
  // `f` is derived from `g` and `div`, so its bound is too: lam multiplies div's.
  const bounds: [number, number, number][] = [
    [r.f,         c.expected_f,         tol(c.expected_g) + c.lam * tol(c.expected_div)],
    [r.g,         c.expected_g,         tol(c.expected_g)],
    [r.div,       c.expected_div,       tol(c.expected_div)],
    [r.threshold, c.expected_threshold, tol(c.expected_threshold)],
    [r.dMax,      c.expected_d_max,     tol(c.expected_d_max)],
  ];
  for (const [a, e, bound] of bounds) assert.ok(Math.abs(a - e) <= bound);
}
```

divsel's own readers — `crates/divsel/tests/golden.rs` (Rust) and
`python/tests/test_golden.py` (Python) — implement exactly this procedure and
are the executable reference for it.

## The contract: every `[divsel choice]` a port must reproduce

The paper's Algorithm 1 leaves several details open. divsel fixed them; the
fixtures encode them; a conforming port must make the **same** choices.
(Line numbers count the paper's `function` header as line 1.) Each rule ends
with the cases that **pin** it — the cases that fail when the rule is broken.
Where a rule is stated but no fixture would catch a deviation, it says so:
those rules are documented choices a port takes on trust, and an honest
citation matters more than an impressive one.

1. **Greedy argmax ties → lowest index.** Whenever two candidates have equal
   marginal gain, the smaller index wins. (Cases 10 and 18 are built around
   it; measured, a highest-index rule also fails cases 1, 2, 3, 8, 9, 12 and
   17 — nine of the 22.)
2. **Sweep fold: ascending `d`, non-strict `>=`.** Line 8 iterates a *set*;
   divsel folds the thresholds in ascending order and line 10 compares `>=`,
   so **the largest threshold attaining the best `f` wins** — including a tie
   with the greedy or diameter-pair incumbent, which the sweep then relabels.
   A parallel sweep must reproduce this exact fold order. (Cases 2, 4, 9, 18
   pin it directly; a strict `>` fold changes the reported stage or threshold
   of 15 of the 22 cases.)
3. **Diametrical-pair check (line 5): strict `>`, guarded by
   `k >= 2 && n >= 2`.** The pair is `(u, v)` from line 3 — the
   exact-diameter pair of rule 9 under `diameter == "exact"`, the `(a, b)`
   the double sweep returned under `"approx"` — evaluated from a reset
   utility (`g(∅) = 0`, then `u`, then `v`); it displaces the greedy
   solution only when its `f` is strictly larger, and it is reported in
   ascending index order. **Reported `threshold` per stage:** `0` for
   `"greedy"`, `d_max` (`d_hat` under approx) for `"diameter_pair"`, the
   winning `d` for `"sweep"`. (Cases 11, 15, 22 pin the pair stage and
   `threshold == d_max`; case 21 pins it at `d_max == 0`. The strictness of
   the `>` itself is **not pinned**: in case 18 the pair ties greedy exactly
   and rule 2's fold then reaches the same `[0, 2]` either way, and in case
   11 the pair wins outright — a port comparing `>=` here passes all 22
   cases. Use the paper's strict `>` regardless.)
4. **`div(|S| <= 1) = d_max`** — the paper's own definition (Sec. 2), not 0.
   This applies to the empty set and singletons. (Case 7.)
5. **The sweep is skipped when `d_max == 0`** (`n == 1`, or every point
   coincident): every threshold would be 0 and re-run line 2. **Only the
   sweep is skipped.** Line 2 runs, and for `n >= 2` the line-5 check of
   rule 3 runs too — its guard is `k >= 2 && n >= 2`, not `d_max > 0` — with
   every `div` equal to 0, so it reduces to `g(pair) > g(greedy)`. The pair
   is `(0, 1)` (every pair ties at distance 0; both the exact reduction and
   the double sweep of rule 9 return the lexicographically smallest), and a
   non-modular utility can let it win. So `stage` is never `"sweep"` for such
   inputs: it is `"greedy"`, or `"diameter_pair"` when the pair strictly
   wins, and `threshold == 0` either way; for `n == 1` it is always
   `"greedy"`. Skipping the sweep cannot change `selected`, `f`, `g` or
   `div`. (Case 21: three coincident points under coverage report
   `"diameter_pair"`. The same points under a linear utility report
   `"greedy"` — greedy already holds the top-`k` weights — which no fixture
   pins.)
6. **Exhaustive mode always contains threshold 0**, from the `u == v` pairs
   of `{dist(u,v)/2}` — so with `d_max > 0` the `d = 0` sweep run duplicates
   line 2 and rule 2 relabels it: **`stage == "greedy"` is unreachable under
   `exhaustive_thresholds` when `d_max > 0`**. The qualifier is load-bearing:
   rule 5 skips the whole sweep at `d_max == 0`, so a degenerate point set
   reports `"greedy"` (or `"diameter_pair"`) in this mode like any other. The exhaustive set is sorted ascending and
   exactly deduplicated; its ceiling is `d_max/2`. (Case 19 pins only that
   the exhaustive set is used at all: its winning threshold `1.5 =
   dist(6, 9)/2` is not an entry of the geometric grid for `d_max = 8`,
   `eps = 0.1` — the nearest grid entries are `1.3809` and `1.5190` — so a
   port sweeping the grid instead reports `1.3809` and fails the `threshold`
   field. The `0` entry — and with it the unreachability of `"greedy"` — and
   the `dist/2` halving are **not pinned**: dropping `0` from the set, or
   not halving it (ceiling `d_max` instead of `d_max/2`), leaves all 22
   cases passing; case 19 still reports `[9, 0, 8]` at `1.5` either way.)
7. **Geometric thresholds by repeated multiplication — never `log` + floor.**
   `D = {(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps}`, built by iterating
   `p *= 1 + eps` in f64 with `eps` and `d_max` widened from f32, each entry
   cast to f32, consecutive duplicates removed. Library `ln`/`powf` precision
   is not portable; the entry **count** is part of this contract. Under
   `diameter == "approx"` the bound `2/eps` widens to **`4/eps`** (the sweep
   must still cover the true `d_max <= 2*d_hat`) and `d_max` in the formula
   is `d_hat`. (The count is pinned by every case whose reported threshold is
   the grid's top entry — 1, 3, 4, 5, 7, 9, 10, 18: at `eps = 0.1` that is
   `i = 31`, `1.1^31 = 19.19 <= 20 < 1.1^32`. The `4/eps` widening is **not
   pinned**: case 20's winning threshold is entry 31, the last one inside the
   `2/eps` grid, so a port keeping `2/eps` under approx passes all 22 cases.)
8. **FacilityLocation:** `sim(i, j) = max(0, 1 − dist(i, j)/scale)` with
   `sim(i, i) = 1`; **`scale = 1.0` for cosine** (the paper's own
   `s(i,j) = 1 − dist`, clamped at 0) and **`scale = d_max`, the exact
   diameter, for euclidean** — exact whatever `diameter` mode is selected,
   see rule 10; `d_max == 0` makes `sim ≡ 1` (the scale falls back to 1.0).
   `g(S) = Σ_i max_{j∈S} sim(i, j)`, `g(∅) = 0`; the marginal of `v` is
   `Σ_i max(0, sim(i, v) − best_i)` with `best_i = max_{j∈S} sim(i, j)` (0 for
   `S = ∅`), which divsel evaluates in f64. `utilities` is `null` for this
   utility — it is built from the vectors alone. (Cases 14–16 pin the scale
   choice. The `max(0, ·)` clamp is **not pinned**: case 14 has 12 pairs with
   cosine distance above 1, yet a port that omits the clamp still reproduces
   cases 14–16. `d_max == 0` under facility location is not pinned either.)
9. **Diameter.** Under `diameter == "exact"`, line 3 is the exact pairwise
   maximum over `u < v`, reduced under the total order *larger distance,
   then smaller `u`, then smaller `v`* — so **diameter ties resolve to the
   lexicographically smallest pair**; `n == 1` gives `(0, 0, 0)`. (Case 22
   pins the tie order: `d_max = 8` is realised by both `(0, 3)` and `(1, 3)`
   and the reported selection is the pair itself. Case 21 pins it too: under
   the largest-pair order its pair would be `(1, 2)` with `g = 5`, no longer
   strictly better than greedy. Case 3 does **not** pin it — its `[0, 3]` is
   re-found by the sweep under either order.)
   Under `diameter == "approx"`, line 3 is a farthest-point double sweep:
   starting at `cur = 0`, each sweep takes `a = argmax_{j != cur} dist(cur, j)`
   then `b = argmax_{j != a} dist(a, j)` — the source index is **excluded**,
   so the pair stays distinct even when every distance is 0; argmax ties →
   lowest index — keeps `(dist(a, b), min(a, b), max(a, b))` under the same
   total order, and continues from `b`. `sweeps == 0` is treated as 1, and
   `sweeps > n` as `n` — the orbit `current_{t+1} = b(current_t)` lives in a
   set of size `n`, so it repeats within `n` steps and no later sweep can
   change the answer; the clamp is result-preserving and a port may omit it
   (it exists so that an unvalidated `sweeps` cannot become a call that never
   returns). `d_hat ∈ [d_max/2, d_max]`. (Case 20 pins `d_hat`: the sweep from index 0
   returns `(0, 1)` at `8.00390…`, not the exact `8.20061…` at `(2, 3)`, and
   `expected_d_max` plus the winning threshold — an entry of the grid built
   from `d_hat` — follow from it. It does **not** pin `sweeps > 1` (the first
   sweep already finds `(0, 1)`) or the source-index exclusion (no approx
   fixture has coincident points).)
10. **Under approx, `d_hat` replaces `d_max`** in the threshold grid (rule 7,
    with the `4/eps` bound), in `div(|S| <= 1)` (rule 4), in the pair stage's
    reported `threshold` (rule 3) and in the reported `d_max` field.
    **Exception: the FacilityLocation scale of rule 8 stays the exact
    diameter** — `FacilityLocation::new` always computes `pts.diameter()`,
    whatever `GistConfig::diameter` says, and the Python binding builds the
    utility the same way. (Case 20 pins the reported `d_max == d_hat` and the
    grid built from it. **Not pinned by any fixture:** the
    `approx + facility_location` combination, and the `div(|S| <= 1) = d_hat`
    fallback — case 20's selection has two points, so its `div` is a
    pairwise distance.)
11. **Cloning a utility must be faithful.** divsel's parallel sweep hands each
    worker a clone of the *reset* utility (`Utility::boxed_clone`); any port
    that parallelises must guarantee a worker never observes another's
    selection state, or its sweep results — and therefore rule 2's fold —
    diverge.
12. **`k > n` clamps** to `n` (the constraint is `|S| <= k`); the result can
    also be *shorter* than `k` when no candidate clears the threshold.
    (Cases 8 and 3 respectively.)
13. **Validation — every one of these is an error, never an empty result.**
    `n == 0` (an empty point matrix) → `EmptyInput`; `k == 0` → `InvalidK`;
    `eps` outside `f32::EPSILON <= eps <= 1` → `InvalidEps` (so `NaN`, `0`,
    negatives and anything above 1 are rejected; `eps == 1` is accepted). The
    **lower** bound is a `[divsel choice]` forced by the grid of rule 7: the
    entries are f32 built by repeated multiplication, so below
    `f32::EPSILON = 1.1920929e-7` two consecutive entries cannot differ and `|D|`
    runs away toward the count of representable f32s in the range — below
    `2^-53`, `1 + eps` is `1` and the loop never advances at all. A port that
    accepts an arbitrarily small `eps` hangs or dies on the allocator instead of
    reporting an error; `lambda` not
    finite or `< 0` → `InvalidLambda` (`NaN` and `+inf` rejected); then the
    utility's own checks (weights length `== n`, each weight finite and
    `>= 0`; coverage rows `== n`, ids below the universe). divsel's core
    checks `k`, `eps`, `lambda`, then the utility; the Python binding raises
    `ValueError` for all of them, the empty matrix included, and additionally
    rejects a `bool` **`k` or `diameter_sweeps`** with `TypeError` (Python's
    `True` would otherwise pass as `1` — one sweep, or a budget of one) — a
    binding-level **`[divsel choice]`** for both, since the core's `k` and
    `sweeps` are plain `usize`. A value of either that does not fit a signed
    64-bit integer is a `ValueError`, never an `OverflowError`, whatever
    int-like it arrived as; the same holds for a **coverage item id** outside
    `[0, 2**32 - 1]`, which is a range error naming its row and never a
    `TypeError` about the shape of the argument. (No fixture
    exercises an error — every case is a valid input — so a port must reject
    these on its own.)
14. **Cosine rows are L2-normalised on construction**; a row that cannot be
    normalised (zero norm, or any non-finite coordinate anywhere) is
    **rejected as an error**. Distance is `clamp(1 − a·b, 0, 2)` on the
    normalised rows; `dist(i, i) == 0` exactly. (Cases 13–14.)
15. **Line 4's candidate set is non-strict:
    `C = {v ∈ V \ S : dist(v, S) >= d}`**, with `dist(v, ∅) = +∞` (the first
    pick is unconstrained) and a selected point never re-admitted (which
    matters at `d = 0`). Two points exactly `d` apart are feasible. (Case 19
    pins it: the winning exhaustive threshold is `1.5 = dist(6, 9)/2`
    exactly, and the selection `[9, 0, 8]` admits point 8 at
    `dist(8, 9) = 1.5 == d`, so `expected_div == expected_threshold == 1.5`.
    A port using strict `>` reproduces case 19's selection at a different
    threshold and fails its `threshold` field; every other case passes.)
16. **Linear (modular) utility: `g(S) = Σ_{v∈S} w_v`**, marginals independent
    of `S`. **`utilities: null` under `utility == "linear"` means uniform unit
    weights** (`w ≡ 1`, so `g(S) = |S|`). Weights are f64, finite and `>= 0`.
    (Cases 1, 3, 8, 9, 10 carry `null`; the other linear cases carry explicit
    dyadic weights.)
17. **Coverage utility: `g(S) = |∪_{v∈S} sets_v|`**, the number of distinct
    item ids covered; the marginal of `v` is the count of its ids not yet
    covered by `S`; an id repeated inside one row counts once; unweighted
    (every item is worth 1); `g(∅) = 0`. `utilities` is one list of
    non-negative integer ids per point. In the Python API — and in the
    fixture, which carries no universe field — the universe is inferred as
    `max id + 1` (0 when every list is empty); the universe only bounds the
    ids, it never changes `g`. (Cases 17, 18, 21.)

18. **`lambda == 0` contributes exactly `0` to `f`, whatever `div` is.**
    divsel evaluates `f(S) = g(S) + lambda * div(S)` as `g(S)` alone when
    `lambda == 0`, instead of forming the product — a **`[divsel choice]`**,
    because `div(S)` really can be `+inf` and `0 * inf` is `NaN` in IEEE-754.
    `Points::new` validates coordinates, not distances, so a point set of
    finite `f32` coordinates far enough apart overflows the squared difference:
    `vectors = [[-3e38], [3e38]]`, `metric = "euclidean"`, `k = 2`,
    `lam = 0`, `eps = 0.1` gives `div = +inf`, and divsel reports
    `f = g = 2.0`, `stage = "sweep"`, `threshold = +inf`, `d_max = +inf`. A port
    that writes the formula literally gets `f = NaN` there, and `NaN` then loses
    rule 3's line-5 `>` and every `>=` in rule 2's fold, so it also reports
    `stage = "greedy"` and `threshold = 0` — three fields diverging with no
    error on either side. The same identity is what the Python stub promises for
    `gist_select_full`, so it is qualified there too. (**Not pinned by any
    fixture, and unpinnable by one:** every fixture input is a dyadic rational
    in `[-4, 4]` by construction, so no case can reach an infinite `div`. Rust
    covers it at `gist::tests::lambda_zero_survives_an_infinite_div` and Python
    at `test_api.py::test_lambda_zero_with_an_infinite_div_is_g_not_nan`.)

### Optional: bit-identity (not required for conformance)

divsel's distance kernels use a fixed logical layout of **16 f32 partial
accumulators** regardless of ISA — separate multiply and add (no FMA), tail
elements folded into accumulator `idx % 16`, final reduction in fixed index
order — which makes its results bit-identical across x86_64 and aarch64. A
port that wants bit-identity with divsel must reproduce that exact
accumulation order. **Conformance does not require it**: the `1e-6` relative
tolerance absorbs any reasonable summation order. That paragraph is about
divsel's *own* kernels agreeing with each other on the same f32 arithmetic. It
is **not** the answer to the next section, which is about a port whose
arithmetic is a different *width*.

### Degenerate geometry: divsel's f32 distances are not the exact ones

Every distance divsel computes is `f32`, by contract (`crates/divsel/src/metric.rs`
explains why: it is what makes the SIMD and scalar kernels bit-identical, and it
is what the golden values were generated from). On most inputs that is invisible
to a port. On **degenerate geometry it is not**, and the direction is the one a
port author does not expect:

> **divsel is the less accurate side.** A port computing distances in float64 is
> *more* accurate, and will legitimately disagree with divsel about `selected`
> and `stage` on these inputs.

Three families provoke it, all of them geometry whose exact distances are
integers:

| family | exact distance | divsel's f32 kernel returns |
|---|---|---|
| exact duplicates (`dist == 0`) | `0` | up to `2.384185791015625e-07` |
| exactly-antipodal unit vectors (`dist == 2`) | `2` | `1.9999998807907104`, sometimes `1.999999761581421` |
| signed-axis vectors (`dist ∈ {0, 1, 2}`) | `0` / `1` / `2` | the same last-ulp drift on the `0` and `2` ends |

Measured here, over 30,000 instances across six vector families and both
metrics, the largest gap between divsel's `d_max` and a float64 recomputation
was `1.906e-06` on euclidean (a relative `5.8e-08`) and `2.409e-07` on cosine;
the independent differential above saw `4.77e-07` as the maximum in its own
draw — 4 ulp of `1.0`, 2 ulp of `2.0`, in f32.

**Why a last ulp becomes a different answer.** Line 3's diameter and line 4's
candidate test are *discrete argmaxes* over those distances, resolved by the
total orders in rules 9 and 15. Degenerate geometry makes distances tie
**exactly**, so the tie-break is decided by whichever value the last ulp made
larger — and divsel's f32 kernel and a float64 port disagree about that. The
argmax then amplifies a `6e-08` relative distance difference into a different
pair, a different `selected`, and an `f` that differs by percent. One measured
instance (`n = 27`, `dim = 4`, `k = 4`, `lam = 8.0`, `eps = 0.75`): `f = 21.4375`
from divsel against `21.0076` from the float64 port — a 2% gap. Note which way
it runs: the port's selection scores **higher** under divsel's own arithmetic.

This is not covered by the tolerance rules either. The tolerances above bound
the *value* of an agreed set; here the sets themselves differ, and
`expected_selected` is an exact-equality field on purpose.

**What a port should do.**

* **Safe to compare on anything:** the 22 committed cases. None of them contains
  degenerate geometry, and the robustness margin (below) keeps every one of them
  at least `6.3e-04` relative away from its nearest competitor — four orders of
  magnitude above the last ulp. Passing all 22 is still the conformance bar and
  nothing here relaxes it.
* **Not a bug signal on its own:** a `selected` or `stage` difference on an
  instance containing exact duplicates, signed-axis vectors, or antipodal pairs.
  A generator drawing only uniform random dyadic coordinates never produces
  those families, which is why divsel's fixtures and Aura's committed
  400-instance test have never seen one. A harness whose generator *does* include
  them will: measured at roughly **2% of instances on a degenerate mix** against
  **1 in 30,000 on a realistic one**.
* **How to classify a difference** rather than argue about it: re-run your own
  Algorithm 1 on **divsel's** f32 distance matrix. If it then reproduces
  divsel's `selected` and `stage`, the two implementations are the same
  algorithm and the difference was arithmetic width. That is exactly the method
  the differential above used, on all 1,164 of its differences.
* **To agree with divsel on degenerate geometry at all**, a port must reproduce
  divsel's f32 kernel — the 16-accumulator layout in the section above. That is
  what the optional bit-identity section is for; it is optional precisely because
  conformance is judged on the 22 cases, where it is not needed.

## What each case pins

| # | name | stage | pins |
|---|---|---|---|
| 1 | `line_pick_widest_scaled` | sweep | rule 2 (top threshold re-finds the pair, `>=` relabels); rule 7 count |
| 2 | `weighted_line_middle_threshold` | sweep | rule 2 (exact `f` tie across thresholds → the later one wins) |
| 3 | `rectangle_short_return` | sweep | rule 12 (result shorter than `k`); rule 7 count — *not* the diameter tie order |
| 4 | `pair_reached_by_sweep_tie` | sweep | rule 2 (the pair's win relabelled by the sweep); rule 7 count |
| 5 | `near_duplicate_cluster_high_lambda` | sweep | sweep behaviour on a near-duplicate cluster (regression guard); rule 7 count |
| 6 | `near_duplicate_cluster_lambda_zero` | greedy | `lambda == 0`; rule 3 (`threshold == 0` for `"greedy"`) |
| 7 | `k_one_div_equals_dmax` | sweep | rule 4; `k == 1` skips the pair check; rule 7 count |
| 8 | `k_exceeds_n_returns_all` | sweep | rule 12 (clamp); rule 16 (`null` → uniform) |
| 9 | `sweep_tie_later_threshold_wins` | sweep | rule 2; rule 7 (32 entries at `eps = 0.1`) |
| 10 | `argmax_tie_lowest_index` | sweep | rule 1; rule 7 count |
| 11 | `diameter_pair_wins` | diameter_pair | rule 3 (stage, `threshold == d_max`) |
| 12 | `greedy_wins_outright` | greedy | `"greedy"` reachable under the geometric grid; rule 3 (`threshold == 0`) |
| 13 | `cosine_linear_random` | sweep | rule 14 |
| 14 | `cosine_facility_location_random` | sweep | rule 8 (`scale = 1.0`); rule 14 |
| 15 | `facility_location_euclidean_n8` | diameter_pair | rule 8 (`scale = d_max`); rule 3 |
| 16 | `facility_location_euclidean_n12` | sweep | rule 8 (`scale = d_max`) |
| 17 | `coverage_hand_counts` | sweep | rule 17 |
| 18 | `coverage_exact_tie_lowest_index` | sweep | rules 1, 2, 17; rule 7 count |
| 19 | `exhaustive_thresholds_linear_n10` | sweep | rule 6 (the exhaustive set is used — `threshold == 1.5` is not a grid entry; *not* the `0` entry or the `d_max/2` ceiling); rule 15 |
| 20 | `approx_diameter_double_sweep` | sweep | rule 9 (`d_hat`); rule 10 (reported `d_max`, grid from `d_hat`) — optional |
| 21 | `coincident_coverage_pair_check` | diameter_pair | rule 5 (line 5 still runs at `d_max == 0`); rule 3 (`threshold == 0`); rule 9 (tie order → `(0, 1)`); rule 17 |
| 22 | `diameter_tie_smallest_pair` | diameter_pair | rule 9 (exact tie order, R-G15); rule 3 |

## Optional cases

* **Case 20 (`approx_diameter_double_sweep`)** may be skipped by a port that
  does not implement the approximate-diameter mode — skipping must be stated
  in the port's conformance report. Every other case is mandatory. To pass
  case 20 a port must implement the double sweep of rule 9 exactly (total
  order included), or `expected_d_max` will not match; what case 20 does and
  does not pin is stated under rules 7, 9 and 10.

## Byte stability

* The committed `test-assets/golden-selection.json` is the contract. It is
  written with LF newlines and `.gitattributes` marks `test-assets/*.json
  -text`, so git never rewrites its bytes on any platform (this repo has
  `core.autocrlf` active on Windows — the attribute is what keeps the
  checked-out bytes identical everywhere).
* **Every platform reproduces the committed file's *values***: CI runs both
  readers on Linux, macOS and Windows — the `golden` job
  (`cargo test -p divsel --test golden`, one cell per OS) and the `python`
  matrix (`pytest python/tests/test_golden.py`, 3 OSes x 4 interpreters).
* **Regeneration** (`python python/tools/gen_golden.py --check`) is
  byte-identical: inputs are dyadic (Gaussian draws are excluded by design), the
  random cases use a fixed seed, and floats are written with Python `repr`
  (shortest round-trip, and CPython's own implementation, not the platform's).
  CI gates it on one cell — Linux, 3.12 — while the committed file was generated
  on Windows, so that gate *is* an assertion of cross-platform byte identity, and
  it is expected to hold for a mechanical reason: every value on the path from
  the inputs to the file comes out of `+ - * /`, `min`/`max` and `sqrt`, all of
  which IEEE-754 makes exactly reproducible; no `libm` transcendental is
  involved anywhere; the SIMD kernels are bit-identical to the scalar reference
  by R-G22 (gated on x86_64 and aarch64), and the parallel sweep folds
  deterministically.
* That is divsel's own generator, though, and it is **not** part of the contract
  a port has to meet. A port is judged on the *values* in the file, through the
  readers, on every platform; reproducing the file's *bytes* is nobody's
  obligation but this generator's.

## Robustness margin (why these 22 instances)

No fixture sits on a knife edge a 1-ulp platform difference could flip: the
generator brute-forces every subset of size `<= k` in pure-Python float64 and
requires the best and second-best distinct `f` values to be at least `1e-4`
relative apart, and `expected_selected` to be the unique subset attaining its
`f` value. Deliberate tie cases are exempt — their ties are **exact** dyadic
arithmetic, identical on every platform, and exist precisely to pin the
tie-breaking rules. The margin-exempt set is **cases 2, 3, 10, 18 and 22**:
case 2 (`f({0,2,4}) == f({0,4,5})`, rule 2), case 3 (`f({0,3}) == f({1,2})`),
case 10 (`f({0,2}) == f({1,2})`, rule 1), case 18 (`f({0,1}) == f({0,2})`,
rules 1 and 2) and case 22 (`f({0,3}) == f({1,3})`, rule 9's tie order). Case
9 is **not** exempt although all 32 of its thresholds tie: those ties are
between thresholds producing the same subset, not between distinct subsets,
and it passes the margin at `1.67e-1`. Each case's `note` states its margin or
its exemption.

That margin covers **these 22 instances**. It says nothing about instances a
port's own harness generates: an instance drawn from the degenerate families
*is* a knife edge, by construction, and the section on
[Degenerate geometry](#degenerate-geometry-divsels-f32-distances-are-not-the-exact-ones)
says what to do about it.
