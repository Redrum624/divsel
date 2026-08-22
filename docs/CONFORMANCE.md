# Conformance: how a port proves it implements GIST the way divsel does

divsel is the **reference implementation** of GIST (arXiv:2405.18754v3): it was
written directly from the paper and is backed by a brute-force oracle over 500
instances proving the `(1/2 − ε)` and `(2/3 − ε)` guarantees. A port — Aura's
`gist_select` (Python), limbic's `gistSelect` (TypeScript), or any other —
proves conformance against **`test-assets/golden-selection.json`** (schema 1),
not against divsel's source.

## The procedure

1. Load `test-assets/golden-selection.json`.
2. For each of the 20 `cases`, rebuild the instance from `vectors`,
   `utilities`, `metric`, `utility`, `k`, `lam`, `eps`,
   `exhaustive_thresholds`, `diameter`, `diameter_sweeps` and run your GIST.
3. Compare against the `expected_*` fields under the tolerance rules below.

All inputs are **dyadic rationals** (multiples of 1/64 in [−4, 4]), so every
language parses them into exactly the same f32/f64 values — there is no input
rounding to disagree about.

### Tolerance rules

| Field | Rule |
|---|---|
| `expected_selected` | **Exact** list equality, order included (selection order; ascending for a diameter pair). |
| `expected_stage` | Exact string equality (`"greedy"` / `"diameter_pair"` / `"sweep"`). |
| `expected_f`, `expected_g`, `expected_div`, `expected_threshold`, `expected_d_max` | `abs(actual − expected) <= f_rel * max(1, abs(expected))` with `f_rel = 1e-6`, read from the file's `tolerance` block. |

Every case must pass. The only optional case is listed under
[Optional cases](#optional-cases).

### Minimal reader (TypeScript-flavoured pseudocode)

```ts
const g = JSON.parse(readFileSync("test-assets/golden-selection.json", "utf8"));
const ok = (a: number, e: number) => Math.abs(a - e) <= g.tolerance.f_rel * Math.max(1, Math.abs(e));
for (const c of g.cases) {
  const r = gistSelect(c.vectors, c.utilities, c); // k, lam, eps, metric, utility, ...
  assert.deepEqual(r.selected, c.expected_selected);
  assert.equal(r.stage, c.expected_stage);
  for (const [a, e] of [[r.f, c.expected_f], [r.g, c.expected_g], [r.div, c.expected_div],
                        [r.threshold, c.expected_threshold], [r.dMax, c.expected_d_max]])
    assert.ok(ok(a, e));
}
```

divsel's own readers — `crates/divsel/tests/golden.rs` (Rust) and
`python/tests/test_golden.py` (Python) — implement exactly this procedure and
are the executable reference for it.

## The contract: every `[divsel choice]` a port must reproduce

The paper's Algorithm 1 leaves several details open. divsel fixed them; the
fixtures encode them; a conforming port must make the **same** choices.
(Line numbers count the paper's `function` header as line 1.)

1. **Greedy argmax ties → lowest index.** Whenever two candidates have equal
   marginal gain, the smaller index wins. (Cases 10, 18.)
2. **Sweep fold: ascending `d`, non-strict `>=`.** Line 8 iterates a *set*;
   divsel folds the thresholds in ascending order and line 10 compares `>=`,
   so **the largest threshold attaining the best `f` wins** — including a tie
   with the greedy or diameter-pair incumbent, which the sweep then relabels.
   A parallel sweep must reproduce this exact fold order. (Cases 2, 4, 9, 18.)
3. **Diametrical-pair check: strict `>`, guarded by `k >= 2 && n >= 2`.**
   The pair displaces the greedy solution only when strictly better; it is
   reported in ascending index order. (Cases 11, 18.)
4. **`div(|S| <= 1) = d_max`** — the paper's own definition (Sec. 2), not 0.
   This applies to the empty set and singletons. (Case 7.)
5. **The sweep is skipped when `d_max == 0`** (n = 1 or all points
   coincident): every threshold would be 0 and re-run line 2. Selection is
   provably unaffected; the observable effect is that such inputs report
   `stage == "greedy"` with `threshold == 0`.
6. **Exhaustive mode always contains threshold 0**, from the `u == v` pairs
   of `{dist(u,v)/2}` — so with `d_max > 0` the `d = 0` sweep run duplicates
   line 2 and rule 2 relabels it: **`stage == "greedy"` is unreachable under
   `exhaustive_thresholds`**. The exhaustive set is sorted ascending and
   exactly deduplicated; its ceiling is `d_max/2`. (Case 19.)
7. **Geometric thresholds by repeated multiplication — never `log` + floor.**
   `D = {(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps}`, built by iterating
   `p *= 1 + eps` in f64 with `eps` and `d_max` widened from f32, each entry
   cast to f32, consecutive duplicates removed. Library `ln`/`powf` precision
   is not portable; the entry **count** is part of this contract. Under
   `diameter == "approx"` the bound `2/eps` widens to **`4/eps`** (the sweep
   must still cover the true `d_max <= 2*d_hat`).
8. **FacilityLocation:** `sim(i, j) = max(0, 1 − dist(i, j)/scale)` with
   `sim(i, i) = 1`; **`scale = 1.0` for cosine** (the paper's own
   `s(i,j) = 1 − dist`, clamped at 0) and **`scale = d_max` (the exact
   diameter) for euclidean**; `d_max == 0` makes `sim ≡ 1`.
   `g(S) = Σ_i max_{j∈S} sim(i, j)`, `g(∅) = 0`. (Cases 14–16.)
9. **`diameter == "approx"` is a farthest-point double sweep.** Starting at
   index 0, each sweep takes `a = argmax_j dist(cur, j)` then
   `b = argmax_j dist(a, j)` (argmax ties → lowest index), keeps
   `(dist(a,b), min(a,b), max(a,b))` under the total order *larger distance,
   then smaller u, then smaller v*, and continues from `b`. `sweeps == 0` is
   treated as 1. `d_hat ∈ [d_max/2, d_max]`. The exact-diameter reduction
   uses the same total order, so diameter ties resolve to the
   lexicographically smallest pair. (Cases 3, 20.)
10. **Under approx, `div(|S| <= 1) = d_hat`** — the estimate replaces `d_max`
    everywhere it is used, including the fallback of rule 4 and the reported
    `d_max` field. (Case 20.)
11. **Cloning a utility must be faithful.** divsel's parallel sweep hands each
    worker a clone of the *reset* utility (`Utility::boxed_clone`); any port
    that parallelises must guarantee a worker never observes another's
    selection state, or its sweep results — and therefore rule 2's fold —
    diverge.
12. **`k > n` clamps** to `n` (the constraint is `|S| <= k`); the result can
    also be *shorter* than `k` when no candidate clears the threshold.
    (Cases 3, 8.)
13. **`k == 0` is an error** (`InvalidK` / `ValueError`), never an empty
    result.
14. **Cosine rows are L2-normalised on construction**; a row that cannot be
    normalised (zero norm, or any non-finite coordinate anywhere) is
    **rejected as an error**. Distance is `clamp(1 − a·b, 0, 2)` on the
    normalised rows; `dist(i, i) == 0` exactly. (Cases 13–14.)

### Optional: bit-identity (not required for conformance)

divsel's distance kernels use a fixed logical layout of **16 f32 partial
accumulators** regardless of ISA — separate multiply and add (no FMA), tail
elements folded into accumulator `idx % 16`, final reduction in fixed index
order — which makes its results bit-identical across x86_64 and aarch64. A
port that wants bit-identity with divsel must reproduce that exact
accumulation order. **Conformance does not require it**: the `1e-6` relative
tolerance absorbs any reasonable summation order.

## Optional cases

* **Case 20 (`approx_diameter_double_sweep`)** may be skipped by a port that
  does not implement the approximate-diameter mode — skipping must be stated
  in the port's conformance report. Every other case is mandatory. To pass
  case 20 a port must implement the double sweep of rule 9 exactly (total
  order included), or `expected_d_max` will not match.

## Byte stability

* The committed `test-assets/golden-selection.json` is the contract. It is
  written with LF newlines and `.gitattributes` marks `test-assets/*.json
  -text`, so git never rewrites its bytes on any platform (this repo has
  `core.autocrlf` active on Windows — the attribute is what keeps the
  checked-out bytes identical everywhere).
* **Every platform reproduces the committed file's *values***: CI runs both
  readers (`cargo test --test golden`, `pytest python/tests/test_golden.py`)
  on Linux, macOS and Windows.
* **Regeneration** (`python python/tools/gen_golden.py --check`) is
  byte-identical on the generating machine: inputs are dyadic (Gaussian draws
  are excluded by design), the random cases use a fixed seed, and floats are
  written with Python `repr` (shortest round-trip). Cross-platform
  regeneration identity is **not** promised — reproducing the *file* is the
  generator's job on one machine; reproducing the *values* is every
  platform's job through the readers.

## Robustness margin (why these 20 instances)

No fixture sits on a knife edge a 1-ulp platform difference could flip: the
generator brute-forces every subset of size `<= k` in pure-Python float64 and
requires the best and second-best distinct `f` values to be at least `1e-4`
relative apart, and `expected_selected` to be the unique subset attaining its
`f` value. Deliberate tie cases (2, 3, 10, 18) are exempt — their ties are
**exact** dyadic arithmetic, identical on every platform, and exist precisely
to pin the tie-breaking rules; each case's `note` states its margin or its
exemption.
