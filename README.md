# bistable-cert

`bistable-cert` is the reference Python implementation of the computable
certificates in *Certified branch-safe selective writing in diffusively
coupled bistable graph memories* (A. Kubański, 2026; prepared for
submission to *Opuscula Mathematica*). The preprint is available at
<https://doi.org/10.5281/zenodo.22903156>.

The package certifies storage of a prescribed binary pattern (Theorem A)
and branch-safe switching of one selected node by a constant pulse
(Theorem B — Theorem 4.3 of the paper — from every initial state in the
certified source box). Certificate decisions use closed-form sufficient conditions;
they do not depend on an ODE simulation. Simulation utilities are included
only for inspection and numerical cross-checking.

## Installation

From the package directory — the one containing `pyproject.toml`
(`bistable-cert/` inside the release bundle):

```bash
python -m pip install .
```

For development and test execution:

```bash
python -m pip install -e ".[dev]"
pytest
```

The distribution name is `bistable-cert`; the Python import name is
`bistable_cert`.

## Reproducibility

The paper's numerical verification script for Section 6 is included at
`reference/reproduce_section6.py`. Every formula in `certificates.py` and
`dynamics.py` is checked against its printed output by
`tests/test_against_paper.py`. The complete release suite contains 111 tests.

## Pre-release audit and correction history

Versions 0.1.0–0.1.5 were internal audit builds under the temporary name
`rlds-network`. The public 1.0.0 release was renamed to avoid confusion
with unrelated RLDS software. The rename and release packaging do not
change the mathematical formulas or certificate logic. The detailed
record below is retained so earlier defects and their regression tests
remain visible; its last section, "Pre-release review of 1.0.0", records
the review of the release candidate itself (input validation added to the
simulation utilities, the corrected description of "Theorem B\*", the
probe's default trial count, and the bundle's directory name).

## Erratum (v0.1.1)

An external review of v0.1.0 found that `certify_storage` checked the
wrong condition: the **a-independent** `eps*q(sigma) < C_beta` (the
paper's eq:Acert) instead of the **a-specific** `eps*(1+a)*q(sigma) <
f_beta(a)` (eq:amarg) that Theorem B's own hypothesis requires ("let a
satisfy eq:common"). eq:Acert only proves *some* floor near `a_star` is
admissible for a pattern — not that the floor a caller actually asks for
(every function defaulted to the paper's own `a=0.65`) is one of them.
Concretely, v0.1.0 could report a write `CERTIFIED` when a non-target
node's true resting equilibrium was already below the claimed floor, with
*zero* pulse applied — confirmed here by direct ODE integration on a
constructed 6-node counterexample, independently of the reviewer's own.

This was a real bug in the package's API layer, not in the paper: the
paper's own `reproduce_section6.py` checked the correct condition all
along. It is fixed in v0.1.1 — `certify_storage` now checks eq:amarg
directly, `admissible_floor_bound()` is added to compute which floors a
pattern actually supports, and every number in this README below was
re-measured after the fix (they differ from v0.1.0's numbers, sometimes
substantially — see each subsection). The review also flagged a grid-
search precision concern (replaced with an exact critical-point solver,
see `certificates.py`) and an invalid `a=0.30` floor used in one probe
script (`a_star≈0.4127` for `beta=2.5`, so `0.30` was never a valid floor
to begin with — fixed). Full detail is in `certificates.py`'s module
docstring and `tests/test_against_paper.py`'s comments, which keep the
old, now-invalid examples on record as regression checks rather than
deleting them quietly.

## Erratum (v0.1.2)

A third external review, of the v0.1.1 bundle, found the eq:amarg fix and
the exact pulse-margin solver both hold up, but the package's *edges* were
not fail-closed: several inputs outside what Theorem A/B actually cover
were silently accepted rather than rejected, and a "CERTIFIED" verdict was
reachable a few float64 ULPs from a true zero. All confirmed by independent
reproduction (not just taken on the review's word) before fixing:

- `amplitude=nan` returned `feasible=True` with `margin=nan`, `T_cert=nan`
  (`nan` compares false against everything, so the old `margin <= 0` guard
  never caught it); `amplitude=inf` was accepted outright. Both now raise.
- A negative `target_node` was silently accepted via Python's own
  negative-indexing (reinterpreted as a node counted from the end, not
  rejected). Now validated to `0 <= target_node < n`.
- `Graph`'s symmetry check used an explicit `atol=1e-12` but left
  `np.allclose`'s default `rtol=1e-5` in effect, making the *effective*
  tolerance ~1e-5, not 1e-12 — a 1e-6 asymmetric matrix was accepted.
  Negativity and the zero-diagonal check used a `1e-12` tolerance that let
  a `-5e-13` weight and a `5e-13` diagonal entry through. All three are now
  exact (negativity, diagonal) or pinned to `rtol=0, atol=1e-9` (symmetry,
  a tolerance for genuine float noise only, not for a 1e-6 error).
- A pattern with two entries both within tolerance of `-1` but not
  float-equal to *each other* (e.g. `-1+3e-13` and `-1-4e-13`) passed
  validation, then was silently miscomputed downstream by exact `==`/`!=`
  comparisons — confirmed to actually flip `opposed_degree`'s output on a
  4-node ring. Validated patterns are now snapped to the exact `±1` they
  were checked against.
- `certify_storage`/`certify_single_node_write` could report `CERTIFIED`
  on a margin only a few float64 ULPs above zero — independently
  reconstructed from scratch here (not the reviewer's own example, which
  didn't come with enough parameters to replicate directly): a random
  graph's margin(U) root, found by exact bisection, then sampled at
  float64 ULPs around that root and cross-checked against an 80-digit
  arbitrary-precision recomputation of the identical formula — 3 sign
  disagreements found out of ~6000 points tested, all with `|margin|`
  under 1.2e-16. Both functions now require clearing a
  `numerical_safety_tolerance` (default `1e-9` — negligible next to any
  real margin seen in this project, from 0.076 up, but far above float
  noise), not just `> 0`.
- `admissible_floor_bound()` crashed with `ValueError: f(a) and f(b) must
  have different signs` on a homogeneous pattern (`q_sigma=0`, where the
  true answer is exactly `s_plus`) and on a load within ~1e-12 of `C_beta`
  (where the true root sits closer to `a_star` than the old fixed bracket
  padding). Rewritten to evaluate the (removable) endpoint singularities
  directly instead of padding around them — `h_beta(a_star)` agrees with
  `C_beta` bit-for-bit, so the true, unpadded bracket is safe to use.
- `CellModel(beta=20)` (a value the model accepts — only `beta > 1` is
  required) crashed computing `s_plus`: for `beta` above ~19–20,
  `tanh(beta*(1-1e-15))` itself rounds to exactly `1.0` in float64, so the
  old bracket no longer contained a sign change. Fixed by using exactly
  `1.0` as the upper bracket (`scipy.optimize.brentq` accepts an exact
  zero at an endpoint as the root itself, with no sign-change needed) —
  confirmed correct through beta=100; unchanged at the paper's beta=2.5.

All of the above are now regression-tested in `tests/test_validation.py`
against the pre-fix behavior, not just the post-fix one. Two wording/
methodology issues in the probe scripts (not the package) were fixed at
the same time — the "roughly 4x" floor-vs-`d_perp*` claim below, and the
sensitivity-sweep's mechanism attribution below — see those two
subsections directly for the corrected numbers and reasoning rather than
restating them here. `T_true`/"branch preserved" wording in the
probe scripts now says "observed equilibrium-start hitting time" and "no
violation found in the sampled points" at least once before reverting to
shorthand, since both are narrower claims than the old phrasing implied
(see `probe_common.py`'s module docstring). One more probe-side check was
added while fixing this: whether the target node, once it first crosses
the floor, stays above it through `T_cert` — which briefly *appeared* to
fail on 39/179 certified writes on the first re-run, traced to a 1-ULP
float artifact in the new check itself (evaluating the crossing instant a
second time, through a different code path, can round a hair either side
of the floor) rather than anything physical, and fixed with the same
numerical-tolerance treatment as above before the number below was
reported.

## Erratum (v0.1.3)

A fourth external review, of the v0.1.2 bundle, confirmed the eq:amarg fix,
the exact pulse-margin solver, and every v0.1.2 fail-closed check hold up
under the standard paper scenario (beta=2.5, a valid graph, default
parameters) — but found six more edges, all independently reproduced
before fixing:

- A **negative `numerical_safety_tolerance`** was accepted outright and
  could push a raw-negative margin over the CERTIFIED line — confirmed: a
  storage margin of `-0.0546538`, and separately a write with
  `amplitude=0` giving `margin=-0.257158` and a **negative** `T_cert=
  -6.41628`, both reported `feasible=True` at `tolerance=-1.0`. Now
  validated finite and `>= 0` in both `certify_storage` and
  `certify_single_node_write` (`0` itself stays legal, as the documented
  opt-out to the raw condition).
- **`Graph` still stored a raw, slightly-asymmetric matrix** within the
  v0.1.2 tolerance — confirmed: `[[0,0],[9e-10,0]]` (just inside
  `atol=1e-9`) was accepted as-is, so `opposed_degree` disagreed with
  itself by row about whether the edge existed, `repr` undercounted edges
  (1 nonzero entry, not 2), and `certify_storage` reported `CERTIFIED` off
  the zeroed-out row. Tightening the tolerance further only moves the same
  problem to a smaller adversarial value, so the fix is structural: the
  matrix actually stored, once confirmed close enough to symmetric to
  accept, is always its exact symmetrization — never the caller's raw
  input.
- **`search_amplitude`'s `tol` was unvalidated** — confirmed: `tol<0` hung
  the bisection loop forever (the loop invariant `b-a >= 0` is always
  greater than a negative number), and `tol=nan`/`inf` skipped the loop
  entirely and silently returned `hi` as though it were the minimal
  amplitude found. Now validated finite and `> 0`, plus a 200-iteration
  cap regardless (even a tiny-but-valid `tol` can stop making progress
  once `b-a` hits float64's own precision floor).
- **`admissible_floor_bound` could return a floor `certify_storage`'s own
  default tolerance then rejected** — confirmed: a load placed `1e-12`
  below `C_beta` got a finite bound back, but `certify_storage` at exactly
  that floor, default tolerance, said NOT CERTIFIED. Both functions now
  share the same `numerical_safety_tolerance` parameter and
  `admissible_floor_bound` solves the correspondingly shifted condition,
  so a floor it returns is one `certify_storage` (same tolerance) actually
  accepts. (Documentation fix alongside this: an earlier note claimed
  `h_beta(a_star)` matches `C_beta` "bit-for-bit" as a general fact —
  checked across a wider beta range, it doesn't; it only happened to match
  exactly at the paper's own beta=2.5. The function's correctness never
  actually depended on that identity, only the docstring's explanation of
  it did, and that's now fixed.)
- **`Graph.from_edges` still accepted a negative node index** — confirmed:
  `(-1, 0, 0.5)` silently became an edge to the LAST node via plain Python
  negative-indexing, the same class of gap v0.1.2 fixed for `target_node`
  but this constructor's own loop had its own, separate index use. Now
  validated against `0 <= i,j < n` before any edge is built.
- **`beta` right at 1 lost catastrophic precision** — confirmed:
  `beta=nextafter(1.0, 2.0)` (accepted by the old `beta > 1` check alone)
  computed `a_star=0.0` EXACTLY, when it should be a tiny *positive*
  number; `sqrt(1+1ULP)` itself already rounds to exactly `1.0` in
  float64. Checked against an 80-digit reference across a beta range:
  `C_beta`'s relative error (a subtraction of two quantities that both
  `-> 0` at the same rate as `beta -> 1`, so a classic cancellation setup)
  is already ~1.6e-8 at `beta=1+1e-4` and exceeds 25% by `beta=1+1e-8`.
  `CellModel` now requires `beta - 1 >= 1e-3` (documented on
  `CellModel.MIN_BETA_MINUS_ONE`, measured at ~1e-10 relative error there,
  comfortable margin to spare) rather than attempting an asymptotic
  rewrite of the closed forms near their own branch point — the paper's
  own working value, beta=2.5, is nowhere near this edge.

Two documentation-only fixes from the same review: `simulation.py`'s
`nontarget_margin` docstring said a positive result means a non-target
node "stayed strictly inside its certified branch for the whole simulated
window" — overstated in exactly the way the probe scripts' own wording was
already corrected for in v0.1.2 (it's a dense-sample check, not a
continuous one; fixed to say so); and this file previously explained two
of the v0.1.2 fixes in full twice — once in that Erratum, once again at
each fix's own subsection — trimmed to a single explanation with a pointer
between them, here.

## Erratum (v0.1.4)

A fifth external review, of the v0.1.3 bundle, again confirmed the
standard paper scenario (beta=2.5, a valid graph, default parameters)
passes cleanly, and characterized its three remaining findings as
adversarial-parameter territory this README already disclaims (an
ever-finer boundary value, an ever-more-extreme magnitude, an ever-less-
standard input type — the chase has no natural end, and a research
package accompanying a paper doesn't owe it one). That framing is right in
general, but treating all three as equally out of scope wasn't checked,
just asserted — so each was checked independently before accepting the
recommendation to stop here:

- **`Graph.from_edges` given a single-use iterator** (a generator
  expression, a `map`/`zip` object) instead of a list — confirmed this is
  not adversarial-parameter territory at all: it's a plain, idiomatic way
  to call the documented `[(i, j, weight), ...]` contract, and it silently
  built an all-zero, zero-edge graph with no error. The v0.1.3 fix for the
  negative-index bug made this function iterate `edges` twice (validate,
  then build `W`), which exhausts a single-use iterator on the first pass.
  Fixed by materializing `edges` into a list up front — see graph.py's own
  CORRECTNESS NOTE (v0.1.4) and
  `test_from_edges_accepts_a_single_use_iterator`.
- **An edge weight near float64's max (~1e308)** — checked directly: it
  does trigger a `RuntimeWarning: overflow encountered in add` from the
  symmetrization step (cosmetic; nothing catches or suppresses it), but
  the overflow propagates to the load term as `inf`, and `certify_storage`
  correctly reports `NOT CERTIFIED` with `margin=-inf` rather than
  anything resembling a false `CERTIFIED`. Fail-safe direction confirmed;
  left as a documented limitation rather than a fix, since silencing the
  warning without addressing the underlying non-finite arithmetic would
  only hide it, not resolve it.
- **A value placed right at the v0.1.2 symmetry tolerance's own boundary**
  (`atol=1e-9`, not the specific `9e-10` case v0.1.3 already fixed) —
  checked both sides directly: an asymmetry of `9.999999e-10` (just
  inside) is accepted and stored as its exact symmetrization, and
  `1.0000001e-9` (just outside) is correctly rejected. The v0.1.3
  structural fix (always store the symmetrization, never the raw input)
  generalizes past the one originally-reported value, as intended.

`admissible_floor_bound`'s own residual gap (documented in its
CORRECTNESS NOTE (v0.1.3)) was also re-checked against this review's
characterization of it as producing an excess rejection rather than a
dangerous false certificate: it does, and provably so, not just by
observation — the function shifts by the unscaled
`numerical_safety_tolerance`, while `certify_storage` only requires
clearing the smaller scaled `tol/(1+floor)`, so the floor returned is
never looser than what `certify_storage` actually accepts, only
potentially tighter.

Net effect: two of the three remaining findings were legitimately out of
scope for this package, confirmed rather than assumed; one was not, and
is now fixed. v0.1.4 is the version this package stops at pending any
further external review.

## Erratum (v0.1.5)

**This corrects the v0.1.4 bullet above about an edge weight near
float64's max.** That bullet checked the claim against one pattern
(alternating signs) and reported the overflow fails closed. A follow-up
to the fifth external review checked a second pattern and it isn't
fail-closed there:

- **The same `~1e308` overflow, with a HOMOGENEOUS pattern** — confirmed
  reproducible: `certify_storage` reported `CERTIFIED` on
  `W=[[0,1e308],[1e308,0]]`, `pattern=[1,1]`. `opposed_degree` selects a
  node's edges to *opposite*-signed neighbors via boolean masking
  (`self.W[k][pattern != pattern[k]]`); for a homogeneous pattern nothing
  is opposite-signed, so it selects nothing and returns an exact `0.0` —
  never touching the `inf` that `W+W.T` overflowed to. The certificate
  this produces is `CERTIFIED` for a graph that, per this package's own
  constructor docstring, should not have been constructible at all ("W
  must contain only finite weights"). Root cause: `Graph.__init__`
  checked finiteness on the caller's raw input, before the
  `W = (W + W.T) / 2` symmetrization step that can itself overflow two
  large-but-finite entries into `inf` — so the check and the thing
  actually stored could disagree. This is the same class of gap v0.1.3
  fixed for the symmetry check itself (checking the pre-symmetrization
  input let an asymmetric raw matrix through; here, checking
  pre-symmetrization finiteness let a matrix that overflows *during*
  symmetrization through). Fixed the same way: finiteness is now checked
  on the matrix actually stored, after symmetrization.

  Unlike the previous bullet's alternating-pattern case, or either of the
  two v0.1.4 findings correctly left as documented limitations, this one
  is a false `CERTIFIED` — the exact failure mode this package exists to
  rule out — so it's fixed rather than documented, regardless of how
  unusual `~1e308` is as an edge weight. A documented "don't use extreme
  magnitudes" caveat is a reasonable thing to ask of a caller; silently
  handing back a wrong CERTIFIED when they do isn't.

Also from the same follow-up: `admissible_floor_bound`'s own docstring
called its return value "the supremum a_max" outright. That overclaims
what the tolerance-padded computation actually guarantees — because of
the padding (see its CORRECTNESS NOTE (v0.1.3)), the true supremum of the
raw condition can be larger than what this function returns, though never
smaller. Reworded to "a conservative admissible floor bound" throughout,
with the guarantee stated in one direction only (every floor it returns
*is* admissible; it does not claim to find the largest one). No
computational change, only what the docstring claims.

Lesson taken from this round specifically: "checked directly" (v0.1.4's
own words, one paragraph up) meant one pattern, not all reachable ones —
worth naming plainly rather than quietly fixing and moving on. 82 → 83
tests. Filed as v0.1.5 rather than amending v0.1.4 in place, so this
correction has its own paper trail instead of rewriting what v0.1.4
claimed.

## Pre-release review of 1.0.0

An external review of the 1.0.0 release candidate, before the first
upload, re-ran the full chain (checksums, wheel against source, 83/83
tests from source and from the installed wheel, the Section 6
reproduction, the 500-graph and topology probes, and an independent
check of the pulse-margin minimiser on 1750 parameter combinations,
maximum deviation ~1.8e-15) and confirmed it. No certificate formula
changed as a result. It raised two points, both confirmed here:

- **The simulation utilities had no input validation.** Reported:
  `simulate_write(target_node=-1)` ran but applied the pulse to *no* node
  (the pulse was built with `np.arange(n) == -1`, which never matches,
  while the trace used Python's negative indexing) — the trajectory was
  bit-identical to a no-pulse run; and a negative `duration` was accepted.
  Checking the rest of the module found more of the same: `duration <
  -extra` integrated the ODE *backward* in time; `duration` or `amplitude`
  = nan, `duration` = inf and `equilibrium(horizon=nan or inf)` hung
  instead of raising; a negative `amplitude` and a `floor` outside
  `(a_star, s_plus)` were accepted although the certificate layer rejects
  both; `equilibrium(residual_tol=nan)` silently disabled its own
  settledness check; and `nontarget_margin` could not skip a negative
  target index. None of this could produce a false `CERTIFIED` — the
  module never certifies — but each case simulated a different system
  than the one requested, or never returned. Fixed by applying the
  certificate layer's own checks (`_check_node_index`,
  `_validate_pattern`, the `floor` domain, `amplitude` finite and `>= 0`)
  plus finite, non-negative durations and positive tolerances; results
  for valid inputs are bit-identical to before. 83 → 111 tests, each
  confirmed to fail (or time out) against the pre-fix module.
- **The README described "Theorem B\*" wrongly** — see "On the name B\*"
  below.

Found while checking the review's numbers: they match the archived
`probes/probe_results.json` exactly, but `certify_probe.py` defaulted to
400 trials (and its own docstring said `--trials 400`), while that file
and the numbers quoted below come from 500. Running the probe as
documented therefore gave different numbers (174 storage-certified, 143
write-certified). The default is now 500, and `--trials 500 --seed 0`
reproduces the archived file record for record. The review also noted,
correctly, that the probe's `T_cert`/hit-time ratios are measured from
the stored equilibrium, not from the least favourable certified start —
now stated where those ratios are reported.

Also found while re-verifying: in the release bundle this directory was
named `bistable_cert/`, the same as the import name, and with an editable
install, starting Python from the bundle's root made it shadow the
installed package as an empty namespace package (`from bistable_cert
import Graph` → `ImportError`). The bundle now names it `bistable-cert/`;
see the bundle's `BUNDLE_README.md`.

## What this is

Given a finite weighted graph of diffusively coupled bistable cells, this
package answers, **before applying anything**:

- `certify_storage(graph, pattern)` — is `pattern` a certified stable
  memory state? (Theorem A)
- `certify_single_node_write(graph, pattern, target_node, amplitude)` — does
  a constant pulse of the given amplitude, applied to `target_node` alone,
  provably switch it while every other node provably keeps its own branch
  throughout the transient, from every initial state in the certified
  source box, off-diagonal ones (`x_i ≠ y_i`) included? (Theorem B —
  Theorem 4.3 of the paper)

Both answers come from the theorems' closed-form sufficient conditions,
evaluated without integrating the network. Numerical integration
(`bistable_cert.simulation`) is available separately, to inspect or
cross-check a certificate — it never decides one.

## What this is not

Version **1.0.0** deliberately implements the Theorem A/B certificate
layer only. It does not implement:

- Section 5's crosstalk bounds (Theorems C/C\*) — the non-target
  displacement envelope
- multi-node writes, write-path planning, or anything that chains several
  single-node writes together

These are natural next layers, but each is its own scope decision, not a
missing feature of this release. See "Status" below for the validation
boundary of version 1.0.0.

**On the name B\*.** The release candidate of this README listed "Theorem
B\*" here and described it as robustness over an interval of admissible
amplitudes. Both parts were wrong. In the paper's development, B\* was the
extension of Theorem B from the diagonal start to the whole certified
source box — off-diagonal initial states, `x_i ≠ y_i` — and that extension
is now part of Theorem 4.3, which is exactly what
`certify_single_node_write` certifies (the `WriteCertificate` docstring
states the full-box guarantee). `reference/reproduce_section6.py` is an
unmodified copy of the paper's own script and still uses the older label:
its full-box corner start is marked "Theorem B\* regime" and its three
write checks B\*1–B\*3. It evaluates the same margin and the same
`T_cert = (1+a)/m` for the equilibrium start and for that corner start.

Robustness in the amplitude needs no separate result for this model: the
pulse margin is nondecreasing in `U` (∂g/∂U = β·sech²(β(r+U)) > 0, and the
two storage conditions do not involve `U`), so a write certified at
amplitude `U` is certified at every larger amplitude, with a `T_cert` no
larger. `search_amplitude` relies on the same monotonicity.

## The central caveat

**`feasible=False` never means "impossible."** Theorem A and Theorem B are
*sufficient* conditions. When a certificate comes back infeasible, it means
this theorem does not certify the write at these parameters — not that no
pulse, no amplitude, and no duration could ever do it. `search_amplitude()`
is the first thing to try before concluding a write is out of reach.

## How conservative is the sufficient condition, really?

Before writing this package, two probe scripts measured this empirically
(not derived — measured, on top of the already-proved theorem). **All
numbers below are from v0.1.1's corrected `certify_storage` (see the
Erratum above); they differ from v0.1.0's, in one case substantially.**

**On 500 random connected graphs** (4–12 nodes): of the 215 with a
storage-certified source pattern, **83.3%** of attempted single-node
writes were certified by Theorem B at all (v0.1.0 claimed 92.6% — the gap
is cases the corrected check now correctly excludes). Of the 179 that
were, the certified duration `T_cert` was typically **3.28x** the actual
time needed (median — essentially unchanged from v0.1.0's 3.3x), with
82.7% under 5x and 96.6% under 10x (also essentially unchanged). **The
tail is where the correction actually shows up**: the maximum ratio
observed dropped from v0.1.0's ~8765x to **22.7x**. The astronomical
outliers were, on inspection, exactly the cases the old a-independent
check let through despite the pattern being right at the edge of (or past)
real storage certification at `a=0.65` — border cases that are now
rejected outright as `NOT CERTIFIED` rather than accepted with a wildly
inflated bound.

**What "actual time needed" measures.** Every probe write starts from the
stored equilibrium — one point of the certified source box, and a
favourable one — while `T_cert` has to cover every start in the box. So
these ratios are not the overhead relative to the least favourable
certified start. On the paper's own 7-node example
(`reference/reproduce_section6.py`, `T_cert = 7.6801`), the equilibrium
start reaches the floor at 0.376·`T_cert` (≈2.66x), while the script's
full-box corner start — `x` and `y` at opposite corners, a deliberately
unfavourable start, though not proved to be the worst one — reaches it at
0.488·`T_cert` (≈2.05x).

**Following up on that tail across four explicit graph families**
(chain, ring, star, complete, every node as write target): chain and ring
stay essentially flat regardless of size (ratio 2.9x–3.3x from n=4 to
n=20 — low, bounded degree, nothing to trigger). Star and complete are
where the correction changes the story, not just the numbers: v0.1.0
reported a 10-node star hub as `CERTIFIED` with `margin≈0.0096` and a 64x
ratio; that source/target pair is **not actually storage-certified at
`a=0.65`** under the real eq:amarg condition, so v0.1.0's number was
itself a false positive, not just a loose-but-valid bound. Under the
fix, the star family stops producing ANY certified write past n=6 (source
pattern itself fails storage certification at n=8+, at `weight=0.5`), and
complete stops past n=4 (n=6's pattern is still storage-certified, but
every one of its 6 possible single-node writes fails the pulse-margin or
target-certificate step; n=8+ fails storage certification outright). The
worst ratio among writes that ARE genuinely certified in this sweep is
**6.05x** (star, n=4, a leaf — not even the hub). In short: high written-
node degree still predicts trouble, but under the corrected check that
trouble now shows up mostly as outright rejection, not as a certified-but-
extremely-loose bound — the earlier "conservativeness blows up with
degree" framing undersold how much of that blowup was actually invalid
certificates. The mechanism identified for the ones that ARE valid is
unchanged: the margin formula's `eps*(a*d_par - d_perp)` term still grows
with the written node's degree.

**Practical consequence for this API:** every `WriteCertificate` carries
`degree_i`, `d_par`, and `d_perp` alongside `margin` and `T_cert`, so a
caller can see *why* a bound is loose, not just that it is. Treat a
certificate on a high-degree written node as correct but possibly very
conservative, and check `margin` before trusting `T_cert` as a useful
(not just valid) bound.

**Does raising the pulse amplitude fix a bad margin? Only below a hard,
closed-form ceiling — and the ceiling depends on ONE thing, not degree.**
As `U -> infinity`, `tanh(beta(r+U)) -> 1` pointwise, so the margin
function converges to an amplitude-independent limit:

```
m_inf(a) = 1 - (1+eps*d_i)*a + eps*(a*d_par - d_perp)
```

A first pass found this ceiling crossed zero at the same total degree on
both the star and complete-graph families and concluded "degree drives
it." That was the wrong mechanism, caught by isolating degree from the
`d_par`/`d_perp` split directly: fixing a star hub's degree at 4.50 and
varying only how those neighbours split between same-side (`d_par`) and
opposite-side (`d_perp`) moved the ceiling from **-0.393 to +0.020** —
same degree, different split, different verdict. Substituting
`d_i = d_par + d_perp` into the formula shows why: the `d_par` terms
cancel completely, leaving

```
m_inf(a) = 1 - a - eps*(1+a)*d_perp
```

— a function of `d_perp` **alone**. This gives a universal,
topology-independent threshold, `opposite_side_degree_threshold()`:

```
d_perp* = (1-a) / (eps*(1+a))
```

A written node's OWN margin is capable of being pushed positive by *some*
finite amplitude if and only if its opposite-side weighted degree is below
`d_perp*` — regardless of same-side degree, total degree, or graph family.
At the paper's own `a=0.65, eps=0.10`, `d_perp* ≈ 2.121`. Confirmed against
`certify_single_node_write` at U=1e4 (exact match, on examples that pass
storage certification — see the scope note just below) and against the
formula's predicted sign on 800 random connected weighted graphs (raised
from v0.1.0's 300, since the corrected storage check passes fewer of them,
see the Erratum) — **267/267** storage-certified cases matched exactly
(it's an algebraic identity, not a statistical tendency, so 100% is the
expected result, not a lucky sample).

**Scope note, sharpened in v0.1.1:** `margin_ceiling`/`d_perp*` describe
only the written node's OWN drift — they say nothing about whether the two
endpoint patterns are storage-certified in the first place (a *separate*,
required condition, checked by `certify_storage`). A positive ceiling is
therefore **necessary but not sufficient** for `certify_single_node_write`
to succeed. v0.1.0's demonstration examples (a 9- and a 10-node star)
conflated the two: their ceiling is positive, but neither pattern is
actually storage-certified at `a=0.65` under the real check, so
`certify_single_node_write` correctly reports `NOT CERTIFIED` at any
amplitude regardless of the ceiling's sign — those examples were replaced
with ones that genuinely pass both conditions (`tests/test_against_paper.py`
keeps the old ones too, now as an explicit "positive ceiling does not
imply certified" regression check).

That separation turns out to resolve cleanly: whenever the TARGET pattern
of a write **is** storage-certified at floor `a`, `margin_ceiling` at that
same `a` is *provably* positive — proved directly from the two closed
forms (storage certification bounds the target node's own post-write
opposed degree, which is exactly `d_perp`, and `tanh(beta*a) < 1` always),
not just observed. So "both endpoints storage-certified, yet no finite
amplitude can ever certify the write" cannot actually happen — confirmed
on 3000 random graphs (752 storage-certified pairs, zero exceptions,
smallest ceiling seen 0.076) on top of the proof. `search_amplitude()`
still checks storage certification and `margin_ceiling()` in that order,
so a write that fails storage certification is reported as such — "no
finite amplitude can certify this write" is now understood to be a
defensive branch for a case the model's own algebra rules out once
storage certification already passed, not a scenario this package's
random-graph testing ever exercises.

**How does the threshold itself move with `eps` and the floor `a`?**
`d_perp* = (1-a)/(eps*(1+a))` is closed-form, so this is direct
substitution, not a new measurement — but it's worth stating plainly
rather than leaving a reader to work it out:

- **In `eps`:** exactly inverse — `d_perp*` at `eps=0.02` is 5x the value
  at `eps=0.10`. Weaker coupling always widens the certifiable range.
- **In the floor `a`:** about **7.2x** tighter across the grid used
  throughout this README (`d_perp*=3.79` at `a=0.45` down to
  `d_perp*=0.53` at `a=0.90`, at the paper's `eps=0.10`) — a stricter
  (higher) certified floor shrinks the budget for opposite-side neighbours
  faster than typical `eps` choices would. (An earlier version of this
  README compared `a=0.30` against `a=0.90` and called it "roughly 4x" —
  `a=0.30` is below `a_star≈0.4127` and was never a valid floor to begin
  with, the same invalid grid point the Erratum above already removed
  elsewhere; the raw formula still evaluates there without complaint since
  nothing in this one-line computation calls `_validate_floor`, which is
  exactly how the error slipped in unnoticed. The two valid numbers give
  ~10.2x if you insist on comparing `a=0.30`'s number anyway — correct
  arithmetic on an invalid input is still not a fact about this model — so
  ~7.2x across the actually-valid `[0.45, 0.90]` grid is the number to
  use.) The paper's own `a=0.65` sits roughly in the middle of the model's
  valid range `(a_star≈0.413, s_plus≈0.986)`, not near either edge.
- This holds across the *entire* valid floor range, not just near 0.65:
  the derivation rests on `g_inf(r,a)`'s minimum always landing at `r=a`,
  which requires the coefficient `-(1+eps*d_i)` to stay negative — true
  unconditionally for every `d_i >= 0`, so nothing about the argument is
  specific to the paper's chosen floor.

**Practical reading:** if a write is infeasible at every amplitude
(`margin_ceiling <= 0`), lowering the certified floor `a` moves the
threshold more than adjusting `eps` typically would — worth knowing before
concluding a graph is simply unwritable at that node.

**Does the ~3.3x typical T_cert/T_true conservativeness itself move with
`a` and `eps`?** A paired sweep — the same 120 random graphs/patterns/
write-nodes re-evaluated at each parameter setting, so differences reflect
the swept parameter and not sampling noise — under the corrected storage
check now reads noticeably differently from v0.1.0's version:

  | `a`  | certified / 120 | median ratio |
  |------|-----------------:|-------------:|
  | 0.45 | 89               | 3.62x        |
  | 0.55 | 72               | 3.30x        |
  | 0.65 | 40               | 3.28x        |
  | 0.80 | 9                | 3.70x        |
  | 0.90 | 3                | 3.29x        |

  (`a=0.30`, used in v0.1.0, is removed — it is below `a_star≈0.4127` for
  `beta=2.5` and was never a valid floor; see the Erratum.)

  | `eps` | certified / 120 | median ratio |
  |-------|-----------------:|-------------:|
  | 0.02  | 120              | 4.08x        |
  | 0.05  | 115              | 3.63x        |
  | 0.10  | 40               | 3.28x        |
  | 0.20  | 7                | 3.53x        |
  | 0.30  | 3                | 3.15x        |

- The **median ratio is stable across the entire valid range**, 3.15x–4.08x
  — actually *more* stable than v0.1.0 reported. v0.1.0 claimed a distinct
  jump to 5.8x at `a=0.90`, attributed to high `a` filtering the sample
  down to near-threshold survivors. That jump **does not survive the
  fix**: at `a=0.90` the median is 3.29x, unremarkable. The near-threshold
  cases that drove the old 5.8x figure were disproportionately exactly the
  ones the corrected eq:amarg check now excludes as invalid rather than
  measures as loose — the same pattern as the topology-probe tail above.
- The **far bigger, and still-correct, effect is on how many writes are
  certifiable at all** — this part of the original finding holds up:
  raising `a` sharply shrinks the certified count (89 → 3 from `a=0.45` to
  `a=0.90`), and so does raising `eps` (120 → 3 from `eps=0.02` to
  `eps=0.30`). **Correction:** an earlier version of this README attributed
  the `a`-sweep's drop partly to the pulse margin / `opposite_side_degree_threshold`
  shrinking. A third external review checked the saved sweep data directly
  and found zero margin-failure outcomes anywhere in either sweep's JSON —
  every non-certified case is `source_not_certified` or
  `target_not_certified`, i.e. storage certification (Theorem A, eq:amarg),
  never the pulse margin. That is the actual, single mechanism behind
  *both* sweeps, and it makes sense directly from eq:amarg's own shape,
  not just from the data: `eps*(1+a)*q(sigma) < f_beta(a)` gets harder on
  BOTH sides as `a` grows (the left side grows linearly in `a`, the right
  side `f_beta(a)` shrinks to 0 as `a -> s_plus`), and obviously harder as
  `eps` grows on the left side alone. The pulse margin / `d_perp*` story
  earlier in this README is a real, separate, and correctly-derived effect
  (confirmed above against 800 random graphs) — it governs whether a
  *storage-certified* write's target-node drift alone can be pushed
  positive by amplitude; it just isn't what is driving this particular
  sweep's certified-count numbers, since every write reaching the margin
  check in this sweep already passed the margin check that lets it (see
  the JSON: `not_certified`, the margin-failure bucket, never appears at
  any point in either table).
- **Sample size caveat:** the `a=0.90` and `eps=0.30` rows rest on only
  3 certified writes each (out of 120 paired specs) — the median at those
  two settings is reported for completeness, not as a stable estimate;
  treat the "stable 3.15x–4.08x across the whole range" reading above as
  solid for the well-populated interior of the grid (40–120 certified
  writes) and merely suggestive at its two sparse ends.
- **Independent cross-check:** the external review that found the
  eq:amarg bug separately re-filtered v0.1.0's own saved specs by hand and
  reported 89 / 40 / 9 / 3 certified at `a=0.45/0.65/0.80/0.90` — an exact
  match to this rewritten probe's own from-scratch count at every point,
  despite being produced independently (their manual re-filter vs. this
  script's full re-run through the fixed package). That agreement is
  strong mutual confirmation that both the diagnosis and the fix are right.

The probe scripts that produced these numbers are not part of the package
API. They are included in the release bundle as supporting research
material, not as hidden package functionality.

## Example

```python
from bistable_cert import Graph, CellModel, certify_single_node_write

g = Graph.from_edges(7, [
    (0,1,0.70),(0,2,0.60),(0,4,0.50),(1,2,0.40),(1,3,0.30),(1,4,0.60),
    (1,5,0.20),(2,3,0.50),(2,4,0.30),(2,5,0.40),(3,6,0.30),(4,5,0.50),
    (4,6,0.40),(5,6,0.60),
])  # the paper's own worked example (Section 6), 0-indexed
model = CellModel(beta=2.5, eps=0.10)
pattern = [-1, 1, 1, 1, -1, -1, -1]

cert = certify_single_node_write(g, pattern, target_node=0, amplitude=0.40, model=model)
print(cert)
# WriteCertificate(CERTIFIED, node=0, T_cert=7.6801, margin=0.21484, degree_i=1.800)
```

## Status

Version 1.0.0 is the first public release. Its scientific scope is frozen
to Theorems A and B; the 0.1.x identifiers below refer to the retained
pre-release audit history.

Every formula is transcribed from, and numerically cross-checked against,
`reference/reproduce_section6.py` (a copy of the script behind the paper's
own Section 6, included here specifically so this claim is checkable) —
see `tests/test_against_paper.py`, which pins exact printed values from
that script plus star-hub and complete-graph cases genuinely re-verified
under the corrected storage check (see the Erratum). Run `pytest` before
trusting any change to `certificates.py`, `graph.py`, or `dynamics.py`
against this suite (111 tests across `test_against_paper.py`, `test_api.py`,
and `test_validation.py`); a mismatch there means a transcription error,
not a new numerical result.

**Input validation** (`tests/test_validation.py`): `Graph` rejects
non-finite weights, non-square matrices, and — as of v0.1.2, tightened
after a third review found the old tolerances let through inputs Theorem
A/B doesn't cover — any negative weight or nonzero diagonal entry (exact,
no tolerance) and any asymmetry beyond `atol=1e-9` (pinned `rtol=0`; the
old check's unset `rtol` defaulted to `1e-5`, effectively ~1e-5 tight, not
the intended ~1e-12). A fourth review then found that within that
tolerance, v0.1.2 still *stored* the caller's raw, slightly-asymmetric
matrix rather than an exactly symmetric one, so row-indexed computations
(`opposed_degree`, the Laplacian, …) could disagree with each other about
whether a near-tolerance edge existed at all — as of v0.1.3, the matrix
actually stored (and used everywhere downstream) is always the exact
symmetrization of whatever passed the closeness check, never the raw
input, so this can no longer happen regardless of how tight any tolerance
is. That symmetrization step can itself overflow two individually-finite
large weights (~1e308) into `inf` — a fifth review found this let a graph
containing a literal infinite coupling weight construct successfully and,
for a homogeneous pattern, get reported `CERTIFIED`; as of v0.1.5,
finiteness is re-checked on the matrix actually stored, after
symmetrization, closing that gap regardless of pattern. `Graph.from_edges`
(v0.1.3) also now rejects an out-of-range node index instead of silently
wrapping it via Python's own negative-indexing, and (v0.1.4) materializes
its `edges` argument into a list up front, so passing a single-use
iterator (a generator expression, a `map`/`zip` object) no longer
silently exhausts it into an empty graph.
`opposed_degree`/`split_degree` reject non-`{-1,+1}`
patterns and (v0.1.2) snap accepted ones to the exact `±1` checked against,
so a near-binary pattern can no longer be silently miscomputed by an exact
`==`/`!=` comparison downstream; every node index (`target_node`, and
`split_degree`'s own index) is checked against `0 <= i < n` (v0.1.2 — a
negative index used to be silently accepted via Python's own
negative-indexing); `certify_single_node_write` (v0.1.2) rejects a
non-finite or negative `amplitude` outright, where it used to report
`CERTIFIED` with `margin=nan`/`T_cert=nan` for `nan`, or an
internally-inconsistent finite-looking certificate for `inf`;
`certify_storage`/`certify_single_node_write` (v0.1.2) require a margin to
clear a `numerical_safety_tolerance` (default `1e-9`), not just be `> 0`,
since a margin within a few float64 ULPs of zero is not reliably signed at
all (see the Erratum above); `CellModel` rejects `beta<=1` (no fold exists
there) and `eps<0`, both required finite, with `eps=0` (no coupling)
handled as a genuine edge case rather than a division error; every `floor`
argument of the certificate functions (and, since the 1.0.0 pre-release
review, of `simulate_write`) is checked against the model's valid domain
`(a_star, s_plus)`;
`admissible_floor_bound()` (v0.1.2) no longer crashes on a homogeneous
pattern or a load near `C_beta`; `CellModel(beta=20)` and beyond (v0.1.2)
no longer crashes computing `s_plus`; and
`bistable_cert.simulation.equilibrium`/`simulate_write` raise
`RuntimeError` (rather than returning silently) if the ODE solver reports
failure, or if `equilibrium` settles somewhere that isn't actually at rest
(checked via the residual `||rhs(Z)||`, not just solver status). Both also
validate their inputs the way the certificate layer does (1.0.0
pre-release review): `simulate_write` rejects a negative, out-of-range or
non-int `target_node`, a non-binary pattern, a negative or non-finite
`amplitude`, `duration` or `extra`, an empty simulated window, a `floor`
outside `(a_star, s_plus)` and `n_points < 2`; `equilibrium` rejects a
non-binary pattern and a non-finite or non-positive `horizon`, `rtol`,
`atol` or `residual_tol`; `WriteSimulation` cannot be built with a target
index outside `0..n-1`, and `nontarget_margin` requires a finite `floor`.
