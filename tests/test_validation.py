"""Input validation added in v0.1.1 in response to an external review's
hygiene list: reject non-binary patterns, beta<=1, eps<0, non-finite
numbers, negative/infinite weights, and out-of-domain floors; special-case
eps=0 rather than dividing by it; surface ODE solver failure and a
non-settled equilibrium instead of returning silently.

v0.1.2 additions (a third external review): NaN/inf amplitude and
out-of-range target_node used to silently produce a "CERTIFIED" result
instead of raising; graph symmetry/negativity/zero-diagonal checks had
tolerances loose enough to accept inputs Theorem A/B does not cover; a
near-binary pattern could be validated but then silently miscomputed via
exact == comparisons downstream; certify_storage/certify_single_node_write
could report CERTIFIED on a margin only a few float64 ULPs from zero, where
the sign itself is not reliably resolved; admissible_floor_bound crashed on
a homogeneous pattern and near-C_beta loads; CellModel(beta=20) crashed
computing s_plus. Every case below was independently reproduced against
the pre-fix code before being fixed -- see certificates.py/graph.py/
dynamics.py's own CORRECTNESS NOTE (v0.1.2) comments for the reproduction
details and root cause of each.

v0.1.3 additions (a fourth external review): a negative
numerical_safety_tolerance could push a raw-negative margin over the
CERTIFIED line (confirmed: -1.0 produced a NEGATIVE T_cert); a graph just
inside the v0.1.2 symmetry tolerance (asymmetry 9e-10) was still stored
asymmetric, not just accepted, corrupting row-indexed computations;
search_amplitude's tol was unvalidated (tol<0 hung the bisection, tol=nan/
inf silently skipped it); admissible_floor_bound's raw bound could be one
certify_storage's own default tolerance then rejected; from_edges still
accepted a negative node index; and beta right at 1 lost catastrophic
precision in a_star/C_beta. See certificates.py/graph.py/dynamics.py's own
CORRECTNESS NOTE (v0.1.3) comments for details.

v0.1.4 addition: a fifth external review, of the v0.1.3 bundle, reported
the standard paper scenario still passes cleanly and that three remaining
items (a single-use iterator passed to from_edges, an edge weight near
float64's max, and a value right at the v0.1.2 symmetry tolerance's own
boundary) looked like the kind of ever-more-exotic adversarial input this
package's README already disclaims. Checked each independently rather
than taking that at face value: the weight-magnitude and tolerance-
boundary cases do hold up (confirmed below and in
test_graph_symmetrizes_instead_of_storing_raw_near_symmetric_input's
neighborhood), but the iterator case does not -- from_edges silently
building an empty graph needs no adversarial value at all, just an
ordinary generator expression, so it's fixed here rather than documented
as a limitation. See graph.py's own CORRECTNESS NOTE (v0.1.4).

CORRECTION (v0.1.5): the v0.1.4 paragraph above says the weight-magnitude
case "holds up" -- checked only against an alternating-sign pattern at the
time. A follow-up to the fifth external review found a homogeneous
pattern is a different story: opposed_degree's boolean masking excludes
same-sign neighbors rather than multiplying them by zero, so it never
touches the inf entries an overflowing W+W.T produces, and returns an
exact q_sigma=0.0 -- certify_storage then reported CERTIFIED off a graph
containing a literal infinite coupling weight (confirmed reproducible).
Unlike the two cases the v0.1.4 paragraph correctly left as documented
limitations, this is a false CERTIFIED, not excess conservatism, so it's
fixed rather than documented: Graph's finiteness check now runs on the
matrix actually stored, after symmetrization, not just on the caller's
raw input before it. Also fixed in the same review: admissible_floor_bound
called its own result "the supremum" outright, which overclaims what the
tolerance-padded computation actually guarantees (a safe lower bound on
the true supremum, not the true supremum itself) -- reworded, no
computational change. See graph.py/certificates.py's own CORRECTNESS NOTE
(v0.1.5) comments for both.

1.0.0 pre-release review (an external review of the release candidate,
before the first upload): the simulation module -- simulate_write,
equilibrium, WriteSimulation -- had no input validation at all. The review
reported target_node=-1 (pulse applied to no node) and a negative duration;
checking the rest of the module found backward-in-time integration, hangs
on nan/inf, a negative amplitude and an out-of-domain floor accepted, and
residual_tol=nan disabling equilibrium's own settledness check. None of
these could produce a false CERTIFIED (the module never certifies), but
each simulated a different system than the one requested, or never
returned. Fixed by applying the certificate layer's own checks; results
for valid inputs are bit-identical to before. See simulation.py's module
docstring.
"""
import numpy as np
import pytest

from bistable_cert import (Graph, CellModel, certify_storage, certify_single_node_write,
                           opposite_side_degree_threshold, search_amplitude)
from bistable_cert.certificates import admissible_floor_bound, NUMERICAL_SAFETY_TOLERANCE
from bistable_cert.simulation import equilibrium, simulate_write, WriteSimulation


def test_graph_rejects_infinite_or_nan_weights():
    for bad in (np.inf, np.nan, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            Graph(np.array([[0.0, bad], [bad, 0.0]]))


def test_opposed_degree_rejects_non_binary_pattern():
    g = Graph.star(4, weight=0.3)
    with pytest.raises(ValueError, match=r"\{-1, \+1\}"):
        g.opposed_degree([1, -1, 0.5, -1])


def test_split_degree_rejects_non_binary_pattern():
    g = Graph.star(4, weight=0.3)
    with pytest.raises(ValueError, match=r"\{-1, \+1\}"):
        g.split_degree(0, [1, -1, 0, -1], target_sign=1.0)


def test_opposed_degree_rejects_wrong_shape():
    g = Graph.star(4, weight=0.3)
    with pytest.raises(ValueError, match="shape"):
        g.opposed_degree([1, -1, 1])


@pytest.mark.parametrize("beta", [1.0, 0.9, -2.0, 0.0, np.inf, np.nan])
def test_cell_model_rejects_beta_at_or_below_one(beta):
    """beta<=1 leaves the uncoupled cell without a fold at all: a_star and
    s_SN degenerate (or go complex/undefined) rather than describing a
    smaller-but-valid bistable regime."""
    with pytest.raises(ValueError, match="beta"):
        CellModel(beta=beta)


@pytest.mark.parametrize("eps", [-0.01, -5.0, np.inf, np.nan])
def test_cell_model_rejects_invalid_eps(eps):
    with pytest.raises(ValueError, match="eps"):
        CellModel(eps=eps)


def test_cell_model_accepts_eps_zero():
    """eps=0 (no coupling at all) is a legitimate, meaningful edge case --
    not something to reject."""
    model = CellModel(eps=0.0)
    assert model.eps == 0.0


def test_opposite_side_degree_threshold_is_infinite_at_eps_zero():
    """With zero coupling, a pulse at the target node cannot reach any
    other node through the graph at all, so no opposite-side degree is
    ever too large -- the threshold is unbounded, not a division error."""
    model = CellModel(eps=0.0)
    assert opposite_side_degree_threshold(model, floor=0.65) == float("inf")


def test_certify_storage_rejects_floor_outside_domain():
    g = Graph.star(5, weight=0.3)
    sigma = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    model = CellModel()
    with pytest.raises(ValueError, match="valid domain"):
        certify_storage(g, sigma, model, floor=0.05)  # below a_star
    with pytest.raises(ValueError, match="valid domain"):
        certify_storage(g, sigma, model, floor=0.999)  # above s_plus


def test_equilibrium_flags_a_pattern_that_never_settles():
    """A pattern with a node whose neighbours overwhelmingly disagree with
    it is not a stable equilibrium of the UNCONTROLLED dynamics at all --
    equilibrium() must say so via a residual check, not return a moving
    point silently labeled 'settled'."""
    g = Graph.star(6, weight=5.0)  # heavily saturated: not storage-certified
    sigma = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])  # hub outvoted 5-to-1
    model = CellModel()
    with pytest.raises(RuntimeError, match="not settled|solver did not complete"):
        equilibrium(g, sigma, model, horizon=5.0)  # too short AND not stable either way


def test_equilibrium_matches_paper_example_within_residual_tolerance():
    """Regression: the validation added above must not reject the paper's
    own genuinely-stable worked example."""
    edges = [(0,1,0.70),(0,2,0.60),(0,4,0.50),(1,2,0.40),(1,3,0.30),(1,4,0.60),
             (1,5,0.20),(2,3,0.50),(2,4,0.30),(2,5,0.40),(3,6,0.30),(4,5,0.50),
             (4,6,0.40),(5,6,0.60)]
    g = Graph.from_edges(7, edges)
    sigma = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    model = CellModel(beta=2.5, eps=0.10)
    p = equilibrium(g, sigma, model)
    assert np.all(np.sign(p) == np.sign(sigma))


# =====================================================================
# v0.1.2: fail-closed amplitude / target_node / graph / pattern checks,
# numerical safety tolerance, and admissible_floor_bound / s_plus fixes.
# =====================================================================

def _paper_graph_and_pattern():
    edges = [(0,1,0.70),(0,2,0.60),(0,4,0.50),(1,2,0.40),(1,3,0.30),(1,4,0.60),
             (1,5,0.20),(2,3,0.50),(2,4,0.30),(2,5,0.40),(3,6,0.30),(4,5,0.50),
             (4,6,0.40),(5,6,0.60)]
    g = Graph.from_edges(7, edges)
    sigma = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    return g, sigma


@pytest.mark.parametrize("bad_amplitude", [float("nan"), float("inf"), -1.0, -1e-9])
def test_certify_single_node_write_rejects_bad_amplitude(bad_amplitude):
    """Confirmed against the pre-fix code: amplitude=nan returned
    feasible=True with margin=nan (nan compares False against everything,
    so the old `m_val <= 0` guard silently let it through); amplitude=inf
    was accepted outright and produced a finite-looking but meaningless
    certificate."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="amplitude"):
        certify_single_node_write(g, sigma, target_node=0, amplitude=bad_amplitude)


@pytest.mark.parametrize("bad_node", [-1, -7, 7, 100])
def test_certify_single_node_write_rejects_out_of_range_target_node(bad_node):
    """Confirmed against the pre-fix code: target_node=-1 was silently
    accepted via Python's own negative-indexing semantics (reinterpreted as
    the LAST node) and returned a normal-looking CERTIFIED result."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="node index|range"):
        certify_single_node_write(g, sigma, target_node=bad_node, amplitude=0.4)


def test_split_degree_rejects_out_of_range_node():
    g = Graph.star(5, weight=0.3)
    sigma = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="node index|range"):
        g.split_degree(-1, sigma, target_sign=1.0)


def test_graph_rejects_asymmetry_too_small_for_the_old_tolerance():
    """Confirmed against the pre-fix code: np.allclose's default rtol=1e-5
    was still in effect despite an explicit atol=1e-12, so a 1e-6 asymmetry
    (5 orders of magnitude above genuine float noise) was silently
    accepted. Fixed by pinning rtol=0."""
    W = np.array([[0.0, 0.5], [0.500001, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        Graph(W)


def test_graph_rejects_tiny_negative_weight():
    """Confirmed against the pre-fix code: -5e-13 was accepted under the
    old `W < -1e-12` tolerance."""
    W = np.array([[0.0, -5e-13], [-5e-13, 0.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        Graph(W)


def test_graph_rejects_tiny_nonzero_diagonal():
    """Confirmed against the pre-fix code: 5e-13 was accepted under the old
    `abs(diag) > 1e-12` tolerance."""
    W = np.array([[5e-13, 0.5], [0.5, 0.0]])
    with pytest.raises(ValueError, match="zero diagonal"):
        Graph(W)


def test_graph_still_accepts_genuine_construction_noise():
    """Regression: the tightened checks above must not become so strict
    they reject an exactly-valid graph built the normal way."""
    g = Graph.ring(6, weight=0.4)
    assert g.n == 6
    g2 = Graph.from_edges(5, [(0, 1, 0.3), (1, 2, 0.5), (2, 3, 0.2), (3, 4, 0.7)])
    assert g2.n == 5


def test_near_binary_pattern_is_snapped_to_exact_signs():
    """Confirmed against the pre-fix code: two entries both within 1e-12 of
    -1 (e.g. -1+3e-13 and -1-4e-13) passed the tolerance check but were not
    float-equal to EACH OTHER, so opposed_degree's exact `!=` comparison
    silently classified them as opposite sides. On ring(4) (where nodes 1
    and 2 are adjacent) this gave [0.5, 1., 1., 0.5] instead of the correct
    [0.5, 0.5, 0.5, 0.5]. Fixed by snapping validated entries to exact
    +-1."""
    g = Graph.ring(4, weight=0.5)
    p_noisy = np.array([1.0, -1.0 + 3e-13, -1.0 - 4e-13, 1.0])
    od = g.opposed_degree(p_noisy)
    np.testing.assert_allclose(od, [0.5, 0.5, 0.5, 0.5])


def test_certify_storage_requires_clearing_numerical_safety_tolerance():
    """A margin that is technically positive but only by a few float64 ULPs
    must not be reported CERTIFIED -- confirmed reproducible (see
    certificates.py's NUMERICAL_SAFETY_TOLERANCE docstring: an independent
    from-scratch search found float64 disagreeing in SIGN with an 80-digit
    mpmath recomputation of the identical formula, at the ~1e-17 scale,
    within a few ULPs of a genuine root). This test constructs a storage
    margin deliberately inside the safety tolerance and checks it is
    rejected by the default, but accepted if the caller explicitly opts
    out (numerical_safety_tolerance=0) and the raw sign is positive."""
    model = CellModel()
    # q_sigma placed so that rhs - lhs is a few multiples of float64 eps,
    # comfortably inside the default 1e-9 safety tolerance but positive.
    target_lhs = model.f(0.65) - 1e-13
    weight = target_lhs / model.eps / (1 + 0.65) / 2
    g = Graph.star(3, weight=weight)
    sigma = np.array([1.0, -1.0, -1.0])
    cert_default = certify_storage(g, sigma, model, floor=0.65)
    assert cert_default.margin_to_threshold > 0  # raw sign is positive
    assert not cert_default.feasible  # but does not clear the safety tolerance
    cert_zero_tol = certify_storage(g, sigma, model, floor=0.65,
                                     numerical_safety_tolerance=0.0)
    assert cert_zero_tol.feasible  # opting out recovers the raw condition


def test_admissible_floor_bound_at_q_sigma_zero_returns_s_plus():
    """Confirmed against the pre-fix code: crashed with ValueError (fixed
    bracket padding stepped past the true root, which sits exactly at
    s_plus when q_sigma=0 -- every floor is trivially admissible for a
    homogeneous pattern, so the supremum of admissible floors is s_plus
    itself)."""
    model = CellModel()
    g = Graph.star(5, weight=0.5)
    sigma_homogeneous = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    bound = admissible_floor_bound(g, sigma_homogeneous, model)
    assert bound == pytest.approx(model.s_plus)


def test_admissible_floor_bound_near_c_beta_does_not_crash():
    """Confirmed against the pre-fix code: crashed with ValueError when
    eps*q_sigma sat within ~1e-12 of C_beta (the padded bracket's lower end
    had already stepped past the true root, which sits extremely close to
    a_star in this regime). Uses numerical_safety_tolerance=0 to isolate
    this robustness fix from the separate v0.1.3 tolerance-consistency
    fix tested below (at the default 1e-9 tolerance, a load THIS close to
    C_beta has no admissible floor left at all -- see the next test)."""
    model = CellModel()
    target = model.C_beta * (1 - 1e-12)
    weight = target / model.eps / 2
    g = Graph.star(3, weight=weight)
    sigma = np.array([1.0, -1.0, -1.0])
    bound = admissible_floor_bound(g, sigma, model, numerical_safety_tolerance=0.0)
    assert model.a_star < bound < model.s_plus
    assert bound == pytest.approx(model.a_star, abs=1e-6)


def test_admissible_floor_bound_matches_old_result_away_from_edges():
    """Regression: the rewritten bracket (no more fixed padding) must give
    the same RAW answer as before (numerical_safety_tolerance=0) in the
    ordinary, non-edge-case regime."""
    model = CellModel()
    g = Graph.star(7, weight=0.4)
    sigma = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    bound = admissible_floor_bound(g, sigma, model, numerical_safety_tolerance=0.0)
    q_sigma = float(g.opposed_degree(sigma).max())
    h_at_bound = model.f(bound) / (1 + bound)
    assert h_at_bound == pytest.approx(model.eps * q_sigma, abs=1e-9)


def test_admissible_floor_bound_default_tolerance_is_consistent_with_certify_storage():
    """v0.1.3: confirmed against the pre-fix code that admissible_floor_bound
    could return a floor certify_storage's own default tolerance then
    rejected (a load placed 1e-12 below C_beta got a finite bound back, but
    certify_storage AT that exact floor, default tolerance, said NOT
    CERTIFIED). At the shared default tolerance, a floor strictly below the
    returned bound must be accepted by certify_storage."""
    model = CellModel()
    g = Graph.star(7, weight=0.4)
    sigma = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    bound = admissible_floor_bound(g, sigma, model)  # default tolerance
    assert certify_storage(g, sigma, model, floor=bound - 1e-6).feasible

    # and the near-C_beta load from the test above: at the DEFAULT
    # tolerance there is no floor left that clears it, so this must
    # honestly report None rather than a bound certify_storage would reject.
    target = model.C_beta * (1 - 1e-12)
    weight = target / model.eps / 2
    g2 = Graph.star(3, weight=weight)
    sigma2 = np.array([1.0, -1.0, -1.0])
    assert admissible_floor_bound(g2, sigma2, model) is None


def test_admissible_floor_bound_rejects_bad_tolerance():
    model = CellModel()
    g = Graph.star(5, weight=0.4)
    sigma = np.array([1.0, -1.0, -1.0, -1.0, -1.0])
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="numerical_safety_tolerance"):
            admissible_floor_bound(g, sigma, model, numerical_safety_tolerance=bad)


def test_cell_model_beta_20_s_plus_does_not_crash():
    """Confirmed against the pre-fix code: CellModel(beta=20).s_plus raised
    ValueError (the old upper bracket, 1-1e-15, no longer brackets the root
    once tanh(beta*(1-1e-15)) itself rounds to exactly 1.0 in float64,
    which happens for beta gtrsim ~19-20)."""
    for beta in (20.0, 40.0, 100.0):
        model = CellModel(beta=beta)
        s = model.s_plus
        assert 0 < s <= 1.0
        assert abs(model.f(s)) < 1e-6  # s is genuinely close to a root of f


def test_cell_model_beta_2_5_s_plus_unchanged():
    """Regression: the s_plus bracket fix must not move the answer for the
    paper's own beta=2.5 -- compared directly against the pre-fix bracket
    (a_star+1e-12, 1-1e-15), not a hand-typed literal."""
    from scipy.optimize import brentq
    model = CellModel(beta=2.5)
    old_bracket_s_plus = brentq(lambda s: model.f(s), model.a_star + 1e-12, 1 - 1e-15)
    assert model.s_plus == pytest.approx(old_bracket_s_plus, abs=1e-9)


# =====================================================================
# v0.1.3 (fourth external review): negative numerical_safety_tolerance,
# symmetrization instead of a tolerance-only symmetry check, search_amplitude
# tol validation, admissible_floor_bound/certify_storage consistency (see
# above), from_edges index validation, beta-near-1 precision guard.
# =====================================================================

@pytest.mark.parametrize("bad_tol", [-1.0, -1e-12, float("nan"), float("inf")])
def test_certify_storage_rejects_bad_numerical_safety_tolerance(bad_tol):
    """Confirmed against the pre-fix code: numerical_safety_tolerance=-1.0
    reported feasible=True off a raw-NEGATIVE margin (-0.0546538)."""
    g, sigma = _paper_graph_and_pattern()
    model = CellModel()
    with pytest.raises(ValueError, match="numerical_safety_tolerance"):
        certify_storage(g, sigma, model, floor=0.65, numerical_safety_tolerance=bad_tol)


def test_certify_single_node_write_rejects_bad_numerical_safety_tolerance():
    """Confirmed against the pre-fix code: amplitude=0 with tolerance=-1.0
    reported feasible=True with a NEGATIVE T_cert (-6.41628) -- a certified
    duration that cannot mean anything physically."""
    g, sigma = _paper_graph_and_pattern()
    model = CellModel()
    with pytest.raises(ValueError, match="numerical_safety_tolerance"):
        certify_single_node_write(g, sigma, target_node=0, amplitude=0.0, model=model,
                                   floor=0.65, numerical_safety_tolerance=-1.0)


def test_graph_symmetrizes_instead_of_storing_raw_near_symmetric_input():
    """Confirmed against the pre-fix code: W=[[0,0],[9e-10,0]] (asymmetry
    just inside the old atol=1e-9) was accepted and STORED AS-IS, so
    opposed_degree/degree disagreed by row about whether the edge existed,
    __repr__ undercounted edges (1 nonzero entry, not 2), and
    certify_storage reported CERTIFIED off node 0's zeroed-out view. Any
    fixed tolerance can be undercut the same way by a smaller adversarial
    value, so the fix is symmetrization, not a tighter threshold: once W is
    confirmed close enough to accept, the matrix actually stored is its
    exact symmetrization."""
    W = np.array([[0.0, 0.0], [9e-10, 0.0]])
    g = Graph(W)
    assert g.W[0, 1] == g.W[1, 0] == pytest.approx(4.5e-10)
    assert "edges=1" in repr(g)
    model = CellModel()
    sigma = np.array([1.0, -1.0])
    cert = certify_storage(g, sigma, model, floor=0.65)
    assert cert.q_sigma == pytest.approx(4.5e-10)  # sees the (tiny) edge from EITHER node now


def test_graph_rejects_weights_that_overflow_during_symmetrization():
    """Confirmed against the pre-fix code: W=[[0,1e308],[1e308,0]] has two
    individually-finite entries, but W+W.T overflows to inf before the /2,
    and the finiteness check (which only looked at the caller's raw input,
    before symmetrization) never saw that. Graph(...) built with no error
    at all, silently storing self.W=[[0,inf],[inf,0]]. Because
    opposed_degree uses boolean masking (excludes same-sign neighbors
    rather than multiplying them by zero), a HOMOGENEOUS pattern never
    touches the inf entries and returns an exact q_sigma=0.0 -- so
    certify_storage reported CERTIFIED off a graph containing a literal
    infinite coupling weight. Fixed by re-checking finiteness on the
    matrix actually stored, after symmetrization."""
    with pytest.raises(ValueError, match="finite"):
        Graph(np.array([[0.0, 1e308], [1e308, 0.0]]))


@pytest.mark.parametrize("bad_tol", [0.0, -0.1, float("nan"), float("inf")])
def test_search_amplitude_rejects_bad_tol(bad_tol):
    """Confirmed against the pre-fix code: tol<0 hung the bisection loop
    (b-a, always >= 0, is always > a negative number); tol=nan/inf skipped
    the loop entirely and silently returned hi as if it were the minimal
    amplitude."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="tol"):
        search_amplitude(g, sigma, target_node=0, tol=bad_tol)


def test_from_edges_rejects_out_of_range_index():
    """Confirmed against the pre-fix code: (-1, 0, 0.5) silently created an
    edge to the LAST node via plain Python negative-indexing."""
    with pytest.raises(ValueError, match="range"):
        Graph.from_edges(5, [(-1, 0, 0.5)])
    with pytest.raises(ValueError, match="range"):
        Graph.from_edges(5, [(0, 5, 0.5)])


def test_from_edges_accepts_a_single_use_iterator():
    """Confirmed against the pre-fix code: passing a generator (or any
    single-use iterator) instead of a list silently built an all-zero,
    zero-edge graph -- the validation pass exhausted it before the
    W-building pass ever ran, and no exception was raised either way."""
    edges_list = [(0, 1, 0.5), (1, 2, 0.7), (2, 3, 0.3)]

    g_from_list = Graph.from_edges(4, edges_list)
    g_from_gen = Graph.from_edges(4, (e for e in edges_list))
    g_from_map = Graph.from_edges(4, map(lambda e: e, edges_list))

    assert np.array_equal(g_from_gen.W, g_from_list.W)
    assert np.array_equal(g_from_map.W, g_from_list.W)
    assert np.count_nonzero(g_from_gen.W) == 6  # 3 edges, symmetric


def test_cell_model_rejects_beta_too_close_to_one():
    """Confirmed against the pre-fix code: beta=nextafter(1.0, 2.0) (which
    only the old `beta > 1` check let through) computed a_star=0.0 EXACTLY
    -- it should be a tiny positive number -- and C_beta's relative error
    vs. an 80-digit reference exceeds 25% by beta=1+1e-8, long before this
    threshold."""
    with pytest.raises(ValueError, match="beta"):
        CellModel(beta=np.nextafter(1.0, 2.0))
    with pytest.raises(ValueError, match="beta"):
        CellModel(beta=1.0 + 1e-8)


def test_cell_model_accepts_beta_comfortably_past_the_documented_threshold():
    """Regression: the paper's own beta=2.5 and a beta just past the
    documented minimum distance from 1 must both still work. Uses 2x the
    threshold, not the exact boundary value, since (1.0 + 1e-3) - 1 is not
    guaranteed to equal 1e-3 exactly in float64 -- testing the precise
    boundary is a float-representation exercise, not what this regression
    is for."""
    CellModel(beta=2.5)
    m = CellModel(beta=1.0 + 2 * CellModel.MIN_BETA_MINUS_ONE)
    assert np.isfinite(m.a_star) and m.a_star > 0
    assert np.isfinite(m.C_beta) and m.C_beta > 0


# =====================================================================
# 1.0.0 pre-release review: the simulation module had no input validation.
# Every case below was reproduced against the pre-fix code first; see
# simulation.py's module docstring for the mechanism of each.
# =====================================================================

_PAPER_MODEL = dict(beta=2.5, eps=0.10)


@pytest.mark.parametrize("bad_node", [-1, -7, 7, True, 6.0])
def test_simulate_write_rejects_invalid_target_node(bad_node):
    """Confirmed against the pre-fix code: target_node=-1 ran without error
    but applied the pulse to NO node -- the pulse was built by comparing
    np.arange(n) == -1, which never matches, while target_sign and the
    reported trace used Python's negative indexing (-1 = last node). The
    trajectory was bit-identical to a no-pulse run. Must now fail exactly
    as certify_single_node_write does for the same index."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="node index"):
        simulate_write(g, sigma, target_node=bad_node, amplitude=0.4, duration=8.0,
                       model=CellModel(**_PAPER_MODEL))


@pytest.mark.parametrize("bad_duration", [-5.0, -20.0, float("nan"), float("inf")])
def test_simulate_write_rejects_bad_duration(bad_duration):
    """Confirmed against the pre-fix code: -5 ran with no pulse at all; -20
    (below -extra) made the horizon negative and solve_ivp integrated
    BACKWARD in time; nan and inf hung instead of raising."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="duration"):
        simulate_write(g, sigma, target_node=0, amplitude=0.4, duration=bad_duration,
                       model=CellModel(**_PAPER_MODEL))


@pytest.mark.parametrize("bad_amplitude", [-0.4, float("nan"), float("inf")])
def test_simulate_write_rejects_bad_amplitude(bad_amplitude):
    """Confirmed against the pre-fix code: a negative amplitude (pushing the
    target the wrong way) was accepted although certify_single_node_write
    rejects it, and nan hung."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="amplitude"):
        simulate_write(g, sigma, target_node=0, amplitude=bad_amplitude, duration=8.0,
                       model=CellModel(**_PAPER_MODEL))


@pytest.mark.parametrize("bad_floor", [5.0, 0.30, float("nan")])
def test_simulate_write_rejects_floor_outside_model_domain(bad_floor):
    """Confirmed against the pre-fix code: floor=5.0 was accepted and simply
    reported hit_time=None. Now checked against (a_star, s_plus) exactly as
    the certificate layer checks its own floor."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match="floor"):
        simulate_write(g, sigma, target_node=0, amplitude=0.4, duration=8.0,
                       model=CellModel(**_PAPER_MODEL), floor=bad_floor)


def test_simulate_write_rejects_bad_pattern_window_and_sampling():
    g, sigma = _paper_graph_and_pattern()
    model = CellModel(**_PAPER_MODEL)
    kw = dict(target_node=0, amplitude=0.4, duration=8.0, model=model)
    with pytest.raises(ValueError, match="pattern"):
        simulate_write(g, [0.5, 1, 1, 1, -1, -1, -1], **kw)
    with pytest.raises(ValueError, match="extra"):
        simulate_write(g, sigma, **kw, extra=-1.0)
    with pytest.raises(ValueError, match="window"):
        simulate_write(g, sigma, target_node=0, amplitude=0.4, duration=0.0,
                       model=model, extra=0.0)
    for bad_n in (1, 0, 2.5, True):
        with pytest.raises(ValueError, match="n_points"):
            simulate_write(g, sigma, **kw, n_points=bad_n)


def test_simulate_write_still_accepts_no_pulse_controls_and_numpy_ints():
    """Regression: amplitude=0 and duration=0 are legitimate no-pulse control
    runs, and a numpy integer target (as produced by loops over np.arange)
    must keep working -- the new checks must not reject any of these."""
    g, sigma = _paper_graph_and_pattern()
    model = CellModel(**_PAPER_MODEL)
    s0 = simulate_write(g, sigma, target_node=np.int64(6), amplitude=0.0, duration=8.0,
                        model=model, n_points=500)
    s1 = simulate_write(g, sigma, target_node=6, amplitude=0.4, duration=0.0,
                        model=model, n_points=500)
    assert s0.hit_time is None and s1.hit_time is None
    assert s0.target_node == 6 and isinstance(s0.target_node, int)


def test_simulate_write_negative_index_regression_value_for_the_valid_index():
    """The node that -1 used to "mean" (the last one, 6) does switch when
    addressed correctly -- the pre-fix -1 run reported hit_time=None for a
    write that was never applied, while this one hits well inside T_cert."""
    g, sigma = _paper_graph_and_pattern()
    s = simulate_write(g, sigma, target_node=6, amplitude=0.4, duration=8.0,
                       model=CellModel(**_PAPER_MODEL))
    assert s.hit_time is not None and s.hit_time < 8.0


@pytest.mark.parametrize("kw", [dict(horizon=float("nan")), dict(horizon=float("inf")),
                                dict(horizon=-50.0), dict(horizon=0.0),
                                dict(residual_tol=float("nan")), dict(residual_tol=-1.0),
                                dict(rtol=float("nan")), dict(atol=0.0)])
def test_equilibrium_rejects_bad_horizon_and_tolerances(kw):
    """Confirmed against the pre-fix code: horizon=nan/inf hung, and
    residual_tol=nan silently DISABLED the settledness check (resid > nan is
    always False), returning an unchecked point as the equilibrium."""
    g, sigma = _paper_graph_and_pattern()
    with pytest.raises(ValueError, match=next(iter(kw))):
        equilibrium(g, sigma, CellModel(**_PAPER_MODEL), **kw)


def test_equilibrium_rejects_non_binary_or_wrong_length_pattern():
    """Confirmed against the pre-fix code: a non-binary pattern was
    integrated as-is (from 0.99*pattern) and a wrong-length one failed with
    an unrelated numpy broadcasting error."""
    g, _ = _paper_graph_and_pattern()
    model = CellModel(**_PAPER_MODEL)
    with pytest.raises(ValueError, match="pattern"):
        equilibrium(g, [0.5, 1, 1, 1, -1, -1, -1], model)
    with pytest.raises(ValueError, match="pattern"):
        equilibrium(g, [-1, 1, 1, 1, -1, -1], model)


def test_write_simulation_rejects_negative_target_node_and_non_finite_floor():
    """nontarget_margin skips the target via j == target_node, which a
    negative index never matches -- so a WriteSimulation must not be
    constructible with one, whoever builds it. A non-finite floor would turn
    the margin into +-inf/nan, which reads as a verdict, not an error."""
    t = np.linspace(0, 1, 5)
    x = np.ones((3, 5))
    with pytest.raises(ValueError, match="target_node"):
        WriteSimulation(t=t, x=x, y=x.copy(), target_node=-1, target_sign=1.0, hit_time=None)
    sim = WriteSimulation(t=t, x=x, y=x.copy(), target_node=0, target_sign=1.0, hit_time=None)
    for bad_floor in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="floor"):
            sim.nontarget_margin(bad_floor)
