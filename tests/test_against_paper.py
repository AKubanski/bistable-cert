"""Cross-checks against reproduce_section6.py's own printed output and
against the two probe scripts run earlier in this project. These are not
generic unit tests -- they pin exact numbers already independently produced
by scripts outside this package, so a silent formula transcription error
cannot hide.
"""
import numpy as np
import pytest

from bistable_cert import (
    Graph, CellModel, certify_storage, certify_single_node_write,
    margin_ceiling, search_amplitude, opposite_side_degree_threshold,
)


# ---------------------------------------------------------------------------
# reproduce_section6.py's own printed output (Opuscula manuscript, Section 6):
#   C_beta  = 0.256185
#   eps*q(sigma)  = 0.150000 < C_beta -> True
#   eps*q(sigma') = 0.140000 < C_beta -> True
#   m_1(U,a) = 0.214842 at r = -0.768625
#   certified duration T >= (1+a)/m = 7.680072
# ---------------------------------------------------------------------------
PAPER_EDGES = [(1,2,0.70),(1,3,0.60),(1,5,0.50),(2,3,0.40),(2,4,0.30),(2,5,0.60),
               (2,6,0.20),(3,4,0.50),(3,5,0.30),(3,6,0.40),(4,7,0.30),(5,6,0.50),
               (5,7,0.40),(6,7,0.60)]
PAPER_SIGMA = [-1, 1, 1, 1, -1, -1, -1]


def _paper_graph():
    edges0 = [(i - 1, j - 1, w) for (i, j, w) in PAPER_EDGES]
    return Graph.from_edges(7, edges0)


def test_paper_constants_match_printed_output():
    model = CellModel(beta=2.5, eps=0.10)
    assert model.C_beta == pytest.approx(0.256185, abs=1e-6)


def test_paper_storage_certificates_match():
    g = _paper_graph()
    model = CellModel(beta=2.5, eps=0.10)
    sigma = np.array(PAPER_SIGMA, dtype=float)
    sigma2 = sigma.copy(); sigma2[0] = -sigma2[0]

    cert0 = certify_storage(g, sigma, model)
    cert1 = certify_storage(g, sigma2, model)

    assert cert0.eps * cert0.q_sigma == pytest.approx(0.150000, abs=1e-6)
    assert cert1.eps * cert1.q_sigma == pytest.approx(0.140000, abs=1e-6)
    assert cert0.feasible and cert1.feasible


def test_paper_write_certificate_matches_printed_output():
    g = _paper_graph()
    model = CellModel(beta=2.5, eps=0.10)
    sigma = np.array(PAPER_SIGMA, dtype=float)

    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=0.40,
                                      model=model, floor=0.65)

    assert cert.feasible
    assert cert.margin == pytest.approx(0.214842, abs=1e-5)
    assert cert.T_cert == pytest.approx(7.680072, abs=1e-4)
    # from reproduce_section6.py: d_par = 1.30, d_perp = 0.50, d_i = 1.80
    assert cert.d_par == pytest.approx(1.30, abs=1e-9)
    assert cert.d_perp == pytest.approx(0.50, abs=1e-9)
    assert cert.degree_i == pytest.approx(1.80, abs=1e-9)


# ---------------------------------------------------------------------------
# topology_results.json (earlier probe run, this project): star graph,
# n=10, hub write (node 0) -- margin=0.00956, ratio=64.64 (extreme
# conservativeness case, used to confirm the degree-dependence finding).
# complete graph, n=10 -- NOT certified (storage certificate already fails).
#
# CORRECTNESS NOTE (v0.1.1): the n=10, weight=0.5 example below was v0.1.0's
# pinned "extreme conservativeness" case. Under the corrected eq:amarg
# storage check (see certificates.py's module docstring) its TARGET pattern
# is not actually storage-certified at floor=0.65 at all -- v0.1.0's
# CERTIFIED verdict on it was itself an instance of the false-positive bug,
# not just numerically imprecise. It is kept below as a regression check
# that the fix rejects it, and replaced with a new, independently-verified
# example for the "high hub degree -> small margin" illustration.
# ---------------------------------------------------------------------------
def _alternating_pattern(n):
    return np.array([1.0 if k % 2 == 0 else -1.0 for k in range(n)])


def test_old_star_hub_example_now_correctly_rejected():
    """v0.1.0 reported this write CERTIFIED. It should not have: the target
    pattern fails the real, a-specific storage condition (eq:amarg) at
    floor=0.65 -- confirmed independently by direct equilibrium integration
    (see the module docstring's 6-node counterexample for the same
    mechanism). This pins the fix, not a new numeric claim."""
    g = Graph.star(10, weight=0.5)
    model = CellModel()
    sigma = _alternating_pattern(10)

    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=0.40,
                                      model=model, floor=0.65)

    assert not cert.feasible
    assert "not storage-certified" in cert.reason


def test_star_hub_high_degree_small_margin():
    """Replacement for the old (invalid) topology-probe pin: a star hub
    example that DOES pass the corrected storage certificate at floor=0.65
    for both endpoints, still showing the same qualitative effect (high
    write-node degree -> small pulse margin -> large T_cert/T_true).
    margin cross-checked against an independent 2,000,001-point brute-force
    grid search over r (not the package's own critical-point solver):
    agreement to 8 decimals (0.09205975 both ways)."""
    g = Graph.star(11, weight=0.3)
    model = CellModel()
    sigma = _alternating_pattern(11)

    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=0.40,
                                      model=model, floor=0.65)

    assert cert.feasible
    assert cert.margin == pytest.approx(0.09205975, abs=1e-6)
    assert cert.degree_i == pytest.approx(3.0, abs=1e-9)
    assert cert.d_par == pytest.approx(1.5, abs=1e-9)
    assert cert.d_perp == pytest.approx(1.5, abs=1e-9)
    assert cert.T_cert == pytest.approx(17.92314258, abs=1e-5)


def test_complete_graph_n10_not_storage_certified():
    g = Graph.complete(10, weight=0.5)
    model = CellModel()
    sigma = _alternating_pattern(10)

    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=0.40,
                                      model=model, floor=0.65)

    assert not cert.feasible
    assert "not storage-certified" in cert.reason


# ---------------------------------------------------------------------------
# amplitude_ceiling_probe.py (this project, follow-up to the topology
# probe): star graph, weight=0.5, a=0.65 -- U -> infinity margin ceiling
# crosses zero between n=10 (ceiling=+0.02000) and n=11 (ceiling=-0.06250).
# Closed form matched certify_single_node_write at U=1e4 to 5 decimals.
# ---------------------------------------------------------------------------
def test_margin_ceiling_matches_probe_star_n10():
    g = Graph.star(10, weight=0.5)
    sigma = _alternating_pattern(10)
    ceiling = margin_ceiling(g, sigma, target_node=0, floor=0.65)
    assert ceiling == pytest.approx(0.02000, abs=1e-4)


def test_margin_ceiling_matches_probe_star_n11_crosses_zero():
    g = Graph.star(11, weight=0.5)
    sigma = _alternating_pattern(11)
    ceiling = margin_ceiling(g, sigma, target_node=0, floor=0.65)
    assert ceiling == pytest.approx(-0.06250, abs=1e-4)
    assert ceiling < 0


def test_margin_ceiling_agrees_with_huge_amplitude_certificate():
    """The closed form must match what a direct (expensive) large-U
    certificate call finds -- this is the check that caught the formula
    right the first time, kept here so it stays right.

    Uses weight=0.3 (not v0.1.0's weight=0.5): at weight=0.5 this n=9 star's
    patterns fail the corrected storage certificate at floor=0.65, so
    certify_single_node_write would report NOT CERTIFIED regardless of the
    ceiling -- this test is specifically about the ceiling formula, so it
    needs an example where storage certification isn't the thing being
    exercised."""
    g = Graph.star(9, weight=0.3)
    sigma = _alternating_pattern(9)
    ceiling = margin_ceiling(g, sigma, target_node=0, floor=0.65)
    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=1e4, floor=0.65)
    assert cert.feasible
    assert ceiling == pytest.approx(0.152, abs=1e-6)
    assert cert.margin == pytest.approx(ceiling, abs=1e-3)


def test_search_amplitude_reports_storage_failure_not_ceiling_on_old_example():
    """v0.1.0 used this star(12, weight=0.5) example to show search_amplitude
    reporting 'no finite amplitude can certify this write' from a negative
    margin_ceiling. Under the corrected storage check that reason is no
    longer reachable here: the SOURCE pattern already fails eq:amarg at
    floor=0.65 (q(sigma) is too large regardless of amplitude), so
    search_amplitude must report that -- the real, earlier-blocking reason
    -- not blame the amplitude ceiling."""
    g = Graph.star(12, weight=0.5)
    sigma = _alternating_pattern(12)
    cert = search_amplitude(g, sigma, target_node=0, lo=0.0, hi=5.0)
    assert not cert.feasible
    assert "not storage-certified" in cert.reason
    assert "no finite amplitude" not in cert.reason


def test_storage_certified_implies_positive_margin_ceiling():
    """Structural fact found while fixing the eq:amarg bug: whenever a
    write's TARGET pattern passes storage certification at some floor a,
    margin_ceiling at that same a is automatically positive. So
    'both endpoints storage-certified, yet no finite amplitude can ever
    certify the write' -- the scenario the old (removed) test demonstrated
    -- cannot actually occur; search_amplitude's AssertionError guard for
    exactly this combination is defensive, not a live code path.

    Proof: certify_storage(target pattern) feasible means
        eps*(1+a)*q(sigma') < f_beta(a) = tanh(beta*a) - a.
    The write-target node i's own post-write opposed degree is exactly
    d_perp (flipping node i turns its former same-side neighbours into
    opposite-side ones under sigma'), and q(sigma') = max_j c_j(sigma') >=
    c_i(sigma') = d_perp. So eps*(1+a)*d_perp < tanh(beta*a) - a, i.e.
        1 - a - eps*(1+a)*d_perp > 1 - tanh(beta*a) > 0
    (strict since tanh is strictly < 1 for any finite argument) -- and the
    left side is exactly margin_ceiling's closed form. This holds for every
    beta > 0 and every a in (0,1), not just the paper's own parameters.

    Checked here two ways: the general algebraic argument above (true
    unconditionally, so a single confirming instance plus the identity
    check is enough), and a 3000-random-graph sweep as a regression guard.
    """
    model = CellModel()
    a = 0.65
    rng = np.random.default_rng(123)
    n_checked = 0
    for _ in range(3000):
        n = rng.integers(4, 15)
        W = np.zeros((n, n))
        perm = rng.permutation(n)
        for k in range(1, n):
            i, j = perm[k], perm[rng.integers(0, k)]
            W[i, j] = W[j, i] = rng.uniform(0.1, 1.0)
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] == 0 and rng.random() < 0.3:
                    W[i, j] = W[j, i] = rng.uniform(0.1, 1.0)
        g = Graph(W)
        sigma = rng.choice([-1.0, 1.0], size=n)
        i0 = int(rng.integers(0, n))
        cert = certify_single_node_write(g, sigma, target_node=i0, amplitude=0.01,
                                          model=model, floor=a)
        if cert.initial_certificate is None or not cert.initial_certificate.feasible:
            continue
        if cert.final_certificate is None or not cert.final_certificate.feasible:
            continue
        n_checked += 1
        ceiling = margin_ceiling(g, sigma, target_node=i0, model=model, floor=a)
        assert ceiling > 0, f"counterexample to the proof above: ceiling={ceiling}"
    assert n_checked >= 300  # sanity: enough storage-certified pairs were sampled


# ---------------------------------------------------------------------------
# eps/floor sensitivity of d_perp* = (1-a)/(eps*(1+a)): the threshold
# scales as exactly 1/eps at fixed a (algebraic, checked here as a
# regression), and drops roughly 4x across a in [0.30, 0.90] at the
# paper's eps=0.10 -- floor choice matters more than eps in that range.
# The r=floor minimum the whole derivation rests on was also confirmed to
# hold unconditionally (coefficient -(1+eps*d_i) < 0 for every d_i >= 0),
# so this is valid across the full floor range, not just a=0.65.
# ---------------------------------------------------------------------------
def test_threshold_scales_as_inverse_eps_at_fixed_floor():
    a = 0.65
    base = opposite_side_degree_threshold(CellModel(eps=0.10), floor=a)
    for eps in [0.02, 0.05, 0.20, 0.50, 1.0]:
        thr = opposite_side_degree_threshold(CellModel(eps=eps), floor=a)
        assert thr == pytest.approx(base * (0.10 / eps), rel=1e-9)


def test_threshold_decreases_as_floor_increases():
    """A stricter (higher) certified floor a tightens the opposite-side
    degree budget -- monotone, not just true at the paper's a=0.65.

    v0.1.0 swept floors down to 0.10; several of those (and 0.30, see
    below) are below this model's a_star=0.4127 and are not valid floors
    at all -- Theorem A's own contraction argument requires floor in the
    open interval (a_star, s_plus) (see certificates.py's _validate_floor).
    The swept range is restricted to that domain."""
    model = CellModel()
    assert model.a_star < 0.45 and 0.95 < model.s_plus  # keeps the range below valid
    floors = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    thresholds = [opposite_side_degree_threshold(model, floor=a) for a in floors]
    assert thresholds == sorted(thresholds, reverse=True)


def test_threshold_rejects_floor_outside_valid_domain():
    """floor=0.30 (v0.1.0's own probe used this) is below a_star=0.4127 for
    beta=2.5 -- not a smaller-but-still-valid margin, but outside the
    domain where Theorem A's proof applies at all (confirmed separately:
    gamma_a = 1 - beta*sech^2(beta*a) is already negative there)."""
    model = CellModel()
    assert model.a_star == pytest.approx(0.412687, abs=1e-6)
    with pytest.raises(ValueError):
        opposite_side_degree_threshold(model, floor=0.30)


def test_threshold_at_paper_floor_matches_probe():
    model = CellModel()
    assert opposite_side_degree_threshold(model, floor=0.65) == pytest.approx(2.12121, abs=1e-5)
    assert opposite_side_degree_threshold(model, floor=0.45) == pytest.approx(3.79310, abs=1e-5)
    assert opposite_side_degree_threshold(model, floor=0.90) == pytest.approx(0.52632, abs=1e-5)


# ---------------------------------------------------------------------------
# universal_threshold_probe.py (this project, follow-up to the amplitude-
# ceiling probe): the degree_i=4.50 star-hub sweep across d_par/d_perp
# splits (k_same=0..5), and a random-graph sign check. Confirms
# margin_ceiling depends on d_perp ALONE, not on d_par or total degree,
# contrary to the earlier ("driven by degree") conclusion.
#
# CORRECTNESS NOTE (v0.1.1): the sign-check sample size below was raised
# from 300 to 800 random graphs. The corrected, stricter eq:amarg storage
# check (see certificates.py) passes fewer of them than v0.1.0's check did
# (267/800 here vs. v0.1.0's 206/300) -- expected, since eq:amarg is
# strictly tighter than the eq:Acert check it replaced (h_beta(floor) <
# C_beta for any floor > a_star). The larger sample restores a comparable
# number of storage-certified cases to check the sign identity on.
# ---------------------------------------------------------------------------
def test_margin_ceiling_depends_on_d_perp_alone_not_d_par():
    """The decisive test: fix degree_i, vary ONLY the d_par/d_perp split.
    If the earlier 'degree drives it' story were the real mechanism,
    ceiling would stay constant here. It does not -- it tracks d_perp
    exactly, confirming d_par cancels out algebraically.

    margin_ceiling is a pure function of the graph/pattern's d_par/d_perp
    split, floor, and eps -- it is well-defined and independent of storage
    certification (see its "SCOPE NOTE" docstring), so this test does not
    gate on certify_storage. None of these six patterns are actually
    storage-certified at floor=0.65 (weight=0.5 on a 10-node star saturates
    q(sigma) too far) -- that is a separate, later-checked condition,
    demonstrated explicitly below for k_same=5."""
    g = Graph.star(10, weight=0.5)
    model = CellModel()
    a = 0.65

    # k_same=0: all 9 leaves opposite the write target -> d_perp=4.50 (max)
    # k_same=5: 5 leaves same side -> d_perp=2.00 (min feasible here)
    expected = {
        0: -0.39250, 1: -0.31000, 2: -0.22750, 3: -0.14500, 4: -0.06250, 5: +0.02000,
    }
    for k_same, expected_ceiling in expected.items():
        sigma = np.array([-1.0] + [1.0] * k_same + [-1.0] * (9 - k_same))
        ceiling = margin_ceiling(g, sigma, target_node=0, model=model, floor=a)
        assert ceiling == pytest.approx(expected_ceiling, abs=1e-4), (
            f"k_same={k_same}: got {ceiling}, expected {expected_ceiling} "
            f"(all cases share degree_i=4.50 -- only the d_par/d_perp split differs)")

        # and the closed d_perp-only formula must reproduce it exactly
        _, d_perp = g.split_degree(0, sigma, target_sign=1.0)
        pure_d_perp_formula = 1 - a - model.eps * (1 + a) * d_perp
        assert ceiling == pytest.approx(pure_d_perp_formula, abs=1e-9)


def test_positive_ceiling_alone_does_not_imply_certified():
    """k_same=5 above has the only positive ceiling in that sweep
    (+0.02000) -- reading margin_ceiling alone would suggest this write
    might be certifiable at a large enough amplitude. It is not: the
    SOURCE pattern (q(sigma), independent of amplitude) fails storage
    certification at floor=0.65. This is exactly the necessary-but-not-
    sufficient relationship margin_ceiling's docstring warns about, made
    concrete on one example rather than left abstract."""
    g = Graph.star(10, weight=0.5)
    model = CellModel()
    sigma = np.array([-1.0] + [1.0] * 5 + [-1.0] * 4)

    ceiling = margin_ceiling(g, sigma, target_node=0, model=model, floor=0.65)
    assert ceiling > 0

    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=1e4,
                                      model=model, floor=0.65)
    assert not cert.feasible
    assert "source pattern is not storage-certified" in cert.reason


def test_opposite_side_degree_threshold_matches_probe():
    model = CellModel()
    threshold = opposite_side_degree_threshold(model, floor=0.65)
    assert threshold == pytest.approx(2.121212, abs=1e-5)


def test_opposite_side_degree_threshold_predicts_ceiling_sign_on_random_graphs():
    """Same 800-graph sweep as universal_threshold_probe.py (raised from
    300 in v0.1.0 -- the corrected, stricter storage check passes fewer of
    them, see the module comment above): the sign of margin_ceiling must
    match (d_perp < threshold) with no exceptions, since this is an
    algebraic identity, not a statistical tendency."""
    model = CellModel()
    a = 0.65
    threshold = opposite_side_degree_threshold(model, floor=a)
    rng = np.random.default_rng(42)
    n_checked = n_match = 0
    for _ in range(800):
        n = rng.integers(4, 15)
        W = np.zeros((n, n))
        perm = rng.permutation(n)
        for k in range(1, n):
            i, j = perm[k], perm[rng.integers(0, k)]
            w = rng.uniform(0.1, 1.0)
            W[i, j] = W[j, i] = w
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] == 0 and rng.random() < 0.3:
                    w = rng.uniform(0.1, 1.0)
                    W[i, j] = W[j, i] = w
        g = Graph(W)
        sigma = rng.choice([-1.0, 1.0], size=n)
        if not certify_storage(g, sigma, model).feasible:
            continue
        i0 = int(rng.integers(0, n))
        ceiling = margin_ceiling(g, sigma, target_node=i0, model=model, floor=a)
        target_sign = -sigma[i0]
        _, d_perp = g.split_degree(i0, sigma, target_sign)
        n_checked += 1
        if (ceiling > 0) == (d_perp < threshold):
            n_match += 1
    assert n_checked >= 200  # sanity: enough storage-certified cases were sampled
    assert n_match == n_checked, f"threshold predicted sign correctly in {n_match}/{n_checked}"


def test_write_certificate_infeasible_does_not_claim_impossibility():
    """The one property that matters most: an infeasible certificate's
    repr and reason must never assert the write cannot happen -- only
    that this theorem does not certify it."""
    g = Graph.complete(10, weight=0.5)
    sigma = _alternating_pattern(10)
    cert = certify_single_node_write(g, sigma, target_node=0, amplitude=0.40, floor=0.65)
    assert not cert.feasible
    forbidden_words = ["impossible", "cannot", "can not", "never happens"]
    text = (cert.reason + repr(cert)).lower()
    for word in forbidden_words:
        assert word not in text
