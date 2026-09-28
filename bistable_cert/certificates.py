"""Certificates from Theorem A (storage) and Theorem B (single-node write;
Theorem 4.3 of the paper).

These functions decide feasibility from the theorems' explicit, closed-form
sufficient conditions -- they never integrate the network to decide
CERTIFIED/NOT CERTIFIED. (Numerical integration is available separately,
in :mod:`bistable_cert.simulation`, for inspecting or cross-checking a
certified write -- not for producing the certificate itself.)

Central caveat, worth repeating in every place a result is displayed:
Theorem A and Theorem B give SUFFICIENT conditions. ``feasible=False`` means
the theorem does not certify the write -- not that the write is impossible.

CORRECTNESS NOTE (v0.1.1): v0.1.0's ``certify_storage`` checked only
``eps*q(sigma) < C_beta`` (the paper's eq:Acert) -- an ``a``-INDEPENDENT
condition that proves *some* admissible floor exists near ``a_star``, but
does NOT prove the SPECIFIC floor a caller actually uses (the package
defaulted every function to ``floor=0.65``, the paper's own value) is
admissible for an arbitrary graph/pattern. The paper's own Theorem B
states its hypothesis as "let a satisfy eq:common"
(``eps*(1+a)*max{q(sigma),q(sigma')} < f_beta(a)``) -- an explicit,
``a``-SPECIFIC condition that v0.1.0 never checked. An external review
caught this and described a concrete example in prose (no runnable code
or exact parameters given); independently reconstructed here as a fresh
6-node counterexample -- see ``tests/test_against_paper.py`` in this
package, and ``eq_common_gap_probe.py``, a standalone reproduction script
kept alongside this project's probe scripts (not part of this package's
own files -- ask for it if you want it). Same underlying bug: a 6-node
star where the old check reported CERTIFIED for a write, yet the TRUE
equilibrium of the source pattern already had a non-target node resting
at 0.602, strictly below the claimed floor 0.65 -- with zero pulse
applied. That is not a transient edge case; it is the storage certificate
itself being wrong at that floor. The paper's own manuscript-reproduction
script, ``reference/reproduce_section6.py`` (included in this package),
checked the correct condition all along (its own line computing
``eps(1+a)Q < f_beta(a)``) -- the gap was introduced only when this
package's API was built on top of it, not in the paper or its own
verification. Root cause: since ``h_beta(a) := f_beta(a)/(1+a)`` is
strictly decreasing on ``(a_star, s_plus)`` with
``h_beta(a) -> C_beta`` only as ``a -> a_star`` from above,
``eps*q(sigma) < C_beta`` guarantees existence of an admissible floor
arbitrarily close to ``a_star`` -- never that a caller-chosen ``floor``
far from ``a_star`` (like 0.65) is one of them. ``certify_storage`` now
takes ``floor`` as an explicit, required-by-default argument and checks
the real, a-specific condition. See ``admissible_floor_bound`` for the
constructive fix: computing the actual floor range a pattern supports,
rather than assuming 0.65 always works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .dynamics import CellModel
from .graph import Graph

#: Default margin a "CERTIFIED" verdict must clear above raw float zero.
#: CORRECTNESS NOTE (v0.1.2): an external review found a specific
#: amplitude at which the exact critical-point solver's own float64
#: evaluation of tanh/cosh disagreed in SIGN with an arbitrary-precision
#: (80-digit) recomputation of the identical formula -- both at the
#: ~1e-17 scale (independently reconstructed here on a fresh example, not
#: just trusted: a from-scratch random graph, its margin(U) root found by
#: brentq, then sampled at float64 ULPs around that root and compared
#: against an mpmath recomputation -- 3 sign disagreements found out of
#: ~6000 points tested, all with |margin| below 1.2e-16). This is not a
#: bug in the critical-point algorithm itself (the earlier grid-vs-exact
#: precision work stands); it is inherent to evaluating any formula in
#: float64 near a true zero-crossing, where the last representable bit of
#: precision cannot reliably resolve the sign. So "CERTIFIED" now requires
#: clearing this safety tolerance, not just `> 0`. 1e-9 is enormous
#: relative to float64 noise (~1e-16) and utterly negligible relative to
#: every margin seen in this project's own probes (median ~0.2, smallest
#: genuine one 0.076) -- it costs nothing on real cases and closes the
#: knife-edge one.
NUMERICAL_SAFETY_TOLERANCE = 1e-9


def _validate_floor(model: CellModel, floor: float) -> None:
    """floor must lie in the model's open domain (a_star, s_plus): outside
    it, f_beta(floor) can be non-positive and the contraction rate
    gamma_a = 1 - beta*sech^2(beta*floor) can be negative (expansion, not
    contraction) -- the entire invariant-box argument (Theorem A's own
    proof) breaks down, not just this package's bookkeeping. Confirmed
    numerically: at floor=0.30 (below a_star=0.4127 for beta=2.5),
    gamma_a = -0.491."""
    if not (model.a_star < floor < model.s_plus):
        raise ValueError(
            f"floor={floor} is outside the model's valid domain "
            f"(a_star={model.a_star:.6f}, s_plus={model.s_plus:.6f}); "
            f"Theorem A's own contraction argument requires floor in that "
            f"open interval (gamma_a = 1-beta*sech^2(beta*floor) must be "
            f"positive). A floor at or below a_star is not a smaller-but-"
            f"still-valid margin -- it invalidates the box-invariance proof.")


@dataclass(frozen=True)
class StorageCertificate:
    """Result of Theorem A applied to one pattern AT A SPECIFIC floor.

    ``feasible=True`` means the pattern is storage-certified at THIS
    ``floor``: eq:amarg, ``eps*(1+floor)*q(sigma) < f_beta(floor)``, holds,
    which is what Theorem A's proof actually uses to show
    ``B_sigma(floor)`` is positively invariant and contains the unique
    stable equilibrium. ``feasible=False`` means this specific floor is
    not certified for this pattern -- a SMALLER floor might still work;
    see ``admissible_floor_bound``.

    ``exists_admissible_floor`` is the weaker, floor-independent condition
    ``eps*q(sigma) < C_beta`` (the paper's eq:Acert): it proves SOME floor
    near ``a_star`` is admissible, but is NOT sufficient by itself to
    conclude THIS ``floor`` is -- see the module docstring for why
    conflating the two was v0.1.0's bug. It is kept here only as a
    diagnostic (e.g. to distinguish "no floor will ever work" from "this
    particular floor is too large").
    """
    feasible: bool
    q_sigma: float
    eps: float
    floor: float
    lhs_amarg: float
    rhs_amarg: float
    exists_admissible_floor: bool
    C_beta: float

    @property
    def margin_to_threshold(self) -> float:
        """rhs_amarg - lhs_amarg: positive and large is comfortably inside
        the certificate at this floor; positive and near zero means close
        to the boundary of what this floor can certify."""
        return self.rhs_amarg - self.lhs_amarg

    def __repr__(self):
        status = "CERTIFIED" if self.feasible else "NOT CERTIFIED"
        return (f"StorageCertificate({status} at floor={self.floor}, "
                f"eps*(1+floor)*q(sigma)={self.lhs_amarg:.5f} vs "
                f"f_beta(floor)={self.rhs_amarg:.5f}, "
                f"margin={self.margin_to_threshold:+.5f})")


def certify_storage(graph: Graph, pattern, model: Optional[CellModel] = None,
                     floor: float = 0.65,
                     numerical_safety_tolerance: float = NUMERICAL_SAFETY_TOLERANCE
                     ) -> StorageCertificate:
    """Theorem A at a specific floor: is ``pattern`` storage-certified with
    branch margin ``floor`` on ``graph``?

    Checks eq:amarg, ``eps*(1+floor)*q(sigma) < f_beta(floor)`` -- the
    ``a``-specific condition Theorem A's proof actually establishes
    ``B_sigma(floor)``'s invariance from, NOT the weaker floor-independent
    eq:Acert (``eps*q(sigma) < C_beta``), which only proves existence of
    *some* admissible floor near ``a_star`` and was v0.1.0's bug -- see the
    module docstring.

    ``feasible`` requires ``rhs - lhs > numerical_safety_tolerance``, not
    just ``lhs < rhs`` -- see ``NUMERICAL_SAFETY_TOLERANCE``'s docstring for
    why a bare ``>`` is not safe this close to a float64 zero. Pass
    ``numerical_safety_tolerance=0`` to recover the raw mathematical
    condition (e.g. for testing against the paper's own printed numbers).
    """
    model = model or CellModel()
    _validate_floor(model, floor)
    if not np.isfinite(numerical_safety_tolerance) or numerical_safety_tolerance < 0:
        raise ValueError(
            f"numerical_safety_tolerance must be finite and >= 0, got "
            f"{numerical_safety_tolerance!r}. CORRECTNESS NOTE (v0.1.3): a fourth "
            f"external review found a NEGATIVE tolerance was accepted outright and "
            f"could push a raw-negative margin over the CERTIFIED line -- confirmed "
            f"reproducible (a storage margin of -0.0546538 reported feasible=True "
            f"with tolerance=-1.0). 0 is still allowed, as the deliberate opt-out "
            f"back to the raw mathematical condition (e.g. to compare against the "
            f"paper's own printed numbers); only a negative or non-finite value, "
            f"which no legitimate use needs, is rejected.")
    q_sigma = float(graph.opposed_degree(pattern).max())
    lhs = model.eps * (1 + floor) * q_sigma
    rhs = model.f(floor)
    return StorageCertificate(
        feasible=(rhs - lhs) > numerical_safety_tolerance,
        q_sigma=q_sigma, eps=model.eps, floor=floor,
        lhs_amarg=lhs, rhs_amarg=rhs,
        exists_admissible_floor=model.eps * q_sigma < model.C_beta,
        C_beta=model.C_beta,
    )


def admissible_floor_bound(graph: Graph, pattern, model: Optional[CellModel] = None,
                            numerical_safety_tolerance: float = NUMERICAL_SAFETY_TOLERANCE):
    """A conservative admissible floor bound: for every ``a`` in
    ``(a_star, a_max)``, ``certify_storage(..., floor=a)`` (same
    ``numerical_safety_tolerance``) IS feasible -- the constructive answer
    to "what floor should I actually use for this pattern?" instead of
    assuming the paper's own 0.65 (chosen for ITS 7-node graph) works for
    an arbitrary one.

    CORRECTNESS NOTE (v0.1.5): earlier revisions of this docstring called
    ``a_max`` "the supremum" outright. A follow-up to the fifth external
    review pointed out that overclaims it: because of the tolerance padding
    below, ``a_max`` can be smaller than the TRUE supremum of the raw
    (tol=0) condition -- safe (every ``a`` below it is genuinely feasible,
    never the reverse), but not exact. Reworded throughout to "conservative
    bound" and "never loosens the true supremum, only potentially tightens
    it"; nothing about the actual computation changed, only what the
    docstring claims about it.

    Returns ``None`` when NO floor clears this function's own
    tolerance-padded bar. That always includes the case where no raw
    admissible floor exists at all (eq:Acert itself fails, ``eps*q(sigma)
    >= C_beta`` -- Theorem A does not apply to this pattern at any floor),
    but can also trigger on a pattern with a genuine raw-admissible sliver
    narrower than ``numerical_safety_tolerance`` itself -- the same
    conservatism as above, at its most visible.

    Derivation: ``h_beta(a) := f_beta(a)/(1+a)`` is continuous and strictly
    decreasing on ``(a_star, s_plus)`` (established in Theorem A's own
    proof), with ``h_beta(a) -> C_beta`` as ``a -> a_star`` and
    ``h_beta(s_plus) = 0`` (since ``f_beta(s_plus) = 0`` by definition of
    ``s_plus``). So eq:amarg, ``eps*q(sigma) < h_beta(a)``, holds exactly
    for ``a`` below the unique root of ``h_beta(a) = eps*q(sigma)``.

    CORRECTNESS NOTE (v0.1.2): the old implementation padded the bracket
    by a fixed ``1e-9*(hi-lo)`` before calling ``brentq``, to dodge
    evaluating ``h`` exactly at its (removable) endpoint singularities. An
    external review found this crashes with ``ValueError: f(a) and f(b)
    must have different signs`` in two reachable cases -- both confirmed
    here: (1) a homogeneous pattern, ``q_sigma=0``, where the "root" is
    exactly AT ``s_plus`` (every floor is admissible, so the true supremum
    is the boundary itself, which the padding steps just past); (2) a
    pattern whose load ``eps*q_sigma`` sits within roughly ``1e-12`` of
    ``C_beta``, where the true root sits closer to ``a_star`` than the pad
    itself, so the padded bracket no longer contains it.

    Fixed by dropping the padding heuristic entirely and evaluating ``h``
    directly at both endpoints. This does NOT rely on ``h(a_star)``
    matching ``model.C_beta`` -- CORRECTION (v0.1.3): an earlier version
    of this note claimed that match holds "bit-for-bit", checked at the
    time only at the paper's own beta=2.5. A fourth external review
    pointed out this is not a general identity, and checking it across a
    wider beta range confirms that: the two are computed via unrelated
    formulas (``h(a_star)`` via ``tanh``/subtraction, ``C_beta`` via
    ``arccosh``/``sqrt``) that agree mathematically but differ by ~1 ULP
    at most tested beta (8 of 14 spot-checked from 1.001 to 10000), and
    happened to match exactly at beta=2.5 by coincidence, not by any
    general property. The actual reason this function's bracket is safe
    doesn't need that identity at all: ``h_lo``/``h_hi`` below are both
    computed by calling the SAME ``h`` closure used for the root search,
    never ``model.C_beta``, so there is no cross-formula agreement to rely
    on in the first place.

    CORRECTNESS NOTE (v0.1.3): a fourth external review additionally found
    this function's result could be inconsistent with ``certify_storage``:
    the bound above solves the RAW condition ``h(a) = eps*q(sigma)``
    exactly, but ``certify_storage`` (since v0.1.2) requires clearing a
    ``numerical_safety_tolerance`` beyond that raw condition by default --
    confirmed reproducible: a pattern placed just inside the raw boundary
    got a finite bound back from this function, yet ``certify_storage`` at
    exactly that floor, with its own default tolerance, reported NOT
    CERTIFIED. Fixed by accepting the same ``numerical_safety_tolerance``
    here and solving the shifted condition ``h(a) - eps*q(sigma) =
    numerical_safety_tolerance`` instead of ``= 0``, so the floor this
    returns is one ``certify_storage`` (at the same tolerance) actually
    accepts, not just the raw theoretical edge. The shift is applied
    directly to ``h`` rather than reproducing ``certify_storage``'s exact
    ``(1+floor)`` scaling of the tolerance (its condition is
    ``(1+floor)*(h(a)-eps*q(sigma)) > tol``, i.e.
    ``h(a)-eps*q(sigma) > tol/(1+floor)``): since ``1+floor > 1``
    throughout the valid domain, shifting by the larger, unscaled ``tol``
    is always at least as conservative, never less -- the returned bound
    is never looser than what ``certify_storage`` actually needs. Pass
    ``numerical_safety_tolerance=0`` to recover the exact raw bound.
    """
    model = model or CellModel()
    if not np.isfinite(numerical_safety_tolerance) or numerical_safety_tolerance < 0:
        raise ValueError(f"numerical_safety_tolerance must be finite and >= 0, got "
                          f"{numerical_safety_tolerance!r}")
    q_sigma = float(graph.opposed_degree(pattern).max())
    target = model.eps * q_sigma + numerical_safety_tolerance
    h = lambda a: model.f(a) / (1 + a)
    lo, hi = model.a_star, model.s_plus
    h_lo, h_hi = h(lo), h(hi)
    if target >= h_lo:
        return None
    if target <= h_hi:
        return hi
    return brentq(lambda a: h(a) - target, lo, hi, xtol=1e-12, rtol=1e-12)


@dataclass(frozen=True)
class WriteCertificate:
    """Result of Theorem B applied to a single-node write.

    Theorem B is Theorem 4.3 of the paper, whose statement covers the full
    two-coordinate source box -- off-diagonal initial states
    (``x_i != y_i``) included. Earlier drafts, and the paper's own
    ``reference/reproduce_section6.py``, call that full-box regime "B*";
    it is part of Theorem 4.3, not a separate, unimplemented result.

    ``feasible=True`` means: for every initial state in the certified
    source box, the constant pulse of amplitude ``U`` held for duration
    ``T_cert`` drives the target node across the switching floor ``a`` in
    finite time while EVERY non-target node provably retains its own
    branch (both internal coordinates) throughout the transient, and the
    network is captured in the adjacent certified box at pulse removal.

    ``feasible=False`` means Theorem B's sufficient conditions failed --
    either the endpoint storage certificates at this floor (see ``reason``
    and ``initial_certificate``/``final_certificate``), or the pulse
    margin. Either way, this is not a proof of impossibility: a smaller
    floor (see ``admissible_floor_bound``) or a larger amplitude (see
    ``search_amplitude``) may certify it.

    ``degree_i``, ``d_par``, ``d_perp`` are included because a probe over
    chain/ring/star/complete/random graphs found the pulse margin's
    conservativeness grows with ``d_perp`` specifically (see
    ``margin_ceiling``) -- worth inspecting before trusting ``T_cert`` as
    a *useful*, not just valid, bound.
    """
    feasible: bool
    reason: str
    initial_certificate: StorageCertificate
    final_certificate: Optional[StorageCertificate] = None
    margin: Optional[float] = None
    T_cert: Optional[float] = None
    degree_i: Optional[float] = None
    d_par: Optional[float] = None
    d_perp: Optional[float] = None
    target_node: Optional[int] = None
    amplitude: Optional[float] = None
    floor: Optional[float] = None

    def __repr__(self):
        if not self.feasible:
            return f"WriteCertificate(NOT CERTIFIED: {self.reason})"
        return (f"WriteCertificate(CERTIFIED, node={self.target_node}, "
                f"T_cert={self.T_cert:.4f}, margin={self.margin:.5f}, "
                f"degree_i={self.degree_i:.3f})")


def _pulse_margin(graph: Graph, target_node: int, target_sign: float, source_pattern,
                   amplitude: float, floor: float, model: CellModel):
    """m_i(U, a) = min_{r in [-1, floor]} g_i(r, U, a), computed EXACTLY
    via critical-point root-finding, not by grid search.

        g_i(r) = tanh(beta(r+U)) - (1+eps*d_i) r + eps(floor*d_par - d_perp)
        g_i'(r) = beta*sech^2(beta(r+U)) - (1+eps*d_i)

    ``sech^2(beta(r+U))`` is unimodal (single peak at r=-U), so g_i' is a
    single "bump" minus a constant: monotone increasing on
    ``(-inf, -U]`` and monotone decreasing on ``[-U, inf)``. Each half is
    therefore individually monotone, so g_i' has at most one root per half
    -- found directly by ``brentq`` when the half's endpoints bracket a
    sign change, with no grid resolution to miss a dip between samples.
    The global minimum of g_i over ``[-1, floor]`` is then the smallest
    value among the two endpoints and any interior critical point found
    this way -- exact up to brentq's tolerance (1e-13), not grid spacing.

    v0.1.0 used a two-stage grid search (401 points, refined to 2001) here
    instead. A 20,000-trial stress test (this project) found that grid
    disagreed with this exact method by at most ~1.1e-11 at the paper's
    own beta=2.5 across a realistic parameter sweep -- far too small to
    flip any feasibility verdict seen in this project's probes. The grid
    version is nonetheless replaced outright: a "CERTIFIED" claim should
    rest on an exact computation when one is this cheap to get, not on a
    numerically adequate approximation.

    CORRECTION (v0.1.2): an earlier version of this docstring additionally
    claimed a second external review's report of a ~1e-9 sign-flipping
    case "could not be independently reproduced at beta=2.5, it required
    beta~40". A third review disputed this and reported reproducing a
    sign disagreement at beta=2.5 after all. That correction is right, but
    the two claims are about different things and conflating them would
    trade one imprecise statement for another: the ~1e-9 figure above was
    always specifically about GRID-vs-EXACT disagreement (a methodology
    gap that no longer exists now that grid search is gone), not about
    this exact solver disagreeing with the true, infinite-precision
    answer. That second, genuinely distinct phenomenon -- this exact
    critical-point solver's OWN float64 evaluation disagreeing in sign
    with an arbitrary-precision recomputation of the identical formula --
    IS reproducible at beta=2.5, confirmed here from scratch (a random
    graph, margin(U)'s root found by brentq, then float64 ULPs around that
    root compared against an 80-digit mpmath recomputation: 3 sign
    disagreements found out of ~6000 points tested, all with |margin|
    under 1.2e-16 -- i.e. exactly where float64 arithmetic runs out of
    resolution, not a defect specific to any one beta). This is inherent
    to floating point near any true zero-crossing and cannot be fixed by
    a better root-finder; see ``NUMERICAL_SAFETY_TOLERANCE``, which is the
    actual fix: a margin this close to zero is no longer reported as
    "CERTIFIED" regardless of which way its float64 sign happens to fall.
    """
    d_par, d_perp = graph.split_degree(target_node, source_pattern, target_sign)
    d_i = float(graph.degree[target_node])
    const = model.eps * (floor * d_par - d_perp)
    beta = model.beta
    C = 1 + model.eps * d_i
    U = amplitude

    def _sech2(x):
        """1/cosh(x)**2, stable for large |x| (e.g. amplitude searches that
        probe U ~ 1e4). np.cosh(x)**2 overflows to inf for |x| gtrsim 355.6
        and the true value correctly underflows to 0 instead; this form
        computes exp(-2|x|), which just underflows cleanly, with no
        intermediate inf and no RuntimeWarning."""
        ax = np.abs(x)
        e = np.exp(-2 * ax)
        return 4 * e / (1 + e) ** 2

    g = lambda r: np.tanh(beta * (r + U)) - C * r + const
    gp = lambda r: beta * _sech2(beta * (r + U)) - C

    peak = -U
    candidates = [-1.0, floor]

    lo1, hi1 = -1.0, min(peak, floor)
    if hi1 > lo1:
        g_lo, g_hi = gp(lo1), gp(hi1)
        if g_lo == 0:
            candidates.append(lo1)
        elif g_hi == 0:
            candidates.append(hi1)
        elif g_lo * g_hi < 0:
            candidates.append(brentq(gp, lo1, hi1, xtol=1e-13, rtol=1e-13))

    lo2, hi2 = max(peak, -1.0), floor
    if hi2 > lo2:
        g_lo, g_hi = gp(lo2), gp(hi2)
        if g_lo == 0:
            candidates.append(lo2)
        elif g_hi == 0:
            candidates.append(hi2)
        elif g_lo * g_hi < 0:
            candidates.append(brentq(gp, lo2, hi2, xtol=1e-13, rtol=1e-13))

    vals = [g(r) for r in candidates]
    kmin = int(np.argmin(vals))
    return float(vals[kmin]), d_par, d_perp, d_i


def certify_single_node_write(graph: Graph, pattern, target_node: int,
                               amplitude: float, model: Optional[CellModel] = None,
                               floor: float = 0.65,
                               numerical_safety_tolerance: float = NUMERICAL_SAFETY_TOLERANCE
                               ) -> WriteCertificate:
    """Theorem B (Theorem 4.3 of the paper): certify writing ``target_node``
    to its opposite sign, from every initial state in the certified source
    box.

    Parameters
    ----------
    graph : Graph
    pattern : array_like of shape (n,)
        The current (source) pattern, entries in {-1, +1}. The target
        pattern is ``pattern`` with ``target_node`` flipped.
    target_node : int
        Index of the node to write. Must satisfy ``0 <= target_node < graph.n``.
    amplitude : float
        Constant pulse amplitude ``U`` applied to ``target_node`` alone.
        Must be finite and non-negative.
    model : CellModel, optional
        Defaults to ``CellModel()`` (beta=2.5, eps=0.10 -- the paper's
        worked-example values).
    floor : float
        The certified branch floor ``a`` (a_star < a < s_plus). Defaults
        to 0.65, the value used throughout the paper's Section 6 -- but
        that value was verified there for ONE specific 7-node graph; on a
        different graph, check ``admissible_floor_bound`` before assuming
        0.65 is valid, or simply try ``certify_storage(..., floor=0.65)``
        first (this function does so for both endpoints and reports
        exactly which one fails, if any).
    numerical_safety_tolerance : float
        See ``NUMERICAL_SAFETY_TOLERANCE``. Applied to both the endpoint
        storage checks (delegated to ``certify_storage``) and the pulse
        margin itself.

    Checks, in order: source pattern storage-certified at ``floor``
    (eq:amarg), target pattern storage-certified at ``floor`` (eq:amarg
    again -- together these two checks are exactly equivalent to the
    paper's joint eq:common, ``eps*(1+floor)*max{q(sigma),q(sigma')} <
    f_beta(floor)``, since checking A<C and B<C separately is equivalent
    to checking max(A,B)<C), then the pulse margin at ``floor``.

    CORRECTNESS NOTE (v0.1.2): an external review found ``amplitude=nan``
    silently produced ``feasible=True`` (with ``margin=nan``, ``T_cert=nan``
    -- ``nan`` compares False against everything, so the old ``m_val <= 0``
    guard never rejected it), and ``amplitude=inf`` was accepted outright
    and produced an internally-inconsistent but finite-looking certificate.
    Both confirmed here directly. A negative ``target_node`` was also
    confirmed accepted (Python's own negative-indexing silently reinterprets
    it as a node counted from the end, rather than raising). Both are now
    validated explicitly, fail-closed, before any certificate logic runs.
    """
    model = model or CellModel()
    _validate_floor(model, floor)
    target_node = graph._check_node_index(target_node)
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError(f"amplitude must be finite and >= 0, got {amplitude!r}")
    pattern = np.asarray(pattern, dtype=float)

    cert0 = certify_storage(graph, pattern, model, floor=floor,
                             numerical_safety_tolerance=numerical_safety_tolerance)
    if not cert0.feasible:
        return WriteCertificate(
            feasible=False,
            reason=(f"source pattern is not storage-certified at floor={floor} "
                    f"(eq:amarg): eps*(1+floor)*q(sigma)={cert0.lhs_amarg:.5f} "
                    f"is not < f_beta(floor)={cert0.rhs_amarg:.5f}"
                    + ("" if cert0.exists_admissible_floor else
                       " -- and no floor exists for this pattern at all (eq:Acert fails too)")),
            initial_certificate=cert0, target_node=target_node, amplitude=amplitude, floor=floor,
        )

    pattern2 = pattern.copy()
    pattern2[target_node] = -pattern2[target_node]
    target_sign = pattern2[target_node]

    cert1 = certify_storage(graph, pattern2, model, floor=floor,
                             numerical_safety_tolerance=numerical_safety_tolerance)
    if not cert1.feasible:
        return WriteCertificate(
            feasible=False,
            reason=(f"target pattern is not storage-certified at floor={floor} "
                    f"(eq:amarg): eps*(1+floor)*q(sigma')={cert1.lhs_amarg:.5f} "
                    f"is not < f_beta(floor)={cert1.rhs_amarg:.5f}"),
            initial_certificate=cert0, final_certificate=cert1,
            target_node=target_node, amplitude=amplitude, floor=floor,
        )

    m_val, d_par, d_perp, d_i = _pulse_margin(
        graph, target_node, target_sign, pattern, amplitude, floor, model)

    if m_val <= numerical_safety_tolerance:
        return WriteCertificate(
            feasible=False, reason="pulse margin m_i(U,a) does not clear the numerical "
                                    "safety tolerance at this amplitude",
            initial_certificate=cert0, final_certificate=cert1,
            margin=m_val, degree_i=d_i, d_par=d_par, d_perp=d_perp,
            target_node=target_node, amplitude=amplitude, floor=floor,
        )

    T_cert = (1 + floor) / m_val
    return WriteCertificate(
        feasible=True, reason="",
        initial_certificate=cert0, final_certificate=cert1,
        margin=m_val, T_cert=T_cert, degree_i=d_i, d_par=d_par, d_perp=d_perp,
        target_node=target_node, amplitude=amplitude, floor=floor,
    )


def margin_ceiling(graph: Graph, pattern, target_node: int,
                    model: Optional[CellModel] = None, floor: float = 0.65) -> float:
    """The U -> infinity limit of the pulse margin m_i(U, a), in closed
    form -- no amplitude search needed.

    IMPORTANT SCOPE NOTE: this concerns ONLY the target node's own drift
    (Theorem B's Step 2/3) -- it says nothing about whether the two
    endpoint patterns are storage-certified at ``floor`` (eq:amarg /
    eq:common, Theorem B's Step 1, checked by :func:`certify_storage`). A
    positive ceiling is NECESSARY but not SUFFICIENT for
    :func:`certify_single_node_write` to return ``feasible=True`` --
    always check storage certification separately (or just call
    :func:`certify_single_node_write`, which checks both, in the right
    order).

    As U grows, tanh(beta(r+U)) -> 1 pointwise for every r in [-1, floor],
    so g_i(r,U,a) converges to the affine, decreasing function
    ``1 - (1+eps*d_i)*r + eps(floor*d_par - d_perp)``, whose minimum over
    ``r in [-1, floor]`` is at ``r = floor``:

        m_inf(a) = 1 - (1+eps*d_i)*a + eps*(a*d_par - d_perp)

    Substituting ``d_i = d_par + d_perp`` collapses this algebraically --
    the ``d_par`` terms cancel completely:

        m_inf(a) = 1 - a - eps*(1+a)*d_perp

    So the ceiling depends on ``d_perp`` ALONE -- see
    :func:`opposite_side_degree_threshold`.
    """
    model = model or CellModel()
    _validate_floor(model, floor)
    target_node = graph._check_node_index(target_node)
    pattern = np.asarray(pattern, dtype=float)
    target_sign = -pattern[target_node]
    d_par, d_perp = graph.split_degree(target_node, pattern, target_sign)
    d_i = float(graph.degree[target_node])
    return 1 - (1 + model.eps * d_i) * floor + model.eps * (floor * d_par - d_perp)


def opposite_side_degree_threshold(model: Optional[CellModel] = None, floor: float = 0.65) -> float:
    """d_perp* = (1-floor) / (eps*(1+floor)): the exact, topology-independent
    threshold on a written node's opposite-side weighted degree, below
    which the TARGET NODE's own margin (see :func:`margin_ceiling`'s scope
    note -- this does not cover endpoint storage certification) can be
    made positive by a large enough amplitude.

    eps=0 (no diffusive coupling at all) is a genuine edge case, not a
    division error to guard against: with zero coupling, no non-target
    node feels the pulse through the graph regardless of its opposite-side
    degree, so the threshold is unbounded -- returned here as +inf rather
    than raising ZeroDivisionError.
    """
    model = model or CellModel()
    _validate_floor(model, floor)
    if model.eps == 0:
        return float("inf")
    return (1 - floor) / (model.eps * (1 + floor))


def search_amplitude(graph: Graph, pattern, target_node: int,
                      model: Optional[CellModel] = None, floor: float = 0.65,
                      lo: float = 0.0, hi: float = 5.0, tol: float = 1e-4):
    """Find the smallest amplitude U in [lo, hi] for which
    :func:`certify_single_node_write` returns ``feasible=True``, by binary
    search on ``margin(U)`` (monotone increasing in U for this model, since
    increasing U only increases tanh(beta(r+U)) pointwise).

    Checks endpoint storage certification FIRST (via
    :func:`certify_single_node_write` at ``amplitude=hi``): if either
    endpoint fails eq:amarg at this ``floor``, no amplitude can fix that --
    amplitude only affects the target node's own margin, never whether the
    two patterns are storage-certified -- and the returned certificate's
    ``reason`` says so explicitly rather than misleadingly blaming the
    amplitude ceiling.

    Only once both endpoints are storage-certified does this check
    :func:`margin_ceiling`: if non-positive, no amplitude in ``[lo, hi]``
    -- or beyond it -- can ever certify the write, and the binary search
    is skipped, with ``reason`` saying so explicitly instead of just
    "infeasible at hi" (which would wrongly suggest trying a larger ``hi``).
    """
    model = model or CellModel()
    _validate_floor(model, floor)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0 or hi <= lo:
        raise ValueError(f"require finite 0 <= lo < hi, got lo={lo!r}, hi={hi!r}")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError(
            f"tol must be finite and > 0, got {tol!r}. CORRECTNESS NOTE (v0.1.3): a "
            f"fourth external review found tol<=0 made the `while b-a > tol` loop "
            f"below either hang (tol<0: b-a, always >= 0, is always greater than a "
            f"negative number, so the loop never terminates) or, for tol=nan/inf, "
            f"skip straight past the loop (any comparison against nan is False) and "
            f"silently return `hi` unbisected as if it were the minimal amplitude.")

    cert_hi = certify_single_node_write(graph, pattern, target_node, hi, model, floor)
    if not cert_hi.feasible and cert_hi.final_certificate is None and cert_hi.margin is None:
        # storage certification itself failed (source or target endpoint) --
        # amplitude is irrelevant, do not consult the margin ceiling at all
        return cert_hi
    if not cert_hi.feasible and cert_hi.margin is None:
        return cert_hi  # target endpoint storage certification failed

    ceiling = margin_ceiling(graph, pattern, target_node, model, floor)
    if ceiling <= 0:
        if cert_hi.feasible:
            raise AssertionError(
                f"margin_ceiling={ceiling:.6f} <= 0 but amplitude={hi} is feasible "
                f"(margin={cert_hi.margin}) -- ceiling formula or certificate disagree")
        return WriteCertificate(
            feasible=False,
            reason=(f"no finite amplitude can certify this write: the pulse-margin "
                    f"ceiling as U -> infinity is {ceiling:.6f} <= 0 (structural limit, "
                    f"see margin_ceiling()); both endpoints ARE storage-certified at "
                    f"floor={floor}, so this is purely a target-node margin limit"),
            initial_certificate=cert_hi.initial_certificate,
            final_certificate=cert_hi.final_certificate,
            degree_i=cert_hi.degree_i, d_par=cert_hi.d_par, d_perp=cert_hi.d_perp,
            target_node=target_node, amplitude=None, floor=floor,
        )

    if not cert_hi.feasible:
        return cert_hi  # ceiling positive, but not reached within [lo, hi]; a larger hi will help
    cert_lo = certify_single_node_write(graph, pattern, target_node, lo, model, floor)
    if cert_lo.feasible:
        return cert_lo  # already feasible at the bottom of the range

    a, b = lo, hi
    # max_iters is defense in depth, not just the tol>0 validation above: even
    # with a tiny-but-valid tol, halving b-a eventually hits float64's own
    # precision limit, where mid=0.5*(a+b) can round to exactly a or b with no
    # further progress -- b-a then stops shrinking and could stay above an
    # extremely small tol forever. 200 iterations is already far more than
    # enough to exhaust float64 precision on any realistic [lo, hi] (each
    # halves the interval; 2**-200 is astronomically past machine epsilon).
    max_iters = 200
    for _ in range(max_iters):
        if b - a <= tol:
            break
        mid = 0.5 * (a + b)
        cert_mid = certify_single_node_write(graph, pattern, target_node, mid, model, floor)
        if cert_mid.feasible:
            b = mid
        else:
            a = mid
    return certify_single_node_write(graph, pattern, target_node, b, model, floor)
