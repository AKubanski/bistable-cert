"""Weighted undirected graphs.

A ``Graph`` is nothing more than a symmetric nonnegative weight matrix and
the machinery Theorems A/B need from it (the weighted degree, the opposed
weighted degree relative to a pattern, and the graph Laplacian). It carries
no assumption about topology -- chain, ring, star, complete, or an
arbitrary irregular graph are all just weight matrices.
"""
from __future__ import annotations

import numpy as np


class Graph:
    """A finite weighted undirected graph on nodes ``0, ..., n-1``.

    Parameters
    ----------
    W : (n, n) array_like
        Symmetric nonnegative weight matrix, zero diagonal. Not copied
        defensively beyond ``np.asarray`` -- treat the constructed graph as
        owning its own copy and do not mutate ``W`` afterwards.
    """

    def __init__(self, W):
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be a square matrix, got shape {W.shape}")
        if not np.all(np.isfinite(W)):
            raise ValueError("W must contain only finite weights (no inf or NaN)")
        # CORRECTNESS NOTE (v0.1.2): a third external review found these
        # checks let through inputs Theorem A/B does not cover -- an
        # external review reproduced a 1e-6 asymmetry, a -5e-13 "negative"
        # weight, and a 5e-13 nonzero diagonal, all accepted, all leading to
        # a downstream CERTIFIED that is not actually backed by the
        # theorems' hypotheses (which need an EXACTLY symmetric, EXACTLY
        # nonnegative, EXACTLY zero-diagonal matrix). Root cause of the
        # symmetry gap specifically: np.allclose's default rtol=1e-5 was
        # still in effect despite the explicit atol=1e-12 -- for weights
        # O(1), that made the *effective* tolerance ~1e-5, not 1e-12. Fixed
        # by pinning rtol=0 so atol is the only tolerance in force, and by
        # tightening all three checks to what a certificate tool should
        # actually require: the caller is responsible for an exactly-valid
        # graph (trivial to satisfy by construction -- from_edges below
        # always produces one), not the library for guessing whether a
        # "slightly invalid" graph is close enough. Negativity and the
        # zero-diagonal check are now exact (no tolerance at all); symmetry
        # keeps a tiny atol=1e-9 (not zero) since two independently
        # computed floats that are mathematically equal can differ by a few
        # ULPs, and rejecting that would make the check needlessly brittle
        # for real inputs. 1e-9 is still ~4 orders of magnitude tighter
        # than the old rtol-induced ~1e-5, and rejects the reviewer's 1e-6
        # example outright.
        if not np.allclose(W, W.T, atol=1e-9, rtol=0):
            raise ValueError("W must be symmetric (undirected graph): "
                              f"max asymmetry {np.max(np.abs(W - W.T)):.3e} exceeds atol=1e-9")
        # CORRECTNESS NOTE (v0.1.3): the check above only confirmed W is
        # CLOSE to symmetric and then stored it as-is, asymmetry and all. A
        # fourth external review found a targeted case that slips under any
        # fixed tolerance almost by definition -- W=[[0,0],[9e-10,0]], just
        # inside the old atol=1e-9 -- and showed the consequence is not
        # cosmetic: opposed_degree/degree/laplacian all index self.W by ROW,
        # so node 0 (row [0,0]) and node 1 (row [9e-10,0]) disagreed about
        # whether an edge between them exists at all, __repr__ undercounted
        # edges (nonzero entries: 1, not 2), and certify_storage reported
        # CERTIFIED off node 0's zeroed-out view. Tightening the tolerance
        # further would only move the same problem to a smaller adversarial
        # value, not remove it -- ANY fixed tolerance can be undercut this
        # way. Fixed at the root: once W is confirmed close enough to
        # symmetric to accept, the matrix actually STORED and used from here
        # on is its exact symmetrization, not the caller's possibly-slightly-
        # asymmetric input -- so every downstream computation sees the same,
        # genuinely symmetric graph regardless of row/column access order,
        # and __repr__'s edge count is never inconsistent with it.
        with np.errstate(over="ignore"):
            # overflow here is expected input for the finiteness check right
            # below, which turns it into one clear ValueError -- suppressed
            # so a caller doesn't see numpy's own raw RuntimeWarning first
            # for a condition this constructor already handles explicitly.
            W = (W + W.T) / 2
        # CORRECTNESS NOTE (v0.1.5): the finiteness check above runs on the
        # caller's RAW input, before symmetrization -- but ``W+W.T`` itself
        # can overflow to inf even when every individual entry the caller
        # passed was finite (two entries near float64's max, ~1e308, sum to
        # ~2e308, which exceeds it).
        #
        # This CORRECTS v0.1.4's own erratum, which checked this package's
        # weight-magnitude behavior against exactly one pattern (alternating
        # signs) and reported the overflow fails closed (correctly NOT
        # CERTIFIED via q_sigma=inf). A follow-up to the fifth external
        # review pointed out a HOMOGENEOUS pattern is a different story, and
        # reproduced it: opposed_degree uses boolean masking
        # (``self.W[k][pattern != pattern[k]]``), which EXCLUDES same-sign
        # entries rather than multiplying them by zero, so a homogeneous
        # pattern (every neighbor same-signed) never selects the inf
        # entries at all and returns an exact 0.0 -- confirmed here too:
        # W=[[0,1e308],[1e308,0]] constructed with no error, silently
        # storing self.W=[[0,inf],[inf,0]] despite this constructor's own
        # contract ("no inf or NaN"), and certify_storage reported CERTIFIED
        # off a graph that contains a literal infinite coupling weight. Not
        # a documentable limitation like the true out-of-scope cases
        # elsewhere in this file -- a false CERTIFIED is exactly the
        # failure mode this whole package exists to rule out, regardless of
        # how unusual the triggering input is. Fixed by re-checking
        # finiteness on the matrix actually being stored, after
        # symmetrization, not just on the input before it -- the same
        # checked-thing-must-match-stored-thing fix already applied to the
        # symmetry gap above (v0.1.3).
        if not np.all(np.isfinite(W)):
            raise ValueError("W must contain only finite weights (no inf or NaN) -- "
                              "this input was finite but W+W.T overflowed during "
                              "symmetrization; scale the weights down")
        if np.any(W < 0):
            raise ValueError("W must be nonnegative (checked exactly, no tolerance -- "
                              "a certificate's graph must actually satisfy Theorem A/B's "
                              "hypothesis, not merely be close to satisfying it)")
        if np.any(np.diag(W) != 0):
            raise ValueError("W must have zero diagonal (no self-loops), checked exactly")
        self.W = W
        self.n = W.shape[0]

    def _check_node_index(self, i) -> int:
        """target_node/i must be an in-range node index. Plain Python
        negative-indexing semantics (W[-1] silently means the LAST node)
        would otherwise let an out-of-range write target through with no
        error at all -- confirmed reachable via certify_single_node_write
        (target_node=-1) before this check existed."""
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise ValueError(f"node index must be an int, got {type(i).__name__}: {i!r}")
        i = int(i)
        if not (0 <= i < self.n):
            raise ValueError(f"node index {i} out of range for a graph with n={self.n} "
                              f"nodes (valid range: 0 <= i < {self.n})")
        return i

    def _validate_pattern(self, pattern) -> np.ndarray:
        """Every certificate in this package (Theorem A/B) is stated for
        pattern in {-1, +1}^n; c_i(sigma), d_par, d_perp are all sums
        gated on sign *agreement*, which silently means something else
        entirely (and certifies nothing the theorems actually cover) for
        any other numeric pattern -- e.g. a 0/1 pattern would make every
        node with value 0 agree with every other 0, which is not a sign
        comparison. Reject anything that is not exactly +-1 per entry."""
        pattern = np.asarray(pattern, dtype=float)
        if pattern.shape != (self.n,):
            raise ValueError(f"pattern must have shape ({self.n},), got {pattern.shape}")
        if not np.all(np.isfinite(pattern)):
            raise ValueError("pattern must contain only finite values")
        if not np.all(np.abs(np.abs(pattern) - 1.0) < 1e-12):
            raise ValueError(
                f"pattern must have every entry in {{-1, +1}} (got values "
                f"{sorted(set(pattern.tolist()))}); Theorem A/B's c_i(sigma) "
                f"and the d_par/d_perp split are defined by SIGN agreement, "
                f"not by numeric equality, so a non-binary pattern silently "
                f"computes something the theorems don't cover.")
        # CORRECTNESS NOTE (v0.1.2): an external review found that two
        # entries both within 1e-12 of -1 (e.g. -1+3e-13 and -1-4e-13) pass
        # the tolerance check above but are NOT float-equal to each other --
        # and opposed_degree/split_degree below compare entries with exact
        # == / != . On a graph where both such nodes are adjacent to a third,
        # this silently miscomputes which side of the pattern each node is
        # on (confirmed: a ring(4) with such a pair gave opposed_degree
        # [0.5,1.,1.,0.5] instead of the correct [0.5,0.5,0.5,0.5] -- the
        # exact-== comparisons split two same-side neighbours apart). Fixed
        # by snapping every validated entry to the exact +-1 it was checked
        # against, so every downstream == / != comparison is exact by
        # construction, not just "close" comparisons that can disagree with
        # each other.
        return np.where(pattern > 0, 1.0, -1.0)

    # ---------------------------------------------------------- construction
    @classmethod
    def from_edges(cls, n, edges):
        """Build from an edge list ``[(i, j, weight), ...]`` (0-indexed).

        Matches the format used in the paper's own worked example
        (``reproduce_section6.py``), where edges are listed 1-indexed in
        the manuscript; convert to 0-indexed before calling this.

        CORRECTNESS NOTE (v0.1.3): a fourth external review found
        ``(-1, 0, 0.5)`` was silently accepted and, via plain Python
        negative-indexing on ``W[i, j] = ...`` below, created an edge to
        the LAST node instead of raising -- confirmed reproducible. Every
        index is now checked against ``0 <= i,j < n`` before use.

        CORRECTNESS NOTE (v0.1.4): that fix made this iterate ``edges``
        twice (once to validate, once to build ``W``) -- confirmed
        reproducible that a single-use iterator (a generator expression, a
        ``map``/``zip`` object, anything not a list/tuple) is exhausted by
        the first pass, so the second pass silently builds an all-zero,
        zero-edge ``W`` instead of raising or building the intended graph.
        Unlike the other adversarial-parameter cases this package
        documents as out of scope, this needs no extreme value -- a plain
        generator expression is ordinary Python, and nothing about the
        documented ``[(i, j, weight), ...]`` contract warns a caller off
        it. Fixed by materializing ``edges`` into a list up front, so both
        passes see the same data regardless of what iterable was passed.
        """
        edges = list(edges)
        for i, j, w in edges:
            if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))) \
                    or isinstance(i, bool) or isinstance(j, bool):
                raise ValueError(f"edge endpoints must be ints, got ({i!r}, {j!r})")
            if not (0 <= i < n) or not (0 <= j < n):
                raise ValueError(f"edge ({i}, {j}, {w}) has an index out of range "
                                  f"for n={n} nodes (valid range: 0 <= i,j < {n})")
            if i == j:
                raise ValueError(f"self-loop at node {i} is not allowed")
        W = np.zeros((n, n))
        for i, j, w in edges:
            W[i, j] = W[j, i] = w
        return cls(W)

    @classmethod
    def chain(cls, n, weight=0.5):
        """Path graph: 0-1-2-...-(n-1)."""
        return cls.from_edges(n, [(i, i + 1, weight) for i in range(n - 1)])

    @classmethod
    def ring(cls, n, weight=0.5):
        """Cycle graph: 0-1-2-...-(n-1)-0."""
        edges = [(i, (i + 1) % n, weight) for i in range(n)]
        return cls.from_edges(n, edges)

    @classmethod
    def star(cls, n, weight=0.5):
        """One hub (node 0) connected to n-1 leaves."""
        return cls.from_edges(n, [(0, i, weight) for i in range(1, n)])

    @classmethod
    def complete(cls, n, weight=0.5):
        """K_n with uniform edge weight."""
        edges = [(i, j, weight) for i in range(n) for j in range(i + 1, n)]
        return cls.from_edges(n, edges)

    # ---------------------------------------------------------------- derived
    @property
    def degree(self):
        """Weighted degree d_i = sum_j W_ij, shape (n,)."""
        return self.W.sum(axis=1)

    @property
    def laplacian(self):
        """Graph Laplacian L = diag(degree) - W."""
        return np.diag(self.degree) - self.W

    def grounded_laplacian(self, i):
        """L_(i): the Laplacian with row/column i removed (Section 5 of the
        paper, the grounded comparison resolvent A_i(a) = gamma_a I + eps L_(i))."""
        idx = [j for j in range(self.n) if j != i]
        return self.laplacian[np.ix_(idx, idx)]

    def opposed_degree(self, pattern):
        """c(sigma): for each node k, the weight of edges to nodes with the
        opposite sign in ``pattern``. Used by both the storage certificate
        (Theorem A, q(sigma) = max_k c_k(sigma)) and the pulse margin
        (Theorem B, via d_par/d_perp at the written node)."""
        pattern = self._validate_pattern(pattern)
        return np.array([
            self.W[k][pattern != pattern[k]].sum() for k in range(self.n)
        ])

    def split_degree(self, i, pattern, target_sign):
        """d_par, d_perp at node i for a write that moves node i's sign to
        ``target_sign``: the weight of i's edges to nodes currently on the
        SAME side as the target sign (d_par) versus the OPPOSITE side
        (d_perp). These are exactly the quantities the pulse margin
        m_i(U, a) needs (Section 4)."""
        i = self._check_node_index(i)
        pattern = self._validate_pattern(pattern)
        if abs(abs(target_sign) - 1.0) > 1e-12:
            raise ValueError(f"target_sign must be +-1, got {target_sign}")
        target_sign = 1.0 if target_sign > 0 else -1.0
        same = self.W[i][pattern == target_sign].sum()
        opp = self.W[i][pattern == -target_sign].sum()
        return float(same), float(opp)

    def __repr__(self):
        return f"Graph(n={self.n}, edges={int(np.count_nonzero(self.W)) // 2})"
