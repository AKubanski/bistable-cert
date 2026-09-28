"""The two-component tanh-cell model and its scalar constants.

Every formula here is copied verbatim from ``reproduce_section6.py`` (the
script behind Section 6 of the Opuscula manuscript) and cross-checked
against its printed output before this package existed (see
``tests/test_against_paper.py``). This module does not derive anything; it
packages the model the paper already proved things about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .graph import Graph


@dataclass(frozen=True)
class CellModel:
    """A single scalar family of parameters shared by every cell in the
    network: the bistability sharpness ``beta`` and the diffusive coupling
    strength ``eps``. Both are dimensionless in the paper's normalization.
    """
    beta: float = 2.5
    eps: float = 0.10

    #: Minimum required distance from beta=1. CORRECTNESS NOTE (v0.1.3): a
    #: fourth external review found beta=nextafter(1.0, 2.0) (accepted by
    #: the old `beta > 1` check alone) computes a_star=0.0 EXACTLY --
    #: mathematically it should be a tiny positive number (~1.49e-8 there,
    #: matching s_SN's own value to leading order) -- because
    #: sqrt(1+1ULP) itself already rounds to exactly 1.0 in float64, so
    #: arccosh(1.0)=0. Checked against an 80-digit mpmath reference across
    #: a range of beta: a_star's own relative error is still tiny
    #: (~1e-9) even at beta=1+1e-8, but C_beta = (s_SN-a_star)/(1+a_star)
    #: is far more sensitive -- s_SN and a_star both -> 0 at the same
    #: sqrt(beta-1) rate as beta -> 1, so their difference u_SN is a
    #: subtraction of two nearly-equal numbers (catastrophic
    #: cancellation): C_beta's relative error is already ~1.6e-8 at
    #: beta=1+1e-4, ~1.3e-4 at beta=1+1e-6, and over 25% by beta=1+1e-8.
    #: At beta=1+1e-3 (this threshold) it measures ~1.1e-10 -- comfortably
    #: below NUMERICAL_SAFETY_TOLERANCE, with real margin to spare.
    #: Properly restoring accuracy all the way to beta=1 would need an
    #: asymptotic (Taylor-in-(beta-1)) rewrite of a_star/s_SN/C_beta,
    #: which risks introducing new errors without the paper's own
    #: derivation to check it against; since the paper's own working
    #: value is beta=2.5, far from this edge, documenting and enforcing a
    #: safe minimum distance is the fix here instead.
    MIN_BETA_MINUS_ONE = 1e-3

    def __post_init__(self):
        if not np.isfinite(self.beta) or self.beta <= 1:
            raise ValueError(
                f"beta must be finite and > 1, got {self.beta}. beta<=1 gives no "
                f"fold at all (a_star=arccosh(sqrt(beta))/beta and s_SN=sqrt(1-1/beta) "
                f"both require beta>1 for the uncoupled cell to be bistable; at "
                f"beta=1 the two branches degenerate to a single point).")
        if self.beta - 1 < self.MIN_BETA_MINUS_ONE:
            raise ValueError(
                f"beta={self.beta} is too close to 1 (beta-1={self.beta-1:.3e} < "
                f"{self.MIN_BETA_MINUS_ONE:.0e}): a_star/s_SN/C_beta lose accuracy "
                f"catastrophically in this regime (a subtraction of two nearly-equal, "
                f"nearly-zero quantities) -- see MIN_BETA_MINUS_ONE's docstring for "
                f"measured error sizes. Use a beta farther from 1, or note that this "
                f"model was never validated near its own bifurcation point (the "
                f"paper's own working value is beta=2.5).")
        if not np.isfinite(self.eps) or self.eps < 0:
            raise ValueError(f"eps must be finite and >= 0 (a diffusive coupling "
                              f"strength), got {self.eps}")

    # ---------------------------------------------------------- scalar constants
    @property
    def a_star(self) -> float:
        """The fold point of the uncoupled cell's odd nonlinearity."""
        b = self.beta
        return float(np.arccosh(np.sqrt(b)) / b)

    @property
    def s_SN(self) -> float:
        b = self.beta
        return float(np.sqrt(1 - 1 / b))

    @property
    def u_SN(self) -> float:
        return self.s_SN - self.a_star

    @property
    def C_beta(self) -> float:
        """Storage-certificate threshold: a pattern sigma is certified
        whenever eps * q(sigma) < C_beta (Theorem A)."""
        return self.u_SN / (1 + self.a_star)

    @property
    def s_plus(self) -> float:
        """The positive stable equilibrium of the uncoupled cell.

        CORRECTNESS NOTE (v0.1.2): the old upper bracket, ``1 - 1e-15``, was
        an external review found ``CellModel(beta=20)`` (a value
        ``__post_init__`` accepts -- only ``beta > 1`` is required) raises
        ``ValueError: f(a) and f(b) must have different signs`` here. Cause:
        as ``beta`` grows, ``tanh(beta*s_plus) -> 1`` faster than ``s_plus``
        itself approaches 1, so for ``beta`` above ~19-20,
        ``tanh(beta*(1-1e-15))`` already rounds to exactly ``1.0`` in
        float64 -- making ``f(1-1e-15)`` land on the SAME (positive) side as
        ``f(a_star+1e-12)``, so the old bracket no longer contains a sign
        change. Using exactly ``1.0`` instead fixes this directly: at that
        same regime ``f(1.0) = tanh(beta) - 1.0`` rounds to exactly ``0.0``
        (confirmed for beta=20/40/100), and ``scipy.optimize.brentq``
        accepts an exact zero at an endpoint as the root itself, with no
        sign-change requirement -- so it returns ``1.0`` directly, which is
        also the mathematically correct answer to float64's own resolution
        (the true ``s_plus`` and ``1.0`` are closer than adjacent floats can
        distinguish at that beta). For beta in the paper's actual working
        range (beta=2.5), this changes nothing: ``f(1.0)`` is comfortably
        negative there (-1.3e-2), so the bracket is exactly as before.
        """
        return float(brentq(lambda s: self.f(s), self.a_star + 1e-12, 1.0))

    def f(self, s: float) -> float:
        """f_beta(s) = tanh(beta s) - s, the uncoupled cell's vector field."""
        return float(np.tanh(self.beta * s) - s)

    def k_a(self, a: float) -> float:
        return self.beta / np.cosh(self.beta * a) ** 2

    def gamma_a(self, a: float) -> float:
        """gamma_a = 1 - k_a(a): the comparison-system decay rate used in
        the grounded resolvent A_i(a) = gamma_a I + eps L_(i) (Section 5)."""
        return 1 - self.k_a(a)

    # ---------------------------------------------------------------- dynamics
    def rhs(self, graph: Graph):
        """Return the ODE right-hand side ``f(t, Z, u)`` for this model on
        ``graph``, where ``Z = [x_1..x_n, y_1..y_n]`` stacks both cell
        coordinates and ``u(t)`` is the control (pulse) as a length-n array.
        """
        L = graph.laplacian
        beta, eps = self.beta, self.eps
        n = graph.n

        def rhs(t, Z, u):
            x, y = Z[:n], Z[n:]
            ut = u(t)
            return np.concatenate([
                -x + np.tanh(beta * (y + ut)) - eps * (L @ x),
                -y + np.tanh(beta * (x + ut)) - eps * (L @ y),
            ])
        return rhs
