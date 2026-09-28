"""Numerical integration -- for inspecting or cross-checking a certificate,
never for producing one. See the module docstring of
:mod:`bistable_cert.certificates` for why that separation matters.

Input validation: these functions make no feasibility judgment, but they
must still integrate the system the caller actually named. They apply the
same checks as the certificate layer -- ``Graph._check_node_index`` for the
target node, ``Graph._validate_pattern`` for the pattern, the model's open
domain ``(a_star, s_plus)`` for ``floor``, ``amplitude`` finite and
``>= 0`` -- plus finite, non-negative durations and positive solver
tolerances. CORRECTNESS NOTE (1.0.0 pre-release review): an external
review of the 1.0.0 release candidate found this module had no input
validation at all, and every case below was reproduced before fixing:

- ``simulate_write(target_node=-1)`` ran without error but applied the
  pulse to NO node: ``target_sign`` and the reported trace used Python's
  negative indexing (-1 = the last node), while the pulse itself was built
  by comparing ``np.arange(n) == target_node``, which -1 never matches. The
  trajectory was bit-identical to a no-pulse run, so a caller saw a
  "failed" write that was never attempted. ``nontarget_margin`` had the
  same mismatch (``j == -1`` never skips the target).
- a negative ``duration`` was accepted: no pulse at all, and once
  ``duration < -extra`` the horizon went negative and ``solve_ivp``
  silently integrated BACKWARD in time.
- ``duration`` or ``amplitude`` = nan, ``duration`` = inf, and
  ``equilibrium(horizon=nan or inf)`` did not raise -- they hung.
- a negative ``amplitude`` (pushing the target the wrong way) and a
  ``floor`` outside the model's domain were accepted, although the
  certificate layer rejects both.
- ``equilibrium(residual_tol=nan)`` silently disabled its own settledness
  check (``resid > nan`` is always False), and a non-binary or
  wrong-length pattern was either integrated as-is or failed with an
  unrelated numpy broadcasting error.

None of these produced a false CERTIFIED -- this module never certifies
anything -- but each returned a trace of a different system than the one
requested, or no answer at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .certificates import _validate_floor
from .dynamics import CellModel
from .graph import Graph


def _check_real(name: str, value, lower: float, *, strict: bool) -> float:
    """``value`` must be finite and ``>= lower`` (``> lower`` if strict).
    Same rule and wording as the certificate layer's ``amplitude`` check."""
    if not np.isfinite(value) or (value <= lower if strict else value < lower):
        op = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {op} {lower:g}, got {value!r}")
    return float(value)


def equilibrium(graph: Graph, pattern, model: CellModel, horizon: float = 300.0,
                 rtol: float = 1e-11, atol: float = 1e-13, residual_tol: float = 1e-6):
    """Integrate from ``0.99 * pattern`` (both coordinates) under zero
    control out to ``horizon`` and return the settled state's x-coordinate.
    Used to seed a write simulation from the actual stored equilibrium
    rather than from the pattern itself.

    Raises ``RuntimeError`` rather than returning a value silently, in two
    cases the caller cannot otherwise detect: the ODE solver reporting
    failure (``sol.success`` False -- e.g. it could not keep the requested
    tolerance and gave up before ``horizon``), or the state at ``horizon``
    not actually being settled (the velocity ``||rhs(Z)||`` there exceeds
    ``residual_tol``, meaning either the pattern isn't stable at all, or
    ``horizon`` was too short for it to converge). A function whose whole
    job is "find the resting point" returning a moving point with no signal
    is exactly the kind of silent wrongness this package exists to avoid --
    a later certificate or simulation built on that point would be checking
    the wrong thing without any indication why.

    Raises ``ValueError`` for a pattern that is not in ``{-1, +1}^n`` and
    for a ``horizon``, ``rtol``, ``atol`` or ``residual_tol`` that is not
    finite and ``> 0`` (see the module docstring for what each did before).
    """
    pattern = graph._validate_pattern(pattern)
    horizon = _check_real("horizon", horizon, 0.0, strict=True)
    rtol = _check_real("rtol", rtol, 0.0, strict=True)
    atol = _check_real("atol", atol, 0.0, strict=True)
    residual_tol = _check_real("residual_tol", residual_tol, 0.0, strict=True)

    rhs = model.rhs(graph)
    zero_u = lambda t: np.zeros(graph.n)
    Z0 = np.concatenate([pattern * 0.99, pattern * 0.99])
    sol = solve_ivp(rhs, [0, horizon], Z0, args=(zero_u,), rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(
            f"equilibrium(): ODE solver did not complete to horizon={horizon} "
            f"(status={sol.status}: {sol.message}); the returned state would not "
            f"be trustworthy as an equilibrium")
    Z_end = sol.y[:, -1]
    velocity = rhs(sol.t[-1], Z_end, zero_u)
    resid = float(np.max(np.abs(velocity)))
    if resid > residual_tol:
        raise RuntimeError(
            f"equilibrium(): state at horizon={horizon} has not settled "
            f"(||rhs(Z)||_inf={resid:.3e} > residual_tol={residual_tol:.1e}); "
            f"either this pattern is not actually a stable equilibrium of "
            f"the uncontrolled dynamics, or horizon is too short -- try a "
            f"larger horizon before trusting this as a resting point")
    return Z_end[:graph.n]


@dataclass
class WriteSimulation:
    """Numerical trace of a certified (or attempted) single-node write."""
    t: np.ndarray
    x: np.ndarray  # shape (n, len(t))
    y: np.ndarray  # shape (n, len(t))
    target_node: int
    target_sign: float
    hit_time: float | None  # first t where min(tau*x_i, tau*y_i) >= floor, or None

    def __post_init__(self):
        # target_node is compared with == against 0..n-1 below, so it must
        # BE one of those -- a negative index would silently never match.
        x, y, t = np.asarray(self.x), np.asarray(self.y), np.asarray(self.t)
        if x.ndim != 2 or x.shape != y.shape or t.shape != (x.shape[1],):
            raise ValueError(f"inconsistent trace shapes: t {t.shape}, x {x.shape}, y {y.shape}")
        n = x.shape[0]
        tn = self.target_node
        if isinstance(tn, bool) or not isinstance(tn, (int, np.integer)) or not (0 <= tn < n):
            raise ValueError(f"target_node must be an int with 0 <= target_node < {n}, got {tn!r}")

    def nontarget_margin(self, floor: float) -> float:
        """min over t and non-target nodes j of sigma_j * s_j(t) - floor,
        evaluated at the ``n_points`` times sampled into ``self.t`` (see
        ``simulate_write``'s ``n_points``) -- a dense-SAMPLE check, not a
        continuous one. Positive means no violation was found in that
        sample; it does not prove every non-target node stayed inside its
        branch at every instant of the continuous trajectory, which this
        method cannot see between two sampled points. CORRECTNESS NOTE
        (v0.1.3): an earlier version of this docstring said "stayed ...
        for the whole simulated window", which overstated exactly this --
        a fourth external review flagged the gap between what is claimed
        and what a fixed-size sample can actually establish.

        ``floor`` must be finite (a non-finite floor would turn the result
        into +-inf or nan, which reads as a verdict rather than an error).
        This method holds no model, so it cannot check ``floor`` against
        ``(a_star, s_plus)`` -- pass the certificate's own ``floor``."""
        if not np.isfinite(floor):
            raise ValueError(f"floor must be finite, got {floor!r}")
        n = self.x.shape[0]
        sigma0 = np.sign(self.x[:, 0])
        worst = np.inf
        for j in range(n):
            if j == self.target_node:
                continue
            worst = min(worst, float((sigma0[j] * self.x[j]).min()),
                        float((sigma0[j] * self.y[j]).min()))
        return worst - floor


def simulate_write(graph: Graph, pattern, target_node: int, amplitude: float,
                    duration: float, model: CellModel = None, floor: float = 0.65,
                    extra: float = 15.0, n_points: int = 4000) -> WriteSimulation:
    """Integrate the true ODE from the pattern's own equilibrium under a
    constant pulse of ``amplitude`` held for ``duration`` at ``target_node``,
    then released. ``duration`` is typically a certificate's ``T_cert``, but
    passing any other value (e.g. to search for the true minimal switching
    time, as the probe scripts do) is equally valid -- this function makes
    no feasibility judgment itself.

    Inputs are validated exactly as ``certify_single_node_write`` validates
    its own: ``target_node`` an int with ``0 <= target_node < n``,
    ``pattern`` in ``{-1, +1}^n``, ``amplitude`` finite and ``>= 0``,
    ``floor`` inside ``(a_star, s_plus)``. In addition ``duration`` and
    ``extra`` must be finite and ``>= 0`` with ``duration + extra > 0``, and
    ``n_points`` an int ``>= 2``. ``amplitude=0`` and ``duration=0`` stay
    legal as no-pulse control runs. Raises ``ValueError`` otherwise -- see
    the module docstring for what each missing check used to do.
    """
    model = model or CellModel()
    target_node = graph._check_node_index(target_node)
    pattern = graph._validate_pattern(pattern)
    amplitude = _check_real("amplitude", amplitude, 0.0, strict=False)
    duration = _check_real("duration", duration, 0.0, strict=False)
    extra = _check_real("extra", extra, 0.0, strict=False)
    if duration + extra <= 0:
        raise ValueError("duration + extra must be > 0 (the simulated window would be empty)")
    _validate_floor(model, floor)
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError(f"n_points must be an int >= 2, got {n_points!r}")

    target_sign = -pattern[target_node]
    n = graph.n

    p_src = equilibrium(graph, pattern, model)
    Z0 = np.concatenate([p_src, p_src])
    rhs = model.rhs(graph)
    pulse_u = lambda t: np.where(np.arange(n) == target_node, target_sign * amplitude * (t <= duration), 0.0)

    horizon = duration + extra
    sol = solve_ivp(rhs, [0, horizon], Z0, args=(pulse_u,),
                     rtol=1e-10, atol=1e-12, dense_output=True, max_step=1e-2)
    if not sol.success:
        raise RuntimeError(
            f"simulate_write(): ODE solver did not complete to horizon={horizon} "
            f"(status={sol.status}: {sol.message}); the returned trace would be "
            f"truncated or unreliable")
    t = np.linspace(0, horizon, n_points)
    Z = sol.sol(t)
    x, y = Z[:n], Z[n:]

    v = np.minimum(target_sign * x[target_node], target_sign * y[target_node])
    hit_mask = v >= floor
    hit_time = float(t[np.argmax(hit_mask)]) if hit_mask.any() else None

    return WriteSimulation(t=t, x=x, y=y, target_node=target_node,
                            target_sign=float(target_sign), hit_time=hit_time)
