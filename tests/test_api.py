"""Basic API behavior, independent of the paper's specific numbers."""
import numpy as np
import pytest

from bistable_cert import Graph, CellModel, certify_storage, certify_single_node_write, search_amplitude
from bistable_cert.simulation import simulate_write, equilibrium


def test_graph_rejects_asymmetric_matrix():
    with pytest.raises(ValueError):
        Graph(np.array([[0, 1], [0, 0]]))


def test_graph_rejects_self_loop():
    with pytest.raises(ValueError):
        Graph(np.array([[1, 0], [0, 0]]))


def test_graph_factories_have_expected_edge_counts():
    assert Graph.chain(5).W.astype(bool).sum() // 2 == 4
    assert Graph.ring(5).W.astype(bool).sum() // 2 == 5
    assert Graph.star(5).W.astype(bool).sum() // 2 == 4
    assert Graph.complete(5).W.astype(bool).sum() // 2 == 10


def test_zero_amplitude_never_certifies_a_write():
    """With U=0 there is no pulse at all; the margin function's dominant
    term at U=0 is bounded by the cell's own uncoupled dynamics and should
    not certify a switch for any reasonable floor."""
    g = Graph.chain(4)
    sigma = [1, -1, 1, -1]
    cert = certify_single_node_write(g, sigma, target_node=1, amplitude=0.0)
    assert not cert.feasible


def test_larger_amplitude_never_hurts_feasibility():
    """Margin m_i(U,a) is monotone increasing in U (tanh(beta(r+U)) is
    monotone increasing in U pointwise in r), so once a write is certified
    at some U it stays certified at any larger U on the same graph/pattern."""
    g = Graph.star(6)
    sigma = [1, -1, 1, -1, 1, -1]
    cert_low = certify_single_node_write(g, sigma, target_node=0, amplitude=0.5)
    cert_high = certify_single_node_write(g, sigma, target_node=0, amplitude=1.5)
    if cert_low.feasible:
        assert cert_high.feasible
        assert cert_high.margin >= cert_low.margin


def test_search_amplitude_finds_a_feasible_point_when_one_exists():
    g = Graph.chain(6)
    sigma = [1, -1, 1, -1, 1, -1]
    cert = search_amplitude(g, sigma, target_node=2, lo=0.0, hi=3.0)
    assert cert.feasible
    # one step smaller should not also be feasible (within the search
    # tolerance), confirming this is close to the actual threshold and not
    # just "some large amplitude that trivially works"
    cert_below = certify_single_node_write(g, sigma, target_node=2, amplitude=cert.amplitude - 0.01)
    assert not cert_below.feasible


def test_simulate_write_runs_and_reports_a_hit_time_for_a_certified_write():
    g = Graph.chain(6)
    model = CellModel()
    sigma = np.array([1, -1, 1, -1, 1, -1], dtype=float)
    cert = certify_single_node_write(g, sigma, target_node=2, amplitude=0.6, model=model)
    assert cert.feasible

    sim = simulate_write(g, sigma, target_node=2, amplitude=0.6,
                          duration=cert.T_cert, model=model, floor=cert.floor)
    assert sim.hit_time is not None
    assert sim.hit_time <= cert.T_cert  # the certificate's duration must be sufficient
    assert sim.nontarget_margin(cert.floor) >= 0  # non-targets never crossed the floor
