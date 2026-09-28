"""bistable_cert -- computable certificates from the paper "Certified
branch-safe selective writing in diffusively coupled bistable graph
memories" (A. Kubanski, 2026; preprint at
https://doi.org/10.5281/zenodo.22903156, prepared for submission to
Opuscula Mathematica).

Version 1.0.0 implements Theorem A (storage certification) and Theorem B
(single-node write certification; Theorem 4.3 of the paper, which covers
every initial state in the certified source box, off-diagonal ones
included -- the regime earlier drafts called B*). It does not implement
Section 5's crosstalk bounds (Theorems C/C*) or multi-node writes -- those
are deliberately outside this release, see the package README.

    from bistable_cert import Graph, CellModel, certify_storage, certify_single_node_write

    g = Graph.star(8, weight=0.5)
    model = CellModel(beta=2.5, eps=0.10)
    pattern = [1, -1, 1, -1, 1, -1, 1, -1]
    cert = certify_single_node_write(g, pattern, target_node=0, amplitude=0.4, model=model)
    print(cert)

Every quantity here is a SUFFICIENT condition from a proved theorem.
``feasible=False`` means the theorem does not certify the write -- never
that the write is impossible. See ``certificates.py`` for the full caveat
and ``probe/`` for measurements of how conservative the sufficient
condition actually is.
"""
from .graph import Graph
from .dynamics import CellModel
from .certificates import (
    StorageCertificate, WriteCertificate,
    certify_storage, certify_single_node_write, search_amplitude, margin_ceiling,
    opposite_side_degree_threshold, admissible_floor_bound,
)
from .simulation import WriteSimulation, simulate_write, equilibrium

__all__ = [
    "Graph", "CellModel",
    "StorageCertificate", "WriteCertificate",
    "certify_storage", "certify_single_node_write", "search_amplitude", "margin_ceiling",
    "opposite_side_degree_threshold", "admissible_floor_bound",
    "WriteSimulation", "simulate_write", "equilibrium",
]

__version__ = "1.0.0"
