"""Confere o grafo do controlador (gerado por scripts/m3_build_graph.py; pulado se não existir)."""

import numpy as np
import pytest

from mosca.body.fly import LEGS
from mosca.brain.graph import Connectome
from mosca.paths import MALECNS_DIR

GRAPH = MALECNS_DIR / "controller_graph_min5.npz"
pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason="grafo não gerado (rode scripts/m3_build_graph.py)")

EXPECTED_MOTORS = {"T1_left": 68, "T1_right": 67, "T2_left": 58, "T2_right": 58, "T3_left": 66, "T3_right": 64}


@pytest.fixture(scope="module")
def graph():
    return Connectome.load(GRAPH)


def test_all_leg_motor_neurons_are_kept(graph):
    for leg in LEGS:
        assert len(graph.groups[f"motor_{leg}"]) == EXPECTED_MOTORS[leg]


def test_every_leg_has_sensory_input(graph):
    for leg in LEGS:
        assert len(graph.groups[f"sensory_{leg}"]) > 100


def test_command_neurons_present_on_both_sides(graph):
    for cell_type in ("DNg100", "DNa01", "DNa02", "MDN"):
        for side in "LR":
            assert len(graph.groups[f"{cell_type}_{side}"]) >= 1


def test_edges_are_valid(graph):
    assert graph.count.min() >= 5
    assert graph.pre.max() < graph.n and graph.post.max() < graph.n
    assert set(np.unique(graph.sign)) <= {-1, 0, 1}


def test_groups_do_not_mix_sides(graph):
    for leg in LEGS:
        side = "L" if leg.endswith("left") else "R"
        for kind in ("motor", "sensory"):
            assert (graph.side[graph.groups[f"{kind}_{leg}"]] == side).all()
