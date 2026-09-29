"""Estratégias evolutivas: a população em colunas equivale a moscas separadas, e o passo sobe o retorno."""

import numpy as np
import pytest
import torch

from mosca.paths import MALECNS_DIR
from mosca.rl.es import PopulationES, centered_ranks

GRAPH = MALECNS_DIR / "controller_graph_min5.npz"


def test_ranks_and_ascent_on_a_simple_problem():
    assert np.allclose(centered_ranks(np.array([3.0, -1.0, 10.0])), [0.0, -0.5, 0.5])
    target = torch.tensor([1.0, -2.0, 0.5])
    p = torch.nn.Parameter(torch.zeros(3))
    es = PopulationES({"p": p}, {"p": 0.1}, lr=0.5, seed=0)
    start = -float(((p - target) ** 2).sum())
    for gen in range(60):
        eps = es.sample(gen, 32)
        cand = p.detach()[None, :] + 0.1 * eps
        es.step(eps, -((cand - target) ** 2).sum(1).numpy())
    assert -float(((p - target) ** 2).sum()) > start + 3.0


@pytest.mark.skipif(not GRAPH.exists(), reason="grafo do controlador não montado")
def test_population_column_matches_a_separate_fly():
    from mosca.brain.controller import ConnectomePolicy, ControllerConfig
    from mosca.brain.graph import Connectome

    cfg = ControllerConfig(control_dt=0.01, substeps=2, tau_scale=0.25, all_synapse_gains=True, haltere_input=True,
                           haltere_scale=0.5, haltere_offset=0.0, net_dtype="float64")
    pol = ConnectomePolicy(Connectome.load(GRAPH), cfg)
    for q in pol.parameters():
        q.requires_grad_(False)
    params = pol.es_parameters()
    gen = torch.Generator().manual_seed(0)
    off = {k: (0.05 * torch.randn(*v.shape, 2, generator=gen, dtype=torch.float64)).to(v.dtype) for k, v in params.items()}
    for v in off.values():
        v[..., 0] = 0.0  # coluna 0: sem variação
    obs = torch.randn(2, 400, generator=gen)
    vcmd = torch.full((2,), 1.5)
    pol.set_population(off)
    r = pol.initial_state(2)
    for _ in range(3):
        r, both = pol(r, obs, vcmd)
    pol.set_population(None)
    r0 = pol.initial_state(1)
    for _ in range(3):
        r0, first = pol(r0, obs[:1], vcmd[:1])
    for k, v in params.items():  # a variação da coluna 1 aplicada direto nos parâmetros
        v.add_(off[k][..., 1])
    r1 = pol.initial_state(1)
    for _ in range(3):
        r1, second = pol(r1, obs[1:], vcmd[1:])
    assert torch.allclose(both[0], first[0], atol=1e-5) and torch.allclose(both[1], second[0], atol=1e-5)
