"""A rede em torch reproduz a referência em numpy (mosca.brain.pugliese) e deixa o gradiente passar."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mosca.brain.pugliese import RateNet, rhythm, sample_params
from mosca.brain.rate_model import PuglieseNet
from mosca.paths import PUGLIESE_DIR


def _random_net(n=60, seed=0):
    rng = np.random.default_rng(seed)
    w = sp.random(n, n, density=0.15, random_state=seed, format="csr")
    w.data = np.round(w.data * 40) * rng.choice([-1, 1], size=w.nnz)  # nº de sinapses com sinal
    return w, sample_params(rng.uniform(0.5, 2.0, n), rng)


def test_matches_the_numpy_reference():
    w, params = _random_net()
    ref = RateNet(w, params)
    net = PuglieseNet(w, params, dtype=torch.float64)
    current = np.zeros(w.shape[0])
    current[:5] = 300.0
    r_np = np.zeros(w.shape[0])
    r_t = torch.zeros(w.shape[0], 1, dtype=torch.float64)
    for _ in range(200):  # 0,4 s em passos de 2 ms
        r_np = np.clip(ref.rk4(r_np, current, 2e-3), 0.0, 1000.0)
        r_t = net(r_t, torch.from_numpy(current)[:, None], dt=2e-3)
    assert r_np.max() > 1.0  # a rede não ficou em silêncio
    assert np.abs(r_t[:, 0].detach().numpy() - r_np).max() < 1e-8 * max(1.0, r_np.max())


def test_gradient_reaches_the_cell_type_factors():
    w, params = _random_net()
    net = PuglieseNet(w, params, cell_type=np.arange(w.shape[0]) % 7)
    r = torch.zeros(w.shape[0], 4)
    current = torch.zeros(w.shape[0], 4)
    current[:5] = 300.0
    for _ in range(10):
        r = net(r, current, dt=1e-2, substeps=2)
    r[10:20].sum().backward()
    for p in (net.log_a, net.log_theta, net.log_tau):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.skipif(not (PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").exists(),
                    reason="rede de Pugliese não baixada (download_assets.py --pugliese)")
def test_dng100_rhythm_with_two_substeps_per_control_step():
    import pandas as pd

    table = pd.read_csv(PUGLIESE_DIR / "wTable_20260210_vncRoisOnly.csv", index_col=0)
    w = sp.csr_matrix(pd.read_csv(PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").drop(columns="bodyId_pre").to_numpy().T.astype(float))
    motor = np.flatnonzero(table["motor module"].notna().to_numpy())
    net = PuglieseNet(w, sample_params(table["size"].to_numpy(), np.random.default_rng(0)))
    current = torch.zeros(w.shape[0], 1)
    r = torch.zeros(w.shape[0], 1)
    trace, peak = [], torch.zeros(w.shape[0])
    with torch.no_grad():
        for k in range(200):  # 2 s em passos de controle de 10 ms, 2 subpassos de RK4
            current[9, 0] = 400.0 if k >= 2 else 0.0
            r = net(r, current, dt=1e-2, substeps=2)
            trace.append(r[motor, 0].numpy())
            peak = torch.maximum(peak, r[:, 0])
    res = rhythm(np.array(trace).T, peak.numpy(), h=1e-2)
    assert res.score > 0.8 and 7.0 <= res.freq_hz <= 15.0


def test_surrogate_gradient_reaches_a_silent_network():
    w, params = _random_net()
    plain = PuglieseNet(w, params)
    surrogate = PuglieseNet(w, params, surrogate=True)
    grads = []
    for net in (plain, surrogate):
        r = torch.zeros(w.shape[0], 2)
        current = torch.full((w.shape[0], 2), 2.0)  # abaixo de todos os limiares: rede calada
        for _ in range(5):
            r = net(r, current, dt=1e-2, substeps=2)
        assert r.abs().max() == 0.0  # a simulação é a mesma: ninguém dispara
        r.sum().backward()
        grads.append(net.log_theta.grad.abs().sum().item())
    assert grads[0] == 0.0 and grads[1] > 0.0


def test_synapse_gains_start_neutral_and_learn():
    w, params = _random_net()
    plain = PuglieseNet(w, params, dtype=torch.float64)
    gained = PuglieseNet(w, params, dtype=torch.float64, gain_rows=np.arange(10, 20))
    assert gained.log_gain.numel() == w[10:20].nnz
    current = torch.zeros(w.shape[0], 2, dtype=torch.float64)
    current[:5] = 300.0
    r1 = r2 = torch.zeros(w.shape[0], 2, dtype=torch.float64)
    for _ in range(20):
        r1, r2 = plain(r1, current, dt=2e-3), gained(r2, current, dt=2e-3)
    assert torch.allclose(r1, r2, atol=1e-9)  # ganhos em 1: mesma rede
    r2[10:20].sum().backward()
    assert gained.log_gain.grad is not None and gained.log_gain.grad.abs().sum() > 0
