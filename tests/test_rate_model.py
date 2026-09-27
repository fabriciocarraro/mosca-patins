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
    r_t = torch.zeros(1, w.shape[0], dtype=torch.float64)
    for _ in range(200):  # 0,4 s em passos de 2 ms
        r_np = np.clip(ref.rk4(r_np, current, 2e-3), 0.0, 1000.0)
        r_t = net(r_t, torch.from_numpy(current)[None], dt=2e-3)
    assert r_np.max() > 1.0  # a rede não ficou em silêncio
    assert np.abs(r_t[0].detach().numpy() - r_np).max() < 1e-8 * max(1.0, r_np.max())


def test_gradient_reaches_the_cell_type_factors():
    w, params = _random_net()
    net = PuglieseNet(w, params, cell_type=np.arange(w.shape[0]) % 7)
    r = torch.zeros(4, w.shape[0])
    current = torch.zeros(4, w.shape[0])
    current[:, :5] = 300.0
    for _ in range(10):
        r = net(r, current, dt=1e-2, substeps=2)
    r[:, 10:20].sum().backward()
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
    current = torch.zeros(1, w.shape[0])
    r = torch.zeros(1, w.shape[0])
    trace, peak = [], torch.zeros(w.shape[0])
    with torch.no_grad():
        for k in range(200):  # 2 s em passos de controle de 10 ms, 2 subpassos de RK4
            current[0, 9] = 400.0 if k >= 2 else 0.0
            r = net(r, current, dt=1e-2, substeps=2)
            trace.append(r[0, motor].numpy())
            peak = torch.maximum(peak, r[0])
    res = rhythm(np.array(trace).T, peak.numpy(), h=1e-2)
    assert res.score > 0.8 and 7.0 <= res.freq_hz <= 15.0
