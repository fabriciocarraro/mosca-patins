import numpy as np
import pytest
import scipy.sparse as sp

from mosca.brain.pugliese import NeuronParams, RateNet, neuron_score, rhythm, sample_params, simulate
from mosca.paths import PUGLIESE_DIR


def test_sine_scores_one_at_the_right_frequency():
    t = np.arange(1750) * 1e-3  # 1,75 s a 1 kHz, como depois de descartar o transiente
    score, freq = neuron_score(np.sin(2 * np.pi * 11.0 * t) + 1.0)
    assert score > 0.95
    assert freq / 1e-3 == pytest.approx(11.0, rel=0.05)


def test_noise_scores_low():
    rng = np.random.default_rng(0)
    score, _ = neuron_score(rng.normal(size=1750))
    assert score < 0.5


def test_network_stays_silent_without_input():
    rng = np.random.default_rng(0)
    n = 50
    w = sp.random(n, n, density=0.1, random_state=1, format="csr") * 20
    net = RateNet(w, sample_params(np.ones(n), rng))
    rates = simulate(net, np.zeros(n), seconds=0.2)
    assert rates.max() == 0.0


def test_single_neuron_settles_at_the_activation():
    # τ dr/dt = r_max·tanh((a/r_max)(I − θ)) − r: com um neurônio só, r vai para esse valor.
    p = NeuronParams(tau=np.array([0.02]), a=np.array([1.0]), theta=np.array([7.5]), r_max=np.array([200.0]))
    net = RateNet(sp.csr_matrix((1, 1)), p)
    rates = simulate(net, np.array([100.0]), seconds=0.5, pulse=(0.0, 1.0))
    assert rates[0, -1] == pytest.approx(200.0 * np.tanh((100.0 - 7.5) / 200.0), rel=1e-3)


@pytest.mark.skipif(not (PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").exists(),
                    reason="rede de Pugliese não baixada (download_assets.py --pugliese)")
def test_dng100_drives_a_leg_rhythm():
    import pandas as pd

    table = pd.read_csv(PUGLIESE_DIR / "wTable_20260210_vncRoisOnly.csv", index_col=0)
    w = pd.read_csv(PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").drop(columns="bodyId_pre").to_numpy()
    net = RateNet(sp.csr_matrix(w.T.astype(float)), sample_params(table["size"].to_numpy(), np.random.default_rng(0)))
    current = np.zeros(len(table))
    current[9] = 400.0  # DNg100, como no experimento publicado
    res = rhythm(simulate(net, current), np.flatnonzero(table["motor module"].notna().to_numpy()))
    assert res.score > 0.9
    assert 7.0 <= res.freq_hz <= 15.0
    assert res.active_motor >= 5
