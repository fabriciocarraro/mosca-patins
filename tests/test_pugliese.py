import numpy as np
import pytest
import scipy.sparse as sp

from mosca.brain.pugliese import (SYNAPSE_COUNTS, NeuronParams, RateNet, estimate_sizes, neuron_score, rhythm,
                                  sample_params, signed_matrix, simulate)
from mosca.paths import MALECNS_DIR, PUGLIESE_DIR

HAS_PUGLIESE = (PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").exists()
HAS_COUNTS = (MALECNS_DIR / SYNAPSE_COUNTS).exists()


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
    rec = simulate(net, np.zeros(n), seconds=0.2)
    assert rec.rates.max() == 0.0
    assert rec.peak.max() == 0.0


def test_single_neuron_settles_at_the_activation():
    # τ dr/dt = r_max·tanh((a/r_max)(I − θ)) − r: com um neurônio só, r vai para esse valor.
    p = NeuronParams(tau=np.array([0.02]), a=np.array([1.0]), theta=np.array([7.5]), r_max=np.array([200.0]))
    net = RateNet(sp.csr_matrix((1, 1)), p)
    rec = simulate(net, np.array([100.0]), seconds=0.5, pulse=(0.0, 1.0))
    assert rec.rates[0, -1] == pytest.approx(200.0 * np.tanh((100.0 - 7.5) / 200.0), rel=1e-3)


def _pugliese_network():
    import pandas as pd

    table = pd.read_csv(PUGLIESE_DIR / "wTable_20260210_vncRoisOnly.csv", index_col=0)
    w = pd.read_csv(PUGLIESE_DIR / "W_20260210_vncRoisOnly.csv").drop(columns="bodyId_pre").to_numpy()
    motor = np.flatnonzero(table["motor module"].notna().to_numpy())
    current = np.zeros(len(table))
    current[9] = 400.0  # DNg100, como no experimento publicado
    return sp.csr_matrix(w.T.astype(float)), table, motor, current


@pytest.mark.skipif(not HAS_PUGLIESE, reason="rede de Pugliese não baixada (download_assets.py --pugliese)")
def test_dng100_drives_a_leg_rhythm():
    w, table, motor, current = _pugliese_network()
    net = RateNet(w, sample_params(table["size"].to_numpy(), np.random.default_rng(0)))
    rec = simulate(net, current, record=motor)
    res = rhythm(rec.rates, rec.peak)
    assert res.score > 0.9
    assert 7.0 <= res.freq_hz <= 15.0
    assert res.active_motor >= 5


@pytest.mark.skipif(not (HAS_PUGLIESE and HAS_COUNTS),
                    reason="faltam a rede de Pugliese ou as contagens de sinapses (--pugliese, --malecns-stats)")
def test_estimated_sizes_keep_the_rhythm_and_shuffled_sizes_break_it():
    w, table, motor, current = _pugliese_network()
    real = table["size"].to_numpy(dtype=float)
    sizes, estimated, model = estimate_sizes(table["bodyId"].to_numpy(), use_real=False)
    assert estimated.all() and model.r > 0.9 and model.spread < 1.6
    mixed, estimated_mixed, _ = estimate_sizes(table["bodyId"].to_numpy())
    assert np.array_equal(mixed[~estimated_mixed], real[~estimated_mixed]) and estimated_mixed.sum() <= 1

    def score(s):
        rec = simulate(RateNet(w, sample_params(s, np.random.default_rng(0))), current, record=motor)
        return rhythm(rec.rates, rec.peak)

    good = score(sizes)
    assert good.score > 0.9 and 7.0 <= good.freq_hz <= 15.0
    assert score(np.random.default_rng(1).permutation(real)).score < 0.5


GRAPH = MALECNS_DIR / "controller_graph_min5.npz"


@pytest.mark.skipif(not (HAS_PUGLIESE and HAS_COUNTS and GRAPH.exists()),
                    reason="faltam o grafo do controlador, a rede de Pugliese ou as contagens de sinapses")
def test_dng100_rhythm_on_the_six_leg_graph():
    from mosca.body.fly import LEGS
    from mosca.brain.graph import Connectome

    c = Connectome.load(GRAPH)
    sizes, _, _ = estimate_sizes(c.body_id)
    motor = np.concatenate([c.groups[f"motor_{leg}"] for leg in LEGS])
    current = np.zeros(c.n)
    current[c.groups["DNg100_R"]] = 400.0
    net = RateNet(signed_matrix(c, "consensus"),
                  sample_params(sizes, np.random.default_rng(0), reference=1.30 * np.nanmedian(sizes)))
    rec = simulate(net, current, record=motor)
    res = rhythm(rec.rates, rec.peak)
    assert res.score > 0.5 and 7.0 <= res.freq_hz <= 15.0
    assert 3 <= res.active_motor <= 30 and res.active < 1000  # ritmo localizado, sem disparo descontrolado
