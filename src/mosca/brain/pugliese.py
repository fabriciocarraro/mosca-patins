"""Modelo de taxa do cordão nervoso de Pugliese et al. (2026), reproduzido pela receita do código deles.

    τ_i·dr_i/dt = max(r_max,i · tanh((a_i / r_max,i) · (I_i + b·Σ_j w_ij r_j − θ_i)), 0) − r_i

- w: nº de sinapses com sinal (acetilcolina +, GABA/glutamato −, resto 0), piso de 5 sinapses.
- b = 0,03 multiplica só a soma sináptica.
- Parâmetros por neurônio, normais truncadas em 0: τ ~ N(0,02; 0,002) s, a* ~ N(1; 0,1),
  θ* ~ N(7,5; 0,6), r_max ~ N(200; 10) Hz. Escala pelo volume: a = a*/s e θ = θ*·s, com
  s = volume do neurônio / mediana dos volumes da rede simulada. Sem essa escala não há ritmo.
- Integração RK4 com passo de 1 ms, que bate com o solver adaptativo deles (r = 0,9999);
  Euler só serve com passo ≤ 0,1 ms.
- Nota de oscilação: porta de `compute_oscillation_score` (autocorrelação normalizada,
  proeminência própria deles, comparação com um seno da mesma frequência).
- Volume: o deles vem do neuPrint (exige login) e só cobre os 4310 neurônios da rede deles.
  Para os outros, `estimate_sizes` estima o volume pela contagem de sinapses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.signal import correlate
from scipy.special import ndtr, ndtri

from mosca.brain.graph import FAST_SIGN, NEUROTRANSMITTERS, Connectome
from mosca.paths import MALECNS_DIR, PUGLIESE_DIR

B = 0.03
PUGLIESE_TABLE = "wTable_20260210_vncRoisOnly.csv"
SYNAPSE_COUNTS = "body-synapse-counts.feather"  # pré e pós de cada neurônio anotado (download_assets --malecns-stats)
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"


@dataclass(frozen=True)
class NeuronParams:
    tau: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    r_max: np.ndarray


def trunc_normal(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    """Normal truncada em 0 por inversão da distribuição acumulada (como `sample_trunc_normal`)."""
    lo = np.clip((0.0 - mean) / sd, -10, 10)
    hi = np.clip(min(100 * sd, 1e6) / sd, -10, 10)
    u = rng.uniform(ndtr(lo), ndtr(hi), size=size)
    return mean + sd * ndtri(np.clip(u, 1e-10, 1 - 1e-10))


def sample_params(sizes: np.ndarray, rng: np.random.Generator, reference: float | None = None) -> NeuronParams:
    """Parâmetros por neurônio. O volume entra relativo a `reference` (por padrão, a mediana da
    própria rede, como no código deles)."""
    n = len(sizes)
    s = np.asarray(sizes, dtype=float).copy()
    median = np.nanmedian(s)
    s[np.isnan(s) | (s == 0)] = median
    s /= median if reference is None else reference
    tau = trunc_normal(rng, 0.02, 0.002, n)
    a = trunc_normal(rng, 1.0, 0.1, n) / s
    theta = trunc_normal(rng, 7.5, 0.6, n) * s
    r_max = trunc_normal(rng, 200.0, 10.0, n)
    return NeuronParams(tau, a, theta, r_max)


@dataclass(frozen=True)
class SizeModel:
    """log(volume) = α + β·log(sinapses de entrada + saída) + desvio médio da superclasse."""

    alpha: float
    beta: float
    offset: dict[str, float]
    r: float  # correlação no ajuste
    spread: float  # erro típico, como fator multiplicativo


def estimate_sizes(body_ids: np.ndarray, use_real: bool = True, pugliese_dir: Path = PUGLIESE_DIR,
                   data_dir: Path = MALECNS_DIR) -> tuple[np.ndarray, np.ndarray, SizeModel]:
    """Volume de cada neurônio: o real (tabela de Pugliese) quando existe e `use_real`, senão o estimado.

    A estimativa é ajustada nos neurônios da rede de Pugliese e usa as sinapses totais de cada
    neurônio no MaleCNS. Nessa rede, trocar todos os volumes reais pelos estimados mantém o
    ritmo do DNg100, e embaralhar os volumes acaba com ele (teste em tests/test_pugliese.py).
    Devolve (volumes, máscara dos estimados, modelo).
    """
    table = pd.read_csv(pugliese_dir / PUGLIESE_TABLE, index_col=0)
    counts = pd.read_feather(data_dir / SYNAPSE_COUNTS, columns=["body", "pre", "post"]).set_index("body")
    superclass = pd.read_feather(data_dir / ANNOTATIONS, columns=["bodyId", "superclass"]).set_index("bodyId").superclass
    synapses = counts.pre + counts.post

    real = pd.Series(table["size"].to_numpy(dtype=float), index=table["bodyId"].to_numpy())
    syn = synapses.reindex(real.index)
    fit = (real > 0) & (syn > 0)
    x, y = np.log(syn[fit].to_numpy(dtype=float)), np.log(real[fit].to_numpy())
    beta, alpha = np.polyfit(x, y, 1)
    resid = pd.Series(y - (alpha + beta * x), index=real.index[fit])
    by_class = resid.groupby(superclass.reindex(resid.index).fillna("")).agg(["mean", "size"])
    offset = {k: float(v) for k, v in by_class["mean"][by_class["size"] >= 20].items() if k}
    shift = superclass.reindex(resid.index).map(offset).fillna(0.0).to_numpy()
    final = resid.to_numpy() - shift
    model = SizeModel(float(alpha), float(beta), offset, float(np.corrcoef(y, y - final)[0, 1]), float(np.exp(final.std())))

    body_ids = np.asarray(body_ids)
    own = real.reindex(body_ids).to_numpy()
    syn_own = synapses.reindex(body_ids).to_numpy(dtype=float)
    shift_own = superclass.reindex(body_ids).map(offset).fillna(0.0).to_numpy()
    with np.errstate(divide="ignore"):
        guess = np.exp(alpha + beta * np.log(syn_own) + shift_own)
    guess[~(syn_own > 0)] = np.nan  # sem sinapse conhecida: sample_params usa a mediana
    estimated = ~(own > 0) if use_real else np.ones(len(body_ids), dtype=bool)
    return np.where(estimated, guess, own), estimated, model


def signed_matrix(c: Connectome, signs: str = "consensus", data_dir: Path = MALECNS_DIR) -> sp.csr_matrix:
    """Pós × pré com sinal × nº de sinapses. "consensus": só o transmissor de consenso, como
    Pugliese; "graph": o sinal do grafo, que completa os incertos pela previsão."""
    if signs == "consensus":
        nt = pd.read_feather(data_dir / NEUROTRANSMITTERS, columns=["body", "consensus_nt"]).set_index("body")
        sign = nt.consensus_nt.reindex(c.body_id).map(FAST_SIGN).fillna(0).to_numpy()
    else:
        sign = c.sign.astype(float)
    values = sign[c.pre] * c.count
    keep = values != 0
    return sp.csr_matrix((values[keep], (c.post[keep], c.pre[keep])), shape=(c.n, c.n))


class RateNet:
    """Rede de taxa em numpy/scipy; `w` é a matriz pós × pré de sinapses com sinal (sem o b)."""

    def __init__(self, w: sp.csr_matrix, params: NeuronParams, b: float = B):
        self.w = (b * w).tocsr()
        self.p = params

    @property
    def n(self) -> int:
        return self.w.shape[0]

    def deriv(self, r: np.ndarray, current: np.ndarray) -> np.ndarray:
        p = self.p
        drive = self.w @ r + current - p.theta
        return (np.maximum(p.r_max * np.tanh((p.a / p.r_max) * drive), 0.0) - r) / p.tau

    def rk4(self, r: np.ndarray, current: np.ndarray, h: float) -> np.ndarray:
        k1 = self.deriv(r, current)
        k2 = self.deriv(r + 0.5 * h * k1, current)
        k3 = self.deriv(r + 0.5 * h * k2, current)
        k4 = self.deriv(r + h * k3, current)
        return r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass(frozen=True)
class Recording:
    rates: np.ndarray  # (neurônios gravados, passos+1), a cada h
    peak: np.ndarray  # (N,) taxa máxima de cada neurônio da rede


def simulate(net: RateNet, current: np.ndarray, seconds: float = 2.0, h: float = 1e-3,
             pulse: tuple[float, float] = (0.02, 1.999), record: np.ndarray | None = None) -> Recording:
    """Simula a partir do repouso, com a corrente ligada durante `pulse`, gravando os neurônios de
    `record` (todos, se None)."""
    steps = round(seconds / h)
    record = np.arange(net.n) if record is None else np.asarray(record)
    out = np.zeros((len(record), steps + 1))
    r = np.zeros(net.n)
    peak = np.zeros(net.n)
    zero = np.zeros(net.n)
    for k in range(steps):
        t = k * h
        # Como no solver deles, a corrente vale nos instantes dentro do pulso.
        on = pulse[0] <= t <= pulse[1]
        r = np.clip(net.rk4(r, current if on else zero, h), 0.0, 1000.0)
        out[:, k + 1] = r[record]
        np.maximum(peak, r, out=peak)
    return Recording(out, peak)


def _peaks(x: np.ndarray, min_prominence: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    is_peak = np.zeros(n, dtype=bool)
    is_peak[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    left_valleys = np.concatenate([[np.inf], np.minimum.accumulate(x)[:-1]])
    right_valleys = np.concatenate([np.minimum.accumulate(x[::-1])[::-1][1:], [np.inf]])
    inside = (np.arange(n) > 0) & (np.arange(n) < n - 1)
    prominence = np.where(inside, x - np.maximum(left_valleys, right_valleys), 0.0)
    return is_peak & (prominence >= min_prominence), prominence


def _raw_score(trace: np.ndarray) -> tuple[float, float]:
    lo, hi = trace.min(), trace.max()
    x = 2 * (trace - lo) / (hi - lo) - 1 if hi - lo > 1e-6 else np.zeros_like(trace)
    ac = correlate(x, x, mode="full", method="fft")
    if np.abs(ac).max() > 1e-6:
        ac = ac / np.abs(ac).max()
    ac = ac[len(ac) // 2 :]
    peaks, prominence = _peaks(ac)
    peaks[0] = False
    if not peaks.any():
        return 0.0, 0.0
    best = int(np.argmax(np.where(peaks, prominence, -np.inf)))
    return float(min(ac[peaks].max(), prominence[peaks].max())), 1.0 / best  # ciclos por amostra


def neuron_score(trace: np.ndarray) -> tuple[float, float]:
    """Nota de 0 a 1 (quão parecido com um seno) e frequência em ciclos por amostra."""
    raw, freq = _raw_score(trace)
    if raw <= 1e-6:
        return 0.0, freq
    t = np.arange(len(trace), dtype=float)
    reference = max(_raw_score(np.sin(2 * np.pi * freq * t))[0], _raw_score(np.cos(2 * np.pi * freq * t))[0])
    return (float(np.clip(raw / reference, 0.0, 1.0)) if reference > 1e-6 else 0.0), freq


@dataclass(frozen=True)
class Rhythm:
    score: float  # média entre os neurônios motores ativos
    freq_hz: float
    active_motor: int
    active: int  # neurônios ativos na rede inteira


def rhythm(motor_rates: np.ndarray, peak: np.ndarray, h: float = 1e-3, skip: float = 0.25) -> Rhythm:
    """Ritmo dos neurônios motores (linhas de `motor_rates`) depois de descartar `skip` s."""
    tail = motor_rates[:, round(skip / h):]
    scores, freqs = [], []
    active = [trace for trace in tail if trace.max() > 0.01]
    for trace in active:
        s, f = neuron_score(trace)
        scores.append(s)
        if f > 0:
            freqs.append(f / h)
    return Rhythm(float(np.mean(scores)) if scores else 0.0, float(np.mean(freqs)) if freqs else float("nan"),
                  len(active), int((peak > 0.01).sum()))
