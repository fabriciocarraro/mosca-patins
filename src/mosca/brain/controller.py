"""Controlador de conectoma: a rede de Pugliese com a fiação do MaleCNS dirigindo as patas.

A cada passo de controle (10 ms):
1. Codificador por pata: os sentidos daquela pata (ângulo e velocidade das 7 juntas e a
   carga no patim, em escalas físicas fixas) viram correntes só nos proprioceptores dela (órgão
   cordotonal, placas de pelos, sensilas). É um mapa linear treinável; nenhum outro neurônio
   recebe entrada sensorial. (A anotação do MaleCNS tem menos proprioceptores na pata da
   frente direita: 23 contra 64 na esquerda.)
2. Comando: a velocidade pedida vira corrente nos dois DNg100 (ganho treinável).
3. A rede avança 10 ms (2 subpassos de RK4), com os parâmetros por tipo celular treináveis.
4. Decodificador por pata: cada neurônio motor move a junta do seu músculo, no sentido da
   tabela (mosca.brain.muscles), com ganho ≥ 0 treinável; motores sem músculo identificado
   têm peso livre só para as juntas da própria pata. A saída é a ação média da política, na
   mesma escala da MLP (desvio da postura canônica, 0,15 rad por unidade).

Nada de camadas escondidas: entre sentidos e comando, de um lado, e os servos, do outro, só
a fiação do conectoma. A exploração do PPO soma ruído à ação média, como na MLP.

Ponto de partida: na referência de volume de 1,30× a mediana (a que dá o ritmo do DNg100
sem treino), qualquer entrada sensorial leva a rede inteira a disparar sem controle. Na
normalização padrão do código de Pugliese (1,0×, a mediana da própria rede) e com o
codificador fraco, ela fica num regime calmo e responsivo: ~2.600 neurônios passam por
atividade e ~14 motores por ambiente, com ação inicial quase nula (a mosca parada na
postura, como a MLP). O gradiente substituto da rede deixa o PPO acordar os neurônios certos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mosca.body.fly import LEGS
from mosca.body.ik import LEG_JOINTS
from mosca.brain.graph import Connectome
from mosca.brain.muscles import leg_decoders
from mosca.brain.pugliese import estimate_sizes, sample_params, signed_matrix
from mosca.brain.rate_model import PuglieseNet

N_JOINTS = len(LEG_JOINTS)


@dataclass(frozen=True)
class ControllerConfig:
    size_ref: float = 1.0  # referência do volume, em múltiplos da mediana da rede (1,0 = como no código deles)
    substeps: int = 2
    control_dt: float = 0.01
    walk_gain: float = 100.0  # corrente nos DNg100 por cm/s pedido, no início
    enc_std: float = 0.5  # desvio inicial dos pesos do codificador (corrente por unidade normalizada)
    dec_gain: float = 0.005  # ação por Hz de cada neurônio motor, no início
    init_std: float = 0.6  # desvio inicial do ruído de exploração, em unidades de ação
    seed: int = 0


def leg_feature_index(leg: int, act_dim: int = 6 * N_JOINTS) -> np.ndarray:
    """Posições, na observação do SkateVecEnv, do ângulo e da velocidade das juntas da pata e
    da carga no seu patim (ver SkateVecEnv._measure)."""
    joints = np.arange(leg * N_JOINTS, (leg + 1) * N_JOINTS)
    loads = 4 * act_dim + 9 + leg
    return np.concatenate([joints, act_dim + joints, [loads]])


# Escalas fixas dos sentidos da pata, na ordem de leg_feature_index: ângulo em relação à postura
# canônica (0,2 rad), velocidade da junta (a observação já traz 0,1 × rad/s: 10 rad/s por
# unidade) e carga no patim (fração do peso, centrada em 1/6, 0,2 por unidade).
FEATURE_CENTER = np.concatenate([np.zeros(2 * N_JOINTS), [1.0 / 6.0]])
FEATURE_SCALE = np.concatenate([np.full(N_JOINTS, 0.2), np.full(N_JOINTS, 1.0), [0.2]])


class ConnectomePolicy(nn.Module):
    def __init__(self, c: Connectome, cfg: ControllerConfig = ControllerConfig(), device="cpu"):
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        sizes, _, _ = estimate_sizes(c.body_id)
        params = sample_params(sizes, rng, reference=cfg.size_ref * np.nanmedian(sizes))
        self.net = PuglieseNet(signed_matrix(c), params, cell_type=c.cell_type, device=device, surrogate=True)
        self.n = c.n
        gen = torch.Generator().manual_seed(cfg.seed)

        # Codificador: um mapa linear por pata, só para os proprioceptores daquela pata.
        self.enc_neurons, self.enc_features = [], []
        self.enc_w, self.enc_b = nn.ParameterList(), nn.ParameterList()
        for k, leg in enumerate(LEGS):
            neurons = torch.as_tensor(c.groups[f"proprio_{leg}"], dtype=torch.long)
            feats = torch.as_tensor(leg_feature_index(k), dtype=torch.long)
            self.register_buffer(f"enc_neurons_{k}", neurons.to(device))
            self.register_buffer(f"enc_features_{k}", feats.to(device))
            self.register_buffer(f"enc_center_{k}", torch.as_tensor(FEATURE_CENTER, dtype=torch.float32, device=device))
            self.register_buffer(f"enc_scale_{k}", torch.as_tensor(FEATURE_SCALE, dtype=torch.float32, device=device))
            self.enc_w.append(nn.Parameter((cfg.enc_std * torch.randn(len(neurons), len(feats), generator=gen)).to(device)))
            self.enc_b.append(nn.Parameter(torch.zeros(len(neurons), device=device)))

        # Comando de velocidade nos dois DNg100.
        dng100 = np.concatenate([c.groups["DNg100_L"], c.groups["DNg100_R"]])
        self.register_buffer("dng100", torch.as_tensor(dng100, dtype=torch.long, device=device))
        self.log_walk_gain = nn.Parameter(torch.tensor(float(np.log(cfg.walk_gain)), device=device))

        # Decodificador: ligações neurônio motor → junta (índice na ação de 42 posições).
        motor, out, sign = [], [], []
        decoders = leg_decoders(c)
        for k, leg in enumerate(LEGS):
            dec = decoders[leg]
            motor.append(dec.motor)
            out.append(k * N_JOINTS + dec.joint)
            sign.append(dec.sign)
        motor, out, sign = (np.concatenate(x) for x in (motor, out, sign))
        self.register_buffer("dec_motor", torch.as_tensor(motor, dtype=torch.long, device=device))
        self.register_buffer("dec_out", torch.as_tensor(out, dtype=torch.long, device=device))
        self.register_buffer("dec_sign", torch.as_tensor(sign, dtype=torch.float32, device=device))
        free = sign == 0
        self.register_buffer("dec_free", torch.as_tensor(free, device=device))
        # Ganho ≥ 0 (em log) para músculos identificados; peso livre (qualquer sinal, começa em 0)
        # para os motores sem músculo identificado.
        init = float(np.log(cfg.dec_gain))
        self.dec_raw = nn.Parameter(torch.where(torch.as_tensor(free), torch.zeros(len(sign)), torch.full((len(sign),), init)).to(device))
        self.dec_bias = nn.Parameter(torch.zeros(6 * N_JOINTS, device=device))
        self.log_std = nn.Parameter(torch.full((6 * N_JOINTS,), float(np.log(cfg.init_std)), device=device))

    @property
    def device(self) -> torch.device:
        return self.log_std.device

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.n, batch, device=self.device)

    def currents(self, obs: torch.Tensor, v_cmd: torch.Tensor) -> torch.Tensor:
        """Corrente externa (N, lote) a partir da observação crua do ambiente (lote, obs) e do comando."""
        batch = obs.shape[0]
        current = torch.zeros(self.n, batch, device=self.device)
        for k in range(len(LEGS)):
            neurons, feats = getattr(self, f"enc_neurons_{k}"), getattr(self, f"enc_features_{k}")
            x = (obs[:, feats] - getattr(self, f"enc_center_{k}")) / getattr(self, f"enc_scale_{k}")
            current = current.index_add(0, neurons, self.enc_w[k] @ x.T + self.enc_b[k][:, None])
        drive = self.log_walk_gain.exp() * v_cmd
        return current.index_add(0, self.dng100, drive[None, :].expand(len(self.dng100), -1))

    def decode(self, r: torch.Tensor) -> torch.Tensor:
        """Ação média (lote, 42) a partir das taxas (N, lote)."""
        weight = torch.where(self.dec_free, self.dec_raw, self.dec_sign * self.dec_raw.exp())
        contrib = weight[:, None] * r[self.dec_motor]
        out = torch.zeros(6 * N_JOINTS, r.shape[1], device=self.device).index_add(0, self.dec_out, contrib)
        return out.T + self.dec_bias

    def forward(self, r: torch.Tensor, obs: torch.Tensor, v_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Um passo de controle: devolve o novo estado (N, lote) e a ação média (lote, 42)."""
        r = self.net(r, self.currents(obs, v_cmd), dt=self.cfg.control_dt, substeps=self.cfg.substeps)
        return r, self.decode(r)
