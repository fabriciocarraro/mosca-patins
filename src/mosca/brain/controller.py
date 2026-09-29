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

Dois corpos (`body`): "skate" (42 saídas, os servos das juntas; o tendão longo, ltm, abaixa o
tarso) e "walk" (M4, a mosca com pés: 42 servos + a adesão das 6 garras, comandada pelo
tendão longo, que no inseto puxa o tendão da garra). O giro pedido entra nos DNa01 e DNa02
do lado da curva (no inseto, eles disparam no giro para o próprio lado).

Ponto de partida: na referência de volume de 1,30× a mediana (a que dá o ritmo do DNg100
sem treino), qualquer entrada sensorial leva a rede inteira a disparar sem controle. Na
normalização padrão do código de Pugliese (1,0×, a mediana da própria rede), com os
proprioceptores logo abaixo do limiar (calados em repouso, ativos quando a pata se mexe) e
um tônus de repouso treinável nos neurônios motores (que quase não mandam sinal de volta
para a rede), ela responde ao movimento das patas sem disparar: ~2.200 neurônios ativos,
motores a ~14 Hz. Sem o tônus, a mosca parada deixa a rede calada e o PPO não aprende nada.
O viés do decodificador é calibrado para que a saída de repouso seja a postura canônica
(ação zero, como a MLP começa); o gradiente substituto da rede alcança os neurônios calados.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from mosca.body.fly import LEGS
from mosca.body.ik import LEG_JOINTS
from mosca.brain.graph import Connectome, sensory_subclass
from mosca.brain.muscles import leg_decoders
from mosca.brain.pugliese import estimate_sizes, sample_params, signed_matrix
from mosca.brain.rate_model import PuglieseNet

N_JOINTS = len(LEG_JOINTS)
GYRO = slice(171, 174)  # velocidade angular do tórax (rad/s) na observação do ambiente


@dataclass(frozen=True)
class ControllerConfig:
    size_ref: float = 1.0  # referência do volume, em múltiplos da mediana da rede (1,0 = como no código deles)
    substeps: int = 2
    control_dt: float = 0.01
    walk_gain: float = 100.0  # corrente nos DNg100 por cm/s pedido, no início
    enc_std: float = 0.5  # desvio inicial dos pesos do codificador (corrente por unidade normalizada)
    prop_offset: float = -2.0  # viés inicial dos proprioceptores em relação ao limiar (abaixo = calados em repouso)
    motor_tone: float = 1.0  # corrente de repouso inicial dos motores acima do limiar
    dec_gain: float = 0.002  # ação por Hz de cada neurônio motor, no início (malha fechada estável)
    init_std: float = 0.6  # desvio inicial do ruído de exploração, em unidades de ação
    seed: int = 0
    body: str = "skate"  # "skate" (42 saídas) ou "walk" (42 + adesão das 6 garras)
    motor_synapse_gains: bool = False  # degrau 2 da escada, só nas ligações que entram nos motores das patas
    all_synapse_gains: bool = False  # degrau 2 completo: ganho positivo por ligação em toda a rede (sinal fixo)
    turn_gain: float = 100.0  # corrente nos DNa01/DNa02 do lado da curva por rad/s pedido
    tau_scale: float = 1.0  # multiplica as constantes de tempo sorteadas (τ ~20 ms no modelo de Pugliese)
    haltere_input: bool = False  # sentido de rotação: o giroscópio do tórax entra nos aferentes dos halteres
    haltere_scale: float = 2.0  # rad/s por unidade na entrada dos halteres
    haltere_offset: float | None = None  # viés inicial dos halteres em relação ao limiar (padrão: prop_offset)
    turn_cells: tuple[str, ...] = ("DNa01", "DNa02")  # descendentes que recebem o comando de giro
    # "float64": estado e parâmetros da rede em precisão dupla. Na GPU, a multiplicação esparsa soma em
    # ordem variável; em float32 a diferença entre duas execuções (1e-3 numa escala de 3 mil) é amplificada
    # pela rede com neurônios de 5 ms até 0,07 na ação em 16 passos, e o PPO (que re-executa a política na
    # atualização) e a re-simulação da captura deixam de reproduzir a coleta. Em float64, 1e-12.
    net_dtype: str = "float32"
    # Determinismo (exige float64): somas exatas na multiplicação esparsa (modo exato da rede, ver
    # mosca.brain.rate_model) e decodificador somando por multiplicação densa, sem atômicas. A mesma
    # política com as mesmas observações dá as mesmas ações bit a bit, e a re-simulação do cérebro
    # (M5, mosca.capture.replay_brain) reproduz a coleta exatamente. Só em float64, a re-simulação de
    # tentativas de 5 s ficou a ~2e-6 das ações gravadas (dentro do critério do M5, sem garantia se a
    # rede ficar mais sensível com o treino). Custa ~17% a mais por passo de controle.
    deterministic: bool = False


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
        if cfg.tau_scale != 1.0:
            params = replace(params, tau=params.tau * cfg.tau_scale)
        leg_motors = np.concatenate([c.groups[f"motor_{leg}"] for leg in LEGS])
        self.net_dtype = torch.float64 if cfg.net_dtype == "float64" else torch.float32
        if cfg.deterministic and cfg.net_dtype != "float64":
            raise ValueError("deterministic=True precisa de net_dtype='float64'")
        self.net = PuglieseNet(signed_matrix(c), params, cell_type=c.cell_type, device=device, surrogate=True,
                               gain_rows=leg_motors if cfg.motor_synapse_gains else None,
                               edge_gains=cfg.all_synapse_gains, dtype=self.net_dtype, exact=cfg.deterministic)
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
        a0, theta0, _ = self.net.neuron_params()
        with torch.no_grad():
            for k in range(len(LEGS)):
                self.enc_b[k].copy_(theta0[getattr(self, f"enc_neurons_{k}"), 0] + cfg.prop_offset)

        # Halteres (opcional): os aferentes dos halteres, os giroscópios da mosca, recebem a velocidade
        # angular do tórax por um mapa linear treinável, como os proprioceptores das patas.
        if cfg.haltere_input:
            halt = torch.as_tensor(sensory_subclass(c, "haltere"), dtype=torch.long, device=device)
            self.register_buffer("halt_neurons", halt)
            self.halt_w = nn.Parameter((cfg.enc_std * torch.randn(len(halt), 3, generator=gen)).to(device))
            offset = cfg.prop_offset if cfg.haltere_offset is None else cfg.haltere_offset
            self.halt_b = nn.Parameter((theta0[halt, 0] + offset).detach().clone().float())

        # Tônus de repouso dos neurônios motores das patas (corrente constante treinável).
        motors = torch.as_tensor(np.concatenate([c.groups[f"motor_{leg}"] for leg in LEGS]), dtype=torch.long, device=device)
        self.register_buffer("motors", motors)
        self.motor_bias = nn.Parameter((theta0[motors, 0] + cfg.motor_tone).detach().clone().float())

        # Comando de velocidade nos dois DNg100.
        dng100 = np.concatenate([c.groups["DNg100_L"], c.groups["DNg100_R"]])
        self.register_buffer("dng100", torch.as_tensor(dng100, dtype=torch.long, device=device))
        self.log_walk_gain = nn.Parameter(torch.tensor(float(np.log(cfg.walk_gain)), device=device))

        # Comando de giro nos descendentes `turn_cells` de cada lado (esquerdo para giro positivo, anti-horário).
        turn_l = np.concatenate([c.groups[f"{cell}_L"] for cell in cfg.turn_cells])
        turn_r = np.concatenate([c.groups[f"{cell}_R"] for cell in cfg.turn_cells])
        self.register_buffer("turn_l", torch.as_tensor(turn_l, dtype=torch.long, device=device))
        self.register_buffer("turn_r", torch.as_tensor(turn_r, dtype=torch.long, device=device))
        self.log_turn_gain = nn.Parameter(torch.tensor(float(np.log(cfg.turn_gain)), device=device))

        # Decodificador: ligações neurônio motor → saída (servos: pata × 7 + junta; adesão: 42 + pata).
        self.n_out = 6 * N_JOINTS + (6 if cfg.body == "walk" else 0)
        motor, out, sign = [], [], []
        decoders = leg_decoders(c)
        for k, leg in enumerate(LEGS):
            dec = decoders[leg]
            if cfg.body == "walk":  # tendão longo -> adesão; motores sem músculo também podem usá-la
                ltm = np.array([str(c.cell_type[i]).startswith("ltm") for i in dec.motor])
                keep = ~ltm
                motor.append(dec.motor[keep])
                out.append(k * N_JOINTS + dec.joint[keep])
                sign.append(dec.sign[keep])
                extra = np.unique(np.concatenate([dec.motor[ltm], dec.motor[dec.sign == 0]]))
                is_ltm = np.array([str(c.cell_type[i]).startswith("ltm") for i in extra], dtype=bool)
                motor.append(extra)
                out.append(np.full(len(extra), 6 * N_JOINTS + k))
                sign.append(np.where(is_ltm, 1, 0).astype(np.int8))
            else:
                motor.append(dec.motor)
                out.append(k * N_JOINTS + dec.joint)
                sign.append(dec.sign)
        motor, out, sign = (np.concatenate(x) for x in (motor, out, sign))
        self.register_buffer("dec_motor", torch.as_tensor(motor, dtype=torch.long, device=device))
        self.register_buffer("dec_out", torch.as_tensor(out, dtype=torch.long, device=device))
        self.register_buffer("dec_sign", torch.as_tensor(sign, dtype=torch.float32, device=device))
        free = sign == 0
        self.register_buffer("dec_free", torch.as_tensor(free, device=device))
        if cfg.deterministic:  # soma por saída como multiplicação densa (o index_add usa atômicas)
            scatter = torch.zeros(self.n_out, len(out))
            scatter[torch.as_tensor(out, dtype=torch.long), torch.arange(len(out))] = 1.0
            self.register_buffer("dec_scatter", scatter.to(device), persistent=False)
        # Ganho ≥ 0 (em log) para músculos identificados; peso livre (qualquer sinal, começa em 0,
        # na escala dec_gain) para os motores sem músculo identificado. Sem a escala, os pesos
        # livres eram 330 vezes mais sensíveis que os ganhos e dominavam a divergência KL do PPO.
        init = float(np.log(cfg.dec_gain))
        self.dec_raw = nn.Parameter(torch.where(torch.as_tensor(free), torch.zeros(len(sign)), torch.full((len(sign),), init)).to(device))
        self.dec_bias = nn.Parameter(torch.zeros(self.n_out, device=device))
        self.log_std = nn.Parameter(torch.full((self.n_out,), float(np.log(cfg.init_std)), device=device))

    # ------------------------------------------------------------ população (estratégias evolutivas)

    def es_parameters(self) -> dict[str, torch.Tensor]:
        """Os parâmetros que as estratégias evolutivas variam, por nome: fatores por tipo celular, tônus
        dos motores, vieses do codificador e dos halteres e o decodificador. Os ganhos por ligação ficam
        fixos (não cabem uma variação por coluna na multiplicação esparsa)."""
        out = {"log_a": self.net.log_a, "log_theta": self.net.log_theta, "log_tau": self.net.log_tau,
               "motor_bias": self.motor_bias, "dec_raw": self.dec_raw, "dec_bias": self.dec_bias,
               "log_walk_gain": self.log_walk_gain}
        for k in range(len(LEGS)):
            out[f"enc_b_{k}"] = self.enc_b[k]
        if self.cfg.haltere_input:
            out["halt_b"] = self.halt_b
        return out

    def set_population(self, offsets: dict[str, torch.Tensor] | None) -> None:
        """Deslocamentos (tamanho do parâmetro, lote) somados a cada coluna do lote, pelos nomes de
        `es_parameters`; None volta a usar os mesmos parâmetros em todas as colunas."""
        self.population = offsets
        self.net.type_offsets = None if offsets is None else {k: v for k, v in offsets.items() if k.startswith("log_")}

    def _col(self, name: str, value: torch.Tensor, batch: int) -> torch.Tensor:
        """Parâmetro como coluna (tamanho, 1) ou, com população, (tamanho, lote)."""
        off = getattr(self, "population", None)
        if off is not None and name in off:
            return value[:, None] + off[name].to(value.dtype)
        return value[:, None].expand(-1, batch)

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parâmetros por grupo, para taxas de aprendizado relativas (ver connectome_train.py)."""
        groups = {"rede": [self.net.log_a, self.net.log_theta, self.net.log_tau],
                "codificador": list(self.enc_w) + list(self.enc_b) + ([self.halt_w, self.halt_b] if self.cfg.haltere_input else []),
                "tonus": [self.motor_bias],
                "decodificador": [self.dec_raw, self.dec_bias],
                "comando": [self.log_walk_gain, self.log_turn_gain],
                "exploracao": [self.log_std]}
        if self.net.has_gains:
            groups["sinapses"] = [self.net.log_gain]
        if self.net.edge_gains:
            groups["sinapses"] = [self.net.log_edge_gain]
        return groups

    @property
    def device(self) -> torch.device:
        return self.log_std.device

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.n, batch, device=self.device, dtype=self.net_dtype)

    def step_net(self, r: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """Avança a rede um passo de controle com a corrente externa (N, lote) dada."""
        return self.net(r, current.to(r.dtype), dt=self.cfg.control_dt, substeps=self.cfg.substeps)

    def currents(self, obs: torch.Tensor, v_cmd: torch.Tensor) -> torch.Tensor:
        """Corrente externa (N, lote) a partir da observação crua do ambiente (lote, obs) e do comando."""
        batch = obs.shape[0]
        current = torch.zeros(self.n, batch, device=self.device)
        for k in range(len(LEGS)):
            neurons, feats = getattr(self, f"enc_neurons_{k}"), getattr(self, f"enc_features_{k}")
            x = (obs[:, feats] - getattr(self, f"enc_center_{k}")) / getattr(self, f"enc_scale_{k}")
            current = current.index_add(0, neurons, self.enc_w[k] @ x.T + self._col(f"enc_b_{k}", self.enc_b[k], batch))
        if self.cfg.haltere_input:
            g = obs[:, GYRO] / self.cfg.haltere_scale
            current = current.index_add(0, self.halt_neurons, self.halt_w @ g.T + self._col("halt_b", self.halt_b, batch))
        current = current.index_add(0, self.motors, self._col("motor_bias", self.motor_bias, batch))
        off = getattr(self, "population", None)
        log_walk = self.log_walk_gain if off is None or "log_walk_gain" not in off else self.log_walk_gain + off["log_walk_gain"]
        drive = log_walk.exp() * v_cmd
        current = current.index_add(0, self.dng100, drive[None, :].expand(len(self.dng100), -1))
        yaw = obs[:, 184]  # giro pedido (rad/s), na observação do ambiente
        turn = self.log_turn_gain.exp()
        current = current.index_add(0, self.turn_l, (turn * torch.relu(yaw))[None, :].expand(len(self.turn_l), -1))
        return current.index_add(0, self.turn_r, (turn * torch.relu(-yaw))[None, :].expand(len(self.turn_r), -1))

    def decode(self, r: torch.Tensor) -> torch.Tensor:
        """Ação média (lote, n_out) a partir das taxas (N, lote)."""
        raw = self._col("dec_raw", self.dec_raw, r.shape[1])
        weight = torch.where(self.dec_free[:, None], self.cfg.dec_gain * raw, self.dec_sign[:, None] * raw.exp())
        contrib = weight * r[self.dec_motor].float()
        if self.cfg.deterministic:
            out = self.dec_scatter @ contrib
        else:
            out = torch.zeros(self.n_out, r.shape[1], device=self.device).index_add(0, self.dec_out, contrib)
        return (out + self._col("dec_bias", self.dec_bias, r.shape[1])).T

    @torch.no_grad()
    def calibrate_rest(self, obs_rest: torch.Tensor, v_cmd: float, seconds: float = 1.0) -> None:
        """Ajusta o viés do decodificador para a saída ser zero com a mosca parada na postura."""
        batch = obs_rest.shape[0]
        r = self.initial_state(batch)
        v = torch.full((batch,), float(v_cmd), device=self.device)
        for _ in range(round(seconds / self.cfg.control_dt)):
            r = self.step_net(r, self.currents(obs_rest, v))
        self.dec_bias -= self.decode(r).mean(dim=0)

    def forward(self, r: torch.Tensor, obs: torch.Tensor, v_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Um passo de controle: devolve o novo estado (N, lote) e a ação média (lote, 42)."""
        r = self.step_net(r, self.currents(obs, v_cmd))
        return r, self.decode(r)


def load_walking(pol: ConnectomePolicy, path: str, device) -> int:
    """Copia do conectoma que anda (checkpoint do m4_distill.py) o que vale nos dois corpos: a rede, os
    codificadores, o tônus e os ganhos de comando; no decodificador, o ganho de cada ligação neurônio
    motor → junta que existe nos dois corpos. Devolve quantos tensores/ganhos foram copiados."""
    src = torch.load(path, weights_only=False, map_location=device)
    weights = dict(src["policy"])
    if src.get("ema"):
        weights.update({k: v.to(device) for k, v in src["ema"].items()})
    own = pol.state_dict()
    fixed = {"net.tau0", "net.a0", "net.theta0", "net.r_max"}
    skip = fixed | {k for k in own if k.startswith("dec_") or k == "log_std"}
    copied = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape and k not in skip}
    own.update(copied)
    pol.load_state_dict(own)
    # Decodificador: ganho de cada ligação (motor, saída) presente nos dois corpos.
    walk_pairs = {(int(m), int(o)): i for i, (m, o) in enumerate(zip(weights["dec_motor"].tolist(), weights["dec_out"].tolist()))}
    n_dec = 0
    with torch.no_grad():
        for i, (m, o) in enumerate(zip(pol.dec_motor.tolist(), pol.dec_out.tolist())):
            j = walk_pairs.get((int(m), int(o)))
            if j is not None and bool(pol.dec_free[i]) == bool(weights["dec_free"][j]):
                pol.dec_raw[i] = weights["dec_raw"][j]
                n_dec += 1
    return len(copied) + n_dec
