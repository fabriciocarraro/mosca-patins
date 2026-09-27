"""Tabela músculo → junta do flybody, com sinal fixo, para o decodificador de cada pata.

No MaleCNS, o tipo de cada neurônio motor da pata é o nome do músculo que ele inerva. Cada
músculo move uma junta do modelo num sentido; o decodificador só pode usar o neurônio nesse
sentido (ganho ≥ 0 vezes o sinal da tabela). Os agrupamentos seguem os módulos motores de
Pugliese et al. (balanço e apoio da coxa, flexão e extensão do fêmur e da tíbia, tarso).

Sentido positivo de cada junta no flybody, medido na postura de patinação (perturbando a
junta e vendo para onde vai a ponta da tíbia; ver tests/test_muscles.py):
- tibia +: o ângulo do joelho abre (extensão).
- femur +: a pata desce (depressão do trocânter, que sustenta o corpo = extensão).
- tarsus +: o tarso sobe (levantamento).
- coxa +: a pata balança para a frente (promoção).
- coxa_twist +: a ponta da pata vai para a frente (rotação anterior).
- femur_twist +: a ponta da pata vai para trás (rotação posterior, a do músculo redutor).
- coxa_abduct +: abdução, pelo nome da junta no flybody (na postura de patinação o efeito
  geométrico muda de pata para pata).

Os tendões longos (ltm, que no inseto flexionam as garras) vão para a depressão do tarso,
porque o patim substitui garra e tarsos distais. Neurônios motores sem músculo identificado
(tipos como MNml78) ficam com peso livre, de qualquer sinal, só para as juntas da própria pata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mosca.body.fly import LEGS
from mosca.body.ik import LEG_JOINTS
from mosca.brain.graph import Connectome

EXTENSION, FLEXION = +1, -1

# tipo do neurônio motor → [(junta, sinal no flybody)]
MUSCLE_JOINTS: dict[str, list[tuple[str, int]]] = {
    # Coxa
    "Tergopleural/Pleural promotor MN": [("coxa", +1)],
    "Pleural remotor/abductor MN": [("coxa", -1), ("coxa_abduct", +1)],
    "Sternal anterior rotator MN": [("coxa_twist", +1)],
    "Sternal posterior rotator MN": [("coxa_twist", -1)],
    "Sternal adductor MN": [("coxa_abduct", -1)],
    # Trocânter e fêmur
    "Sternotrochanter MN": [("femur", EXTENSION)],
    "Tergotr. MN": [("femur", EXTENSION)],
    "Tr extensor MN": [("femur", EXTENSION)],
    "Tr flexor MN": [("femur", FLEXION)],
    "Acc. tr flexor MN": [("femur", FLEXION)],
    "Fe reductor MN": [("femur_twist", +1)],
    # Tíbia
    "Ti extensor MN": [("tibia", EXTENSION)],
    "Ti flexor MN": [("tibia", FLEXION)],
    "Acc. ti flexor MN": [("tibia", FLEXION)],
    # Tarso (+ = levantar)
    "Ta levator MN": [("tarsus", +1)],
    "Ta depressor MN": [("tarsus", -1)],
    "ltm MN": [("tarsus", -1)],
    "ltm1-tibia MN": [("tarsus", -1)],
    "ltm2-femur MN": [("tarsus", -1)],
}


@dataclass(frozen=True)
class LegDecoder:
    """Ligações neurônio motor → junta de uma pata.

    `motor[k]` é o índice do neurônio no grafo; `joint[k]` a junta (0 a 6, na ordem de
    LEG_JOINTS); `sign[k]` é +1 ou −1 para músculos identificados e 0 para peso livre.
    """

    leg: str
    motor: np.ndarray
    joint: np.ndarray
    sign: np.ndarray


def leg_decoders(c: Connectome) -> dict[str, LegDecoder]:
    out = {}
    for leg in LEGS:
        motor, joint, sign = [], [], []
        for i in c.groups[f"motor_{leg}"]:
            links = MUSCLE_JOINTS.get(str(c.cell_type[i]))
            if links is None:  # sem músculo identificado: peso livre para cada junta da pata
                links = [(j, 0) for j in LEG_JOINTS]
            for name, s in links:
                motor.append(int(i))
                joint.append(LEG_JOINTS.index(name))
                sign.append(s)
        out[leg] = LegDecoder(leg, np.array(motor), np.array(joint), np.array(sign, dtype=np.int8))
    return out
