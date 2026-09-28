"""Percurso de teste do M6: reta e slalom entre cones, com um piloto externo que dá o giro pedido.

Os cones ficam em linha no eixo x, depois de uma reta de `lead` cm; a mosca deve passar cada cone
por um lado, alternando (o primeiro pela esquerda, y > 0). Quem patina é a política; o piloto só
faz o papel do controle remoto: aponta para um alvo ao lado do próximo cone (perseguição pura) e
pede um giro proporcional ao erro de rumo, cortado em ±`yaw_max`. É o mesmo comando de giro do
treino (que no conectoma entra no DNa02); a velocidade pedida é constante.

Os cones não existem na física (não há contato): são posições, conferidas quando a mosca cruza a
linha de cada cone, e desenhadas só na renderização (`draw_cones`).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Slalom:
    lead: float = 2.0  # cm de reta até o primeiro cone
    spacing: float = 3.0  # cm entre cones
    cones: int = 4
    offset: float = 0.4  # cm: o alvo fica a esse tanto do cone, do lado da passagem
    gain: float = 3.0  # rad/s de giro pedido por rad de erro de rumo
    yaw_max: float = 1.5  # rad/s

    def cone_x(self) -> np.ndarray:
        return self.lead + self.spacing * np.arange(self.cones)

    def side(self, k: int) -> float:
        return 1.0 if k % 2 == 0 else -1.0  # +1: passa com o cone à direita (mosca em y > 0)


class Pilot:
    """Um piloto por tentativa: dá o giro pedido a cada passo e confere a passagem pelos cones."""

    def __init__(self, course: Slalom, n: int):
        self.course, self.n = course, n
        self.next = np.zeros(n, dtype=int)  # próximo cone de cada tentativa
        self.passed = np.zeros(n, dtype=int)
        self.missed = np.zeros(n, dtype=bool)

    def command(self, xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """Giro pedido (rad/s) a partir da posição (n, 2) e do rumo (n,) do tórax."""
        c = self.course
        xs = c.cone_x()
        yaw = np.zeros(self.n)
        for i in range(self.n):
            k = self.next[i]
            while k < c.cones and xy[i, 0] >= xs[k]:  # cruzou a linha do cone k: confere o lado
                if np.sign(xy[i, 1]) == c.side(k):
                    self.passed[i] += 1
                else:
                    self.missed[i] = True
                k += 1
            self.next[i] = k
            target = (xs[k], c.side(k) * c.offset) if k < c.cones else (xs[-1] + c.spacing, 0.0)
            want = np.arctan2(target[1] - xy[i, 1], target[0] - xy[i, 0])
            err = (want - heading[i] + np.pi) % (2 * np.pi) - np.pi
            yaw[i] = np.clip(c.gain * err, -c.yaw_max, c.yaw_max)
        return yaw

    def done(self) -> np.ndarray:
        """Percurso completo: todos os cones pelo lado certo."""
        return (self.passed == self.course.cones) & ~self.missed


def thorax_pose(env) -> tuple[np.ndarray, np.ndarray]:
    """Posição (n, 2) em cm e rumo (n,) em rad do tórax de cada ambiente."""
    xy = np.array([d.xpos[env.thorax, :2] for d in env.datas])
    heading = np.array([np.arctan2(d.xmat[env.thorax, 3], d.xmat[env.thorax, 0]) for d in env.datas])
    return xy, heading


def draw_cones(scene: mujoco.MjvScene, course: Slalom, radius: float = 0.08, height: float = 0.25) -> None:
    """Acrescenta os cones à cena já montada (só na imagem: o modelo e a física não mudam)."""
    for k, x in enumerate(course.cone_x()):
        if scene.ngeom >= scene.maxgeom:
            return
        rgba = np.array([1.0, 0.45, 0.0, 1.0] if k % 2 == 0 else [1.0, 0.8, 0.0, 1.0], dtype=np.float32)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([radius, height / 2, 0.0]), np.array([x, 0.0, height / 2]),
                            np.eye(3).ravel(), rgba)
        scene.ngeom += 1
