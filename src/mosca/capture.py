"""Captura fiel das tentativas de treino (M5) e re-simulação bit a bit.

Por iteração do treino grava-se, para cada tentativa da leva:
- o número da tentativa ("Tentativa #N"), de onde saem a semente e o estado inicial;
- a velocidade pedida e o empurrão inicial (ajuda do currículo);
- as ações aplicadas a cada passo de controle, em float32 exato (média + ruído, antes do
  corte que o ambiente faz), e quantos passos a tentativa durou;
- o estado da física (qpos, qvel) a cada `checkpoint_every` passos, para conferir que a
  re-simulação não divergiu.

A física do MuJoCo é determinística: re-simular uma tentativa com as mesmas ações, sozinha
num ambiente de lote 1, reproduz os estados gravados bit a bit (teste em
tests/test_capture.py). A versão da política de cada iteração fica em outro arquivo, para
re-simular a atividade do cérebro no painel do vídeo.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from mosca.env.skate_env import EnvConfig, SkateVecEnv


def library_versions() -> dict[str, str]:
    """Versões que entram no manifesto da execução (a física depende da versão do MuJoCo)."""
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for name in ("mujoco", "numpy", "torch", "scipy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


@dataclass
class IterationCapture:
    iteration: int
    attempts: np.ndarray  # (n,) números das tentativas
    v_cmd: np.ndarray  # (n,)
    push: np.ndarray  # (n,)
    steps: np.ndarray  # (n,) passos executados por tentativa
    actions: np.ndarray  # (T, n, A) float32
    checkpoint_every: int
    qpos: np.ndarray  # (K, n, nq) estado da física a cada checkpoint_every passos (antes do passo)
    qvel: np.ndarray  # (K, n, nv)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, iteration=self.iteration, attempts=self.attempts, v_cmd=self.v_cmd, push=self.push,
                            steps=self.steps, actions=self.actions, checkpoint_every=self.checkpoint_every,
                            qpos=self.qpos, qvel=self.qvel)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "IterationCapture":
        z = np.load(path, allow_pickle=False)
        return IterationCapture(int(z["iteration"]), z["attempts"], z["v_cmd"], z["push"], z["steps"], z["actions"],
                                int(z["checkpoint_every"]), z["qpos"], z["qvel"])


class Recorder:
    """Acompanha uma leva de tentativas durante a coleta."""

    def __init__(self, env: SkateVecEnv, iteration: int, attempts, v_cmd, push, checkpoint_every: int = 50):
        self.env, self.iteration, self.every = env, iteration, checkpoint_every
        self.attempts, self.v_cmd, self.push = np.asarray(attempts), np.asarray(v_cmd, float), np.asarray(push, float)
        self.actions = np.zeros((env.max_steps, env.n, env.act_dim), np.float32)
        self.qpos, self.qvel = [], []
        self.t = 0

    def before_step(self, actions: np.ndarray) -> None:
        """Chame com as ações que vão para env.step, antes de chamá-lo."""
        if self.t % self.every == 0:
            self.qpos.append(np.stack([d.qpos.copy() for d in self.env.datas]))
            self.qvel.append(np.stack([d.qvel.copy() for d in self.env.datas]))
        self.actions[self.t] = actions
        self.t += 1

    def finish(self) -> IterationCapture:
        return IterationCapture(self.iteration, self.attempts, self.v_cmd, self.push, self.env.steps.copy(),
                                self.actions[: self.t], self.every, np.array(self.qpos), np.array(self.qvel))


def replay(cap: IterationCapture, index: int, env_cfg: EnvConfig, on_step=None, on_substep=None) -> dict:
    """Re-simula a tentativa `index` da leva num ambiente de lote 1 e confere os estados gravados.

    `on_step(env, t)` é chamado depois de cada passo de controle e `on_substep(env, t, sub)`
    depois de cada subpasso de física (para renderizar em câmera lenta). Devolve as
    estatísticas da tentativa e o maior desvio encontrado nos estados gravados.
    """
    cfg = EnvConfig(**{**env_cfg.__dict__, "n_envs": 1, "n_threads": 1})
    env = SkateVecEnv(cfg)
    env.reset(cap.attempts[index : index + 1], cap.v_cmd[index : index + 1], cap.push[index : index + 1])
    step = {"t": 0}
    if on_substep is not None:
        env.substep_callback = lambda e, sub: on_substep(e, step["t"], sub)
    d = env.datas[0]
    worst = 0.0
    for t in range(int(cap.steps[index])):
        if t % cap.checkpoint_every == 0:
            k = t // cap.checkpoint_every
            worst = max(worst, float(np.abs(d.qpos - cap.qpos[k, index]).max()), float(np.abs(d.qvel - cap.qvel[k, index]).max()))
        step["t"] = t
        env.step(cap.actions[t, index][None])
        if on_step is not None:
            on_step(env, t)
    stats = env.episode_stats()[0]
    env.close()
    return {**stats, "max_deviation": worst}
