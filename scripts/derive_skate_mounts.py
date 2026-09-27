"""Deriva a montagem dos patins a partir de moscas reais.

Para cada pata, nos quadros em que a garra está apoiada (a até 40 µm da garra mais baixa),
em todos os 100 trechos da amostra de caminhada, mede no referencial do tarso:
- a direção "para baixo" (SKATE_DOWN): o patim fica paralelo ao chão dessa postura;
- a direção da frente do corpo, projetada no chão (SKATE_FORWARD): o patim aponta para a
  frente da mosca nessa postura média de apoio, e não ao longo do pé (nas patas do meio e
  de trás o pé aponta para o lado, e as juntas não conseguem girar o patim até a frente).
As médias viram constantes em src/mosca/body/skates.py.

Também imprime a postura média de apoio (altura e inclinação do tórax, posição da ponta de
cada garra no referencial do tórax, ângulos médios das juntas de cada pata), que vira a
postura canônica "parada de patins" em src/mosca/body/stance.py.

Uso:
    python scripts/derive_skate_mounts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS, build_fly  # noqa: E402
from mosca.body.poses import WalkingFrame  # noqa: E402
from mosca.body.poses import apply_frame  # noqa: E402
from mosca.paths import WALKING_SAMPLE  # noqa: E402

STANCE_TOL = 0.004  # cm


def main() -> None:
    model = build_fly()
    data = mujoco.MjData(model)
    claws = {leg: model.body(f"claw_{leg}").id for leg in LEGS}
    tarsi = {leg: model.body(f"tarsus_{leg}").id for leg in LEGS}
    thorax = model.body("thorax").id
    down = {leg: [] for leg in LEGS}
    forward = {leg: [] for leg in LEGS}
    tips = {leg: [] for leg in LEGS}
    joints = {leg: [] for leg in LEGS}
    heights, pitches, rolls = [], [], []
    leg_qadr = {
        leg: [model.jnt_qposadr[model.joint(f"{j}_{leg}").id]
              for j in ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus")]
        for leg in LEGS
    }
    claw_tip_local = {leg: model.geom(f"tarsal_claw_{leg}_collision").id for leg in LEGS}

    with h5py.File(WALKING_SAMPLE, "r") as f:
        names = tuple(n.decode() for n in f["id2name/joints"][()])
        for key in sorted(f["trajectories"].keys()):
            traj = f["trajectories"][key]
            qpos, root = traj["qpos"][()], traj["root_qpos"][()]
            for i in range(0, len(qpos), 10):
                apply_frame(model, data, WalkingFrame(names, qpos[i].astype(float), root[i].astype(float)), center=False)
                mujoco.mj_kinematics(model, data)
                ground = min(data.xpos[b][2] for b in claws.values())
                thorax_rot = data.xmat[thorax].reshape(3, 3)
                heading = thorax_rot[:, 0].copy()
                heading[2] = 0.0
                heading /= np.linalg.norm(heading)
                heights.append(data.xpos[thorax][2] - ground)
                pitches.append(np.arcsin(np.clip(thorax_rot[2, 0], -1, 1)))
                rolls.append(np.arcsin(np.clip(thorax_rot[2, 1], -1, 1)))
                for leg in LEGS:
                    if data.xpos[claws[leg]][2] - ground < STANCE_TOL:
                        rot = data.xmat[tarsi[leg]].reshape(3, 3)
                        down[leg].append(rot.T @ np.array([0.0, 0.0, -1.0]))
                        forward[leg].append(rot.T @ heading)
                        tip = data.geom_xpos[claw_tip_local[leg]]
                        tips[leg].append(thorax_rot.T @ (tip - data.xpos[thorax]))
                        joints[leg].append(data.qpos[leg_qadr[leg]].copy())

    for name, vectors in (("SKATE_DOWN", down), ("SKATE_FORWARD", forward)):
        print(f"{name} = {{")
        for leg in LEGS:
            v = np.mean(vectors[leg], axis=0)
            spread = np.linalg.norm(v)
            v = v / spread
            print(f'    "{leg}": ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}),  # {len(vectors[leg])} quadros, |média| {spread:.2f}')
        print("}")

    print(f"\nSTANCE_THORAX_HEIGHT = {np.median(heights):.4f}  # cm acima da garra mais baixa (mediana)")
    print(f"STANCE_THORAX_PITCH = {np.median(pitches):.4f}  # rad (nariz para cima > 0? ver sinal)")
    print(f"# rolagem mediana {np.degrees(np.median(rolls)):+.2f} graus (ignorada: postura simétrica)")
    print("STANCE_CLAW_TIP = {  # centro da garra, referencial do tórax (cm), mediana em apoio")
    for leg in LEGS:
        v = np.median(tips[leg], axis=0)
        print(f'    "{leg}": ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}),')
    print("}")
    print("STANCE_JOINTS = {  # coxa_abduct, coxa_twist, coxa, femur_twist, femur, tibia, tarsus (mediana em apoio)")
    for leg in LEGS:
        v = np.median(joints[leg], axis=0)
        print(f'    "{leg}": ({", ".join(f"{x:.3f}" for x in v)}),')
    print("}")


if __name__ == "__main__":
    main()
