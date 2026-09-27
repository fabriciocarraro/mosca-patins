"""Modelo de contato da lâmina, isolado num trenó de 1 mg (o peso da mosca) com dois patins.

Cada patim tem as mesmas quatro lâminas curtas do modelo da mosca (runner_segments), com
atrito anisotrópico num par de contato explícito com o chão. O centro de massa do trenó
fica 100 µm acima das lâminas, para testar também o efeito de empinar ao frear.
"""

import mujoco
import numpy as np
import pytest

from mosca.body.skates import SkateConfig, runner_segments

CFG = SkateConfig()
SLED_MASS = 1e-3  # g


def _sled(yaw_deg: float = 0.0) -> tuple[mujoco.MjModel, mujoco.MjData]:
    runners, pairs = [], []
    fric = f"{CFG.friction_along} {CFG.friction_across} 0.005 0.0001 0.0001"
    solimp = " ".join(map(str, CFG.solimp))
    segments = runner_segments(CFG)
    for skate_y in (0.02, -0.02):
        for tag, fromto in segments:
            name = f"r_{tag}_{'l' if skate_y > 0 else 'r'}"
            shifted = fromto + np.array([0, skate_y, 0, 0, skate_y, 0])
            runners.append(
                f'<geom name="{name}" type="capsule" size="{CFG.runner_radius}" '
                f'fromto="{" ".join(map(str, shifted))}" mass="{CFG.mass / len(segments)}" contype="0" conaffinity="0"/>'
            )
            pairs.append(
                f'<pair geom1="floor" geom2="{name}" condim="3" friction="{fric}" '
                f'solref="{CFG.solref[0]} {CFG.solref[1]}" solimp="{solimp}"/>'
            )
    a = np.radians(yaw_deg) / 2
    xml = f"""
    <mujoco>
      <option timestep="0.0002" gravity="0 0 -981" cone="elliptic" impratio="{CFG.impratio}" noslip_iterations="3"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="sled" pos="0 0 0.0001" quat="{np.cos(a)} 0 0 {np.sin(a)}">
          <freejoint/>
          <geom type="box" size="0.02 0.03 0.002" pos="0 0 0.01" mass="{SLED_MASS}" contype="0" conaffinity="0"/>
          {''.join(runners)}
        </body>
      </worldbody>
      <contact>{''.join(pairs)}</contact>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(1000):  # assenta no chão
        mujoco.mj_step(model, data)
    return model, data


def _weight(model):
    return model.body_subtreemass[1] * 981.0


@pytest.mark.parametrize("yaw", [0.0, 30.0])
def test_slides_only_along_the_blade(yaw):
    # Empurrão de 5% do peso em x do mundo por 0,2 s: termina a 6–8 cm/s, a faixa da mosca.
    model, data = _sled(yaw)
    for _ in range(1000):
        data.xfrc_applied[1, :3] = [0.05 * _weight(model), 0.0, 0.0]
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.qvel[:2]) > 3.0
    heading = np.degrees(np.arctan2(data.qvel[1], data.qvel[0]))
    assert heading == pytest.approx(yaw, abs=1.0)


def test_lateral_grip_holds_at_half_the_friction_limit():
    model, data = _sled()
    for _ in range(2500):
        data.xfrc_applied[1, :3] = [0.0, 0.5 * CFG.friction_across * _weight(model), 0.0]
        mujoco.mj_step(model, data)
    assert abs(data.qvel[1]) < 1e-3  # cm/s


def test_glide_deceleration_is_friction_times_g():
    model, data = _sled()
    data.qvel[0] = 3.0
    v0, n = data.qvel[0], 500
    for _ in range(n):
        mujoco.mj_step(model, data)
    decel = (v0 - data.qvel[0]) / (n * model.opt.timestep)
    assert decel == pytest.approx(CFG.friction_along * 981.0, rel=0.02)


def test_runners_barely_sink():
    model, data = _sled()
    penetration = max(-data.contact[i].dist for i in range(data.ncon))
    assert penetration < 0.05 * CFG.runner_radius
