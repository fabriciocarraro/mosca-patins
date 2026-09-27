"""Tabela músculo → junta: sentido das juntas no flybody e cobertura dos neurônios motores."""

import mujoco
import numpy as np
import pytest

from mosca.body.fly import LEGS
from mosca.body.ik import LEG_JOINTS, LegIK
from mosca.body.skates import build_skater
from mosca.body.stance import skating_stance
from mosca.brain.muscles import MUSCLE_JOINTS, leg_decoders
from mosca.paths import MALECNS_DIR

GRAPH = MALECNS_DIR / "controller_graph_min5.npz"


@pytest.fixture(scope="module")
def posed():
    m = build_skater()
    d = mujoco.MjData(m)
    d.qpos[:] = skating_stance(m, LegIK(m)).qpos
    d.qpos[:7] = [0, 0, 0.14, 1, 0, 0, 0]  # tórax nivelado, olhando para +x
    mujoco.mj_kinematics(m, d)
    return m, d


def _effect(m, d, leg, joint, delta=0.05):
    """Posições (fêmur, tíbia, tarso, patim) antes e depois de girar a junta em +delta."""
    names = ("femur", "tibia", "tarsus", "skate")
    before = {k: d.xpos[m.body(f"{k}_{leg}").id].copy() for k in names}
    adr = m.jnt_qposadr[m.joint(f"{joint}_{leg}").id]
    d.qpos[adr] += delta
    mujoco.mj_kinematics(m, d)
    after = {k: d.xpos[m.body(f"{k}_{leg}").id].copy() for k in names}
    d.qpos[adr] -= delta
    mujoco.mj_kinematics(m, d)
    return before, after


def _knee(p):
    u, v = p["femur"] - p["tibia"], p["tarsus"] - p["tibia"]
    return np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))


@pytest.mark.parametrize("leg", LEGS)
def test_joint_directions_match_the_table(posed, leg):
    m, d = posed
    b, a = _effect(m, d, leg, "tibia")
    assert _knee(a) > _knee(b)  # tibia +: extensão
    b, a = _effect(m, d, leg, "femur")
    assert a["tarsus"][2] < b["tarsus"][2]  # femur +: a pata desce (depressão do trocânter)
    b, a = _effect(m, d, leg, "tarsus")
    assert (a["skate"] - a["tarsus"])[2] > (b["skate"] - b["tarsus"])[2]  # tarsus +: levantamento
    for joint, forward in (("coxa", True), ("coxa_twist", True), ("femur_twist", False)):
        b, a = _effect(m, d, leg, joint)
        assert (a["tarsus"][0] > b["tarsus"][0]) == forward, joint


def test_antagonists_have_opposite_signs():
    def sign(muscle, joint):
        return dict(MUSCLE_JOINTS[muscle])[joint]

    assert sign("Ti extensor MN", "tibia") == -sign("Ti flexor MN", "tibia")
    assert sign("Tr extensor MN", "femur") == -sign("Tr flexor MN", "femur")
    assert sign("Ta levator MN", "tarsus") == -sign("Ta depressor MN", "tarsus")
    assert sign("Tergopleural/Pleural promotor MN", "coxa") == -sign("Pleural remotor/abductor MN", "coxa")
    assert sign("Sternal anterior rotator MN", "coxa_twist") == -sign("Sternal posterior rotator MN", "coxa_twist")
    assert sign("Sternal adductor MN", "coxa_abduct") == -sign("Pleural remotor/abductor MN", "coxa_abduct")


@pytest.mark.skipif(not GRAPH.exists(), reason="grafo não gerado (rode scripts/m3_build_graph.py)")
def test_every_leg_motor_neuron_drives_only_its_own_leg():
    from mosca.brain.graph import Connectome

    c = Connectome.load(GRAPH)
    decoders = leg_decoders(c)
    for leg, dec in decoders.items():
        assert set(dec.motor.tolist()) == set(c.groups[f"motor_{leg}"].tolist())
        assert set(dec.joint.tolist()) == set(range(len(LEG_JOINTS)))  # toda junta recebe algum neurônio
        named = np.array([str(c.cell_type[i]) in MUSCLE_JOINTS for i in dec.motor])
        assert (dec.sign[named] != 0).all() and (dec.sign[~named] == 0).all()
    t1 = decoders["T1_left"]
    assert (t1.sign != 0).all()  # na pata da frente todo neurônio motor tem músculo identificado
