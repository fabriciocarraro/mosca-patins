"""Piloto do percurso de teste (M6): confere a passagem pelos cones e o sentido do giro pedido."""

import numpy as np

from mosca.env.course import Pilot, Slalom


def _drive(course, xs, ys):
    pilot = Pilot(course, 1)
    for x, y in zip(xs, ys):
        pilot.command(np.array([[x, y]]), np.array([0.0]))
    return pilot


def test_zigzag_through_the_cones_completes_the_course():
    c = Slalom()
    x = np.linspace(0.0, c.cone_x()[-1] + 1.0, 500)
    y = 0.4 * np.cos(np.pi * (x - c.lead) / c.spacing) * (x > c.lead - c.spacing / 2)
    pilot = _drive(c, x, y)
    assert pilot.passed[0] == c.cones and pilot.done()[0]


def test_straight_line_misses_the_cones():
    c = Slalom()
    x = np.linspace(0.0, c.cone_x()[-1] + 1.0, 500)
    pilot = _drive(c, x, np.full_like(x, -0.3))  # sempre à direita: erra o primeiro cone
    assert pilot.missed[0] and not pilot.done()[0]


def test_pilot_turns_toward_the_first_gate_and_saturates():
    c = Slalom()
    pilot = Pilot(c, 2)
    yaw = pilot.command(np.zeros((2, 2)), np.array([0.0, np.pi / 2]))
    assert yaw[0] > 0  # alvo à esquerda (y > 0) do rumo +x: giro anti-horário
    assert yaw[1] == -c.yaw_max  # olhando para +y, precisa virar à direita, além do limite
