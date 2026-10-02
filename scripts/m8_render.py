"""M8: renderiza uma tentativa re-simulada (saída do m8_trace.py) com o painel do cérebro ao lado.

Quadro de 1280×720:
- à esquerda, a mosca, na postura re-simulada a cada 0,2 ms (câmera lenta de verdade, sem interpolar), com régua de
  1 mm no plano da mosca, velocidade e pedido;
- à direita, os 23.117 neurônios do controlador, cada um no centro das suas sinapses no MaleCNS (vista de cima, a
  cabeça para cima), com brilho pela atividade re-simulada naquele passo de controle. Cores: motores das patas,
  sensoriais das patas, descendentes (DNg100 e DNa02 destacados) e o resto do cordão nervoso.

Sai um MP4 (ffmpeg) e, com `--gif`, também um GIF.

Uso:
    python scripts/m8_render.py runs/final_s0/video/tentativa_1.npz --slow 20 --out outputs/tentativa_1.mp4
    python scripts/m8_render.py runs/final_s0/video/tentativa_43905.npz --slow 4 --seconds 2 --gif outputs/t.gif
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.skates import build_skater  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402

W, H, FLY_W = 1280, 720, 800
PANEL_W = W - FLY_W
BG = np.array([14, 16, 22], np.float32)
RATE_REF = 60.0  # Hz para brilho cheio
COLORS = {  # (cor, nome na legenda)
    "other": ((90, 100, 125), "outros do cordão nervoso"),
    "sensory": ((80, 200, 255), "sensoriais das patas"),
    "motor": ((255, 170, 60), "motores das patas"),
    "dn": ((190, 120, 255), "descendentes do cérebro"),
    "DNg100": ((120, 255, 120), "DNg100 (andar)"),
    "DNa02": ((255, 90, 140), "DNa02 (virar)"),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else
                 ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class BrainPanel:
    """Nuvem de pontos dos neurônios, projetada de cima (x para a direita, z para baixo)."""

    def __init__(self, graph: Connectome, positions: Path):
        z = np.load(positions)
        assert (z["body_id"] == graph.body_id).all(), "posições de outro grafo"
        xyz = z["syn_centroid"].copy()
        miss = ~np.isfinite(xyz[:, 0])
        xyz[miss] = z["soma"][miss]
        xyz[~np.isfinite(xyz[:, 0])] = np.nanmedian(xyz, axis=0)
        u, v = xyz[:, 0], xyz[:, 2]
        top, bottom = 70, H - 110
        lo_u, hi_u, lo_v, hi_v = *np.percentile(u, [0.5, 99.5]), *np.percentile(v, [0.5, 99.5])
        scale = min((PANEL_W - 60) / (hi_u - lo_u), (bottom - top) / (hi_v - lo_v))
        self.px = np.clip(((u - (lo_u + hi_u) / 2) * scale + PANEL_W / 2).astype(int), 0, PANEL_W - 1)
        self.py = np.clip(((v - lo_v) * scale + top).astype(int), 0, H - 1)
        kind = np.full(graph.n, "other", dtype=object)
        kind[graph.superclass == "descending_neuron"] = "dn"
        for name, idx in graph.groups.items():
            if name.startswith("sensory_") or name.startswith("proprio_"):
                kind[idx] = "sensory"
            elif name.startswith("motor_"):
                kind[idx] = "motor"
        for cell in ("DNg100", "DNa02"):
            for side in ("L", "R"):
                kind[graph.groups.get(f"{cell}_{side}", [])] = cell
        self.color = np.array([COLORS[k][0] for k in kind], np.float32) / 255.0
        self.highlight = np.isin(kind, ["DNg100", "DNa02"])

    def render(self, rates: np.ndarray) -> np.ndarray:
        img = np.zeros((H, PANEL_W, 3), np.float32)
        level = np.clip(rates.astype(np.float32) / RATE_REF, 0.0, 1.0)
        np.add.at(img, (self.py, self.px), self.color * 0.10)
        glow = np.zeros_like(img)
        np.add.at(glow, (self.py, self.px), self.color * level[:, None])
        glow = gaussian_filter(glow, sigma=(1.2, 1.2, 0)) * 3.0 + glow
        big = self.highlight & (level > 0.05)
        for dy in (-2, -1, 0, 1, 2):  # descendentes destacados: ponto maior
            for dx in (-2, -1, 0, 1, 2):
                np.add.at(glow, (np.clip(self.py[big] + dy, 0, H - 1), np.clip(self.px[big] + dx, 0, PANEL_W - 1)),
                          self.color[big] * level[big, None] * 0.6)
        out = BG / 255.0 + img + glow
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def ruler_pixels(model: mujoco.MjModel, distance: float, height: int) -> float:
    """Pixels por cm no plano do alvo da câmera (perspectiva, campo vertical do modelo)."""
    fovy = np.deg2rad(model.vis.global_.fovy)
    return height / (2 * distance * np.tan(fovy / 2))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--out", default="")
    p.add_argument("--gif", default="", help="também grava um GIF (menor, 480 px de altura)")
    p.add_argument("--slow", type=float, default=20.0, help="câmera lenta")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=float, default=0.0, help="início do trecho (s simulados)")
    p.add_argument("--seconds", type=float, default=0.0, help="duração do trecho (s simulados; 0 = até o fim)")
    p.add_argument("--distance", type=float, default=0.9, help="distância da câmera (cm)")
    p.add_argument("--elevation", type=float, default=-30.0, help="graus; abaixo de −23 o horizonte sai do quadro")
    p.add_argument("--label", default="", help="rótulo extra (ex.: 'treino, com variação da população')")
    args = p.parse_args()

    z = np.load(args.trace)
    meta = json.loads(str(z["meta"]))
    qpos, rates = z["qpos"], z["rates"]
    control_dt, sub = meta["control_dt"], meta["substeps"]
    timestep = control_dt / sub
    graph = Connectome.load(MALECNS_DIR / meta.get("graph", "controller_graph_min5.npz"))
    panel = BrainPanel(graph, MALECNS_DIR / "controller_positions.npz")

    model = build_skater()
    model.vis.global_.offwidth, model.vis.global_.offheight = FLY_W, H
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=FLY_W)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, model.body("thorax").id
    cam.distance, cam.elevation, cam.azimuth = args.distance, args.elevation, 120
    px_per_cm = ruler_pixels(model, args.distance, H)

    total = len(qpos) * timestep
    end = total if args.seconds <= 0 else min(total, args.start + args.seconds)
    frame_dt = 1.0 / (args.fps * args.slow)
    times = np.arange(args.start, end, frame_dt)
    f_big, f_mid, f_small = font(34, True), font(20), font(16)
    out = args.out or str(Path("outputs") / f"tentativa_{meta['video_number']}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                          stdin=subprocess.PIPE)
    stats = meta["stats"]
    window = max(1, round(0.2 / timestep))  # velocidade pela média de 200 ms
    for t in times:
        k = min(len(qpos) - 1, int(round(t / timestep)))
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        fly = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(fly)
        k0 = max(0, k - window)
        speed = float(np.linalg.norm(qpos[k, :2] - qpos[k0, :2])) / max((k - k0) * timestep, 1e-9)
        d.text((24, 18), f"Tentativa #{meta['video_number']:,}".replace(",", "."), font=f_big, fill=(15, 15, 20))
        d.text((24, 62), f"geração {meta['generation']}" + (f" · {args.label}" if args.label else ""), font=f_mid,
               fill=(40, 40, 50))
        d.text((24, H - 70), f"t = {t:.3f} s   ·   câmera lenta {args.slow:g}×", font=f_mid, fill=(40, 40, 50))
        d.text((24, H - 42), f"{speed:.1f} cm/s (pedido: {meta['v_cmd']:.1f})", font=f_mid, fill=(40, 40, 50))
        bar = px_per_cm * 0.1  # 1 mm
        x1, y1 = FLY_W - 40, H - 40
        d.line([(x1 - bar, y1), (x1, y1)], fill=(15, 15, 20), width=4)
        d.text((x1 - bar, y1 - 28), "1 mm", font=f_small, fill=(15, 15, 20))
        step = min(len(rates) - 1, int(t / control_dt))
        brain = Image.fromarray(panel.render(rates[step]))
        b = ImageDraw.Draw(brain)
        b.text((20, 16), "cordão nervoso da mosca (MaleCNS)", font=f_mid, fill=(230, 230, 240))
        b.text((20, 42), f"{graph.n:,} neurônios · brilho = atividade".replace(",", "."), font=f_small,
               fill=(160, 165, 180))
        y = H - 100
        for key in ("motor", "sensory", "dn", "DNg100", "DNa02"):
            col, name = COLORS[key]
            b.ellipse([(20, y + 4), (30, y + 14)], fill=col)
            b.text((38, y - 1), name, font=f_small, fill=(200, 200, 215))
            y += 19
        frame = Image.new("RGB", (W, H))
        frame.paste(fly, (0, 0))
        frame.paste(brain, (FLY_W, 0))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"tentativa #{meta['video_number']} (geração {meta['generation']}): {len(times)} quadros, "
          f"{stats['seconds']:.2f} s simulados, {stats['speed']:.2f} cm/s, {'caiu' if stats['fell'] else 'não caiu'} "
          f"| física {meta['physics_max_deviation']:.3g}, cérebro {meta['brain_max_action_diff']:.3g} -> {out}")
    if args.gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                        "fps=15,scale=-1:480:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                        args.gif], check=True)
        print(f"GIF: {args.gif}")


if __name__ == "__main__":
    main()
