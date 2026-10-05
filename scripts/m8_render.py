"""M8: renderiza uma tentativa re-simulada (saída do m8_trace.py) com o painel do cérebro ao lado.

Quadro de 1280×720:
- à esquerda, a mosca, na postura re-simulada a cada 0,2 ms (câmera lenta de verdade, sem interpolar), com régua de
  1 mm no plano da mosca, velocidade e pedido;
- à direita, os 23.117 neurônios do controlador, cada um no centro das suas sinapses no MaleCNS (vista de cima, a
  cabeça para cima), com brilho pela atividade re-simulada naquele passo de controle. Cores: motores das patas,
  sensoriais das patas, descendentes (DNg100 e DNa02 destacados) e o resto do cordão nervoso.

Com `--compare`, duas tentativas ou testes lado a lado (ex.: o mesmo teste com e sem o DNg100 calado), cada uma com
um painel pequeno do cérebro. Sai um MP4 (ffmpeg) e, com `--gif`, também um GIF.

Uso:
    python scripts/m8_render.py runs/final_s0/video/tentativa_1.npz --slow 20 --out outputs/tentativa_1.mp4
    python scripts/m8_render.py runs/final_s0/video/tentativa_43905.npz --slow 4 --seconds 2 --gif outputs/t.gif
    python scripts/m8_render.py runs/final_s0/video/teste1_v3.5_base_g2000.npz \
        --compare runs/final_s0/video/teste1_v3.5_DNg100_calado_g2000.npz --slow 4
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


LEG_NAMES = {"T1": "frente", "T2": "meio", "T3": "trás", "left": "E", "right": "D"}


def br(x: float, digits: int = 1) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{digits}f}".replace(".", ",")


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

    def render(self, rates: np.ndarray, rate_ref: float = RATE_REF) -> np.ndarray:
        img = np.zeros((H, PANEL_W, 3), np.float32)
        level = np.clip(rates.astype(np.float32) / rate_ref, 0.0, 1.0)
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


INTERVENTIONS = {"base": "sem intervenção", "DNg100_calado": "DNg100 calado", "DNa02_L": "DNa02 esquerdo estimulado",
                 "DNa02_R": "DNa02 direito estimulado"}


def ruler_pixels(model: mujoco.MjModel, distance: float, height: int) -> float:
    """Pixels por cm no plano do alvo da câmera (perspectiva, campo vertical do modelo)."""
    fovy = np.deg2rad(model.vis.global_.fovy)
    return height / (2 * distance * np.tan(fovy / 2))


class Trace:
    def __init__(self, path: str):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.qpos, self.rates = z["qpos"], z["rates"]
        self.control_dt = self.meta["control_dt"]
        self.timestep = self.control_dt / self.meta["substeps"]
        self.seconds = len(self.qpos) * self.timestep
        self.window = max(1, round(0.2 / self.timestep))  # velocidade pela média de 200 ms

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Postura, atividade e velocidade no instante t (a última, depois do fim da tentativa)."""
        k = min(len(self.qpos) - 1, int(round(t / self.timestep)))
        k0 = max(0, k - self.window)
        speed = float(np.linalg.norm(self.qpos[k, :2] - self.qpos[k0, :2])) / max((k - k0) * self.timestep, 1e-9)
        return self.qpos[k], self.rates[min(len(self.rates) - 1, int(t / self.control_dt))], speed

    def title(self) -> tuple[str, str]:
        m = self.meta
        if m.get("mode") == "teste":
            label = INTERVENTIONS.get(m["intervention"], m["intervention"].replace("_", " "))
            if m["intervention"] == "base" and getattr(self, "by_graph", False):
                name = "Conectoma embaralhado" if "shuffled" in m.get("graph", "") else "Conectoma real"
                return name, f"teste {m['test'] + 1} · geração {m['generation']} · sem variação"
            return f"Teste {m['test'] + 1} · {label}", f"geração {m['generation']} · teste, sem variação"
        return f"Tentativa #{m['video_number']:,}".replace(",", "."), f"geração {m['generation']}"

    def summary(self) -> str:
        m, s = self.meta, self.meta["stats"]
        check = (f"física {m['physics_max_deviation']:.3g}, cérebro {m['brain_max_action_diff']:.3g}"
                 if "physics_max_deviation" in m else "teste re-simulado do checkpoint")
        return (f"{self.title()[0]} (geração {m['generation']}): {s['seconds']:.2f} s simulados, {s['speed']:.2f} cm/s, "
                f"{'caiu' if s['fell'] else 'não caiu'} | {check}")


class FlyView:
    def __init__(self, width: int, distance: float, elevation: float, crop: int = 0):
        self.model = build_skater()
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = max(width, 640), H
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=H, width=width)
        self.crop = crop or width
        self.cam = mujoco.MjvCamera()
        self.cam.type, self.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, self.model.body("thorax").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, elevation, 120
        self.width = self.crop
        self.full_width = width
        self.px_per_cm = ruler_pixels(self.model, distance, H)

    def render(self, qpos: np.ndarray, title: str, subtitle: str, lines: list[str], fonts) -> Image.Image:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        img = Image.fromarray(self.renderer.render())
        if self.crop < self.full_width:
            left = (self.full_width - self.crop) // 2
            img = img.crop((left, 0, left + self.crop, H))
        d = ImageDraw.Draw(img)
        f_big, f_mid, f_small = fonts
        d.text((24, 18), title, font=f_big, fill=(15, 15, 20))
        d.text((24, 62), subtitle, font=f_mid, fill=(40, 40, 50))
        for i, line in enumerate(reversed(lines)):
            d.text((24, H - 42 - 28 * i), line, font=f_mid, fill=(40, 40, 50))
        bar = self.px_per_cm * 0.1  # 1 mm
        x1, y1 = self.width - 40, H - 40
        d.line([(x1 - bar, y1), (x1, y1)], fill=(15, 15, 20), width=4)
        d.text((x1 - bar, y1 - 28), "1 mm", font=f_small, fill=(15, 15, 20))
        return img


def brain_image(panel: BrainPanel, rates: np.ndarray, n: int, fonts, legend: bool = True) -> Image.Image:
    _, f_mid, f_small = fonts
    img = Image.fromarray(panel.render(rates))
    b = ImageDraw.Draw(img)
    b.text((20, 16), "cordão nervoso da mosca (MaleCNS)", font=f_mid, fill=(230, 230, 240))
    b.text((20, 42), f"{n:,} neurônios · brilho = atividade".replace(",", "."), font=f_small, fill=(160, 165, 180))
    if legend:
        y = H - 100
        for key in ("motor", "sensory", "dn", "DNg100", "DNa02"):
            col, name = COLORS[key]
            b.ellipse([(20, y + 4), (30, y + 14)], fill=col)
            b.text((38, y - 1), name, font=f_small, fill=(200, 200, 215))
            y += 19
    return img


def render_rhythm(path: str, args) -> None:
    """Cena do ritmo sem treino (m3_rhythm_full.py --save-trace): o painel do cérebro e, ao lado, a atividade dos
    neurônios motores mais ativos, com o cursor do tempo."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    rates = z["rates"].astype(np.float32)  # (T, N), a cada meta["control_dt"]
    dt = meta["control_dt"]
    graph = Connectome.load(MALECNS_DIR / meta["graph"])
    panel = BrainPanel(graph, MALECNS_DIR / "controller_positions.npz")
    motor = np.concatenate([graph.groups[k] for k in sorted(graph.groups) if k.startswith("motor_")])
    settled = rates[int(0.3 / dt):, motor]
    top = motor[np.argsort(settled.std(axis=0))[::-1][:6]]
    leg_of = {int(i): k.split("_", 1)[1] for k in graph.groups if k.startswith("motor_") for i in graph.groups[k]}
    peaks = rates.max(axis=0)
    rate_ref = args.rate_ref or max(0.5, float(np.median(peaks[peaks > 0.2])))
    f_big, f_mid, f_small = font(30, True), font(20), font(16)
    x0, x1, y0, row = PANEL_W + 60, W - 40, 230, 70
    t_end = len(rates) * dt if args.seconds <= 0 else min(len(rates) * dt, args.start + args.seconds)
    t_axis = np.arange(len(rates)) * dt
    out = args.out or str(Path("outputs") / (Path(path).stem + ".mp4"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                          stdin=subprocess.PIPE)
    times = np.arange(args.start, t_end, 1.0 / (args.fps * args.slow))
    for t in times:
        k = min(len(rates) - 1, int(t / dt))
        frame = Image.new("RGB", (W, H), tuple(int(c) for c in BG))
        frame.paste(Image.fromarray(panel.render(rates[k], rate_ref)), (0, 0))
        d = ImageDraw.Draw(frame)
        d.text((20, 16), "cordão nervoso da mosca (MaleCNS)", font=f_mid, fill=(230, 230, 240))
        d.text((20, 42), "sem nenhum treino · brilho relativo", font=f_small, fill=(160, 165, 180))
        stim = " e ".join(s.replace("_R", " direito").replace("_L", " esquerdo") for s in meta["stim"])
        d.text((x0, 40), f"Estímulo no {stim}", font=f_big, fill=(120, 255, 120))
        d.text((x0, 84), "(o neurônio de comando de andar)", font=f_mid, fill=(200, 200, 215))
        d.text((x0, 130), f"ritmo de {br(meta['freq_hz'])} Hz nos neurônios motores das patas", font=f_mid,
               fill=(255, 190, 110))
        d.text((x0, 160), "parâmetros publicados de Pugliese et al.; fiação do MaleCNS", font=f_small,
               fill=(160, 165, 180))
        for j, i in enumerate(top):  # traços dos motores mais ativos até o instante t
            yb = y0 + row * (j + 1)
            y = rates[: k + 1, i] / max(rates[:, i].max(), 1e-6) * (row - 12)
            xs = x0 + (t_axis[: k + 1] - args.start) / max(t_end - args.start, 1e-9) * (x1 - x0)
            keep = xs >= x0
            if keep.sum() > 1:
                d.line(list(zip(xs[keep], yb - y[keep])), fill=COLORS["motor"][0], width=2)
            d.line([(x0, yb), (x1, yb)], fill=(50, 55, 65), width=1)
            leg = leg_of.get(int(i), "_").split("_")
            d.text((x0 - 70, yb - 22), " ".join(LEG_NAMES.get(part, part) for part in leg),
                   font=f_small, fill=(160, 165, 180))
        d.text((x0, y0 + row * 7 + 10), f"t = {br(t, 3)} s · câmera lenta {args.slow:g}×", font=f_mid, fill=(200, 200, 215))
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"ritmo ({stim}, {meta['freq_hz']:.2f} Hz, brilho cheio a {rate_ref:.1f} Hz): {len(times)} quadros -> {out}")
    if args.gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                        "fps=15,scale=-1:480:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                        args.gif], check=True)
        print(f"GIF: {args.gif}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--compare", default="",
                   help="segunda tentativa ou teste, lado a lado (ex.: o mesmo teste com intervenção)")
    p.add_argument("--out", default="")
    p.add_argument("--gif", default="", help="também grava um GIF (menor, 480 px de altura)")
    p.add_argument("--slow", type=float, default=20.0, help="câmera lenta")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--start", type=float, default=0.0, help="início do trecho (s simulados)")
    p.add_argument("--seconds", type=float, default=0.0, help="duração do trecho (s simulados; 0 = até o fim)")
    p.add_argument("--distance", type=float, default=0.9, help="distância da câmera (cm)")
    p.add_argument("--elevation", type=float, default=-30.0, help="graus; abaixo de −23 o horizonte sai do quadro")
    p.add_argument("--label", default="", help="rótulo extra (ex.: 'treino, com variação da população')")
    p.add_argument("--rate-ref", type=float, default=0.0,
                   help="taxa (Hz) de brilho cheio; padrão 60, e na cena do ritmo o percentil 90 dos picos")
    args = p.parse_args()
    if json.loads(str(np.load(args.trace)["meta"])).get("mode") == "ritmo":
        render_rhythm(args.trace, args)
        return

    traces = [Trace(args.trace)] + ([Trace(args.compare)] if args.compare else [])
    if len({tr.meta.get("graph") for tr in traces}) > 1:  # real × embaralhado: o título diz qual é qual
        for tr in traces:
            tr.by_graph = True
    graph = Connectome.load(MALECNS_DIR / traces[0].meta.get("graph", "controller_graph_min5.npz"))
    panel = BrainPanel(graph, MALECNS_DIR / "controller_positions.npz")
    width = FLY_W if len(traces) == 1 else W // 2
    strip = 200  # lado a lado: faixa do cérebro (o centro do painel, onde fica o cordão nervoso)
    view = FlyView(width, args.distance, args.elevation) if len(traces) == 1 else \
        FlyView(FLY_W, args.distance, args.elevation, crop=width - strip)
    fonts = font(34, True), font(20), font(16)

    total = max(tr.seconds for tr in traces)
    end = total if args.seconds <= 0 else min(total, args.start + args.seconds)
    times = np.arange(args.start, end, 1.0 / (args.fps * args.slow))
    out = args.out or str(Path("outputs") / (Path(args.trace).stem + ".mp4"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                          stdin=subprocess.PIPE)
    for t in times:
        frame = Image.new("RGB", (W, H))
        for col, tr in enumerate(traces):
            qpos, rates, speed = tr.at(t)
            title, subtitle = tr.title()
            if args.label:
                subtitle += f" · {args.label}"
            lines = [f"t = {br(min(t, tr.seconds), 3)} s   ·   câmera lenta {args.slow:g}×",
                     f"{br(speed)} cm/s (pedido: {br(tr.meta['v_cmd'])})"]
            fly = view.render(qpos, title, subtitle, lines, fonts)
            frame.paste(fly, (col * width, 0))
            if len(traces) == 1:
                frame.paste(brain_image(panel, rates, graph.n, fonts), (FLY_W, 0))
            else:  # painel pequeno no canto de cada lado
                left = (PANEL_W - strip) // 2
                band = Image.fromarray(panel.render(rates)).crop((left, 0, left + strip, H))
                ImageDraw.Draw(band).text((14, 16), "cérebro", font=fonts[1], fill=(230, 230, 240))
                frame.paste(band, (col * width + width - strip, 0))
        if len(traces) == 2:
            ImageDraw.Draw(frame).line([(width, 0), (width, H)], fill=(20, 20, 25), width=3)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    for tr in traces:
        print(tr.summary())
    print(f"{len(times)} quadros -> {out}")
    if args.gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                        "fps=15,scale=-1:480:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",
                        args.gif], check=True)
        print(f"GIF: {args.gif}")


if __name__ == "__main__":
    main()
