"""M8: cartela do cérebro para o vídeo: o painel do cordão nervoso (cada neurônio no centro das suas sinapses no
MaleCNS, com o brilho de um instante de uma tentativa re-simulada) com rótulos (patas da frente, do meio e de trás,
DNg100 e DNa02) e, ao lado, o que é fixo e o que o treino ajusta. Sai um PNG de 1280×720.

Uso:
    python scripts/m8_brain_card.py runs/evolui_u/video/teste2_v3.5_base_g1400.npz --at 2.0 --out outputs/cerebro.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m8_render import BG, COLORS, H, PANEL_W, W, BrainPanel, font  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402

LEGS_PT = {"T1": "patas da frente", "T2": "patas do meio", "T3": "patas de trás"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="traço do m8_trace.py (dá o brilho de um instante)")
    p.add_argument("--at", type=float, default=2.0, help="instante do brilho (s simulados)")
    p.add_argument("--out", default="outputs/cerebro.png")
    args = p.parse_args()

    z = np.load(args.trace)
    meta = json.loads(str(z["meta"]))
    graph = Connectome.load(MALECNS_DIR / meta.get("graph", "controller_graph_min5.npz"))
    panel = BrainPanel(graph, MALECNS_DIR / "controller_positions.npz")
    rates = z["rates"][min(len(z["rates"]) - 1, int(args.at / meta["control_dt"]))]
    img = Image.new("RGB", (W, H), tuple(int(c) for c in BG))
    left = W - PANEL_W - 110  # o painel à direita; o cordão nervoso ocupa só o meio dele, e os rótulos ficam ao lado
    img.paste(Image.fromarray(panel.render(rates)), (left, 0))
    d = ImageDraw.Draw(img)
    f_title, f_text, f_lab, f_small = font(40, True), font(25), font(21, True), font(17)

    def label(idx: np.ndarray, text: str, color, side: int = -1, dy: int = 0) -> None:
        x, y = float(np.median(panel.px[idx])) + left, float(np.median(panel.py[idx])) + dy
        mid = left + PANEL_W / 2
        tx = mid - 115 if side < 0 else mid + 115
        d.line([(x, y), (tx + (8 if side < 0 else -8), y)], fill=color, width=2)
        d.text((tx, y), text, font=f_lab, fill=color, anchor="rm" if side < 0 else "lm")

    for seg, name in LEGS_PT.items():
        idx = np.concatenate([graph.groups[f"motor_{seg}_left"], graph.groups[f"motor_{seg}_right"]])
        label(idx, name, COLORS["motor"][0])
    label(np.concatenate([graph.groups["DNg100_L"], graph.groups["DNg100_R"]]), "DNg100: “ande”", COLORS["DNg100"][0], +1)
    label(np.concatenate([graph.groups["DNa02_L"], graph.groups["DNa02_R"]]), "DNa02: “vire”", COLORS["DNa02"][0], +1, 26)

    x0 = 50
    d.text((x0, 60), "O cordão nervoso de", font=f_title, fill=(235, 235, 245))
    d.text((x0, 108), "uma mosca de verdade", font=f_title, fill=(235, 235, 245))
    lines = [
        ("a “medula espinhal” dela", (150, 155, 170)),
        ("", None),
        (f"{graph.n:,}".replace(",", ".") + " neurônios, mapeados um a um", (215, 220, 232)),
        ("", None),
        ("Fixo: quem se liga com quem", (120, 255, 120)),
        ("Ajustado pelo treino: a força das ligações", (255, 200, 90)),
    ]
    y = 168
    for text, color in lines:
        if text:
            d.text((x0, y), text, font=f_text, fill=color)
        y += 40
    d.text((x0, H - 60), "dados: MaleCNS (Janelia) · brilho = atividade num instante de um teste re-simulado", font=f_small,
           fill=(130, 135, 150))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"cartela do cérebro -> {args.out}")


if __name__ == "__main__":
    main()
