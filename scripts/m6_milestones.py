"""M6: encontra, no registro de tentativas de um treino, as que vão para o vídeo.

Critérios do plano (docs/plano.md, captura fiel), sempre sobre o registro completo, sem
escolher a dedo:
- números pré-registrados: #1, #10, #100, #1.000, #10.000...;
- marcos automáticos: primeira tentativa acima de cada velocidade média, primeiro deslize
  (≥25% do tempo deslizando a ≥1 cm/s), cada novo recorde de distância e a primeira queda (este último
  acrescentado em 02/10/2026, depois de ver o registro do treino final);
- uma amostra aleatória fixa de 5% (semente registrada).

Tentativas com empurrão inicial (ajuda do currículo) não contam para marcos e recordes e
aparecem marcadas. A numeração do vídeo começa em #1 (a tentativa 0 do registro é a #1).

Uso:
    python scripts/m6_milestones.py --run conectoma_d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.paths import RUNS  # noqa: E402

SPEEDS = (0.5, 1.0, 2.0, 3.0, 4.0)
SAMPLE_SEED = 20260927


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sample", type=float, default=0.05)
    args = parser.parse_args()

    run_dir = RUNS / args.run
    attempts = [json.loads(line) for line in open(run_dir / "attempts.jsonl", encoding="utf-8")]
    attempts.sort(key=lambda a: a["attempt"])
    n = len(attempts)
    picked: dict[int, list[str]] = {}

    def pick(a, why):
        picked.setdefault(a["attempt"], []).append(why)

    k = 1
    while k <= n:  # pré-registrados (numeração do vídeo a partir de 1)
        pick(attempts[k - 1], f"pré-registrada #{k}")
        k *= 10

    unaided = [a for a in attempts if a["push"] == 0.0]
    for v in SPEEDS:
        first = next((a for a in unaided if not a["fell"] and a["speed"] >= v), None)
        if first:
            pick(first, f"primeira a ≥{v:g} cm/s sem cair")
    glide = next((a for a in unaided if a["glide_frac"] >= 0.25 and a["speed"] >= 1.0), None)
    if glide:
        pick(glide, "primeiro deslize (≥25% do tempo a ≥1 cm/s)")
    fall = next((a for a in unaided if a["fell"]), None)
    if fall:  # marco acrescentado em 02/10/2026, depois de ver o registro do final_s0 (o vídeo diz isso)
        pick(fall, "primeira queda")
    record = 0.0
    for a in unaided:
        if a["distance"] > record + 0.05:  # recorde de distância (margem de 0,5 mm)
            record = a["distance"]
            pick(a, f"recorde de distância: {record:.2f} cm")

    rng = np.random.default_rng(SAMPLE_SEED)
    for i in np.flatnonzero(rng.random(n) < args.sample):
        pick(attempts[i], "amostra aleatória")

    rows = []
    for number in sorted(picked):
        a = attempts[number]
        rows.append({"video_number": number + 1, "attempt": number, "iteration": a.get("it", a.get("gen")),
                     "reasons": picked[number],
                     "v_cmd": a.get("v_cmd"), "push": a["push"], "fell": a["fell"], "seconds": a["seconds"], "speed": a["speed"],
                     "distance": a["distance"], "glide_frac": a["glide_frac"]})
    (run_dir / "milestones.json").write_text(json.dumps({"sample_seed": SAMPLE_SEED, "attempts": n, "picked": rows},
                                                        indent=2, ensure_ascii=False), encoding="utf-8")
    marks = [r for r in rows if r["reasons"] != ["amostra aleatória"]]
    print(f"{n} tentativas; {len(rows)} escolhidas ({len(marks)} por número ou marco, o resto da amostra de "
          f"{args.sample:.0%}) -> {run_dir / 'milestones.json'}")
    for r in marks:
        aid = " (com empurrão)" if r["push"] else ""
        asked = "" if r["v_cmd"] is None else \
            (" (pedido: ficar parada)" if r["v_cmd"] == 0 else f" (pedido {r['v_cmd']:.1f} cm/s)")
        print(f"  #{r['video_number']:<7d} iteração {r['iteration']:4d}: {'; '.join(r['reasons'])}{aid}{asked} | "
              f"{r['speed']:5.2f} cm/s, {r['distance']:5.2f} cm, deslizando {r['glide_frac']:.0%}, "
              f"{'caiu' if r['fell'] else 'não caiu'} em {r['seconds']:.2f} s")


if __name__ == "__main__":
    main()
