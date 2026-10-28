"""M7: tabelas do resultado com a receita final (registro de 24/10/2026 em docs/m7_controles.md), em markdown.

Lê, de cada execução: runs/NOME/m7_eval.json (m7_eval.py a cada 50 gerações ou iterações), runs/NOME/metrics.jsonl
(curvas de aprendizado), runs/probe_NOME_1400.log (m6_eval.py --probes, só nos conectomas) e
outputs/NOME_1400_marcha.json (m8_gait.py no teste fixo 1). Aplica a regra de leitura registrada à métrica primária
(primeira geração que cumpre o critério do M2): diferença só com separação completa das 3 sementes.

Uso:
    python scripts/m7_summary.py
    python scripts/m7_summary.py --root runs/m7f_resultados   # cópia dos arquivos do Spark
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ARMS = {  # braço -> execuções das sementes 0, 1 e 2
    "Real": ["final2_s0", "m7f_real_s1", "m7f_real_s2"],
    "Embaralhado": ["m7f_emb_s0", "m7f_emb_s1", "m7f_emb_s2"],
    "MLP": ["m7f_mlp_s0", "m7f_mlp_s1", "m7f_mlp_s2"],
}
END, PER_GEN = 1400, 64
LEGS = ["T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right"]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace(".", ",").replace("-", "−")


def first_ok(evals: dict) -> int | None:
    return next((g for g in sorted(evals) if evals[g]["m2_ok"]), None)


def gait_criteria(g: dict, m2: dict) -> list[bool]:
    """Os 5 critérios de marcha natural (docs/m6_regras_novas.md)."""
    ground = [g["patins"][leg]["no_chao"] for leg in LEGS]
    share = [g["patins"][leg]["parcela_do_empurrao"] for leg in LEGS]
    left = sum(s for leg, s in zip(LEGS, share) if leg.endswith("left"))
    return [min(ground) >= 0.5 and sum(ground) / 6 >= 0.75,
            max(share) <= 0.6 and min(left, 1 - left) >= 0.25,
            g["desliza_seis"] >= 0.25,
            m2["m2_fast"] >= 0.8 and m2["m2_falls"] < 0.1,
            abs(g["giro_medio"]) < 0.3]


def turns_both(s: dict) -> bool:
    """Vira quando pedido, para os dois lados (o critério de curvas do docs/m6_treino_final2.md)."""
    return s["turn_left"] >= 0.25 and s["turn_right"] <= -0.25 and s["turn_ok"] >= 0.5


def separated(a: list[int | None], b: list[int | None]) -> bool:
    """As 3 sementes de a cumprem o M2 antes de todas as de b (ou só elas cumprem)."""
    late = [END + 1 if x is None else x for x in b]
    return all(x is not None for x in a) and max(a) < min(late)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="pasta com runs/ e outputs/")
    args = p.parse_args()
    root = Path(args.root)

    evals, metrics, probes, gaits = {}, {}, {}, {}
    for runs in ARMS.values():
        for run in runs:
            evals[run] = {int(g): s for g, s in
                          json.loads((root / "runs" / run / "m7_eval.json").read_text(encoding="utf-8")).items()}
            rows = {}
            for line in (root / "runs" / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
                m = json.loads(line)
                rows[m.get("gen", m.get("it"))] = m  # retomadas repetem gerações: vale a última
            metrics[run] = [rows[k] for k in sorted(rows)]
            log = root / "runs" / f"probe_{run}_1400.log"
            if log.exists():
                text = log.read_text(encoding="utf-8")
                calm = re.search(r"DNg100 calados: (\S+) cm/s \(sem intervenção (\S+)\)", text)
                turn = re.search(r"efeito no giro médio (\S+) rad/s \(esquerdo\) e (\S+) rad/s \(direito\).*?"
                                 r"lado estimulado em (\d+)%", text)
                probes[run] = (float(calm[1]), float(calm[2]), float(turn[1]), float(turn[2]), int(turn[3]))
            gait = root / "outputs" / f"{run}_1400_marcha.json"
            if gait.exists():
                gaits[run] = json.loads(gait.read_text(encoding="utf-8"))

    print("## Métrica primária: primeira geração (ou iteração) que cumpre o critério do M2\n")
    print("| Braço | Semente 0 | Semente 1 | Semente 2 |")
    print("|---|---|---|---|")
    firsts = {}
    for arm, runs in ARMS.items():
        firsts[arm] = [first_ok(evals[r]) for r in runs]
        cells = [f"{g} (#{g * PER_GEN:,})".replace(",", ".") if g is not None else f"não cumpre até {END}"
                 for g in firsts[arm]]
        print(f"| {arm} | " + " | ".join(cells) + " |")
    print("\nEntre parênteses, as tentativas de treino até a avaliação (64 por geração ou iteração).\n")
    names = list(ARMS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            verdict = (f"{a} antes" if separated(firsts[a], firsts[b]) else
                       f"{b} antes" if separated(firsts[b], firsts[a]) else "empate")
            print(f"- {a} × {b}: {verdict}")

    print(f"\n## Na geração (ou iteração) {END} (m7_eval.py)\n")
    print("| Execução | M2 | ≥3 cm/s | Velocidade | Desliza | Quedas | Giro pedindo +0,5 / −0,5 rad/s | "
          "Acerto nas curvas | Parada |")
    print("|---|---|---|---|---|---|---|---|---|")
    for runs in ARMS.values():
        for r in runs:
            s = evals[r].get(END)
            if s is None:
                print(f"| {r} | (sem avaliação na {END}) |||||||| ")
                continue
            print(f"| {r} | {'sim' if s['m2_ok'] else 'não'} | {pct(s['m2_fast'])} | {num(s['m2_speed'])} cm/s | "
                  f"{pct(s['m2_glide'])} | {pct(s['m2_falls'])} | {num(s['turn_left'], sign=True)} / "
                  f"{num(s['turn_right'], sign=True)} | {pct(s['turn_ok'])} | {pct(s['stand_ok'])} |")

    print(f"\n## Ao longo do treino (m7_eval.py, {len(evals['final2_s0'])} avaliações por execução)\n")
    print("| Execução | Avaliações no critério do M2 | Viram para os dois lados | Melhor acerto nas curvas | "
          "Parada mínima | Currículo no fim (velocidade / giro) |")
    print("|---|---|---|---|---|---|")
    for runs in ARMS.values():
        for r in runs:
            e = evals[r]
            ok = sum(s["m2_ok"] for s in e.values())
            both = [g for g in sorted(e) if turns_both(e[g])]
            best_turn = max(e, key=lambda g: e[g]["turn_ok"])
            turn = (f"{pct(e[best_turn]['turn_ok'])} (na {best_turn})" if e[best_turn]["turn_ok"] > 0 else "0%")
            last = metrics[r][-1]
            top = next((m.get("gen", m.get("it")) for m in metrics[r] if m["v_max"] >= 4.0), None)
            print(f"| {r} | {ok} de {len(e)} | {len(both)} de {len(e)}{f' (a 1ª na {both[0]})' if both else ''} | "
                  f"{turn} | {pct(min(s['stand_ok'] for s in e.values()))} | {num(last['v_max'], 1)} cm/s"
                  f"{f', desde a {top}' if top is not None else ''} / {num(last['yaw_max'], 2)} rad/s |")
    print("\nViram para os dois lados: giro médio ≥ +0,25 rad/s pedindo +0,5, ≤ −0,25 pedindo −0,5 e acerto ≥50% "
          "(o critério de curvas do `docs/m6_treino_final2.md`). O acerto sozinho engana: quem gira sempre para um lado "
          "acerta as curvas desse lado.")

    print(f"\n## Provas causais na geração {END} (conectomas, m6_eval.py --probes pedindo 3 cm/s)\n")
    print("| Execução | DNg100 calados | DNa02 esquerdo | DNa02 direito | Vira para o lado estimulado |")
    print("|---|---|---|---|---|")
    for arm in ("Real", "Embaralhado"):
        for r in ARMS[arm]:
            if r in probes:
                calm, base, left, right, side = probes[r]
                print(f"| {r} | {num(calm)} cm/s (sem intervenção {num(base)}) | {num(left, sign=True)} rad/s | "
                      f"{num(right, sign=True)} rad/s | {side}% |")

    print(f"\n## Marcha na geração (ou iteração) {END} (m8_gait.py, teste fixo 1 pedindo 3,5 cm/s)\n")
    print("| Execução | Velocidade | Giro | Desliza (com os seis) | Patins no chão (FE FD ME MD TE TD) | "
          "Maior parcela do empurrão | Critérios 1–5 |")
    print("|---|---|---|---|---|---|---|")
    short = {"T1_left": "frente E", "T1_right": "frente D", "T2_left": "meio E", "T2_right": "meio D",
             "T3_left": "trás E", "T3_right": "trás D"}
    for runs in ARMS.values():
        for r in runs:
            g = gaits.get(r)
            if g is None or END not in evals[r]:
                print(f"| {r} | (sem medida) |||||| ")
                continue
            crit = gait_criteria(g, evals[r][END])
            top = max(LEGS, key=lambda leg: g["patins"][leg]["parcela_do_empurrao"])
            ground = " ".join(f"{100 * g['patins'][leg]['no_chao']:.0f}" for leg in LEGS)
            print(f"| {r} | {num(g['vel_media'])} cm/s | {num(g['giro_medio'], sign=True)} rad/s | "
                  f"{pct(g['desliza'])} ({pct(g['desliza_seis'])}) | {ground} | {short[top]} "
                  f"{pct(g['patins'][top]['parcela_do_empurrao'])} | "
                  f"{''.join('✓' if c else '✗' for c in crit)} ({sum(crit)} de 5) |")
    print("\nPatins no chão: % do tempo, na ordem frente, meio e trás, esquerdo (E) e direito (D). Os 5 critérios de marcha "
          "natural (`docs/m6_regras_novas.md`):\n\n"
          "1. patins no chão (cada um ≥50%, média ≥75%);\n"
          "2. empurrão dividido (nenhum patim >60%, cada lado ≥25%);\n"
          "3. desliza com os seis no chão ≥25% do tempo;\n"
          "4. velocidade e quedas do critério do M2;\n"
          "5. reto (|giro médio| <0,3 rad/s).")


if __name__ == "__main__":
    main()
