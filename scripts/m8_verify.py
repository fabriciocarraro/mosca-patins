"""M8: conferência de autenticidade dos clipes do vídeo (rode onde a execução rodou, na mesma máquina dos traços).

Para cada traço (.npz do m8_trace.py) usado na montagem:
- tentativa gravada: o número da tentativa existe no registro da execução (attempts.jsonl) com a mesma geração e o
  mesmo resultado; os hashes da captura e da política batem com os arquivos; e a física refeita a partir da captura dá
  as mesmas posturas, bit a bit. Com `--brain`, o cérebro também é re-simulado (diferença na ação < 1e-4);
- teste a partir de checkpoint (patins, caminhada ou grade): o hash do checkpoint bate, e o teste refeito com os mesmos
  argumentos dá as mesmas posturas, bit a bit.

O desenho do `m8_render.py` só depende das posturas: posturas idênticas dão as mesmas imagens.

Uso (no Spark):
    python scripts/m8_verify.py runs/final2_s0/video/*.npz --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosca.capture import IterationCapture, replay  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig  # noqa: E402
from mosca.paths import RUNS  # noqa: E402

DEFAULTS = {"test": 0, "tests": 0, "every": 10, "speed": 3.5, "yaw": 0.0, "seconds": 0.0, "silence": "", "stim": "",
            "stim_scale": 1.0, "walk": False}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def legacy_args(meta: dict) -> dict:
    """Argumentos de traços antigos, sem o campo "args": saem do rótulo da intervenção."""
    a = dict(DEFAULTS, test=meta.get("test", 0), speed=meta["v_cmd"], yaw=meta.get("yaw", 0.0),
             walk=meta.get("body") == "walk", tests=meta.get("tests", 0), every=meta.get("every", 10))
    tag = meta.get("intervention", "base")
    if tag.endswith("_calado"):
        a["silence"] = tag[: -len("_calado")]
    elif tag != "base":
        parts = tag.split("_")
        if parts[-1].startswith("x"):
            a["stim_scale"] = float(parts.pop()[1:])
        a["stim"] = "_".join(parts)
    return a


def check_capture(meta: dict, qpos: np.ndarray, brain: bool, device) -> list[str]:
    problems = []
    run_dir = RUNS / meta["run"]
    gen, a = meta["generation"], meta["attempt"]
    cap_path, pol_path = run_dir / "capture" / f"it{gen:05d}.npz", run_dir / "capture" / f"policy_it{gen:05d}.pt"
    if sha256(cap_path) != meta["capture_sha256"]:
        problems.append("hash da captura diferente")
    if sha256(pol_path) != meta["policy_sha256"]:
        problems.append("hash da política diferente")
    rows = [json.loads(line) for line in open(run_dir / "attempts.jsonl", encoding="utf-8") if f'"attempt": {a},' in line]
    row = next((r for r in rows if r["attempt"] == a), None)
    if row is None:
        problems.append("tentativa fora do registro")
    elif row.get("gen", row.get("it")) != gen or bool(row["fell"]) != bool(meta["stats"]["fell"]) \
            or abs(row["speed"] - meta["stats"]["speed"]) > 1e-9:
        problems.append("registro com outra geração ou outro resultado")
    manifest = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env_dict = dict(manifest["env"])
    env_dict["reward"] = RewardConfig(**env_dict["reward"])
    cap = IterationCapture.load(cap_path)
    poses = []
    phys = replay(cap, meta["env"], EnvConfig(**env_dict),
                  on_substep=lambda env, t, sub: poses.append(env.datas[0].qpos.astype(np.float32)))
    if phys["max_deviation"] != 0.0:
        problems.append(f"física desviou da captura ({phys['max_deviation']:.3g})")
    if not np.array_equal(np.stack(poses), qpos):
        problems.append("posturas diferentes das do traço")
    if brain:
        diff = brain_diff(manifest, EnvConfig(**env_dict), cap, pol_path, gen, meta["env"], device)
        if diff >= 1e-4:
            problems.append(f"cérebro re-simulado difere ({diff:.3g})")
    return problems


def brain_diff(manifest: dict, env_cfg: EnvConfig, cap: IterationCapture, pol_path: Path, gen: int, env: int,
               device) -> float:
    """Re-simula o cérebro da leva inteira da geração (como no m8_trace.py e no m5_rehearsal.py) e devolve a maior
    diferença entre a ação recalculada e a gravada, na tentativa do ambiente `env`."""
    from mosca.brain.controller import ConnectomePolicy, ControllerConfig
    from mosca.brain.graph import Connectome
    from mosca.capture import noise_beta, replay_brain
    from mosca.env.skate_env import attempt_seed
    from mosca.paths import MALECNS_DIR

    ctrl = dict(manifest["controller"])
    ctrl["turn_cells"] = tuple(ctrl["turn_cells"])
    run_args = manifest["args"]
    graph_path = Path(run_args["graph"])
    if not graph_path.exists():
        graph_path = MALECNS_DIR / graph_path.name
    pol = ConnectomePolicy(Connectome.load(graph_path), ControllerConfig(**ctrl), device=device)
    pol.load_state_dict(torch.load(pol_path, map_location=device, weights_only=True))
    evolution = "es_dim" in manifest
    population = None
    if evolution:
        from mosca.rl.es import PopulationES

        es = PopulationES(pol.es_parameters(), manifest["sigma"], 0.0, run_args["seed"])
        population = es.offsets(es.sample(gen, env_cfg.n_envs), device)
    beta = 0.0 if evolution else noise_beta(env_cfg.control_dt, run_args["noise_tau"])
    diff = replay_brain(cap, env_cfg, pol, attempt_seed(run_args["seed"], -1 - gen), beta, population=population,
                        action_noise=not evolution)
    return float(diff[env])


def check_test(meta: dict, qpos: np.ndarray, device) -> list[str]:
    import m8_trace

    problems = []
    ckpt = Path(meta["checkpoint"])
    if sha256(ckpt) != meta["checkpoint_sha256"]:
        problems.append("hash do checkpoint diferente")
    a = SimpleNamespace(**{**DEFAULTS, **(meta.get("args") or legacy_args(meta)), "checkpoint": str(ckpt)})
    if meta.get("device") and meta["device"] != str(device):
        problems.append(f"traço feito em {meta['device']}, conferência em {device}")
    if meta.get("body") == "walk":
        again = m8_trace.trace_walk(a, device, save=False)
    elif meta["mode"] == "grade":
        again = m8_trace.trace_grid(a, device, save=False)
    else:
        again = m8_trace.trace_test(a, device, save=False)
    if again.shape != qpos.shape or not np.array_equal(again, qpos):
        problems.append("o teste refeito deu outras posturas")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+")
    p.add_argument("--brain", action="store_true", help="re-simula também o cérebro das tentativas gravadas")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=4.0)
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    failed = 0
    for path in args.traces:
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        qpos = z["qpos"]
        if meta["mode"] == "captura":
            what = f"tentativa #{meta['video_number']:,}".replace(",", ".") + f", geração {meta['generation']}"
            problems = check_capture(meta, qpos, args.brain, device)
        else:
            what = f"{meta['mode']} {meta.get('intervention', '')}, {Path(meta['checkpoint']).name}"
            problems = check_test(meta, qpos, device)
        failed += bool(problems)
        print(f"{'OK    ' if not problems else 'FALHOU'} {Path(path).name}: {what}" +
              ("" if not problems else " | " + "; ".join(problems)), flush=True)
    print(f"{len(args.traces) - failed} de {len(args.traces)} traços conferidos")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
