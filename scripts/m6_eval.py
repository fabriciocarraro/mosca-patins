"""M6: avalia o conectoma de patins (checkpoint do connectome_train.py) nos testes fixos.

Testes: as mesmas 20 tentativas de sementes fixas do M2 (nunca aparecem no treino), a partir do
repouso, sem empurrão. Sem `--stochastic`, com a ação média da política (é o que vale para os
critérios); com ele, com o ruído de exploração do treino. Critério do M2: ≥80% dos testes com
média ≥ `--min-speed` em 5 s, ≥25% do tempo deslizando e <10% de quedas.

- `--yaw Y`: metade dos testes pede giro de +Y e metade de −Y (rad/s); o erro de giro é o do giro
  médio de 200 ms em relação ao pedido.
- `--probes`: provas causais, pareadas com os mesmos testes sem intervenção: calar os DNg100 (a
  mosca deve parar) e estimular o DNa02 de um lado com a corrente de um comando de giro de 1 rad/s
  (deve virar para esse lado).
- `--sheet` / `--gif`: folha de quadros (um a cada 0,25 s) ou GIF do teste 0.

Uso:
    python scripts/m6_eval.py runs/patina_e/latest.pt --speed 1 --sheet outputs/patina_e_folha.png
    python scripts/m6_eval.py runs/patina_e/latest.pt --speed 1 --stochastic
    python scripts/m6_eval.py runs/NOME/latest.pt --speed 3.5 --probes --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.brain.controller import ConnectomePolicy, ControllerConfig  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.capture import ColoredNoise, noise_beta  # noqa: E402
from mosca.env.skate_env import EnvConfig, RewardConfig, SkateVecEnv  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402
from mosca.walking.evaluate import side_current  # noqa: E402

TEST_ATTEMPT_BASE = 10**9  # os mesmos testes do M2 e da avaliação do treino


def load(path: str, device, control_dt: float = 0.0, substeps: int = 0) -> tuple[ConnectomePolicy, Connectome, dict, dict]:
    """`control_dt`/`substeps` > 0 trocam o passo de controle do checkpoint (teste de sensibilidade)."""
    ckpt = torch.load(path, weights_only=False, map_location=device)
    ctrl = dict(ckpt["controller"])
    ctrl["turn_cells"] = tuple(ctrl["turn_cells"])
    if control_dt > 0:
        ctrl["control_dt"] = control_dt
        ckpt["env_cfg"] = {**ckpt["env_cfg"], "control_dt": control_dt}
    if substeps > 0:
        ctrl["substeps"] = substeps
    graph_path = Path(ckpt["args"]["graph"])
    if not graph_path.exists():
        graph_path = MALECNS_DIR / graph_path.name
    graph = Connectome.load(graph_path)
    pol = ConnectomePolicy(graph, ControllerConfig(**ctrl), device=device)
    pol.load_state_dict(ckpt["policy"])
    return pol, graph, ckpt, ckpt["args"]


def make_env(ckpt: dict, n: int, threads: int) -> SkateVecEnv:
    cfg = dict(ckpt["env_cfg"])
    cfg["reward"] = RewardConfig(**cfg["reward"])
    cfg.update(n_envs=n, n_threads=threads)
    return SkateVecEnv(EnvConfig(**cfg))


def run(env: SkateVecEnv, pol: ConnectomePolicy, speed: float, yaw: np.ndarray, noise: ColoredNoise | None = None,
        silence=None, extra_current=None, on_step=None) -> list[dict]:
    """Uma leva de testes; devolve as estatísticas de cada tentativa."""
    n, device = env.n, pol.device
    obs, _ = env.reset(TEST_ATTEMPT_BASE + np.arange(n), np.full(n, speed), yaw_cmd=yaw)
    v = torch.full((n,), float(speed), device=device)
    r = pol.initial_state(n)
    std = pol.log_std.detach().exp()
    with torch.no_grad():
        while env.alive.any():
            current = pol.currents(torch.as_tensor(obs, dtype=torch.float32, device=device), v)
            if extra_current is not None:
                current = current + extra_current
            r = pol.step_net(r, current)
            if silence is not None:
                r[silence] = 0.0
            action = pol.decode(r)
            if noise is not None:
                action = action + std * noise.step()
            obs, *_ = env.step(action.cpu().numpy())
            if on_step is not None:
                on_step(env)
    return env.episode_stats()


def table(episodes: list[dict]) -> None:
    print(" teste  dur(s)  queda  vel(cm/s)  rolamento  desliza  patins no chão  giro pedido  giro médio  erro de giro  |giro|")
    for k, e in enumerate(episodes):
        print(f"  {k:4d}  {e['seconds']:6.2f}  {'sim' if e['fell'] else 'não':>5}  {e['speed']:9.2f}  {e['rolling']:9.2f}"
              f"  {e['glide_frac']:7.2f}  {e['grounded_frac']:14.2f}  {e['yaw_cmd']:11.2f}  {e['yaw_rate']:10.2f}"
              f"  {e['yaw_error']:12.2f}  {e['yaw_abs_rate']:6.2f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--tests", type=int, default=20)
    p.add_argument("--speed", type=float, default=3.0, help="velocidade pedida (cm/s)")
    p.add_argument("--min-speed", type=float, default=3.0, help="média mínima do critério do M2 (cm/s)")
    p.add_argument("--yaw", type=float, default=0.0, help="giro pedido (±, metade dos testes para cada lado)")
    p.add_argument("--stochastic", action="store_true", help="com o ruído de exploração do treino")
    p.add_argument("--probes", action="store_true", help="provas causais (DNg100 calado, DNa02 de um lado)")
    p.add_argument("--control-dt", type=float, default=0.0, help="troca o passo de controle do treino (s)")
    p.add_argument("--substeps", type=int, default=0, help="troca os subpassos de RK4 por passo de controle")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-gpu-mem-gb", type=float, default=6.0)
    p.add_argument("--gif", default="", help="GIF do teste 0 (câmera lenta 4×)")
    p.add_argument("--sheet", default="", help="folha de quadros (PNG) do teste 0, um a cada 0,25 s")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_mem_gb * 2**30 / total), device)
    pol, graph, ckpt, train_args = load(args.checkpoint, device, args.control_dt, args.substeps)
    state = ckpt["state"]
    n = args.tests
    env = make_env(ckpt, n, args.threads)
    yaw = np.where(np.arange(n) % 2 == 0, args.yaw, -args.yaw)
    beta = noise_beta(env.cfg.control_dt, train_args.get("noise_tau", 0.05))

    def noise():
        return ColoredNoise(0, (n, env.act_dim), beta, device) if args.stochastic else None

    print(f"checkpoint: iteração {state['it']}, {state['attempts']} tentativas de treino, v_max {state['v_max']:.1f} cm/s, "
          f"giro máximo {state.get('yaw_max', 0.0):.2f} rad/s; desvio do ruído {pol.log_std.exp().mean().item():.2f}")
    if args.stochastic:
        print("(com ruído de exploração: não vale para os critérios)")
    base = run(env, pol, args.speed, yaw, noise())
    table(base)
    fast = np.mean([not e["fell"] and e["seconds"] >= env.cfg.episode_seconds - 1e-9 and e["speed"] >= args.min_speed
                    for e in base])
    glide = np.mean([e["glide_frac"] for e in base])
    falls = np.mean([e["fell"] for e in base])
    print(f"\nvelocidade pedida {args.speed:g} cm/s: média {np.mean([e['speed'] for e in base]):.2f} cm/s; "
          f"média ≥{args.min_speed:g} cm/s em {env.cfg.episode_seconds:g} s em {fast:.0%} dos testes (critério ≥80%)")
    print(f"tempo deslizando: {glide:.0%} (critério ≥25%); quedas: {falls:.0%} (critério <10%); "
          f"erro de giro {np.mean([e['yaw_error'] for e in base]):.2f} rad/s, |giro| instantâneo "
          f"{np.mean([e['yaw_abs_rate'] for e in base]):.2f} rad/s")
    print("critério do M2:", "OK" if fast >= 0.8 and glide >= 0.25 and falls < 0.1 else "ainda não")

    if args.probes:
        dng100 = torch.as_tensor(np.concatenate([graph.groups["DNg100_L"], graph.groups["DNg100_R"]]), device=device)
        quiet = run(env, pol, args.speed, yaw, noise(), silence=dng100)
        print(f"\nDNg100 calados: {np.mean([e['speed'] for e in quiet]):.2f} cm/s "
              f"(sem intervenção {np.mean([e['speed'] for e in base]):.2f}); quedas {np.mean([e['fell'] for e in quiet]):.0%}")
        sides = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        stim = run(env, pol, args.speed, yaw, noise(), extra_current=side_current(pol, graph.groups, "DNa02", sides, device))
        delta = np.array([s["yaw_rate"] - b["yaw_rate"] for s, b in zip(stim, base)])
        both = np.array([not s["fell"] and not b["fell"] for s, b in zip(stim, base)])
        right = (np.sign(delta) == sides) & both
        print(f"DNa02 de um lado: efeito no giro médio {delta[sides > 0].mean():+.2f} rad/s (esquerdo) e "
              f"{delta[sides < 0].mean():+.2f} rad/s (direito), contra os mesmos testes sem estímulo; "
              f"vira para o lado estimulado em {right.mean():.0%} dos testes")
    env.close()

    if args.gif or args.sheet:
        from PIL import Image, ImageDraw

        env = make_env(ckpt, 1, 1)
        renderer = mujoco.Renderer(env.model, height=480, width=854)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, env.thorax
        cam.distance, cam.elevation, cam.azimuth = 0.9, -22, 120
        frames = []

        def grab(e):
            renderer.update_scene(e.datas[0], camera=cam)
            frames.append(Image.fromarray(renderer.render()))

        one_noise = ColoredNoise(0, (1, env.act_dim), beta, device) if args.stochastic else None
        run(env, pol, args.speed, yaw[:1], one_noise, on_step=grab)
        label = f"{Path(args.checkpoint).parent.name} it {state['it']}, {args.speed:g} cm/s" + (" (com ruído)" if args.stochastic else "")
        if args.gif:
            Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=40, loop=0)
            print(f"GIF: {args.gif} ({len(frames)} quadros a 100 Hz, reproduzido a 25 fps = 4× mais lento)")
        if args.sheet:
            every = max(1, round(0.25 / env.cfg.control_dt))
            picked = frames[::every][:20]
            cols, (w, h) = 4, (427, 240)
            sheet = Image.new("RGB", (cols * w, -(-len(picked) // cols) * h), "white")
            for k, frame in enumerate(picked):
                tile = frame.resize((w, h))
                ImageDraw.Draw(tile).text((6, 4), f"t = {k * every * env.cfg.control_dt:.2f} s  {label}", fill="black")
                sheet.paste(tile, ((k % cols) * w, (k // cols) * h))
            Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
            sheet.save(args.sheet)
            print(f"folha de quadros: {args.sheet} ({len(picked)} quadros, um a cada {every * env.cfg.control_dt:.2f} s)")
        env.close()


if __name__ == "__main__":
    main()
