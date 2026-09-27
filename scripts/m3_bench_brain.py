"""M3: custo da rede de Pugliese no grafo de 6 patas, por passo de controle (10 ms, 2 subpassos de RK4).

Mede o passo sem gradiente (coleta das tentativas) com lotes de vários tamanhos e o passo
com gradiente através do tempo (atualização do PPO), com a memória limitada para não
disputar a máquina compartilhada.

Uso:
    python scripts/m3_bench_brain.py --device cuda --max-mem-gb 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosca.body.fly import LEGS  # noqa: E402
from mosca.brain.graph import Connectome  # noqa: E402
from mosca.brain.pugliese import estimate_sizes, sample_params, signed_matrix  # noqa: E402
from mosca.brain.rate_model import PuglieseNet  # noqa: E402
from mosca.paths import MALECNS_DIR  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-mem-gb", type=float, default=6.0)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    parser.add_argument("--bptt-batch", type=int, default=64)
    parser.add_argument("--bptt-steps", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_mem_gb * 2**30 / total), device)

    c = Connectome.load(MALECNS_DIR / "controller_graph_min5.npz")
    sizes, _, _ = estimate_sizes(c.body_id)
    params = sample_params(sizes, np.random.default_rng(0), reference=1.30 * np.nanmedian(sizes))
    net = PuglieseNet(signed_matrix(c), params, cell_type=c.cell_type, device=device)
    print(f"{device}: {c.n:,} neurônios, {net.w.values().numel():,} ligações, {net.log_a.numel():,} tipos celulares")
    stim = torch.zeros(c.n, device=device)
    stim[torch.as_tensor(c.groups["DNg100_R"], device=device)] = 400.0
    motor = torch.as_tensor(np.concatenate([c.groups[f"motor_{leg}"] for leg in LEGS]), device=device)

    with torch.no_grad():
        for batch in args.batches:
            r = torch.zeros(c.n, batch, device=device)  # estado (neurônios × lote)
            current = stim[:, None].expand(-1, batch).contiguous()
            for _ in range(5):
                r = net(r, current, dt=1e-2, substeps=2)
            sync(device)
            t0 = time.perf_counter()
            steps = 50
            for _ in range(steps):
                r = net(r, current, dt=1e-2, substeps=2)
            sync(device)
            ms = 1e3 * (time.perf_counter() - t0) / steps
            print(f"sem gradiente, lote {batch:4d}: {ms:7.2f} ms por passo de controle "
                  f"({ms / batch * 1e3:7.1f} µs por ambiente)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    r = torch.zeros(c.n, args.bptt_batch, device=device)
    current = stim[:, None].expand(-1, args.bptt_batch).contiguous()
    sync(device)
    t0 = time.perf_counter()
    loss = 0.0
    for _ in range(args.bptt_steps):
        r = net(r, current, dt=1e-2, substeps=2)
        loss = loss + r[motor].mean()
    loss.backward()
    sync(device)
    ms = 1e3 * (time.perf_counter() - t0) / args.bptt_steps
    mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    print(f"com gradiente (BPTT de {args.bptt_steps} passos), lote {args.bptt_batch}: {ms:.2f} ms por passo "
          f"(ida e volta), pico de memória {mem:.2f} GB")


if __name__ == "__main__":
    main()
