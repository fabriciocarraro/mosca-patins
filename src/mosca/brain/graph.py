"""Grafo do conectoma para o controlador das patas (MaleCNS v1.0).

Recorte: cordão nervoso (intrínsecos, sensoriais, motores, eferentes) mais os neurônios
descendentes e ascendentes, com ligações de pelo menos `min_synapses` sinapses. Por padrão,
só contam as sinapses dentro das regiões do cordão nervoso (como em Pugliese et al.): as que
descendentes e ascendentes trocam no cérebro, que não é simulado, formavam laços que faziam
a rede disparar sem controle (scripts/m3_vnc_weights.py gera essa contagem). O sinal de
cada ligação vem do neurotransmissor do neurônio pré-sináptico: acetilcolina excita, GABA
e glutamato inibem (como em Pugliese et al.). Para "incerto", usa a previsão por tipo
celular e depois a individual; outros transmissores (serotonina, histamina, etc.) ficam de
fora das ligações rápidas. A poda tira quem não tem caminho até um neurônio motor de pata.

Grupos por pata: neurônios motores (subclasses fl/ml/hl, por neurômero e lado), sensoriais
(proprioceptivos e táteis que entram pelos nervos daquela pata, pelo lado de entrada) e, entre
eles, só os proprioceptivos (órgão cordotonal, placas de pelos, sensilas campaniformes).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mosca.body.fly import LEGS
from mosca.paths import MALECNS_DIR

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_VNC = "connectome-weights-vnc.feather"  # pares do recorte; weight_vnc = sinapses no cordão nervoso

REGION = ("vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent",
          "ascending_neuron", "sensory_ascending", "descending_neuron")
FAST_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
LEG_MOTOR_SUBCLASS = {"T1": "fl", "T2": "ml", "T3": "hl"}
LEG_NERVES = {"T1": ("ProLN", "ProCN"), "T2": ("MesoLN",), "T3": ("MetaLN",)}
SENSORY_CLASSES = ("mechanosensory_proprioceptive", "mechanosensory_tactile")
COMMAND_TYPES = ("DNg100", "DNg97", "DNb08", "DNa01", "DNa02", "DNg13", "MDN", "DNp01", "DNp09")


@dataclass
class Connectome:
    body_id: np.ndarray  # (N,)
    superclass: np.ndarray  # (N,) texto
    cell_type: np.ndarray  # (N,) texto ("" sem tipo)
    side: np.ndarray  # (N,) "L", "R" ou ""
    sign: np.ndarray  # (N,) +1, -1 ou 0 (sem ligação rápida)
    in_synapses: np.ndarray  # (N,) sinapses recebidas (de qualquer parceiro contado, sem limiar)
    pre: np.ndarray  # (E,) índices
    post: np.ndarray  # (E,) índices
    count: np.ndarray  # (E,) nº de sinapses
    groups: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(self.body_id)

    def save(self, path: Path) -> None:
        """Salva sem objetos Python (texto como unicode de tamanho fixo), para ler sem pickle."""
        arrays = {f"group__{k}": v for k, v in self.groups.items()}
        np.savez_compressed(path, body_id=self.body_id, superclass=np.asarray(self.superclass, dtype=str),
                            cell_type=np.asarray(self.cell_type, dtype=str), side=np.asarray(self.side, dtype=str),
                            sign=self.sign, in_synapses=self.in_synapses,
                            pre=self.pre, post=self.post, count=self.count, **arrays)

    @staticmethod
    def load(path: Path) -> "Connectome":
        z = np.load(path, allow_pickle=False)
        groups = {k[len("group__"):]: z[k] for k in z.files if k.startswith("group__")}
        return Connectome(z["body_id"], z["superclass"], z["cell_type"], z["side"], z["sign"], z["in_synapses"],
                          z["pre"], z["post"], z["count"], groups)


def _signs(nt: pd.DataFrame, bodies: np.ndarray) -> np.ndarray:
    table = nt.set_index("body").reindex(bodies)
    sign = np.zeros(len(bodies), dtype=np.int8)
    for column in ("predicted_nt", "celltype_predicted_nt", "consensus_nt"):  # a última vence
        s = table[column].map(FAST_SIGN)
        sign = np.where(s.notna(), s.fillna(0).astype(np.int8), sign)
    return sign


def _reaches(targets: np.ndarray, pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    """Neurônios com caminho dirigido até algum dos alvos (busca reversa)."""
    order = np.argsort(post, kind="stable")
    starts = np.searchsorted(post[order], np.arange(n + 1))
    seen = np.zeros(n, dtype=bool)
    seen[targets] = True
    queue = deque(targets.tolist())
    while queue:
        v = queue.popleft()
        for u in pre[order[starts[v] : starts[v + 1]]]:
            if not seen[u]:
                seen[u] = True
                queue.append(u)
    return seen


def build_connectome(min_synapses: int = 5, data_dir: Path = MALECNS_DIR, vnc_only: bool = True) -> Connectome:
    ann = pd.read_feather(data_dir / ANNOTATIONS)
    ann = ann[ann.superclass.isin(REGION)].reset_index(drop=True)
    nt = pd.read_feather(data_dir / NEUROTRANSMITTERS)

    if vnc_only:
        weights = pd.read_feather(data_dir / WEIGHTS_VNC, columns=["body_pre", "body_post", "weight_vnc"])
        weights = weights.rename(columns={"weight_vnc": "weight"})
    else:
        weights = pd.read_feather(data_dir / WEIGHTS, columns=["body_pre", "body_post", "weight"])
    region_ids = ann.bodyId.to_numpy()
    in_syn = weights[weights.body_post.isin(region_ids)].groupby("body_post").weight.sum()
    weights = weights[(weights.weight >= min_synapses) & weights.body_pre.isin(region_ids) & weights.body_post.isin(region_ids)]
    index = pd.Series(np.arange(len(ann)), index=region_ids)
    pre, post = index[weights.body_pre].to_numpy(), index[weights.body_post].to_numpy()
    count = weights.weight.to_numpy().astype(np.int32)
    del weights

    motor = np.flatnonzero(ann.superclass.eq("vnc_motor") & ann.subclass.isin(LEG_MOTOR_SUBCLASS.values()))
    keep = _reaches(motor, pre, post, len(ann))
    sign = _signs(nt, region_ids)

    ann = ann[keep].reset_index(drop=True)
    remap = np.full(len(keep), -1)
    remap[keep] = np.arange(keep.sum())
    edge_keep = keep[pre] & keep[post]
    pre, post, count = remap[pre[edge_keep]], remap[post[edge_keep]], count[edge_keep]
    sign = sign[keep]

    side = np.where(ann.superclass.eq("vnc_sensory"), ann.rootSide, ann.somaSide).astype(object)
    side = np.where(pd.Series(side).isin(["L", "R"]), side, "").astype(str)
    groups: dict[str, np.ndarray] = {}
    for leg in LEGS:
        neuromere, side_code = leg.split("_")[0], "L" if leg.endswith("left") else "R"
        groups[f"motor_{leg}"] = np.flatnonzero(
            ann.superclass.eq("vnc_motor") & ann.subclass.eq(LEG_MOTOR_SUBCLASS[neuromere])
            & ann.somaNeuromere.eq(neuromere) & (side == side_code))
        groups[f"sensory_{leg}"] = np.flatnonzero(
            ann.superclass.eq("vnc_sensory") & ann["class"].isin(SENSORY_CLASSES)
            & ann.entryNerve.isin(LEG_NERVES[neuromere]) & (side == side_code))
        groups[f"proprio_{leg}"] = groups[f"sensory_{leg}"][
            ann["class"].to_numpy()[groups[f"sensory_{leg}"]] == "mechanosensory_proprioceptive"]
    for cell_type in COMMAND_TYPES:
        for side_code in ("L", "R"):
            idx = np.flatnonzero(ann.type.eq(cell_type) & (side == side_code))
            if len(idx):
                groups[f"{cell_type}_{side_code}"] = idx

    return Connectome(
        body_id=ann.bodyId.to_numpy(),
        superclass=ann.superclass.astype(str).to_numpy(),
        cell_type=ann.type.fillna("").astype(str).to_numpy(),
        side=side,
        sign=sign,
        in_synapses=in_syn.reindex(ann.bodyId).fillna(0).to_numpy().astype(np.int32),
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        count=count,
        groups=groups,
    )
