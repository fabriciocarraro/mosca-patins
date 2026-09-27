"""Leitor mínimo de checkpoints do TensorFlow (formato TensorBundle), sem depender do TF.

Serve para converter uma vez a política de caminhada do flybody (SavedModel do TF) para
numpy/PyTorch. O arquivo `.index` é uma tabela SSTable (formato do LevelDB) que mapeia o
nome de cada tensor para um BundleEntryProto (tipo, forma, posição no arquivo de dados);
o `.data-00000-of-00001` guarda os bytes crus, em little-endian.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_MAGIC = 0xDB4775248B80FB57
_DTYPES = {1: np.float32, 2: np.float64, 3: np.int32, 9: np.int64, 10: np.bool_, 19: np.float16}


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _block(buf: bytes, offset: int, size: int) -> list[tuple[bytes, bytes]]:
    """Entradas de um bloco da SSTable (chaves com prefixo compartilhado)."""
    data = buf[offset : offset + size]
    if buf[offset + size] != 0:
        raise ValueError("bloco comprimido: não suportado")
    n_restarts = struct.unpack_from("<I", data, len(data) - 4)[0]
    end = len(data) - 4 - 4 * n_restarts
    out, pos, key = [], 0, b""
    while pos < end:
        shared, pos = _varint(data, pos)
        unshared, pos = _varint(data, pos)
        vlen, pos = _varint(data, pos)
        key = key[:shared] + data[pos : pos + unshared]
        pos += unshared
        out.append((key, data[pos : pos + vlen]))
        pos += vlen
    return out


def _fields(msg: bytes) -> dict[int, list]:
    """Campos de uma mensagem protobuf (varint, fixo de 32/64 bits ou com comprimento)."""
    out: dict[int, list] = {}
    pos = 0
    while pos < len(msg):
        tag, pos = _varint(msg, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _varint(msg, pos)
        elif wire == 1:
            value, pos = msg[pos : pos + 8], pos + 8
        elif wire == 2:
            n, pos = _varint(msg, pos)
            value, pos = msg[pos : pos + n], pos + n
        elif wire == 5:
            value, pos = msg[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"tipo de campo {wire} não suportado")
        out.setdefault(field, []).append(value)
    return out


def read_checkpoint(prefix: Path) -> dict[str, np.ndarray]:
    """Todos os tensores numéricos de um checkpoint (`prefix` sem a extensão, ex.: .../variables/variables)."""
    prefix = Path(prefix)
    index = prefix.with_name(prefix.name + ".index").read_bytes()
    data = prefix.with_name(prefix.name + ".data-00000-of-00001").read_bytes()
    footer = index[-48:]
    if struct.unpack("<Q", footer[-8:])[0] != _MAGIC:
        raise ValueError("não é uma SSTable do TensorFlow")
    pos = 0
    _, pos = _varint(footer, pos)  # metaindex (offset)
    _, pos = _varint(footer, pos)  # metaindex (tamanho)
    idx_off, pos = _varint(footer, pos)
    idx_size, pos = _varint(footer, pos)
    tensors = {}
    for _, handle in _block(index, idx_off, idx_size):
        off, p = _varint(handle, 0)
        size, _ = _varint(handle, p)
        for key, value in _block(index, off, size):
            if not key:  # cabeçalho do bundle
                continue
            f = _fields(value)
            dtype = _DTYPES.get(f.get(1, [0])[0])
            if dtype is None:  # textos etc.
                continue
            shape = [(_fields(d).get(1, [0])[0]) for d in _fields(f[2][0]).get(2, [])] if 2 in f else []
            offset, nbytes = f.get(4, [0])[0], f.get(5, [0])[0]
            tensors[key.decode()] = np.frombuffer(data[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()
    return tensors
