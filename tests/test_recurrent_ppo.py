import torch

from mosca.rl.recurrent_ppo import chunks_with_data


def test_chunks_cover_exactly_the_executed_steps():
    valid = torch.zeros(10, 3, dtype=torch.bool)
    valid[:10, 0] = True  # tentativa inteira
    valid[:5, 1] = True  # caiu no passo 5
    valid[:1, 2] = True  # caiu logo no primeiro passo
    k, b = chunks_with_data(valid, chunk=4)  # trechos começam em 0, 4 e 8
    got = sorted(zip(k.tolist(), b.tolist()))
    assert got == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
