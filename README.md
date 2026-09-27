# Mosca de patins

Uma mosca-da-fruta simulada, controlada pelo conectoma real, aprende a andar de patins. O
vídeo final mostra a evolução com tentativas reais ("Tentativa #N"), capturadas de forma
fiel durante o treino.

- Corpo: modelo [flybody](https://github.com/TuragaLab/flybody) (MuJoCo), carregado direto pelo `mujoco`, com patins acrescentados.
- Cérebro (a partir do M3): cordão nervoso e neurônios descendentes do [MaleCNS v1.0](https://male-cns.janelia.org/).
- Plano completo: [docs/plano.md](docs/plano.md).

## Estado

| Etapa | Situação |
|---|---|
| M0 Ferramentas | Ok nas duas máquinas, com replay de 5 s idêntico bit a bit. Notebook: 6,6 mil passos de física/s em 1 thread, 4,3 mil passos de controle/s com 16 threads. DGX Spark: 11,9 mil passos de física/s em 1 thread, 13,9 mil passos de controle/s com 20 threads (física fica na CPU). |
| M1 Física dos patins | Contato validado no trenó; na mosca inteira os patins rolam; marcha programada anda a ~2 cm/s. Achado: marchas fixas só conseguem "andar de pato" (patins plantados), não deslizar. O deslize fica para o M2, com realimentação e bônus de rolamento. |
| M2 MLP patina | Em andamento: ambiente de RL em lote e PPO prontos, primeiro treino no Spark. |
| M3 em diante | A fazer |

## Instalação

Notebook (Windows) ou DGX Spark (Linux Arm), Python 3.11 ou mais novo:

```bash
git clone https://github.com/fabriciocarraro/mosca-patins
cd mosca-patins
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,render,data]"   # Linux: .venv/bin/python
python scripts/download_assets.py --walking-sample
```

## Comandos

| Comando | O que faz |
|---|---|
| `pytest -q` | testes (modelo da mosca, replay bit a bit, contato das lâminas) |
| `python scripts/bench_physics.py` | M0: velocidade da física e determinismo |
| `python scripts/render_preview.py --skates --camera side` | imagem da mosca de patins |
| `python scripts/m1_skate_physics.py` | M1: checagens de física com a mosca inteira |
| `python scripts/m1_gait_search.py --require-rolling` | M1: busca de marcha programada (CMA-ES) |
| `python scripts/derive_skate_mounts.py` | refaz a montagem dos patins e a postura a partir de moscas reais |
| `python scripts/m2_train.py --run NOME` | M2: treino da MLP patinadora por PPO (saídas em `runs/NOME/`) |
| `python scripts/m2_eval.py runs/NOME/latest.pt` | M2: os 20 testes fixos e os critérios de pronto |

## Estrutura

```
src/mosca/body/fly.py      mosca base (asas, boca e antenas rígidas; modo de andar do flybody)
src/mosca/body/skates.py   patins: 4 lâminas curtas por patim, atrito anisotrópico
src/mosca/body/stance.py   postura canônica "parada de patins" (medianas de moscas reais)
src/mosca/body/ik.py       cinemática inversa das patas
src/mosca/body/gait.py     marchas periódicas programadas (M1)
src/mosca/body/poses.py    poses reais, ajuste de altura, controles que seguram a pose
src/mosca/env/skate_env.py ambiente de RL em lote (física em threads, recompensa, quedas)
src/mosca/rl/ppo.py        PPO com tentativas completas por versão da política
```

## Créditos e licenças

Nada de terceiros fica versionado: `scripts/download_assets.py` baixa tudo com versões fixadas.

- flybody (Vaxenburg et al., Nature 2025): código Apache-2.0; dados de caminhada e políticas no figshare da Janelia, GPL-3.0+.
- MaleCNS v1.0 (Berg et al., Cell 2026): CC-BY 4.0.
