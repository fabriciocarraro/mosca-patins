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
| M1 Física dos patins | Contato validado no trenó; na mosca inteira os patins rolam; marchas programadas deslizam a 2 a 2,6 cm/s (rolamento 0,99). Uma versão anterior dizia que elas só "andavam de pato": era erro de medida (velocidade do patim lida no referencial de inércia, com os eixos trocados), corrigido. |
| M2 MLP patina | Quase pronto. Execução E, sem ruído, a 3 cm/s pedidos: 2,7 cm/s, deslizando 98% do tempo, sem quedas (critério: ≥3 cm/s, ≥25%, <10%). O mesmo resultado com passo de física 2× e 4× menor (não é artefato numérico). Ela ainda vai girando devagar. As execuções A a D não saíram do lugar: ruído branco que empurrava a mosca por vibração, recompensa que pagava por ficar parada e bônus de rolamento medido no eixo errado. |
| M3 Conectoma montado | Em andamento. Grafo do controlador: 23.117 neurônios e 1,04 milhão de ligações (só sinapses do cordão nervoso), com os 381 motores das patas. Ritmo do DNg100 reproduzido na rede de Pugliese et al. (100% das réplicas, 11,5 Hz) e no nosso grafo de 6 patas, com um fator global de excitabilidade calibrado (100% das réplicas, ~10 Hz, na pata da frente esquerda). Com os dois DNg100 não há faixa rítmica: coordenar as patas fica para o treino (M4). Falta a tabela músculo → junta. |
| M4 em diante | A fazer |

## Instalação

Notebook (Windows) ou DGX Spark (Linux Arm), Python 3.11 ou mais novo:

```bash
git clone https://github.com/fabriciocarraro/mosca-patins
cd mosca-patins
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,render,data,brain]"   # Linux: .venv/bin/python
python scripts/download_assets.py --walking-sample
python scripts/download_assets.py --malecns --pugliese --malecns-stats   # cérebro (M3)
```

O PyTorch entra à parte: a versão para CPU no notebook e a cu130 no Spark.

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
| `python scripts/m2_eval.py runs/NOME/latest.pt` | M2: os 20 testes fixos e os critérios de pronto (`--sheet`, `--stochastic`) |
| `python scripts/m3_vnc_weights.py` | M3: sinapses do cordão nervoso por par de neurônios (baixa 6,8 GB; rode no Spark) |
| `python scripts/m3_build_graph.py` | M3: grafo do controlador e contagens por pata |
| `python scripts/m3_rhythm_test.py` | M3: ritmo do DNg100 na rede de Pugliese et al. |
| `python scripts/m3_rhythm_full.py` | M3: ritmo do DNg100 no grafo de 6 patas |

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
src/mosca/brain/graph.py   grafo do controlador (MaleCNS): recorte, sinais, grupos por pata
src/mosca/brain/pugliese.py  modelo de taxa de Pugliese et al., volume estimado, nota de ritmo
src/mosca/brain/rate_model.py  rede de taxa em lote (torch), para o controlador
```

## Créditos e licenças

Nada de terceiros fica versionado: `scripts/download_assets.py` baixa tudo com versões fixadas.

- flybody (Vaxenburg et al., Nature 2025): código Apache-2.0; dados de caminhada e políticas no figshare da Janelia, GPL-3.0+.
- MaleCNS v1.0 (Berg et al., Cell 2026): CC-BY 4.0.
- Modelo de taxa e rede do cordão nervoso de Pugliese et al. (bioRxiv 2025, [código](https://github.com/smpuglie/Pugliese_2026) MIT): equação, parâmetros e nota de ritmo reproduzidos aqui.
