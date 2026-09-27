# CLAUDE.md

Guia para sessões do Claude Code neste projeto. Responda em português do Brasil.

## O projeto

Uma mosca simulada (corpo do flybody no MuJoCo), controlada pelo conectoma do MaleCNS,
aprende a andar de patins; o vídeo final mostra tentativas reais ("Tentativa #N"). O plano
aprovado, com as etapas M0–M8 e os critérios de pronto, está em `docs/plano.md`; o estado de
cada etapa está no README.

## Onde roda

- Notebook Windows: código, testes rápidos, prévias. Ambiente em `.venv` (`.venv\Scripts\python`).
- DGX Spark do usuário (aarch64, CUDA 13, memória unificada): treino e renderização, com o
  repositório clonado em `~/mosca-patins` e ambiente criado pelo `uv`. A máquina é
  compartilhada com outros jobs do usuário: rode com `nice`, limite núcleos e memória
  (estourar a GPU trava a máquina inteira) e confira o espaço em disco antes de gravar logs.

## Comandos

- Testes: `python -m pytest -q`
- Assets (não versionados): `python scripts/download_assets.py --walking-sample --malecns --pugliese --malecns-stats`
- Física: `scripts/bench_physics.py`, `scripts/m1_skate_physics.py`, `scripts/m1_gait_search.py`
- Treino M2 (no Spark, com `nice`): `scripts/m2_train.py --run NOME`; avaliação: `scripts/m2_eval.py`
- Cérebro M3: `scripts/m3_vnc_weights.py` e `scripts/m3_build_graph.py` (Spark), `scripts/m3_rhythm_full.py`

## Regras do projeto

- Fidelidade: nenhum clipe do vídeo pode ser encenado. Cada tentativa precisa ser
  re-simulável a partir do que foi gravado; o replay da física deve continuar idêntico
  bit a bit (há teste para isso).
- Unidades CGS (cm, g, s), como no flybody. Não confiar nos padrões do MuJoCo para geoms
  novos (densidade, atrito de torção/rolamento, maciez de contato): tudo explícito.
- Mudanças de física passam pelos testes de contato do trenó (`tests/test_skate_contact.py`).
- Commit e push só quando o usuário pedir; mensagens de commit em português.

## Achados que orientam os próximos passos

- Marchas programadas fixas só conseguem "andar de pato" (patins plantados, rolamento ≤ 0,11),
  mesmo com servos 3× mais fortes. O deslize precisa de realimentação: é o teste do M2
  (MLP treinada por reforço, com bônus de rolamento como no humanoide de patins da ETH).
- As patas giram pouco o patim (±15° na frente, 5–15° só para fora no meio e atrás).
  Alavancas se o deslize não surgir: ângulo de montagem com a ponta para fora, comprimento
  do patim, atrito de rolamento.
- M2: antes de mudar hiperparâmetros, olhe a política (`m2_eval.py --sheet`, com e sem
  `--stochastic`). Duas armadilhas já vistas: ruído de exploração que faz a mosca andar
  sozinho (a política média fica parada) e termos de recompensa que pagam por ficar parada.
  A avaliação do treino usa a velocidade já liberada pelo currículo.
- M3: o modelo de Pugliese só oscila numa faixa estreita de excitabilidade. Contam só as
  sinapses do cordão nervoso (as do cérebro entre descendentes e ascendentes fazem a rede
  disparar), o volume vem da tabela deles ou da estimativa pelas sinapses, e a referência do
  volume é 1,30× a mediana do nosso grafo (com a mediana própria, silêncio; com 1,40×, disparo).
