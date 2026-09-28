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
| M2 MLP patina | Ok (execução E, iteração 55, ~1 h de treino no Spark). Nos 20 testes sem ruído, pedindo 3,5 cm/s: média de 3,02 a 3,09 cm/s em 5 s a partir do repouso, deslizando 98% do tempo, sem quedas (critério: ≥3 cm/s em ≥80% dos testes, ≥25%, <10%). O mesmo resultado com passo de física 2× e 4× menor; com a postura congelada ela para (não há energia de graça). Falta de controle de rumo: ela vira ~25°/s para a direita. As execuções A a D não saíram do lugar: ruído branco que empurrava a mosca por vibração, recompensa que pagava por ficar parada e bônus de rolamento medido no eixo errado. |
| M3 Conectoma montado | Em andamento. Grafo do controlador: 23.117 neurônios e 1,04 milhão de ligações (só sinapses do cordão nervoso), com os 381 motores das patas. Ritmo do DNg100 reproduzido na rede de Pugliese et al. (100% das réplicas, 11,5 Hz) e no nosso grafo de 6 patas, com um fator global de excitabilidade calibrado (100% das réplicas, ~10 Hz, na pata da frente esquerda). Com os dois DNg100 não há faixa rítmica: coordenar as patas fica para o treino (M4). Tabela músculo → junta pronta (sinal de cada junta conferido pela geometria; na pata da frente todo neurônio motor tem músculo identificado, nas outras 12 a 14 ficam com peso livre). Rede em torch com RK4: 2 subpassos de 5 ms por passo de controle reproduzem o passo de 1 ms (correlação 0,9995). |
| M4 Conectoma anda | Ok (28/09/2026; anda_r5I, iteração 6, avaliado pelo `m4_eval.py`): sozinho, sem professora, 98% das caminhadas retas de 5 s bem-sucedidas (0,95, 1,82 e 2,67 cm/s para 1, 2 e 3 pedidos, desviando menos de 0,5 rad/s); o DNa02 de um lado vira a mosca para o próprio lado em 100% dos testes (+0,75 / −0,77 rad/s contra a mesma caminhada sem estímulo); calar o DNg100 faz a mosca parar (0,09 cm/s). Como chegamos lá: a política do flybody só anda reagindo em menos de 10 ms, e o conectoma com neurônios de τ 20 ms respondia em 20 a 40 ms e só ficava em pé; com τ de 5 ms (dentro da faixa biológica) ele passou a andar. A destilação (DAgger) usa ganho por ligação (degrau 2, sinal fixo), erro depois do filtro dos atuadores, referência que acompanha a mosca, os halteres como sentido de rotação (grupo sensorial declarado), dados espelhados esquerda-direita, o comando de giro só no DNa02, tentativas mandadas ficar paradas e ganhos por ligação aprendendo devagar; foram várias etapas partindo da anterior (anda_r5L → r5H → r5M → r5D → r5I), todas registradas. No vídeo: dizer que a força das ligações e a constante de tempo foram ajustadas e que os halteres entram como sentido. Controle (M7): com esta receita o embaralhado também anda, então a comparação real × embaralhado fica para a patinação (a virada do plano). |
| M5 Ensaio da captura | Ok (29/09/2026, no Spark): num treino curto com `--capture --deterministic`, o `m5_rehearsal.py` regenerou 4 tentativas sorteadas de 5 s a partir do manifesto (config.json, versões e política de cada iteração), com a física idêntica bit a bit e o cérebro re-simulado com diferença 0 nas ações (critério: <1e-4). Sem o modo determinístico, só em float64: ~2e-6. |
| M6 Treino final | Em andamento: PPO de patins partindo do conectoma que anda (execução patina_e, com crítico aquecido primeiro, âncora no andar e exploração maior). Até a iteração 9, sem ruído a política ainda fica parada. Falta o currículo de curvas e slalom. |
| M7 em diante | A fazer |

## Instalação

Notebook (Windows) ou DGX Spark (Linux Arm), Python 3.11 ou mais novo:

```bash
git clone https://github.com/fabriciocarraro/mosca-patins
cd mosca-patins
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,render,data,brain]"   # Linux: .venv/bin/python
python scripts/download_assets.py --walking-sample
python scripts/download_assets.py --malecns --pugliese --malecns-stats   # cérebro (M3)
python scripts/download_assets.py --walking-policy                         # professora de caminhada (M4)
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
| `python scripts/m2_eval.py runs/NOME/latest.pt --speed 3.5` | M2: os 20 testes fixos e os critérios de pronto (`--sheet`, `--stochastic`) |
| `python scripts/m3_vnc_weights.py` | M3: sinapses do cordão nervoso por par de neurônios (baixa 6,8 GB; rode no Spark) |
| `python scripts/m3_build_graph.py` | M3: grafo do controlador e contagens por pata |
| `python scripts/m3_rhythm_test.py` | M3: ritmo do DNg100 na rede de Pugliese et al. |
| `python scripts/m3_rhythm_full.py` | M3: ritmo do DNg100 no grafo de 6 patas |
| `python scripts/m3_bench_brain.py --device cuda` | M3: custo da rede por passo de controle (coleta e gradiente) |
| `python scripts/m7_shuffle_graph.py --seed 0` | M7: grafo de controle com o conectoma embaralhado (mesmas ligações e sinais por neurônio) |
| `python scripts/m4_distill.py --run NOME --device cuda` | M4: o conectoma aprende a andar imitando a política do flybody (DAgger) |
| `python scripts/m4_slow_teacher.py --run NOME --init-from CKPT` | M4: professora lenta (MLP com a latência do conectoma, PPO com currículo de latência) |
| `python scripts/m4_bc_probe.py --variants "base:" "rapido:tau_scale=0.25"` | M4: triagem de configurações do conectoma por imitação offline |
| `python scripts/m4_eval.py --ckpt runs/NOME/best.pt --device cuda` | M4: critério de pronto (retas, curvas, DNa02 sozinho, DNg100 calado) |
| `python scripts/connectome_train.py --run NOME --device cuda` | conectoma aprende a patinar por PPO (rede na GPU, física na CPU; `--capture` grava as tentativas) |
| `python scripts/replay_attempt.py --run NOME --attempt N --gif saida.gif --slow 20` | re-simula a tentativa N bit a bit a partir da captura e grava um GIF em câmera lenta real (subpasso a subpasso) |
| `python scripts/m5_rehearsal.py --run NOME --count 3` | M5: regenera tentativas sorteadas a partir da captura (física bit a bit, cérebro re-simulado com a política da coleta) |
| `python scripts/m6_eval.py runs/NOME/latest.pt --speed 3.5 --probes` | M6: testes fixos do conectoma de patins (critério do M2, curvas com `--yaw`, provas causais do DNg100 e do DNa02, `--sheet`/`--gif`, `--stochastic`) |
| `python scripts/m6_sensitivity.py --init-from CKPT --device cuda` | M6: sensibilidade da ação a cada grupo de parâmetros, para as taxas relativas do PPO |
| `python scripts/m6_milestones.py --run NOME` | tentativas que vão para o vídeo: números pré-registrados, marcos, recordes e amostra fixa de 5% |

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
src/mosca/brain/rate_model.py  rede de Pugliese em torch (lote, RK4, fatores por tipo celular), para o controlador
src/mosca/brain/muscles.py  tabela músculo → junta com sinal fixo (decodificador por pata)
src/mosca/brain/controller.py  controlador de conectoma: codificador por pata, comando no DNg100, decodificador pelos músculos
src/mosca/rl/recurrent_ppo.py  PPO com ator recorrente (trechos de retropropagação truncada)
src/mosca/capture.py       captura fiel das tentativas e re-simulação bit a bit
src/mosca/walking/         M4: professora (política de caminhada do flybody em numpy) e ambiente de caminhada
```

## Créditos e licenças

Nada de terceiros fica versionado: `scripts/download_assets.py` baixa tudo com versões fixadas.

- flybody (Vaxenburg et al., Nature 2025): código Apache-2.0; dados de caminhada e políticas no figshare da Janelia, GPL-3.0+ (a política de caminhada é a professora do M4).
- MaleCNS v1.0 (Berg et al., Cell 2026): CC-BY 4.0.
- Modelo de taxa e rede do cordão nervoso de Pugliese et al. (bioRxiv 2025, [código](https://github.com/smpuglie/Pugliese_2026) MIT): equação, parâmetros e nota de ritmo reproduzidos aqui.
