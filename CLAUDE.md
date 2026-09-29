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

- Velocidade de patim: `mj_objectVelocity` com `mjOBJ_XBODY` (referencial do patim). Com
  `mjOBJ_BODY` o MuJoCo usa o referencial de inércia, com os eixos reordenados; esse erro
  fez o M1 concluir que as marchas programadas "andavam de pato" (na verdade deslizam,
  rolamento 0,99) e estragou o bônus de rolamento das execuções A a E.
- A MLP da execução E desliza 98% do tempo com movimentos pequenos das patas (juntas
  oscilando 1–6°, 5–8 Hz); o resultado não muda com passo de física menor.
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
  O ritmo é de ~10 Hz, mas balanço e apoio da coxa ficam só ~0,2 de ciclo defasados (também
  na rede deles): no vídeo, falar em ritmo, não em alternância de passada.
- Controlador de conectoma: com a referência 1,30×, qualquer entrada sensorial faz a rede
  disparar; o controle usa 1,0×, proprioceptores 2 abaixo do limiar e tônus nos motores
  (sem o tônus a rede fica calada com a mosca parada e o PPO não aprende). Gradiente
  substituto na rede; `torch.clamp` zera o gradiente no limite (use passagem direta). Antes
  de mudar hiperparâmetros do PPO, meça a sensibilidade da ação a cada grupo de parâmetros:
  pesos sem escala num grupo dominam a divergência KL e travam a taxa dos outros.
- Professora do M4 (política de caminhada do flybody): as 59 ações vêm na ordem da política
  (adesão, cabeça, abdômen, patas), diferente da ordem dos atuadores do modelo; o remapeamento
  está em `mosca.walking.teacher.ACTION_ORDER`. Sem ele, a mosca cai na hora. Os sensores da
  observação são a média dos 10 subpassos de cada passo de controle (2 ms), como no flybody.
- Para saber onde um controlador de conectoma perde a informação, grave a atividade numa
  coleta conduzida pela professora e meça o teto linear (regressão das ações dela) a partir
  de cada camada: sentidos crus, proprioceptores, pré-motores, motores. No M4, a informação
  chegava aos pré-motores e sumia nos motores; parâmetros por tipo celular não resolveram.
- Latência decide o M4: a política do flybody só anda reagindo em menos de 10 ms (com as
  ações atrasadas 10 ms, ou filtradas com τ de 20 ms, todas as moscas caem em menos de 1 s).
  O comando dela é liga-desliga: 73% da variância é um chacoalhar que o filtro dos atuadores
  (τ 10 ms) apaga. O conectoma com τ de 20 ms responde aos sentidos das patas em 20–40 ms e só
  aprendeu a ficar em pé (anda_c: 100% sem cair, velocidade 0,02). Com τ×0,25 (5 ms), a
  imitação offline chega a 0,27 de erro filtrado, contra 0,72 (`m4_bc_probe.py`). A MLP
  destilada com 20 ms de atraso e 20 ms de filtro cai em 0,1 s; a professora lenta
  (`m4_slow_teacher.py`) parte da MLP sem latência e sobe a latência aos poucos no PPO.
- DAgger com a MLP: com o erro cru e a referência fixa, o aluno piora quando β cai (o erro é
  dominado pelo chacoalhar, e a professora manda "correr para alcançar" quando ele fica para
  trás); com o erro filtrado e a referência que acompanha a mosca (`--ref-leak-tau 0.2`), ele
  melhora. Avalie sempre pelo teste sem professora (% que chega aos 5 s e velocidade), não
  pelo erro.
- Conectoma que anda (τ×0,25, ganho por ligação, `--filtered-loss --ref-leak-tau 0.2`): segue a
  velocidade pedida, mas gira em círculos (não sente rotação) e ignorava o giro pedido porque os
  DNa01/DNa02 (neurônios grandes, limiar 90–200) paravam de disparar quando o ganho de giro caía no
  treino: use `--turn-gain 1500 --cmd-lr-mult 0`. Meça com `m4_eval.py` (velocímetro e giroscópio),
  não pela distância em linha reta do teste do treino, que pune quem anda em círculos. O embaralhado
  anda tão bem quanto o real e os ganhos mudam pouco (|log ganho| médio ~0,05): com neurônios
  rápidos, a rede funciona como um reservatório com entrada e saída treinadas.
- Marcha simétrica: o conectoma que anda desviava conforme a velocidade (1 cm/s à direita, 3 cm/s à
  esquerda). Halteres (`--haltere-input`) sozinhos não bastaram; com `--mirror` (metade dos trechos
  espelhados, patas trocadas em ângulo absoluto: nas direitas, os eixos são o espelho dos das
  esquerdas, conferido pelas garras) e taxa 3× menor, 97% das retas contam. O DAgger piora depois
  de ~20 iterações (erro sobe, aluno cai): use `--ema`, pegue o best.pt e rode `m4_eval.py`.
- Treinos no Spark já travaram em `torch.save` (esperando num futex, com o latest.tmp pela metade; aconteceu
  com dois treinos ao mesmo tempo e também com um só, o evolui_b na geração 559): o monitor precisa acusar
  checkpoint parado, e `--resume` retoma. No Spark, `runs/supervise.sh EXECUÇÃO SCRIPT_DE_RETOMADA` faz isso
  sozinho (checkpoint parado há 15 min com o processo vivo: mata e retoma). Não rode `pgrep -f` com o padrão
  dentro do próprio comando ssh: ele casa com o bash do ssh e o `kill` derruba a sessão (código 127).
- M6, primeiro teste (patina_a: PPO de patins partindo do conectoma que anda): de patins, ele cai em
  55–70% das tentativas de 5 s (a "Tentativa #1" do plano). Com ganhos por ligação, a divergência KL
  passou do alvo já na primeira época em toda iteração e a taxa caiu ao piso: o grupo "sinapses"
  (1 milhão de ganhos, cada um andando ~lr por passo no Adam) precisa de multiplicador bem menor;
  meça a sensibilidade por grupo antes do treino final.
- Halteres: com `--haltere-scale 2` e o viés 2 abaixo do limiar, uma rotação de 1,3 rad/s acendia 2 a
  4 dos 203 aferentes (a mosca não sentia o giro). Use `--haltere-scale 0.5 --haltere-offset 0`.
- DNa02: meça o efeito contra a mesma caminhada sem estímulo (pareada, `mosca.walking.evaluate`);
  o sinal do giro absoluto confunde o efeito com o desvio próprio da marcha. O teste do treino é o
  próprio critério do M4 (`quick_m4`) e escolhe o best.pt; cada checkpoint tem suas manias (gira a
  uma velocidade, cai a outra), então compare vários antes de escolher.
- PPO de patins (M6): pelo `m6_sensitivity.py`, um passo de sinal de 1e-4 dá KL de 3e-3 nos ganhos
  por ligação, 1e-4 nos parâmetros por tipo celular, 3e-5 no decodificador e ~1e-7 no codificador:
  multiplicadores ~1 / 5 / 10 / 100 com taxa-base ~1e-4.
- M4 cumprido com `--syn-lr-mult 0.3` (ganhos por ligação 10× mais lentos que antes): com 3, cada
  atualização mexia demais na política e o DAgger desandava depois de ~10 iterações; com 0,3, os testes
  do critério ficaram estáveis (98%/100% na iteração 6, 92%/100% na 9). Checkpoint: runs/anda_r5I/best_it6.pt
  no Spark e no notebook.
- Determinismo da rede na GPU: a `torch.sparse.mm` (cuSPARSE) soma em ordem variável, e o conectoma
  com neurônios de 5 ms amplifica a diferença (0,07 na ação em 16 passos no começo da tentativa). Isso
  inflava a KL do PPO de patins (a atualização re-executa a política) e quebraria a re-simulação do
  cérebro na captura (M5). Use `--net-dtype float64` no PPO e na captura (a multiplicação fica 2 a 3×
  mais lenta). O Triton do venv não compila no Spark (faltam os cabeçalhos do Python do sistema).
- Captura do treino final: `--capture --deterministic --net-dtype float64`. O modo determinístico põe
  taxas e pesos numa grade binária antes da multiplicação esparsa (somas exatas em float64, em qualquer
  ordem; folga de 12× com os pesos do M4, impressa no início do treino) e soma o decodificador por
  multiplicação densa: a re-simulação do cérebro (`m5_rehearsal.py`, a leva inteira no mesmo lote) dá
  diferença 0 nas ações. Só em float64 ela ficou a ~2e-6 em tentativas de 5 s (a rede não amplificou o
  erro como se temia). O cérebro só é re-simulável na mesma máquina da coleta.
- KL do PPO de patins: `kl_fim` é a do minilote em que a época parou (passou de 2× o alvo) e salta por
  ser estimada em 16 trechos; a média (`kl` no metrics.jsonl) fica em 0,005–0,016. Com passos coerentes a
  KL cresce com o quadrado do nº de passos, e a taxa cai sozinha até caber uma época (~7e-6 com os
  multiplicadores atuais). A norma do gradiente do ator antes do corte é ~1e8.
- Teto linear para imitar a MLP patinadora (`m6_bc_probe.py --ceiling`: conectoma que anda com as
  observações da MLP do M2, 1 − R² filtrado no teste): as ações dela são quase lineares nos sentidos das
  patas (0,005). Lendo os motores: todos, livre, 0,25; só os da própria pata 0,74; pelo decodificador
  anatômico (músculo → junta, sinal fixo) 0,88 (com qualquer sinal, 0,84). A informação está nos motores
  das outras patas: o treino teria de reorganizar quais motores disparam em cada pata, e por isso o PPO
  não tira a política média do ponto fixo. A imitação offline ponta a ponta (taxa 3e-3) cai de 31 e para
  em ~2 depois de 17 épocas (melhor 1,96): nem treinado ele imita a MLP melhor que a média das ações (no
  M4, 0,27). De patins, o conectoma que anda fica parado num ponto fixo (rede ativa, ~22 Hz nos motores),
  também com controle a 2 ms; o ruído de 1,0 derruba ~20% das tentativas (0,6: 2–3%).
- Patins travados (`--glide-friction 1`, atrito ao longo = de lado): o conectoma que anda também fica
  parado (0,07–0,09 cm/s, a 10 ms ou a 2 ms de controle). A caminhada do M4 não passa para o corpo de
  patins (botas no lugar de tarsos e garras, sem adesão), então soltar os patins aos poucos não parte de
  um andar.
- Estratégias evolutivas (`m6_es.py`, OpenAI-ES sobre 14 mil parâmetros: fatores por tipo celular, tônus,
  vieses do codificador e decodificador anatômico; ganhos por ligação fixos; cada ambiente do lote é uma
  variação) tiram o conectoma do ponto fixo onde o PPO travou: com σ×2 (`--sigma-type 0.1 --sigma-tone 2
  --sigma-enc 1 --sigma-dec-raw 0.2 --sigma-dec-bias 0.1`), a média da população anda 1,6 cm/s pedindo 2
  já na geração 9 (evolui_a). Comece o currículo em `--v-start 2`: pedindo 0,3–1 cm/s, ficar parada paga
  parte da recompensa de velocidade e as variações evoluem para ficar em pé melhor.
- evolui_b (a partir da geração 11 do evolui_a, `--lr 0.05`): com `--lr 0.3` a média fazia um passeio aleatório
  (64 amostras em 14 mil dimensões) e saía do ponto que andava. O currículo precisa ser julgado pela avaliação
  da média (a população perturbada acerta 5–9%), e o `--yaw-filtered` tirou a mosca dos círculos. Na geração
  365 cumpre o critério do M2 (3,5 cm/s pedidos: 3,24 de média, 95% ≥3, 45% deslizando, 0% de quedas), mas
  com ~6,6× os passos da MLP. Pedindo 0 cm/s ela patinava a 1,5 cm/s até entrar `--p-stand 0.3` com testes de
  parada no acerto do currículo (e o ganho do DNg100 entre os parâmetros evoluídos): aí para em 75–88%.
- Para calar um neurônio com controle a 10 ms, inibir com corrente muito negativa (`m6_eval.py --probes`):
  zerar a taxa só no fim de cada passo deixa o neurônio disparar nos 5 subpassos de RK4 seguintes (com isso,
  o DNg100 "calado" ainda dava 2,4–2,6 cm/s; inibido, 0,29 cm/s contra 3,52, na geração 448 do evolui_b).
