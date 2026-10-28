# M7: controles (registrado antes de rodar a patinação, 02/10/2026)

## Pergunta

Com a mesma receita e o mesmo orçamento, o conectoma real aprende a patinar melhor ou mais rápido que o
conectoma embaralhado (mesmos neurônios, mesmo nº de ligações e sinais por neurônio, fiação sorteada:
`scripts/m7_shuffle_graph.py --seed 0`)? E uma rede comum (MLP), treinada pelo método com que ela patinou no M2?
O vídeo mostra o resultado que sair, inclusive empate.

## Braços

### Conectoma real e embaralhado: a mesma receita do começo ao fim

**Etapa A, andar (destilação, já rodando antes deste registro):**

- uma execução por grafo, semente 0;
- receita fixada em 28/09/2026 nos scripts de lançamento: `m7_real_s1` / `m7_emb_s1` (20 iterações a partir do
  zero) e `m7_real_s2` / `m7_emb_s2` (a receita do anda_r5I, 30 iterações a partir do best.pt da etapa 1);
- a etapa 2 foi interrompida na iteração 7 (para liberar o Spark) e retomada em 02/10/2026 até as 30 iterações,
  sem mudar nada;
- a patinação parte do best.pt da etapa 2 (maior pontuação no teste sem professora do próprio treino), ande ele
  ou não.

Na iteração 6, nenhum dos dois cumpria o critério do M4. O real ficava parado (0,02 cm/s). O embaralhado andava a
~2 cm/s, mas girando (+0,6 a +0,9 rad/s), e calar o DNg100 o deixava a 0,5 cm/s.

**Etapa B, patinar (estratégias evolutivas):**

- a receita do treino final (`docs/m6_treino_final.md`), já com os dois desvios valendo desde a geração 0
  (rolamento 1,5, deslize 0,5, giro 1,0 condicionado à velocidade);
- sementes 0, 1 e 2;
- 1000 gerações (64 mil tentativas por execução), sem captura, checkpoints a cada 25 gerações.

```
m6_es.py --run m7p_{real,emb}_sN --seed N --init-from runs/m7_{real,emb}_s2/best.pt [--graph <embaralhado>] \
  --deterministic --net-dtype float64 --envs 64 --generations 1000 --eval-every 5 \
  --sigma-type 0.1 --sigma-tone 2 --sigma-enc 1 --sigma-dec-raw 0.2 --sigma-dec-bias 0.1 \
  --lr 0.05 --lr-warm 0.3 --lr-warm-gens 10 \
  --v-start 2 --v-final 4 --phase2-gen 30 --p-stand 0.3 --yaw-start 0.3 --yaw-final 1.5 \
  --turn-side-gains --yaw-filtered --w-roll 1.5 --w-glide 0.5 --w-yaw 1.0 --yaw-gated
```

### MLP: PPO, a partir do zero

- a receita da execução E do M2 (`m2_train.py`): ruído correlacionado, desvio inicial 0,6, empurrão inicial que
  some em 200 iterações;
- a mesma tarefa do treino final: os mesmos pesos da recompensa, tentativas paradas e curvas a partir da
  iteração 30, currículo de 2 a 4 cm/s e depois curvas até ±1,5 rad/s;
- sementes 0, 1 e 2; 1000 iterações (64 mil tentativas, o mesmo orçamento de tentativas).

```
m2_train.py --run m7p_mlp_sN --seed N --iters 1000 --envs 64 --init-std 0.6 \
  --v-start 2 --v-final 4 --v-step 0.5 --phase2-gen 30 --p-stand 0.3 \
  --yaw-start 0.3 --yaw-final 1.5 --yaw-step 0.25 --yaw-filtered --yaw-gated \
  --w-vel 1 --w-yaw 1.0 --sigma-yaw 0.5 --w-up 0.1 --w-roll 1.5 --sigma-roll 0.5 --w-slip 0.2 \
  --w-rate 0.01 --w-leg-floor 0.1 --w-glide 0.5
```

Diferenças declaradas:

- a MLP não passa pela etapa de andar, porque no M2 ela aprendeu a patinar do zero;
- o PPO julga o currículo pelas tentativas de treino, como no M2; as estratégias evolutivas o julgam pela
  avaliação da média, porque a população perturbada acerta pouco.

### Fora deste registro

O grafo aleatório do mesmo tamanho e o BANC ficam de fora deste registro. Se rodarem, terão registro próprio
antes.

## Métricas (`scripts/m7_eval.py`, nos checkpoints a cada 50 gerações ou iterações)

As mesmas 20 tentativas de teste do M2 (sementes 10⁹ + 0…19), com a ação média da política e a partir do repouso:

1. **Primária: primeira geração em que cumpre o critério do M2.** Pedindo 3,5 cm/s: ≥80% com média ≥3 cm/s em 5 s,
   ≥25% do tempo deslizando e <10% de quedas. Quem não cumpre até a geração 1000 fica empatado em "não cumpriu".
2. **Curvas.** Pedindo 3 cm/s, com giro de ±0,5 rad/s (metade para cada lado):
   - diferença entre o giro médio dos dois lados (ideal 1,0 rad/s);
   - acerto: sem cair, até o fim, erro médio de giro abaixo de 0,25 rad/s (metade do pedido; quem não gira fica
     com ~0,5) e velocidade a menos de 25% da pedida.
3. **Parada.** Pedindo 0 cm/s: fração sem cair e com |velocidade| <0,5 cm/s.
4. **Na geração 1000, só nos conectomas**: as provas causais do `m6_eval.py --probes` (DNg100 calado; DNa02 de um
   lado).
5. **Curvas de aprendizado**: nível do currículo e avaliação da média (velocidade, deslize, acerto) por geração,
   do metrics.jsonl de cada execução.

## Como se lê

As três sementes de cada braço aparecem todas, sem escolher.

- **Diferença declarada só com separação completa na métrica primária.** As 3 sementes de um braço cumprem o M2
  antes de todas as 3 do outro (ou só elas cumprem). Sem diferença real, isso acontece por acaso com
  probabilidade 1/10 (1/20 para cada lado).
- **Qualquer outro padrão é empate.**
- As métricas 2 a 5 são descritivas.

## Resultado da etapa A (02/10/2026, antes de qualquer resultado da patinação)

As duas caminhadas terminaram as 30 iterações; a patinação parte do best.pt de cada uma, como registrado.

| Grafo | best.pt | Retas de 5 s (1 / 2 / 3 cm/s) | Velocidades | DNa02 lado certo | Quedas |
|---|---|---|---|---|---|
| Real | iteração 3 | 0% (0% / 0% / 0%) | 0,02 / 0,02 / 0,03 cm/s | 38% | 8% |
| Embaralhado | iteração 27 | 42% (94% / 0% / 31%) | 0,91 / 0,00 / 1,22 cm/s | 94% (efeito ~0) | 2% |

Com esta receita de dois estágios, o conectoma real não aprendeu a andar e o embaralhado andou em parte. O real que
anda (anda_r5I, M4) saiu de uma sequência mais longa de etapas, que o embaralhado não teve. A comparação da
patinação mede a receita inteira a partir do zero, não a melhor caminhada que cada grafo consegue.

## Braço extra: linhagem longa da caminhada (registrado em 02/10/2026, antes de rodar)

Motivo: com a receita de dois estágios, o real não andou e o embaralhado andou em parte, então a comparação acima
mistura o andar com o patinar. Este braço compara os dois grafos partindo da caminhada mais trabalhada de cada um.
Os 9 treinos acima continuam como registrados.

**Etapa A (andar), só no embaralhado.** A mesma sequência de etapas que produziu o anda_r5I, com o mesmo checkpoint
de partida em cada uma:

| Etapa | Parte de | Iterações |
|---|---|---|
| anda_r5L_emb | do zero (já existe, mesma receita do r5L) | — |
| anda_r5H_emb | it00020 do r5L_emb | 80 |
| anda_r5M_emb | best.pt do r5H_emb | 80 |
| anda_r5D_emb | best.pt do r5M_emb | 20 (só até o it00020, o que a etapa seguinte usa) |
| anda_r5I_emb | it00020 do r5D_emb | 40, e a patinação parte do best.pt |

- Os argumentos de cada etapa saem do config.json da etapa real, trocando só execução, grafo, ponto de partida e nº
  de iterações. `runs/m7l_cmd.py` confere que o parser devolve os mesmos argumentos.
- As opções criadas depois das etapas reais ficam no valor padrão, que reproduz o comportamento de então.
- Roda com o código de hoje; as etapas reais rodaram com o código de 26 a 28/09.

**Etapa B (patinar).** A mesma receita e o mesmo orçamento dos braços acima (1000 gerações, sementes 0, 1 e 2):

- `m7l_real_sN` parte de `runs/anda_r5I/best_it6.pt`, o conectoma que anda do M4;
- `m7l_emb_sN` parte de `runs/anda_r5I_emb/best.pt`.

**Métricas e regra de leitura:** as mesmas (`m7_eval.py`; diferença só com separação completa das 3 sementes na
métrica primária). Se o embaralhado não andar no fim da etapa A, a patinação roda assim mesmo, como nos braços acima.

## Pausa (05/10/2026, decidida pelo usuário)

A análise da marcha (`scripts/m8_gait.py`) mostrou que nenhum braço patina como um animal faria:

- **Conectoma real:** desliza sobre dois patins de um lado e chuta com a pata do meio esquerda, com duas patas no ar.
- **Embaralhado:** anda de lado, em círculos.
- **MLP:** desliza sobre três patins, com três no ar.

A causa provável é a recompensa. O rolamento e o deslize contavam só os patins apoiados, então levantar patins
compensava. A receita vai ser corrigida (patins no ar passam a contar, custo de energia do plano ligado) e testada
antes de um novo treino final.

Como a comparação do vídeo precisa usar a mesma receita do treino final, o M7 foi pausado com os checkpoints guardados:

| Treino | Geração em que parou |
|---|---|
| m7p_real_s2 | 748 |
| m7p_emb_s2 | 307 |
| m7l_real_s1 | 31 |
| m7l_emb_s1 | 26 |
| m7p_mlp_s2, m7l_real_s2, m7l_emb_s2 | não começaram |

Os treinos já terminados ficam como resultado da receita antiga. Com a receita nova, o M7 será registrado de novo antes
de rodar.

## Novo registro com a receita final (24/10/2026, antes de rodar)

**Por quê.** O M7 foi pausado porque a receita antiga premiava patins no ar. A receita final agora é a do segundo
treino final gravado (`docs/m6_treino_final2.md`), em três fases fixas até a geração 1400:

- gerações 0–499: patins no ar contam;
- a partir da 500: entra a regra de apoio;
- a partir da 1000: entra o empurrão inicial.

A comparação do vídeo usa essa receita.

**Braços** (3 sementes cada; 1400 gerações ou iterações, 89.600 tentativas por execução, sem captura):

| Braço | Parte de | Execuções |
|---|---|---|
| Conectoma real | `runs/anda_r5I/best_it6.pt`, o conectoma que anda (M4) | semente 0: o próprio `final2_s0`; sementes 1 e 2: `m7f_real_s1` e `m7f_real_s2` |
| Conectoma embaralhado | `runs/anda_r5I_emb/best.pt`, a mesma linhagem de caminhada no grafo embaralhado (`controller_graph_shuffled_s0_min5.npz`) | `m7f_emb_s0`, `m7f_emb_s1` e `m7f_emb_s2` |
| MLP (PPO, do zero) | — | `m7f_mlp_s0`, `m7f_mlp_s1` e `m7f_mlp_s2` |

- **Os dois conectomas usam os argumentos do `final2_s0`.** Mudam só a semente, o ponto de partida e o grafo. As fases
  trocam nas gerações 500 e 1000, como no treino gravado.
- **A semente 0 do real é o próprio treino gravado.** A receita é idêntica e o treino é determinístico, então uma
  execução separada daria os mesmos números.
- **A MLP usa o método com que patinou no M2:** ruído correlacionado, desvio inicial 0,6 e empurrão inicial nas
  primeiras 200 iterações. A tarefa é a mesma: `--count-lifted` desde o começo e `--w-contact 1.0` a partir da
  iteração 500.

```
m2_train.py --run m7f_mlp_sN --seed N --iters 500 --envs 64 --threads 4 --torch-threads 2 --init-std 0.6 \
  --v-start 2 --v-final 4 --v-step 0.5 --phase2-gen 30 --p-stand 0.3 \
  --yaw-start 0.3 --yaw-final 1.5 --yaw-step 0.25 --yaw-filtered --yaw-gated \
  --w-vel 1 --w-yaw 1.0 --sigma-yaw 0.5 --w-up 0.1 --w-roll 1.5 --sigma-roll 0.5 --w-slip 0.2 \
  --w-rate 0.01 --w-leg-floor 0.1 --w-glide 0.5 --count-lifted
# na iteração 500, retomada com --w-contact 1.0 --iters 1400 (registrada em resumes.jsonl)
```

**Diferenças declaradas:**

- A MLP não passa pela etapa de andar.
- O empurrão inicial da MLP vem no começo (iterações 0–199), como no M2. O das estratégias evolutivas entrou na geração
  1000 só porque foi achado depois.
- O PPO julga o currículo pelas tentativas de treino. As estratégias evolutivas o julgam pela avaliação da média.

**Fora deste registro:**

- **Os braços da receita curta de caminhada (`m7p_*`).** Neles o real não andou, então a comparação misturaria o andar
  com o patinar.
- **Os resultados da receita antiga** (`m7p_*` e `m7l_*`) ficam como histórico, e os treinos pausados não são retomados.
- **O grafo aleatório e o BANC** continuam de fora.

**Métricas e regra de leitura.** As mesmas do registro de 02/10 (`m7_eval.py` a cada 50 gerações ou iterações), com a
primária contada até a geração 1400:

1. **Primária:** primeira geração em que cumpre o critério do M2.
2. **Curvas.**
3. **Parada.**
4. **Provas causais** na geração 1400, só nos conectomas.
5. **Curvas de aprendizado.**
6. **Nova, descritiva:** os 5 critérios de marcha natural (`m8_gait.py`, teste fixo 1) na geração 1400.

A diferença só é declarada com separação completa das 3 sementes na métrica primária. Qualquer outro padrão é empate.

**No Spark.** Uma fila roda no máximo 3 treinos ao mesmo tempo, contando o `final2_s0`, nesta ordem:

> emb_s0, real_s1, emb_s1, real_s2, emb_s2, mlp_s0, mlp_s1, mlp_s2

Cada treino tem o próprio script de vigia, como o do `final2_s0`:

- escolhe a fase pela geração do checkpoint;
- mata e retoma se o checkpoint ficar parado 15 min;
- relança se o processo cair sem avançar.

Duração prevista: ~3–4 dias.

## Resultado (28/10/2026)

Os 9 treinos foram até a geração (ou iteração) 1400 com a receita registrada. As únicas retomadas foram as trocas de fase
previstas (gerações 500 e 1000; na MLP, iteração 500), registradas no resumes.jsonl de cada execução. As medidas foram
feitas no Spark:

- `m7_eval.py` a cada 50 gerações ou iterações;
- na 1400, as provas causais (`m6_eval.py --probes`) e a marcha (`m8_gait.py`, teste fixo 1).

As tabelas saem de `scripts/m7_summary.py`, e a cena do vídeo, de `scripts/m8_m7_video.py`.

### Métrica primária: primeira avaliação no critério do M2

| Braço | Semente 0 | Semente 1 | Semente 2 |
|---|---|---|---|
| Real | 500 (#32.000) | 800 (#51.200) | 650 (#41.600) |
| Embaralhado | 250 (#16.000) | 1150 (#73.600) | 1200 (#76.800) |
| MLP | 100 (#6.400) | 100 (#6.400) | 100 (#6.400) |

Entre parênteses, as tentativas de treino até a avaliação (64 por geração ou iteração). Pela regra registrada:

- **Real × embaralhado: empate.** As sementes se intercalam: o embaralhado tem a mais rápida (250) e as duas mais
  lentas (1150 e 1200).
- **A MLP cumpre antes dos dois conectomas, com separação completa.** As 3 sementes cumprem na iteração 100, a segunda
  avaliação (6.400 tentativas). A primeira semente de cada conectoma só cumpre com 16.000 tentativas (embaralhado) e
  32.000 (real).

**O que a diferença mede.** A comparação é entre receitas inteiras, com as diferenças declaradas no registro:

- a MLP treina por PPO, do zero, com o empurrão inicial nas iterações 0–199;
- os conectomas treinam por estratégias evolutivas, partindo da caminhada, com o empurrão só a partir da geração 1000.

O resultado não separa a rede do método de treino.

### Descritivas

**Na geração (ou iteração) 1400** (`m7_eval.py`):

| Execução | M2 | ≥3 cm/s | Velocidade | Desliza | Quedas | Giro pedindo +0,5 / −0,5 rad/s | Acerto nas curvas | Parada |
|---|---|---|---|---|---|---|---|---|
| final2_s0 | sim | 100% | 3,26 cm/s | 49% | 0% | −0,10 / −0,09 | 0% | 100% |
| m7f_real_s1 | não | 40% | 3,00 cm/s | 49% | 0% | −0,24 / −0,24 | 0% | 100% |
| m7f_real_s2 | não | 100% | 3,26 cm/s | 16% | 0% | +0,10 / +0,18 | 0% | 100% |
| m7f_emb_s0 | sim | 100% | 3,59 cm/s | 69% | 0% | +0,57 / +0,26 | 0% | 100% |
| m7f_emb_s1 | sim | 100% | 3,74 cm/s | 72% | 0% | +0,28 / +0,09 | 0% | 100% |
| m7f_emb_s2 | sim | 100% | 3,42 cm/s | 52% | 0% | −0,35 / −0,62 | 50% | 100% |
| m7f_mlp_s0 | sim | 100% | 3,20 cm/s | 58% | 0% | +0,53 / −0,20 | 50% | 100% |
| m7f_mlp_s1 | sim | 100% | 3,44 cm/s | 73% | 0% | +1,01 / −0,16 | 0% | 100% |
| m7f_mlp_s2 | não | 0% | 1,04 cm/s | 58% | 0% | −0,68 / −1,14 | 0% | 100% |

- **Critério do M2 na 1400:** o embaralhado cumpre nas 3 sementes, a MLP em 2 e o real só na semente 0. No real, a
  semente 1 passa de 3 cm/s em 40% dos testes, e a 2 desliza 16% do tempo. A MLP s2 anda a 1,04 cm/s, girando para a
  direita.
- **Ao longo do treino** (tabela mais abaixo):
  - o embaralhado s0 cumpre da geração 250 até o fim (24 das 28 avaliações), e os outros conectomas entram e saem do
    critério;
  - a MLP s1 cumpre da iteração 100 até o fim (27 de 28), e a s0 de 100 a 800 e de novo na 1350–1400;
  - a MLP s2 só cumpre nas iterações 100–150 e 250–300. Depois desanda, com quedas de 15% a 90% entre as iterações 450
    e 1050.

**Curvas.**

- **Conectomas:** nenhum vira quando pedido. Só o embaralhado s2 virou para os dois lados, numa única avaliação
  (geração 850: +0,49 / −0,27 rad/s, acerto 55%), e perdeu isso depois. Os acertos de até 50% dos outros vêm de virar
  bem para um lado só.
- **MLP:** vira para os dois lados em parte das avaliações (s0: 10 de 28; s1: 11 de 28; s2: nenhuma), mas não fica. Na
  1400, a s0 e a s1 viram para a esquerda e quase nada para a direita (+0,53 / −0,20 e +1,01 / −0,16 rad/s).

**Parada.** Na 1400, todos param quando pedido em 100% dos testes. O real perdeu a parada no começo do treino e a
recuperou:

- final2_s0: 20% na geração 50;
- semente 2: 0–30% entre as gerações 50 e 300.

**Provas causais (1400, só os conectomas).**

- Calar os DNg100 para os 6 (≤0,01 cm/s, contra 2,2–3,6 cm/s sem intervenção).
- O DNa02 de um lado vira a mosca para o lado estimulado em 5%, 65% e 20% dos testes no real, e em 50%, 100% e 100% no
  embaralhado. No grafo embaralhado, o DNa02 continua sendo o neurônio que recebe o comando de giro, mas com as ligações
  sorteadas.

| Execução | DNg100 calados | DNa02 esquerdo | DNa02 direito | Vira para o lado estimulado |
|---|---|---|---|---|
| final2_s0 | −0,00 cm/s (sem intervenção 3,02) | −0,28 rad/s | +0,10 rad/s | 5% |
| m7f_real_s1 | 0,00 cm/s (sem intervenção 2,20) | −0,04 rad/s | −0,07 rad/s | 65% |
| m7f_real_s2 | 0,01 cm/s (sem intervenção 3,08) | +0,01 rad/s | +0,47 rad/s | 20% |
| m7f_emb_s0 | −0,00 cm/s (sem intervenção 3,46) | −0,07 rad/s | −0,32 rad/s | 50% |
| m7f_emb_s1 | 0,00 cm/s (sem intervenção 3,64) | +0,16 rad/s | −0,17 rad/s | 100% |
| m7f_emb_s2 | 0,01 cm/s (sem intervenção 3,06) | +0,69 rad/s | −0,36 rad/s | 100% |

**Curvas de aprendizado** (metrics.jsonl e `m7_eval.py`):

- O currículo de velocidade chega a 4 cm/s nas gerações 455–595 no real e 250–1155 no embaralhado. Na MLP, chega na
  iteração 62 nas sementes 0 e 1; a semente 2 parou em 3 cm/s na iteração 53 e não passou daí.
- O de giro parou em 0,55 rad/s nas 3 sementes do real. No embaralhado, parou em 1,50, 0,55 e 0,80 rad/s; na MLP, em
  0,80, 1,30 e 0,30 rad/s.

O PPO julga o currículo pelas tentativas de treino, e as estratégias evolutivas pela avaliação da média. Por isso, os
níveis não se comparam entre os braços.

| Execução | Avaliações no critério do M2 | Viram para os dois lados | Melhor acerto nas curvas | Parada mínima | Currículo no fim (velocidade / giro) |
|---|---|---|---|---|---|
| final2_s0 | 6 de 28 | 0 de 28 | 50% (na 1250) | 20% | 4,0 cm/s, desde a 455 / 0,55 rad/s |
| m7f_real_s1 | 11 de 28 | 0 de 28 | 35% (na 1350) | 100% | 4,0 cm/s, desde a 590 / 0,55 rad/s |
| m7f_real_s2 | 3 de 28 | 0 de 28 | 15% (na 1300) | 0% | 4,0 cm/s, desde a 595 / 0,55 rad/s |
| m7f_emb_s0 | 24 de 28 | 0 de 28 | 50% (na 250) | 100% | 4,0 cm/s, desde a 250 / 1,50 rad/s |
| m7f_emb_s1 | 6 de 28 | 0 de 28 | 0% | 100% | 4,0 cm/s, desde a 1155 / 0,55 rad/s |
| m7f_emb_s2 | 5 de 28 | 1 de 28 (a 1ª na 850) | 55% (na 850) | 100% | 4,0 cm/s, desde a 635 / 0,80 rad/s |
| m7f_mlp_s0 | 17 de 28 | 10 de 28 (a 1ª na 100) | 100% (na 100) | 100% | 4,0 cm/s, desde a 62 / 0,80 rad/s |
| m7f_mlp_s1 | 27 de 28 | 11 de 28 (a 1ª na 100) | 100% (na 100) | 75% | 4,0 cm/s, desde a 62 / 1,30 rad/s |
| m7f_mlp_s2 | 4 de 28 | 0 de 28 | 40% (na 150) | 95% | 3,0 cm/s / 0,30 rad/s |

Viram para os dois lados: giro médio ≥ +0,25 rad/s pedindo +0,5, ≤ −0,25 pedindo −0,5 e acerto ≥50% (o critério de
curvas do `docs/m6_treino_final2.md`). O acerto sozinho engana: quem gira sempre para um lado acerta as curvas desse lado.

**Marcha (1400).**

| Execução | Velocidade | Giro | Desliza (com os seis) | Patins no chão (FE FD ME MD TE TD) | Maior parcela do empurrão | Critérios 1–5 |
|---|---|---|---|---|---|---|
| final2_s0 | 3,23 cm/s | +0,31 rad/s | 45% (2%) | 70 97 54 58 87 99 | meio E 35% | ✓✓✗✓✗ (3 de 5) |
| m7f_real_s1 | 3,08 cm/s | −0,14 rad/s | 48% (0%) | 0 96 89 89 91 28 | meio D 72% | ✗✗✗✗✓ (1 de 5) |
| m7f_real_s2 | 3,26 cm/s | +0,43 rad/s | 17% (0%) | 12 88 93 58 83 80 | meio D 78% | ✗✗✗✓✗ (1 de 5) |
| m7f_emb_s0 | 3,66 cm/s | +0,44 rad/s | 70% (0%) | 92 0 77 95 34 38 | meio E 65% | ✗✗✗✓✗ (1 de 5) |
| m7f_emb_s1 | 3,80 cm/s | +0,06 rad/s | 74% (0%) | 98 0 97 85 0 71 | meio D 45% | ✗✓✗✓✓ (3 de 5) |
| m7f_emb_s2 | 3,46 cm/s | −0,25 rad/s | 53% (0%) | 34 70 71 49 84 81 | meio E 36% | ✗✓✗✓✓ (3 de 5) |
| m7f_mlp_s0 | 3,19 cm/s | +0,34 rad/s | 59% (0%) | 96 34 14 83 95 15 | meio D 44% | ✗✓✗✓✗ (2 de 5) |
| m7f_mlp_s1 | 3,40 cm/s | +0,71 rad/s | 74% (0%) | 0 99 87 0 17 99 | meio E 71% | ✗✗✗✓✗ (1 de 5) |
| m7f_mlp_s2 | 1,10 cm/s | −0,85 rad/s | 59% (0%) | 0 100 100 33 11 94 | meio E 64% | ✗✗✗✗✗ (0 de 5) |

Patins no chão: % do tempo, na ordem frente, meio e trás, esquerdo (E) e direito (D). Os 5 critérios de marcha natural
(`docs/m6_regras_novas.md`):

1. patins no chão (cada um ≥50%, média ≥75%);
2. empurrão dividido (nenhum patim >60%, cada lado ≥25%);
3. desliza com os seis no chão ≥25% do tempo;
4. velocidade e quedas do critério do M2;
5. reto (|giro médio| <0,3 rad/s).

- **Ninguém patina com marcha natural.** Nenhum cumpre o critério 3 (deslizar com os seis patins no chão: 0–2% do
  tempo), e só o real s0 cumpre o 1.
- **No real s0, o critério 5 está no limite.** A mesma política deu +0,31 rad/s agora e +0,29 em 07/10 (o `final2_s0`
  reproduz a evolui_u bit a bit). O `m8_gait.py` roda a rede em float32 na GPU, fora do modo determinístico.

### No vídeo e no post

- **Cena 16:** troca o gancho ("esse treino já tá rodando") pelo resultado: o embaralhado patina tão bem quanto o real
  (empate pela regra), e a MLP aprende antes, pela receita dela. A cena mostra os testes das 9 execuções ao longo do
  treino (`outputs/v3_controles.mp4`).
- **Post:** as diferenças de receita da MLP (PPO, do zero) entram nas letras miúdas.
- **O que o vídeo não promete:** nenhum dos três braços vira quando pedido de forma estável nem patina com marcha natural.
