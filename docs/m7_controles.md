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
