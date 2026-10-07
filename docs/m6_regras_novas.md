# M6: teste das regras novas (registrado em 05/10/2026, antes de ver os resultados)

## Por quê

O treino final (`final_s0`) cumpriu o critério do M2, mas não patina como um animal faria. A análise da marcha
(`scripts/m8_gait.py`) mostra que ele desliza sobre dois patins do lado direito, chuta com a pata do meio esquerda e
mantém a frente e a trás esquerdas no ar. Os outros controladores também inventaram marchas com patas no ar:

- **Conectoma embaralhado:** anda de lado, em círculos.
- **MLP:** desliza sobre três patins.

A causa provável é a recompensa. O rolamento era a média só dos patins apoiados, e o deslize contava quando todos os
apoiados rolavam: levantar um patim que freia ou empurra aumentava os dois. Levantar e sacudir pata também não custava
nada, porque o custo de energia do plano nunca foi ligado.

## Regras novas (sem impor gesto nem simetria)

- `--count-lifted`: o rolamento vira a média dos seis patins (patim no ar vale 0), e o deslize só conta com os seis no
  chão, rolando.
- `--w-cot`: o custo de transporte (energia por distância) do plano. Na marcha atual ele vale ~3; na MLP, ~0,5.

## Testes, todos com a receita do treino final e sem captura

| Execução | Controlador | Regras | Duração |
|---|---|---|---|
| mlp_c1 | MLP (PPO) | count-lifted | 400 iterações |
| mlp_c2 | MLP (PPO) | count-lifted + w-cot 0,2 | 400 iterações |
| mlp_c3 | MLP (PPO) | count-lifted + w-cot 0,5 | 400 iterações |
| evolui_o | conectoma (estratégias evolutivas, do anda_r5I) | count-lifted + w-cot 0,2 | 500 gerações |
| evolui_p | conectoma (estratégias evolutivas, do anda_r5I) | count-lifted + w-cot 0,5 | 500 gerações |

## Como se mede a marcha

Pelo `m8_gait.py`, no teste fixo 1 pedindo 3,5 cm/s, mais os testes do `m7_eval.py`. Uma marcha conta como natural
quando cumpre todos estes critérios:

1. **Patins no chão:** cada patim fica no chão em ≥50% do tempo (nenhuma pata no ar o tempo todo), e a média é ≥75%.
2. **Empurrão dividido:** nenhum patim faz mais de 60% do empurrão, e cada lado faz ≥25%.
3. **Desliza de verdade:** ≥25% do tempo deslizando com os seis patins no chão.
4. **Anda e não cai:** o critério do M2 em velocidade e quedas (≥80% dos testes com média ≥3 cm/s pedindo 3,5; <10% de
   quedas).
5. **Reto quando pedido reto:** |giro médio| <0,3 rad/s no teste fixo.

## Regra de decisão

- O novo treino final usa as regras da variante de conectoma (evolui_o ou evolui_p) que cumprir mais critérios na
  geração 500. No empate, fica a de custo de energia menor (0,2).
- Se nenhuma cumprir os critérios 1 a 3, a decisão volta ao usuário: as opções seguintes são o patim com a ponta para
  fora (alavanca do plano) ou ensinar o gesto.
- As MLPs servem de diagnóstico: mostram se a tarefa, com as regras novas, leva uma rede livre a uma marcha natural.

## Desvio do teste (05/10/2026, 15h30, antes de qualquer resultado de conectoma)

Situação na geração ~70:

- **Os dois conectomas com custo de energia estão parados:** 0,10 cm/s no evolui_o (custo 0,2) e 0,15 cm/s no evolui_p
  (custo 0,5), pedindo 2. Na mesma altura, o treino final já andava a 1,2 cm/s.
- **As MLPs com custo de energia (mlp_c2 e mlp_c3) também ficam paradas quando se pede 3,5.**
- **A MLP só com `--count-lifted` (mlp_c1) já patina:** 3,44 cm/s pedindo 3,5, os seis patins no chão 89–97% do tempo,
  empurrão dividido entre as duas patas do meio (41% esquerda, 58% direita), reto.

O plano original dizia que o custo de transporte "só entra depois do primeiro movimento", e o teste o ligou desde a
geração 0. Por isso o evolui_p (custo 0,5) foi parado na geração ~70 e substituído pelo **evolui_q**: a mesma receita,
só com `--count-lifted`. O custo de energia poderá entrar depois do primeiro movimento, numa retomada registrada. O
evolui_o continua.

A regra de decisão passa a comparar evolui_o e evolui_q, com os mesmos critérios 1 a 5. No empate, fica o evolui_q, o
de regra mais simples.

## Segundo desvio (05/10/2026, 19h50)

Resultados das MLPs na iteração 400:

- **mlp_c3 (custo de energia 0,5):** pedindo 2 cm/s, é a marcha natural. Os seis patins ficam no chão 92–100% do tempo,
  as duas patas do meio dividem o empurrão (36% e 47%), desliza 99% e anda reto. Mas não passou de 2 cm/s no currículo
  e, pedindo 3,5, fica parada.
- **mlp_c1 (só `--count-lifted`):** chega a 4,1 cm/s pedindo 3,5, mas voltou a levantar as patas do meio para remadas
  longas (meio D só 16% no chão). Desliza com os seis no chão só 2% do tempo, contra 70% na iteração 125.

Nos conectomas:

- **evolui_o (custo de energia desde a geração 0):** continuou parado (0,21 cm/s na geração 226) e foi interrompido.
- **evolui_q:** anda (2,5–2,9 cm/s pedindo 2), mas com o mesmo chute da pata do meio esquerda (77% do empurrão), fazendo
  curva (−0,95 rad/s) e quase sem deslizar com os seis.

Ramificações abertas, com o custo de energia entrando depois do primeiro movimento, como o plano previa:

- **evolui_q2:** o evolui_q a partir da geração 164, com `--w-cot 0.2`.
- **mlp_c1b:** a mlp_c1 a partir da iteração 125 (quando ainda patinava com os seis no chão), com `--w-cot 0.2`.

O evolui_q continua sem custo, como controle. A regra de decisão passa a comparar evolui_q e evolui_q2.

## Terceira ramificação (05/10/2026, 20h)

O usuário achou a marcha da mlp_c3 (pedindo 2 cm/s) "muito mais natural". Para testar o conectoma na mesma condição,
foi aberta a **evolui_q3**: o evolui_q a partir da geração 219, com o custo de energia da mlp_c3 (`--w-cot 0.5`) e
velocidade-alvo de 2,5 cm/s (`--v-final 2.5`, contra 4 na receita final).

Baixar a velocidade-alvo mudaria o critério do M2 (≥3 cm/s pedindo 3,5), que é decisão do usuário: esta ramificação só
dá o dado. A regra de decisão compara evolui_q, evolui_q2 e evolui_q3 pelos critérios 1 a 5. Na q3, os critérios de
velocidade (4) são lidos à velocidade-alvo dela (2,5 pedidos, ≥2 cm/s de média) e ficam marcados como tal.

## Resultado (06/10/2026)

Medido pelo `m8_gait.py` no teste fixo 1 e pelo `m7_eval.py`:

| Execução | Critério 1 (patins no chão) | 2 (empurrão dividido) | 3 (desliza com os seis) | 4 (M2) | 5 (reto) |
|---|---|---|---|---|---|
| evolui_q (geração 500) | não: meio D no ar o tempo todo, média 49% | sim (meio E 52%; esquerda 68%, direita 32%) | não (0%) | sim (3,14 cm/s, 100% ≥3) | sim (−0,11 rad/s) |
| evolui_q2 (geração 500) | não | não | não | não (parada, 0,2 cm/s) | — |
| evolui_q3 (geração 500) | não | não | não | não (parada, 0,2 cm/s) | — |
| mlp_c1b (MLP, iteração 400) | sim (66–95%, média 84%) | sim (meio D 58%, meio E 32%) | sim (41%) | 3,11 cm/s no teste fixo | sim (−0,12 rad/s) |

- **O custo de energia faz o conectoma parar,** desde a geração 0 (evolui_o) ou depois do primeiro movimento (evolui_q2 e
  q3). Na MLP isso não acontece: com o mesmo custo, ela mantém a marcha natural e faz curvas de ±1 rad/s com erro de
  0,15 rad/s.
- **Só com `--count-lifted`, o conectoma real volta ao chute com a pata do meio esquerda,** agora com mais patins no
  chão. É a quinta vez que ele chega a esse gesto.
- **Nenhuma variante de conectoma cumpre os critérios 1 a 3.** Pela regra registrada, a decisão volta ao usuário.

## Regra de apoio (registrada em 06/10/2026, antes de rodar; escolhida pelo usuário)

Termo `--w-contact`: a cada passo, paga a fração dos seis patins que estão no chão. Ele já existia no ambiente (peso 0)
e passa a ser exposto no `m6_es.py`. Segue `--count-lifted`, sem custo de energia.

- Diferente do custo de energia, este termo não pune o movimento: ficar parada com os seis no chão rende `w` por passo, e
  patinar como o evolui_q rende ~2,6 + 0,55·`w`. Mexer-se continua valendo mais enquanto `w` < ~5.
- Na geração 500 do evolui_q, 55% dos patins estavam no chão, e esse número não mudou da geração 100 à 500.

| Execução | Parte de | Regra de apoio | Duração |
|---|---|---|---|
| evolui_r | evolui_q, geração 500 | `--w-contact 1.0` | até a geração 1000 |
| evolui_s | evolui_q, geração 500 | `--w-contact 2.0` | até a geração 1000 |
| evolui_t | do zero (anda_r5I) | `--w-contact 1.0` | 500 gerações |

**Critérios e regra de decisão:**

- Os mesmos critérios 1 a 5. Segue para o novo treino final a variante que cumprir mais critérios no fim (no empate, a de
  `w` menor).
- Se nenhuma cumprir os critérios 1 a 3, a decisão volta ao usuário: ensinar o gesto, patim com a ponta para fora ou
  usar o que temos.

## Resultado da regra de apoio (07/10/2026)

No teste fixo 1, pedindo 3,5 cm/s, e pelo `m7_eval.py` (critério 4 = velocidade e quedas do M2):

| Execução | 1: patins no chão (média) | 2: empurrão dividido | 3: desliza com os seis | 4: velocidade e quedas | 5: reto | Critérios |
|---|---|---|---|---|---|---|
| evolui_r (`w` 1,0, geração 1000) | não (68%, trás E 40%) | sim (máx. 37%; esq. 49%) | não (0%) | sim (95% ≥3 cm/s) | não (−0,32) | 2 |
| evolui_s (`w` 2,0, geração 1000) | não (73%, trás E 39%) | sim (máx. 40%; esq. 53%) | não (2%) | não (60% ≥3 cm/s) | sim (−0,25) | 2 |
| evolui_t (`w` 1,0, do zero, geração 500) | não (59%) | não (meio E 61%) | não (0%) | — | sim (−0,04) | 1 |

- **Comparação com a geração 500 do evolui_q:**
  - a fração de patins no chão subiu de 49% para 68–73%;
  - o empurrão deixou de vir quase todo da pata do meio esquerda, e nenhum patim passa de 40%;
  - o deslize com os seis continua perto de zero, e a velocidade caiu um pouco.
- **Pela regra registrada, nenhuma variante cumpre os critérios 1 a 3, e a decisão volta ao usuário.** No empate, a
  melhor é a evolui_r, a de `w` menor.
- **Uma diferença que pode explicar o resultado:** todas as MLPs treinaram com o empurrão inicial que some
  (`push_iters` 200 no `m2_train.py`: até 50% das tentativas começam já andando, e a fração cai a zero em 200
  iterações). O `m6_es.py` nunca usou essa ajuda.
  - Com o empurrão, a mosca começa deslizando com os seis patins no chão, e o treino descobre cedo que vale manter essa
    postura.
  - Partindo do repouso, o conectoma nunca visita esse estado (deslize com os seis em 0%), e o bônus correspondente nunca
    entra.

## Empurrão inicial (registrado em 07/10/2026, antes de rodar; escolhido pelo usuário)

**O que é.** `m6_es.py --push-gens N`: até `--push-max` dos pares de variações (padrão 0,5) começa a tentativa já
andando na velocidade pedida. A fração cai linearmente a zero em N gerações, contadas a partir de `--push-from`. É a mesma
ajuda que todas as MLPs tiveram (`push_iters` 200 no `m2_train.py`), e está no plano ("empurrão inicial... que some com o
tempo").

**O que não muda.**
- As avaliações e os testes continuam partindo do repouso.
- As tentativas com empurrão ficam marcadas no registro (campo `push`), não contam para marcos e recordes, e no vídeo
  aparecem rotuladas.
- O empurrão entra na captura. Um teste com captura e empurrão re-simulou 8 tentativas com física bit a bit e cérebro
  com diferença 0.

**Variantes**, todas com `--count-lifted --w-contact 1.0`, sem custo de energia:

| Execução | Parte de | Empurrão | Duração |
|---|---|---|---|
| evolui_u | evolui_r, geração 1000 | `--push-gens 200 --push-from 1000` | até a geração 1400 |
| evolui_v | do zero (anda_r5I) | `--push-gens 200` | 500 gerações |

**Critérios e regra de decisão.** Os mesmos critérios 1 a 5. Segue a variante que cumprir mais critérios no fim (no
empate, a evolui_v, que parte do zero). Se nenhuma cumprir os critérios 1 a 3, a decisão volta ao usuário.
