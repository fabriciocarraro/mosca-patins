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
