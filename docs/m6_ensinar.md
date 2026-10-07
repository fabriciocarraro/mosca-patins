# M6: ensinar o gesto (registrado em 07/10/2026, antes de rodar; escolhido pelo usuário)

## Por quê

O conectoma treinado por estratégias evolutivas nunca chegou à marcha natural (seis patins no chão, empurrões das patas
do meio, deslize), com nenhuma das regras testadas (`docs/m6_regras_novas.md`). O melhor, a evolui_u, cumpre 4 de 5
critérios e não desliza com os seis. Uma MLP com as mesmas regras chega lá.

O usuário escolheu ensinar o gesto: o conectoma imita a MLP, como imitou a política do flybody para aprender a andar no
M4, e depois evolui sozinho. **No vídeo: o gesto foi ensinado por uma IA comum que aprendeu a patinar com as mesmas
regras; o conectoma aprendeu a executá-lo com a própria fiação.**

## Professora

A **mlp_c1b** (iteração 400). Nos testes do M7, pelo `m7_eval.py`:

| Teste | Resultado |
|---|---|
| Critério do M2 | cumpre (100% ≥3 cm/s, 3,15 cm/s, sem quedas) |
| Patins no chão | 83% |
| Desliza com os seis | 42% |
| Trabalho por lado | 50% / 50% |
| Curvas pedindo ±0,5 rad/s | +0,55 / −0,54 rad/s, acerto 100% |
| Parar quando pedido | 100% |

## Viabilidade (sonda offline, `m6_bc_probe.py`)

- **Teto linear (erro filtrado, 1 − R²):**
  - sentidos crus: 0,008 (as ações dela são quase lineares nos sentidos das patas);
  - proprioceptores: 0,31;
  - motores: 0,36;
  - pelo decodificador anatômico: 0,77.
- **Imitação offline ponta a ponta (30 épocas):** 0,96, ainda caindo. Com a MLP antiga, o melhor tinha sido 2,09; na
  caminhada (M4), 0,27.
- **Leitura:** o gargalo é o decodificador anatômico. O treino tem de reorganizar quais motores disparam em cada pata,
  como no M4.

## Treino (`scripts/m6_dagger.py`)

**Montagem:**

- **Aluno:** o conectoma dos patins do `m6_es.py`, partindo do conectoma que anda (`runs/anda_r5I/best_it6.pt`), com
  ganho por ligação (sinal fixo), decodificador anatômico, τ × 0,25 e halteres. Rede em float32 no treino.
- **DAgger como no M4:**
  - β (fração conduzida pela professora) cai de 1 a 0,3 em 20 iterações;
  - intervenção da professora a cada passo, de 0,3 a 0,1 em 30 iterações;
  - trechos de 16 passos de 10 ms, com 6 de aquecimento;
  - erro depois do filtro dos atuadores;
  - taxa 3e-4; multiplicadores: sinapses 0,3, decodificador 10, codificador 10, tônus 3, comando 0;
  - média móvel dos parâmetros (0,995).
- **Pedidos:** 1–4 cm/s, 20% de pedidos de parar, curvas de até ±1 rad/s trocando a cada ~1,5 s.

**Variantes:**

| Execução | Espelhamento esquerda-direita | Iterações |
|---|---|---|
| ensina_a | sim (`--mirror`) | 60 |
| ensina_b | não | 60 |

O espelhamento é aproximado: o corpo do flybody não é perfeitamente simétrico. Trocar os ângulos entre os lados dá a
posição espelhada dos patins com erro de 2–4%, e a direção da lâmina com erro de 5–10°. Por isso há uma variante sem
espelhamento.

**Teste do aluno sozinho, a cada 3 iterações:** os testes do M7 (critério do M2, curvas, parada), a fração de patins no
chão e o deslize com os seis, com a média móvel dos parâmetros.

## Critério de sucesso

- **O aluno sozinho cumpre os mesmos 5 critérios de marcha natural** (`docs/m6_regras_novas.md`). Além disso: curvas para
  os dois lados (diferença entre os giros pedindo +0,5 e −0,5 ≥0,5 rad/s) e parar quando pedido (≥80%).
- **Se passar:** o passo seguinte, registrado antes de rodar, é o treino final gravado. Estratégias evolutivas partindo
  do aluno, com as regras novas, verificando que a marcha se mantém. Captura desde a tentativa #1 dessa fase.
- **Parada:** se, até a iteração 60, nenhuma variante tiver o aluno sozinho com ≥50% dos patins no chão e acima de
  1,5 cm/s pedindo 3,5, a decisão volta ao usuário.
