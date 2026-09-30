# M6: treino final gravado (registrado antes de rodar, 30/09/2026)

## Receita

Estratégias evolutivas (`scripts/m6_es.py`) partindo do conectoma que anda (`runs/anda_r5I/best_it6.pt`),
com captura desde a tentativa #1 e o controlador determinístico:

```
m6_es.py --run final_sN --seed N --init-from runs/anda_r5I/best_it6.pt \
  --capture --deterministic --net-dtype float64 \
  --envs 64 --generations 2000 --eval-every 5 \
  --sigma-type 0.1 --sigma-tone 2 --sigma-enc 1 --sigma-dec-raw 0.2 --sigma-dec-bias 0.1 \
  --lr 0.05 --lr-warm 0.3 --lr-warm-gens 10 \
  --v-start 2 --v-final 4 --phase2-gen 30 --p-stand 0.3 --yaw-start 0.3 --yaw-final 1.5 \
  --turn-side-gains --yaw-filtered
```

- Parâmetros evoluídos: fatores por tipo celular (a, θ, τ), tônus dos motores, vieses do codificador e dos
  halteres, decodificador anatômico (músculo → junta, sinal fixo), ganhos dos comandos de velocidade (DNg100)
  e de giro (DNa02, um por lado). Os ganhos por ligação ficam os do conectoma que anda (M4).
- Fase 1 (gerações 0–29): só retas. Fase 2 (a partir da geração 30): 30% dos pares pedem para ficar parados,
  e curvas de ±0,3 rad/s, que abrem até ±1,5 rad/s depois que a velocidade pedida chega a 4 cm/s.
- Recompensa: a do PPO de patins, com o giro medido pela média de 200 ms.

## Sementes e escolha

Três sementes (0, 1 e 2) rodam em paralelo. Na geração 50, segue só uma, pela regra:

1. Entre as sementes cuja média (avaliação sem perturbação) anda ≥1 cm/s nas avaliações das gerações 45 e 50,
   a que tiver a maior fração de tempo deslizando na média dessas duas avaliações.
2. Se nenhuma deslizar ≥15% nessas avaliações, rodam mais três sementes (3, 4 e 5) com a mesma regra.

As outras sementes ficam gravadas (capture/ e metrics.jsonl) e o vídeo diz quantas execuções vieram antes da
escolhida, incluindo as exploratórias (evolui_a a evolui_m, patina_a a patina_e, sem captura).

## Critérios (do plano)

M6: critério do M2 (≥80% de 20 testes com média ≥3 cm/s em 5 s pedindo 3,5, ≥25% deslizando, <10% de
quedas), slalom completo e provas causais (DNg100 calado para a mosca; DNa02 de um lado a vira para esse lado),
medidos com `m6_eval.py`. O critério "em até 3× os passos do M2" não vale para este método: as estratégias
evolutivas usaram ~6,6× nos exploratórios (desvio a registrar no README).

## Escolha na geração 50 (01/10/2026)

Avaliações da média nas gerações 45 e 50 (velocidade pedida 2 cm/s):

| Semente | Velocidade | Deslizando (média) | Resultado |
|---|---|---|---|
| 0 | 1,34 / 1,20 cm/s | 17,5% | segue (anda ≥1 cm/s e é a que mais desliza, ≥15%) |
| 1 | 1,57 / 1,75 cm/s | 7% | parada na geração 52, captura guardada |
| 2 | 0,07 / 0,05 cm/s | — | parada na geração 52 (não saiu do lugar), captura guardada |
