# M6: segundo treino final gravado (registrado em 24/10/2026, antes de rodar)

## Por quê

- **O primeiro treino final (`final_s0`, `docs/m6_treino_final.md`) cumpriu o critério do M2,** mas com a recompensa
  antiga, em que levantar patins pagava. A marcha que saiu não é a de um animal: desliza sobre dois patins e chuta
  com a pata do meio esquerda.
- **Os testes seguintes acharam, por etapas, a melhor receita para o conectoma:** a evolui_u (`docs/m6_regras_novas.md`),
  com 4 dos 5 critérios de marcha natural. Falta só o deslize com os seis patins. Ensinar o gesto (`docs/m6_ensinar.md`)
  e andar de patins travados antes de soltar (`docs/m6_andar_soltar.md`) não fizeram melhor.
- **A evolui_u não tem captura.** O vídeo só mostra tentativas gravadas do treino que aparece, desde a #1. Por isso a
  receita roda de novo, agora com captura.

## Receita

Os argumentos da evolui_q (que começou a linhagem da evolui_u), mais `--capture`:

```
m6_es.py --run final2_s0 --seed 0 --init-from runs/anda_r5I/best_it6.pt \
  --capture --deterministic --net-dtype float64 \
  --envs 64 --eval-every 5 --threads 4 --device cuda --max-gpu-mem-gb 8 \
  --sigma-type 0.1 --sigma-tone 2 --sigma-enc 1 --sigma-dec-raw 0.2 --sigma-dec-bias 0.1 \
  --lr 0.05 --lr-warm 0.3 --lr-warm-gens 10 \
  --v-start 2 --v-final 4 --phase2-gen 30 --p-stand 0.3 --yaw-start 0.3 --yaw-final 1.5 \
  --turn-side-gains --yaw-filtered --w-roll 1.5 --w-glide 0.5 --w-yaw 1.0 --yaw-gated --count-lifted
```

São três fases, em gerações fixadas de antemão:

| Gerações | Tentativas no vídeo | Regra | Argumentos acrescentados | Como na |
|---|---|---|---|---|
| 0–499 | #1 a #32.000 | patins no ar contam | `--generations 500` | evolui_q |
| 500–999 | #32.001 a #64.000 | + regra de apoio | `--w-contact 1.0 --generations 1000` | evolui_r |
| 1000–1399 | #64.001 a #89.600 | + empurrão inicial | `--w-contact 1.0 --push-gens 200 --push-from 1000 --generations 1400` | evolui_u |

- **Cada troca de fase é uma retomada do mesmo treino:** a numeração das tentativas continua, e os argumentos de cada
  retomada ficam em `runs/final2_s0/resumes.jsonl`. No vídeo, as trocas de regra aparecem rotuladas na tentativa em
  que acontecem.
- **As fases repetem a ordem em que os testes exploratórios as acharam.** Neste treino, elas não dependem de resultado
  nenhum.
- **O empurrão inicial** vale nas gerações 1000–1199: até metade dos pares começa a tentativa já andando, fração que cai
  a zero. Essas tentativas ficam marcadas (campo `push`) e aparecem rotuladas.
- **No Spark,** um script escolhe a fase pela geração do último checkpoint e cuida das falhas conhecidas:
  - mata e retoma se o checkpoint ficar parado 15 min (o travamento no `torch.save`);
  - relança se o processo cair sem avançar (a falha ao criar o contexto da GPU).

  A retomada é exata, porque o checkpoint guarda o estado completo e o sorteio de cada geração depende só da semente e
  do número da geração.

## Reprodução esperada

O treino é determinístico. A semente, o código e o controlador são os mesmos (float64, somas exatas), e a captura só
grava. Por isso ele deve reproduzir a evolui_q, a evolui_r e a evolui_u geração por geração.

- **Conferência:** cada linha do `metrics.jsonl` (retornos, velocidades, avaliações) é comparada com a da mesma geração
  na evolui_q, na evolui_r ou na evolui_u. A primeira conferência é a geração 0, logo no começo.
- **Se as linhas forem idênticas,** o checkpoint da geração 1400 é o da evolui_u. Mesmo assim, as medidas do fim são
  refeitas no checkpoint deste treino.
- **Se divergirem,** registro a primeira geração divergente e a causa, se aparecer. O treino segue com as mesmas fases e
  é julgado pelos próprios resultados. Se, na geração 1400, ele cumprir menos critérios de marcha que a evolui_u, a
  decisão volta ao usuário.

## Curvas: exploratório em paralelo, sem captura

**O problema.** A evolui_u não vira quando pedido:

- nos testes de curva do `m7_eval.py` (±0,5 rad/s a 3 cm/s), o giro médio é −0,10 e −0,09 rad/s;
- o currículo de curvas parou em ±0,55 rad/s desde a geração ~600 (acerto 69–75%, abaixo dos 80% que o fazem subir).
  Com a tolerância de 0,5 rad/s (`--yaw-tol`), andar reto já passa até ±0,5 rad/s: o currículo abriu sem que ela
  virasse.

**Provas causais na geração 1400** (`m6_eval.py --probes`, a 3 cm/s):

- calar o DNg100 a para: 3,02 → 0,00 cm/s;
- o DNa02 está invertido: o esquerdo a vira para a direita (−0,28 rad/s) e o direito, para a esquerda (+0,10 rad/s).
  Ela vira para o lado estimulado em só 5% dos testes. O primeiro treino final chegou ao mesmo DNa02 invertido.
- **O DNa01 também está invertido, e nos dois lados:** com a mesma corrente, o esquerdo vira para a direita (−0,31
  rad/s) e o direito, para a esquerda (+0,38). Com o triplo da corrente, o DNa02 esquerdo dá −0,55 rad/s.

**O que isso indica:** a via de giro existe e é forte o bastante, mas age para o lado oposto. O comando de virar à
esquerda entra no DNa02 esquerdo, e quanto mais o ganho de giro sobe, mais ela vira para o lado errado. Por isso a
recompensa de giro não consegue aumentar esse ganho.

Uma explicação possível vem dos patinadores: quem vira para a esquerda empurra com a perna de fora, a direita. Se o
DNa02 reforça as patas do próprio lado, como na caminhada, de patins isso empurra a mosca para o outro lado. É só uma
hipótese, e a variante curvas_b a testa.

**Ramificações**, todas partindo da evolui_u na geração 1400 (o mesmo ponto em que o treino gravado estará na geração
1400, se a reprodução for exata). Têm os argumentos da última fase, sem captura, e vão até a geração 1700 (~8–10 h):

| Execução | Mudança | O que testa |
|---|---|---|
| curvas_a | `--w-yaw 3.0 --sigma-yaw 0.3 --yaw-tol 0.25` | A variante aprovada. A recompensa de giro fica 3× maior e mais estreita, e o currículo só abre curvas maiores quando ela vira de fato: erro de giro <0,25 rad/s, o limiar do `m7_eval.py`. |
| curvas_b | a mesma, mais `--turn-cross` | Se a via existe mas está cruzada: o comando de virar à esquerda entra no DNa02 direito, e vice-versa. |

**Medidas** nas gerações 1500, 1600 e 1700: `m7_eval.py`, `m6_eval.py --probes` e `m8_gait.py` no teste fixo 1.

**Critério "vira quando pedido":** nos testes de curva do `m7_eval.py` (±0,5 rad/s a 3 cm/s), três condições:

- giro médio ≥ +0,25 rad/s quando se pede +0,5;
- giro médio ≤ −0,25 rad/s quando se pede −0,5;
- acerto das curvas ≥50%.

Além disso, ela precisa manter o que a evolui_u já cumpre:

- critério do M2;
- parada quando pedido em ≥80% dos testes;
- os critérios de marcha natural 1, 2 e 5.

**Regra de decisão:**

- **Se a curvas_a cumprir:** a mesma fase entra no treino gravado a partir da geração 1400, com adendo neste documento
  antes de rodar. Ela vai até a primeira geração, entre 1500, 1600 e 1700, em que o critério foi cumprido. Depois, o
  slalom e o DNa02 entram nas medidas.
- **Se só a curvas_b cumprir:** a decisão volta ao usuário. O comando de giro passaria a entrar no lado oposto, e a prova
  causal mostraria o DNa02 de um lado virando a mosca para o outro. Isso tem explicação física (o empurrão da perna de
  fora), mas é o contrário do que o DNa02 faz na mosca que anda.
- **Se nenhuma cumprir:** o treino gravado termina na geração 1400. O vídeo diz que ela patina reto e para quando pedido,
  mas ainda não vira quando o comando pede, e troca o slalom pela pista reta com os 20 testes. A exceção é uma variante
  que esteja melhorando: se a diferença entre os giros para os dois lados subir de 1500 a 1700, ela pode seguir mais
  300 gerações, com registro antes.

## Medidas no fim (geração 1400, ou depois da fase de curvas)

- **Critério do M2:** 20 testes fixos pedindo 3,5 cm/s (`m7_eval.py`).
- **Os 5 critérios de marcha natural:** `m8_gait.py` no teste fixo 1, mais o `m7_eval.py`.
- **Provas causais** (`m6_eval.py --probes`): o DNg100 calado para a mosca; o DNa02 de um lado a vira para esse lado.
- **Slalom** (`m6_eval.py --slalom`), se a fase de curvas entrar.
- **Ensaio da captura (M5):** o `m5_rehearsal.py` re-simula 3 tentativas sorteadas. A física tem de sair idêntica bit a
  bit, e o cérebro com diferença <1e-4.
- **Passos de treino:** o critério "em até 3× os passos do M2" não vale para as estratégias evolutivas, o mesmo desvio
  do primeiro treino final. Registro os passos até a primeira geração que cumpre o critério do M2.

## Disco e máquina

- O `final_s0` ocupou 29 GB em 2000 gerações com captura (~14 MB por geração). Este treino deve ocupar ~20 GB, e há
  99 GB livres.
- **Duração prevista:** ~85 s por geração, ou ~33 h para as 1400 gerações (a evolui_q, a evolui_r e a evolui_u somaram
  34,6 h).
- **Recursos:** `nice` 10, 4 núcleos de física e limite de 8 GB na GPU.

## No vídeo

- O vídeo diz que a receita foi achada em execuções exploratórias anteriores, sem captura, e que este treino a repete
  gravando desde a tentativa #1:
  - evolui_a a evolui_y;
  - patina_a a patina_e;
  - as MLPs de diagnóstico;
  - os testes de ensinar o gesto;
  - final_s0, final_s1 e final_s2.
- As trocas de regra (tentativas #32.001 e #64.001) e as tentativas com empurrão aparecem rotuladas.
