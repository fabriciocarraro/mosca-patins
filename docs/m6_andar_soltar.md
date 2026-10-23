# M6: andar de patins travados e soltar aos poucos (registrado em 22/10/2026, antes de rodar; ideia do usuário)

## Por quê

Ideia do usuário: aprender primeiro a andar com as seis patas e depois usar a mesma lógica nos patins, como uma pessoa
que aprende a patinar.

O plano já partia do conectoma que anda (M4), mas a caminhada não passou para os patins:

- **Andar empurra o chão para trás com o pé parado,** e o patim rola justamente nessa direção.
- **O conectoma que anda, no corpo de patins, fica parado,** mesmo com os patins travados (0,07–0,09 cm/s). A bota no lugar
  do pé (sem garra nem adesão) muda o que as patas sentem.

A versão que ainda não foi testada tem duas partes:

1. **Andar no próprio corpo de patins, com os patins travados,** e soltá-los aos poucos, do atrito ao longo da lâmina
   igual ao de lado (1,0, patim que se comporta como pé) até o do patim livre (0,01). O passo de andar teria de virar
   deslize, como num iniciante.
2. **Montar os patins com a ponta para fora.** Com a lâmina em "V", o empurrão para trás de uma pata vira em parte um
   empurrão de lado, que é o que a lâmina transmite. A alternância das patas, a lógica da caminhada, viraria passada de
   patinação.

Nada disso impõe o gesto: muda só a ordem em que as coisas são aprendidas e o ângulo do patim.

## Opções novas (desligadas por padrão; os treinos anteriores não mudam)

- **`EnvConfig.toe_out_deg` / `--toe-out`:** ângulo de montagem com a ponta para fora, espelhado entre os lados. Com
  35°, os patins ficam a ~12° nas patas da frente, ~25° nas do meio e ~29° nas de trás na postura de patinação. A mosca
  assenta sem cair, com os seis patins no chão. O desenho do vídeo usa o mesmo ângulo, gravado em cada traço.
- **`--glide-start/--glide-end/--glide-gens/--glide-from`** no `m6_es.py` e no `m2_train.py`: o atrito ao longo da lâmina cai
  em escala logarítmica de 1,0 até o do modelo (0,01). O valor de cada leva entra na captura, e a re-simulação o aplica.

## Testes

Todos com as regras do treino final + `--count-lifted --w-contact 1.0` (as que mais aproximaram o conectoma da marcha
natural), sem empurrão inicial (com os patins travados, ele não faz sentido) e sem captura.

| Execução | Controlador | Soltar os patins | Ponta para fora | Duração |
|---|---|---|---|---|
| mlp_w1 | MLP (PPO, do zero) | 1,0 → 0,01 nas primeiras 150 iterações | 0° | 400 iterações |
| mlp_w2 | MLP | idem | 35° | 400 iterações |
| mlp_w3 | MLP (controle) | não (patins livres) | 35° | 400 iterações |
| evolui_x | conectoma (do anda_r5I) | 1,0 → 0,01 nas primeiras 400 gerações | 0° | 1000 gerações |
| evolui_y | conectoma (do anda_r5I) | idem | 35° | 1000 gerações |

O controle do conectoma sem soltar os patins e sem ponta para fora já existe: o evolui_r (mesmas regras, 2 de 5 critérios).

## Critérios e regra de decisão

Os mesmos 5 critérios de marcha natural (`docs/m6_regras_novas.md`), medidos no fim com os patins livres e o ângulo de
cada variante, pelo `m8_gait.py` e pelo `m7_eval.py`. Além disso, curvas para os dois lados e parada.

- **Se um conectoma cumprir os critérios 1 a 3:** a receita dele vira a do novo treino final gravado, com registro
  próprio antes de rodar.
- **Se nenhum cumprir:** a decisão volta ao usuário.
- **As MLPs são diagnóstico:** mostram se andar e soltar os patins, com ou sem ponta para fora, leva uma rede livre a uma
  passada natural.

## Desvio nas MLPs (22/10/2026, 9h50, antes do resultado dos conectomas)

**O que se viu nas MLPs com patins travados (iteração 50):** as três ficaram paradas, com os seis patins no chão. A
fase de andar de botas não aconteceu. O resultado final delas também foi fraco:

- o currículo não passou de 2 cm/s;
- a melhor, a mlp_w1, anda a 2 cm/s com 74% dos patins no chão, mas desliza com os seis só 14% do tempo;
- as duas com ponta para fora ficaram piores.

**Causa provável, um erro do desenho do teste:** a regra de apoio (`--w-contact`) paga por patim no chão, e andar exige
levantar as patas. Com os patins travados, ficar parada com os seis no chão rende quase o mesmo que tentar andar. Além
disso, os testes tiraram o empurrão inicial, que todas as MLPs bem-sucedidas tinham.

**Conectomas na geração 100, com os patins travados:**

- evolui_x (ponta 0°) também ficou parado;
- evolui_y (ponta 35°) aprendeu a dar passos com as seis patas: patins no chão 30–76% do tempo, empurrão dividido entre
  as seis (10–27% cada), 1,2 cm/s, mas em círculo (−1,2 rad/s).

**Correção, só nas MLPs:** duas MLPs novas, sem a regra de apoio (só `--count-lifted`, como a mlp_c1) e com o empurrão
inicial padrão. Os conectomas seguem como registrados.

| Execução | Soltar os patins | Ponta para fora | Duração |
|---|---|---|---|
| mlp_w4 | 1,0 → 0,01 em 150 iterações | 0° | 400 iterações |
| mlp_w5 | idem | 35° | 400 iterações |

## Resultado (23/10/2026)

Teste fixo 1 pelo `m8_gait.py` e os 20 testes do `m7_eval.py`, com os patins livres no fim.

**Fase de andar de botas (patins travados, atrito 1,0):**

| Execução | O que fez |
|---|---|
| evolui_y (ponta 35°) | andou na geração 100: passos com as seis patas, empurrão dividido, mas em círculo |
| mlp_w4 (corrigida, ponta 0°) | andou na iteração 50: 0,7 cm/s, as seis patas no chão 34–54% do tempo, como numa caminhada |
| evolui_x (ponta 0°) | ficou parado |
| mlp_w1, mlp_w2, mlp_w3 (com a regra de apoio) | ficaram paradas |

**Depois de soltar os patins (geração 1000 / iteração 400), pedindo 3,5 cm/s:**

| Execução | Patins no chão | Empurrão | Desliza com os seis | Velocidade | Reto | Critérios |
|---|---|---|---|---|---|---|
| evolui_x (ponta 0°) | 82% (meio E 32%) | 82% do lado esquerdo | 10% (no `m7_eval`: 8%) | 1,6 cm/s | sim | 1 |
| evolui_y (ponta 35°) | 55% | meio E 50%; esquerda 68% | 0% | 2,8 cm/s | sim | 2 |
| mlp_w4 (ponta 0°) | 64% | meio D 61%, meio E 35% | 0% | 3,8 cm/s | sim | — |
| mlp_w5 (ponta 35°) | ~56% | meio E 80% | 0% | 2,2 cm/s pedindo 2 | não | — |

- **Andar de botas acontece:** no conectoma, só com a ponta para fora; na MLP, depois de tirar a regra de apoio. Quando
  os patins soltam, o passo vira remada com as patas do meio, que levantam para voltar, e não vira deslize com os seis
  patins no chão.
- **A ponta para fora** ajudou o conectoma a andar de patins travados, mas piorou a MLP e não trouxe deslize.
- **Nenhum conectoma cumpre os critérios 1 a 3,** e a decisão volta ao usuário. O melhor conectoma continua sendo a
  evolui_u (4 de 5, `docs/m6_regras_novas.md`).
