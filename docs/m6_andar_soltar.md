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
