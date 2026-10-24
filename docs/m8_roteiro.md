# M8: roteiro do vídeo (rascunho, atualizado em 24/10/2026)

Parte do plano (`docs/plano.md`, "O vídeo que queremos no final") e muda onde os resultados foram outros. Regra de
sempre: nenhum clipe encenado. Cada clipe sai de uma tentativa gravada ou de um teste re-simulável a partir de um
checkpoint, e leva no canto o número da tentativa e a geração.

**O treino que aparece é o segundo treino final gravado (`final2_s0`, `docs/m6_treino_final2.md`).** Ele repete, com
captura desde a tentativa #1, a receita achada nos exploratórios. O treino é determinístico, então as tentativas são
as mesmas da linhagem evolui_q → evolui_r → evolui_u. Os números abaixo saem dessa linhagem e serão conferidos no
registro do próprio `final2_s0` quando ele terminar.

## Formato

- **Quadro de 1280×720.** A mosca à esquerda. À direita, o painel do cérebro: cada neurônio no centro das suas
  sinapses no MaleCNS, com brilho pela atividade re-simulada.
- **Câmera lenta de verdade, sempre rotulada.** A física é refeita a cada 0,2 ms, sem interpolar.
- **Régua de 1 mm** no plano da mosca.
- **Ferramentas:**
  - `m8_trace.py`, no Spark: re-simula a tentativa, com física bit a bit e cérebro com diferença 0;
  - `m8_render.py`, no notebook: desenha a mosca e o painel;
  - `m8_curves.py`: gráficos;
  - `m8_gait.py`: diagrama de contato e empurrão de cada patim;
  - `m6_milestones.py`: escolha das tentativas.
- **Duração-alvo:** 4 a 6 minutos.

## Cenas

### 1. Abertura: "Tentativa #1"

- **O que se vê:** a Tentativa #1, a mosca de patins tentando sair do lugar. Ela fica parada (0,2 cm/s em 5 s). Em
  seguida, a primeira queda do registro, a #9 (geração 0, caiu em 2,47 s).
- **A primeira geração é a mesma do primeiro treino final:** mesmo ponto de partida, mesma semente, e a recompensa não
  muda o que acontece na tentativa. Por isso a #1 e a #9 já renderizadas continuam valendo.
- **Mudança em relação ao plano:** o plano imaginava a #1 escorregando e caindo, mas a #1 real não cai. O marco
  "primeira queda" foi criado depois de ver o registro, e o vídeo diz isso.
- **Narração (sugestão):** "Esta é a tentativa número 1. A mosca nunca viu um patim."

### 2. Prova de que é a mosca, antes de qualquer treino

- **O que se vê:** só o painel do cérebro, em tela cheia, com a rede sem nenhum treino, os parâmetros publicados de
  Pugliese et al. e a fiação do MaleCNS. Estimular o DNg100 faz surgir sozinho um ritmo de ~10 Hz nos neurônios
  motores da pata, com o traço da atividade embaixo.
- **Cuidado:** falar em ritmo, não em passada alternada. A coordenação entre as patas não sai da fiação sozinha (M3).
- **Estado:** rascunho renderizado (`m3_rhythm_full.py --save-trace`, `m8_render.py`): DNg100 direito, 9,5 Hz nos
  motores da pata da frente esquerda. O brilho é relativo (os picos são de poucos Hz), e a tela diz isso.

### 3. O que é treinado e o que é fixo

- **Animação curta.** Entram os sentidos das patas (proprioceptores) e os comandos (DNg100 "andar", DNa02 "virar");
  saem os neurônios motores, cada um ligado ao seu músculo com sinal fixo.
- **Fixo:** quem liga com quem e o sinal de cada ligação.
- **Ajustado:** a força das ligações (no andar), a constante de tempo e a excitabilidade por tipo de célula, o tônus
  dos motores e a escala da entrada e da saída.
- **Transparências desta cena:**
  - τ de 5 ms (dentro da faixa biológica);
  - halteres como sentido de rotação;
  - cérebro de macho num corpo de fêmea;
  - treinado por computador, não pelo aprendizado da própria mosca.

### 4. Primeiro, andar (M4)

- **O que se vê:** o conectoma que anda, num teste fixo re-simulado a partir do checkpoint `anda_r5I/best_it6`, sem
  patins. Ele aprendeu imitando a política de caminhada do flybody. O DNa02 de um lado o vira para esse lado.
- **Estado:** o modo de teste a partir de checkpoint existe (`m8_trace.py --checkpoint`), mas só para o corpo de
  patins. Falta o corpo de caminhada.

### 5. Montagem do treino de patins

Números reais, todos da lista automática de marcos. Saem da linhagem e serão conferidos no `final2_s0`.

**Marcos de velocidade e distância** (tentativas de treino, cada uma uma variação sorteada da rede):

| Tentativa | Geração | O que mostra |
|---|---|---|
| #47 | 0 | primeira acima de 0,5 cm/s sem cair |
| #75 | 1 | primeira acima de 1 cm/s |
| #130 | 2 | primeira acima de 2 cm/s |
| #318 | 4 | primeira acima de 3 cm/s |
| #444 | 6 | primeira acima de 4 cm/s |
| #4.658 | 72 | recorde de distância: 27,2 cm em 5 s |
| #10.000 | 156 | pré-registrada; pedido "fique parada", e ela fica |

**Arco da história:** a marcha muda em três fases, e cada troca de regra aparece rotulada na tentativa em que acontece.

| Tentativas | Regra | O que muda na marcha (testes fixos, sem variação) |
|---|---|---|
| #1 a #32.000 | patins no ar não pagam mais nada | ela patina empurrando sobretudo com a pata do meio esquerda (52% do empurrão), com a pata do meio direita no ar o tempo todo (49% dos patins no chão na geração 500) |
| #32.001 a #64.000 | regra de apoio: cada patim no chão conta | os patins no chão sobem para ~68%, e o empurrão se divide entre as patas |
| #64.001 a #89.600 | empurrão inicial, que some em 200 gerações | patins no chão 77%, nenhuma pata no ar o tempo todo, nenhum patim com mais de 35% do empurrão |

- **Gráficos:**
  - `m8_curves.py`: velocidade e % deslizando por tentativa, com as trocas de regra marcadas;
  - `m8_gait.py`: diagrama de contato e parcela do empurrão de cada patim nas gerações 100, 500, 1000 e 1400. Mostra
    as patas descendo para o chão a cada fase.
- **Rótulos:**
  - os clipes de treino levam "treino (variação da população)";
  - os de teste levam "teste, sem variação";
  - as tentativas com empurrão levam "com empurrão inicial".
- **Estado:**
  - #1, #2 e #9 re-simuladas, do primeiro treino final, que tem a mesma geração 0;
  - falta re-simular os marcos e os testes das gerações 100, 500, 1000 e 1400 no `final2_s0`;
  - falta uma "colagem" de vários clipes curtos com o contador de tentativas.

### 6. Painel do cérebro

Fica ao lado o tempo todo. Uma vez, de perto, mostrar o DNg100 aceso enquanto ela patina.

### 7. Provas causais (testes fixos, a partir do checkpoint final, geração 1400)

- **DNg100 calado:** inibição contínua. Ela para (3,02 → 0,00 cm/s, pedindo 3). Clipe lado a lado, mesmo teste com e
  sem intervenção.
- **DNa02 de um lado:** ela não aprendeu a virar quando pedido.
  - Estimular o DNa02 esquerdo a vira para a direita (−0,28 rad/s). O direito a vira um pouco para a esquerda (+0,10).
  - Dois treinos exploratórios focados em curvas não mudaram isso. Com outras correntes e outras marchas, o sentido
    do giro muda.
  - Mostrar como é: "o neurônio de virar mexe no rumo, mas ela ainda não aprendeu a usá-lo".
  - O conectoma que anda (M4) vira para o lado estimulado em 100% dos testes, e pode aparecer como contraste.
- **Estado:** DNg100 renderizado lado a lado no primeiro treino final. Falta refazer no `final2_s0`.

### 8. Desafio final: a pista reta com os 20 testes

- **Sem curvas, o slalom fica de fora** (decisão registrada em `docs/m6_treino_final2.md`).
- **No lugar:** uma grade com as 20 corridas de teste, pedindo 3,5 cm/s em linha reta, sem escolher a melhor. Na
  evolui_u, a mesma geração: 100% acima de 3 cm/s, 3,26 cm/s de média, sem quedas.
- **Estado:** falta a grade no `m8_render.py`.

### 9. A virada: conectoma real × embaralhado × MLP (M7)

- Curvas de aprendizado lado a lado, as 3 sementes de cada braço, com a regra de leitura registrada antes
  (`docs/m7_controles.md`, registro de 24/10/2026).
- Também um clipe de teste de cada braço na mesma geração.
- O resultado entra como sair, inclusive empate.
- **Estado:** treinos na fila do Spark (~3–4 dias, a partir de 24/10).

### 10. Transparências e créditos

- Patins idealizados (lâmina sem aderência).
- Rede de taxa, não de neurônios que disparam.
- O que foi treinado e como.
- **A receita foi achada em exploratórios sem captura,** e o treino gravado a repete desde a tentativa #1:
  - as trocas de regra (#32.001 e #64.001) e o empurrão inicial aparecem rotulados;
  - quantas execuções vieram antes: patina_a a patina_e (PPO), evolui_a a evolui_y (estratégias evolutivas), as MLPs
    de diagnóstico, os testes de ensinar o gesto, o primeiro treino final (final_s0, com as sementes 1 e 2 paradas na
    geração 52) e os dois exploratórios de curvas.
- Créditos: MaleCNS (Berg et al., CC-BY), flybody (Vaxenburg et al.), Pugliese et al.

## Ferramentas

Prontas:

- `m8_trace.py`: tentativas gravadas, e testes a partir de checkpoint com e sem intervenção (corpo de patins);
- `m8_render.py`: quadro com painel, lado a lado (`--compare`) e a cena do ritmo (só painel e traços);
- `m8_gait.py`: diagrama de contato, ângulos e empurrão de cada patim;
- `m3_rhythm_full.py --save-trace`: atividade da rede no teste de ritmo.

Faltam:

1. Corpo de caminhada (M4) no `m8_trace.py` e no `m8_render.py`.
2. No `m8_render.py`:
   - grade dos 20 testes da pista reta;
   - zoom em neurônios destacados.
3. Versão do diagrama de marcha para o vídeo (as quatro gerações lado a lado).
4. Montagem: cartelas de texto e concatenação com ffmpeg. Narração e trilha ficam com o editor (DaVinci).
5. Conferência de autenticidade: um script que, para cada clipe da montagem, confere número da tentativa, geração,
   hashes da captura e da política, e o desvio 0 da re-simulação (os metadados já vão no `.npz` de cada tentativa).
