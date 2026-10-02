# M8: roteiro do vídeo (rascunho, 02/10/2026)

Parte do plano (`docs/plano.md`, "O vídeo que queremos no final") e muda onde os resultados foram outros. Regra de
sempre: nenhum clipe encenado. Cada clipe sai de uma tentativa gravada ou de um teste re-simulável a partir de um
checkpoint, e leva no canto o número da tentativa e a geração.

## Formato

- **Quadro de 1280×720.** A mosca à esquerda. À direita, o painel do cérebro: cada neurônio no centro das suas
  sinapses no MaleCNS, com brilho pela atividade re-simulada.
- **Câmera lenta de verdade, sempre rotulada.** A física é refeita a cada 0,2 ms, sem interpolar.
- **Régua de 1 mm** no plano da mosca.
- **Ferramentas:**
  - `m8_trace.py`, no Spark: re-simula a tentativa, com física bit a bit e cérebro com diferença 0;
  - `m8_render.py`, no notebook: desenha a mosca e o painel;
  - `m8_curves.py`: gráficos;
  - `m6_milestones.py`: escolha das tentativas.
- **Duração-alvo:** 4 a 6 minutos.

## Cenas

### 1. Abertura: "Tentativa #1"

- **O que se vê:** a Tentativa #1 do treino final, a mosca de patins tentando sair do lugar, e ela fica parada
  (0,2 cm/s em 5 s). Em seguida, a primeira queda do registro, a #9 (geração 0, caiu em 2,47 s).
- **Mudança em relação ao plano:** o plano imaginava a #1 escorregando e caindo; a #1 real não cai.
- **A decidir:** mostrar a #9 exige um marco novo, "primeira queda". É uma regra automática (a primeira tentativa
  que cai), mas foi criada depois de ver o registro, e o vídeo diz isso.
- **Narração (sugestão):** "Esta é a tentativa número 1. A mosca nunca viu um patim."
- **Estado:** #1 re-simulada e renderizada; #9 a re-simular.

### 2. Prova de que é a mosca, antes de qualquer treino

- **O que se vê:** só o painel do cérebro, em tela cheia, com a rede sem nenhum treino, os parâmetros publicados
  de Pugliese et al. e a fiação do MaleCNS. Estimular o DNg100 faz surgir sozinho um ritmo de ~10 Hz nos
  neurônios motores da pata, com o traço da atividade embaixo.
- **Cuidado:** falar em ritmo, não em passada alternada. A coordenação entre as patas não sai da fiação sozinha
  (M3).
- **Estado:** rascunho renderizado (`m3_rhythm_full.py --save-trace`, `m8_render.py`): DNg100 direito, 9,5 Hz nos motores
  da pata da frente esquerda; brilho relativo (os picos são de poucos Hz), dito na tela.

### 3. O que é treinado e o que é fixo

- **Animação curta.** Entram os sentidos das patas (proprioceptores) e os comandos (DNg100 "andar", DNa02
  "virar"); saem os neurônios motores, cada um ligado ao seu músculo com sinal fixo.
- **Fixo:** quem liga com quem e o sinal de cada ligação.
- **Ajustado:** a força das ligações (no andar), a constante de tempo e a excitabilidade por tipo de célula, o
  tônus dos motores e a escala da entrada e da saída.
- **Transparências desta cena:**
  - τ de 5 ms (dentro da faixa biológica);
  - halteres como sentido de rotação;
  - cérebro de macho num corpo de fêmea;
  - treinado por computador, não pelo aprendizado da própria mosca.

### 4. Primeiro, andar (M4)

- **O que se vê:** o conectoma que anda, num teste fixo re-simulado a partir do checkpoint `anda_r5I/best_it6`, sem
  patins. Ele aprendeu imitando a política de caminhada do flybody.
- **Estado:** o modo de teste a partir de checkpoint existe (`m8_trace.py --checkpoint`), mas só para o corpo de
  patins; falta o corpo de caminhada.

### 5. Montagem do treino de patins

Números reais, todos da lista automática de marcos.

**Marcos de velocidade e distância:**

| Tentativa | Geração | O que mostra |
|---|---|---|
| #47 | 0 | primeira acima de 0,5 cm/s sem cair |
| #85 | 1 | primeira acima de 1 cm/s |
| #269 | 4 | primeira acima de 3 cm/s; ela anda de patins dando passos, sem deslizar |
| #3.744 | 58 | primeira acima de 4 cm/s |

Recordes de distância até 26,2 cm (#43.474, geração 679).

**Arco da história:**

- Nas primeiras ~34 mil tentativas ela vai cada vez mais rápido, mas "andando" sobre os patins: 1 a 5% do tempo
  deslizando.
- Na tentativa #34.305 (geração 536), a regra do treino passou a premiar o deslize. O vídeo diz isso, com a curva
  marcada.
- Em ~10 mil tentativas, o deslize sobe para ~60%.
- Gráfico: `m8_curves.py` (velocidade e % deslizando por tentativa, mudanças de regra marcadas).

**Quedas:** 11% nas primeiras 100 gerações, ~0% depois da 1.500.

**Rótulos:** os clipes de treino levam "treino (variação da população)", porque cada tentativa é uma variação
sorteada da rede. Os de teste levam "teste, sem variação".

**Estado:**
- #1, #2 e #43.905 a #43.908 re-simuladas;
- falta re-simular os marcos;
- falta uma "colagem" de vários clipes curtos com o contador de tentativas.

### 6. Painel do cérebro

Fica ao lado o tempo todo. Uma vez, de perto, mostrar o DNg100 aceso enquanto ela patina.

### 7. Provas causais (testes fixos, a partir do checkpoint final)

- **DNg100 calado:** inibição contínua. Ela para (3,76 → 0,01 cm/s na geração 686). Clipe lado a lado, mesmo
  teste com e sem intervenção.
- **DNa02 de um lado:** no treino final, não vira de forma consistente para o lado estimulado (na geração 686,
  vira para a esquerda com os dois lados; na ~800, para o lado oposto ao estimulado).
  - Se as curvas não destravarem até a geração 2000, mostrar o resultado como ele é: "ela ainda não aprendeu a
    virar".
  - O conectoma que anda (M4) vira para o lado certo em 100% dos testes, e pode aparecer como contraste.
- **Estado:** DNg100 renderizado lado a lado (teste 1, geração 1725: 3,62 → 0,00 cm/s). Na mesma geração, o DNa02 de
  qualquer lado quase não muda o giro (−0,22 → −0,18 e −0,19 rad/s).

### 8. Desafio final: slalom

O plano previa uma grade com as 20 corridas de teste, sem escolher a melhor. Sem curvas, o slalom falha.

- **Opção A:** mostrar a grade assim mesmo ("o desafio que ela ainda não venceu").
- **Opção B:** trocar por uma pista reta com os 20 testes (critério do M2 cumprido).

A decisão fica para quando o final_s0 terminar.

### 9. A virada: conectoma real × embaralhado × MLP (M7)

- Curvas de aprendizado lado a lado, as 3 sementes de cada braço, com a regra de leitura registrada antes
  (`docs/m7_controles.md`).
- Também um clipe de teste de cada braço na mesma geração.
- O resultado entra como sair, inclusive empate.
- **Estado:** treinos na fila do Spark (2 a 3 dias).

### 10. Transparências e créditos

- Patins idealizados (lâmina sem aderência).
- Rede de taxa, não de neurônios que disparam.
- O que foi treinado e como, e as duas mudanças de recompensa no meio do treino gravado.
- Quantas execuções vieram antes da escolhida:
  - exploratórias patina_a a patina_e (PPO) e evolui_a a evolui_m (estratégias evolutivas);
  - sementes 1 e 2 do treino final, paradas na geração 52 pela regra registrada.
- Créditos: MaleCNS (Berg et al., CC-BY), flybody (Vaxenburg et al.), Pugliese et al.

## Ferramentas

Prontas:

- `m8_trace.py`: tentativas gravadas, e testes a partir de checkpoint com e sem intervenção (corpo de patins);
- `m8_render.py`: quadro com painel, lado a lado (`--compare`) e a cena do ritmo (só painel e traços);
- `m3_rhythm_full.py --save-trace`: atividade da rede no teste de ritmo.

Faltam:

1. Corpo de caminhada (M4) no `m8_trace.py` e no `m8_render.py`.
2. No `m8_render.py`:
   - grade de 20 testes (slalom ou reta);
   - zoom em neurônios destacados.
3. Montagem: cartelas de texto e concatenação com ffmpeg; narração e trilha ficam com o editor (DaVinci).
4. Conferência de autenticidade: um script que, para cada clipe da montagem, confere número da tentativa, geração,
   hashes da captura e da política, e o desvio 0 da re-simulação (os metadados já vão no `.npz` de cada tentativa).
