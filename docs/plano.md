# Mosca de patins — plano do projeto

## Contexto

Primeiro episódio da série "a mosca aprendendo a ser gente": uma mosca simulada, cujo controlador é o conectoma real, aprende a andar de patins. O vídeo mostra a evolução com tentativas reais ("Tentativa #N"), capturadas de forma fiel durante o treino. Projeto não monetizado. Na busca (listas da comunidade, GitHub, matérias), não apareceu nenhuma mosca simulada de patins, de skate ou com rodas nos pés.

**Resumo da abordagem:**
- **Corpo:** o `flybody` (MuJoCo), com patins acrescentados.
- **Controlador:** o cordão nervoso e os neurônios descendentes do MaleCNS, rodando como rede de taxa com a fiação congelada (a mesma equação do modelo de ritmo de passos de Pugliese et al.).
- **Ordem de treino:**
  1. Uma rede comum (MLP) prova que a física e a recompensa permitem patinar.
  2. O conectoma aprende a andar, por destilação.
  3. O conectoma aprende a patinar, por PPO.
- **Captura:** grava o suficiente para re-simular cada tentativa exatamente.

**Máquinas:**
- **Notebook** (Windows 11, Ryzen 7 5800H, RTX 3050 4 GB, 14 GB RAM, WSL2 Ubuntu): código, testes rápidos, prévias.
- **NVIDIA DGX Spark** (GB10: 20 núcleos Arm, GPU Blackwell sm_121, 128 GB de memória unificada a 273 GB/s, Ubuntu 24.04, CUDA 13, PyTorch 2.9 cu130, Python 3.12): treino e renderização. Na memória unificada, estourar a GPU trava a máquina inteira, então todo treino roda com limite de memória.

**Pasta:** `C:\Users\Fabricio\Documents\GitHub\mosca-patins` (repositório Git, clonado também no Spark).

## O vídeo que queremos no final

1. **Abertura, "Tentativa #1":** a mosca tenta andar normalmente de patins, escorrega, abre espacate e cai de costas com as rodinhas girando.
2. **Prova de que é a mosca, antes de qualquer treino:** estimular o DNg100, o neurônio de comando de andar, faz o cordão nervoso gerar sozinho um ritmo de 7 a 15 Hz nas patas. É um resultado publicado (Pugliese et al.), que vamos reproduzir.
3. **Explicação curta:** o conectoma, o que entra (sensores das patas), o que sai (neurônios motores), o que é treinado e o que é fixo.
4. **Montagem com números reais:** primeiro segundo em pé, primeiro deslize, o gesto de patinação surgindo, primeira curva, recordes de distância.
5. **Painel do cérebro** ao lado o tempo todo, com os neurônios das patas acendendo.
6. **Provas causais:** silenciar o DNg100 faz a mosca parar; estimular o DNa02 de um lado faz ela virar. Bônus, se o treino incluir ré: estimular o MDN, o neurônio "moonwalker", e ela patina de ré.
7. **Desafio final:** slalom entre cones, mostrado como uma grade com as 20 corridas de teste, sem escolher a melhor.
8. **Virada:** a mesma receita com o conectoma embaralhado e com uma MLP. Curvas de aprendizado lado a lado, com o resultado que sair.

**Linguagem visual:**
- Câmera lenta de ~20×, sempre rotulada; a mosca dá 10 a 15 passos por segundo.
- Régua de 1 mm na tela.
- Desfoque de movimento nas rodas, que giram ~100 vezes por segundo e piscariam no vídeo.

## Decisões principais

| Decisão | Escolha | Motivo |
|---|---|---|
| Corpo | `fruitfly.xml` do flybody (Apache-2.0), carregado pelo `mujoco` em Python, sem dm_control/Acme | Mesmo código no Windows e no Spark (Arm) |
| Conectoma | MaleCNS v1.0 (CC-BY 4.0, arquivos Feather abertos, sem login) | Os neurônios motores das patas vêm nomeados pelo músculo, e o ritmo do DNg100 foi confirmado nele. O corpo do flybody é de fêmea: isso é dito no vídeo, e o BANC (fêmea, CC-BY) entra como braço de controle |
| Recorte da rede | Cordão nervoso + neurônios descendentes e ascendentes, ligações com ≥5 sinapses, só neurônios com caminho até um neurônio motor da pata | É onde mora o controle das patas, e roda na velocidade da física: estimativa de ~20 a 25 mil neurônios e ~1 milhão de ligações. O MaleCNS inteiro no mesmo limiar tem 6,2 milhões de ligações e fica como variante |
| Modelo de neurônio | Equação de Pugliese et al.: `τ·dr/dt = −r + r_max·max(tanh(a(b·Σw·r + I − θ)), 0)`, com estado de um passo para o outro e um passo de Euler a cada 2 ms | Preserva o ritmo do cordão nervoso e os laços de realimentação; parte de parâmetros publicados |
| Controlador | Sem camadas escondidas. Codificador linear por pata, que só alimenta os proprioceptores daquela pata. Decodificador linear por pata, que só lê os neurônios motores dela, com sinal fixo pelo músculo | Deixa o mérito com a fiação da mosca |
| Ordem do treino | MLP patinadora → conectoma anda (destilação) → conectoma patina (PPO) | Prova física e recompensa antes de gastar tempo no conectoma |
| Física | MuJoCo na CPU: 64 a 128 ambientes em threads (`mujoco.rollout`), em dois grupos alternados para CPU e GPU trabalharem ao mesmo tempo | A banda de memória da GPU fica para o cérebro, e o `noslip` e o atrito anisotrópico só são garantidos na CPU. MuJoCo Warp só entra se o M0 medir menos de ~3 mil passos/s |
| Captura | Estado inicial, controles e ruído de cada tentativa; o vídeo é re-simulado | Fidelidade exata, câmera livre e câmera lenta de verdade |

## Arquitetura

Laço de controle a cada 2 ms:

```
corpo (MuJoCo) → sensores de cada pata → codificador da pata → proprioceptores da pata
                                                                   ↓
comando "ande / vire / ré" → DNg100, DNa01/DNa02, MDN → cordão nervoso (rede de taxa)
                                                                   ↓
corpo (MuJoCo) ← atuadores da pata ← decodificador da pata ← neurônios motores da pata
```

### 1. Corpo e patins

**Base (flybody):**
- Unidades CGS; corpo de 0,297 cm e ~0,98 mg.
- Física a 0,2 ms e controle a 2 ms.
- Cada pata tem 11 juntas e 8 atuadores de posição, mais adesão na garra.
- Asas recolhidas, como no modo de andar do flybody.

**Bota:**
- Fica presa ao primeiro segmento do tarso. As juntas 2 a 5 do tarso e o tendão delas saem do modelo, e a junta `tarsus` vira o tornozelo.
- O patim fica num corpo-filho próprio, sem junta. Se ficasse no corpo da garra, o atuador de adesão, que age em tudo que está nele, grudaria o patim no chão.

**Dois modelos de patim, ambos testados no M1:**
- **Lâmina (padrão de treino).** Uma cápsula rígida por pata, com atrito anisotrópico definido num par de contato explícito:
  - baixo ao longo do patim (~0,01) e alto de lado (~1,0);
  - sem peças girando, é o modelo mais estável;
  - a cápsula é a única forma do MuJoCo em que o atrito acompanha o eixo do objeto, e a pesquisa confirmou que isso funciona na escala da mosca.
- **Rodas de verdade (validação).** 2 rodas elipsoides em linha por bota, com dobradiça, numa classe própria definida fora das classes do flybody.
  - Sem classe própria, elas herdam `armature` 1e-6, que equivale a ~20 mg por roda, 20 vezes a mosca inteira. Herdam também amortecimento 0,01, que faz a simulação explodir.
  - Pontos de partida: raio de 40 a 70 µm, densidade 1, `limited="false"`, `armature` ~1e-11, amortecimento ~1e-9, `frictionloss` de 1e-6 a 1e-5.
- A política treinada num modelo é testada no outro, como a ETH fez no humanoide de patins, para garantir que ela não explora um defeito da simulação.
- No vídeo, as rodas giram exatamente na velocidade real do patim ao longo do chão.

**Armadilhas de escala, todas resolvidas com valores explícitos:**
- A densidade padrão do MuJoCo é 1000: vale no SI, mas em CGS dá 1 kg por cm³.
- Atrito de torção e de rolamento têm unidade de comprimento, e os padrões são maiores que a própria roda.
- Com a maciez de contato padrão, uma roda de 40 µm afundaria 154 µm. Por isso o contato usa `solref` de 0,0004 a 0,001 s e `solimp` (0,95, 0,99, 0,01), como no flybody, com cones elípticos, `impratio` ~10 e `noslip` de 1 a 3.

**Alavancas de projeto:**
- **Comprimento da lâmina:** curta se comporta como uma roda só; longa, como patim em linha.
- **Ângulo de montagem com a ponta para fora:** num robô quadrúpede de patins, patins paralelos ao corpo deixaram a velocidade quase incontrolável.
- Conferir quanto a coxa e o fêmur conseguem girar cada patim; o tarso das patas do meio aponta para o lado.

**Demais ajustes:**
- A massa de cada patim fica no máximo igual à da pata (~0,016 mg).
- A resistência do ar fica ligada: nessas velocidades ela é 10 a 35% da resistência ao rolamento.
- As pernas colidem com o chão, com penalidade, para nada atravessar o piso na câmera.
- Queda: tórax, cabeça ou abdômen no chão, altura baixa ou inclinação acima do limite.

### 2. Cérebro

**Dados** (MaleCNS v1.0, arquivos Feather em `storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`):
- Anotações (13 MB), neurotransmissores (42 MB) e pesos (1,1 GB).
- Sinal pelo `consensus_nt`: acetilcolina +, GABA e glutamato −. O `predicted_nt` erra, por exemplo, ao chamar os corpos de Kenyon de dopaminérgicos.

**Grupos por pata** (perna via `somaNeuromere` T1/T2/T3 ou nervo, lado via `somaSide`/`rootSide`):
- **Motores:** 381 neurônios motores das patas (`vnc_motor`, subclasses fl/ml/hl): T1 135, T2 116, T3 130. O tipo de cada um é o nome do músculo (extensor da tíbia, flexor do trocânter, rotadores da coxa...). Uma tabela nossa leva músculo → junta do flybody → sinal (extensor +, flexor −). Os ~53 sem músculo identificado ficam com peso livre dentro da própria pata.
- **Sensoriais:** proprioceptores das patas: órgão cordotonal, placas de pelos e os proprioceptores sem órgão anotado. Poucas sensilas campaniformes (sensores de carga) aparecem anotadas nas patas (12), então as forças de contato entram pelos proprioceptores disponíveis.
- **Comando:** DNg100 (andar; inicia a marcha até em moscas sem cabeça), DNg97/oDN1 e DNb08 como reforço; DNa01/DNa02 assimétricos para virar; MDN para ré.
- **Inclinação:** começa sem entrada própria. Se o treino mostrar que falta, entra por um grupo sensorial declarado (por exemplo, as placas de pelos do pescoço), e isso é dito no vídeo.

**Dinâmica:**
- Parâmetros iniciais de Pugliese et al.: b = 0,03; a ~ N(1; 0,1); θ ~ N(7,5; 0,6); r_max ~ N(200; 10) Hz; τ ~ N(20; 2) ms.
- Treináveis por tipo celular: a, θ e τ. Isso dá simetria esquerda/direita de graça.
- Treino com retropropagação truncada no tempo (32 a 64 passos).
- Poda exata: sai quem não tem caminho até um neurônio motor da pata; neurônios que nenhuma entrada alcança viram viés constante.

**Escada de flexibilidade** (sobe só se o degrau anterior falhar, e o vídeo diz qual foi usado):
1. Parâmetros por tipo celular.
2. Mais um ganho positivo por ligação, com o sinal sempre fixo.

**Lição de quem tentou antes:**
- O therealfly (MaleCNS + flybody, sem treino) gerou uma oscilação global de 45 Hz, com as duas patas em fase, igual à do conectoma embaralhado.
- O fly-cord-robots reproduziu o modelo de Pugliese num corpo, mas a realimentação das patas ficou desprezível ou disparou.
- Conclusão: a fiação sozinha, sem treinar nada, não anda. O treino dos parâmetros por tipo celular é o que se testa aqui.

**Variantes de controle:**
- conectoma embaralhado (mesmo nº de ligações e sinais por neurônio);
- grafo aleatório do mesmo tamanho;
- MLP com o mesmo nº de parâmetros;
- BANC (fêmea);
- sistema nervoso inteiro, se couber no tempo.

**Referência de implementação:** código de Pugliese (JAX, MIT) e o fly-cord-robots, que reproduziu o modelo com correlação de 0,9993.

### 3. Treino

**MLP patinadora:** a mesma tarefa com uma rede comum. Serve para acertar física, recompensa e currículo, e depois vira braço de controle.

**Conectoma anda** (destilação, limite de 3 semanas):
- Professor: a política de andar publicada pelo flybody (MLP 741→512×4→59). Os pesos são convertidos uma vez para PyTorch no WSL2 do notebook (Linux x86, onde o TensorFlow 2.8 dessa política instala sem dor); depois disso o projeto não depende de TensorFlow.
- Dados: as ~16 mil trajetórias de moscas reais (80 min, 3 GB) do figshare da Janelia.
- O aluno recebe comando de velocidade e virada no lugar da trajetória futura.
- Se estourar o prazo, o ponto de partida vira só "ficar em pé".

**Conectoma patina** (PPO):
- **Partida:** o crítico aquece primeiro, com a política congelada. Uma penalidade por se afastar do "andar" vai sumindo aos poucos.
- **Parâmetros:** γ 0,996, λ 0,95, clip 0,1 no início, taxa de aprendizado ajustada para KL-alvo de 0,01 (1e-4 para os parâmetros dos neurônios).
- **Crítico:** uma MLP que enxerga o estado completo do simulador (dito no vídeo).
- **Ruído de exploração:** colorido ou gSDE, porque ruído branco a 500 Hz parece tremedeira.
- **Recompensa**, na escala da mosca (ela anda a 1,3–2,8 cm/s):
  - seguir a velocidade média de 200 ms e a virada pedida, e ficar em pé;
  - bônus de rolamento: patim no chão movendo-se ao longo do próprio eixo, na velocidade do corpo. Foi esse termo que fez o humanoide de patins da ETH deslizar em vez de dar passinhos;
  - penalidade por derrapagem lateral, que também pega trapaças da física;
  - custo de transporte (energia por distância), que só entra depois do primeiro movimento;
  - penalidade por ações bruscas;
  - dados espelhados esquerda/direita, para o movimento sair simétrico;
  - fim da tentativa em qualquer contato que não seja do patim, ou se a mosca capotar. Sem bônus por continuar viva.
- **Currículo**, avançando com 80% de sucesso: equilibrar parada → faixa de velocidade pedida que se abre conforme ela acerta, até 10 cm/s → curvas → slalom (→ ré, para o bônus do MDN).
- **Ajudas opcionais** que somem com o tempo (rampa leve, empurrão inicial) aparecem rotuladas no vídeo. O percurso de teste não tem nenhuma.
- **Expectativa honesta:** em quase todos os precedentes, o que surge sozinho é o "swizzle", a ampulheta dos iniciantes em que os patins abrem e fecham sem sair do chão. Empurrar e deslizar só surgiu sem ajuda num caso (a ETH); nos outros foi imposto. Se o swizzle dominar, é um ótimo momento de vídeo: "ela descobriu sozinha o que todo iniciante aprende".
- **Alavancas para empurrar e deslizar** que mantêm o gesto emergente: lâminas mais longas, resistência de deslize ajustada para empurrar compensar, bônus de deslize, comandos de velocidade intermitentes. Relógio de fase ou movimento humano capturado impõem o gesto: ficam de fora, ou são ditos no vídeo.
- **Métricas registradas** (viram gráficos no vídeo): fração do tempo rolando contra dando passos, fração do tempo deslizando, ângulo do patim no empurrão, derrapagem lateral, e custo de transporte comparado ao de andar.

**Tentativa = episódio completo:** começa parada de patins e termina na queda ou em 5 s. Cada tentativa roda inteira com uma única versão da política; o PPO atualiza entre levas de tentativas completas.

**Orçamento:** 10⁸ a 10⁹ passos. Com a rede recortada, o ritmo é limitado pela física (estimativa de 5 a 10 mil passos/s), então 10⁸ passos levam 3 a 6 horas. O M0 mede o valor real.

### 4. Captura fiel

- **O que se grava por tentativa:**
  - semente, derivada de hash(execução, tentativa);
  - estado inicial do simulador;
  - controles e ruído a 500 Hz;
  - versão da política;
  - estado a cada 0,5 s, para detectar divergência.

  São 1 a 5 KB por passo, 5 a 25 GB no total.
- **Re-simulação:** a física é re-simulada deterministicamente, inclusive a 5 kHz para a câmera lenta. A atividade do cérebro também é re-simulada, sem gravar os ~333 KB por passo que ela ocuparia.
- **Numeração:** "Tentativa #N" é o índice global do episódio de treino, atribuído quando ele começa, somando todos os ambientes em paralelo.
- **O que fica salvo sempre:**
  - números pré-registrados (#1, #10, #100, #1.000, #10.000...);
  - uma amostra aleatória de 5%;
  - marcos automáticos: primeiro 1 s em pé, primeiro deslize, primeira curva, primeiro slalom e cada novo recorde de distância.
- **Rótulos:** clipes de treino (com ruído de exploração) e de teste (sem ruído) aparecem identificados como tais.
- **Percurso de teste:** pista reta mais slalom, com 20 sementes fixas, igual para todos os braços. Cada checkpoint (em escala logarítmica) roda os 20.
- **Manifesto:** ID da tentativa, hash do checkpoint, semente e versões fixadas de contêiner e bibliotecas. Um ID pequeno vai no canto do quadro. Logs e código são publicados.
- O vídeo diz quantas execuções de treino vieram antes da que aparece.

### 5. Renderização e edição

- Rascunhos com o renderizador do MuJoCo. Versão final pelo exportador USD do MuJoCo, depois Blender.
- **Painel do cérebro:** nuvem de pontos com as posições reais dos neurônios, com brilho pela atividade re-simulada.
  - Posições pela `somaLocation`, que cobre 84% dos neurônios. Os axônios sensoriais, que não têm corpo celular no volume, usam o centro das sinapses.
  - Antes de usar, conferir que a re-simulação reproduz as ações gravadas, com diferença abaixo de 1e-4.
- Tomada de abertura com a anatomia real do conectoma, pelo neuVid (Blender).
- Textos, recordes e gráficos com Python/ffmpeg ou DaVinci.

## Estrutura do repositório

```
mosca-patins/
  assets/                      baixados por script, não versionados: flybody, MaleCNS, dados e política do flybody
  src/mosca/body/skates.py     remove o tarso distal; acrescenta bota, lâmina ou rodas e os pares de contato
  src/mosca/body/env.py        ambientes em lote: observações, recompensa, queda, currículo
  src/mosca/brain/graph.py     monta o grafo, recorte, poda, grupos por pata, tabela músculo → junta
  src/mosca/brain/model.py     rede de taxa, codificadores e decodificadores por pata, variantes
  src/mosca/train/mlp.py       patinadora MLP
  src/mosca/train/distill.py   conectoma aprende a andar
  src/mosca/train/ppo.py       PPO com tentativas completas e ganchos de captura
  src/mosca/capture/           gravador, marcos, manifesto, re-simulação
  src/mosca/render/            MuJoCo, USD, painel do cérebro, sobreposições
  scripts/                     testes de física, benchmarks, conversão da política, teste de ritmo do DNg100
  configs/                     experimentos
```

**Licenças:** flybody Apache-2.0; dados e políticas do figshare GPL-3.0+; MaleCNS e BANC CC-BY. Nada disso fica versionado: tudo é baixado por script, e os créditos entram no README e no vídeo.

## Etapas e critérios de pronto

| Etapa | Pronto quando |
|---|---|
| M0 Ferramentas | flybody roda com `mujoco` puro no Windows e no Spark; replay de 5 s idêntico bit a bit; passos/s de física e de matriz esparsa medidos |
| M1 Física dos patins (decide se segue) | Nos dois modelos (lâmina e rodas): afunda <5% do raio; na rampa, acelera dentro de 10% do previsto, o que prova que não há efeito volante; derrapagem lateral <0,01 cm/s sob meia força de atrito; desacelera no deslize como previsto; um swizzle e um V programados à mão levam a mosca a ≥1 cm/s com deslizes de ≥0,2 s; cada patim alcança o giro necessário; mosca deslizando nunca ganha energia |
| M2 MLP patina | ≥80% de 20 testes com média ≥3 cm/s em 5 s, ≥25% do tempo deslizando e <10% de quedas; 50 clipes aleatórios auditados atrás de trapaças da física |
| M3 Conectoma montado | Grupos por pata conferidos contra as contagens publicadas; tabela músculo → junta pronta; rede estável sob ruído; com os parâmetros de Pugliese, o DNg100 gera ritmo de 7 a 15 Hz nos neurônios motores da coxa; custo por passo dentro do previsto |
| M4 Conectoma anda | ≥90% de caminhadas de 5 s bem-sucedidas a 1–3 cm/s; DNa02 faz virar |
| M5 Ensaio da captura | Um script independente regenera 3 tentativas sorteadas a partir do manifesto: física idêntica bit a bit, cérebro com diferença <1e-4 |
| M6 Treino final | Captura ligada desde a tentativa #1; atinge o critério do M2 em até 3× os passos do M2; slalom completo; provas causais funcionando |
| M7 Controles | Embaralhado, aleatório, MLP (e BANC, se der tempo): 3 sementes cada, mesmo orçamento, métricas registradas antes de rodar |
| M8 Vídeo | Roteiro, renderização final, painel do cérebro, textos e transparências |

## Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Crítica "isso é só uma IA comum", como aconteceu com a Eon e com o Digital Sphinx (que andou com o conectoma de um verme) | Sem camadas escondidas, decodificador com sinal fixo, provas causais, controles com o mesmo orçamento, e dizer quantos neurônios alcançam as patas |
| A fiação não aprende a andar nem com parâmetros treinados (quem tentou sem treino falhou) | O M3 e o M4 vêm cedo, com prazo; escada de flexibilidade; plano B de partir de "ficar em pé" |
| O PPO não descobre o deslize: fica parado, anda com os patins de lado ou explora defeitos de contato | MLP primeiro, bônus de rolamento, velocidade-alvo acima da de andar, currículo, auditoria de clipes, validação cruzada lâmina ↔ rodas |
| Física instável com patins minúsculos | Classe própria e valores explícitos; os testes do M1 decidem antes de qualquer treino |
| Controles não mostram diferença e a virada perde força | Apresentar como pergunta, com o resultado que sair |
| Spark trava por falta de memória | Limites de memória e lote dimensionados pelos benchmarks do M0 |
| Tentativa misturando versões da política, tremedeira do ruído, escolha seletiva de clipes | Tentativas completas por versão, ruído colorido, números pré-registrados, grade dos 20 testes |

## Transparência no vídeo

- Os patins são idealizados, sem a aderência que existe nessa escala.
- O cérebro é uma rede de taxa, não de neurônios que disparam.
- O vídeo diz o que foi treinado e como: por retropropagação, não pelo aprendizado da própria mosca.
- Um crítico MLP ajudou no treino; ajudas de currículo, acelerações e câmera lenta aparecem rotuladas.
- Só uma parte dos 166.700 neurônios entra no controle das patas.
- É um cérebro de macho num corpo de fêmea.
- Créditos: MaleCNS (Berg et al., Cell 2026, CC-BY), flybody (Vaxenburg et al., Nature 2025), Pugliese et al.

## Verificação

- **Testes automatizados:**
  - contagens dos grupos por pata (381 motores; T1/T2/T3 = 135/116/130);
  - sinais pelo `consensus_nt`;
  - máscaras por pata dos codificadores e decodificadores;
  - recompensa e queda.
- **Física:** os cenários do M1 viram testes com tolerância.
- **Ritmo:** reproduzir o ritmo do DNg100 e comparar com o código de Pugliese.
- **Determinismo:** replay bit a bit da física; cérebro re-simulado com diferença <1e-4 das ações gravadas.
- **Treino:** taxa de sucesso no percurso de teste com as 20 sementes fixas, e curvas de aprendizado de todos os braços com o mesmo orçamento.
- **Autenticidade:** um script confere, para cada clipe do vídeo, que o ID da tentativa, o checkpoint e a semente existem nos logs e reproduzem as mesmas imagens.

## Referências principais

- [flybody](https://github.com/TuragaLab/flybody) e o [artigo na Nature](https://www.nature.com/articles/s41586-025-09029-4); [figshare com dados e políticas](https://doi.org/10.25378/janelia.25309105)
- [MaleCNS: download](https://male-cns.janelia.org/download/) e [Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/)
- [Pugliese et al., ritmo de passos](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v2.full) e o [código](https://github.com/smpuglie/Pugliese_cpg_2025)
- [FlyGM](https://arxiv.org/abs/2602.17997); [therealfly](https://github.com/fruitflydev/therealfly); [fly-cord-robots](https://github.com/SakshayMahna/fly-cord-robots)
- [Humanoide de patins em linha da ETH](https://arxiv.org/abs/2606.31807); [quadrúpede de patins](https://arxiv.org/html/2603.18408); [Skaterbots](https://crl.ethz.ch/papers/Skaterbots.pdf)
- [Guia de PyTorch no DGX Spark](https://github.com/natolambert/dgx-spark-setup); [exportador USD do MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html)
