# Pesquisa: como treinar o conectoma a patinar (M6), 29/09/2026

Motivo: com o decodificador anatômico (cada neurônio motor só mexe a junta do seu músculo, com sinal
fixo), o PPO não tira o conectoma do ponto fixo nos patins (136 iterações), a imitação offline da MLP
patinadora estaciona em 1 − R² ~2 e o teto linear pelo decodificador anatômico é 0,88 (pelos 381 motores,
livre, 0,25). Patins travados também não funcionam. Pesquisa em três frentes, feita na web.

## O que outros fizeram com conectomas num corpo

- **FlyGM** (arXiv 2602.17997): conectoma do cérebro (FlyWire) como grafo fixo no flybody; cada neurônio
  tem um estado atualizado por uma MLP compartilhada. Imitação da política de caminhada do flybody e depois
  PPO. O decodificador é aprendido (lê os neurônios eferentes), não anatômico. Venceu grafos embaralhados.
- **flyvis** (Lappalainen et al., Nature 2024): só 734 parâmetros treinados (por tipo celular), BPTT, com
  leitura aprendida.
- **BAAIWorm** (C. elegans, 2024): para a locomoção em malha fechada, trocou o mapa neuromuscular anatômico
  por uma matriz linear aprendida de todos os motores para os músculos.
- **Olivares, Izquierdo & Beer** (C. elegans, 2021): algoritmo evolutivo ajustando os parâmetros de um
  cordão anatomicamente restrito em malha fechada com o corpo.
- **NeuroMechFly v2 / Eon**: o conectoma produz comandos descendentes que acionam controladores de baixo
  nível feitos à mão ou aprendidos; não acionam neurônios motores diretamente.
- **flykski** (GitHub WKDev/flykski; só o README conferido): mosca de esqui com o MaleCNS, treinada por
  estratégias evolutivas; relata desempenho igual para conectoma real, embaralhado e aleatório.
- **nfly** (MaleCNS, 8,4 milhões de ganhos): PPO puro resolve o CartPole e falha no Pong; o autor conclui que
  "a atribuição de crédito é o problema em aberto".
- Nenhum controlador de conectoma com corpo encontrado usa decodificador anatômico com sinal fixo por junta:
  todos usam leituras aprendidas que juntam muitos neurônios (inferido do desenho deles; ninguém publicou a
  comparação).

## Por que o PPO trava aqui

- Dinâmica caótica (Metz et al., arXiv 2111.05803): em sistemas caóticos, o gradiente pela recorrência
  explode e nenhum corte resolve; estimadores de caixa-preta (estratégias evolutivas, REINFORCE) têm variância
  menor. É o nosso quadro: norma do gradiente ~1e8 e picos de KL.
- Ruído nas ações com atuadores antagonistas (extensor/flexor) explora mal; ruído em espaço latente ou
  correlacionado entre atuadores explora melhor (Lattice, github.com/amathislab/lattice; DEP-RL, arXiv 2206.00484).
- Estratégias evolutivas só aprendem se as perturbações mudam o comportamento (Salimans et al., arXiv
  1703.03864): antes de tudo, medir se alguma variação sai do ponto fixo.

## Métodos candidatos, mantendo o decodificador anatômico

1. **Estratégias evolutivas num espaço reduzido** (OpenAI-ES ou PGPE+ClipUp): perturbar parâmetros por
   tipo/neurônio, tônus e ganhos do decodificador (milhares, não o milhão de ganhos por ligação), com
   amostragem antitética e notas pela ordem. Cada ambiente do lote roda uma variação (os parâmetros por
   neurônio cabem no lote; os ganhos por ligação não). Não precisa de gradiente pela recorrência. Custo
   estimado 10–100× o da MLP (300–1000 gerações de 64–256). Precedentes: Evolving Connectivity (NeurIPS 2023,
   rede de disparos com lei de Dale resolve o Humanoid), Olivares/Izquierdo/Beer. No vídeo: "a cada geração,
   64 moscas tentam; as melhores guiam a próxima" (e cada tentativa continua re-simulável pela semente).
2. **PPO com menos parâmetros e exploração melhor**: congelar os ganhos por ligação (o grupo que domina a KL
   e o caos), treinar parâmetros por tipo, tônus, codificador e decodificador (como FlyGM e flyvis), BPTT curto,
   e ruído correlacionado no espaço dos neurônios motores levado às juntas pelo decodificador (estilo Lattice).
3. **Guia pelo estado, sem copiar ações** (se 1 e 2 não saírem do lugar): recompensa por mover o corpo e os
   patins como a MLP patinadora (velocidade, rumo, altura; nenhum ângulo de junta, estilo DeepMimic), com
   partidas em estados dela (RSI) e JSRL. No vídeo: "era premiado por se mover como uma treinadora artificial,
   sem ninguém dizer que junta mexer".
4. **Forças auxiliares que somem** ("rodinhas"): honesto e sem professora, mas precisa de um ponto de partida
   que se mova.

Descartados como método principal: kickstarting e DAPG (exigem as mesmas ações e recriam o platô),
simulação diferenciável (MuJoCo Warp ainda não tem gradiente; portar para MJX/JAX é caro e o caos continua).

## Nota sobre o decodificador livre

Na literatura, a leitura aprendida é o padrão (FlyGM, flyvis, BAAIWorm). Mas, com ela, a rede tende a
funcionar como um reservatório (o embaralhado anda tão bem quanto o real já no M4, e o flykski relata o
mesmo), e o mérito passa para a leitura. Por decisão do usuário, ela não é o resultado principal; pode
entrar como braço de controle declarado no M7.
