# CLAUDE.md

Guia para sessões do Claude Code neste projeto. Responda em português do Brasil.

## O projeto

Uma mosca simulada (corpo do flybody no MuJoCo), controlada pelo conectoma do MaleCNS,
aprende a andar de patins; o vídeo final mostra tentativas reais ("Tentativa #N"). O plano
aprovado, com as etapas M0–M8 e os critérios de pronto, está em `docs/plano.md`; o estado de
cada etapa está no README.

## Onde roda

- Notebook Windows: código, testes rápidos, prévias. Ambiente em `.venv` (`.venv\Scripts\python`).
- DGX Spark do usuário (aarch64, CUDA 13, memória unificada): treino e renderização, com o
  repositório clonado em `~/mosca-patins` e ambiente criado pelo `uv`. A máquina é
  compartilhada com outros jobs do usuário: rode com `nice`, limite núcleos e memória
  (estourar a GPU trava a máquina inteira) e confira o espaço em disco antes de gravar logs.

## Comandos

- Testes: `python -m pytest -q`
- Assets (não versionados): `python scripts/download_assets.py --walking-sample --malecns --pugliese --malecns-stats`
- Física: `scripts/bench_physics.py`, `scripts/m1_skate_physics.py`, `scripts/m1_gait_search.py`
- Treino M2 (no Spark, com `nice`): `scripts/m2_train.py --run NOME`; avaliação: `scripts/m2_eval.py`
- Cérebro M3: `scripts/m3_vnc_weights.py` e `scripts/m3_build_graph.py` (Spark), `scripts/m3_rhythm_full.py`

## Regras do projeto

- Fidelidade: nenhum clipe do vídeo pode ser encenado. Cada tentativa precisa ser
  re-simulável a partir do que foi gravado; o replay da física deve continuar idêntico
  bit a bit (há teste para isso).
- Unidades CGS (cm, g, s), como no flybody. Não confiar nos padrões do MuJoCo para geoms
  novos (densidade, atrito de torção/rolamento, maciez de contato): tudo explícito.
- Mudanças de física passam pelos testes de contato do trenó (`tests/test_skate_contact.py`).
- Commit e push só quando o usuário pedir; mensagens de commit em português.

## Achados que orientam os próximos passos

- Velocidade de patim: `mj_objectVelocity` com `mjOBJ_XBODY` (referencial do patim). Com
  `mjOBJ_BODY` o MuJoCo usa o referencial de inércia, com os eixos reordenados; esse erro
  fez o M1 concluir que as marchas programadas "andavam de pato" (na verdade deslizam,
  rolamento 0,99) e estragou o bônus de rolamento das execuções A a E.
- A MLP da execução E desliza 98% do tempo com movimentos pequenos das patas (juntas
  oscilando 1–6°, 5–8 Hz); o resultado não muda com passo de física menor.
- As patas giram pouco o patim (±15° na frente, 5–15° só para fora no meio e atrás).
  Alavancas se o deslize não surgir: ângulo de montagem com a ponta para fora, comprimento
  do patim, atrito de rolamento.
- M2: antes de mudar hiperparâmetros, olhe a política (`m2_eval.py --sheet`, com e sem
  `--stochastic`). Duas armadilhas já vistas: ruído de exploração que faz a mosca andar
  sozinho (a política média fica parada) e termos de recompensa que pagam por ficar parada.
  A avaliação do treino usa a velocidade já liberada pelo currículo.
- M3: o modelo de Pugliese só oscila numa faixa estreita de excitabilidade. Contam só as
  sinapses do cordão nervoso (as do cérebro entre descendentes e ascendentes fazem a rede
  disparar), o volume vem da tabela deles ou da estimativa pelas sinapses, e a referência do
  volume é 1,30× a mediana do nosso grafo (com a mediana própria, silêncio; com 1,40×, disparo).
  O ritmo é de ~10 Hz, mas balanço e apoio da coxa ficam só ~0,2 de ciclo defasados (também
  na rede deles): no vídeo, falar em ritmo, não em alternância de passada.
- Controlador de conectoma: com a referência 1,30×, qualquer entrada sensorial faz a rede
  disparar; o controle usa 1,0×, proprioceptores 2 abaixo do limiar e tônus nos motores
  (sem o tônus a rede fica calada com a mosca parada e o PPO não aprende). Gradiente
  substituto na rede; `torch.clamp` zera o gradiente no limite (use passagem direta). Antes
  de mudar hiperparâmetros do PPO, meça a sensibilidade da ação a cada grupo de parâmetros:
  pesos sem escala num grupo dominam a divergência KL e travam a taxa dos outros.
- Professora do M4 (política de caminhada do flybody): as 59 ações vêm na ordem da política
  (adesão, cabeça, abdômen, patas), diferente da ordem dos atuadores do modelo; o remapeamento
  está em `mosca.walking.teacher.ACTION_ORDER`. Sem ele, a mosca cai na hora. Os sensores da
  observação são a média dos 10 subpassos de cada passo de controle (2 ms), como no flybody.
- Para saber onde um controlador de conectoma perde a informação, grave a atividade numa
  coleta conduzida pela professora e meça o teto linear (regressão das ações dela) a partir
  de cada camada: sentidos crus, proprioceptores, pré-motores, motores. No M4, a informação
  chegava aos pré-motores e sumia nos motores; parâmetros por tipo celular não resolveram.
- Latência decide o M4: a política do flybody só anda reagindo em menos de 10 ms (com as
  ações atrasadas 10 ms, ou filtradas com τ de 20 ms, todas as moscas caem em menos de 1 s).
  O comando dela é liga-desliga: 73% da variância é um chacoalhar que o filtro dos atuadores
  (τ 10 ms) apaga. O conectoma com τ de 20 ms responde aos sentidos das patas em 20–40 ms e só
  aprendeu a ficar em pé (anda_c: 100% sem cair, velocidade 0,02). Com τ×0,25 (5 ms), a
  imitação offline chega a 0,27 de erro filtrado, contra 0,72 (`m4_bc_probe.py`). A MLP
  destilada com 20 ms de atraso e 20 ms de filtro cai em 0,1 s; a professora lenta
  (`m4_slow_teacher.py`) parte da MLP sem latência e sobe a latência aos poucos no PPO.
- DAgger com a MLP: com o erro cru e a referência fixa, o aluno piora quando β cai (o erro é
  dominado pelo chacoalhar, e a professora manda "correr para alcançar" quando ele fica para
  trás); com o erro filtrado e a referência que acompanha a mosca (`--ref-leak-tau 0.2`), ele
  melhora. Avalie sempre pelo teste sem professora (% que chega aos 5 s e velocidade), não
  pelo erro.
- Conectoma que anda (τ×0,25, ganho por ligação, `--filtered-loss --ref-leak-tau 0.2`): segue a
  velocidade pedida, mas gira em círculos (não sente rotação) e ignorava o giro pedido porque os
  DNa01/DNa02 (neurônios grandes, limiar 90–200) paravam de disparar quando o ganho de giro caía no
  treino: use `--turn-gain 1500 --cmd-lr-mult 0`. Meça com `m4_eval.py` (velocímetro e giroscópio),
  não pela distância em linha reta do teste do treino, que pune quem anda em círculos. O embaralhado
  anda tão bem quanto o real e os ganhos mudam pouco (|log ganho| médio ~0,05): com neurônios
  rápidos, a rede funciona como um reservatório com entrada e saída treinadas.
