# M8: roteiro da narração do corte v3 (24/10/2026)

Texto falado de cada cena do corte `docs/m8_corte_v3.json` (1 min 56 s com a voz provisória). As legendas saem
desse texto. Pra gravar ou trocar a voz, mantenha uma fala por cena; a montagem estica cada cena até a fala caber.

| Início | Cena | Fala |
|---|---|---|
| 0:00 | Gancho: a patinadora final, em tempo real | Essa mosca é virtual, mas os neurônios das patas dela são ligados igualzinho aos de uma mosca de verdade. |
| 0:06 | Gancho: em câmera lenta | E ela aprendeu a patinar. |
| 0:08 | Tentativa #1 | No começo foi assim: tentativa número um, e ela nem sai do lugar. |
| 0:13 | Close no patim | Os patins são lâminas que só deslizam pra frente… |
| 0:16 | Tentativa #9, a primeira queda | e na tentativa nove veio o primeiro tombo. |
| 0:21 | O cérebro (cartela com rótulos) | Por dentro, roda o cordão nervoso de uma mosca real, tipo a medula espinhal dela: 23 mil neurônios, mapeados um por um. As ligações são as dela; o treino só mexe na força de cada uma. |
| 0:33 | Ritmo sem treino | Sem treino nenhum, ligar um neurônio só já faz o circuito das patas pulsar no ritmo de andar. |
| 0:39 | Ela andando | Primeiro, ela aprendeu a andar. |
| 0:41 | Primeiras tentativas de patins | Nos patins, ela aprende na tentativa e erro, ganhando pontos por ir na velocidade pedida e por deslizar. |
| 0:48 | A brecha do primeiro treino (patas no ar em amarelo) | Só que, no primeiro treino, ela achou uma brecha: patinar com as patas no ar, essas em amarelo, dava mais pontos. Espertinha, né? |
| 0:57 | Curva dos patins no chão, com as trocas de regra | Daí a gente corrigiu a regra e recomeçou do zero, gravando cada tentativa. A cada rodada, 64 versões do cérebro tentam, e as melhores puxam a próxima. |
| 1:08 | Antes e depois (32 mil × 89 mil tentativas) | Mesmo assim, ela insistia numa pata no ar, até que cada patim no chão passou a valer ponto, e rolou um empurrãozinho no começo. Aí as patas desceram. |
| 1:17 | Clímax: câmera lenta e tempo real, com o rótulo dos 70 km/h | Depois de oitenta e nove mil e seiscentas tentativas, olha ela aí, a mais de três centímetros por segundo. Em tamanho de gente, isso dá uns setenta quilômetros por hora! |
| 1:28 | Os 20 testes | E não foi sorte: vinte de vinte testes, sem escolher nenhum. |
| 1:33 | DNg100 (“ande”) desligado | Quer a prova de que quem manda é o cérebro dela? Desliga os dois neurônios de “ande”… e ela para na hora. |
| 1:39 | DNa02 (“vire”) estimulado | Já o neurônio de “vire” vira a mosca pro lado errado. Fazer curva é o próximo desafio. |
| 1:45 | Gancho do conectoma embaralhado | E se as ligações fossem sorteadas? Esse treino já tá rodando… será que ela patina igual? |
| 1:51 | Fecho | Tudo simulado, nada encenado. Detalhes na descrição! |

**Como trocar a voz:**

- voz neural da Microsoft: `python scripts/m8_narrate.py docs/m8_corte_v3.json --engine edge --voice pt-BR-AntonioNeural`;
- áudios gravados (um por cena, `seg00.wav` a `seg17.wav`): `python scripts/m8_narrate.py docs/m8_corte_v3.json --engine files --dir PASTA`;

Depois disso, troque `narration_dir` no plano e rode o `m8_assemble.py`.

A conta dos 70 km/h: 3,3 cm/s divididos pelos 0,297 cm do corpo dão ~11 corpos por segundo; numa pessoa de
1,75 m, ~19,5 m/s, ou ~70 km/h.
