# M8: roteiro da narração do corte v3 (24/10/2026)

Uma fala por cena do corte `docs/m8_corte_v3.json`. As legendas saem do texto, e a montagem estica cada cena até a
fala caber, então dá pra falar no seu ritmo. Mas, pra caber nos 2 minutos, a soma das falas deve ficar em até ~1 min
45 s. O tempo de referência é o da voz provisória.

## Como gravar

- **Um arquivo por cena:** `seg00` a `seg17`, em qualquer formato que o celular ou o gravador gere (wav, mp3, m4a…).
- **Pasta:** pode ser `outputs/narracao/gravada/` (fica fora do Git) ou qualquer outra; me diga qual.
- **Respiro nas pontas:** deixe ~0,5 s de silêncio antes e depois. O script corta o silêncio das pontas e filtra o ronco grave.
- **Volume:** não precisa ajustar. A montagem normaliza a voz e abaixa a trilha quando você fala.
- **Texto:** leia como está escrito na coluna "Fala" (números por extenso). A legenda mostra os números em algarismos.
- **Pra conferir o ritmo:** o corte v3 tem a voz provisória no lugar certo de cada fala.

Depois de gravar:

```
python scripts/m8_narrate.py docs/m8_corte_v3.json --engine files --dir outputs/narracao/gravada
```

Em seguida, troque `narration_dir` no plano para `outputs/narracao/files` e rode o `m8_assemble.py`.

## Falas

| Arquivo | Cena | Fala | Referência |
|---|---|---|---|
| seg00 | Gancho: a patinadora final, em tempo real | Essa mosca é virtual, mas os neurônios das patas dela são ligados igualzinho aos de uma mosca de verdade. | 5.9 s |
| seg01 | Gancho: em câmera lenta | E ela aprendeu a patinar. | 1.3 s |
| seg02 | Tentativa #1 | No começo foi assim: tentativa número um, e ela nem sai do lugar. | 3.9 s |
| seg03 | Close no patim | Os patins são lâminas que só deslizam pra frente… | 2.6 s |
| seg04 | Tentativa #9, a primeira queda | e na tentativa nove veio o primeiro tombo. | 2.2 s |
| seg05 | O cérebro (cartela com rótulos) | Por dentro, roda o cordão nervoso de uma mosca real, tipo a medula espinhal dela: 23 mil neurônios, mapeados um por um. As ligações são as dela; o treino só mexe na força de cada uma. | 12.0 s |
| seg06 | Ritmo sem treino | Sem treino nenhum, ligar um neurônio só já faz o circuito das patas pulsar no ritmo de andar. | 5.3 s |
| seg07 | Ela andando | Primeiro, ela aprendeu a andar. | 1.8 s |
| seg08 | Primeiras tentativas de patins | Nos patins, ela aprende na tentativa e erro, ganhando pontos por ir na velocidade pedida e por deslizar. | 6.0 s |
| seg09 | A brecha do primeiro treino (patas no ar em amarelo) | Só que, no primeiro treino, ela achou uma brecha: patinar com as patas no ar, essas em amarelo, dava mais pontos. Espertinha, né? | 8.6 s |
| seg10 | Curva dos patins no chão, com as trocas de regra | Daí a gente corrigiu a regra e recomeçou do zero, gravando cada tentativa. A cada rodada, 64 versões do cérebro tentam, e as melhores puxam a próxima. | 10.3 s |
| seg11 | Antes e depois (32 mil × 89 mil tentativas) | Mesmo assim, ela insistia numa pata no ar, até que cada patim no chão passou a valer ponto, e rolou um empurrãozinho no começo. Aí as patas desceram. | 9.0 s |
| seg12 | Clímax: câmera lenta e tempo real, com o rótulo dos 70 km/h | Depois de oitenta e nove mil e seiscentas tentativas, olha ela aí, a mais de três centímetros por segundo. Em tamanho de gente, isso dá uns setenta quilômetros por hora! | 10.3 s |
| seg13 | Os 20 testes | E não foi sorte: vinte de vinte testes, sem escolher nenhum. | 3.8 s |
| seg14 | DNg100 (“ande”) desligado | Quer a prova de que quem manda é o cérebro dela? Desliga os dois neurônios de “ande”… e ela para na hora. | 5.9 s |
| seg15 | DNa02 (“vire”) estimulado | Já o neurônio de “vire” vira a mosca pro lado errado. Fazer curva é o próximo desafio. | 5.3 s |
| seg16 | Gancho do conectoma embaralhado | E se as ligações fossem sorteadas? Esse treino já tá rodando… será que ela patina igual? | 5.6 s |
| seg17 | Fecho | Tudo simulado, nada encenado. Detalhes na descrição! | 3.9 s |

Total de referência: 104 s de fala.

A conta dos 70 km/h: 3,3 cm/s divididos pelos 0,297 cm do corpo dão ~11 corpos por segundo; numa pessoa de
1,75 m, ~19,5 m/s, ou ~70 km/h.
