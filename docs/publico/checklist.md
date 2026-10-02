# Checklist para abrir o repositório (GPLv3)

Primeira verificação em 01/10/2026 (`git ls-files`, `git log --all` e grep nos arquivos
rastreados). **Atualizado em 02/10/2026**, depois da preparação para abrir: só caminhos
abaixo; nenhum valor de credencial foi lido nem copiado para cá.

Decisões do usuário (02/10): abrir o **repositório atual, com o histórico inteiro** (sem
reescrever); publicar só a pesquisa **técnica**; preparar agora e **virar público só entre
08 e 10/10**, antes do prazo de 11/10. O organizador usa o último commit anterior ao prazo —
documentação depois do prazo não conta.

## Feito

- [x] **LICENSE presente e rastreado**: `LICENSE` é o texto completo da GNU GPL v3
  (35 KB), já versionado.
- [x] **Nenhum segredo versionado**: `.gitignore` cobre `.env`, `*.env`,
  `credentials.json`, `minio-credentials.json`, `data/`, `submissions/`, `runs/`,
  `logs/`. `git log --all -- .env minio-credentials.json credentials.json` volta
  vazio: esses arquivos nunca entraram no histórico.
- [x] **Arquivos que citam credenciais só citam os nomes das variáveis**:
  `.env.example` (`BUCKET_ACCESS_KEY`, `BUCKET_ACCESS_SECRET`, `TEAM_NAME`,
  `SUBMISSION_BUCKET`, `S3_ENDPOINT`) e `src/s3.py` (lê do ambiente). Sem valores.
- [x] **Sem arquivos grandes**: maior arquivo rastreado é `experiments.jsonl`
  (~300 KB); nenhum blob acima de 1 MB em todo o histórico. Dado do organizador e
  parquet de envio ficam fora do git.
- [x] **Dados externos documentados** (exigência do prêmio): tabela com fonte,
  licença e comando de download no `README.md` (§4) e em `NOTICE`.
- [x] **Caminho de disco local embutido no código** (01/10): `src/adsb.py` usa
  `data/adsb` como padrão (pode ser um link para outro disco) e `PRC_ADSB_RAIZ` troca o
  lugar. Não há mais ponto de montagem com UUID da máquina no código.
- [x] **README público em inglês**: `docs/publico/README.en.md` virou o `README.md` do
  repositório (links relativos corrigidos: `LICENSE`, `NOTICE`, `docs/…`), atualizado para a
  campeã atual (três corretores sobre `20260930-130008-e76_base`) e para a melhor nota
  oficial (243,9755, v36). Ganhou as seções de **desenho do modelo** (§1), **validação** (§2,
  incluindo a régua A/B por metades de dias e a política de envios) e **"What did not work"**
  (§3, negativos medidos com números e links para `docs/research/`). Todo comando e flag
  documentado foi conferido contra o `--help` real dos scripts.
- [x] **Diário em pt-BR**: o `README.md` antigo virou `docs/diario.md`, sem as seções de
  placar, de Discord e de leitura de repositórios de concorrentes; o conteúdo técnico
  (decisões de método, experimentos, resultados na simulação e comandos) ficou.
- [x] **Scripts descartáveis de `.superpowers/`**: os 10 arquivos que continuavam
  rastreados saíram do índice (`git rm --cached`); `.gitignore` já cobria o diretório.
- [x] **Notas internas fora do índice** (`git rm --cached` + `.gitignore`, arquivos
  continuam no disco e os geradores seguem funcionando): `docs/auditoria/`,
  `docs/projecao.md`, `saltos.json`, `placar/`, e as pesquisas que são leitura de Discord e
  de repositórios de concorrentes (`docs/research/2026-09-26-praticas-ml.md`,
  `2026-09-27-concorrentes.md`, `2026-10-01-repos-concorrentes.md`). `docs/esteira.md` fica:
  é relatório técnico da esteira e o serviço `prc-esteira` faz commit dele.
- [x] **Aviso de copyright**: `NOTICE` com titular e ano (`Copyright (C) 2026 team
  outgoing-boat`), o parágrafo padrão da GPLv3 e as atribuições de dados; referenciado no
  `README.md` (§7).
- [x] **Aviso de copyright do `apt.dat`** (X-Plane, GPL): o aviso exigido pela
  especificação está em `NOTICE`. Não redistribuímos o `apt.dat` (fica em `data/`, fora do
  git) nem trechos dele em `src/mapa.py` ou em `docs/`.
- [x] **Referências mortas no código e nos documentos novos**: `src/pista.py`, `src/roma.py`,
  `docs/research/2026-10-02-roma-t-d-g.md`, `docs/research/2026-10-02-espelho-chegadas.md`,
  `docs/mapa.md` e `docs/diario.md` não apontam mais para caminhos que saíram do índice.
  Os planos antigos de `docs/superpowers/plans/` e duas pesquisas de setembro ainda citam
  esses nomes — ficam na lista de pendências abaixo, junto com o conteúdo deles.

## Pendências antes de publicar

- [ ] **Virar a visibilidade para pública** em `https://github.com/EnioAguiar/prc-taxiout-2026`
  (hoje privado) **entre 08 e 10/10**, e confirmar que a organização do desafio vai forkar
  esse endereço. Nada foi empurrado para o GitHub nesta preparação.
- [ ] **Caminho pessoal em `experiments.jsonl`**: 18 ocorrências de `/home/enio` (campos de
  caminho de arquivos de corrida). Decidir entre limpar os caminhos ou aceitar que o nome de
  usuário apareça. Não é segredo; é só exposição do nome da conta local.
- [x] **Material interno misturado a conteúdo técnico (decidido em 02/10):** o repositório abre com
  todo o histórico, então essas leituras já ficam visíveis nos commits antigos. Sai da versão final
  só `docs/research/2026-09-24-competicoes-e-placar.md` (foto do placar e leitura de READMEs de
  outras equipes; fica no disco, no `.gitignore`). **Ficam**, porque mostram de onde veio cada
  passo do método: o spec do pipeline, os planos 4 e 8–12, `literatura-taxiout.md`,
  `forense-dados.md`, `diagnostico-v4.md`, `docs/mapa.md` e `ferramentas/projecao.py`.
- [ ] **Atualizar a campeã e o README entre 08 e 10/10**: se a esteira promover membros
  novos até lá, refazer §1.3 (`champion.json`), a tabela de versões e a nota oficial do
  `README.md` **antes** do prazo — o organizador usa o último commit anterior a 11/10.
- [ ] **Último passo antes de abrir**: rodar `bin/run -m pytest tests/ -q` e
  `ferramentas/auditoria.py`, conferir que o `champion.json` publicado corresponde à melhor
  nota enviada e repetir o grep de credenciais. Hoje
  `git ls-files -z | xargs -0 grep -lniE 'secret|access_key|token'` devolve
  `.env.example`, `README.md`, `docs/publico/checklist.md` e `src/s3.py` — todos só com
  **nomes** de variáveis, nenhum valor.
