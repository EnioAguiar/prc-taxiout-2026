# Checklist para abrir o repositório (GPLv3)

Verificado em 01/10/2026 com `git ls-files`, `git log --all` e grep nos arquivos
rastreados (129 arquivos). Só caminhos abaixo; nenhum valor de credencial foi lido
nem copiado para cá.

## OK (nada a fazer)

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
  (292 KB); nenhum blob acima de 1 MB em todo o histórico. Dado do organizador e
  parquet de envio ficam fora do git.
- [x] **Dados externos documentados** (exigência do prêmio): tabela com fonte,
  licença e comando de download em `docs/publico/README.en.md` e no `README.md`.

## Pendências antes de publicar

- [ ] **README público em inglês**: publicar `docs/publico/README.en.md` como
  `README.md` do repositório público (ou linkado em destaque). O `README.md` atual é
  um diário de trabalho em pt-BR, com estratégia, placar, notas de Discord e passos
  de inscrição — revisar o que fica. Ao mover para a raiz, trocar o link
  `../../LICENSE` por `LICENSE`.
- [ ] **Caminho pessoal em `experiments.jsonl`**: 18 ocorrências de `/home/enio`
  (campos de caminho de arquivos de corrida). Decidir entre limpar os caminhos ou
  aceitar que apareça o nome de usuário.
- [ ] **Caminho de disco local embutido no código**: `src/adsb.py:32` usa como
  padrão um ponto de montagem com UUID da máquina
  (`/mnt/<uuid>/prc-adsb`) quando `PRC_ADSB_RAIZ` não está definido. Trocar o padrão
  por algo relativo (ex.: `data/adsb`) ou exigir a variável — hoje quem clonar o
  repositório recebe um caminho que não existe.
- [ ] **Scripts descartáveis rastreados em `.superpowers/`**: apesar de
  `.gitignore` ter `.superpowers/`, 10 arquivos continuam versionados
  (`.superpowers/noite2/cacador3.py`, `cacador4.py`, `copia_multi.sh`,
  `detetive2.py`, `ens2.sh`, `ocupante.py`, `resultados.txt`, `run.sh`,
  `sem_adsb_proprio.sh`, `sem_lirf.sh`). São experimentos descartados; decidir entre
  remover do índice (`git rm --cached`) ou manter como histórico de pesquisa.
- [ ] **Notas internas a decidir**: `docs/auditoria/*.md` (5 auditorias diárias da
  máquina, com nomes de serviços e estado do PC), `docs/projecao.md` e `saltos.json`
  (projeção de placar e fila de apostas), `docs/esteira.md`,
  `docs/research/2026-09-27-concorrentes.md` e
  `docs/research/2026-09-26-praticas-ml.md` (leitura de Discord e de repositórios de
  concorrentes). Nada aqui é segredo técnico, mas é material interno — escolher o
  que publicar.
- [ ] **Aviso de copyright**: a GPLv3 pede o cabeçalho com titular e ano. Hoje ele só
  existe no README em inglês ("Copyright © 2026 team `outgoing-boat`"); avaliar
  incluir o cabeçalho curto nos arquivos de `src/` ou um `NOTICE`/seção equivalente.
- [ ] **Aviso de copyright do `apt.dat`** (X-Plane, GPL): a especificação pede manter
  o aviso ao redistribuir. Não redistribuímos o `apt.dat` (fica em `data/`, fora do
  git) — confirmar que nada em `src/mapa.py` ou em `docs/` traz trechos do dado.
- [ ] **Repositório de destino**: `origin` aponta para
  `https://github.com/EnioAguiar/prc-taxiout-2026` (privado). Confirmar visibilidade
  pública e que a organização do desafio vai forkar esse endereço.
- [ ] **Último passo antes de abrir**: rodar `pytest -q` e `ferramentas/auditoria.py`,
  conferir que `champion.json` publicado corresponde à melhor nota enviada e repetir
  o grep de credenciais (`git ls-files -z | xargs -0 grep -lniE 'secret|access_key|token'`
  deve devolver apenas `.env.example`, `LICENSE` e `src/s3.py`).
