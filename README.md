# O Falsificador e o Detetive

Apresentação interativa (5 min, em português) explicando GANs para o público leigo.
Uma DCGAN treinada na CPU gera rostos; o app usa imagens pré-geradas, sem servidor nem internet.

## 1. Preparar os dados (~35 mil rostos do CelebA, 64×64)
    uv run python scripts/prepare_data.py

## 2. Treinar (CPU, ~1,1 s/passo → 15 mil passos ≈ 4,5 h)
    nohup uv run python scripts/train.py --steps 15000 > runs/faces/train.log 2>&1 &
    tail -f runs/faces/train.log                      # acompanhar
    uv run python scripts/train.py --steps 15000 --resume   # retomar se parar

Acompanhe a evolução em `runs/faces/grids/` (uma grade por snapshot).

## 3. Exportar as imagens para o app
Escolha 4 passos para as 4 rodadas do jogo (do mais fácil ao mais difícil):

    uv run python scripts/export_assets.py --stages 100,600,2000,15000

## 4. Apresentar
Abra `app/index.html` no navegador (Chrome/Firefox) e tecle **F** para tela cheia.

| Tecla | Ação |
|---|---|
| → / PageDown | avançar (funciona com passador de slides) |
| ← / PageUp | voltar |
| 1–4 ou clique | escolher o rosto falso no jogo |
| Espaço | pausar/animar (timelapse e morph) |
| N | novo par de rostos (morph) |

Roteiro: título → história (5 passos) → jogo (4 rodadas + placar) → timelapse → morph → mundo real.
