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

## 3. Modelo grande pré-treinado (rodada final e morph)
Progressive GAN (Karras et al., 2018) treinada no CelebA-HQ 512×512, com pesos do pytorch_GAN_zoo.
O gerador foi reimplementado em `scripts/pgan.py`; carregamos só os tensores (`weights_only=True`).

    curl -L -o data/pretrained/pgan_celebahq512.pth \
      https://dl.fbaipublicfiles.com/gan_zoo/PGAN/celebaHQ16_december_s7_i96000-9c72988c.pth

Fotos reais em alta (mesmo recorte da PGAN), um único arquivo do CelebA-HQ:

    uv run python -c "from huggingface_hub import hf_hub_download as d; d('Chris1/celebA-HQ', 'data/train-00000-of-00046.parquet', repo_type='dataset', local_dir='data/raw_hq')"

## 4. Escolher os melhores falsos (curadoria)
Gera candidatos e folhas de contato numeradas em `data/candidates/<conjunto>/sheet_*.png`:

    uv run python scripts/curate.py ours --n 600 --show 96   # nosso DCGAN, com truncamento
    uv run python scripts/curate.py pgan --n 96              # modelo grande

Anote os números escolhidos em `curation.json` (`ours`, `pgan`, `title`, `morph`).

## 5. Exportar as imagens para o app
Rodadas: 200 e 800 passos (falsos aleatórios), nosso modelo final (curado) e o modelo grande.

    uv run python scripts/export_assets.py --stages 200,800

## 6. Apresentar
Abra `app/index.html` no navegador (Chrome/Firefox) e tecle **F** para tela cheia.

| Tecla | Ação |
|---|---|
| → / PageDown | avançar (funciona com passador de slides) |
| ← / PageUp | voltar |
| 1–4 ou clique | escolher o rosto falso no jogo |
| Espaço | pausar/animar (timelapse e morph) |
| N | novo par de rostos (morph) |

Roteiro: título → história (5 passos) → jogo (4 rodadas + placar) → timelapse → morph → mundo real.
