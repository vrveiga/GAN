"""Gera as imagens usadas pela apresentação.

Rodadas do jogo:
  - estágios do NOSSO DCGAN (ex.: 200, 800) com falsos aleatórios daquele momento do treino
  - estágio final do nosso DCGAN com os falsos escolhidos à mão (curation.json -> "ours")
  - rodada final com a Progressive GAN pré-treinada (curation.json -> "pgan"),
    comparada com fotos reais do CelebA-HQ na mesma resolução

Saída em app/assets/ + manifest.json / manifest.js.

Uso:
  uv run python scripts/export_assets.py --stages 200,800
"""
import argparse
import io
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid

import pgan
from train import NZ, Generator

OUT = Path("app/assets")
CARD = 256  # tamanho das imagens do jogo


def to_pil(t, size=None):
    """tensor 3xHxW em [-1,1] -> PIL; redimensiona para `size` (lado) se dado."""
    if size and t.shape[-1] != size:
        mode = "area" if t.shape[-1] > size else "bilinear"
        t = F.interpolate(t.unsqueeze(0), size=size, mode=mode,
                          **({} if mode == "area" else {"align_corners": False}))[0]
    arr = ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def upscale(img, size=CARD):
    return img.resize((size, size), Image.LANCZOS)


def degrade(img, res):
    """Reduz para res x res e amplia de volta para CARD, igual às imagens do nosso DCGAN."""
    if res:
        img = img.resize((res, res), Image.LANCZOS)
    return img.resize((CARD, CARD), Image.LANCZOS)


def load_G(path):
    G = Generator()
    G.load_state_dict(torch.load(path))
    return G.eval()


def slerp(a, b, t):
    """Interpolação esférica: mais natural que linear para ruído gaussiano."""
    an, bn = a / a.norm(), b / b.norm()
    omega = torch.acos((an * bn).sum().clamp(-1, 1))
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def save(img, name, **kw):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/faces")
    p.add_argument("--data", default="data/faces64.npy")
    p.add_argument("--hq", default="data/raw_hq/data/train-00000-of-00046.parquet")
    p.add_argument("--curation", default="curation.json")
    p.add_argument("--stages", default="200,800", help="estágios iniciais do nosso DCGAN (falsos aleatórios)")
    p.add_argument("--per-stage", type=int, default=12)
    p.add_argument("--n-real", type=int, default=48)
    p.add_argument("--morph-frames", type=int, default=40)
    p.add_argument("--pgan-res", type=int, default=64,
                   help="resolução efetiva da PGAN e das fotos HQ (64 = igual ao nosso modelo; 0 = sem reduzir)")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    run = Path(a.run)
    cur = json.loads(Path(a.curation).read_text())
    if OUT.exists():
        shutil.rmtree(OUT)
    M = {}

    step_of = {int(re.findall(r"\d+", f.name)[0]): f for f in run.glob("snapshots/G_step*.pt")}
    final_step = max(step_of)

    with torch.no_grad():
        # ---- fotos reais: CelebA 64px (mesmo recorte do nosso modelo) ----
        data = np.load(a.data, mmap_mode="r")
        M["real"] = [save(upscale(Image.fromarray(np.asarray(data[k]))), f"real/real_{i:02d}.png")
                     for i, k in enumerate(rng.choice(len(data), a.n_real, replace=False))]

        # ---- fotos reais: CelebA-HQ 256px (mesmo recorte da PGAN) ----
        col = pq.read_table(a.hq, columns=["image"]).column("image")
        idx = rng.choice(len(col), a.n_real, replace=False)
        M["real_hq"] = []
        for i, k in enumerate(idx):
            img = Image.open(io.BytesIO(col[int(k)].as_py()["bytes"])).convert("RGB")
            M["real_hq"].append(save(degrade(img, a.pgan_res), f"real_hq/real_{i:02d}.png"))

        M["stages"] = []
        # ---- estágios iniciais do nosso DCGAN: amostras aleatórias, sem escolher ----
        for s in [int(x) for x in a.stages.split(",")]:
            G = load_G(step_of[s])
            imgs = G(torch.randn(a.per_stage, NZ))
            fakes = [save(upscale(to_pil(t)), f"game/ours{s:06d}/fake_{i:02d}.png") for i, t in enumerate(imgs)]
            M["stages"].append({"model": "ours", "step": s, "fakes": fakes, "realset": "real"})

        # ---- nosso DCGAN final: escolhidos à mão ----
        ck = torch.load("data/candidates/ours/z.pt")
        G = load_G(step_of[final_step])
        imgs = G(ck["z"][cur["ours"]])
        fakes = [save(upscale(to_pil(t)), f"game/ours_final/fake_{i:02d}.png") for i, t in enumerate(imgs)]
        M["stages"].append({"model": "ours", "step": final_step, "fakes": fakes, "realset": "real"})

        # ---- Progressive GAN pré-treinada ----
        P = pgan.load_pgan()
        zp = torch.load("data/candidates/pgan/z.pt")["z"]
        game_ids = [i for i in cur["pgan"] if i not in cur["title"]]  # quem aparece no título não entra no jogo
        imgs = torch.cat([P(c) for c in zp[game_ids].split(8)])
        fakes = [save(degrade(to_pil(t, CARD), a.pgan_res), f"game/pgan/fake_{i:02d}.png") for i, t in enumerate(imgs)]
        M["stages"].append({"model": "big", "step": None, "fakes": fakes, "realset": "real_hq"})

        M["title"] = [save(degrade(to_pil(t, CARD), a.pgan_res), f"title/face_{i}.jpg", quality=92)
                      for i, t in enumerate(P(zp[cur["title"]]))]

        # ---- morph (PGAN, 512px) ----
        M["morph"] = []
        for pi, (ia, ib) in enumerate(cur["morph"]):
            za, zb = zp[ia], zp[ib]
            zs = torch.stack([slerp(za, zb, float(t)) for t in np.linspace(0, 1, a.morph_frames)])
            frames = torch.cat([P(c) for c in zs.split(8)])
            M["morph"].append([save(degrade(to_pil(t, CARD), a.pgan_res) if a.pgan_res else to_pil(t),
                                    f"morph/pair{pi}_{fi:02d}.jpg", quality=90)
                               for fi, t in enumerate(frames)])
            print("morph", pi + 1, "/", len(cur["morph"]), flush=True)

        # ---- timelapse: os mesmos 8 ruídos em todos os snapshots do nosso DCGAN ----
        fixed_z = torch.load(run / "fixed_z.pt")[:8]
        M["timelapse"] = []
        for i, (s, f) in enumerate(sorted(step_of.items())):
            grid = make_grid(load_G(f)(fixed_z), nrow=4, padding=2, pad_value=-1.0)
            img = to_pil(grid)
            img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
            M["timelapse"].append({"step": s, "img": save(img, f"timelapse/frame_{i:03d}.png")})

    log_path = run / "log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        M["accuracy"] = [[e["step"], e["accD"]] for e in log[::4]]

    (OUT / "manifest.json").write_text(json.dumps(M, indent=1))
    # também como JS, para o app abrir direto do disco (file://) sem servidor
    (OUT / "manifest.js").write_text("window.MANIFEST = " + json.dumps(M) + ";\n")
    print("ok:", OUT, {k: len(v) for k, v in M.items()})


if __name__ == "__main__":
    main()
