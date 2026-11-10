"""Gera candidatos de rostos falsos e folhas de contato numeradas para escolha manual.

  ours: nosso DCGAN final, com truncamento; ordenado pelo nosso detetive treinado
  pgan: Progressive GAN pré-treinada (CelebA-HQ 512), sem truncamento

Saída em data/candidates/<conjunto>/:
  z.pt          - ruídos (latentes) de cada candidato, na ordem das folhas
  sheet_N.png   - folhas de contato com o número de cada candidato

Depois, anote os escolhidos em curation.json, por exemplo:
  {"ours": [3, 7, 12, ...], "pgan": [0, 5, ...], "morph": [[1, 9], [4, 22], ...]}

Uso:
  uv run python scripts/curate.py ours --n 600 --show 96
  uv run python scripts/curate.py pgan --n 96
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import pgan
from train import NZ, Discriminator, Generator

PER_SHEET = 48  # 8 x 6
COLS = 8


def to_pil(t, size):
    t = F.interpolate(t.unsqueeze(0), size=size, mode="area" if t.shape[-1] > size else "bilinear",
                      align_corners=None if t.shape[-1] > size else False)[0]
    arr = ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def sheets(imgs, out, size):
    pad = 4
    for s in range(0, len(imgs), PER_SHEET):
        chunk = imgs[s:s + PER_SHEET]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (size + pad) + pad, rows * (size + pad) + pad), "black")
        d = ImageDraw.Draw(sheet)
        for i, im in enumerate(chunk):
            x, y = pad + (i % COLS) * (size + pad), pad + (i // COLS) * (size + pad)
            sheet.paste(im, (x, y))
            d.rectangle((x, y, x + 30, y + 16), fill="black")
            d.text((x + 3, y + 2), str(s + i), fill="yellow")
        sheet.save(out / f"sheet_{s // PER_SHEET}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["ours", "pgan"])
    p.add_argument("--n", type=int, default=600, help="quantos gerar")
    p.add_argument("--show", type=int, default=96, help="quantos mostrar nas folhas")
    p.add_argument("--psi", type=float, default=None, help="truncamento (1 = sem)")
    p.add_argument("--run", default="runs/faces")
    p.add_argument("--step", type=int, default=15000)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    out = Path("data/candidates") / a.which
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("sheet_*.png"):
        f.unlink()

    with torch.no_grad():
        if a.which == "ours":
            psi = a.psi if a.psi is not None else 0.75
            G = Generator()
            G.load_state_dict(torch.load(Path(a.run) / "snapshots" / f"G_step{a.step:06d}.pt"))
            G.eval()
            D = Discriminator()
            D.load_state_dict(torch.load(Path(a.run) / "latest.pt")["D"])
            D.eval()
            z = torch.randn(a.n, NZ) * psi
            imgs = torch.cat([G(c) for c in z.split(100)])
            order = torch.cat([D(c) for c in imgs.split(100)]).argsort(descending=True)[: a.show]
            z, imgs = z[order], imgs[order]
            pils = [to_pil(t, 128) for t in imgs]
            size = 128
        else:
            psi = 1.0  # a PGAN normaliza o ruído na entrada: truncar por escala não tem efeito
            G = pgan.load_pgan()
            z = torch.randn(min(a.n, a.show), pgan.NZ) * psi
            pils = []
            for c in z.split(8):
                pils += [to_pil(t, 160) for t in G(c)]
                print(len(pils), "/", len(z), flush=True)
            size = 160

    torch.save({"z": z, "psi": psi}, out / "z.pt")
    sheets(pils, out, size)
    print("ok:", out, len(pils), "candidatos")


if __name__ == "__main__":
    main()
