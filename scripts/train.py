"""Treina uma DCGAN 64x64 em rostos (CelebA) na CPU.

Salva:
  runs/<nome>/snapshots/G_step<N>.pt   - pesos do gerador em momentos escolhidos
  runs/<nome>/grids/step<N>.png        - grade 8x8 com ruído fixo (para acompanhar)
  runs/<nome>/log.json                 - perdas e acurácia do discriminador
  runs/<nome>/latest.pt                - checkpoint completo para retomar (--resume)

Uso:
  uv run python scripts/train.py --steps 15000
  uv run python scripts/train.py --steps 15000 --resume
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

NZ = 100  # tamanho do vetor de ruído (espaço latente)


class Generator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NZ, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),       # 4x4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),          # 32x32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),                                         # 64x64
        )

    def forward(self, z):
        return self.net(z.view(-1, NZ, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # logit
        )

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def snapshot_steps(max_steps):
    """Passos quase logarítmicos: muitos no início (quando tudo muda rápido), poucos no fim."""
    s = {0, 10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000}
    s |= set(range(4000, max_steps + 1, 1000))
    return {x for x in s if x <= max_steps} | {max_steps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/faces64.npy")
    p.add_argument("--run", default="runs/faces")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bench", type=int, default=0, help="só mede o tempo de N passos e sai")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    run = Path(a.run)
    (run / "snapshots").mkdir(parents=True, exist_ok=True)
    (run / "grids").mkdir(parents=True, exist_ok=True)

    data = torch.from_numpy(np.load(a.data)).permute(0, 3, 1, 2).contiguous()  # N,3,64,64 uint8
    n = len(data)
    print(f"{n} rostos carregados")

    G, D = Generator(), Discriminator()
    G.apply(init_weights)
    D.apply(init_weights)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, NZ, generator=torch.Generator().manual_seed(1234))
    torch.save(fixed_z, run / "fixed_z.pt")

    step, log = 0, []
    if a.resume and (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt")
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        step, log = ck["step"], ck["log"]
        print(f"retomando do passo {step}")

    snaps = snapshot_steps(a.steps)

    def save_snapshot(s):
        G.eval()
        with torch.no_grad():
            imgs = G(fixed_z)
        G.train()
        save_image(imgs, run / "grids" / f"step{s:06d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        torch.save(G.state_dict(), run / "snapshots" / f"G_step{s:06d}.pt")

    def save_latest():
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(),
                    "optD": optD.state_dict(), "step": step, "log": log}, run / "latest.pt")
        (run / "log.json").write_text(json.dumps(log))

    if step == 0 and not a.bench:
        save_snapshot(0)

    ones = torch.ones(a.batch)
    zeros = torch.zeros(a.batch)
    t0 = time.time()
    end = step + a.bench if a.bench else a.steps
    while step < end:
        idx = torch.randint(0, n, (a.batch,))
        real = data[idx].float().div_(127.5).sub_(1.0)
        flip = torch.rand(a.batch) < 0.5
        real[flip] = real[flip].flip(-1)  # espelhamento horizontal aleatório

        # --- Detetive (discriminador): aprende a separar real de falso ---
        z = torch.randn(a.batch, NZ)
        fake = G(z)
        d_real, d_fake = D(real), D(fake.detach())
        lossD = bce(d_real, ones * 0.9) + bce(d_fake, zeros)  # rótulo suave 0.9 estabiliza
        optD.zero_grad(set_to_none=True)
        lossD.backward()
        optD.step()

        # --- Falsificador (gerador): aprende a enganar o detetive ---
        lossG = bce(D(fake), ones)
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        optG.step()
        step += 1

        if step % 25 == 0:
            acc = ((d_real > 0).float().mean() + (d_fake < 0).float().mean()).item() / 2
            log.append({"step": step, "lossD": round(lossD.item(), 4),
                        "lossG": round(lossG.item(), 4), "accD": round(acc, 4)})
        if step % 100 == 0:
            el = time.time() - t0
            print(f"passo {step}/{end}  D={lossD.item():.3f}  G={lossG.item():.3f}  "
                  f"{el:.0f}s decorridos", flush=True)
        if a.bench:
            continue
        if step in snaps:
            save_snapshot(step)
        if step % 500 == 0:
            save_latest()

    el = time.time() - t0
    if a.bench:
        print(f"BENCH: {a.bench} passos em {el:.1f}s -> {el / a.bench:.3f}s/passo")
        return
    save_latest()
    print(f"fim: {step} passos em {el / 60:.1f} min")


if __name__ == "__main__":
    main()
