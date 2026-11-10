"""Gera as imagens usadas pela apresentação a partir dos snapshots treinados.

Saída em app/assets/:
  real/real_XX.png                 - rostos reais do dataset
  game/stage<S>/fake_XX.png        - falsos de cada estágio do treino (rodadas do jogo)
  timelapse/frame_XXX.png          - grade de rostos com ruído fixo ao longo do treino
  morph/pair<P>_XX.png             - interpolação no espaço latente (slider)
  manifest.json                    - lista de tudo acima + curva de acurácia do detetive

Uso:
  uv run python scripts/export_assets.py --stages 100,600,2000,15000
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from train import NZ, Discriminator, Generator

OUT = Path("app/assets")
SCALE = 4  # 64px -> 256px (suavizado) para ficar nítido no monitor


def to_pil(t, scale=SCALE):
    """tensor 3xHxW em [-1,1] -> PIL ampliada."""
    arr = ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * scale, img.height * scale), Image.LANCZOS)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/faces")
    p.add_argument("--data", default="data/faces64.npy")
    p.add_argument("--stages", required=True, help="passos usados como rodadas do jogo, ex: 100,600,2000,15000")
    p.add_argument("--fakes-per-stage", type=int, default=12)
    p.add_argument("--n-real", type=int, default=48)
    p.add_argument("--morph-pairs", type=int, default=4)
    p.add_argument("--morph-frames", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    run = Path(a.run)
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = {}

    snaps = sorted(run.glob("snapshots/G_step*.pt"), key=lambda f: int(re.findall(r"\d+", f.name)[0]))
    step_of = {int(re.findall(r"\d+", f.name)[0]): f for f in snaps}
    stages = [int(s) for s in a.stages.split(",")]
    final_step = max(step_of)

    # Detetive final: usado para escolher os falsos mais convincentes de cada estágio
    D = None
    if (run / "latest.pt").exists():
        D = Discriminator()
        D.load_state_dict(torch.load(run / "latest.pt")["D"])
        D.eval()

    with torch.no_grad():
        # --- rostos reais ---
        data = np.load(a.data, mmap_mode="r")
        (OUT / "real").mkdir(parents=True)
        idx = rng.choice(len(data), a.n_real, replace=False)
        manifest["real"] = []
        for i, k in enumerate(idx):
            img = Image.fromarray(np.asarray(data[k]))
            img = img.resize((64 * SCALE, 64 * SCALE), Image.LANCZOS)
            name = f"real/real_{i:02d}.png"
            img.save(OUT / name)
            manifest["real"].append(name)

        # --- falsos por estágio (rodadas do jogo) ---
        manifest["stages"] = []
        for s in stages:
            G = load_G(step_of[s])
            z = torch.randn(a.fakes_per_stage * 4, NZ)
            imgs = G(z)
            if D is not None:  # pega os que o detetive final acha mais "reais"
                order = D(imgs).argsort(descending=True)
                imgs = imgs[order]
            imgs = imgs[: a.fakes_per_stage]
            d = OUT / "game" / f"stage{s:06d}"
            d.mkdir(parents=True)
            names = []
            for i, t in enumerate(imgs):
                name = f"game/stage{s:06d}/fake_{i:02d}.png"
                to_pil(t).save(OUT / name)
                names.append(name)
            manifest["stages"].append({"step": s, "fakes": names})

        # --- timelapse: os mesmos 8 ruídos em todos os snapshots ---
        fixed_z = torch.load(run / "fixed_z.pt")[:8]
        (OUT / "timelapse").mkdir(parents=True)
        manifest["timelapse"] = []
        for i, (s, f) in enumerate(sorted(step_of.items())):
            G = load_G(f)
            grid = make_grid(G(fixed_z), nrow=4, padding=2, pad_value=-1.0)
            name = f"timelapse/frame_{i:03d}.png"
            to_pil(grid, scale=3).save(OUT / name)
            manifest["timelapse"].append({"step": s, "img": name})

        # --- morph no espaço latente (gerador final) ---
        G = load_G(step_of[final_step])
        (OUT / "morph").mkdir(parents=True)
        manifest["morph"] = []
        cand = torch.randn(64, NZ)
        if D is not None:
            cand = cand[D(G(cand)).argsort(descending=True)]
        for pi in range(a.morph_pairs):
            za, zb = cand[2 * pi], cand[2 * pi + 1]
            frames = []
            for fi, t in enumerate(np.linspace(0, 1, a.morph_frames)):
                img = G(slerp(za, zb, float(t)).unsqueeze(0))[0]
                name = f"morph/pair{pi}_{fi:02d}.png"
                to_pil(img).save(OUT / name)
                frames.append(name)
            manifest["morph"].append(frames)

    # --- curva de acurácia do detetive ---
    log_path = run / "log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text())
        manifest["accuracy"] = [[e["step"], e["accD"]] for e in log[::4]]

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # também como JS, para o app abrir direto do disco (file://) sem servidor
    (OUT / "manifest.js").write_text("window.MANIFEST = " + json.dumps(manifest) + ";\n")
    print("ok:", OUT, {k: len(v) for k, v in manifest.items()})


if __name__ == "__main__":
    main()
