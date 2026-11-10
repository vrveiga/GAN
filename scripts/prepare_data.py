"""Baixa um subconjunto do CelebA (rostos recortados) e salva como array 64x64 uint8."""
import io
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image

REPO = "tglcourse/CelebA-faces-cropped-128"
N_SHARDS = 2
SIZE = 64
OUT = "data/faces64.npy"

files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
               if f.startswith("data/train-"))[:N_SHARDS]
arrays = []
for f in files:
    path = hf_hub_download(REPO, f, repo_type="dataset", local_dir="data/raw")
    table = pq.read_table(path)
    col = table.column(table.column_names[0]).to_pylist()
    for i, item in enumerate(col):
        img = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        arrays.append(np.asarray(img.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.uint8))
        if i % 5000 == 0:
            print(f, i, flush=True)
    del table, col
faces = np.stack(arrays)
np.save(OUT, faces)
print("salvo", OUT, faces.shape)
