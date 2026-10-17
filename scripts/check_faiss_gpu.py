"""Server check: does the GPU give the same answers as the CPU? Usage: python scripts/check_faiss_gpu.py <data_dir> [device]
(device defaults to cuda; use 'cpu' only to test this script)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from src import clustering as km
from src import embeddings as emb

data_dir, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cuda")
km.check_available(device)
d = emb.embedding_dir(data_dir, "sceptr", "train")
ids = emb.load_dataset_meta(d)["sample_id"].tolist()[:20]
keys, E = emb.concat_embeddings(emb.h5_path(d), ids)
print(f"{len(keys):,} sequences, dim {E.shape[1]}")
for spherical in (True, False):
    base = km.fit_base(keys, E, 200, spherical, device=device, seed=1)
    i_cpu, d_cpu = km.nearest_centroid(base["centroids"], E, spherical, "cpu")
    i_dev, d_dev = km.nearest_centroid(base["centroids"], E, spherical, device)
    print(f"spherical={spherical}: nearest-centroid agreement {np.mean(i_cpu == i_dev):.5f}, "
          f"max |distance difference| {np.abs(d_cpu - d_dev).max():.2e}")
