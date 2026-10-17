#!/usr/bin/env python
"""Does each GPU component of this project actually run on this machine's GPU?  Run it in sceptr_env on the server:

    python scripts/check_gpu_stack.py

Every check runs in its own process (a CUDA failure such as "no kernel image is available" can abort the whole
process), takes a few seconds, and ends with the flags that are safe to use. Checks:
  torch     PyTorch kernels on the GPU (matmul, cdist, top-k): SCEPTR --device cuda
  sceptr    a small SCEPTR embedding on the GPU against the CPU result
  xgboost   a small multi-class fit with device=cuda: --xgboost_device cuda in tuning / run_pipeline
  faiss     FAISS GPU add / search / k-means: --kmeans_device cuda
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])

CHECKS = {
"torch": """
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA device"
print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
print("arch list of this build:", torch.cuda.get_arch_list())
a = torch.rand(2000, 64, device="cuda"); b = torch.rand(5000, 64, device="cuda")
d = torch.cdist(a, b, p=1); v, i = torch.topk(d, 5, dim=1, largest=False)
m = (a @ a.T).sum(); torch.cuda.synchronize()
print("matmul / cdist / topk ran on the GPU")
""",
"sceptr": """
import sys
sys.path.insert(0, %r)
import numpy as np, pandas as pd, torch
import sceptr
assert torch.cuda.is_available(), "no CUDA device"
rng = np.random.default_rng(0); aa = list("ACDEFGHIKLMNPQRSTVWY")
tcrs = pd.DataFrame({"TRBV": ["TRBV10-1"] * 300, "CDR3B": ["C" + "".join(rng.choice(aa, 10)) + "F" for _ in range(300)]})
m = sceptr.variant.default(); m.disable_hardware_acceleration(); cpu = m.calc_vector_representations(tcrs)
m.enable_hardware_acceleration(); gpu = m.calc_vector_representations(tcrs)
print("max |GPU - CPU| over 300 TCRs:", float(np.abs(cpu - gpu).max()))
assert np.allclose(cpu, gpu, atol=1e-3), "GPU and CPU embeddings differ"
""" % ROOT,
"xgboost": """
import numpy as np, xgboost as xgb
assert xgb.build_info().get("USE_CUDA"), "this XGBoost build has no CUDA support (device=cuda would silently use the CPU)"
rng = np.random.default_rng(0); X = rng.random((2000, 20)); y = rng.integers(0, 3, 2000)
b = xgb.train({"objective": "multi:softprob", "num_class": 3, "tree_method": "hist", "device": "cuda"},
              xgb.DMatrix(X, label=y), 10)
p = b.predict(xgb.DMatrix(X)); assert p.shape == (2000, 3)
print("xgboost", xgb.__version__, "trained on the GPU")
""",
"faiss": """
import sys
import numpy as np
import faiss
if not (hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0):
    sys.exit("FAISS sees no GPU")
res = faiss.StandardGpuResources()
x = np.random.default_rng(0).random((2000, 16)).astype("float32")
for make in (faiss.IndexFlatL2, faiss.IndexFlatIP):
    g = faiss.index_cpu_to_gpu(res, 0, make(16))
    kind = type(g).__name__
    g.add(x)
    g.search(x[:10], 1)
    clus = faiss.Clustering(16, 8)
    clus.niter = 2
    clus.train(x, faiss.index_cpu_to_gpu(res, 0, make(16)))
print("add / search / k-means ran; index type on the GPU path:", kind)
if "Gpu" not in kind:
    print("NOTE: that is not a GPU index class: this may be a CPU-only build that falls back silently")
""",
}
FLAGS = {"torch": "PyTorch on the GPU (needed by SCEPTR --device cuda)", "sceptr": "run_embeddings.sh --device cuda",
         "xgboost": "--xgboost_device cuda (XGBoost in tune_hyperparameters / run_pipeline)",
         "faiss": "--kmeans_device cuda (otherwise leave the default, cpu)"}


def main():
    rows = []
    for name, code in CHECKS.items():
        t0 = time.time()
        try:
            res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
            ok, out = res.returncode == 0, (res.stdout + res.stderr).strip()
            if not ok:
                out = f"exit code {res.returncode}\n" + out[-700:]
        except subprocess.TimeoutExpired:
            ok, out = False, "timed out after 300 s"
        rows.append((name, ok, time.time() - t0))
        print(f"\n--- {name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f} s)")
        print("\n".join("    " + line for line in out.splitlines()[-12:]))
    print("\n=== summary")
    for name, ok, _ in rows:
        print(f"  {'OK  ' if ok else 'FAIL'} {name:8s} {'-> safe: ' if ok else '-> do not use: '}{FLAGS[name]}")
    sys.exit(0 if all(ok for _, ok, _ in rows) else 1)


if __name__ == "__main__":
    main()
