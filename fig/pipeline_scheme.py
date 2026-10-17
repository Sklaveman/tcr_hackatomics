"""Block scheme of the final pipeline (SCEPTR + Fisher filter + FAISS k-means + XGBoost).
Run:  python fig/pipeline_scheme.py   -> fig/pipeline_scheme.{png,pdf,svg}"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
C = {"data": ("#e8f1fa", "#3b6ea5"), "embed": ("#e9f5ec", "#3f8f5a"), "label": ("#fdf0e1", "#c7782c"),
     "model": ("#f1e9f7", "#7a4fa3"), "out": ("#fbe9e9", "#b04a4a"), "tune": ("#f3f3f3", "#666666")}

fig, ax = plt.subplots(figsize=(17, 10.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")


def box(x, y, w, h, title, body="", kind="data", fs=8.6, tfs=9.8, left=False):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=ec)
    if body:
        ax.text(x + (1.5 if left else w / 2), y + h - 3.7, body, ha="left" if left else "center", va="top",
                fontsize=fs, color="#222222", linespacing=1.35)


def arrow(p, q, rad=0.0, color="#444444", lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2))


# ---------------- row A: data -> embeddings ----------------
yA, hA = 58.5, 10.5
box(1, yA, 19, hA, "Raw repertoires", "train: 411 (labelled)\ntest_bulk: 117\ntest_sc: 128\n(CDR3 aa, V, J, clone_count)", "data")
box(26, yA, 22, hA, "Shared preprocessing", "canonical CDR3 only; junk / ambiguous\nV,J calls and rows without V or J dropped;\n14 V genes canonicalised;\nduplicates collapsed → key cdr3_aa+V_J", "data")
box(54, yA, 22, hA, "SCEPTR embedding", "pretrained, beta chain + V gene\n64-d, unit norm; rows with V genes\nthe model rejects (~1.3 %) are dropped\nscripts/generate_sceptr_embeddings.py", "embed")
box(82, yA, 17, hA, "Embedding store", "<sample_id>_embed.csv\nper_patient_tcr_embed.h5\nmetadata.tsv, config\n(one folder per dataset)", "embed")
for a, b in ((20, 26), (48, 54), (76, 82)):
    arrow((a + 0.4, yA + hA / 2), (b - 0.4, yA + hA / 2))

# ---------------- row B: per-split chain ----------------
yB, hB = 27.0, 23.5
ax.add_patch(FancyBboxPatch((1, yB), 98, hB, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#fcfcfc", ec="#888888", lw=1.4, ls=(0, (5, 3))))
ax.text(50, yB + hB - 1.0, "One split = one CV fold (tuning) or all training samples (final fit).  Every label-based step is fitted on that split's training samples only",
        ha="center", va="top", fontsize=10.2, fontweight="bold", color="#555555")
w, gap, x0, yb, hb = 14.4, 1.9, 2.3, yB + 2.2, 16.4
steps = [
    ("1  Fisher filter", "one-vs-rest one-sided Fisher\ntest on CDR3 k-mers\n(k = 3, ends trimmed)\nkeep best p % per class\n→ surviving training\nsequences", "label"),
    ("2  FAISS k-means", "K centroids fitted on\nthe survivors\ncosine distance (SCEPTR)\nCPU (GPU optional)", "model"),
    ("3  Cluster radius", "per cluster: q-quantile of\nmember–centroid distances\nclusters with < min size\nassigned members dropped,\nrest renumbered", "model"),
    ("4  Assignment", "every sequence of every\nsample (also test samples)\n→ nearest centroid,\ninside its radius, else\n'unassigned'", "model"),
    ("5  Sample features", "share of a sample's\nclonotypes per cluster\n+ unassigned, evenness,\nlog_depth;\nclass-aware prevalence\nfilter", "label"),
    ("6  XGBoost", "3-class softprob\nbalanced sample weights\ninner early stopping\n→ P(autoimmunity,\ncancer, infection)", "model"),
]
for i, (t, b, k) in enumerate(steps):
    x = x0 + i * (w + gap)
    box(x, yb, w, hb, t, b, k, fs=8.3, tfs=9.6)
    if i:
        arrow((x - gap + 0.2, yb + hb / 2), (x - 0.2, yb + hb / 2))
ax.text(x0 + 0.2, yB + 0.6, "orange = uses labels (fitted on training samples of the split)     purple = model fitted on the split",
        fontsize=8.2, color="#666666", va="bottom")

arrow((90.5, yA - 0.3), (90.5, yB + hB + 0.5), color="#3f8f5a")
ax.text(91.3, (yA + yB + hB) / 2 - 0.2, "embeddings of train,\ntest_bulk, test_sc", fontsize=8.6, color="#3f8f5a", va="center")

# ---------------- row C: tuning and final fit ----------------
yC, hC = 1.2, 20.0
box(1, yC, 46, hC, "Hyperparameter tuning  (tune_hyperparameters.py / run_tuning.sh)",
    "CV folds built once (stratified, 3-class); the whole chain above is refit in every fold\n\n"
    "Stage 1   grid: Fisher strictness × K × radius quantile q (fixed classifier)\n"
    "              score = macro one-vs-rest ROC-AUC → best combination wins\n"
    "Stage 2   Optuna search of the XGBoost parameters on the cached fold features\n"
    "Closing  combined configuration re-evaluated by CV (out-of-fold metrics)\n\n"
    "Output   hyperparameters.json, cv_metrics, stage-1 leaderboard", "tune", fs=8.6, left=True)
box(53, yC, 46, hC, "Final model  (run_pipeline.py --stage fit_predict)",
    "refit the chain with the tuned parameters on ALL training samples\n(Fisher filter on, same as in tuning)\n\n"
    "predict test_bulk and test_sc  (+ test_sc booster without log_depth)\n"
    "SHAP: cluster importance and direction of the effect\n\n"
    "Output   predictions_test_bulk.tsv, predictions_test_sc.tsv,\n"
    "              cluster_importance.csv, saved cluster model + booster", "out", fs=8.6, left=True)
arrow((24, yB - 0.3), (24, yC + hC + 0.4), color="#666666")
ax.text(25, (yB + yC + hC) / 2 - 0.3, "refit per fold", fontsize=8.6, color="#666666", va="center")
arrow((76, yB - 0.3), (76, yC + hC + 0.4), color="#b04a4a")
ax.text(77, (yB + yC + hC) / 2 - 0.3, "refit once", fontsize=8.6, color="#b04a4a", va="center")
arrow((47.4, yC + hC / 2), (52.6, yC + hC / 2), color="#555555", lw=2.2)
ax.text(50, yC + hC / 2 + 1.6, "tuned\nparameters", fontsize=8.2, ha="center", color="#555555")

ax.text(50, 71.4, "TCR repertoire → disease group: SCEPTR + Fisher filter + FAISS k-means + XGBoost", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#222222")
fig.savefig(OUT / "pipeline_scheme.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "pipeline_scheme.pdf", bbox_inches="tight")
fig.savefig(OUT / "pipeline_scheme.svg", bbox_inches="tight")
print("saved", OUT)
