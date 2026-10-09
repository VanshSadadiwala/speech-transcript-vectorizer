"""Execution pipeline and evaluation runner.

Usage (from the project root):
    python -m src.pipeline                       # runs on data/annotated/intents.csv
    python -m src.pipeline --data my.csv --out results

Input CSV needs a `text` column (raw transcript) and, for the classification
experiment, an `intent` column.  Outputs go to the results/ folder:
    preprocessing_stats.csv, config_comparison.csv, threshold_sweep.csv,
    top_collocations.csv, feature_matrix_sample.csv, summary.json and PNG charts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .preprocessing import PreprocessConfig, TranscriptPreprocessor, preprocessing_stats
    from .models import AugmentedBoWVectorizer, CollocationExtractor
except ImportError:  # executed as a plain script: python src/pipeline.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import PreprocessConfig, TranscriptPreprocessor, preprocessing_stats
    from src.models import AugmentedBoWVectorizer, CollocationExtractor

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "annotated" / "intents.csv"
SEED = 42

# default feature settings used across the project
DEFAULTS = dict(min_df=1, min_freq=5, pmi_threshold=3.0, ngram_sizes=(2,))


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "text" not in df.columns:
            df = df.rename(columns={df.columns[0]: "text"})
    else:  # one transcript per line
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame({"text": lines})
    return df.dropna(subset=["text"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
CONFIGS = {
    "BoW only": dict(use_collocations=False),
    "BoW + ALL bigrams (no PMI filter)": dict(use_collocations=True, min_freq=1, pmi_threshold=-1e9, ngram_sizes=(2,)),
    "BoW + PMI bigrams": dict(use_collocations=True, ngram_sizes=(2,)),
    "BoW + PMI bigrams + trigrams": dict(use_collocations=True, ngram_sizes=(2, 3)),
}


def _make_vec(overrides: dict) -> AugmentedBoWVectorizer:
    kw = dict(DEFAULTS)
    kw.update(overrides)
    return AugmentedBoWVectorizer(**kw)


def cross_validate(token_docs, y, vec_kwargs, folds=5, low_data=False):
    """Stratified CV. The vectorizer (incl. PMI) is fitted on the TRAIN part only
    (no leakage).  low_data=True swaps the roles: train on 1 fold, test on 4."""
    y = np.asarray(y)
    docs = np.asarray(token_docs, dtype=object)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    accs, f1s, n_feats = [], [], []
    for a, b in skf.split(docs, y):
        tr, te = (b, a) if low_data else (a, b)
        vec = _make_vec(vec_kwargs)
        Xtr = vec.fit_transform(list(docs[tr]))
        Xte = vec.transform(list(docs[te]))
        clf = LinearSVC(random_state=SEED).fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        n_feats.append(Xtr.shape[1])
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)), float(np.mean(n_feats))


def compare_configs(token_docs, y=None) -> pd.DataFrame:
    rows = []
    for name, kw in CONFIGS.items():
        vec = _make_vec(kw)
        X = vec.fit_transform(token_docs)
        s = vec.stats(X)
        row = {"configuration": name, "bow_features": s["bow_features"],
               "collocation_features": s["collocation_features"],
               "total_features": s["total_features"], "sparsity_pct": s["sparsity_pct"]}
        if y is not None:
            a, sd, f1, _ = cross_validate(token_docs, y, kw)
            la, _, lf1, _ = cross_validate(token_docs, y, kw, low_data=True)
            row.update({"cv_accuracy": round(a, 4), "cv_acc_std": round(sd, 4), "cv_macro_f1": round(f1, 4),
                        "lowdata_accuracy": round(la, 4), "lowdata_macro_f1": round(lf1, 4)})
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_sweep(token_docs, y=None, thresholds=(-1, 0, 1, 2, 3, 4, 5, 6, 7), min_freq=5) -> pd.DataFrame:
    full = CollocationExtractor(min_freq=min_freq, pmi_threshold=-1e9, ngram_sizes=(2,)).fit(token_docs)
    n_bow = len(AugmentedBoWVectorizer(use_collocations=False).fit(token_docs).bow_feature_names_)
    rows = []
    for t in thresholds:
        n_col = int((full.scored_["pmi"] >= t).sum())
        row = {"pmi_threshold": t, "min_freq": min_freq, "collocation_features": n_col,
               "total_features": n_bow + n_col}
        if y is not None:
            kw = dict(use_collocations=True, min_freq=min_freq, pmi_threshold=t, ngram_sizes=(2,))
            row["cv_accuracy"] = round(cross_validate(token_docs, y, kw)[0], 4)
            row["lowdata_accuracy"] = round(cross_validate(token_docs, y, kw, low_data=True)[0], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def min_freq_sweep(token_docs, values=(1, 2, 3, 5, 8), threshold=3.0) -> pd.DataFrame:
    rows = []
    for m in values:
        ce = CollocationExtractor(min_freq=m, pmi_threshold=threshold).fit(token_docs)
        rows.append({"min_freq": m, "pmi_threshold": threshold, "collocation_features": len(ce.selected_)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
def make_charts(out: Path, sweep: pd.DataFrame, top: pd.DataFrame, comp: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(sweep["pmi_threshold"], sweep["total_features"], marker="o", color="#0E9F8E")
    ax.set_xlabel("PMI threshold"); ax.set_ylabel("Total feature columns", color="#0E9F8E")
    ax.set_title("PMI threshold vs feature-matrix width (min_freq=5)")
    if "cv_accuracy" in sweep:
        ax2 = ax.twinx()
        ax2.plot(sweep["pmi_threshold"], sweep["cv_accuracy"], marker="s", color="#F59E0B", label="5-fold")
        ax2.plot(sweep["pmi_threshold"], sweep["lowdata_accuracy"], marker="^", ls="--", color="#B45309", label="low-data")
        ax2.set_ylabel("Accuracy"); ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "chart_threshold_sweep.png", dpi=150); plt.close(fig)

    t = top.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(t["phrase"], t["pmi"], color="#0F1B2D")
    ax.set_xlabel("PMI (bits)"); ax.set_title("Top collocations by PMI")
    fig.tight_layout(); fig.savefig(out / "chart_top_collocations.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(comp)); w = 0.4
    ax.bar(x - w / 2, comp["bow_features"], w, label="BoW columns", color="#0F1B2D")
    ax.bar(x + w / 2, comp["collocation_features"], w, label="Collocation columns", color="#0E9F8E")
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" + ", "\n+ ").replace(" (", "\n(") for c in comp["configuration"]], fontsize=7)
    ax.set_ylabel("Columns"); ax.set_title("Feature columns per configuration"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "chart_config_features.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# main runner
# ---------------------------------------------------------------------------
def run_pipeline(data_path=DEFAULT_DATA, out_dir=ROOT / "results", verbose=True) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)
    y = df["intent"].values if "intent" in df.columns else None
    if verbose:
        print(f"Loaded {len(df)} transcripts" + (f", {len(set(y))} intents" if y is not None else ""))

    pre = TranscriptPreprocessor(PreprocessConfig())
    docs = pre.process_many(df["text"])
    pstats = preprocessing_stats(list(df["text"]), docs)
    pd.DataFrame([pstats]).to_csv(out / "preprocessing_stats.csv", index=False)
    if verbose:
        print("Preprocessing:", pstats)

    # main augmented matrix
    vec = AugmentedBoWVectorizer(**DEFAULTS)
    X = vec.fit_transform(docs)
    mstats = vec.stats(X)
    top = vec.collocations_.selected_df_.copy()
    top.to_csv(out / "top_collocations.csv", index=False)
    vec.to_dataframe(X.tocsr()[:25]).to_csv(out / "feature_matrix_sample.csv", index=False)
    if verbose:
        print("Matrix:", mstats)
        print(top.head(10).to_string(index=False))

    comp = compare_configs(docs, y)
    comp.to_csv(out / "config_comparison.csv", index=False)
    sweep = threshold_sweep(docs, y)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    mf = min_freq_sweep(docs)
    mf.to_csv(out / "min_freq_sweep.csv", index=False)
    make_charts(out, sweep, top if len(top) else vec.collocations_.scored_, comp)

    summary = {"preprocessing": pstats, "matrix": mstats, "defaults": {k: list(v) if isinstance(v, tuple) else v for k, v in DEFAULTS.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print("\nConfiguration comparison:\n", comp.to_string(index=False))
        print("\nThreshold sweep:\n", sweep.to_string(index=False))
        print(f"\nAll results saved in: {out}")
    return {"summary": summary, "comparison": comp, "sweep": sweep, "top": top}


def main():
    ap = argparse.ArgumentParser(description="BoW + PMI collocation feature pipeline")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    run_pipeline(a.data, a.out)


if __name__ == "__main__":
    main()
