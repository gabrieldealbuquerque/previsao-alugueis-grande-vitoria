import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessamento import MIN_PREDICTED_RENT

def prediction_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), MIN_PREDICTED_RENT)
    abs_error = np.abs(y_pred - y_true)
    ape = abs_error / y_true
    mse = np.mean((y_pred - y_true) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(mse)),
        "medae": float(np.median(abs_error)),
        "mape": float(np.mean(ape) * 100),
        "mdape": float(np.median(ape) * 100),
        "within_15_pct": float(np.mean(ape <= 0.15) * 100),
        "r2": float(1 - ss_res / ss_tot) if ss_tot else 0.0,
    }


def safe_group_strata(frame, groups, min_count=5):
    raw = frame["cidade"].astype(str) + "|" + frame["tipo_base"].astype(str)
    grouped = pd.DataFrame({"cluster_hash": groups.astype(str), "stratum": raw}).drop_duplicates(
        "cluster_hash"
    )
    counts = grouped["stratum"].value_counts()
    grouped["stratum"] = grouped["stratum"].where(
        grouped["stratum"].map(counts).ge(min_count), "outros"
    )
    # Se nem a soma das classes raras forma cinco clusters, desativa a
    # estratificação categórica em vez de criar um estrato inválido.
    if grouped["stratum"].value_counts().min() < min_count:
        grouped["stratum"] = "geral"
    return groups.astype(str).map(grouped.set_index("cluster_hash")["stratum"])


def persistent_fold_ids(frame, groups, path, n_splits=5):
    """Mantém clusters antigos no mesmo fold e distribui apenas clusters novos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    strata = safe_group_strata(frame, groups, min_count=n_splits)
    clusters = pd.DataFrame(
        {"cluster_hash": groups.astype(str), "stratum": strata.astype(str)}
    ).drop_duplicates("cluster_hash")
    assignments = {}
    if path.exists():
        stored = pd.read_csv(path, dtype={"cluster_hash": str})
        valid = stored["fold"].between(0, n_splits - 1)
        assignments = dict(zip(stored.loc[valid, "cluster_hash"], stored.loc[valid, "fold"].astype(int)))

    fold_totals = np.zeros(n_splits, dtype=int)
    stratum_totals = {}
    current_clusters = set(clusters["cluster_hash"])
    for cluster, fold in assignments.items():
        if cluster in current_clusters:
            fold_totals[fold] += 1
    for stratum, subset in clusters.groupby("stratum"):
        counts = np.zeros(n_splits, dtype=int)
        for cluster in subset["cluster_hash"]:
            if cluster in assignments:
                counts[assignments[cluster]] += 1
        stratum_totals[stratum] = counts

    pending = clusters[~clusters["cluster_hash"].isin(assignments)].copy()
    pending["order"] = pending["cluster_hash"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    for row in pending.sort_values(["stratum", "order"]).itertuples(index=False):
        per_stratum = stratum_totals.setdefault(row.stratum, np.zeros(n_splits, dtype=int))
        candidates = np.flatnonzero(per_stratum == per_stratum.min())
        fold = min(candidates, key=lambda candidate: (fold_totals[candidate], candidate))
        assignments[row.cluster_hash] = int(fold)
        per_stratum[fold] += 1
        fold_totals[fold] += 1

    persisted = clusters.copy()
    persisted["fold"] = persisted["cluster_hash"].map(assignments).astype(int)
    persisted.sort_values("cluster_hash").to_csv(path, index=False)
    fold_ids = groups.astype(str).map(assignments).astype(int).to_numpy()
    if pd.DataFrame({"cluster": groups.astype(str), "fold": fold_ids}).groupby("cluster")["fold"].nunique().max() != 1:
        raise RuntimeError("Um cluster foi distribuído em mais de um fold.")
    return fold_ids
