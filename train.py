"""Treina o modelo único (XGBoost + comparáveis) e gera o joblib de produção."""

import argparse
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.preprocessamento import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_model_input_contract,
    prepare_model_frame,
    property_cluster_groups,
)
from src.modelo import (
    ApartmentComparableBlendRegressor,
    AreaScaledWeightedLogTargetRegressor,
    HouseComparableBlendRegressor,
    build_xgboost_regressor,
)
from src.validacao import (
    persistent_fold_ids,
    prediction_metrics,
)

MODEL_NAME = "xgb_comparables"
MAX_OOF_MAPE = 15.6
N_FOLDS = 5
ROOT_DIR = Path(__file__).resolve().parent


def build_model():
    """Um XGBoost com comparáveis determinísticos para casas e apartamentos."""
    house_model = HouseComparableBlendRegressor(
        AreaScaledWeightedLogTargetRegressor(build_xgboost_regressor()),
    )
    return ApartmentComparableBlendRegressor(house_model)


def main():
    parser = argparse.ArgumentParser(description="Treina o modelo unico de previsao de aluguel.")
    parser.add_argument("--dataset", default=ROOT_DIR / "dados/processados/dataset_alugueis_reclassificado.csv")
    parser.add_argument("--folds", default=ROOT_DIR / "dados/validacao/folds_reclassificado.csv")
    parser.add_argument("--predictions-out", default=ROOT_DIR / "dados/validacao/predicoes_modelo.csv")
    parser.add_argument("--results-out", default=ROOT_DIR / "dados/validacao/resultados_modelo.csv")
    parser.add_argument("--model-out", default=ROOT_DIR / "modelo_aluguel.joblib")
    args = parser.parse_args()

    started = time.time()
    raw = pd.read_csv(args.dataset)
    raw["_contract_row_id"] = np.arange(len(raw))
    frame = prepare_model_frame(raw, training=True).reset_index(drop=True)
    frame = apply_model_input_contract(frame, raw)
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN].to_numpy(float)
    groups = property_cluster_groups(frame)
    folds = persistent_fold_ids(frame, groups, args.folds, n_splits=N_FOLDS)

    oof = np.empty(len(frame), dtype=float)
    for fold in range(N_FOLDS):
        train, validation = folds != fold, folds == fold
        fitted = clone(build_model()).fit(x.loc[train], y[train])
        oof[validation] = np.maximum(fitted.predict(x.loc[validation]), 1)
        print(f"Fold {fold + 1}/{N_FOLDS}: MAPE {prediction_metrics(y[validation], oof[validation])['mape']:.2f}%")

    metrics = prediction_metrics(y, oof)
    if metrics["mape"] > MAX_OOF_MAPE:
        raise RuntimeError(
            f"MAPE OOF {metrics['mape']:.4f}% excedeu o teto de {MAX_OOF_MAPE:.1f}%; joblib nao foi salvo"
        )

    house = frame["tipo_base"].astype(str).eq("Casa").to_numpy()
    rows = [{"modelo": MODEL_NAME, "fold": fold, **prediction_metrics(y[folds == fold], oof[folds == fold])}
            for fold in range(N_FOLDS)]
    rows.append({"modelo": MODEL_NAME, "fold": "OOF", **metrics})
    rows.append({"modelo": f"{MODEL_NAME}_casas", "fold": "OOF", **prediction_metrics(y[house], oof[house])})
    rows.append({"modelo": f"{MODEL_NAME}_apartamentos", "fold": "OOF", **prediction_metrics(y[~house], oof[~house])})
    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results_out, index=False)

    output = frame[["id", "cidade", "bairro_norm", "tipo_base", TARGET_COLUMN]].copy()
    output["cluster_hash"], output["fold"], output[MODEL_NAME] = groups.to_numpy(), folds, oof
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.predictions_out, index=False)

    final_model = clone(build_model()).fit(x, y)
    artifact = {
        "model": final_model,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "chosen_model": MODEL_NAME,
        "metrics": metrics,
        "training_rows_raw": int(len(raw)),
        "training_rows_fit": int(len(frame)),
        "folds": N_FOLDS,
        "training_seconds": float(time.time() - started),
        "runtime_versions": {
            package: version(package)
            for package in ("xgboost", "scikit-learn", "joblib", "numpy", "pandas", "scipy")
        },
    }
    joblib.dump(artifact, args.model_out, compress=3)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Tempo total: {artifact['training_seconds']:.1f}s")
    print(f"Modelo salvo em: {args.model_out}")


if __name__ == "__main__":
    main()
