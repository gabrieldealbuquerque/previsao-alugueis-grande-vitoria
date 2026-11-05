import argparse
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessamento import FEATURE_COLUMNS, MIN_PREDICTED_RENT, prepare_model_frame


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT_DIR / "examples/entrada.csv"
DEFAULT_OUTPUT = ROOT_DIR / "resultado.csv"
DEFAULT_MODEL = ROOT_DIR / "modelo_aluguel.joblib"
EXPECTED_XGBOOST_VERSION = "3.3.0"


def validate_runtime_versions():
    installed = version("xgboost")
    if installed != EXPECTED_XGBOOST_VERSION:
        raise RuntimeError(
            "Versao incompativel do XGBoost: "
            f"instalada={installed}, esperada={EXPECTED_XGBOOST_VERSION}. "
            "Instale as dependencias fixadas em requirements.txt antes de "
            "carregar modelo_aluguel.joblib."
        )


def main():
    parser = argparse.ArgumentParser(description="Gera predicoes de aluguel para imoveis da Grande Vitoria.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    validate_runtime_versions()
    artifact = joblib.load(args.model)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    feature_columns = artifact.get("features", FEATURE_COLUMNS) if isinstance(artifact, dict) else FEATURE_COLUMNS

    raw = pd.read_csv(input_path)
    ids = raw["id"].astype(str) if "id" in raw.columns else pd.Series([f"amostra_{i+1}" for i in range(len(raw))])
    prepared = prepare_model_frame(raw, training=False)
    x = prepared.reindex(columns=feature_columns)
    predictions = np.maximum(model.predict(x), MIN_PREDICTED_RENT)

    result = pd.DataFrame(
        {
            "id": ids.to_numpy(),
            "valor_previsto": np.rint(predictions).astype(int),
        }
    )
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
