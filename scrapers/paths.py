"""Caminhos compartilhados pelos coletores e scripts de dados."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "dados" / "brutos"
PROCESSED_DIR = ROOT_DIR / "dados" / "processados"
