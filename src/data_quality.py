import re
import unicodedata

import numpy as np
import pandas as pd


DAILY_STRONG_RE = re.compile(
    r"temporada|temporaria|temporario|diaria|diarias|/\s*dia\b|[-_]dia\b|\d+\s*dia\b|por\s+dia\b|hospedagem|airbnb|pernoite|"
    r"locacao diaria|aluguel diario|pacote.{0,30}(?:feriado|carnaval|reveillon)",
    re.I,
)
DAILY_WEAK_RE = re.compile(r"feriado|reveillon|carnaval|semana santa|fim de semana|final de semana|veraneio", re.I)
MONTHLY_RE = re.compile(
    r"/\s*m[eê]s\b|por\s+m[eê]s\b|mensal|loca[cç][aã]o anual|aluguel anual|"
    r"contrato.{0,30}(?:12|doze)\s+meses|moradia fixa|residencial anual|"
    r"contrato\s+anual|contrato.{0,30}(?:1|2|um|dois)\s+anos?\b|"
    r"contrato.{0,30}cartorio|contrato\s+de\s+aluguel|alugo\s+somente\s+com\s+contrato|"
    r"aluguel.{0,50}/m[eê]s|cau[cç][aã]o|fiador|seguro\s+fian[cç]a|"
    r"titulo de capitaliza[cç][aã]o|credpago|credaluga|contrato de loca[cç][aã]o|"
    r"contrato minimo|comprova[cç][aã]o de renda|renda comprovada|inquilino|"
    r"seguro contra incendio|loca[cç][aã]o residencial|aluguel residencial|fins residenciais",
    re.I,
)
SALE_ONLY_RE = re.compile(r"(?:^|[-_/\s])venda(?:[-_/\s]|$)|\b(?:vendo|vende-se)\b", re.I)


def normalize_periodicity_text(value) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def classify_periodicity(row) -> tuple[str, str]:
    explicit = str(row.get("periodicidade") or "").upper().strip()
    evidence = str(row.get("evidencia_periodicidade") or "").strip()
    if explicit == "DAILY":
        return explicit, evidence or "campo_estruturado"

    source = normalize_periodicity_text(row.get("fonte"))
    url = normalize_periodicity_text(row.get("url"))
    description = normalize_periodicity_text(row.get("descricao"))
    property_type = normalize_periodicity_text(row.get("tipo"))
    text = f"{url} {property_type} {description}"

    match = DAILY_STRONG_RE.search(text)
    if match:
        return "DAILY", f"texto:{match.group(0)}"

    if source == "mgfimoveis":
        if explicit == "MONTHLY" and evidence.startswith("mgf:oferta_estruturada_instock"):
            return "MONTHLY", evidence
        mgf_text = re.sub(
            r"(?:aluguel|condominio)\s+por\s+r?\$?\s*[\d\.,]+\s*/\s*mes\b",
            " ",
            text,
        )
        match = MONTHLY_RE.search(mgf_text)
        if match:
            return "MONTHLY", f"mgf:texto_explicito:{match.group(0)}"
        return "UNKNOWN", "mgf:sem_evidencia_independente_do_rotulo_portal"

    if explicit == "MONTHLY":
        return explicit, evidence or "campo_estruturado"

    match = DAILY_WEAK_RE.search(url)
    if match:
        return "DAILY", f"url:{match.group(0)}"

    match = MONTHLY_RE.search(text)
    if match:
        return "MONTHLY", f"texto:{match.group(0)}"

    match = DAILY_WEAK_RE.search(description)
    if match:
        return "DAILY", f"texto:{match.group(0)}"

    if source == "itamarimoveis":
        return "MONTHLY", "itamar:rent_payment_nao_daily"
    if source == "netimoveis" and "/locacao-" in url:
        return "MONTHLY", "netimoveis:url_locacao"
    return "UNKNOWN", "sem_evidencia_de_periodicidade"


def add_periodicity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    classified = out.apply(classify_periodicity, axis=1, result_type="expand")
    out["periodicidade"] = classified[0]
    out["evidencia_periodicidade"] = classified[1]

    if "id" in out and "fonte" in out:
        syndicated = out["fonte"].astype(str).str.lower().isin(["zapimoveis", "vivareal"])
        syndicated_ids = out.loc[syndicated & out["id"].notna(), "id"].astype(str)
        for listing_id, indexes in syndicated_ids.groupby(syndicated_ids).groups.items():
            periods = set(out.loc[indexes, "periodicidade"])
            if "DAILY" in periods:
                out.loc[indexes, "periodicidade"] = "DAILY"
                out.loc[indexes, "evidencia_periodicidade"] = "id_sindicado:daily"
            elif "MONTHLY" in periods:
                out.loc[indexes, "periodicidade"] = "MONTHLY"
                out.loc[indexes, "evidencia_periodicidade"] = "id_sindicado:monthly"

    fingerprint_columns = ["cidade", "bairro", "rua", "tipo"]
    if all(column in out for column in fingerprint_columns) and "fonte" in out:
        fingerprint = pd.DataFrame(index=out.index)
        for column in fingerprint_columns:
            fingerprint[column] = out[column].map(normalize_periodicity_text)
        for column in ["preco_aluguel", "area_util_m2", "quartos"]:
            fingerprint[column] = pd.to_numeric(out.get(column), errors="coerce").round()
        complete = fingerprint.notna().all(axis=1) & fingerprint["rua"].ne("")
        fingerprint["key"] = fingerprint.apply(lambda row: "|".join(map(str, row)), axis=1)
        for _, indexes in fingerprint.loc[complete].groupby("key").groups.items():
            if out.loc[indexes, "fonte"].astype(str).nunique() < 2:
                continue
            if out.loc[indexes, "periodicidade"].eq("DAILY").any():
                out.loc[indexes, "periodicidade"] = "DAILY"
                out.loc[indexes, "evidencia_periodicidade"] = "fingerprint_multifonte:daily"

    price = pd.to_numeric(out.get("preco_aluguel"), errors="coerce")
    out["preco_aluguel_mensal"] = price.where(out["periodicidade"].eq("MONTHLY"))
    out["motivo_exclusao"] = np.select(
        [out["periodicidade"].eq("DAILY"), out["periodicidade"].eq("UNKNOWN")],
        ["periodicidade_daily", "periodicidade_nao_confirmada"],
        default="",
    )
    return out
