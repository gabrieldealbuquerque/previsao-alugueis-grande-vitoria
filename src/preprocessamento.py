import hashlib
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from src.data_quality import add_periodicity_columns


TARGET_COLUMN = "preco_aluguel"
MIN_PREDICTED_RENT = 1.0

RMGV_CITIES = {
    "vitoria": "Vitoria",
    "vila velha": "Vila Velha",
    "serra": "Serra",
    "cariacica": "Cariacica",
    "viana": "Viana",
    "guarapari": "Guarapari",
    "fundao": "Fundao",
}

BAIRROS_LITORANEOS = {
    "Vitoria": {"Bento Ferreira", "Enseada Do Sua", "Ilha Do Boi", "Ilha Do Frade", "Jardim Camburi", "Jardim Da Penha", "Mata Da Praia", "Praia Do Canto", "Santa Helena"},
    "Vila Velha": {"Barra Do Jucu", "Coqueiral De Itaparica", "Interlagos", "Itapua", "Praia Da Costa", "Praia De Itaparica", "Ponta Da Fruta", "Praia Dos Recifes"},
    "Serra": {"Bicanga", "Carapebus", "Jacaraipe", "Manguinhos", "Nova Almeida", "Praia De Capuba"},
    "Guarapari": {"Aldeia Da Praia", "Areia Preta", "Bacutia", "Castanheiras", "Enseada Azul", "Meaipe", "Peracanga", "Praia Do Morro", "Setiba"},
    "Fundao": {"Praia Grande"},
}

FEATURE_COLUMNS = [
    "cidade",
    "bairro_norm",
    "rua_norm",
    "tipo_base",
    "area_m2",
    "area_total_m2",
    "area_construida_m2",
    "area_terreno_m2",
    "quartos",
    "banheiros",
    "suites",
    "vagas",
    "condominio",
    "iptu",
    "tem_condominio",
    "tem_iptu",
    "quartos_missing",
    "banheiros_missing",
    "suites_missing",
    "vagas_missing",
    "area_m2_missing",
    "area_construida_missing",
    "area_terreno_missing",
    "condominio_missing",
    "is_apartamento",
    "mobiliado",
    "armarios",
    "varanda",
    "piscina",
    "churrasqueira",
    "elevador",
    "portaria",
    "vista_mar",
    "praia_proxima",
    "bairro_litoraneo",
    "alto_padrao",
    "condominio_fechado",
    "pet_friendly_flag",
    "ar_condicionado",
    "area_servico",
    "academia",
    "playground",
    "salao_festas",
    "sauna",
    "seguranca_24h",
    "portao_eletronico",
    "closet",
    "quintal",
    "area_gourmet",
    "dependencia_empregada",
    "garagem_coberta",
    "energia_solar",
    "casa_terrea",
    "novo",
    "reformado",
    "amenidades_total",
    "lazer_total",
    "seguranca_total",
    "comodos_total",
    "cidade_bairro",
    "cidade_tipo",
    "bairro_tipo",
    "cidade_bairro_tipo",
    "area_faixa",
    "condominio_faixa",
    "comparavel_bairro",
    "comparavel_condominio",
    "comparavel_rua",
    "quartos_tipo",
    "banheiros_por_quarto",
    "suites_por_quarto",
    "vagas_por_quarto",
    "area_por_quarto",
    "condominio_por_m2",
    "iptu_por_m2",
    "log_area_m2",
    "log_area_construida",
    "log_area_terreno",
    "construida_por_terreno",
    "log_condominio",
    "andar",
    "andar_missing",
    "terreo",
    "dormitorios_por_area",
    "descricao_modelo",
]

MODEL_INPUT_COLUMNS = [
    "id", "cidade", "bairro", "rua", "tipo", "area_m2", "area_total_m2",
    "quartos", "banheiros", "suites", "vagas", "condominio", "iptu", "andar",
    "mobiliado", "armarios", "varanda", "piscina", "churrasqueira", "elevador",
    "portaria", "vista_mar", "praia_proxima", "alto_padrao", "condominio_fechado",
    "pet_friendly", "ar_condicionado", "area_servico", "academia", "playground",
    "salao_festas", "sauna", "seguranca_24h", "portao_eletronico", "closet",
    "quintal", "area_gourmet", "garagem_coberta", "energia_solar", "casa_terrea",
    "novo", "reformado", "area_construida_m2", "area_terreno_m2", "descricao",
]

TEXT_FEATURES = ["descricao_modelo"]

TARGET_BASIC_CATEGORIES = ["cidade", "tipo_base", "cidade_tipo", "quartos_tipo"]
TARGET_HIGH_CARDINALITY_CATEGORIES = [
    "bairro_norm",
    "rua_norm",
    "cidade_bairro",
    "bairro_tipo",
    "cidade_bairro_tipo",
    "area_faixa",
    "condominio_faixa",
    "comparavel_bairro",
    "comparavel_condominio",
    "comparavel_rua",
]
TARGET_CATEGORICAL_FEATURES = TARGET_BASIC_CATEGORIES + TARGET_HIGH_CARDINALITY_CATEGORIES
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in TARGET_CATEGORICAL_FEATURES + TEXT_FEATURES]

ALIASES = {
    "id": ["id", "codigo", "identificador"],
    "cidade": ["cidade", "municipio"],
    "bairro": ["bairro"],
    "rua": ["rua", "logradouro", "endereco"],
    "tipo": ["tipo", "tipo_imovel", "imovel"],
    "preco_aluguel": ["preco_aluguel", "valor_aluguel", "aluguel", "preco", "valor"],
    "condominio": ["condominio", "valor_condominio"],
    "iptu": ["iptu"],
    "area_util_m2": ["area_util_m2", "area_m2", "area", "area_util"],
    "area_total_m2": ["area_total_m2", "area_total"],
    "area_construida_m2": ["area_construida_m2", "area_construida"],
    "area_terreno_m2": ["area_terreno_m2", "area_terreno", "terreno_m2"],
    "quartos": ["quartos", "dormitorios", "dorms"],
    "banheiros": ["banheiros", "banheiro"],
    "suites": ["suites", "suite"],
    "vagas": ["vagas", "garagens", "garagem"],
    "url": ["url", "link"],
    "descricao": ["descricao", "description", "detalhes"],
    "pet_friendly": ["pet_friendly", "aceita_pet"],
    "anunciante": ["anunciante", "vendedor", "corretor", "imobiliaria", "corretora"],
    "fonte": ["fonte", "source", "origem", "portal"],
    "data_coleta": ["data_coleta", "coleta", "data_coleta_dt", "data"],
    "andar_informado": ["andar", "pavimento"],
}

NON_RESIDENTIAL_RE = re.compile(
    r"galpao|deposito comercial|loja|sala[-_ ]?comercial|ponto[-_ ]?comercial|"
    r"imovel[-_ ]?comercial|casa[-_ ]?comercial|uso comercial|finalidade comercial|locacao[-_ ]?comercial|pousada|hotel|hostel|sitio|chacara|"
    r"fazenda|(?:aluguel|venda) de (?:terreno|lote)|terreno (?:vazio|comercial)|lote comercial|predio comercial|galeria comercial|temporada|diaria|feriado|repouso|salao comercial|"
    r"clinica (?:para )?locacao|consultorio medico|sublocacao de consultorio|"
    r"ideal para escritorios|espaco ideal.{0,30}(?:clinica|escritorio)|"
    r"locacao de espaco.{0,30}(?:festas|eventos)|espaco para festas e eventos|locacao para empresa|aluga-se para empresa|"
    r"predio.{0,40}fins comerciais|ideal para fins comerciais",
    re.I,
)
INCLUSIVE_FEES_RE = re.compile(
    r"condominio incluso|condominio inclusos|inclui condominio|"
    r"com (?:iptu e )?condominio inclusos?|pacote de locacao",
    re.I,
)
SHARED_HOME_RE = re.compile(
    r"quarto.{0,40}(?:republica|estudante)|(?:republica|pensionato).{0,40}quarto|"
    r"dividir (?:aluguel|apartamento|casa)|vaga (?:em|para) quarto|"
    r"alugo quarto|aluguel de quarto|quarto mobiliado para locacao|"
    r"suite mobiliada.{0,40}(?:sexo masculino|sexo feminino|homem|mulher)",
    re.I,
)
SHORT_STAY_RE = re.compile(
    r"temporada|temporaria|temporario|diaria|diarias|feriado|reveillon|carnaval|"
    r"semana santa|fim de semana|final de semana|pacote.{0,30}(?:diaria|feriado|carnaval|reveillon)|"
    r"hospedagem|hospedar|airbnb|veraneio|ferias|férias|pernoite|"
    r"ja aluguei|já aluguei|imovel alugado|imóvel alugado|indisponivel|indisponível|"
    r"acomoda(?:m|r)?\s+\d+\s+pessoas?|acomodacao para\s+\d+|"
    r"por[-_ ]?r[-_ ]?\d+(?:[\.,]?\d+)?[-_ ]?(?:a)?dia\b|"
    r"\d+dia\b|/dia\b",
    re.I,
)
SALE_LISTING_RE = re.compile(
    r"\bvenda\b|\bvendo\b|\bvende\b|financi|parcelad|"
    r"\bavista\b|a vista|valor avista|aceito (?:carro|veiculo)|sem entrada",
    re.I,
)

TEXT_FLAG_PATTERNS = {
    "mobiliado": r"mobiliad|semi mobiliad|mobília|mobilia",
    "armarios": r"arm[aá]rio|planejad",
    "varanda": r"varanda|sacada",
    "piscina": r"piscina",
    "churrasqueira": r"churrasqueira|gourmet",
    "elevador": r"elevador",
    "portaria": r"portaria|porteiro|24 horas|24h",
    "vista_mar": r"vista (?:para o )?mar|vista permanente|frente (?:para o )?mar",
    "praia_proxima": r"(?:pr[oó]xim|perto|ao lado|poucos passos).{0,25}praia|praia.{0,25}(?:pr[oó]xim|perto)|\d+\s*(?:m|metros).{0,15}praia|beira[- ]?mar|orla|cal[cç]ad[aã]o",
    "alto_padrao": r"alto padr[aã]o|luxo|luxuos|sofistica",
    "condominio_fechado": r"condom[ií]nio fechado|condominio fechado",
    "ar_condicionado": r"ar[- ]?condicionado|climatizad",
    "area_servico": r"[aá]rea de servi[cç]o|lavanderia",
    "academia": r"academia|fitness",
    "playground": r"playground|parquinho",
    "salao_festas": r"sal[aã]o de festas",
    "sauna": r"sauna",
    "seguranca_24h": r"seguran[cç]a 24|vigil[aâ]ncia 24",
    "portao_eletronico": r"port[aã]o eletr[oô]nico",
    "closet": r"closet",
    "quintal": r"quintal|jardim privativo|area externa",
    "area_gourmet": r"area gourmet|espaco gourmet|varanda gourmet",
    "dependencia_empregada": r"dependencia (?:completa )?de empregad|\bdce\b",
    "garagem_coberta": r"vaga coberta|garagem coberta",
    "energia_solar": r"energia solar|painel solar|fotovoltaic",
    "casa_terrea": r"casa terrea|imovel terreo",
    "novo": r"novo|novissimo|obra nova|obra conclu|pronta|pronto para morar|reformado|reforma",
    "reformado": r"reformad|reformando|re-modelad|acabamento novo",
}

for _flag in TEXT_FLAG_PATTERNS:
    ALIASES[f"{_flag}_informado"] = [_flag]


def normalize_text(value) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode().strip().lower()


def property_cluster_groups(df: pd.DataFrame) -> pd.Series:
    frame = pd.DataFrame(df).reset_index(drop=True).copy()
    size = len(frame)
    parent = list(range(size))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    def values(column, default=""):
        if column not in frame:
            return pd.Series(default, index=frame.index)
        return frame[column]

    city = values("cidade").map(normalize_text)
    neighborhood = values("bairro_norm", "desconhecido").map(normalize_text)
    street = values("rua_norm", "desconhecido").map(normalize_text)
    kind = values("tipo_base").map(normalize_text)
    rooms = pd.to_numeric(values("quartos", np.nan), errors="coerce").round().fillna(-1).astype(int)
    bathrooms = pd.to_numeric(values("banheiros", np.nan), errors="coerce").round().fillna(-1).astype(int)
    parking = pd.to_numeric(values("vagas", np.nan), errors="coerce").round().fillna(-1).astype(int)
    area = pd.to_numeric(values("area_m2", np.nan), errors="coerce")
    source = values("fonte").map(normalize_text)
    listing_id = values("id").fillna("").astype(str).str.strip()

    shared_ids = {}
    for index in range(size):
        if source.iat[index] not in {"zapimoveis", "vivareal"} or not listing_id.iat[index]:
            continue
        key = listing_id.iat[index]
        if key in shared_ids:
            union(index, shared_ids[key])
        else:
            shared_ids[key] = index

    description = values("descricao").map(normalize_text)
    description_key = description.str.replace(r"\s+", " ", regex=True).str.strip()
    exact_descriptions = {}
    for index in range(size):
        current = description_key.iat[index]
        if len(current) < 100 or len(current.split()) < 12:
            continue
        key = (city.iat[index], neighborhood.iat[index], kind.iat[index], current)
        if key in exact_descriptions:
            union(index, exact_descriptions[key])
        else:
            exact_descriptions[key] = index
    building = description.str.extract(
        r"\b(?:edificio|ed\.)\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,3})",
        expand=False,
    ).fillna("")
    building = building.str.replace(
        r"\b(?:com|em|no|na|de|do|da|e|possui|oferece|localizado|localizada)\b.*$", "", regex=True
    ).str.strip()
    building = building.where(
        ~building.str.match(r"^(?:com|conta|possui|oferece|localizado|localizada|proximo|frente|novo)(?:\s|$)"),
        "",
    )

    address_groups = {}
    for index in range(size):
        has_numbered_address = bool(re.search(r"\d", street.iat[index]))
        building_name = building.iat[index]
        if not np.isfinite(area.iat[index]) or (not has_numbered_address and len(building_name) < 4):
            continue
        location = f"address:{street.iat[index]}" if has_numbered_address else f"building:{building_name}"
        key = (city.iat[index], neighborhood.iat[index], location, kind.iat[index], rooms.iat[index])
        address_groups.setdefault(key, []).append(index)

    for indexes in address_groups.values():
        indexes.sort(key=lambda index: area.iat[index])
        for position, left in enumerate(indexes):
            for right in indexes[position + 1:]:
                tolerance = max(2.0, 0.02 * max(area.iat[left], area.iat[right]))
                if area.iat[right] - area.iat[left] > tolerance:
                    break
                bathroom_match = bathrooms.iat[left] >= 0 and bathrooms.iat[left] == bathrooms.iat[right]
                parking_match = parking.iat[left] >= 0 and parking.iat[left] == parking.iat[right]
                if not (bathroom_match or parking_match):
                    continue
                union(left, right)

    # Anúncios republicados por portais diferentes frequentemente mudam apenas
    # pontuação ou uma frase curta. A comparação fica restrita por localização,
    # tipo, quartos e área; o preço nunca participa da identidade.
    approximate_groups = {}
    for index in range(size):
        if len(description_key.iat[index]) < 60 or not np.isfinite(area.iat[index]):
            continue
        key = (city.iat[index], neighborhood.iat[index], kind.iat[index], rooms.iat[index])
        approximate_groups.setdefault(key, []).append(index)
    for indexes in approximate_groups.values():
        indexes.sort(key=lambda index: area.iat[index])
        for position, left in enumerate(indexes):
            left_tokens = set(description_key.iat[left].split())
            for right in indexes[position + 1:]:
                tolerance = max(3.0, 0.05 * max(area.iat[left], area.iat[right]))
                if area.iat[right] - area.iat[left] > tolerance:
                    break
                if source.iat[left] == source.iat[right]:
                    continue
                right_tokens = set(description_key.iat[right].split())
                union_size = len(left_tokens | right_tokens)
                token_similarity = len(left_tokens & right_tokens) / union_size if union_size else 0
                sequence_similarity = SequenceMatcher(
                    None, description_key.iat[left], description_key.iat[right], autojunk=False
                ).ratio()
                if token_similarity >= 0.82 or sequence_similarity >= 0.90:
                    union(left, right)

    components = {}
    for index in range(size):
        components.setdefault(find(index), []).append(index)

    cluster_ids = {}
    for root, indexes in components.items():
        signatures = []
        for index in indexes:
            stable_id = f"{source.iat[index]}:{listing_id.iat[index]}" if listing_id.iat[index] else ""
            semantic = "|".join(
                [
                    city.iat[index], neighborhood.iat[index], street.iat[index], kind.iat[index],
                    str(rooms.iat[index]),
                    f"{area.iat[index]:.1f}" if np.isfinite(area.iat[index]) else "",
                    description_key.iat[index],
                ]
            )
            signatures.append(stable_id or semantic)
        digest = hashlib.sha256(min(signatures).encode("utf-8")).hexdigest()[:20]
        cluster_ids[root] = f"property_{digest}"

    return pd.Series(
        [cluster_ids[find(index)] for index in range(size)],
        index=df.index,
        dtype="object",
    )


def parse_floor(text: str) -> float:
    if not text:
        return np.nan
    normalized = normalize_text(text)
    if re.search(r"\bterre?o\b", normalized):
        return 0.0
    match = re.search(r"(\d{1,2})\s*(?:º|°)?\s*andar", normalized)
    if not match:
        match = re.search(r"andar\s*(\d{1,2})", normalized)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except ValueError:
        return np.nan


def parse_count_from_text(text: str, names: str) -> float:
    normalized = normalize_text(text)
    patterns = [
        rf"(\d{{1,2}})\s*(?:{names})",
        rf"(?:{names})\s*[:\-]?\s*(\d{{1,2}})",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return np.nan


def parse_number(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    text = re.sub(r"[^\d,.\-]", "", text)
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def coerce_number_series(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def _copy_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower_to_original = {c.lower(): c for c in out.columns}
    for canonical, aliases in ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            original = lower_to_original.get(alias.lower())
            if original is not None:
                out[canonical] = out[original]
                break
        if canonical not in out.columns:
            out[canonical] = np.nan
    return out


def _clean_numeric_range(df: pd.DataFrame, column: str, low: float, high: float) -> None:
    df.loc[~df[column].between(low, high), column] = np.nan


def _canonical_city(value) -> str:
    key = normalize_text(value)
    if key in RMGV_CITIES:
        return RMGV_CITIES[key]
    text = "" if pd.isna(value) else str(value).strip()
    return text or "Desconhecida"


def _canonical_type(row) -> str:
    text = normalize_text(f"{row.get('tipo', '')} {row.get('url', '')}")
    if re.search(r"apartamento|cobertura|flat|studio|kitnet|kitchenette", text):
        return "Apartamento"
    if re.search(r"casa|sobrado", text):
        return "Casa"
    return "Desconhecido"


def _property_subtype(row) -> str:
    text = normalize_text(f"{row.get('tipo', '')} {row.get('descricao', '')}")
    if re.search(r"cobertura", text):
        return "Cobertura"
    if re.search(r"kitnet|kitchenette", text):
        return "Kitnet"
    if re.search(r"studio|loft", text):
        return "Studio"
    if re.search(r"flat", text):
        return "Flat"
    if re.search(r"casa (?:em |de )?condominio|condominio de casas", text):
        return "Casa em condominio"
    if re.search(r"sobrado|duplex|triplex", text):
        return "Sobrado"
    return "Apartamento padrao" if _canonical_type(row) == "Apartamento" else "Casa padrao"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _model_area(values: pd.Series) -> np.ndarray:
    """Use neutral area scaling when this optional input is unavailable."""
    area = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.where(np.isfinite(area) & (area > 0), area, 1.0)


def prepare_model_frame(
    raw_df: pd.DataFrame,
    *,
    training: bool,
) -> pd.DataFrame:
    """Normalize, clean and engineer the columns used by the rental model."""
    df = add_periodicity_columns(_copy_aliases(raw_df))

    for column in [
        "preco_aluguel",
        "condominio",
        "iptu",
        "area_util_m2",
        "area_total_m2",
        "area_construida_m2",
        "area_terreno_m2",
        "quartos",
        "banheiros",
        "suites",
        "vagas",
        "andar_informado",
    ]:
        df[column] = coerce_number_series(df[column])

    for column in ["cidade", "bairro", "rua", "tipo", "url", "anunciante", "fonte", "data_coleta"]:
        df[column] = df[column].fillna("").astype(str)
    df["descricao"] = df["descricao"].fillna("").astype(str)
    description_text = df["descricao"].map(normalize_text)
    url_text = (
        df["url"]
        .map(normalize_text)
        .str.replace(r"https?://[^/]+", " ", regex=True)
        .str.replace(r"[-_/]+", " ", regex=True)
    )
    description_text = description_text.where(description_text.str.len() > 5, url_text)
    df["descricao_modelo"] = (
        description_text
        .str.replace(r"r\$?\s*[\d\.,]+", " ", regex=True)
        .str.replace(r"\d+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", "sem descricao")
    )

    df["cidade_norm"] = df["cidade"].map(normalize_text)
    df["cidade"] = df["cidade"].map(_canonical_city)
    df["tipo_base"] = df.apply(_canonical_type, axis=1)
    df["subtipo"] = df.apply(_property_subtype, axis=1)
    df["anunciante"] = df["anunciante"].str.strip().replace({"": "Desconhecido", "nan": "Desconhecido"})
    df["fonte"] = df["fonte"].str.strip().replace({"": "Desconhecido", "nan": "Desconhecido"})
    df["data_coleta_dt"] = pd.to_datetime(df["data_coleta"], errors="coerce")
    data_min = df["data_coleta_dt"].min()
    df["ano_coleta"] = df["data_coleta_dt"].dt.year
    df["mes_coleta"] = df["data_coleta_dt"].dt.month
    df["dia_coleta"] = df["data_coleta_dt"].dt.day
    df["dia_semana_coleta"] = df["data_coleta_dt"].dt.dayofweek
    df["mes_sin"] = np.sin(2.0 * np.pi * df["data_coleta_dt"].dt.month / 12.0)
    df["mes_cos"] = np.cos(2.0 * np.pi * df["data_coleta_dt"].dt.month / 12.0)
    if pd.notna(data_min):
        df["data_ordinal_dias"] = (df["data_coleta_dt"] - data_min).dt.days
    else:
        df["data_ordinal_dias"] = np.nan

    searchable = (
        df["url"] + " " + df["tipo"] + " " + df["bairro"] + " " + df["rua"] + " " + df["descricao"]
    ).map(normalize_text)

    area_text = searchable
    built = area_text.str.extract(
        r"(?:area\s+(?:total\s+)?construida|construcao)\D{0,20}(\d{2,4}(?:[\.,]\d+)?)|(\d{2,4}(?:[\.,]\d+)?)\s*m(?:2)?\s*(?:de\s+)?area\s+construida",
        expand=True,
    ).bfill(axis=1).iloc[:, 0].map(parse_number)
    land = area_text.str.extract(
        r"(?:area\s+(?:total\s+)?do\s+terreno|terreno)\D{0,20}(\d{2,5}(?:[\.,]\d+)?)|(\d{2,5}(?:[\.,]\d+)?)\s*m(?:2)?\s*(?:de\s+)?terreno",
        expand=True,
    ).bfill(axis=1).iloc[:, 0].map(parse_number)
    df["area_construida_m2"] = df["area_construida_m2"].fillna(built)
    df["area_terreno_m2"] = df["area_terreno_m2"].fillna(land)

    count_text = df["descricao"] + " " + df["tipo"] + " " + df["url"]
    count_specs = {
        "quartos": ("quartos?|dormitorios?", 8),
        "banheiros": ("banheiros?", 10),
        "suites": ("suites?", 8),
        "vagas": ("vagas?", 10),
    }
    for column, (names, maximum) in count_specs.items():
        parsed = count_text.map(lambda value, pattern=names: parse_count_from_text(value, pattern))
        invalid = df[column].isna() | ~df[column].between(0, maximum)
        df.loc[invalid & parsed.between(0, maximum), column] = parsed

    if training:
        df = df[df["periodicidade"].eq("MONTHLY")].copy()
        if "anuncio_atual" in df:
            recency = df["anuncio_atual"].map(normalize_text)
            confirmed_current = recency.isin(["true", "1", "sim"])
            mgf = df["fonte"].eq("mgfimoveis")
            stale = recency.isin(["false", "0", "nao"])
            # No MGF, contrato anual prova a periodicidade, mas apenas a página
            # consultada pode provar que o anúncio continua atual.
            df = df[~stale & (~mgf | confirmed_current)].copy()
        else:
            df = df[~df["fonte"].eq("mgfimoveis")].copy()
        df = df[df["cidade_norm"].isin(RMGV_CITIES)].copy()
        searchable = searchable.loc[df.index]
        df = df[df["tipo_base"].isin(["Apartamento", "Casa"])].copy()
        searchable = searchable.loc[df.index]
        non_residential = searchable.str.contains(NON_RESIDENTIAL_RE, na=False)
        short_stay = searchable.str.contains(SHORT_STAY_RE, na=False)
        sale_listing = searchable.str.contains(SALE_LISTING_RE, na=False)
        shared_home = searchable.str.contains(SHARED_HOME_RE, na=False)
        inclusive_fees = searchable.str.contains(INCLUSIVE_FEES_RE, na=False)
        structured_monthly = df["fonte"].eq("itamarimoveis")
        explicit_annual = searchable.str.contains(r"locacao anual|aluguel anual", na=False)
        explicit_rental = searchable.str.contains(r"locacao|aluguel|alugar", na=False)
        non_residential &= ~(structured_monthly & df["tipo_base"].eq("Apartamento"))
        short_stay &= ~(structured_monthly & explicit_annual)
        sale_listing &= ~(structured_monthly & explicit_rental)
        shared_home &= ~structured_monthly
        # O alvo do projeto é aluguel puro. Pacotes que incorporam condomínio
        # ou IPTU não são comparáveis quando a parcela não é informada.
        rejected = non_residential | short_stay | sale_listing | shared_home | inclusive_fees
        syndicated = df["fonte"].isin(["zapimoveis", "vivareal"]) & df["id"].notna()
        rejected_ids = set(df.loc[syndicated & rejected, "id"].astype(str))
        rejected |= syndicated & df["id"].astype(str).isin(rejected_ids)
        df = df[~rejected].copy()
        df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)].copy()

        # Alguns agregadores devolvem anúncios de outra cidade dentro da busca.
        # Ignora o sufixo padronizado do portal e descarta conflitos explícitos
        # no restante da URL, sem consultar o preço ou qualquer dado de teste.
        def city_conflict(row) -> bool:
            url_text = normalize_text(row.get("url", ""))
            current = normalize_text(row.get("cidade", "")).replace(" ", "-")
            url_text = re.sub(rf"-aluguel-es-{re.escape(current)}-\d+/?$", "", url_text)
            current_norm = normalize_text(row.get("cidade", ""))
            return any(
                other != current_norm and re.search(rf"(?:^|[-_/]){re.escape(other.replace(' ', '-'))}(?:[-_/]|$)", url_text)
                for other in RMGV_CITIES
            )

        conflicts = df.apply(city_conflict, axis=1)
        df = df[~conflicts].copy()

    _clean_numeric_range(df, "area_util_m2", 15, 1200)
    _clean_numeric_range(df, "area_total_m2", 15, 2500)
    _clean_numeric_range(df, "area_construida_m2", 15, 2500)
    _clean_numeric_range(df, "area_terreno_m2", 15, 10000)
    _clean_numeric_range(df, "condominio", 0, 5000)
    _clean_numeric_range(df, "iptu", 0, 10000)
    _clean_numeric_range(df, "quartos", 0, 8)
    _clean_numeric_range(df, "banheiros", 0, 10)
    _clean_numeric_range(df, "suites", 0, 8)
    _clean_numeric_range(df, "vagas", 0, 10)
    df.loc[df["condominio"].gt(0) & df["condominio"].le(10), "condominio"] = np.nan
    df.loc[df["iptu"].gt(0) & df["iptu"].le(10), "iptu"] = np.nan

    df["area_m2"] = df["area_util_m2"].fillna(df["area_total_m2"])
    df.loc[df["area_m2"].notna() & ~df["area_m2"].between(15, 1200), "area_m2"] = np.nan
    df.loc[df["suites"].notna() & df["quartos"].notna() & (df["suites"] > df["quartos"]), "suites"] = np.nan
    df.loc[
        df["banheiros"].notna()
        & df["quartos"].notna()
        & (df["banheiros"] > df["quartos"] + 4),
        "banheiros",
    ] = np.nan

    if training:
        df = df[df["area_m2"].notna()].copy()
        df = df[df["data_coleta_dt"].notna()].copy()

    for column in ["bairro", "rua"]:
        df[column] = df[column].fillna("").astype(str).str.strip()
        df.loc[df[column] == "", column] = "Desconhecido"
    df["bairro_norm"] = df["bairro"].map(normalize_text).str.title().replace("", "Desconhecido")
    df["rua_norm"] = df["rua"].map(normalize_text).str.title().replace("", "Desconhecido")
    df["bairro_litoraneo"] = [
        int(bairro in BAIRROS_LITORANEOS.get(cidade, set()))
        for cidade, bairro in zip(df["cidade"], df["bairro_norm"])
    ]

    for column in ["quartos", "banheiros", "suites", "vagas", "area_m2", "condominio"]:
        df[f"{column}_missing"] = df[column].isna().astype(int)
    df["area_construida_missing"] = df["area_construida_m2"].isna().astype(int)
    df["area_terreno_missing"] = df["area_terreno_m2"].isna().astype(int)

    df["tem_condominio"] = df["condominio"].notna().astype(int)
    df["tem_iptu"] = df["iptu"].notna().astype(int)
    df["is_apartamento"] = (df["tipo_base"] == "Apartamento").astype(int)
    for column, pattern in TEXT_FLAG_PATTERNS.items():
        extracted = searchable.loc[df.index].str.contains(pattern, regex=True, na=False).astype(int)
        informed = df[f"{column}_informado"].map(normalize_text)
        informed_true = informed.isin(["1", "true", "sim", "s", "yes"])
        informed_false = informed.isin(["0", "false", "nao", "n", "no"])
        df[column] = extracted
        df.loc[informed_true, column] = 1
        df.loc[informed_false, column] = 0
    df["descricao_len"] = searchable.loc[df.index].str.len().astype(float).fillna(0)
    df["descricao_len"] = np.where(df["descricao_len"] < 1, 1, df["descricao_len"])
    df["descricao_num_palavras"] = searchable.loc[df.index].str.split().str.len().fillna(0).astype(int)
    df["amenidades_total"] = df[list(TEXT_FLAG_PATTERNS)].sum(axis=1)
    df["lazer_total"] = df[["piscina", "churrasqueira", "academia", "playground", "salao_festas", "sauna"]].sum(axis=1)
    df["seguranca_total"] = df[["portaria", "seguranca_24h", "portao_eletronico", "condominio_fechado"]].sum(axis=1)
    df["comodos_total"] = df[["quartos", "banheiros", "suites"]].fillna(0).sum(axis=1)
    df["descricao_num_palavras"] = np.where(df["descricao_num_palavras"] < 1, 1, df["descricao_num_palavras"])
    df["descricao_digits"] = searchable.loc[df.index].str.count(r"\d").astype(int).fillna(0)
    df["descricao_digits_ratio"] = (df["descricao_digits"] / df["descricao_len"]).clip(0, 1)
    df["descricao_tem_numero"] = df["descricao_digits"] > 0
    df["descricao_tem_numero"] = df["descricao_tem_numero"].astype(int)
    original_searchable = (
        df["url"] + " " + df["tipo"] + " " + df["bairro"] + " " + df["rua"] + " " + df["descricao"]
    )
    df["descricao_has_titulo_caps"] = original_searchable.loc[df.index].str.findall(r"[A-ZÀ-Ú]{3,}").str.len().fillna(0).astype(int)
    df["descricao_has_titulo_caps"] = np.where(df["descricao_has_titulo_caps"] > 0, 1, 0)
    df["andar"] = df["andar_informado"].fillna(searchable.loc[df.index].map(parse_floor)).astype(float)
    df["andar_missing"] = df["andar"].isna().astype(int)
    df["terreo"] = (df["andar"] == 0.0).astype(int)
    df["ruas_sem_especifica"] = searchable.loc[df.index].str.contains(
        r"\b(?:rua|rua\.?|avenida|av\.?|travessa|alameda)\b",
        regex=True,
        na=False,
    ).astype(int)
    df["is_venda"] = searchable.loc[df.index].str.contains(
        r"\b(?:venda|vender|vendo)\b",
        regex=True,
        na=False,
    ).astype(int)
    df["dormitorios_por_area"] = _safe_divide(df["quartos"], df["area_m2"])
    pet_values = df["pet_friendly"].astype("object").where(df["pet_friendly"].notna(), "false")
    df["pet_friendly_flag"] = pet_values.astype(str).map(normalize_text)
    df["pet_friendly_flag"] = df["pet_friendly_flag"].isin(["true", "1", "sim", "yes"]).astype(int)

    df["cidade_bairro"] = df["cidade"].astype(str) + "|" + df["bairro_norm"].astype(str)
    df["cidade_tipo"] = df["cidade"].astype(str) + "|" + df["tipo_base"].astype(str)
    df["bairro_tipo"] = df["bairro_norm"].astype(str) + "|" + df["tipo_base"].astype(str)
    df["cidade_bairro_tipo"] = df["cidade_bairro"] + "|" + df["tipo_base"].astype(str)
    df["fonte_cidade"] = df["fonte"].astype(str) + "|" + df["cidade"].astype(str)
    df["fonte_tipo"] = df["fonte"].astype(str) + "|" + df["tipo_base"].astype(str)
    df["fonte_bairro"] = df["fonte"].astype(str) + "|" + df["bairro_norm"].astype(str)
    df["area_faixa"] = (df["area_m2"] / 10).round().fillna(-1).astype(int).astype(str)
    df["condominio_faixa"] = (df["condominio"] / 100).round().fillna(-1).astype(int).astype(str)
    df["comparavel_bairro"] = (
        df["cidade_bairro_tipo"] + "|" + df["quartos"].fillna(-1).round().astype(int).astype(str) + "|" + df["area_faixa"]
    )
    df["comparavel_condominio"] = df["comparavel_bairro"] + "|" + df["condominio_faixa"]
    df["comparavel_rua"] = (
        df["cidade"].astype(str) + "|" + df["bairro_norm"].astype(str) + "|" + df["rua_norm"].astype(str)
        + "|" + df["tipo_base"].astype(str) + "|" + df["quartos"].fillna(-1).round().astype(int).astype(str)
    )
    df["quartos_tipo"] = df["quartos"].fillna(-1).round().astype(int).astype(str) + "|" + df["tipo_base"].astype(str)
    df["banheiros_por_quarto"] = _safe_divide(df["banheiros"], df["quartos"])
    df["suites_por_quarto"] = _safe_divide(df["suites"], df["quartos"])
    df["vagas_por_quarto"] = _safe_divide(df["vagas"], df["quartos"])
    df["area_por_quarto"] = _safe_divide(df["area_m2"], df["quartos"])
    df["condominio_por_m2"] = _safe_divide(df["condominio"], df["area_m2"])
    df["iptu_por_m2"] = _safe_divide(df["iptu"], df["area_m2"])
    df["log_area_m2"] = np.log1p(df["area_m2"])
    df["log_area_construida"] = np.log1p(df["area_construida_m2"])
    df["log_area_terreno"] = np.log1p(df["area_terreno_m2"])
    df["construida_por_terreno"] = _safe_divide(df["area_construida_m2"], df["area_terreno_m2"])
    df["log_condominio"] = np.log1p(df["condominio"].clip(lower=0))

    if training:
        # IDs repetidos dentro da mesma fonte representam o mesmo anúncio. Não
        # use o alvo como chave de deduplicação.
        valid_id = df["id"].notna() & df["id"].astype(str).str.strip().ne("")
        with_id = df[valid_id].drop_duplicates(subset=["fonte", "id"], keep="last")
        without_id = df[~valid_id]
        df = pd.concat([with_id, without_id], axis=0).sort_index().copy()
        syndicated = df["fonte"].isin(["zapimoveis", "vivareal"]) & df["id"].notna()
        syndicated_rows = df[syndicated].copy()
        syndicated_rows["_quality"] = (
            syndicated_rows["descricao"].str.len().fillna(0)
            + syndicated_rows.notna().sum(axis=1)
        )
        syndicated_rows = syndicated_rows.sort_values("_quality").drop_duplicates("id", keep="last")
        df = pd.concat([df[~syndicated], syndicated_rows.drop(columns="_quality")]).sort_index().copy()
    return df


def apply_model_input_contract(prepared_df: pd.DataFrame, original_raw: pd.DataFrame) -> pd.DataFrame:
    metadata = prepared_df.reset_index(drop=True).copy()
    source = original_raw.iloc[metadata["_contract_row_id"].astype(int)].reset_index(drop=True)
    canonical = _copy_aliases(source)
    contract_raw = source.reindex(columns=MODEL_INPUT_COLUMNS).copy()
    contract_raw["area_m2"] = canonical["area_util_m2"]
    contract_raw["andar"] = canonical["andar_informado"]
    contract_raw[TARGET_COLUMN] = metadata[TARGET_COLUMN].to_numpy()
    contract = prepare_model_frame(contract_raw, training=False).reset_index(drop=True)
    metadata.loc[:, FEATURE_COLUMNS] = contract[FEATURE_COLUMNS]
    return metadata


def make_one_hot_encoder(min_frequency: int = 5):
    kwargs = {"handle_unknown": "ignore", "min_frequency": min_frequency}
    try:
        return OneHotEncoder(**kwargs, sparse_output=True)
    except TypeError:
        return OneHotEncoder(**kwargs, sparse=True)


class GroupAwareTargetPreprocessor(BaseEstimator, TransformerMixin):
    """Prepara números, texto e categorias sem vazar o alvo entre imóveis."""

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing

    @staticmethod
    def _groups(frame):
        return property_cluster_groups(frame).reset_index(drop=True)

    @staticmethod
    def _category(series):
        return series.fillna("Desconhecido").astype(str)

    def _fit_maps(self, frame, target):
        self.global_target_ = float(np.mean(target))
        self.target_maps_ = {}
        for column in TARGET_HIGH_CARDINALITY_CATEGORIES:
            values = self._category(frame[column])
            stats = pd.DataFrame({"value": values, "target": target}).groupby("value")["target"].agg(["mean", "count"])
            self.target_maps_[column] = (
                (stats["mean"] * stats["count"] + self.smoothing * self.global_target_)
                / (stats["count"] + self.smoothing)
            )

    def _encode_with_maps(self, frame):
        encoded = np.empty((len(frame), len(TARGET_HIGH_CARDINALITY_CATEGORIES)), dtype=float)
        for position, column in enumerate(TARGET_HIGH_CARDINALITY_CATEGORIES):
            encoded[:, position] = (
                self._category(frame[column])
                .map(self.target_maps_[column])
                .fillna(self.global_target_)
                .to_numpy(float)
            )
        return encoded

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        target = np.asarray(y, dtype=float)
        self.numeric_features_ = NUMERIC_FEATURES
        self.numeric_ = Pipeline(
            [("imputer", SimpleImputer(strategy="median"))]
        ).fit(frame[self.numeric_features_])
        self.basic_ = make_one_hot_encoder(min_frequency=1).fit(
            frame[TARGET_BASIC_CATEGORIES].fillna("Desconhecido").astype(str)
        )
        self.text_ = TfidfVectorizer(
            max_features=1500,
            min_df=3,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        ).fit(frame["descricao_modelo"].fillna("sem descricao").astype(str))
        self._fit_maps(frame, target)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        target = np.asarray(y, dtype=float)
        self.numeric_features_ = NUMERIC_FEATURES
        groups = self._groups(frame)
        folds = min(5, groups.nunique())
        encoded = np.empty((len(frame), len(TARGET_HIGH_CARDINALITY_CATEGORIES)), dtype=float)
        for train_indexes, validation_indexes in GroupKFold(n_splits=folds).split(frame, target, groups):
            train = frame.iloc[train_indexes]
            validation = frame.iloc[validation_indexes]
            global_target = float(np.mean(target[train_indexes]))
            for position, column in enumerate(TARGET_HIGH_CARDINALITY_CATEGORIES):
                values = self._category(train[column])
                stats = pd.DataFrame({"value": values, "target": target[train_indexes]}).groupby("value")["target"].agg(["mean", "count"])
                mapping = (stats["mean"] * stats["count"] + self.smoothing * global_target) / (stats["count"] + self.smoothing)
                encoded[validation_indexes, position] = (
                    self._category(validation[column]).map(mapping).fillna(global_target).to_numpy(float)
                )
        self.fit(frame, target)
        numeric = self.numeric_.transform(frame[self.numeric_features_])
        basic = self.basic_.transform(frame[TARGET_BASIC_CATEGORIES].fillna("Desconhecido").astype(str))
        text = self.text_.transform(frame["descricao_modelo"].fillna("sem descricao").astype(str))
        return sparse.hstack([sparse.csr_matrix(numeric), basic, text, sparse.csr_matrix(encoded)], format="csr")

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        numeric = self.numeric_.transform(frame[self.numeric_features_])
        basic = self.basic_.transform(frame[TARGET_BASIC_CATEGORIES].fillna("Desconhecido").astype(str))
        text = self.text_.transform(frame["descricao_modelo"].fillna("sem descricao").astype(str))
        encoded = self._encode_with_maps(frame)
        return sparse.hstack([sparse.csr_matrix(numeric), basic, text, sparse.csr_matrix(encoded)], format="csr")
