"""
Scraper do portal MGF Imóveis (mgfimoveis.com.br) — segunda fonte do dataset
de aluguéis da Grande Vitória, para ampliar as ~236 linhas do Chaves na Mão.

Como funciona:
  1. Pagina as buscas /aluguel/{tipo}/{cidade}?page=N coletando as URLs das
     páginas de detalhe (es.mgfimoveis.com.br/...-{id}).
  2. Busca cada página de detalhe e lê o JSON-LD `Product`: o preço vem em
     offers.price e os demais atributos (quartos, suítes, banheiros, vagas,
     condomínio, IPTU, área, endereço) vêm na descrição gerada pelo portal
     ("O imóvel ... possui 3 dormitórios, ... está localizado em Rua X,
     Bairro, Cidade.").
  3. Mescla com o dataset do Chaves na Mão (raw_listings.jsonl) e regrava
     dataset_alugueis_grande_vitoria.csv com as duas fontes.

A coleta é retomável: cada anúncio é gravado em raw_mgf_listings.jsonl assim
que processado e execuções seguintes pulam ids já coletados.

Consulte o robots.txt e os termos atuais do portal antes de iniciar uma coleta.

Uso:
  .venv/bin/python -m scrapers.scraper_mgf              # coleta + merge
  .venv/bin/python -m scrapers.scraper_mgf --merge-only # só regenera o CSV final
"""

import csv
import argparse
import html as htmllib
import json
import random
import re
import sys
import time
import unicodedata
from datetime import date

import requests
import pandas as pd

from src.data_quality import add_periodicity_columns

from scrapers.paths import PROCESSED_DIR, RAW_DIR
from scrapers.scraper import CITIES, HEADERS, RESIDENTIAL_TYPES, flatten, parse_brl

BASE = "https://www.mgfimoveis.com.br"
TYPES = ["apartamento", "casa"]
# teto de anúncios por busca (cidade x tipo): as buscas grandes têm 2-3.5 mil
# anúncios; coletar tudo levaria horas. O portal ordena por relevância/data,
# então o teto mantém os anúncios mais recentes.
MAX_DETAILS_PER_SEARCH = 600
CARDS_PER_PAGE = 30
DELAY_RANGE = (0.6, 1.2)

RAW_FILE = RAW_DIR / "raw_mgf_listings.jsonl"
NETIMOVEIS_RAW_FILE = RAW_DIR / "raw_netimoveis_listings.jsonl"
PORTAL_RAW_FILES = [
    RAW_DIR / name
    for name in (
        "raw_olx_listings.jsonl",
        "raw_zap_listings.jsonl",
        "raw_vivareal_listings.jsonl",
        "raw_imovelweb_listings.jsonl",
        "raw_itamar_listings.jsonl",
        "raw_tiosam_listings.jsonl",
    )
]
CSV_FILE = PROCESSED_DIR / "dataset_alugueis_grande_vitoria.csv"

DETAIL_URL_RE = re.compile(r'href="(https://es\.mgfimoveis\.com\.br/[^"#]+)"')
ID_RE = re.compile(r"-(\d+)$")
LDJSON_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)

RMGV = {
    "vitoria", "vila velha", "serra", "cariacica",
    "viana", "guarapari", "fundao",
}


def norm(text: str) -> str:
    """minúsculas sem acento, p/ comparar nomes de cidade/bairro"""
    return (
        unicodedata.normalize("NFKD", text or "")
        .encode("ascii", "ignore")
        .decode()
        .strip()
        .lower()
    )


def fetch(session: requests.Session, url: str) -> str | None:
    for attempt in (1, 2):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == 1:
                print(f"    erro em {url}: {exc}; retry em 8s")
                time.sleep(8)
    return None


def ldjson_products(html: str) -> list[dict]:
    out = []
    for m in LDJSON_RE.finditer(html):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        for item in d if isinstance(d, list) else [d]:
            if isinstance(item, dict) and item.get("@type") == "Product":
                out.append(item)
    return out


def collect_urls(session: requests.Session, tipo: str, city: str, max_details: int) -> list[str]:
    """Pagina a busca e devolve as URLs de detalhe (até o teto)."""
    tag = f"{tipo}/{city}"
    urls: dict[str, None] = {}  # dict preserva ordem e deduplica
    page = 1
    while len(urls) < max_details:
        url = f"{BASE}/aluguel/{tipo}/{city}"
        if page > 1:
            url += f"?page={page}"
        html = fetch(session, url)
        if html is None:
            break
        if page == 1:
            for p in ldjson_products(html):
                print(f"  [{tag}] total anunciado: {p.get('offers', {}).get('offerCount')}")
        before = len(urls)
        for u in DETAIL_URL_RE.findall(html):
            urls.setdefault(u)
        new = len(urls) - before
        print(f"  [{tag}] pg {page}: {new} novos, acumulado {len(urls)}")
        if new == 0:  # fim da paginação (o portal repete a última página)
            break
        page += 1
        time.sleep(random.uniform(*DELAY_RANGE))
    return list(urls)[:max_details]


# padrões da descrição gerada pelo portal
DESC_PATTERNS = {
    "quartos": r"(\d+)\s+dormitóri",
    "suites": r"(\d+)\s+suíte",
    "banheiros": r"(\d+)\s+banheiro",
    "vagas": r"(\d+)\s+vaga",
    "condominio": r"condomínio por R\$\s*([\d.,]+)",
    "iptu": r"IPTU por R\$\s*([\d.,]+)",
    "area_total_m2": r"([\d.,]+)m²\s+de área total",
    "area_util_m2": r"([\d.,]+)m²\s+de área(?!\s+total)",
}
LOCATED_RE = re.compile(r"está localizad[oa] em ([^.]+)")
RENT_RE = re.compile(r"aluguel por R\$[\d.,]+\s*/m[êe]s", re.I)


def parse_detail(url: str, html: str, tipo_busca: str) -> dict | None:
    """Extrai o registro tabular do JSON-LD Product da página de detalhe."""
    products = ldjson_products(html)
    if not products:
        return None
    prod = products[0]
    desc = htmllib.unescape(re.sub(r"<[^>]+>", " ", prod.get("description", "")))

    # só aluguel mensal (as buscas às vezes trazem venda/temporada agregadas)
    if not RENT_RE.search(desc):
        return None
    # aluguel de temporada não é comparável (preço costuma ser diária)
    if re.search(r"temporada|diária|por dia", url + " " + desc, re.I):
        return None

    offers = prod.get("offers", {})
    preco = offers.get("price")
    if not preco:
        m = re.search(r"aluguel por R\$\s*([\d.,]+)", desc)
        preco = parse_brl(m.group(1)) if m else None
    if not preco:
        return None

    fields = {}
    for key, pat in DESC_PATTERNS.items():
        m = re.search(pat, desc)
        fields[key] = m.group(1) if m else None

    rua = bairro = cidade = None
    m = LOCATED_RE.search(desc)
    if m:
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
        if len(parts) >= 2:
            cidade, bairro = parts[-1], parts[-2]
            resto = ", ".join(parts[:-2]) or None
            # só considera rua se parecer logradouro (evita texto livre)
            if resto and re.match(
                r"(?i)(rua|av\.?|avenida|travessa|alameda|rodovia|estrada|praça|beco|largo)\b",
                resto,
            ):
                rua = resto
        elif parts:
            cidade = parts[-1]

    m = ID_RE.search(url)
    return {
        "id": int(m.group(1)) if m else url,
        "fonte": "mgfimoveis",
        "data_coleta": date.today().isoformat(),
        "cidade": cidade,
        "bairro": bairro,
        "rua": rua,
        "cep": None,
        "latitude": None,
        "longitude": None,
        "tipo": tipo_busca.capitalize(),
        "preco_aluguel": float(preco),
        "condominio": parse_brl(fields["condominio"]),
        "iptu": parse_brl(fields["iptu"]),
        "area_util_m2": parse_brl(fields["area_util_m2"]),
        "area_total_m2": parse_brl(fields["area_total_m2"]),
        "quartos": int(fields["quartos"]) if fields["quartos"] else None,
        "banheiros": int(fields["banheiros"]) if fields["banheiros"] else None,
        "suites": int(fields["suites"]) if fields["suites"] else None,
        "vagas": int(fields["vagas"]) if fields["vagas"] else None,
        "pet_friendly": True if re.search(r"aceita (pet|animais)", desc, re.I) else None,
        "anunciante": "Proprietário" if "direto com o proprietário" in desc else None,
        "descricao": desc,
        "url": url,
    }


def load_done() -> dict[int, dict]:
    """Anúncios MGF já coletados em execuções anteriores (retomada)."""
    done: dict[int, dict] = {}
    try:
        with open(RAW_FILE, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                row_id = row["id"]
                previous = done.get(row_id)
                if previous is not None:
                    merged = dict(row)
                    for field, value in previous.items():
                        if merged.get(field) in (None, "") and value not in (None, ""):
                            merged[field] = value
                    if len(str(previous.get("descricao") or "")) > len(str(merged.get("descricao") or "")):
                        merged["descricao"] = previous["descricao"]
                    row = merged
                done[row_id] = row
    except FileNotFoundError:
        pass
    return done


def chavesnamao_rows() -> list[dict]:
    """Reconstrói as linhas do Chaves na Mão a partir do raw (mesma lógica
    do scraper.py original)."""
    rows = []
    try:
        with open(RAW_DIR / "raw_listings.jsonl", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                row = flatten(raw, "")
                if raw.get("transaction") == "RENT" and RESIDENTIAL_TYPES.search(row["tipo"] or ""):
                    period = norm(str(raw.get("locationPeriod") or ""))
                    if "dia" in period:
                        row["periodicidade"] = "DAILY"
                        row["evidencia_periodicidade"] = f"chaves:locationPeriod={raw.get('locationPeriod')}"
                    elif "ano" in period or "mes" in period:
                        row["periodicidade"] = "MONTHLY"
                        row["evidencia_periodicidade"] = f"chaves:locationPeriod={raw.get('locationPeriod')}"
                    row["descricao"] = raw.get("description") or raw.get("descriptionRaw") or raw.get("title")
                    rows.append(row)
    except FileNotFoundError:
        print("aviso: dados/brutos/raw_listings.jsonl não encontrado; CSV só com MGF")
    return rows


def load_extra_rows(path: str) -> list[dict]:
    rows = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[(str(row.get("fonte")), str(row.get("id")))] = row
    except FileNotFoundError:
        pass
    return list(rows.values())


# faixa plausível de aluguel mensal: abaixo disso são placeholders
# ("R$ 1, a combinar"), acima são preços de venda publicados como aluguel
PRECO_MIN, PRECO_MAX = 300, 30_000

NON_RESIDENTIAL_TEXT_RE = re.compile(
    r"galp[aã]o|dep[oó]sito|loja|sala[-_ ]?comercial|ponto[-_ ]?comercial|"
    r"im[oó]vel[-_ ]?comercial|casa[-_ ]?comercial|uso comercial|finalidade comercial|loca[cç][aã]o[-_ ]?comercial|pousada|hotel|hostel|s[ií]tio|ch[aá]cara|"
    r"fazenda|terreno|lote|pr[eé]dio comercial|galeria comercial|temporada|di[aá]ria|feriado|repouso|sal[aã]o comercial",
    re.I,
)
SALE_TEXT_RE = re.compile(r"\bvendo\b|\bvenda\b|financio|parcel[ao]|entrada|aceito", re.I)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_row_values(row: dict) -> dict:
    """Remove valores numericos impossiveis gerados por texto livre do MGF."""
    row = dict(row)
    ranges = {
        "area_util_m2": (15, 1200),
        "area_total_m2": (15, 2500),
        "condominio": (0, 5000),
        "iptu": (0, 10000),
        "quartos": (0, 8),
        "banheiros": (0, 10),
        "suites": (0, 8),
        "vagas": (0, 10),
    }
    for field, (lo, hi) in ranges.items():
        value = _as_float(row.get(field))
        if value is None or not lo <= value <= hi:
            row[field] = None
        elif field in {"quartos", "banheiros", "suites", "vagas"}:
            row[field] = int(value)
        else:
            row[field] = value

    if row.get("suites") is not None and row.get("quartos") is not None and row["suites"] > row["quartos"]:
        row["suites"] = None
    if (
        row.get("banheiros") is not None
        and row.get("quartos") is not None
        and row["banheiros"] > row["quartos"] + 4
    ):
        row["banheiros"] = None
    return row


def likely_residential_rent(row: dict) -> bool:
    text = " ".join(str(row.get(k) or "") for k in ("url", "tipo", "bairro", "rua"))
    if NON_RESIDENTIAL_TEXT_RE.search(text):
        return False
    price = _as_float(row.get("preco_aluguel")) or 0
    return not (price > 10_000 and SALE_TEXT_RE.search(text))


def property_identity_keys(row: dict) -> list[tuple]:
    """Chaves de identidade reproduzíveis que nunca usam o preço-alvo."""
    city = norm(str(row.get("cidade") or ""))
    neighborhood = norm(str(row.get("bairro") or ""))
    kind = norm(str(row.get("tipo") or "").split()[0])
    street = norm(str(row.get("rua") or ""))
    description = re.sub(r"\s+", " ", norm(str(row.get("descricao") or ""))).strip()
    rooms = row.get("quartos")
    area = row.get("area_util_m2") or row.get("area_total_m2")
    source = norm(str(row.get("fonte") or ""))
    listing_id = str(row.get("id") or "").strip()
    keys = []
    if source in {"zapimoveis", "vivareal"} and listing_id:
        keys.append(("shared_id", listing_id))
    if street and re.search(r"\d", street) and city and neighborhood and kind and rooms is not None and area is not None:
        keys.append(("address", city, neighborhood, street, kind, rooms, round(float(area) / 2) * 2))
    if len(description) >= 100 and len(description.split()) >= 12:
        keys.append(("description", city, neighborhood, kind, description))
    return keys


def write_csv(mgf_rows: list[dict]):
    """Junta as duas fontes, tira duplicados óbvios e regrava o CSV final."""
    rows = chavesnamao_rows()
    rows = [
        clean_row_values(r) for r in rows
        if r.get("preco_aluguel")
        and PRECO_MIN <= r["preco_aluguel"] <= PRECO_MAX
        and likely_residential_rent(r)
    ]
    n_cnm = len(rows)
    mgf_rows = [
        clean_row_values(r) for r in mgf_rows
        if PRECO_MIN <= r["preco_aluguel"] <= PRECO_MAX and likely_residential_rent(r)
    ]
    netimoveis_rows = [
        clean_row_values(r) for r in load_extra_rows(NETIMOVEIS_RAW_FILE)
        if r.get("preco_aluguel")
        and PRECO_MIN <= r["preco_aluguel"] <= PRECO_MAX
        and likely_residential_rent(r)
    ]
    portal_rows = []
    for raw_file in PORTAL_RAW_FILES:
        portal_rows.extend(
            clean_row_values(r) for r in load_extra_rows(raw_file)
            if r.get("preco_aluguel")
            and PRECO_MIN <= r["preco_aluguel"] <= PRECO_MAX
            and likely_residential_rent(r)
        )

    rows_by_key = {}
    evidence_flags = {
        "mobiliado", "armarios", "varanda", "piscina", "churrasqueira",
        "elevador", "portaria", "vista_mar", "praia_proxima", "alto_padrao",
        "condominio_fechado", "pet_friendly", "ar_condicionado", "area_servico",
        "academia", "playground", "salao_festas", "sauna", "seguranca_24h",
        "portao_eletronico", "closet", "quintal", "area_gourmet",
        "garagem_coberta", "energia_solar", "casa_terrea", "novo", "reformado",
    }
    for row in rows:
        for identity_key in property_identity_keys(row):
            rows_by_key.setdefault(identity_key, row)
    dupes = 0
    added_by_source = {"mgfimoveis": 0, "netimoveis": 0, "olx": 0, "zapimoveis": 0, "vivareal": 0, "imovelweb": 0, "itamarimoveis": 0, "tiosamimoveis": 0}
    for r in sorted(mgf_rows + netimoveis_rows + portal_rows, key=lambda r: (str(r.get("fonte")), str(r.get("id")))):
        keys = property_identity_keys(r)
        existing = next((rows_by_key[key] for key in keys if key in rows_by_key), None)
        if existing is not None:
            dupes += 1
            fields = [
                "rua", "cep", "area_total_m2", "suites", "pet_friendly",
                "anunciante", "descricao", "url",
            ]
            existing_score = sum(existing.get(field) not in (None, "") for field in fields) + len(str(existing.get("descricao") or "")) / 200
            candidate_score = sum(r.get(field) not in (None, "") for field in fields) + len(str(r.get("descricao") or "")) / 200
            if candidate_score > existing_score:
                merged = dict(r)
                for field, value in existing.items():
                    if merged.get(field) in (None, "") and value not in (None, ""):
                        merged[field] = value
                existing.clear()
                existing.update(merged)
            else:
                for field, value in r.items():
                    if existing.get(field) in (None, "") and value not in (None, ""):
                        existing[field] = value
                if len(str(r.get("descricao") or "")) > len(str(existing.get("descricao") or "")):
                    existing["descricao"] = r["descricao"]
            # Flags positivos são evidência cumulativa entre anúncios
            # sindicados; um parser sem evidência não deve apagar o `True`
            # explícito encontrado por outra fonte.
            for field in evidence_flags:
                values = {norm(str(existing.get(field) or "")), norm(str(r.get(field) or ""))}
                if values & {"1", "true", "sim", "s", "yes"}:
                    existing[field] = True
            for identity_key in property_identity_keys(existing) + keys:
                rows_by_key.setdefault(identity_key, existing)
            continue
        rows.append(r)
        for identity_key in keys:
            rows_by_key.setdefault(identity_key, r)
        added_by_source[r.get("fonte", "mgfimoveis")] = added_by_source.get(r.get("fonte", "mgfimoveis"), 0) + 1

    rows = add_periodicity_columns(pd.DataFrame(rows)).to_dict("records")
    fieldnames = []
    for r in rows:
        for field in r.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"\nCSV final: {len(rows)} linhas ({n_cnm} chavesnamao + "
        f"{added_by_source.get('mgfimoveis', 0)} mgfimoveis + "
        f"{added_by_source.get('netimoveis', 0)} netimoveis, "
        f"{added_by_source.get('olx', 0)} olx + "
        f"{added_by_source.get('zapimoveis', 0)} zapimoveis + "
        f"{added_by_source.get('vivareal', 0)} vivareal + "
        f"{added_by_source.get('imovelweb', 0)} imovelweb, "
        f"{added_by_source.get('itamarimoveis', 0)} itamarimoveis, "
        f"{added_by_source.get('tiosamimoveis', 0)} tiosamimoveis, "
        f"{dupes} duplicados entre fontes removidos)"
    )


def main():
    global DELAY_RANGE
    parser = argparse.ArgumentParser(description="Coleta MGF Imoveis e gera o CSV final.")
    parser.add_argument("cities", nargs="*", help="slugs opcionais, ex.: es-vitoria")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--max-details", type=int, default=MAX_DETAILS_PER_SEARCH)
    parser.add_argument("--refresh-missing", action="store_true")
    parser.add_argument("--delay-min", type=float, default=DELAY_RANGE[0])
    parser.add_argument("--delay-max", type=float, default=DELAY_RANGE[1])
    args = parser.parse_args()
    DELAY_RANGE = (args.delay_min, args.delay_max)

    done = load_done()
    cities = args.cities or CITIES
    if not args.merge_only:
        session = requests.Session()
        print(f"{len(done)} anúncios MGF já coletados (retomada)\n")
        raw_f = open(RAW_FILE, "a", encoding="utf-8")
        for city in cities:
            for tipo in TYPES:
                print(f"== aluguel/{tipo}/{city} ==")
                urls = collect_urls(session, tipo, city, args.max_details)
                pend = []
                for u in urls:
                    m = ID_RE.search(u)
                    if not m:
                        continue
                    row_id = int(m.group(1))
                    if row_id not in done or (
                        args.refresh_missing
                        and not str(done[row_id].get("descricao") or "").strip()
                    ):
                        pend.append(u)
                print(f"  [{tipo}/{city}] {len(urls)} urls, {len(pend)} pendentes")
                for i, u in enumerate(pend, 1):
                    html = fetch(session, u)
                    if html:
                        row = parse_detail(u, html, tipo)
                        if row and norm(row["cidade"] or "") in RMGV:
                            done[row["id"]] = row
                            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            raw_f.flush()
                    if i % 50 == 0:
                        print(f"  [{tipo}/{city}] detalhe {i}/{len(pend)}, total geral {len(done)}")
                    time.sleep(random.uniform(*DELAY_RANGE))
        raw_f.close()
        print(f"\nColeta MGF concluída: {len(done)} anúncios")

    write_csv(list(done.values()))


if __name__ == "__main__":
    main()
