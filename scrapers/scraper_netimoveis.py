"""
Scraper da Netimoveis para complementar a base de aluguel residencial na
Grande Vitoria.

Fonte:
  - robots.txt permite rastreamento geral e aponta sitemap publico;
  - o scraper usa apenas URLs de detalhe presentes no sitemap
    `sitemap-imoveis_*.xml` e filtra locacao residencial no Espirito Santo.

Uso:
  python -m scrapers.scraper_netimoveis
  python -m scrapers.scraper_netimoveis --merge-only
"""

import html as htmllib
import json
import random
import re
import argparse
import time
import unicodedata
from datetime import date
from urllib.parse import unquote

import requests

from scrapers.paths import RAW_DIR
from scrapers.scraper import HEADERS, parse_brl
from scrapers.scraper_mgf import write_csv, load_done


BASE = "https://www.netimoveis.com"
SITEMAP_INDEX = f"{BASE}/sitemap.xml"
RAW_FILE = RAW_DIR / "raw_netimoveis_listings.jsonl"
DELAY_RANGE = (0.25, 0.6)

RMGV_SLUGS = {
    "vitoria": "Vitoria",
    "vila-velha": "Vila Velha",
    "serra": "Serra",
    "cariacica": "Cariacica",
    "viana": "Viana",
    "guarapari": "Guarapari",
    "fundao": "Fundao",
}
RESIDENTIAL_TYPES = (
    "apartamento",
    "casa",
    "casa-em-condominio",
    "casa-geminada",
    "cobertura",
    "flat",
    "kitchenette",
    "kitnet",
    "studio",
)

APARTMENT_TYPES = (
    "apartamento",
    "cobertura",
    "flat",
    "kitchenette",
    "kitnet",
    "studio",
)

LOC_RE = re.compile(r"<loc>(.*?)</loc>")
ID_RE = re.compile(r"/(\d+)$")


def norm(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text or "")
        .encode("ascii", "ignore")
        .decode()
        .strip()
        .lower()
    )


def title_from_slug(slug: str | None) -> str | None:
    if not slug:
        return None
    return " ".join(part.capitalize() for part in unquote(slug).split("-") if part)


def fetch(session: requests.Session, url: str) -> str | None:
    for attempt in (1, 2):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == 1:
                print(f"    erro em {url}: {exc}; retry em 6s")
                time.sleep(6)
    return None


def text_content(html: str) -> str:
    text = htmllib.unescape(re.sub(r"<[^>]+>", " ", html))
    return re.sub(r"\s+", " ", text).strip()


def parse_slug(url: str) -> tuple[str | None, str | None, str | None]:
    slug = url.rstrip("/").split("/")[-2]
    if not slug.startswith("locacao-") or "-espirito-santo-" not in slug:
        return None, None, None
    kind, rest = slug[len("locacao-") :].split("-espirito-santo-", 1)
    city_slug = None
    bairro_slug = None
    for candidate in sorted(RMGV_SLUGS, key=len, reverse=True):
        if rest == candidate or rest.startswith(candidate + "-"):
            city_slug = candidate
            bairro_slug = rest[len(candidate) :].lstrip("-") or None
            break
    return kind, city_slug, bairro_slug


def is_residential_detail(url: str) -> bool:
    kind, city_slug, _ = parse_slug(url)
    return bool(
        kind
        and city_slug in RMGV_SLUGS
        and any(kind == t or kind.startswith(t + "-") for t in RESIDENTIAL_TYPES)
        and "casa-comercial" not in kind
    )


def classify_property_type(kind: str | None) -> str | None:
    """Classifica também slugs com complementos, como apartamento-2-quartos."""
    normalized = norm(kind or "")
    if any(normalized == item or normalized.startswith(item + "-") for item in APARTMENT_TYPES):
        return "Apartamento"
    if normalized == "casa" or normalized.startswith("casa-"):
        return "Casa"
    return None


def collect_urls(session: requests.Session) -> list[str]:
    index_html = fetch(session, SITEMAP_INDEX)
    if not index_html:
        return []
    sitemap_urls = [
        u for u in LOC_RE.findall(index_html)
        if "/sitemap-imoveis_" in u
    ]
    urls: dict[str, None] = {}
    for sitemap_url in sitemap_urls:
        html = fetch(session, sitemap_url)
        if not html:
            continue
        before = len(urls)
        for url in LOC_RE.findall(html):
            if is_residential_detail(url):
                urls.setdefault(url)
        print(f"  [{sitemap_url.rsplit('/', 1)[-1]}] {len(urls) - before} novos")
        time.sleep(random.uniform(*DELAY_RANGE))
    return list(urls)


def first_number(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.I)
    value = parse_brl(m.group(1)) if m else None
    return int(round(value)) if value is not None else None


def price_after(label: str, text: str) -> float | None:
    m = re.search(label + r"\s+R\$\s*([\d.,]+)", text, re.I)
    return parse_brl(m.group(1)) if m else None


def description_from_text(text: str) -> str | None:
    m = re.search(r"Mais sobre este imóvel\s+(.*?)\s+(?:ENTRE EM CONTATO|Características)", text, re.I)
    return m.group(1).strip() if m else None


def parse_detail(url: str, html: str) -> dict | None:
    page_text = text_content(html)
    preco = price_after("Valor de locação", page_text)
    if not preco:
        m = re.search(r"por R\$\s*([\d.,]+)", page_text, re.I)
        preco = parse_brl(m.group(1)) if m else None
    if not preco:
        return None

    kind, city_slug, bairro_slug = parse_slug(url)
    if city_slug not in RMGV_SLUGS:
        return None
    m_id = ID_RE.search(url)
    tipo = classify_property_type(kind)
    if tipo is None:
        return None

    area = first_number(r"área aproximada\s+([\d.,]+)\s*m²", page_text)
    quartos = first_number(r"(\d+)\s+quartos?", page_text)
    banheiros = first_number(r"(\d+)\s+banheiros?", page_text)
    suites = first_number(r"(\d+)\s+su[ií]tes?", page_text)
    vagas = first_number(r"(\d+)\s+vagas?", page_text) or first_number(r"(\d+)\s+vaga\b", page_text)

    return {
        "id": int(m_id.group(1)) if m_id else url,
        "fonte": "netimoveis",
        "data_coleta": date.today().isoformat(),
        "cidade": RMGV_SLUGS[city_slug],
        "bairro": title_from_slug(bairro_slug),
        "rua": None,
        "cep": None,
        "latitude": None,
        "longitude": None,
        "tipo": tipo,
        "preco_aluguel": float(preco),
        "condominio": price_after("Condomínio", page_text),
        "iptu": price_after("Iptu", page_text),
        "area_util_m2": float(area) if area else None,
        "area_total_m2": None,
        "quartos": quartos,
        "banheiros": banheiros,
        "suites": suites,
        "vagas": vagas,
        "pet_friendly": True if re.search(r"aceita (pet|animais)|pet friendly", page_text, re.I) else None,
        "anunciante": None,
        "descricao": description_from_text(page_text),
        "url": url,
    }


def load_done_netimoveis() -> dict[int | str, dict]:
    done = {}
    try:
        with open(RAW_FILE, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["id"]] = row
    except FileNotFoundError:
        pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--cities", nargs="+", choices=["Serra", "Vila Velha", "Vitoria", "Cariacica", "Viana", "Guarapari", "Fundao"])
    parser.add_argument("--types", nargs="+", choices=["Casa", "Apartamento"])
    args = parser.parse_args()
    done = load_done_netimoveis()
    if not args.merge_only:
        session = requests.Session()
        urls = collect_urls(session)
        if args.cities or args.types:
            selected_cities = {norm(city).replace(" ", "-") for city in (args.cities or [])}
            selected_types = set(args.types or [])
            filtered = []
            for url in urls:
                kind, city, _ = parse_slug(url)
                property_type = "Apartamento" if kind and kind.startswith(("apartamento", "kitnet", "studio", "flat", "cobertura")) else "Casa"
                if selected_cities and city not in selected_cities:
                    continue
                if selected_types and property_type not in selected_types:
                    continue
                filtered.append(url)
            urls = filtered
        pending = [
            u for u in urls
            if not ((m := ID_RE.search(u)) and int(m.group(1)) in done)
        ]
        print(f"{len(urls)} URLs Netimoveis, {len(pending)} pendentes")
        with open(RAW_FILE, "a", encoding="utf-8") as raw_f:
            for i, url in enumerate(pending, 1):
                html = fetch(session, url)
                if html:
                    row = parse_detail(url, html)
                    if row:
                        done[row["id"]] = row
                        raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        raw_f.flush()
                if i % 25 == 0:
                    print(f"  detalhe {i}/{len(pending)}, total {len(done)}")
                time.sleep(random.uniform(*DELAY_RANGE))
        print(f"Coleta Netimoveis concluida: {len(done)} anuncios")

    write_csv(list(load_done().values()))


if __name__ == "__main__":
    main()
