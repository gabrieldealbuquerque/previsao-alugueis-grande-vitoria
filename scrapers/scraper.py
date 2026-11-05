"""
Scraper de anúncios de aluguel residencial (casas e apartamentos) da
Grande Vitória - ES, a partir do portal Chaves na Mão.

Como funciona:
  As páginas de busca do portal são renderizadas com Next.js (App Router) e
  embutem os dados completos de cada anúncio em chunks `self.__next_f.push`
  (formato "React Flight"). Em vez de parsear HTML frágil, este script
  decodifica esses chunks, faz json.loads em cada linha do payload e coleta
  todo objeto que tenha as chaves de um card de anúncio (id, prices, location).

Saída:
  - dados/processados/dataset_alugueis_grande_vitoria.csv
  - dados/brutos/raw_listings.jsonl

Uso:
  .venv/bin/python -m scrapers.scraper             # coleta todas as cidades
  .venv/bin/python -m scrapers.scraper es-vitoria  # coleta só uma cidade
"""

import csv
import json
import random
import re
import sys
import time
from datetime import date

import requests

from scrapers.paths import PROCESSED_DIR, RAW_DIR

BASE = "https://www.chavesnamao.com.br"
# Região Metropolitana da Grande Vitória
CITIES = [
    "es-vitoria",
    "es-vila-velha",
    "es-serra",
    "es-cariacica",
    "es-viana",
    "es-guarapari",
    "es-fundao",
]
# buscas já filtradas por tipo residencial no próprio portal
SEARCH_PATHS = ["apartamentos-para-alugar", "casas-para-alugar"]
# tipos aceitos no dataset final (residenciais)
RESIDENTIAL_TYPES = re.compile(
    r"apartamento|casa|cobertura|kitnet|kitchenette|flat|studio|sobrado", re.I
)
PER_PAGE = 15          # cards por página observados no portal
MAX_PAGES = 200        # trava de segurança por cidade
DELAY_RANGE = (1.0, 2.5)  # segundos entre requisições (educado com o servidor)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "pt-BR,pt;q=0.9",
}

FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\]\)')
FLIGHT_LINE_RE = re.compile(r"^[0-9a-fA-F]+:(.*)$")


def fetch(session: requests.Session, url: str) -> str:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def decode_flight_blob(html: str) -> str:
    """Concatena os chunks `self.__next_f.push([1, "..."])` num payload único."""
    chunks = []
    for m in FLIGHT_RE.finditer(html):
        # o conteúdo é uma string JavaScript escapada; json.loads decodifica
        chunks.append(json.loads('"' + m.group(1) + '"'))
    return "".join(chunks)


REF_RE = re.compile(r"^\$([0-9a-fA-F]+)(?::(.*))?$")


def _follow_path(node, tokens: list[str]):
    """Segue um caminho de referência flight (ex.: 0:props:children:1:...)."""
    for tok in tokens:
        if isinstance(node, list):
            # elemento React serializado: ["$", tipo, key, props]
            if len(node) == 4 and node[0] == "$" and not tok.isdigit():
                node = {"type": node[1], "key": node[2], "props": node[3]}[tok]
            else:
                node = node[int(tok)]
        elif isinstance(node, dict):
            node = node[tok]
        else:
            raise KeyError(tok)
    return node


def resolve_refs(obj, trees: dict, depth: int = 0):
    """Substitui referências '$id:caminho' pelos objetos deduplicados
    originais (o React Flight deduplica subárvores repetidas)."""
    if depth > 12:
        return obj
    if isinstance(obj, str):
        if obj == "$undefined":
            return None
        m = REF_RE.match(obj)
        if m and m.group(1) in trees:
            try:
                target = _follow_path(
                    trees[m.group(1)],
                    m.group(2).split(":") if m.group(2) else [],
                )
                return resolve_refs(target, trees, depth + 1)
            except (KeyError, IndexError, ValueError, TypeError):
                return None
        return obj
    if isinstance(obj, dict):
        return {k: resolve_refs(v, trees, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_refs(v, trees, depth + 1) for v in obj]
    return obj


def extract_listings(html: str) -> dict[int, dict]:
    """Extrai os objetos de anúncio embutidos no payload flight da página."""
    trees: dict[str, object] = {}
    for line in decode_flight_blob(html).split("\n"):
        m = FLIGHT_LINE_RE.match(line)
        if not m:
            continue
        try:
            trees[m.group(0).split(":", 1)[0]] = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue  # linhas de texto/refs do formato flight, sem interesse

    # cada anúncio aparece mais de uma vez na página (renderizações
    # diferentes do card); mesclamos as versões preferindo valores não nulos
    listings: dict[int, dict] = {}

    def merge_into(cur: dict, new: dict):
        for k, v in new.items():
            if v is None:
                continue
            if cur.get(k) is None:
                cur[k] = v
            elif isinstance(cur[k], dict) and isinstance(v, dict):
                for k2, v2 in v.items():
                    if cur[k].get(k2) is None and v2 is not None:
                        cur[k][k2] = v2

    def walk(obj):
        if isinstance(obj, dict):
            if "prices" in obj and "location" in obj and "id" in obj:
                resolved = resolve_refs(obj, trees)
                merge_into(listings.setdefault(resolved["id"], {}), resolved)
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    for tree in trees.values():
        walk(tree)
    return listings


def total_listings(html: str) -> int | None:
    """Lê o total de anúncios do bloco JSON-LD RealEstateListing da página."""
    for m in re.finditer(
        r'<script type="application/ld\+json">(.*?)</script>', html, re.S
    ):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("@type") == "Product":
            offers = d.get("offers", {})
            if offers.get("offerCount"):
                return int(offers["offerCount"])
        if isinstance(d, dict) and d.get("@type") == "RealEstateListing":
            m2 = re.match(r"\s*([\d.]+)", d.get("name", ""))
            if m2:
                return int(m2.group(1).replace(".", ""))
    return None


def parse_brl(text) -> float | None:
    """'R$ 2.350' -> 2350.0"""
    if not text:
        return None
    digits = re.sub(r"[^\d,]", "", str(text)).replace(",", ".")
    try:
        return float(digits)
    except ValueError:
        return None


def to_number(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def tipo_from_url(url: str) -> str | None:
    """Deduz o tipo do imóvel pelo slug da URL do anúncio."""
    m = re.match(r"/imovel/([a-z-]+?)-para-alugar", url or "")
    if not m:
        return None
    return m.group(1).replace("-", " ").capitalize()


def as_dict(value) -> dict:
    """Alguns campos vêm como referência flight (string '$26:...') em vez de
    objeto; nesses casos tratamos como ausente."""
    return value if isinstance(value, dict) else {}


def flatten(raw: dict, city_slug: str) -> dict:
    """Converte o JSON do card num registro tabular para o dataset."""
    loc = as_dict(raw.get("location"))
    geo = as_dict(loc.get("geoposition"))
    prices = as_dict(raw.get("prices"))

    def count(field):
        return as_dict(raw.get(field)).get("count")

    return {
        "id": raw.get("id"),
        "fonte": "chavesnamao",
        "data_coleta": date.today().isoformat(),
        "cidade": as_dict(loc.get("city")).get("name") or city_slug,
        "bairro": as_dict(loc.get("neighborhood")).get("name"),
        "rua": as_dict(loc.get("street")).get("name"),
        "cep": loc.get("zipCode"),
        "latitude": to_number(geo.get("lat")),
        "longitude": to_number(geo.get("lon")),
        "tipo": as_dict(raw.get("realtyType")).get("name") or tipo_from_url(raw.get("url", "")),
        "preco_aluguel": prices.get("rawPrice") or parse_brl(prices.get("main")),
        "condominio": parse_brl(prices.get("condominiumFee")),
        "iptu": parse_brl(prices.get("iptuValue")),
        "area_util_m2": to_number(as_dict(raw.get("area")).get("useful")),
        "area_total_m2": to_number(as_dict(raw.get("area")).get("total")),
        "quartos": count("bedrooms"),
        "banheiros": count("bathrooms"),
        "suites": count("suites"),
        "vagas": count("garages"),
        "pet_friendly": raw.get("petFriendly"),
        "anunciante": as_dict(raw.get("publisher")).get("name"),
        "url": BASE + raw.get("url", ""),
    }


def scrape_search(session: requests.Session, path: str, city_slug: str) -> dict[int, dict]:
    tag = f"{path}/{city_slug}"
    found: dict[int, dict] = {}
    expected = None
    for page in range(1, MAX_PAGES + 1):
        url = f"{BASE}/{path}/{city_slug}/"
        if page > 1:
            url += f"?pg={page}"
        try:
            html = fetch(session, url)
        except requests.RequestException as exc:
            print(f"  [{tag}] pg {page}: erro {exc}; tentando de novo em 10s")
            time.sleep(10)
            try:
                html = fetch(session, url)
            except requests.RequestException as exc2:
                print(f"  [{tag}] pg {page}: falhou de novo ({exc2}); parando busca")
                break

        if expected is None:
            expected = total_listings(html)
            print(f"  [{tag}] total anunciado pelo portal: {expected}")

        page_listings = extract_listings(html)
        new = {k: v for k, v in page_listings.items() if k not in found}
        found.update(new)
        print(f"  [{tag}] pg {page}: {len(page_listings)} cards, {len(new)} novos, acumulado {len(found)}")

        # fim da paginação: página vazia ou só repetições (o portal repete a
        # última página quando pg passa do limite)
        if not page_listings or not new:
            break
        if expected and len(found) >= expected:
            break
        time.sleep(random.uniform(*DELAY_RANGE))
    return found


def main():
    cities = sys.argv[1:] or CITIES
    session = requests.Session()
    all_raw: dict[int, dict] = {}

    for city in cities:
        for path in SEARCH_PATHS:
            print(f"== {path}/{city} ==")
            all_raw.update(scrape_search(session, path, city))
            time.sleep(random.uniform(*DELAY_RANGE))

    rows = []
    with open(RAW_DIR / "raw_listings.jsonl", "w", encoding="utf-8") as f:
        for raw in all_raw.values():
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
            row = flatten(raw, "")
            # apenas residencial de aluguel (a busca já filtra, mas garantimos)
            if raw.get("transaction") == "RENT" and RESIDENTIAL_TYPES.search(row["tipo"] or ""):
                rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(PROCESSED_DIR / "dataset_alugueis_grande_vitoria.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nColetados {len(all_raw)} anúncios ({len(rows)} residenciais de aluguel).")
    print("Arquivos salvos em dados/processados/ e dados/brutos/")


if __name__ == "__main__":
    main()
