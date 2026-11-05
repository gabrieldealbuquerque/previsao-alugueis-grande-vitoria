import concurrent.futures
import html
import json
import re
from datetime import date

import requests

from scrapers.paths import RAW_DIR


BASE = "https://www.tiosamimoveis.com.br"
SEARCH = BASE + "/finalidade/aluguel-anual/{page}"
OUTPUT = RAW_DIR / "raw_tiosam_listings.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def json_ld(text):
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        text,
        re.I | re.S,
    ):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("@type") == "RealEstateListing":
            return value
    return None


def first_number(pattern, text):
    match = re.search(pattern, text, re.I)
    return float(match.group(1).replace(".", "").replace(",", ".")) if match else None


def parse_detail(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None
    listing = json_ld(response.text)
    if not listing:
        return None
    title = html.unescape(str(listing.get("name") or ""))
    description = html.unescape(str(listing.get("description") or ""))
    combined = f"{title} {description}"
    if re.search(r"loja|comercial|ponto comercial|terreno|lote", combined, re.I):
        return None
    if not re.search(r"apartamento|casa|cobertura|kitnet|studio", combined, re.I):
        return None
    address = listing.get("address") or {}
    offer = listing.get("offers") or {}
    if "AluguelAnual" not in str(offer.get("availability") or ""):
        return None
    street = address.get("streetAddress")
    neighborhood = None
    for name in ["Praia do Morro", "Muquiçaba", "Centro", "Meaípe", "Santa Mônica", "Enseada Azul", "Nova Guarapari", "Praia da Cerca", "Setiba", "Aeroporto"]:
        if re.search(rf"\b{re.escape(name)}\b", combined + " " + str(street), re.I):
            neighborhood = name
            break
    return {
        "id": re.sub(r"\W+", "-", url.rstrip("/").rsplit("/", 1)[-1]),
        "fonte": "tiosamimoveis",
        "data_coleta": date.today().isoformat(),
        "cidade": "Guarapari",
        "bairro": neighborhood,
        "rua": street,
        "tipo": "Apartamento" if re.search(r"apartamento|cobertura|kitnet|studio", combined, re.I) else "Casa",
        "preco_aluguel": float(offer.get("price")),
        "condominio": first_number(r"condom[ií]nio\D{0,20}R\$\s*([\d.,]+)", combined),
        "iptu": first_number(r"IPTU\D{0,20}R\$\s*([\d.,]+)", combined),
        "area_util_m2": first_number(r"(\d+(?:[.,]\d+)?)\s*m[²2]", combined),
        "quartos": first_number(r"(\d+)\s*quartos?", combined),
        "banheiros": first_number(r"(\d+)\s*banheiros?", combined),
        "suites": first_number(r"(\d+)\s*su[ií]tes?", combined),
        "vagas": first_number(r"(\d+)\s*vagas?", combined),
        "descricao": combined,
        "url": url,
        "periodicidade": "MONTHLY",
        "evidencia_periodicidade": "tiosam:categoria_aluguel_anual",
    }


def main():
    links = set()
    for page in range(1, 8):
        suffix = "?view=list" if page == 1 else f"page/{page}/?view=list"
        response = requests.get(SEARCH.format(page=suffix), headers=HEADERS, timeout=30)
        if response.status_code != 200:
            continue
        links.update(
            match.group(1)
            for match in re.finditer(
                r'<a[^>]+href=["\'](https://www\.tiosamimoveis\.com\.br/imovel/[^"\']+)["\']',
                response.text,
                re.I,
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rows = [row for row in executor.map(parse_detail, sorted(links)) if row]
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"links={len(links)} residenciais={len(rows)}")


if __name__ == "__main__":
    main()
