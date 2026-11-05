"""Coleta anúncios residenciais públicos da OLX e do ZAP com Playwright.

Os dois portais entregam os cards no HTML renderizado. A coleta usa apenas as
páginas de busca, aplica atraso entre páginas e é retomável pelos arquivos
``raw_olx_listings.jsonl`` e ``raw_zap_listings.jsonl``.

Uso:
  PLAYWRIGHT_BROWSERS_PATH=.venv/playwright-browsers \
    .venv/bin/python -m scrapers.scraper_portais --max-pages 20
  .venv/bin/python -m scrapers.scraper_portais --merge-only
"""

import argparse
import json
import os
import random
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from scrapers.paths import RAW_DIR, ROOT_DIR
from scrapers.scraper import FLIGHT_LINE_RE, decode_flight_blob
from scrapers.scraper_mgf import load_done, write_csv


OLX_RAW = RAW_DIR / "raw_olx_listings.jsonl"
ZAP_RAW = RAW_DIR / "raw_zap_listings.jsonl"
VIVA_RAW = RAW_DIR / "raw_vivareal_listings.jsonl"
IMOVELWEB_RAW = RAW_DIR / "raw_imovelweb_listings.jsonl"
OLX_URL = (
    "https://www.olx.com.br/imoveis/aluguel/estado-es/"
    "norte-do-espirito-santo/vitoria/grande-vitoria"
)
ZAP_CITIES = {
    "Vitoria": "vitoria",
    "Vila Velha": "vila-velha",
    "Serra": "serra",
    "Cariacica": "cariacica",
    "Viana": "viana",
    "Guarapari": "guarapari",
    "Fundao": "fundao",
}
ZAP_TYPES = {"Casa": "casas", "Apartamento": "apartamentos"}
VIVA_TYPES = {"Casa": "casa_residencial", "Apartamento": "apartamento_residencial"}
ZAP_PRICE_BANDS = [
    (300, 1000),
    (1001, 1500),
    (1501, 2000),
    (2001, 2500),
    (2501, 3000),
    (3001, 4000),
    (4001, 6000),
    (6001, 10_000),
    (10_001, 30_000),
]
OLX_PRICE_BANDS = ZAP_PRICE_BANDS
CITY_NAMES = {
    "vitoria": "Vitoria",
    "vila velha": "Vila Velha",
    "serra": "Serra",
    "cariacica": "Cariacica",
    "viana": "Viana",
    "guarapari": "Guarapari",
    "fundao": "Fundao",
}
RESIDENTIAL_RE = re.compile(
    r"apartamento|\bapto\b|\bapt\b|casa|sobrado|cobertura|kitnet|studio|flat|loft",
    re.I,
)
NON_RESIDENTIAL_RE = re.compile(
    r"sala comercial|ponto comercial|im[oó]vel comercial|uso comercial|loja|sal[aã]o comercial|galp[aã]o|terreno|lote|pr[eé]dio comercial|temporada|di[aá]ria",
    re.I,
)
SHARED_HOME_RE = re.compile(
    r"procuro|preciso|quarto para (?:alugar|estudante)|dividir (?:aluguel|apartamento)|pens[aã]o|vaga em quarto",
    re.I,
)


def norm(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value or "")
        .encode("ascii", "ignore")
        .decode()
        .strip()
        .lower()
    )


def number(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"\d[\d.]*", value)
    return float(match.group().replace(".", "")) if match else None


def first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.I)
    return int(match.group(1)) if match else None


def clean_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def listing_id(url: str) -> str:
    match = re.search(r"(?:id-|-)(\d{7,})(?:/|$)", clean_url(url))
    return match.group(1) if match else clean_url(url)


def location(text: str, fallback_city: str | None = None) -> tuple[str | None, str | None]:
    for raw_city, canonical in CITY_NAMES.items():
        match = re.search(
            rf"(?:^|\n)\s*{re.escape(raw_city)}\s*,\s*([^\n]+)",
            norm(text),
            re.I,
        )
        if match:
            bairro_norm = match.group(1).strip(" ,-.")
            original = next(
                (line for line in text.splitlines() if norm(line).startswith(raw_city + ",")),
                "",
            )
            bairro = original.split(",", 1)[1].strip() if "," in original else bairro_norm.title()
            return canonical, bairro

        match = re.search(
            rf"(?:^|\n)\s*([^\n,]+),\s*{re.escape(raw_city)}(?:\s|$)",
            norm(text),
            re.I,
        )
        if match:
            return canonical, match.group(1).strip().title()
    return fallback_city, None


def base_row(source: str, url: str, title: str, text: str, fallback_city=None, fallback_type=None) -> dict | None:
    combined = f"{title}\n{text}"
    if (
        (source != "olx" and not fallback_type and not RESIDENTIAL_RE.search(combined))
        or (not fallback_type and NON_RESIDENTIAL_RE.search(combined))
        or SHARED_HOME_RE.search(combined)
    ):
        return None
    price_match = re.search(r"R\$\s*[\d.]+\s*(?:/\s*m[eê]s)?", text, re.I)
    price = number(price_match.group()) if price_match else None
    if price is None or not 300 <= price <= 30_000:
        return None
    city, bairro = location(text, fallback_city)
    if not city:
        return None
    lower = norm(combined)
    tipo = fallback_type or ("Apartamento" if re.search(r"apartamento|\bapto\b|\bapt\b|cobertura|kitnet|studio|flat|loft", lower) else "Casa")
    area = first_int(r"(\d{1,4})\s*m[²2]", text)
    return {
        "id": listing_id(url),
        "fonte": source,
        "data_coleta": date.today().isoformat(),
        "cidade": city,
        "bairro": bairro,
        "rua": None,
        "cep": None,
        "latitude": None,
        "longitude": None,
        "tipo": tipo,
        "preco_aluguel": price,
        "condominio": number((re.search(r"Cond(?:om[ií]nio)?\.?\s*R\$\s*[\d.]+", text, re.I) or [None])[0]),
        "iptu": number((re.search(r"IPTU\s*R\$\s*[\d.]+", text, re.I) or [None])[0]),
        "area_util_m2": area,
        "area_total_m2": None,
        "quartos": first_int(r"(?:Quantidade de quartos|quartos?)\s*\n?\s*(\d+)", text),
        "banheiros": first_int(r"(?:Quantidade de banheiros|banheiros?)\s*\n?\s*(\d+)", text),
        "suites": first_int(r"(\d+)\s*su[ií]tes?", combined),
        "vagas": first_int(r"(?:Quantidade de vagas de garagem|vagas?)\s*\n?\s*(\d+)", text),
        "pet_friendly": True if re.search(r"aceita (?:pet|animais)|permitido animais", lower) else None,
        "anunciante": None,
        "descricao": title,
        "url": clean_url(url),
    }


def parse_olx(page) -> list[dict]:
    cards = []

    def walk(value):
        if isinstance(value, dict):
            if "listId" in value and "priceValue" in value:
                cards.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for line in decode_flight_blob(page.content()).splitlines():
        match = FLIGHT_LINE_RE.match(line)
        if not match:
            continue
        try:
            walk(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue

    rows = []
    for card in cards:
        properties = {item.get("name"): item.get("value") for item in card.get("properties", [])}
        details = " ".join(
            str(properties.get(name) or "")
            for name in ("real_estate_type", "re_features", "re_complex_features")
        )
        text = "\n".join(
            filter(None, [card.get("priceValue"), properties.get("condominio"), properties.get("iptu"), card.get("location")])
        )
        row = base_row("olx", card["url"], f"{card.get('subject', '')} {details}", text)
        if not row:
            continue
        location_details = card.get("locationDetails") or {}
        row.update(
            {
                "id": str(card["listId"]),
                "cidade": location_details.get("municipality") or row["cidade"],
                "bairro": location_details.get("neighbourhood") or row["bairro"],
                "condominio": number(properties.get("condominio")),
                "iptu": number(properties.get("iptu")),
                "area_util_m2": number(properties.get("size")),
                "quartos": first_int(r"(\d+)", str(properties.get("rooms") or "")),
                "banheiros": first_int(r"(\d+)", str(properties.get("bathrooms") or "")),
                "vagas": first_int(r"(\d+)", str(properties.get("garage_spaces") or "")),
                "descricao": f"{card.get('subject', '')}. {details}".strip(),
            }
        )
        rows.append(row)
    return rows


def parse_zap(page, city: str) -> list[dict]:
    cards = page.locator("a[href*='/imovel/']").evaluate_all(
        "els => els.map(a => ({url: a.href, title: a.querySelector('h2')?.innerText || '', text: a.innerText || ''}))"
    )
    rows = []
    for card in cards:
        if "/aluguel-" not in card["url"]:
            continue
        row = base_row("zapimoveis", card["url"], card["title"], card["text"], city)
        if row:
            street = re.search(
                r"(?:^|\n)\s*((?:Rua|Avenida|Av\.?|Alameda|Travessa|Rodovia|Estrada|Pra[cç]a)\s+[^\n]+)",
                card["text"],
                re.I,
            )
            if street:
                row["rua"] = street.group(1).strip()
            rows.append(row)
    return rows


def parse_viva(page, city: str) -> list[dict]:
    cards = page.locator("a[href*='/imovel/']").evaluate_all(
        "els => els.map(a => ({url: a.href, title: a.querySelector('h2')?.innerText || '', text: a.innerText || ''}))"
    )
    rows = []
    seen = set()
    for card in cards:
        url = clean_url(card["url"])
        if url in seen:
            continue
        seen.add(url)
        row = base_row("vivareal", url, card["title"], card["text"], city)
        if row:
            row["descricao"] = f"{card['title']}. {card['text']}".strip()[:3000]
            street = re.search(
                r"(?:^|\n)\s*((?:Rua|Avenida|Av\.?|Alameda|Travessa|Rodovia|Estrada|Pra[cç]a)\s+[^\n]+)",
                card["text"],
                re.I,
            )
            if street:
                row["rua"] = street.group(1).strip()
            rows.append(row)
    return rows


def parse_imovelweb(page, city: str) -> list[dict]:
    cards = page.locator("div[data-qa='posting PROPERTY']").evaluate_all(
        "els => els.map(e => ({url: e.querySelector(\"a[href*='/propriedades/']\")?.href || '', text: e.innerText || ''}))"
    )
    rows = []
    fallback_type = "Apartamento" if "apartamentos-" in page.url else "Casa"
    for card in cards:
        if not card["url"]:
            continue
        lines = [line.strip() for line in card["text"].splitlines() if line.strip()]
        title = max(lines, key=len, default="")
        row = base_row("imovelweb", card["url"], title, card["text"], city, fallback_type)
        if row:
            row["descricao"] = card["text"][:3000]
            row["quartos"] = first_int(r"(\d+)\s*quartos?", card["text"]) or row.get("quartos")
            row["banheiros"] = first_int(r"(\d+)\s*ban(?:heiros?|\.)", card["text"]) or row.get("banheiros")
            row["vagas"] = first_int(r"(\d+)\s*vagas?", card["text"]) or row.get("vagas")
            street = re.search(
                r"(?:^|\n)\s*((?:Rua|Avenida|Av\.?|Alameda|Travessa|Rodovia|Estrada|Pra[cç]a)\s+[^\n]+)",
                card["text"],
                re.I,
            )
            if street:
                row["rua"] = street.group(1).strip()
            rows.append(row)
    return rows


def load_raw(path: Path) -> dict[str, dict]:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def append_new(path: Path, rows: list[dict], done: dict[str, dict], refresh: bool = False) -> int:
    new = [row for row in rows if refresh or str(row["id"]) not in done]
    if new:
        with path.open("a", encoding="utf-8") as stream:
            for row in new:
                done[str(row["id"])] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(new)


def collect(args) -> None:
    sources = set(args.source)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headful,
            args=["--disable-blink-features=AutomationControlled"],
        )
        jobs = []
        if "olx" in sources:
            bands = OLX_PRICE_BANDS if args.olx_price_bands else [(None, None)]
            selected_types = args.types or [None]
            selected_rooms = args.rooms or [None]
            for property_type in selected_types:
                for rooms in selected_rooms:
                    for minimum, maximum in bands:
                        parameters = []
                        if minimum is not None:
                            parameters.extend([f"ps={minimum}", f"pe={maximum}"])
                        if property_type is not None:
                            segment = "casas" if property_type == "Casa" else "apartamentos"
                            room_segment = f"{rooms}-quartos/" if rooms is not None else ""
                            base_url = OLX_URL.replace("/aluguel/", f"/aluguel/{segment}/{room_segment}")
                        else:
                            base_url = OLX_URL
                        url = base_url + ("?" + "&".join(parameters) if parameters else "")
                        jobs.append(("olx", None, url, OLX_RAW, parse_olx))
        if "zap" in sources:
            selected_cities = set(args.cities or ZAP_CITIES)
            selected_types = args.types or [None]
            for city, slug in ZAP_CITIES.items():
                if city not in selected_cities:
                    continue
                for property_type in selected_types:
                    segment = ZAP_TYPES.get(property_type, "imoveis")
                    base = f"https://www.zapimoveis.com.br/aluguel/{segment}/es%2B{slug}/"
                    bands = ZAP_PRICE_BANDS if args.zap_price_bands else [(None, None)]
                    for minimum, maximum in bands:
                        url = base
                        if minimum is not None:
                            url += f"?precoMinimo={minimum}&precoMaximo={maximum}"
                        jobs.append(("zap", city, url, ZAP_RAW, parse_zap))
        if "viva" in sources:
            selected_cities = set(args.cities or ZAP_CITIES)
            selected_types = args.types or list(VIVA_TYPES)
            for city, slug in ZAP_CITIES.items():
                if city not in selected_cities:
                    continue
                for property_type in selected_types:
                    segment = VIVA_TYPES[property_type]
                    base = f"https://www.vivareal.com.br/aluguel/espirito-santo/{slug}/{segment}/"
                    bands = ZAP_PRICE_BANDS if args.viva_price_bands else [(None, None)]
                    for minimum, maximum in bands:
                        url = base
                        if minimum is not None:
                            url += f"?precoMinimo={minimum}&precoMaximo={maximum}"
                        jobs.append(("viva", city, url, VIVA_RAW, parse_viva))
        if "imovelweb" in sources:
            selected_cities = set(args.cities or ZAP_CITIES)
            selected_types = args.types or list(ZAP_TYPES)
            selected_rooms = args.rooms or [None]
            selected_orders = [None, "menor", "maior"] if args.imovelweb_price_orders else [None]
            selected_features = [None, "areap-mobiliado", "areac-piscina", "areac-churrasqueira", "areac-elevador", "areap-varanda"] if args.imovelweb_features else [None]
            for city, slug in ZAP_CITIES.items():
                if city not in selected_cities:
                    continue
                for property_type in selected_types:
                    for rooms in selected_rooms:
                        for feature in selected_features:
                            for order in selected_orders:
                                segment = "casas" if property_type == "Casa" else "apartamentos"
                                room_segment = f"-{rooms}-quartos" if rooms is not None else ""
                                feature_segment = f"-{feature}" if feature is not None else ""
                                order_segment = f"-ordem-precio-{order}" if order is not None else ""
                                url = f"https://www.imovelweb.com.br/{segment}-aluguel-{slug}-es{room_segment}{feature_segment}{order_segment}.html"
                                jobs.append(("imovelweb", city, url, IMOVELWEB_RAW, parse_imovelweb))

        for source, city, base_url, path, parser in jobs:
            # Cookies de uma busca anterior acionam bloqueio prematuro no ZAP;
            # cada cidade usa um contexto limpo, sem contornar desafios HTTP.
            context = browser.new_context(
                locale="pt-BR",
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.set_default_timeout(20_000)
            done = load_raw(path)
            empty_pages = 0
            for page_number in range(args.start_page, args.start_page + args.max_pages):
                separator = "&" if "?" in base_url else "?"
                parameter = "o" if source == "olx" else "pagina"
                if source == "imovelweb" and page_number > 1:
                    url = base_url.replace(".html", f"-pagina-{page_number}.html")
                else:
                    url = base_url if page_number == 1 else f"{base_url}{separator}{parameter}={page_number}"
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                    page.wait_for_timeout(args.wait_ms)
                    rows = parser(page, city) if source in {"zap", "viva", "imovelweb"} else parser(page)
                except PlaywrightTimeoutError:
                    print(f"[{source}/{city or 'RMGV'}] página {page_number}: timeout")
                    continue
                new_count = append_new(path, rows, done, refresh=args.refresh)
                status = response.status if response else "?"
                print(f"[{source}/{city or 'RMGV'}] página {page_number}: HTTP {status}, {len(rows)} válidos, {new_count} novos")
                empty_pages = empty_pages + 1 if not rows or not new_count else 0
                if empty_pages >= 2:
                    break
                time.sleep(random.uniform(args.delay_min, args.delay_max))
            context.close()
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta OLX e ZAP Imóveis com Playwright.")
    parser.add_argument("--source", nargs="+", choices=["olx", "zap", "viva", "imovelweb"], default=["olx", "zap"])
    parser.add_argument("--cities", nargs="+", choices=list(ZAP_CITIES), help="cidades do ZAP")
    parser.add_argument("--types", nargs="+", choices=list(ZAP_TYPES))
    parser.add_argument("--rooms", nargs="+", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--zap-price-bands", action="store_true", help="particiona buscas do ZAP por preço")
    parser.add_argument("--olx-price-bands", action="store_true", help="particiona buscas da OLX por preço")
    parser.add_argument("--viva-price-bands", action="store_true", help="particiona buscas da Viva Real por preço")
    parser.add_argument("--imovelweb-price-orders", action="store_true")
    parser.add_argument("--imovelweb-features", action="store_true")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--wait-ms", type=int, default=2500)
    parser.add_argument("--delay-min", type=float, default=0.8)
    parser.add_argument("--delay-max", type=float, default=1.8)
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="regrava anúncios existentes com o parser atual")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT_DIR / ".venv/playwright-browsers"))
    if not args.merge_only:
        collect(args)
    write_csv(list(load_done().values()))


if __name__ == "__main__":
    main()
