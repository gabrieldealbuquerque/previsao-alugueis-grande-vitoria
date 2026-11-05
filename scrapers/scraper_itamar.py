import argparse
import json
import time
from datetime import date
from urllib.request import Request, urlopen

from scrapers.paths import RAW_DIR


BASE_URL = "https://www.itamarimoveis.com"
API_URL = BASE_URL + "/api/listings/para-alugar?PAGE={}"
RAW_FILE = RAW_DIR / "raw_itamar_listings.jsonl"
RESIDENTIAL_TYPES = {"APARTMENT", "HOUSE", "PENTHOUSE_APARTMENT", "STUDIO_APARTMENT", "OUTHOUSE"}


def first_number(value):
    if isinstance(value, list):
        value = next((item for item in value if item is not None), None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_page(page):
    request = Request(API_URL.format(page), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def convert(item):
    property_type = item.get("property_type")
    payment = item.get("rent_payment")
    if property_type not in RESIDENTIAL_TYPES or payment == "DAILY":
        return None
    city = str(item.get("city") or "")
    if city.casefold() != "guarapari":
        return None
    bedrooms = first_number(item.get("bedrooms"))
    if property_type == "OUTHOUSE" and not bedrooms:
        return None
    amenities = set(item.get("amenities") or [])
    full_address = str(item.get("full_address") or "")
    street = full_address.split(" - ", 1)[0].strip()
    type_map = {
        "APARTMENT": "Apartamento",
        "HOUSE": "Casa",
        "PENTHOUSE_APARTMENT": "Cobertura",
        "STUDIO_APARTMENT": "Studio",
        "OUTHOUSE": "Kitnet",
    }
    url = str(item.get("url") or "")
    if url.startswith("/"):
        url = BASE_URL + url
    reference = str(item.get("property_full_reference") or url.rsplit("/", 1)[-1])
    return {
        "id": "itamar_" + reference,
        "fonte": "itamarimoveis",
        "data_coleta": date.today().isoformat(),
        "cidade": city,
        "bairro": item.get("neighborhood_display") or item.get("neighborhood"),
        "rua": street,
        "tipo": type_map[property_type],
        "preco_aluguel": first_number(item.get("rent_price")),
        "condominio": first_number(item.get("condo_fees")),
        "iptu": first_number(item.get("property_tax")),
        "area_util_m2": first_number(item.get("area")),
        "area_total_m2": first_number(item.get("area")),
        "quartos": bedrooms,
        "banheiros": first_number(item.get("bathrooms")),
        "suites": first_number(item.get("suites")),
        "vagas": first_number(item.get("garages")),
        "pet_friendly": "PET_FRIENDLY" in amenities,
        "anunciante": "Itamar Imoveis",
        "url": url,
        "descricao": item.get("listing_description") or "",
        "periodicidade": "MONTHLY",
        "evidencia_periodicidade": f"itamar:rent_payment={payment}",
        "preco_aluguel_mensal": first_number(item.get("rent_price")),
        "motivo_exclusao": "",
        "mobiliado": "FURNISHED" in amenities,
        "armarios": any(value.startswith("CABINET_") for value in amenities),
        "varanda": bool({"BALCONY", "VERANDA"} & amenities),
        "piscina": any("SWIMMING_POOL" in value for value in amenities),
        "churrasqueira": "BARBECUE_GRILL" in amenities,
        "elevador": "ELEVATOR" in amenities,
        "portaria": bool({"CONCIERGE_24H", "DOORMAN", "RECEPTION"} & amenities),
        "vista_mar": bool({"SEA_VIEW", "OCEAN_VIEW", "FRONT_SEA"} & amenities),
        "condominio_fechado": "GATED_COMMUNITY" in amenities,
        "ar_condicionado": "AIR_CONDITIONING" in amenities,
        "area_servico": bool({"LAUNDRY_ROOM", "LAUNDRY_ROOM_BR"} & amenities),
        "academia": bool({"ACADEMY", "FITNESS_ROOM", "GYM_ROOM"} & amenities),
        "playground": "PLAYGROUND" in amenities,
        "salao_festas": "PARTY_ROOM" in amenities,
        "sauna": any("SAUNA" in value for value in amenities),
        "portao_eletronico": "ELECTRIC_GATE" in amenities,
        "quintal": bool({"BACKYARD", "GARDEN"} & amenities),
        "area_gourmet": any("GOURMET" in value for value in amenities),
        "garagem_coberta": "COVERED_GARAGE" in amenities,
        "energia_solar": any("SOLAR" in value for value in amenities),
        "novo": "NEW_PROPERTY" in amenities,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    existing = {}
    if RAW_FILE.exists():
        for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                existing[str(row["id"])] = row
            except (json.JSONDecodeError, KeyError):
                pass
    first = fetch_page(1)
    total = int(first.get("count") or len(first.get("data") or []))
    pages = max(1, (total + 11) // 12)
    collected = {}
    for page in range(1, pages + 1):
        payload = first if page == 1 else fetch_page(page)
        for item in payload.get("data") or []:
            row = convert(item)
            if row and row.get("preco_aluguel"):
                collected[row["id"]] = row
        if page < pages:
            time.sleep(args.delay)
    existing.update(collected)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in existing.values())
    RAW_FILE.write_text(content, encoding="utf-8")
    print(f"{len(collected)} locacoes residenciais mensais/anuais coletadas; {len(existing)} salvas")


if __name__ == "__main__":
    main()
