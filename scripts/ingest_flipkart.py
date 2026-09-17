"""
scripts/ingest_flipkart.py
---------------------------
Converts data/flipkart_com-ecommerce_sample.csv into product dicts matching
our schema, then ingests them into Chroma via src.ingestion.ingest_products().

Note on fields: colors, visible_features, style, text_visible_in_image are
left empty for every product here. Those are vision.py-derived fields that
only exist on the QUERY side (when a user searches by uploading a photo) --
the catalog itself has no images run through vision.py, so there's nothing
to put there. product_json_to_text() (used inside ingest_products) already
skips empty fields, so this works correctly without any special-casing.

Run from project root:
    python -m scripts.ingest_flipkart
"""

import ast
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion import ingest_products

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_PATH = "data/flipkart_com-ecommerce_sample.csv"
BACKUP_JSON_PATH = "data/flipkart_converted.json"  # saved for inspection/debugging
BATCH_SIZE = 64           # products embedded per ingest_products() call
BATCH_DELAY_SECONDS = 2   # pause between batches to stay under NVIDIA's rate limit
ROW_LIMIT = 10          # set to e.g. 200 for a quick test run before doing all 20k


# --------------------------------------------------------------------------- #
# Field parsers (raw CSV string -> clean Python values)
# Verified against the real 20,000-row flipkart_com-ecommerce_sample.csv.
# --------------------------------------------------------------------------- #

def parse_category_tree(raw_tree_string: str) -> Dict[str, Any]:
    if not raw_tree_string or not isinstance(raw_tree_string, str):
        return {"category": "", "subcategory": "", "extra_levels": []}
    try:
        tree_list = ast.literal_eval(raw_tree_string)
        breadcrumb = tree_list[0]
    except (ValueError, SyntaxError, IndexError):
        return {"category": "", "subcategory": "", "extra_levels": []}

    levels = [lvl.strip() for lvl in breadcrumb.split(">>")]
    return {
        "category": levels[0] if len(levels) > 0 else "",
        "subcategory": levels[1] if len(levels) > 1 else "",
        "extra_levels": levels[2:] if len(levels) > 2 else [],
    }


def parse_image_list(raw_image_string: str) -> str:
    if not raw_image_string or not isinstance(raw_image_string, str):
        return ""
    try:
        images = ast.literal_eval(raw_image_string)
        return images[0] if images else ""
    except (ValueError, SyntaxError, IndexError):
        return ""


def parse_rating(raw_rating: Any) -> float:
    try:
        return float(raw_rating)
    except (TypeError, ValueError):
        return 0.0


def parse_flipkart_specs(raw_spec_string: str) -> Dict[str, str]:
    """
    Handles a real inconsistency in the dataset: when a product has only
    ONE spec, "product_specification" is a single dict directly instead
    of a list containing one dict. Both shapes are normalized here.
    """
    if not raw_spec_string or not isinstance(raw_spec_string, str):
        return {}
    try:
        python_like = raw_spec_string.replace("=>", ":")
        parsed = ast.literal_eval(python_like)
    except (ValueError, SyntaxError):
        return {}

    spec_data = parsed.get("product_specification", [])
    if isinstance(spec_data, dict):
        spec_data = [spec_data]

    specs = {}
    for item in spec_data:
        if not isinstance(item, dict):
            continue
        key, value = item.get("key"), item.get("value")
        if key and value:
            specs[key.strip().lower()] = str(value).strip()
    return specs


def extract_materials_from_specs(specs: Dict[str, str]) -> List[str]:
    fabric = specs.get("fabric") or specs.get("material")
    if not fabric:
        return []
    return [m.strip().lower() for m in re.split(r"\s+", fabric) if m.strip()]


def safe_str(value: Any, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_and_truncate_description(text: str, max_chars: int = 500) -> str:
    """
    Collapses messy whitespace (tabs/newlines common in this dataset) and
    caps length at a sentence boundary. Longest real description in this
    CSV is ~5300 chars -- well under Nemotron's context limit, but long
    descriptions dilute the embedding signal with marketing boilerplate.
    """
    if not text:
        return text
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.5:
        return truncated[: last_period + 1]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


# --------------------------------------------------------------------------- #
# Row -> schema dict
# --------------------------------------------------------------------------- #

def flipkart_row_to_product(row: pd.Series, generated_id: int) -> Dict[str, Any]:
    category_info = parse_category_tree(row.get("product_category_tree", ""))
    specs = parse_flipkart_specs(row.get("product_specifications", ""))
    materials = extract_materials_from_specs(specs)

    brand = safe_str(row.get("brand"), default="Unknown") or "Unknown"

    price = safe_float(row.get("discounted_price"))
    if price == 0.0:
        price = safe_float(row.get("retail_price"))

    rating = parse_rating(row.get("product_rating"))
    if rating == 0.0:
        rating = parse_rating(row.get("overall_rating"))

    search_keywords = list(dict.fromkeys(
        category_info["extra_levels"]
        + [specs.get("type", "")]
        + [specs.get("ideal for", "")]
        + [specs.get("pattern", "")]
    ))
    search_keywords = [kw for kw in search_keywords if kw]

    return {
        "id": generated_id,
        "product_name": safe_str(row.get("product_name")),
        "brand": brand,
        "category": category_info["category"],
        "subcategory": category_info["subcategory"],
        "model": specs.get("style code", ""),
        "colors": [],                 # vision.py fills this on query side only
        "materials": materials,
        "style": "",                  # vision.py fills this on query side only
        "visible_features": [],       # vision.py fills this on query side only
        "description": clean_and_truncate_description(safe_str(row.get("description"))),
        "text_visible_in_image": [],  # vision.py fills this on query side only
        "search_keywords": search_keywords,
        "condition": "new",
        "price": price,
        "rating": rating,
        "image_url": parse_image_list(row.get("image", "")),
        "product_url": safe_str(row.get("product_url")),
    }


# --------------------------------------------------------------------------- #
# Main pipeline: CSV -> product dicts -> ingest_products() in batches
# --------------------------------------------------------------------------- #

def convert_csv(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    products, skipped = [], 0
    for i, row in df.iterrows():
        try:
            product = flipkart_row_to_product(row, generated_id=i + 1)
            if not product["product_name"]:
                skipped += 1
                continue
            products.append(product)
        except Exception as e:
            logger.warning(f"Skipping row {i} due to parse error: {e}")
            skipped += 1

    logger.info(f"Converted {len(products)} products ({skipped} skipped).")
    return products


def ingest_in_batches(
    products: List[Dict[str, Any]],
    batch_size: int = BATCH_SIZE,
    delay_seconds: float = BATCH_DELAY_SECONDS,
) -> int:
    """
    Calls ingest_products() in chunks rather than all at once. This matters
    because vectorstore.add_texts() sends its whole text list to NVIDIA's
    embedding API in one go -- passing all 20,000 texts at once risks
    hitting request-size and per-minute rate limits. Batching + a short
    delay keeps this well within NVIDIA's ~40 requests/minute free-tier
    limit.
    """
    total = 0
    num_batches = (len(products) + batch_size - 1) // batch_size

    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        batch_num = i // batch_size + 1
        start = time.perf_counter()
        try:
            count = ingest_products(batch)
            total += count
            elapsed = time.perf_counter() - start
            logger.info(
                f"Batch {batch_num}/{num_batches}: ingested {count} products "
                f"in {elapsed:.2f}s (total so far: {total})"
            )
        except Exception as e:
            logger.error(f"Batch {batch_num}/{num_batches} FAILED: {e}")

        if i + batch_size < len(products):
            time.sleep(delay_seconds)

    return total


def main():
    logger.info(f"Reading {CSV_PATH} ...")
    products = convert_csv(CSV_PATH, limit=ROW_LIMIT)

    # Save the converted data too, so you can inspect/debug without re-parsing
    with open(BACKUP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    logger.info(f"Backup written to {BACKUP_JSON_PATH}")

    logger.info(f"Ingesting {len(products)} products in batches of {BATCH_SIZE} ...")
    start = time.perf_counter()
    total = ingest_in_batches(products)
    elapsed = time.perf_counter() - start

    logger.info(f"Done. Ingested {total}/{len(products)} products in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()