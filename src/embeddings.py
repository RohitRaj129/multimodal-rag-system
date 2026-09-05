"""
embeddings.py
-------------
Converts product data (from vision.py's JSON output, or plain text queries)
into vector embeddings using NVIDIA's Nemotron 3 Embed 1B model, served via
NVIDIA NIM (build.nvidia.com) with an OpenAI-compatible API.
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

try:
    from src.config import EMBEDDING_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL
except ImportError:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

logger = logging.getLogger(__name__)

if not NVIDIA_API_KEY:
    raise EnvironmentError("NVIDIA_API_KEY not found. Add it to .env")

_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

INPUT_TYPE_QUERY = "query"
INPUT_TYPE_PASSAGE = "passage"


def embed_text(text, model=EMBEDDING_MODEL, input_type=INPUT_TYPE_PASSAGE):
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    response = _client.embeddings.create(
        model=model, input=[text],
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return response.data[0].embedding


def embed_batch(texts, model=EMBEDDING_MODEL, input_type=INPUT_TYPE_PASSAGE, batch_size=32):
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        response = _client.embeddings.create(
            model=model, input=chunk,
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        vectors.extend([item.embedding for item in response.data])
    return vectors


def product_json_to_text(product: Dict[str, Any]) -> str:
    def join_list(val):
        return ", ".join(v for v in (val or []) if v)
    parts = [
        product.get("product_name", ""), product.get("brand", ""),
        product.get("category", ""), product.get("subcategory", ""),
        product.get("model", ""), join_list(product.get("colors")),
        join_list(product.get("materials")), product.get("style", ""),
        join_list(product.get("visible_features")), product.get("description", ""),
        join_list(product.get("text_visible_in_image")),
        join_list(product.get("search_keywords")), product.get("condition", ""),
    ]
    return " | ".join(p.strip() for p in parts if p and p.strip())


def embed_product(product, model=EMBEDDING_MODEL):
    text = product_json_to_text(product)
    return embed_text(text, model=model, input_type=INPUT_TYPE_PASSAGE)


def embed_products_batch(products, model=EMBEDDING_MODEL):
    texts = [product_json_to_text(p) for p in products]
    return embed_batch(texts, model=model, input_type=INPUT_TYPE_PASSAGE)


def embed_query(query, model=EMBEDDING_MODEL):
    return embed_text(query, model=model, input_type=INPUT_TYPE_QUERY)