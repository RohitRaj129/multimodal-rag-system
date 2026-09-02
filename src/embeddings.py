# this file is for handling embeddings

"""
embeddings.py
-------------
Converts product data (from vision.py's JSON output, or plain text queries)
into vector embeddings using NVIDIA's Nemotron 3 Embed 1B model, served via
NVIDIA NIM (build.nvidia.com) with an OpenAI-compatible API.

Requires:
    pip install openai python-dotenv

.env must contain:
    NVIDIA_API_KEY=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

Usage:
    from src.embeddings import embed_text, embed_product, embed_batch, embed_query

    vec = embed_query("red leather handbag with gold buckle")
    vec = embed_product(product_json_dict)
    vecs = embed_batch([text1, text2, text3])
"""

import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

try:
    from src.config import EMBEDDING_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL
except ImportError:
    # Fallback defaults if config.py doesn't have these yet
    import os
    from dotenv import load_dotenv

    load_dotenv()
    EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

logger = logging.getLogger(__name__)

if not NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY not found. Add it to your .env file: "
        "NVIDIA_API_KEY=nvapi-xxxxxxxx"
    )

_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# NIM embedding calls expect an "input_type" — "query" for search queries,
# "passage" for documents/catalog items being indexed. Using the right one
# on each side measurably improves retrieval quality.
INPUT_TYPE_QUERY = "query"
INPUT_TYPE_PASSAGE = "passage"


# --------------------------------------------------------------------------- #
# Core embedding calls
# --------------------------------------------------------------------------- #

def embed_text(
    text: str,
    model: str = EMBEDDING_MODEL,
    input_type: str = INPUT_TYPE_PASSAGE,
) -> List[float]:
    """
    Embed a single string of text into a vector using NVIDIA Nemotron 3 Embed.

    Args:
        text: The text to embed.
        model: NIM embedding model name (default from config).
        input_type: "passage" when embedding catalog/document text,
                    "query" when embedding a user's search query.

    Returns:
        List[float]: the embedding vector (2048-dim).
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    try:
        response = _client.embeddings.create(
            model=model,
            input=[text],
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed for text='{text[:60]}...': {e}")
        raise


def embed_batch(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    input_type: str = INPUT_TYPE_PASSAGE,
    batch_size: int = 32,
) -> List[Optional[List[float]]]:
    """
    Embed multiple texts. NIM supports true batching in a single request,
    so this chunks the input list and sends batch_size texts per call.

    Args:
        texts: List of strings to embed.
        model: NIM embedding model name.
        input_type: "passage" or "query".
        batch_size: how many texts to send per API call.

    Returns:
        List of embedding vectors, same order as input. Failed items are None.
    """
    vectors: List[Optional[List[float]]] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        try:
            response = _client.embeddings.create(
                model=model,
                input=chunk,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend([item.embedding for item in response.data])
        except Exception as e:
            logger.error(f"Batch embedding failed for chunk starting at {start}: {e}")
            vectors.extend([None] * len(chunk))

    return vectors


# --------------------------------------------------------------------------- #
# Product-specific helpers
# --------------------------------------------------------------------------- #

def product_json_to_text(product: Dict[str, Any]) -> str:
    """
    Flatten a product JSON (from vision.py's output schema) into a single
    text blob optimized for embedding + retrieval quality.
    """

    def join_list(val: Optional[List[str]]) -> str:
        return ", ".join(v for v in (val or []) if v)

    parts = [
        product.get("product_name", ""),
        product.get("brand", ""),
        product.get("category", ""),
        product.get("subcategory", ""),
        product.get("model", ""),
        join_list(product.get("colors")),
        join_list(product.get("materials")),
        product.get("style", ""),
        join_list(product.get("visible_features")),
        product.get("description", ""),
        join_list(product.get("text_visible_in_image")),
        join_list(product.get("search_keywords")),
        product.get("condition", ""),
    ]

    text = " | ".join(p.strip() for p in parts if p and p.strip())
    return text


def embed_product(product: Dict[str, Any], model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Convert a product JSON dict directly into an embedding vector.
    Uses input_type="passage" since products are the documents being indexed.
    """
    text = product_json_to_text(product)
    return embed_text(text, model=model, input_type=INPUT_TYPE_PASSAGE)


def embed_products_batch(
    products: List[Dict[str, Any]], model: str = EMBEDDING_MODEL
) -> List[Optional[List[float]]]:
    """
    Embed a list of product JSON dicts in one pass.
    Useful for scripts/ingest_catalog.py.
    """
    texts = [product_json_to_text(p) for p in products]
    return embed_batch(texts, model=model, input_type=INPUT_TYPE_PASSAGE)


def embed_query(query: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Embed a user's search query. Uses input_type="query", which is tuned
    differently from "passage" for better query-to-document retrieval.
    Use this in retrieval.py instead of embed_text() directly.
    """
    return embed_text(query, model=model, input_type=INPUT_TYPE_QUERY)


# --------------------------------------------------------------------------- #
# Quick manual test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_product = {
        "product_name": "Leather Tote Bag",
        "category": "Bags",
        "subcategory": "Tote",
        "brand": "Fossil",
        "model": "",
        "colors": ["brown", "tan"],
        "materials": ["leather"],
        "visible_features": ["gold hardware", "double handles", "zip closure"],
        "style": "casual",
        "condition": "new",
        "text_visible_in_image": ["FOSSIL"],
        "description": "A spacious brown leather tote with gold-tone hardware.",
        "search_keywords": ["tote", "leather bag", "brown handbag"],
    }

    text_repr = product_json_to_text(sample_product)
    print("Text used for embedding:\n", text_repr, "\n")

    vector = embed_product(sample_product)
    print(f"Embedding length: {len(vector)}")  # should be 2048
    print(f"First 5 values: {vector[:5]}")
    print(f"Time taken: {elapsed:.3f} seconds")

    query_vector = embed_query("brown leather handbag")
    print(f"\nQuery embedding length: {len(query_vector)}")
    print(f"Time taken: {elapsed:.3f} seconds")