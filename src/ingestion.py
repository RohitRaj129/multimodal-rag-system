"""
ingestion.py
------------
1. ingest_products()   -> batch embed catalog JSONs -> Chroma (run once/catalog update)
2. query_from_vision()  -> vision.py JSON output -> query -> Chroma retrieval
"""

from typing import List, Dict, Any, Optional
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from src.embeddings import product_json_to_text, NVIDIA_API_KEY  # your existing flattener

embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2",
    api_key=NVIDIA_API_KEY,
    truncate="NONE",
)

vectorstore = Chroma(
    collection_name="products",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


# --------------------------------------------------------------------------- #
# 1. INGESTION — batch product JSONs -> vectors -> Chroma
# --------------------------------------------------------------------------- #

def ingest_products(products: List[Dict[str, Any]]) -> int:
    """
    Take multiple product JSON dicts (your catalog data), embed each,
    store in Chroma. Returns count ingested.
    """
    texts, ids, metadatas = [], [], []

    for p in products:
        texts.append(product_json_to_text(p))
        ids.append(str(p["id"]))
        metadatas.append({
            "brand": p.get("brand", ""),
            "category": p.get("category", ""),
            "price": float(p.get("price", 0)),
            "rating": float(p.get("rating", 0)),
            "image_url": p.get("image_url", ""),
            "product_url": p.get("product_url", ""),
        })

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(ids)


# --------------------------------------------------------------------------- #
# 2. QUERY — vision.py output -> search string -> retrieval
# --------------------------------------------------------------------------- #

def vision_json_to_query(vision_json: Dict[str, Any], user_text: str = "") -> str:
    """
    Convert uploaded image's vision analysis into a search query string.
    Combines with optional user text ("under 5000", "show alternatives").
    """
    parts = [
        vision_json.get("category", ""),
        vision_json.get("color", vision_json.get("colors", "")),
        vision_json.get("description", ""),
        user_text,
    ]
    return " ".join(str(p) for p in parts if p).strip()


def query_from_vision(
    vision_json: Dict[str, Any],
    user_text: str = "",
    top_k: int = 5,
    max_price: Optional[float] = None,
) -> List[Any]:
    """
    Full query path: uploaded image's vision JSON -> query string -> Chroma search.
    """
    query = vision_json_to_query(vision_json, user_text)

    filter_dict = {"price": {"$lte": max_price}} if max_price else None

    results = vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
    return results