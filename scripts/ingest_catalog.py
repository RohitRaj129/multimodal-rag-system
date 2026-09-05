import json
from src.ingestion import ingest_products

with open("data/catalog.json") as f:
    catalog = json.load(f)

count = ingest_products(catalog)
print(f"{count} products embedded and stored")