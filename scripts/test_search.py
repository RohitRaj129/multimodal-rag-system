from src.ingestion import vectorstore

query = "black running shoes"   # try something matching your catalog data
results = vectorstore.similarity_search(query, k=3)

for r in results:
    print("---")
    print("Text:", r.page_content)
    print("Metadata:", r.metadata)