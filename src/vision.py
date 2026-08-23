import ollama

response = ollama.chat(
    model='llava:7b',
    format='json',
    messages=[{
        'role': 'user',
        'content': '''Analyze the product shown in this image for an e-commerce product search and recommendation system.

Return ONLY valid JSON. Do not include markdown, explanations, or any text outside the JSON object.

Use this exact structure:

{
  "product_name": "",
  "category": "",
  "subcategory": "",
  "brand": "",
  "model": "",
  "colors": [],
  "materials": [],
  "visible_features": [],
  "style": "",
  "condition": "",
  "text_visible_in_image": [],
  "description": "",
  "search_keywords": []
}

Instructions:

1. Identify the general product category and subcategory.
2. Identify the brand only if it is clearly visible or strongly identifiable from the image. Otherwise use "unknown".
3. Identify the exact product name or model only if it is clearly visible or confidently identifiable. Otherwise use "unknown".
4. List all clearly visible colors.
5. Identify materials only when they can reasonably be determined from the image. Otherwise use an empty array.
6. List important visually observable product features.
7. Describe the product's style or design when applicable. If not applicable, use "unknown".
8. Estimate the visible condition of the product, such as "new", "used", "worn", or "unknown". Do not assume condition from image quality.
9. Extract any readable text, logos, labels, model numbers, or markings visible in the image.
10. Write a concise but descriptive product description suitable for semantic search.
11. Generate relevant search keywords based ONLY on information visible in the image.
12. Never invent specifications, prices, dimensions, technical details, model numbers, materials, or other information that cannot be determined from the image.
13. If information is unavailable or uncertain, use "unknown" for string fields and [] for list fields.
14. Keep the output concise and focused on information useful for product identification, similarity search, and e-commerce retrieval.''',
        'images': ['data/sample_images/shoe.jpg']
    }]
)
print(response['message']['content'])