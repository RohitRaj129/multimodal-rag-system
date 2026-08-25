import json
import ollama


PROMPT = """
Analyze the image and identify the main object or product shown.

Return ONLY valid JSON. Do not include markdown, explanations, or any text outside the JSON object.

Use exactly this structure:

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
- Base the output only on information that can be reliably determined from the image.
- Do not guess or invent information.
- Use "unknown" when a value cannot be determined.
- Use [] when no reliable values can be identified for a list field.
- Identify visible text, logos, labels, and markings only when they can actually be seen.
- Describe only visually observable characteristics.
- Keep the description concise and useful for semantic search.
- Generate relevant search keywords based only on the visual information.
- Do not infer hidden specifications or information that cannot be verified from the image.
- Do not assume the object's condition from the image quality or photography style.
- Ensure the response is valid JSON with the exact field names provided above.

Return ONLY the JSON object.
"""


def analyze_image(image_path):
    try:
        response = ollama.chat(
            model="qwen2.5vl:7b",
            format="json",
            options={
                'temperature': 0.0,
                'num_ctx': 8192          # bump context window
                },
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                    "images": [image_path]
                }
            ]
        )

        return json.loads(response["message"]["content"])

    except json.JSONDecodeError:
        raise ValueError("Vision model returned invalid JSON.")

    except Exception as e:
        raise RuntimeError(f"Vision analysis failed: {e}")


if __name__ == "__main__":
    result = analyze_image("data/sample_images/shoe.jpg")
    print(json.dumps(result, indent=2))