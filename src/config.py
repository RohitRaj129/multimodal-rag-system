"""
config.py
---------
Central place for API keys, model names, paths. Loaded once, imported everywhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# --- Models ---
VISION_MODEL = "qwen2.5vl:7b"                              # Ollama, local
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"   # NVIDIA NIM, cloud
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- ChromaDB ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "products"

# --- Retrieval defaults ---
DEFAULT_TOP_K = 5