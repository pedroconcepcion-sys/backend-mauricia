# pre_download.py
import os
print("⏳ Iniciando descarga anticipada de modelos...")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    # Esto fuerza la descarga del modelo a la caché del sistema
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Modelo de Embeddings descargado con éxito.")
except Exception as e:
    print(f"❌ Error descargando modelo: {e}")

print("🚀 Pre-carga completada.")