import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURACIÓN ---
CARPETA_DATA = "data"
# REVISA: Asegúrate que este sea el nombre EXACTO de tu archivo en la carpeta data
NOMBRE_ARCHIVO = "usach_doct_Inform_data.md" 
RUTA_DB = "chroma_db"

def main():
    # 1. Cargar el Markdown
    ruta_completa = os.path.join(CARPETA_DATA, NOMBRE_ARCHIVO)
    
    if not os.path.exists(ruta_completa):
        print(f"❌ ERROR: No encuentro el archivo {ruta_completa}")
        return

    print(f"📚 Leyendo archivo: {NOMBRE_ARCHIVO}...")
    with open(ruta_completa, "r", encoding="utf-8") as f:
        texto_raw = f.read()

    # --- VERIFICACIÓN DE SEGURIDAD ---
    # Si esta línea imprime "No", el problema es el archivo de origen, no el código.
    if "3.836.655" in texto_raw:
        print("✅ El precio del arancel SÍ está en el archivo original.")
    else:
        print("⚠️ ALERTA: El precio del arancel NO aparece en el archivo .md. Revisa la ingesta.")

    # 2. CHUNKING ROBUSTO (Estrategia de Ventana Grande)
    # Usamos chunks de 1000 caracteres con 300 de solapamiento.
    # Esto asegura que si el precio está lejos del título, igual entren en el mismo pedazo.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = text_splitter.create_documents([texto_raw])
    
    # Agregamos metadata simple para ayudar al debug
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i
        # Un truco: añadimos el nombre del archivo al contenido para dar contexto
        doc.page_content = f"FUENTE: Documentación Oficial Postgrado USACH\nCONTENIDO:\n{doc.page_content}"

    print(f"🧩 Documento dividido en {len(docs)} fragmentos grandes.")

    # 3. Embeddings
    print("🧠 Cargando modelo de embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Guardar
    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)

    print("💾 Guardando nueva memoria...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    print("✅ ¡CEREBRO RECREADO EXITOSAMENTE!")

if __name__ == "__main__":
    main()