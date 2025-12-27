import os
import re
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURACIÓN ---
CARPETA_DATA = "data"
NOMBRE_ARCHIVO = "usach_doct_Inform_data.md" 
RUTA_DB = "chroma_db"

def limpiar_y_unir_titulos(texto):
    """
    Detecta patrones donde un título Markdown (#) está separado de su valor (ej: un precio)
    por un salto de línea y los une en una sola línea.
    
    Transforma:
       #### ARANCEL
       $ 3.836.655
       
    En:
       #### ARANCEL: $ 3.836.655
    """
    # Expresión regular explicada:
    # 1. (#+ .*)  -> Busca uno o más '#' seguido de texto (El título)
    # 2. \n       -> Busca un salto de línea
    # 3. (\$ .*)  -> Busca un signo '$' seguido de texto (El precio)
    patron = r'(#+ .*?)\n(\$.*)'
    
    # Reemplazamos por: Grupo 1 + ": " + Grupo 2
    texto_limpio = re.sub(patron, r'\1: \2', texto)
    
    return texto_limpio

def main():
    # 1. Cargar el Markdown
    ruta_completa = os.path.join(CARPETA_DATA, NOMBRE_ARCHIVO)
    
    if not os.path.exists(ruta_completa):
        print(f"❌ ERROR: No encuentro el archivo {ruta_completa}")
        return

    print(f"📚 Leyendo archivo: {NOMBRE_ARCHIVO}...")
    with open(ruta_completa, "r", encoding="utf-8") as f:
        texto_raw = f.read()

    # --- PASO NUEVO: LIMPIEZA AUTOMÁTICA ---
    print("🧹 Aplicando pegamento semántico (uniendo Títulos con Precios)...")
    texto_procesado = limpiar_y_unir_titulos(texto_raw)
    
    # Verificación rápida
    if "ARANCEL:" in texto_procesado:
        print("✅ Corrección aplicada: Se detectó 'ARANCEL:' unido al precio.")
    else:
        print("ℹ️ No se detectaron patrones rotos de precio (o ya estaban bien).")

    # 2. CHUNKING
    # Usamos un chunk size moderado. Al haber unido las líneas, 
    # es mucho más difícil que el precio quede huérfano.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = text_splitter.create_documents([texto_procesado])
    
    # Metadata para trazabilidad
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i
        doc.page_content = f"FUENTE: Postgrado USACH Oficial\nCONTENIDO:\n{doc.page_content}"

    print(f"🧩 Documento dividido en {len(docs)} fragmentos optimizados.")

    # 3. Embeddings e Indexado
    print("🧠 Cargando modelo de embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)

    print("💾 Guardando nueva memoria...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    print("✅ ¡CEREBRO RECREADO! Ahora los precios están pegados a sus etiquetas.")

if __name__ == "__main__":
    main()