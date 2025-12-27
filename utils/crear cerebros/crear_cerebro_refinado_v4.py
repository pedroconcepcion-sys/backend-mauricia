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

def estandarizar_formato_precios(texto):
    """
    CORRECCIÓN DINÁMICA (Sin valores fijos):
    Busca patrones donde un título de costo (Arancel/Matrícula) quedó separado
    de su valor numérico por un salto de línea.
    
    Ejemplo entrada:
       #### ARANCEL
       $ 5.000.000 (O cualquier otro valor)
       
    Ejemplo salida:
       #### ARANCEL: $ 5.000.000
    """
    
    # EXPRESIÓN REGULAR EXPLICADA:
    # 1. (?i) -> Insensible a mayúsculas/minúsculas
    # 2. (.*(?:arancel|matr[ií]cula|costo|valor).*?) -> Grupo 1: Cualquier línea que tenga palabras clave de dinero
    # 3. \n\s* -> Salto de línea y espacios opcionales
    # 4. (\$[:\d].*) -> Grupo 2: La siguiente línea empieza con signo $
    patron = r'(?i)(.*(?:arancel|matr[ií]cula|costo|valor).*?)\n\s*(\$\s*.*)'
    
    # Unimos Grupo 1 + ": " + Grupo 2
    texto_arreglado = re.sub(patron, r'\1: \2', texto)
    
    return texto_arreglado

def main():
    # 1. Cargar archivo
    ruta_completa = os.path.join(CARPETA_DATA, NOMBRE_ARCHIVO)
    if not os.path.exists(ruta_completa):
        print("❌ Error: Archivo no encontrado.")
        return

    print(f"📚 Procesando: {NOMBRE_ARCHIVO}...")
    with open(ruta_completa, "r", encoding="utf-8") as f:
        texto_raw = f.read()

    # 2. LIMPIEZA ESTRUCTURAL (Aquí está la mejora)
    # No importa el precio, solo importa que esté debajo del título.
    texto_procesado = estandarizar_formato_precios(texto_raw)
    
    # Debug: Verificar si funcionó la unión sin saber el precio
    if "ARANCEL:" in texto_procesado or "MATRÍCULA:" in texto_procesado.upper():
        print("✅ Estructura corregida: Los precios se unieron a sus títulos.")
    
    # 3. Chunking
    # Usamos chunks generosos para dar contexto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = text_splitter.create_documents([texto_procesado])
    
    # Metadata para RAG
    for doc in docs:
        doc.page_content = f"CONTEXTO DOCUMENTACIÓN USACH:\n{doc.page_content}"

    # 4. Guardar Embeddings
    print("🧠 Generando cerebro vectorial...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    print("✅ ¡Cerebro actualizado! Ahora es resistente a cambios de precios.")

if __name__ == "__main__":
    main()