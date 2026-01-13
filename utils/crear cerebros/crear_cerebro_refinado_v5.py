import os
import re
import shutil
# Importamos los cargadores necesarios para carpetas, markdown y PDF
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACIÓN ---
CARPETA_DATA = "data"
RUTA_DB = "chroma_db_local"

def estandarizar_formato_precios(texto):
    """
    TU FUNCIÓN ORIGINAL (INTACTA):
    Une títulos de costos con sus valores cuando están separados por saltos de línea.
    """
    # Patrón: Título (Arancel/Matrícula) + salto línea + Signo $
    patron = r'(?i)(.*(?:arancel|matr[ií]cula|costo|valor).*?)\n\s*(\$\s*.*)'
    texto_arreglado = re.sub(patron, r'\1: \2', texto)
    return texto_arreglado

def main():
    # 1. Validación de carpeta
    if not os.path.exists(CARPETA_DATA):
        print(f"❌ Error: La carpeta '{CARPETA_DATA}' no existe. Ejecuta primero la ingesta.")
        return

    print(f"📚 Escaneando carpeta: {CARPETA_DATA}...")

    documentos_totales = []

    # --- A) CARGAR ARCHIVOS MARKDOWN (.md) ---
    # Usamos DirectoryLoader para cargar TODOS los .md que genere tu crawler
    try:
        loader_md = DirectoryLoader(CARPETA_DATA, glob="*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs_md = loader_md.load()
        print(f"   - Archivos Markdown encontrados: {len(docs_md)}")
        
        # Aplicamos TU limpieza de precios a cada documento MD
        for doc in docs_md:
            # 1. Limpieza de precios
            doc.page_content = estandarizar_formato_precios(doc.page_content)
            # 2. Etiqueta de contexto para la IA
            doc.page_content = f"CONTEXTO WEB USACH (CRAWLER):\n{doc.page_content}"
        
        documentos_totales.extend(docs_md)
    except Exception as e:
        print(f"   ⚠️ Alerta leyendo MDs: {e}")

# --- B) CARGAR ARCHIVOS PDF (.pdf) ---
    try:
        loader_pdf = DirectoryLoader(CARPETA_DATA, glob="*.pdf", loader_cls=PyPDFLoader)
        docs_pdf = loader_pdf.load()
        print(f"   - Archivos PDF encontrados: {len(docs_pdf)}")

        # 🗺️ MAPA DE LINKS: Asigna URL según palabras clave en el nombre del archivo
        # REGLA: El nombre del archivo en tu carpeta 'data' debe contener estas palabras clave.
        URLS_POR_PROGRAMA = {
            "doctorado": "https://www.postgradosudesantiago.cl/wp-content/uploads/2023/Malla_Doctorado_Informatica.pdf",
            "magister": "https://www.postgradosudesantiago.cl/wp-content/uploads/2023/Malla_Magister_Informatica.pdf",
            "diplomado": "https://www.postgradosudesantiago.cl/link_diplomado_ejemplo.pdf"
        }

        for doc in docs_pdf:
            # 1. Obtenemos el nombre del archivo desde los metadatos
            ruta_archivo = doc.metadata.get('source', '').lower()
            
            # 2. Determinamos de qué programa es
            etiqueta_programa = "PROGRAMA DESCONOCIDO"
            url_descarga = "No disponible"

            if "doctorado" in ruta_archivo:
                etiqueta_programa = "DOCTORADO EN CIENCIAS DE LA INGENIERÍA, MENCIÓN INFORMÁTICA"
                url_descarga = URLS_POR_PROGRAMA["doctorado"]
            elif "magister" in ruta_archivo or "magíster" in ruta_archivo:
                etiqueta_programa = "MAGÍSTER EN INGENIERÍA INFORMÁTICA"
                url_descarga = URLS_POR_PROGRAMA["magister"]
            elif "diplomado" in ruta_archivo:
                etiqueta_programa = "DIPLOMADO EN CIBERSEGURIDAD (EJEMPLO)"
                url_descarga = URLS_POR_PROGRAMA["diplomado"]

            # 3. Limpieza de precios (Tu función maestra)
            doc.page_content = estandarizar_formato_precios(doc.page_content)
            
            # 4. INYECCIÓN INTELIGENTE DE CONTEXTO
            # Ahora el encabezado cambia según el archivo
            header = (
                f"DOCUMENTO OFICIAL/MALLA (PDF) DEL: {etiqueta_programa}.\n"
                f"📥 PUEDES DESCARGAR EL PDF AQUÍ: {url_descarga}\n"
                f"--------------------------------------------------\n"
            )
            doc.page_content = header + doc.page_content
            
        documentos_totales.extend(docs_pdf)
    except Exception as e:
        print(f"   ⚠️ Alerta leyendo PDFs: {e}")

    # Verificación final
    if not documentos_totales:
        print("❌ No encontré ningún documento (.md o .pdf). Revisa la carpeta 'data'.")
        return

    # 3. Chunking (Igual que tu V4) para algoritmo genetico
    print(f"✂️  Dividiendo {len(documentos_totales)} documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documentos_totales)
    print(f"   -> Se generaron {len(chunks)} fragmentos (chunks).")

    # 4. Guardar Embeddings (Igual que tu V4)
    print("🧠 Generando cerebro vectorial...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)
        print("   (Base de datos anterior eliminada para sobrescribir)")


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    
    # --- AHORA SÍ USAMOS LA VARIABLE ---
    # Le preguntamos al objeto cuántos vectores tiene dentro
    cantidad_vectores = vectorstore._collection.count()
    
    print(f"✅ ¡Cerebro actualizado! Se guardaron {cantidad_vectores} fragmentos de información en el disco.")
    

if __name__ == "__main__":
    main()