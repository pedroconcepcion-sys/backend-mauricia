import os
import re
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACIÓN ---
CARPETA_DATA = "data"
RUTA_DB = "chroma_db"

# 🗺️ TUS URLS ORIGINALES
URLS_POR_PROGRAMA = {
    "doctorado": "https://www.postgradosudesantiago.cl/wp-content/uploads/2023/Malla_Doctorado_Informatica.pdf",
    "magister": "https://www.postgradosudesantiago.cl/wp-content/uploads/2023/Malla_Magister_Informatica.pdf",
    "diplomado": "https://www.postgradosudesantiago.cl/diplomados" # Ajusta si tienes URL específica
}

def limpiar_texto_maestro(texto):
    """
    Combina TU función de precios con la limpieza de basura web.
    """
    # 1. ELIMINAR BASURA (Trackers de Twitter/Pixels que confunden a la IA)
    patron_basura = r'(?i)(events=%5B%5B%22pageview|tw_document_href=|integration=advertiser|p_id=Twitter).*'
    texto = re.sub(patron_basura, '', texto)
    
    # 2. UNIR PRECIOS (Tu lógica original para juntar "Arancel" con "$ Valor")
    patron_precio = r'(?i)(.*(?:arancel|matr[ií]cula|costo|valor).*?)\n\s*(\$\s*.*)'
    texto = re.sub(patron_precio, r'\1: \2', texto)
    
    return texto

def main():
    # 1. Validación de carpeta
    if not os.path.exists(CARPETA_DATA):
        print(f"❌ Error: La carpeta '{CARPETA_DATA}' no existe.")
        return

    print(f"📚 Escaneando carpeta: {CARPETA_DATA}...")
    documentos_totales = []

    # --- A) CARGAR ARCHIVOS MARKDOWN (.md) ---
    try:
        loader_md = DirectoryLoader(CARPETA_DATA, glob="*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs_md = loader_md.load()
        print(f"   - Archivos Markdown encontrados: {len(docs_md)}")
        
        for doc in docs_md:
            doc.page_content = limpiar_texto_maestro(doc.page_content)
            doc.page_content = f"CONTEXTO WEB USACH (CRAWLER):\n{doc.page_content}"
        
        documentos_totales.extend(docs_md)
    except Exception as e:
        print(f"   ⚠️ Alerta leyendo MDs: {e}")

    # --- B) CARGAR ARCHIVOS PDF (.pdf) ---
    try:
        loader_pdf = DirectoryLoader(CARPETA_DATA, glob="*.pdf", loader_cls=PyPDFLoader)
        docs_pdf = loader_pdf.load()
        print(f"   - Archivos PDF encontrados: {len(docs_pdf)}")

        for doc in docs_pdf:
            # 1. Identificar Programa (TU LÓGICA RECUPERADA)
            nombre_archivo = doc.metadata.get('source', '').lower()
            etiqueta_programa = "PROGRAMA DESCONOCIDO"
            url_descarga = "No disponible"

            if "doctorado" in nombre_archivo:
                etiqueta_programa = "DOCTORADO EN CIENCIAS DE LA INGENIERÍA, MENCIÓN INFORMÁTICA"
                url_descarga = URLS_POR_PROGRAMA["doctorado"]
            elif "magister" in nombre_archivo or "magíster" in nombre_archivo:
                etiqueta_programa = "MAGÍSTER EN INGENIERÍA INFORMÁTICA"
                url_descarga = URLS_POR_PROGRAMA["magister"]
            elif "diplomado" in nombre_archivo:
                etiqueta_programa = "DIPLOMADO EN CIBERSEGURIDAD (EJEMPLO)"
                url_descarga = URLS_POR_PROGRAMA.get("diplomado", "No disponible")

            # 2. Limpieza Maestra
            doc.page_content = limpiar_texto_maestro(doc.page_content)
            
            # 3. Inyección de Header con URL
            header = (
                f"DOCUMENTO OFICIAL/MALLA (PDF) DEL: {etiqueta_programa}.\n"
                f"📥 PUEDES DESCARGAR EL PDF AQUÍ: {url_descarga}\n"
                f"--------------------------------------------------\n"
            )
            doc.page_content = header + doc.page_content
            
        documentos_totales.extend(docs_pdf)
    except Exception as e:
        print(f"   ⚠️ Alerta leyendo PDFs: {e}")

    if not documentos_totales:
        print("❌ No encontré documentos.")
        return

    # --- C) CHUNKING MEJORADO (LA CORRECCIÓN CRÍTICA) ---
    # Usamos 1500/400 para evitar cortar tablas de precios y párrafos largos
    print(f"✂️  Dividiendo {len(documentos_totales)} documentos (Chunk=1500, Overlap=400)...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,    # Más grande para capturar contexto completo
        chunk_overlap=400,  # Overlap grande para asegurar continuidad de precios
        separators=["\n\n", "\n", "####", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documentos_totales)
    print(f"   -> Se generaron {len(chunks)} fragmentos robustos.")

    # --- D) GUARDAR EN CHROMA ---
    print("🧠 Generando cerebro vectorial...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)
        print("   (Base de datos anterior eliminada)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    
    cantidad = vectorstore._collection.count()
    print(f"✅ ¡Cerebro actualizado! {cantidad} fragmentos guardados con URLs y limpieza.")

if __name__ == "__main__":
    main()