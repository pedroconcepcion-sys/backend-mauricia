import os
import shutil
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURACIÓN ---
CARPETA_DATA = "data"
NOMBRE_ARCHIVO = "usach_doct_Inform_data.md" # Asegúrate que este sea tu archivo real
RUTA_DB = "chroma_db"

def main():
    # 1. Cargar el Markdown
    ruta_completa = os.path.join(CARPETA_DATA, NOMBRE_ARCHIVO)
    
    print(f"📚 Leyendo archivo: {NOMBRE_ARCHIVO}...")
    with open(ruta_completa, "r", encoding="utf-8") as f:
        texto_raw = f.read()

    # --- MEJORA CLAVE: CHUNKING ESTRUCTURADO ---
    # Esto asegura que si dice "## ARANCEL", ese título viaje pegado al precio.
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # Primera pasada: Dividir por estructura lógica
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(texto_raw)

    print(f"   estructura detectada: {len(md_header_splits)} secciones lógicas.")

    # Segunda pasada: Controlar el tamaño para que quepa en el contexto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Un poco más pequeño para ser más precisos
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    docs = text_splitter.split_documents(md_header_splits)

    # Inyectamos metadatos extra en el texto para ayudar al modelo
    # Esto ayuda a que el modelo sepa de qué habla el fragmento aunque sea corto
    for doc in docs:
        header_path = ""
        if "Header 1" in doc.metadata: header_path += f"{doc.metadata['Header 1']} > "
        if "Header 2" in doc.metadata: header_path += f"{doc.metadata['Header 2']}"
        
        # Prependemos el contexto al contenido para fortalecer la búsqueda
        doc.page_content = f"SECCIÓN: {header_path}\nCONTENIDO:\n{doc.page_content}"

    print(f"🧩 Documento final dividido en {len(docs)} fragmentos optimizados.")
    
    # Verificación visual (Debugging)
    print("--- MUESTRA DE UN CHUNK (Lo que leerá el agente) ---")
    print(docs[2].page_content) 
    print("----------------------------------------------------")

    # 3. Embeddings (Igual que antes)
    print("🧠 Cargando modelo de embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Guardar
    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)

    print("💾 Guardando nueva memoria optimizada...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=RUTA_DB
    )
    print("✅ ¡CEREBRO REFINADO LISTO!")

if __name__ == "__main__":
    main()