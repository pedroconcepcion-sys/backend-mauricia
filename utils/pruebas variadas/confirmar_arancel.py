from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

def confirmar_dato_exacto():
    print("🕵️ Iniciando búsqueda forense del Arancel...")
    
    # 1. Configurar
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    if not os.path.exists("chroma_db"):
        print("❌ Error: No existe la carpeta chroma_db")
        return

    vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
    
    # 2. BÚSQUEDA EXACTA
    # Buscamos el número específico del arancel. 
    # Si la base de datos lo tiene, aparecerá aquí.
    monto_clave = "3.836.655"
    print(f"🎣 Buscando el rastro de: {monto_clave} ...")
    
    # Pedimos los 5 mejores resultados por si acaso no sale primero
    docs = vector_db.similarity_search(monto_clave, k=5)
    
    encontrado = False
    for i, doc in enumerate(docs):
        if monto_clave in doc.page_content:
            print(f"\n✅ ¡EUREKA! Dato encontrado en el documento #{i+1}")
            print("--------------------------------------------------")
            
            # Buscamos la posición del precio
            idx = doc.page_content.find(monto_clave)
            
            # Mostramos un pedacito antes y después para ver si se unió el título
            inicio = max(0, idx - 40)
            fin = min(len(doc.page_content), idx + 40)
            fragmento = doc.page_content[inicio:fin]
            
            print(f"CONTEXTO VISUAL:\n...{fragmento}...")
            print("--------------------------------------------------")
            
            # Verificamos la unión
            if ":" in fragmento and "$" in fragmento:
                print("🌟 CONFIRMADO: El título y el precio están pegados con ':'")
            else:
                print("⚠️ OJO: El dato está, pero el formato visual sigue raro.")
            
            encontrado = True
            break # Ya lo encontramos, no seguimos buscando
            
    if not encontrado:
        print("❌ FATAL: El número 3.836.655 no existe en ninguna parte de la base de datos.")

if __name__ == "__main__":
    confirmar_dato_exacto()