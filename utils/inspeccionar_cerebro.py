from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

def mirar_cerebro():
    print("🔍 Conectando al cerebro vectorial...")
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    if os.path.exists("chroma_db"):
        vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
        
        # Buscamos directamente el concepto de dinero
        print("🎣 Pescando documentos sobre 'Arancel'...")
        docs = vector_db.similarity_search("precio arancel anual", k=1)
        
        if docs:
            print("\n📄 DOCUMENTO RECUPERADO DE LA BD:")
            print("====================================")
            # Imprimimos el contenido exacto que tiene guardado
            print(docs[0].page_content) 
            print("====================================")
            
            if ":" in docs[0].page_content and "$" in docs[0].page_content:
                print("\n✅ CONFIRMADO: El documento guardado tiene el formato unido.")
            else:
                print("\n⚠️ ALERTA: El documento se ve separado.")
        else:
            print("❌ No encontré documentos.")
    else:
        print("❌ No existe la carpeta chroma_db")

if __name__ == "__main__":
    mirar_cerebro()