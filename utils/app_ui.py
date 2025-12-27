import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Agente Postgrados USACH", page_icon="🎓")

# Cargar variables
load_dotenv()

# --- CACHÉ DE RECURSOS (Para no cargar modelos cada vez que das Enter) ---
@st.cache_resource
def cargar_cerebro():
    print("⚙️ Cargando recursos del sistema...")
    
    # 1. Embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 2. Vector DB
    carpeta_db = "chroma_db"
    if os.path.exists(carpeta_db):
        vector_db = Chroma(persist_directory=carpeta_db, embedding_function=embedding_function)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    else:
        st.error("❌ No se encontró la base de datos 'chroma_db'. Ejecuta crear_cerebro.py primero.")
        return None, None

    # 3. LLM (Ollama)
    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=0.1
    )
    
    # Warm-up rápido
    try:
        llm.invoke("test")
    except:
        st.warning("⚠️ Asegúrate de que Ollama esté corriendo en segundo plano.")

    return retriever, llm

# Cargar el cerebro una sola vez
retriever, llm = cargar_cerebro()

# --- INTERFAZ GRÁFICA ---
st.title("🎓 Asistente Postgrados USACH")
st.markdown("Pregunta sobre aranceles, mallas, fechas y requisitos del programa.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LÓGICA DEL CHAT ---
if prompt := st.chat_input("Ej: ¿Cuál es el arancel del Doctorado?"):
    
    # 1. Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # A. RAG: Buscar contexto
        if retriever:
            with st.status("🔍 Consultando base de conocimientos...", expanded=False):
                docs = retriever.invoke(prompt)
                context_text = "\n\n".join([d.page_content for d in docs])
                st.write(docs) # Muestra qué documentos encontró (útil para debug)

            # B. Prompt del Sistema
            system_msg = (
                "Eres el Asistente IA de Postgrados USACH. Responde basado en el siguiente CONTEXTO.\n"
                "Si la info no está ahí, di que no sabes.\n"
                f"CONTEXTO:\n{context_text}"
            )

            # C. Streaming de respuesta
            messages_payload = [
                SystemMessage(content=system_msg),
                HumanMessage(content=prompt)
            ]
            
            for chunk in llm.stream(messages_payload):
                content = chunk.content
                if content:
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.005)
            
            message_placeholder.markdown(full_response)
        
        else:
            st.error("Error en el sistema de recuperación.")

    # 3. Guardar en historial
    st.session_state.messages.append({"role": "assistant", "content": full_response})