import os
import time
import re
import sys
from dotenv import load_dotenv

# --- IMPORTS DE LANGCHAIN, MEMORIA Y VECTORSTORE ---
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Componentes clave para el historial de chat
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# =============================================================================
# 0. CONFIGURACIÓN INICIAL
# =============================================================================
load_dotenv()

CARPETA_DB = "chroma_db" 
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SESSION_ID = "sesion_usuario_local"  

# Configuración de búsqueda
MAX_CONTEXT_CHARS = 12000  
K_NORMAL = 4              
K_DINERO = 10 

# =============================================================================
# 1. LAZY LOADING: VARIABLES GLOBALES (VACÍAS AL INICIO)
# =============================================================================
# Aquí está el secreto. Al principio no hay NADA cargado.
sistema_cargado = False
vector_db = None
conversational_rag_chain = None
store = {}  # Memoria de sesiones

# =============================================================================
# 2. PROMPT DEL SISTEMA
# =============================================================================
SYSTEM_PROMPT_V3 = (
    "Eres MauricIA, la asistente oficial de Postgrados USACH.\n"
    "Tus instrucciones son INVIOLABLES. Responde basándote en el CONTEXTO y el HISTORIAL.\n"
    "\n"
    "🧠 PROTOCOLO DE RAZONAMIENTO (NO IMPRIMIR):\n"
    "1. ANALIZA EL HISTORIAL MENTALMENTE: Revisa si el usuario ya mencionó un programa.\n"
    "2. DETECCIÓN DE AMBIGÜEDAD: Si no sabes el programa, pregunta.\n"
    "⛔ PROHIBICIONES DE FORMATO: NO uses etiquetas como 'Respuesta:', 'Paso 1:'.\n"
    "\n"
    "🚨 REGLAS DE SEGURIDAD:\n"
    "- ⛔ NO ACADÉMICO: Recetas, gym, clima -> 'No tengo información sobre eso'.\n"
    "- ✅ INFORMACIÓN VÁLIDA: Costos, Mallas, Becas, Requisitos y CONTACTO.\n"
    "- 📝 Si preguntan por Profesores o Líneas de investigación: responde que estará pronto en el contexto.\n"
    "- 📝 Nota mínima pregrado: responde que no influye.\n"
    "- 📝 Co-tutela o carrera distinta: responde que SÍ es posible.\n"
    "💰 REGLAS FINANCIERAS:\n"
    "- MATRÍCULA (~$167.000) != ARANCEL (Millones).\n"
    "- PROHIBIDO MULTIPLICAR o sumar valores.\n"
    "📝 FORMATO: Respuesta directa, cálida, usa viñetas y entrega LINKS si hay."
)

# Respuestas rápidas (No necesitan IA)
RESP_NO_ACADEMICO = "No tengo información sobre servicios no académicos, solo sobre postgrados."
RESP_BLOQUEO = "Lo siento, solo puedo responder consultas sobre Postgrados USACH."

# Filtros Regex
INYECCION_PROHIBIDA = ["ignora", "ignore", "olvida", "jailbreak", "modo desarrollador"]
NO_ACADEMICO_KW = ["receta", "cocina", "pizza", "sushi", "chiste", "clima", "piscina", "gym", "casino"]
SALUDOS_KW = {"hola", "holi", "buenas", "buenos", "dias", "saludos", "hey", "que", "tal", "mauricia"}
KW_DINERO = ("cuanto", "precio", "valor", "costo", "sale", "arancel", "matricula")

_re_inyeccion = re.compile("|".join(re.escape(x) for x in INYECCION_PROHIBIDA), re.IGNORECASE)
_re_noacad = re.compile("|".join(re.escape(x) for x in NO_ACADEMICO_KW), re.IGNORECASE)

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
def es_saludo_puro(user_input: str) -> bool:
    t = re.sub(r'[^\w\s]', '', (user_input or "").lower().strip())
    words = t.split()
    return len(words) < 6 and any(w in SALUDOS_KW for w in words)

def es_consulta_dinero(user_input: str) -> bool:
    return any(k in (user_input or "").lower() for k in KW_DINERO)

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =============================================================================
# 4. FUNCIÓN DE CARGA PESADA (SE EJECUTA SOLO CUANDO ES NECESARIO)
# =============================================================================
def inicializar_sistema():
    """Carga Modelos y Base de Datos. Tarda unos segundos."""
    global vector_db, conversational_rag_chain, sistema_cargado
    
    print("⚙️  Despertando a MauricIA (Cargando modelos en RAM)...")
    
    # 1. Cargar LLM
    MODO_LOCAL = False # En Render siempre es False
    
    try:
        if MODO_LOCAL:
            llm = ChatOllama(
                base_url=os.getenv("OLLAMA_BASE_URL"),
                model=os.getenv("OLLAMA_MODEL"), 
                temperature=0.0
            )
        else:
            if not os.getenv("GITHUB_TOKEN"):
                print("❌ Error: Falta GITHUB_TOKEN")
                return False
                
            llm = ChatOpenAI(
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=os.getenv("MODEL_NAME"),
                api_key=os.getenv("GITHUB_TOKEN"),
                temperature=0.0,
                max_tokens=300
            )
    except Exception as e:
        print(f"❌ Error cargando LLM: {e}")
        return False

    # 2. Cargar Embeddings y Chroma
    print("   - Conectando memoria a largo plazo...", end=" ")
    try:
        embedding_function = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
        
        if os.path.exists(CARPETA_DB):
            vector_db = Chroma(
                persist_directory=CARPETA_DB,
                embedding_function=embedding_function
            )
            print("✅ ChromaDB conectado.")
        else:
            print("❌ Error: No existe la carpeta chroma_db")
            return False
            
    except Exception as e:
        print(f"❌ Error cargando Chroma: {e}")
        return False

    # 3. Construir la Cadena RAG
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_V3),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "CONTEXTO RECUPERADO:\n{context}\n\nPREGUNTA DEL USUARIO:\n{input}")
    ])
    
    chain = qa_prompt | llm | StrOutputParser()
    
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    sistema_cargado = True
    print("🚀 Sistema MauricIA iniciado correctamente.")
    return True

# =============================================================================
# 5. PUNTO DE ENTRADA PRINCIPAL (LO QUE LLAMA LA API)
# =============================================================================
def obtener_respuesta_agente(user_input: str, session_id: str = SESSION_ID) -> str:
    global sistema_cargado
    
    user_input = (user_input or "").strip()
    if not user_input: return "..."

    # --- FASE 1: Filtros Rápidos (No requieren IA ni carga) ---
    if _re_inyeccion.search(user_input): return RESP_BLOQUEO
    if _re_noacad.search(user_input): return RESP_NO_ACADEMICO
    
    # Saludo rápido sin cargar el modelo pesado
    if es_saludo_puro(user_input):
        return "¡Hola! Soy MauricIA, tu asistente de Postgrados USACH. ¿Sobre qué programa te gustaría informarte hoy?"

    # --- FASE 2: LAZY LOADING (Carga Perezosa) ---
    # Solo cargamos la IA si realmente necesitamos pensar
    if not sistema_cargado:
        exito = inicializar_sistema()
        if not exito:
            return "⚠️ Error crítico: No pude inicializar mi base de conocimientos. Revisa los logs del servidor."

    # --- FASE 3: Generación de Respuesta ---
    try:
        k_val = K_DINERO if es_consulta_dinero(user_input) else K_NORMAL
        query_search = user_input
        if es_consulta_dinero(user_input):
            query_search += " arancel matrícula costo valor"

        # Búsqueda Vectorial
        docs = vector_db.similarity_search(query_search, k=k_val)
        
        contexto_str = "\n\n".join([d.page_content for d in docs])
        if len(contexto_str) > MAX_CONTEXT_CHARS:
            contexto_str = contexto_str[:MAX_CONTEXT_CHARS]
        if not docs:
            contexto_str = "No se encontró información específica."

        # Invocación al LLM
        respuesta = conversational_rag_chain.invoke(
            {"input": user_input, "context": contexto_str},
            config={"configurable": {"session_id": session_id}}
        )
        return respuesta

    except Exception as e:
        print(f"Error generando respuesta: {e}")
        return "Lo siento, tuve un problema interno procesando tu solicitud. Intenta de nuevo."

# =============================================================================
# 6. MODO LOCAL (SOLO PARA PRUEBAS EN TU PC)
# =============================================================================
if __name__ == "__main__":
    print("\n🎓 MODO CLI LOCAL")
    while True:
        txt = input("\n🧑 Tú: ")
        if txt.lower() == "salir": break
        print("🤖 MauricIA:", obtener_respuesta_agente(txt))