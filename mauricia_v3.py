import os
import time
import re
import sys
from dotenv import load_dotenv

# --- IMPORTS LIGEROS PARA PRODUCCIÓN ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Cambio clave aquí
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# =============================================================================
# 0. CONFIGURACIÓN INICIAL
# =============================================================================
load_dotenv()

# Usamos la carpeta que generaste con OpenAI
CARPETA_DB = "chroma_db_prod" 
# Modelo de OpenAI: rápido, barato y no consume RAM en el servidor
MODELO_EMBEDDINGS = "text-embedding-3-small"
SESSION_ID = "sesion_usuario_local"  

MAX_CONTEXT_CHARS = 12000  
K_NORMAL = 4              
K_DINERO = 10 

# =============================================================================
# 1. LAZY LOADING: VARIABLES GLOBALES
# =============================================================================
sistema_cargado = False
vector_db = None
conversational_rag_chain = None
store = {} 

# =============================================================================
# 2. PROMPT DEL SISTEMA (Tu lógica original intacta)
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

RESP_NO_ACADEMICO = "No tengo información sobre servicios no académicos, solo sobre postgrados."
RESP_BLOQUEO = "Lo siento, solo puedo responder consultas sobre Postgrados USACH."

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
# 4. INICIALIZACIÓN LIGERA (OPENAI CLOUD)
# =============================================================================
def inicializar_sistema():
    global vector_db, conversational_rag_chain, sistema_cargado
    
    print("☁️ Conectando con el cerebro en la nube (OpenAI Mode)...")
    
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("❌ Error: GITHUB_TOKEN no configurado.")
        return False
    
    try:
        # 1. Cargar LLM (GPT-4o mini)
        llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("MODEL_NAME"),
            api_key=api_key,
            temperature=0.0,
            max_tokens=300
        )

        # 2. Embeddings de OpenAI (No consumen RAM local)
        embedding_function = OpenAIEmbeddings(
            model=MODELO_EMBEDDINGS,
            api_key=api_key
        )
        
        # 3. Conectar ChromaDB
        if os.path.exists(CARPETA_DB):
            vector_db = Chroma(
                persist_directory=CARPETA_DB,
                embedding_function=embedding_function
            )
            print("✅ ChromaDB (OpenAI) conectado.")
        else:
            print(f"❌ Error: No existe la carpeta {CARPETA_DB}")
            return False
            
        # 4. Construir Cadena RAG
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
        return True

    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return False

# =============================================================================
# 5. OBTENER RESPUESTA (Con Lazy Loading)
# =============================================================================
def obtener_respuesta_agente(user_input: str, session_id: str = SESSION_ID) -> str:
    global sistema_cargado
    
    user_input = (user_input or "").strip()
    if not user_input: return "..."

    if _re_inyeccion.search(user_input): return RESP_BLOQUEO
    if _re_noacad.search(user_input): return RESP_NO_ACADEMICO
    
    if es_saludo_puro(user_input):
        return "¡Hola! Soy MauricIA, tu asistente de Postgrados USACH. ¿Sobre qué programa te gustaría informarte hoy?"

    if not sistema_cargado:
        if not inicializar_sistema():
            return "⚠️ El cerebro está teniendo problemas para iniciar. Revisa los logs."

    try:
        k_val = K_DINERO if es_consulta_dinero(user_input) else K_NORMAL
        query_search = user_input
        if es_consulta_dinero(user_input):
            query_search += " arancel matrícula costo valor"

        # Búsqueda
        docs = vector_db.similarity_search(query_search, k=k_val)
        
        contexto_str = "\n\n".join([d.page_content for d in docs])
        if len(contexto_str) > MAX_CONTEXT_CHARS:
            contexto_str = contexto_str[:MAX_CONTEXT_CHARS]

        # Invocación
        respuesta = conversational_rag_chain.invoke(
            {"input": user_input, "context": contexto_str},
            config={"configurable": {"session_id": session_id}}
        )
        return respuesta

    except Exception as e:
        print(f"Error: {e}")
        return "Lo siento, tuve un problema procesando tu solicitud. ¿Podrías intentar de nuevo?"

if __name__ == "__main__":
    print("\n🎓 MAURICIA CLOUD READY")
    while True:
        txt = input("\n🧑 Tú: ")
        if txt.lower() == "salir": break
        print("🤖 MauricIA:", obtener_respuesta_agente(txt))