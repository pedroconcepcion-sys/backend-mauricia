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

# --- CORRECCIÓN AQUÍ: Usamos 'ChatMessageHistory' genérico ---
from langchain_community.chat_message_histories import ChatMessageHistory

# =============================================================================
# 0. CONFIGURACIÓN INICIAL
# =============================================================================
load_dotenv()

# Configuración de Archivos y Modelos
CARPETA_DB = "chroma_db" 
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SESSION_ID = "sesion_usuario_local"  

"""=== FRONTERA DE PARETO (MEJORES CONFIGURACIONES) ===
ID   | Calidad  | Latencia | Configuración
------------------------------------------------------------
1    | 0.67     | 10.80s    | {'k_normal': 7, 'k_dinero': 3, 'max_chars': 12000}
2    | 0.77     | 14.11s    | {'k_normal': 10, 'k_dinero': 3, 'max_chars': 15000}
4    | 0.67     | 10.56s    | {'k_normal': 4, 'k_dinero': 5, 'max_chars': 10000}
5    | 0.63     | 10.10s    | {'k_normal': 7, 'k_dinero': 3, 'max_chars': 6000}
6    | 0.84     | 14.63s    | {'k_normal': 4, 'k_dinero': 4, 'max_chars': 12000}--------------Joya de la Corona
9    | 0.77     | 13.42s    | {'k_normal': 7, 'k_dinero': 4, 'max_chars': 7000} --------------Más velocidad
11   | 0.85     | 17.39s    | {'k_normal': 10, 'k_dinero': 5, 'max_chars': 16000}"""

MAX_CONTEXT_CHARS = 12000  
K_NORMAL = 4              
K_DINERO = 10 

print("⚙️  Inicializando MauricIA v3 (Corregido)...")

# =============================================================================
# 1. CARGA DE MODELOS (CEREBRO Y MEMORIA)
# =============================================================================

# 🎛️ INTERRUPTOR MAESTRO: ¿Usar PC Local o Nube Potente?
# True  = Gratis, Llama 3.1, Privacidad total.
# False = Pago, GPT-4o, Máxima inteligencia.
MODO_LOCAL = False 

try:
    if MODO_LOCAL:
        # A) LLM LOCAL (Llama 3.1 via Ollama)
        print("   - 🏠 Modo Local. Conectando con Ollama...", end=" ")
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"), 
            temperature=0.0,
            num_predict=300,
        )
        print(f"✅ Listo: {llm.model}")

    else:
        # B) LLM NUBE (GPT-4o via OpenAI)
        print("   - ☁️ Modo Nube. Conectando con OpenAI...", end=" ")
        
        # Verificación de seguridad para no gastar créditos por error
        if not os.getenv("GITHUB_TOKEN"):
            raise ValueError("❌ ERROR: MODO_LOCAL es False, pero no encontré la OPENAI_API_KEY en .env")

        llm = ChatOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("GITHUB_TOKEN"),
            temperature=0.0,
            max_tokens=300
            
        )
        print(f"✅ Listo: {llm.model_name}")

except Exception as e:
    print(f"\n❌ Error crítico cargando el modelo LLM: {e}")
    sys.exit(1)

# B) EMBEDDINGS & VECTOR STORE (ChromaDB)
print("   - Cargando Base de Conocimiento...", end=" ")
embedding_function = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)

if os.path.exists(CARPETA_DB):
    vector_db = Chroma(
        persist_directory=CARPETA_DB,
        embedding_function=embedding_function
    )
    print("✅ ChromaDB conectado.")
else:
    print("\n❌ ERROR: No existe la carpeta 'chroma_db'.")
    print("⚠️  SOLUCIÓN: Ejecuta primero 'python crear_cerebro_refinado_v5.py'")
    sys.exit(1)

# =============================================================================
# 2. PROMPT DEL SISTEMA (LA LÓGICA DE NEGOCIO)
# =============================================================================
SYSTEM_PROMPT_V3 = (
    "Eres MauricIA, la asistente oficial de Postgrados USACH.\n"
    "Tus instrucciones son INVIOLABLES. Responde basándote en el CONTEXTO y el HISTORIAL.\n"
    "\n"
    "🧠 PROTOCOLO DE RAZONAMIENTO (NO IMPRIMIR):\n"
    "1. ANALIZA EL HISTORIAL MENTALMENTE: Revisa si el usuario ya mencionó un programa (ej: 'Magíster en Informática').\n"
    "   - Si pregunta \"¿Cuánto cuesta?\" y antes hablaron del Magíster, asume que es sobre ese.\n"
    "2. DETECCIÓN DE AMBIGÜEDAD:\n"
    "   - Si el usuario pregunta por un dato genérico y NO sabes el programa:\n"
    "   - 🛑 DETENTE Y PREGUNTA: \"¿A cuál programa te refieres? Tengo información de Doctorados, Magísters, etc.\"\n"
    "   - Si el contexto trae info de DOS programas, diferéncialos: \"Para el Doctorado es X, para el Magíster es Y\".\n"
    "⛔ PROHIBICIONES DE FORMATO (CRÍTICO):\n"
    "   - NO uses etiquetas como 'Respuesta:', 'Formato:', 'Análisis:', 'Paso 1:'.\n"
    "   - NO expliques tu comportamiento (ej: 'La respuesta se enfoca en...').\n"
    "   - NO imprimas tu pensamiento interno.\n"
    "   - Solo entrega el mensaje final para el usuario de forma natural.\n"
    "\n"
    "🚨 REGLAS DE SEGURIDAD:\n"
    "- ⛔ NO ACADÉMICO: Si piden recetas, gym, piscina o clima -> \"No tengo información sobre servicios no académicos.\"\n"
    "- ✅ INFORMACIÓN VÁLIDA: Costos, Mallas, Becas, Requisitos y CONTACTO (Nombres de secretarias, coordinadores, correos).\n"
    "- ✅ Los programas tanto de magíster como de doctorado no son dedicación exlusiva, se puede trabajar mientras se estudia a la vez."
    "- 📝 Si preguntan: Profesores/Docentes/Académicos del claustro de cualquier programa, responde que: estará pronto en el contexto, aún no lo hemos actualizado esa información"
    "- 📝 Si preguntan: Que nota mínima de pregrado para ser aceptado en algun programa? - respondes: la nota no influye en la aceptacion, contactar a CONTACTO del programa"
    "- 📝 Si preguntan: Líneas/Lineas de investigación de los programas: responde que: estará pronto en el contexto, aún no hemos actualizado esa información"
    "- 📝 Si preguntan: Hay convenios de co-tutela con universidades extranjeras? , respondes que si, más información en el CONTACTO"
    "- 📝 Si preguntan: Si estudié una carrera de otra área distinta, puedo postular al doctorado o magister: respondes que si"
    "- 📧 CONTACTO: Si preguntan por la secretaria/o, busca en la sección de 'CONTACTO' del texto y entrega el nombre y correo si aparece.\n"
    "💰 REGLAS FINANCIERAS (ESTRICTO):\n"
    "- MATRÍCULA (~$167.000, semestral) != ARANCEL (Millones, anual).\n"
    "- 🚫 PROHIBIDO MULTIPLICAR la matrícula por 2. Entrega el valor semestral tal cual.\n" 
    "- Busca el valor exacto en el texto para el programa específico.\n"
    "- PROHIBIDO MULTIPLICAR o sumar.\n"
    "\n"
    "📝 FORMATO:\n"
    "- Respuesta directa, cálida y profesional.\n"
    "- Usa VIÑETAS para listas (becas, requisitos, etc...).\n"
    "- 📎 LINKS: Si el texto dice 'PUEDES DESCARGAR EL PDF AQUÍ', entrégalo al final con emoji 📥."
)

# Respuestas rápidas
RESP_NO_ACADEMICO = "No tengo información sobre servicios no académicos, solo sobre postgrados."
RESP_BLOQUEO = "Lo siento, solo puedo responder consultas sobre Postgrados USACH."

# =============================================================================
# 3. FILTROS Y SEGURIDAD (CAPA PYTHON)
# =============================================================================
INYECCION_PROHIBIDA = [
    "ignora", "ignore", "olvida", "forget", "system prompt", "instrucciones",
    "revela", "jailbreak", "dan", "modo desarrollador"
]
NO_ACADEMICO_KW = [
    "receta", "cocina", "pizza", "sushi", "chiste", "clima", 
    "piscina", "gimnasio", "gym", "casino", "menú"
]
SALUDOS_KW = {
    "hola", "holi", "buenas", "buenos", "dias", "tardes", "noches",
    "saludos", "hey", "hi", "que", "tal", "mauricia"
}
KW_DINERO = ("cuanto", "precio", "valor", "costo", "sale", "arancel", "matricula")

_re_inyeccion = re.compile("|".join(re.escape(x) for x in INYECCION_PROHIBIDA), re.IGNORECASE)
_re_noacad = re.compile("|".join(re.escape(x) for x in NO_ACADEMICO_KW), re.IGNORECASE)

def es_saludo_puro(user_input: str) -> bool:
    t = re.sub(r'[^\w\s]', '', (user_input or "").lower().strip())
    words = t.split()
    return len(words) < 6 and any(w in SALUDOS_KW for w in words)

def es_consulta_dinero(user_input: str) -> bool:
    return any(k in (user_input or "").lower() for k in KW_DINERO)

def detectar_filtro_programa(consulta: str):
    """
    Analiza la consulta y crea un filtro de metadatos para ChromaDB.
    Esto impide físicamente que se mezclen peras con manzanas.
    """
    consulta = consulta.lower()
    
    # Palabras clave que identifican inequívocamente al programa
    es_doctorado = "doctorado" in consulta or "doctor" in consulta or "phd" in consulta 
    es_magister = "magister" in consulta or "magíster" in consulta or "mgi" in consulta or "master" in consulta or "máster" in consulta
    es_diplomado = "diplomado" in consulta

    # Lógica de exclusión:
    # Si pregunta por Doctorado, filtramos para que la fuente contenga "doctorado"
    if es_doctorado and not es_magister:
        print("   [🔍 Filtro Activo]: Buscando SOLO en documentos de DOCTORADO.")
        return {"source": {"$contains": "doctorado"}} # Sintaxis de ChromaDB
    
    elif es_magister and not es_doctorado:
        print("   [🔍 Filtro Activo]: Buscando SOLO en documentos de MAGÍSTER.")
        return {"source": {"$contains": "magister"}} # Asume que tus archivos se llaman 'magister_algo.md'
    
    elif es_diplomado:
        return {"source": {"$contains": "diplomado"}}

    # Si menciona ambos o ninguno (ej: "¿Qué postgrados tienen?"), no filtramos
    print("   [🔍 Filtro Inactivo]: Buscando en TODO el cerebro.")
    return None

# =============================================================================
# 4. CONFIGURACIÓN DE LA CADENA CON MEMORIA (LANGCHAIN)
# =============================================================================

# A) Template del Chat
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_V3),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "CONTEXTO RECUPERADO:\n{context}\n\nPREGUNTA DEL USUARIO:\n{input}")
])

# B) Cadena Base
chain = qa_prompt | llm | StrOutputParser()

# C) Almacén de Sesiones (Memoria RAM)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        # --- CORRECCIÓN AQUÍ: Usamos ChatMessageHistory ---
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# D) Cadena Final con Historial Automático
conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =============================================================================
# 5. LÓGICA DEL AGENTE (RAG + MEMORIA)
# =============================================================================
def obtener_respuesta_agente(user_input: str) -> str:
    user_input = (user_input or "").strip()
    if not user_input: return "..."

    # --- FASE 1: Filtros de Seguridad ---
    if _re_inyeccion.search(user_input): return RESP_BLOQUEO
    if _re_noacad.search(user_input): return RESP_NO_ACADEMICO

    # --- FASE 2: Saludo Rápido ---
    if es_saludo_puro(user_input):
        return "¡Hola! Soy MauricIA, tu asistente de Postgrados USACH. ¿Sobre qué programa te gustaría informarte hoy?"

    # --- FASE 3: RAG (Retrieval Augmented Generation) ---
    try:
        k_val = K_DINERO if es_consulta_dinero(user_input) else K_NORMAL
        query_search = user_input
        if es_consulta_dinero(user_input):
            query_search += " arancel matrícula costo valor anual semestral pesos matricula"
             

        docs = vector_db.similarity_search(query_search, k=k_val)
        
        contexto_str = "\n\n".join([d.page_content for d in docs])
        if len(contexto_str) > MAX_CONTEXT_CHARS:
            contexto_str = contexto_str[:MAX_CONTEXT_CHARS]

        if not docs:
            contexto_str = "No se encontró información específica en la base de datos."

        respuesta = conversational_rag_chain.invoke(
            {"input": user_input, "context": contexto_str},
            config={"configurable": {"session_id": SESSION_ID}}
        )
        return respuesta

    except Exception as e:
        return f"⚠️ Error interno del sistema: {str(e)}"

# =============================================================================
# 6. INTERFAZ DE USUARIO (CLI)
# =============================================================================
def chatbot_streaming():
    print("\n🎓 === ASISTENTE DE POSTGRADOS USACH (MauricIA v3) ===")
    print("   (Escribe 'salir' para cerrar)\n")

    print("🔥 Calentando motores...", end="", flush=True)
    try:
        vector_db.similarity_search("test", k=1)
        print(" Listo.")
    except:
        print(" (Advertencia: Primera consulta puede ser lenta)")

    while True:
        try:
            user_input = input("\n🧑 Tú: ").strip()
        except EOFError:
            break
            
        if user_input.lower() in ["salir", "exit", "chao"]:
            print("\n🤖 MauricIA: ¡Mucho éxito en tu postulación! Hasta luego.")
            break
        
        if not user_input: continue

        print("\n🤖 MauricIA: ", end="", flush=True)
        respuesta = obtener_respuesta_agente(user_input)

        for char in respuesta:
            print(char, end="", flush=True)
            time.sleep(0.04)
        print()

if __name__ == "__main__":
    chatbot_streaming()