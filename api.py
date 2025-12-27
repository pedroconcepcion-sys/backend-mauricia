import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- IMPORTAMOS TU LÓGICA EXISTENTE ---
# Asegúrate de que tu archivo original se llame 'mauricia_v3.py'
# y que 'obtener_respuesta_agente' esté disponible.
from mauricia_v3 import obtener_respuesta_agente, SESSION_ID

# 1. Crear la APP
app = FastAPI(title="API MauricIA USACH", version="3.0")

# 2. Configurar CORS (Permisos)
# Esto permite que tu HTML (que corre en el navegador) hable con Python (que corre en tu PC)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, cambia esto por tu dominio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Modelo de datos (Qué esperamos recibir del usuario)
class ConsultaUsuario(BaseModel):
    mensaje: str
    session_id: str = "usuario_web_default"

# 4. ENDPOINT: Estado del sistema
@app.get("/")
def home():
    return {"status": "online", "bot": "MauricIA v3"}

# 5. ENDPOINT: Chat (El corazón del sistema)
@app.post("/chat")
def chat_endpoint(consulta: ConsultaUsuario):
    """
    Recibe el mensaje del frontend, lo pasa a MauricIA, y devuelve la respuesta.
    """
    try:
        print(f"📩 Recibido: {consulta.mensaje}")
        
        # Llamamos a tu función maestra (la que ya tienes programada)
        respuesta = obtener_respuesta_agente(consulta.mensaje)
        
        return {"respuesta": respuesta}
    
    except Exception as e:
        print(f"❌ Error API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 6. Arrancar el servidor automáticamente si ejecutas este archivo
if __name__ == "__main__":
    print("🚀 Iniciando Servidor API MauricIA...")
    uvicorn.run(app, host="0.0.0.0", port=8000)