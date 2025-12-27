import time
import optuna

from optuna.samplers import NSGAIISampler

# Importamos los datos de prueba
from procesamiento.dataset_para_test import CASOS_PRUEBA

# --- CORRECCIÓN AQUÍ: Quitamos 'armar_query_optimizada' de la lista ---
from mauricia_v3 import vector_db, conversational_rag_chain, SESSION_ID

# --- FUNCIÓN DE EVALUACIÓN ---
def evaluar_respuesta(respuesta_ia: str, palabras_clave: list) -> float:
    """Retorna 1.0 si encuentra las palabras clave, 0.0 si no."""
    texto = (respuesta_ia or "").lower()
    matches = 0
    for palabra in palabras_clave:
        if palabra.lower() in texto:
            matches += 1
    
    # Puntaje parcial
    if len(palabras_clave) == 0: return 0.0
    if matches == len(palabras_clave):
        return 1.0
    elif matches > 0:
        return 0.5 + (0.5 * matches / len(palabras_clave))
    return 0.0

# --- LÓGICA RAG PERSONALIZADA PARA LA PRUEBA ---
def ejecutar_rag_experimental(pregunta, k, max_chars):
    """
    Simula el agente pero inyectando los K experimentales del algoritmo genético.
    """
    # 1. Búsqueda con K variable (el gen que estamos probando)
    docs = vector_db.similarity_search(pregunta, k=k)
    
    # 2. Recorte de contexto variable (el otro gen)
    contexto_str = "\n\n".join([d.page_content for d in docs])
    if len(contexto_str) > max_chars:
        contexto_str = contexto_str[:max_chars]
        
    if not docs:
        contexto_str = "No info."

    # 3. Generación (Usamos la cadena real de MauricIA v3)
    # Usamos un session_id de prueba para no mezclar con tu chat personal
    respuesta = conversational_rag_chain.invoke(
        {"input": pregunta, "context": contexto_str},
        config={"configurable": {"session_id": "sesion_optuna_training"}}
    )
    return respuesta, len(contexto_str)

# --- FUNCIÓN OBJETIVO (LO QUE SE OPTIMIZA) ---
def objective(trial):
    # 🧬 GENES A MUTAR (Hiperparámetros)
    k_normal = trial.suggest_int("k_normal", 4, 10)       
    k_dinero = trial.suggest_int("k_dinero", 2, 5)       
    max_chars = trial.suggest_int("max_chars", 5000, 16000, step=1000) 

    puntajes_calidad = []
    tiempos_respuesta = []
    
    print(f"\n🧬 Gen {trial.number}: K_NORM={k_normal}, K_DIN={k_dinero}, CHARS={max_chars}")

    for caso in CASOS_PRUEBA:
        pregunta_base = caso["pregunta"]
        
        # LÓGICA MANUAL DE OPTIMIZACIÓN DE QUERY (Reemplaza a la función borrada)
        if caso["tipo"] == "dinero":
            k_usado = k_dinero
            query_final = pregunta_base + " arancel matrícula costo valor anual semestral pesos"
        else:
            k_usado = k_normal
            query_final = pregunta_base

        # Medimos tiempo
        start_time = time.time()
        
        try:
            respuesta, _ = ejecutar_rag_experimental(query_final, k_usado, max_chars)
        except Exception as e:
            print(f"⚠️ Error en trial: {e}")
            respuesta = ""
            
        latencia = time.time() - start_time
        
        # Evaluamos calidad
        score = evaluar_respuesta(respuesta, caso["debe_contener"])
        
        puntajes_calidad.append(score)
        tiempos_respuesta.append(latencia)

    # --- CÁLCULO DE FITNESS ---
    promedio_calidad = sum(puntajes_calidad) / max(len(puntajes_calidad), 1)
    promedio_latencia = sum(tiempos_respuesta) / max(len(tiempos_respuesta), 1)

    # Guardamos métrica secundaria
    trial.set_user_attr("avg_latency", promedio_latencia)
    
    # Optuna minimizará o maximizará según la config del estudio abajo
    return promedio_calidad, promedio_latencia

if __name__ == "__main__":
    # Configuración del Algoritmo Genético (NSGA-II)
    # population_size: Cuántos "individuos" crea por generación
    sampler = NSGAIISampler(population_size=10, mutation_prob=0.15) 
    
    study = optuna.create_study(
        directions=["maximize", "minimize"], # Obj 1: Calidad (Max), Obj 2: Tiempo (Min)
        sampler=sampler,
        study_name="evolucion_mauricia"
    )

    print("🚀 Iniciando evolución de hiperparámetros...")
    print("   (Esto tomará unos minutos dependiendo de la cantidad de trials)...")
    
    # Ajusta n_trials según tu tiempo. 
    # 20 trials * 5 preguntas = 100 llamadas al LLM.
    study.optimize(objective, n_trials=20) 

    print("\n🏆 === FRONTERA DE PARETO (MEJORES CONFIGURACIONES) ===")
    print(f"{'ID':<4} | {'Calidad':<8} | {'Latencia':<8} | {'Configuración'}")
    print("-" * 60)
    
    for trial in study.best_trials:
        calidad = trial.values[0]
        latencia = trial.values[1]
        params = trial.params
        print(f"{trial.number:<4} | {calidad:.2f}     | {latencia:.2f}s    | {params}")