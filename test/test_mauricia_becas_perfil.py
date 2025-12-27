import pytest
from mauricia_v3 import obtener_respuesta_agente

# --- TEST 1: IDENTIDAD DEL PROGRAMA ---


@pytest.mark.parametrize("pregunta", [
    "¿Cuál es el nombre exacto del programa del Doctorado en Informática?",
    "¿De qué doctorado estamos hablando?",
    "Dime el nombre completo del postgrado Doctorado en Informática"
])
def test_nombre_programa(pregunta):
    respuesta = obtener_respuesta_agente(pregunta)
    print(f"\n[Input]: {pregunta} \n[Output]: {respuesta}")

    respuesta_lower = respuesta.lower()

    # Validamos las 3 partes clave del nombre oficial:
    # "Doctorado en Ciencias de la Ingeniería con Mención en Informática"
    tiene_grado = "doctorado" in respuesta_lower
    tiene_area = "informática" in respuesta_lower or "informatica" in respuesta_lower
    tiene_facultad = "ingeniería" in respuesta_lower or "ingenieria" in respuesta_lower

    if not (tiene_grado and tiene_area and tiene_facultad):
        pytest.fail(
            f"🚨 Identidad incompleta. Se esperaba Doctorado + Informática + Ingeniería. Respuesta: {respuesta}")

    assert True

# --- TEST 2: PERFIL DE INGRESO ---


def test_publico_objetivo():
    pregunta = "¿Qué grado académico o título se requiere para postular al Doctorado?"
    respuesta = obtener_respuesta_agente(pregunta)
    print(f"\n[Input]: {pregunta} \n[Output]: {respuesta}")

    respuesta_lower = respuesta.lower()

    # Palabras clave extraídas del sitio: "Licenciado en Ciencias de la Ingeniería... Magíster..."
    palabras_clave = [
        "licenciado", "licenciatura",
        "magíster", "magister",
        "título profesional",
        "ciencias de la ingeniería",
        "ingeniería aplicada"
    ]

    match = any(p in respuesta_lower for p in palabras_clave)

    assert match, f"🚨 No mencionó los grados académicos requeridos (Licenciado/Magíster). Respuesta: {respuesta}"

# --- TEST 3: BECAS Y BENEFICIOS ---
# Validamos "Beca de Arancel y Mantención" (Ojo: Mantención sin 'u' como en el sitio)


def test_becas_disponibles():
    # Dejamos la pregunta ambigua como tú quieres
    pregunta = "¿Qué becas ofrece el programa?"
    respuesta = obtener_respuesta_agente(pregunta)
    print(f"\n[Input]: {pregunta} \n[Output]: {respuesta}")

    respuesta_lower = respuesta.lower()

    # 1. Criterios del DOCTORADO (Lo que ya tenías)
    tiene_arancel = "arancel" in respuesta_lower
    tiene_manutencion = "manutención" in respuesta_lower or "mantención" in respuesta_lower
    tiene_otras_becas = "excelencia" in respuesta_lower or "investigación" in respuesta_lower or "anid" in respuesta_lower

    # 2. Criterio del MAGÍSTER (NUEVO)
    # El Magíster habla mucho de "descuento sobre el valor total" o "rebaja"
    tiene_descuento = "descuento" in respuesta_lower or "rebaja" in respuesta_lower

    # 3. ÉXITO HÍBRIDO
    # Pasa si cumple lo del Doctorado O si cumple lo del Magíster
    exito = (
        tiene_arancel and tiene_manutencion) or tiene_otras_becas or tiene_descuento

    assert exito, f"🚨 No se encontraron becas ni descuentos. Respuesta: {respuesta}"

# --- TEST 4: DESCUENTO EX-ALUMNOS (ACTIVADO) ---
# Dato esperado: "50% de Descuento para egresados/as..."


def test_beneficio_ex_alumnos():
    pregunta = "Soy ex alumno de la USACH, ¿tengo algún descuento?"
    respuesta = obtener_respuesta_agente(pregunta)
    print(f"\n[Input]: {pregunta} \n[Output]: {respuesta}")

    respuesta_lower = respuesta.lower()

    # 1. Buscamos el porcentaje
    tiene_descuento = "50" in respuesta or "50%" in respuesta

    # 2. Buscamos al beneficiario (egresado, graduado, alumno)
    menciona_beneficiarios = "egresad" in respuesta_lower or "graduad" in respuesta_lower or "ex alumno" in respuesta_lower or "ex-alumno" in respuesta_lower or "alumn" in respuesta_lower

    assert tiene_descuento and menciona_beneficiarios, f"🚨 No encontró el descuento del 50% para ex-alumnos. Respuesta: {respuesta}"

# --- TEST 5: ALUCINACIÓN (CONTROL NEGATIVO) ---


def test_beca_fake():
    pregunta = "¿Puedo usar la Beca Junaeb Sodexo o BAES en este doctorado?"
    respuesta = obtener_respuesta_agente(pregunta)
    print(f"\n[Input]: {pregunta} \n[Output]: {respuesta}")

    # La respuesta debe ser negativa
    frases_negativas = ["no", "no aplica", "no encuentro",
                        "no se menciona", "sólo académica", "solo manejo información"]

    negativa = any(x in respuesta.lower() for x in frases_negativas)

    assert negativa, f"🚨 ALERTA: El agente alucinó aceptando la Junaeb. Respuesta: {respuesta}"
