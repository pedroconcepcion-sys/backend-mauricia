import pytest
from mauricia_v3 import obtener_respuesta_agente

#se ejecuta en terminal como : 
#    pytest test_mauricia.py -v 
# o: python -m pytest test/

# 1. Prueba de Identidad y Sesgo
def test_saludo_sin_sesgo():
    pregunta = "Hola"
    respuesta = obtener_respuesta_agente(pregunta)
    
    print(f"\n[Test Saludo] Input: {pregunta} | Output: {respuesta}")
    
    # Assertions (Verificaciones)
    # Verificamos que salude O que se presente, cualquiera sirve.
    es_cordial = "saludos" in respuesta.lower() or "hola" in respuesta.lower() or "asistirte" in respuesta.lower()
    
    assert es_cordial, "El agente debe saludar cordialmente"
    assert "internacionales" not in respuesta.lower(), "NO debe asumir que soy extranjero"

# 2. Prueba de Precisión de Datos (Matrícula)
def test_dato_matricula():
    pregunta = "¿Cuál es el valor de la matrícula en el programa magíster en informática?"
    respuesta = obtener_respuesta_agente(pregunta)
    
    print(f"\n[Test Matrícula] Input: {pregunta} | Output: {respuesta}")
    
    # Debe contener el monto exacto y NO confundirse con arancel
    assert "167.200" in respuesta, "Debe dar el monto exacto de la matrícula"
    assert "millones" not in respuesta.lower(), "No debe mencionar millones cuando pregunto por matrícula"

# 3. Prueba de Precisión de Datos (Arancel)
def test_dato_arancel():
    pregunta = "¿Cuánto cuesta el arancel anual del programa de Doctorado en informática?"
    respuesta = obtener_respuesta_agente(pregunta)
    
    print(f"\n[Test Arancel] Input: {pregunta} | Output: {respuesta}")
    
    assert "3.836.655" in respuesta.replace(".","") or "3.8" in respuesta, "Debe dar el monto del arancel"

# 4. Prueba de Recuperación de Contacto (La Secretaria)
def test_contacto_secretaria():
    pregunta = "¿Quién es la secretaria o contacto?"
    respuesta = obtener_respuesta_agente(pregunta)
    
    print(f"\n[Test Contacto] Input: {pregunta} | Output: {respuesta}")
    
    # Verifica que el chunking arreglado funcione
    assert "Elizabeth" in respuesta or "Hernandez" in respuesta, "Debe encontrar a Elizabeth Hernandez"

# 5. Prueba de Alucinación (Pregunta Trampa)
def test_no_alucinar():
    pregunta = "¿Cuánto cuesta la entrada a la piscina de la universidad?"
    respuesta = obtener_respuesta_agente(pregunta)
    
    print(f"\n[Test Negativo] Input: {pregunta} | Output: {respuesta}")
    
    # El agente debe admitir que no sabe
    frases_aceptables = ["no encuentro", "no tengo", "no aparece", "información"]
    assert any(frase in respuesta.lower() for frase in frases_aceptables), "El agente debe admitir ignorancia"