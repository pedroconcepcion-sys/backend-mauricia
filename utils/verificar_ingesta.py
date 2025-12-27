import os
import re

# Configuración
CARPETA_DATA = "data"
NOMBRE_ARCHIVO = "usach_doct_Inform_data.md" # Asegúrate que el nombre sea correcto

def estandarizar_dinero(texto):
    """La misma función exacta que usamos en tu script principal"""
    patron = r'(?i)(.*(?:arancel|matr[ií]cula|costo|valor).*?)\n\s*(\$.*)'
    texto_arreglado = re.sub(patron, r'\1: \2', texto)
    return texto_arreglado

def auditar():
    ruta = os.path.join(CARPETA_DATA, NOMBRE_ARCHIVO)
    
    if not os.path.exists(ruta):
        print("❌ No encuentro el archivo para auditar.")
        return

    print("--- 1. LEYENDO ARCHIVO ORIGINAL ---")
    with open(ruta, "r", encoding="utf-8") as f:
        texto_raw = f.read()
    
    # Buscamos dónde está el arancel en el original (para comparar)
    # Buscamos "3.8" porque sabemos que es parte del precio
    indice = texto_raw.find("3.8")
    if indice != -1:
        print(f"ORIGINAL (alrededor del precio):\n{texto_raw[indice-50:indice+50]}")
        print("------------------------------------------------")
    else:
        print("⚠️ No encontré '3.8' en el original. Quizás el precio es distinto.")

    print("\n--- 2. APLICANDO MAGIA PYTHON ---")
    texto_procesado = estandarizar_dinero(texto_raw)
    
    print("\n--- 3. RESULTADO FINAL (Lo que verá la IA) ---")
    indice_nuevo = texto_procesado.find("3.8")
    if indice_nuevo != -1:
        fragmento = texto_procesado[indice_nuevo-50:indice_nuevo+50]
        print(f"PROCESADO:\n{fragmento}")
        
        # VERIFICACIÓN AUTOMÁTICA
        if "ARANCEL: $" in fragmento or "ARANCEL : $" in fragmento or "Arancel: $" in fragmento:
            print("\n✅ ÉXITO: El título y el precio están en la misma línea.")
        else:
            print("\n❌ FALLO: Siguen separados o el regex no coincidió.")
            print(f"Debug Regex: Revisa si tu archivo original tiene saltos de línea extraños.")

if __name__ == "__main__":
    auditar()