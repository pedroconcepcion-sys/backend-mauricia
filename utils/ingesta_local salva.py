import asyncio
import os
from crawl4ai import AsyncWebCrawler

# URL objetivo
URL_OBJETIVO = "https://www.postgradosudesantiago.cl/doctorado-en-ciencias-de-la-ingenieria-con-mencion-en-informatica/"

async def main():
    print(f"🕵️  Iniciando agente de extracción en: {URL_OBJETIVO}")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=URL_OBJETIVO,
            word_count_threshold=10, 
            bypass_cache=True
        )

        if result.success:
            print("\n✅ Extracción exitosa!")
            
            # --- CORRECCIÓN AQUÍ ---
            carpeta = "data"
            nombre_archivo = "usach_doct_Inform_data.md"
            
            # Unimos carpeta + nombre para crear la ruta completa
            # Esto crea algo como: "data/usach_doct_Inform_data.md"
            ruta_completa = os.path.join(carpeta, nombre_archivo)
            
            # Aseguramos que la carpeta exista
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)
                print(f"📁 Carpeta '{carpeta}' creada.")

            # USAMOS ruta_completa EN LUGAR DE nombre_archivo
            with open(ruta_completa, 'w', encoding='utf-8') as f:
                f.write(result.markdown)
            
            print(f"💾 Información guardada correctamente en: {os.path.abspath(ruta_completa)}")
            
            print("\n--- VISTA PREVIA DEL CONTENIDO (MARKDOWN) ---")
            print(result.markdown[:500])
            print("---------------------------------------------")
            
        else:
            print(f"❌ Error al extraer: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())