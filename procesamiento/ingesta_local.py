import asyncio
import os
import sys
from crawl4ai import AsyncWebCrawler

async def main():
    print("\n🕵️  === INGESTADOR DE PROGRAMAS USACH ===")
    
    # 1. PEDIR DATOS AL USUARIO
    url_objetivo = input("🌐 Pega la URL del programa aquí: ").strip()
    if not url_objetivo:
        print("❌ Error: Debes ingresar una URL.")
        return

    nombre_input = input("📄 Nombre para guardar el archivo (ej: magister_informatica): ").strip()
    if not nombre_input:
        print("❌ Error: Debes dar un nombre al archivo.")
        return
    
    # Aseguramos que termine en .md
    if not nombre_input.endswith(".md"):
        nombre_input += ".md"

    print(f"\n🚀 Iniciando extracción en: {url_objetivo}...")
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url_objetivo,
            word_count_threshold=10, 
            bypass_cache=True
        )

        if result.success:
            print("\n✅ Extracción exitosa!")
            
            carpeta = "data"
            ruta_completa = os.path.join(carpeta, nombre_input)
            
            if not os.path.exists(carpeta):
                os.makedirs(carpeta)
                print(f"📁 Carpeta '{carpeta}' creada.")

            with open(ruta_completa, 'w', encoding='utf-8') as f:
                f.write(result.markdown)
            
            print(f"💾 Información guardada en: {ruta_completa}")
            print(f"🎉 ¡Listo! Ahora tienes '{nombre_input}' junto a los otros archivos.")
            
        else:
            print(f"❌ Error al extraer: {result.error_message}")

if __name__ == "__main__":
    # CORRECCIÓN PARA WINDOWS + PLAYWRIGHT
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())