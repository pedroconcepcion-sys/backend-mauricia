# dataset.py
# Conjunto de datos de evaluación para optimización con Optuna.
# IMPORTANTE: Asegúrate de que las palabras en "debe_contener" realmente existan en tus PDFs.

CASOS_PRUEBA = [
    # =========================================
    # 💰 CATEGORÍA 1: DINERO (K_DINERO)
    # Buscamos precisión numérica y diferenciación.
    # =========================================
    {
        "pregunta": "¿Cuál es el arancel anual del Doctorado en Informática?",
        "tipo": "dinero",
        "debe_contener": ["arancel", "millones", "anual"], 
        "peso": 1.0
    },
        {
        "pregunta": "¿Cuál es el arancel anual del Magíster en Informática?",
        "tipo": "dinero",
        "debe_contener": ["arancel", "millones", "anual"],
        "peso": 1.0
    },
    {
        "pregunta": "¿Cuál es el valor total del Magíster en Ingeniería Informática?",
        "tipo": "dinero",
        "debe_contener": ["arancel", "millones", "total"], 
        "peso": 1.0
    },
    {
        "pregunta": "¿Cuánto cuesta la matrícula semestral para los postgrados?",
        "tipo": "dinero",
        "debe_contener": ["matrícula", "167", "semestral"], # El valor aprox de matrícula
        "peso": 1.0
    },
    {
        "pregunta": "¿Existen descuentos o rebajas para ex-alumnos de la USACH?",
        "tipo": "dinero",
        "debe_contener": ["descuento", "egresados/as", "50%"], 
        "peso": 0.8
    },

    # =========================================
    # 📚 CATEGORÍA 2: ACADÉMICO Y REQUISITOS (K_NORMAL)
    # Buscamos contexto amplio y listas.
    # =========================================
    {
        "pregunta": "¿Cuáles son los requisitos para postular a un Doctorado?",
        "tipo": "normal",
        "debe_contener": ["grado", "magíster", "licenciado", "Curriculum", "certificado"],
        "peso": 1.0
    },
    {
        "pregunta": "¿Cuánto dura el Magíster en Informática?",
        "tipo": "normal",
        "debe_contener": ["semestres", "8", "ocho"],
        "peso": 0.8
    },
    {
        "pregunta": "¿Qué líneas de investigación tiene el Doctorado de Informática?",
        "tipo": "normal",
        "debe_contener": ["Biología", "Web", "Sistemas", "Complejos"],
        "peso": 1.0
    },
    {
        "pregunta": "¿Cuál es la modalidad del magister en informática?",
        "tipo": "normal",
        "debe_contener": ["presencial", "presencial"],
        "peso": 0.7
    },

    # =========================================
    # 🎓 CATEGORÍA 3: BECAS Y BENEFICIOS (K_NORMAL)
    # Suele requerir leer secciones específicas.
    # =========================================
    {
        "pregunta": "¿Qué becas internas ofrece la universidad?",
        "tipo": "normal",
        "debe_contener": ["beca", "arancel", "mantención", "Apoyo", "investigación"],
        "peso": 1.0
    },
    {
        "pregunta": "¿Se puede postular a becas ANID?",
        "tipo": "normal",
        "debe_contener": ["ANID", "acreditados", "participar"],
        "peso": 0.9
    },

    # =========================================
    # 📧 CATEGORÍA 4: CONTACTO Y ADMIN (K_NORMAL)
    # Prueba si llega al final del documento (footer).
    # =========================================
    {
        "pregunta": "¿Cuál es el correo de contacto para consultas del Magíster?",
        "tipo": "normal",
        "debe_contener": ["@", "usach.cl", "correo", "email"],
        "peso": 1.0
    },
    {
        "pregunta": "¿Quién es el director o coordinador del programa de magíster en informática?",
        "tipo": "normal",
        "debe_contener": ["director", "inoztroza", "dr", "mario"], # Si sabes el nombre, ponlo aquí
        "peso": 0.8
    },
]