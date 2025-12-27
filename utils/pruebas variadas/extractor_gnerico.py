import json
import re
from typing import Any, Optional, Dict

from langchain_core.messages import HumanMessage


def is_extractive_question(q: str) -> bool:
    """Heurística genérica: preguntas que suelen requerir extracción literal."""
    q = q.lower()
    triggers = [
        "cuánto", "cuanto", "precio", "valor", "costo", "arancel", "matrícula", "matricula",
        "fecha", "cuándo", "cuando", "plazo", "postulación", "postulacion",
        "requisito", "documento", "duración", "duracion", "modalidad",
        "contacto", "correo", "email", "teléfono", "telefono", "horario",
        "acreditación", "acreditacion","director"
    ]
    return any(t in q for t in triggers)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extrae el primer JSON válido desde el texto."""
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1].strip()

    # Normaliza comillas “raras”
    candidate = candidate.replace("“", "\"").replace("”", "\"")

    try:
        return json.loads(candidate)
    except Exception:
        return None


def extract_fields_generic(llm: Any, context: str, question: str) -> Optional[Dict[str, Any]]:
    """
    Usa el LLM como extractor (modo controlado) y devuelve dict (JSON).
    Si no logra JSON, devuelve None.
    """
    extractor_prompt = f"""
Eres un extractor de información. SOLO puedes usar el CONTEXTO.
Devuelve ÚNICAMENTE un JSON válido (sin explicación, sin markdown).

Reglas:
- No inventes ni deduzcas. No calcules.
- Si un dato no está explícito en el CONTEXTO, usa null.
- Extrae valores EXACTOS tal como aparecen (montos, periodicidad, fechas, emails, links).
- Si hay MATRÍCULA y ARANCEL, NO los mezcles.

JSON schema:
{{
  "found": true/false,
  "items": [
    {{
      "field": "string (ej: arancel, matricula, duracion, modalidad, requisitos, contacto, fechas_postulacion, etc.)",
      "value": "string|null",
      "unit_or_period": "string|null (ej: anual, semestral, CLP, etc.)",
      "evidence": "string|null (fragmento literal del contexto donde aparece)"
    }}
  ],
  "missing_question": "string|null (UNA pregunta si falta especificación, si aplica)"
}}

CONTEXTO:
{context}

PREGUNTA:
{question}
""".strip()

    resp = llm.invoke([HumanMessage(content=extractor_prompt)])
    return extract_json(getattr(resp, "content", "") or "")
