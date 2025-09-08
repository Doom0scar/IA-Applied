# gemini_eval_demo.py
"""
==============================================================
Demo UNIVERSAL (enriquecido): Generar y EVALUAR con Gemini
==============================================================
Este archivo integra en un solo script:
- Generación (flash) y evaluación (pro + fallbacks) con guardrails.
- Política "Unknown" (doc-only) y evidencia para capear 5★.
- Estructura de salida del juez (JSON) con señales ricas.
- Comparación pairwise RAW vs GUIDED (A/B).
- Modo batch (stub local o Vertex si tienes runner real).

IMPORTANTE:
- Si tienes módulos reales como `vertex_batch_runner`, `schemas`,
  `evidence_policy`, `pairwise` en tus repos, el script intentará importarlos.
  Si no están, usa fallbacks definidos aquí mismo con las MISMAS firmas públicas.
"""

# ============ 0) CONFIG DE MODELOS (roles) ============
GENERATOR_MODEL = "gemini-2.0-flash"  # Generador rápido/eco
JUDGE_MODELS    = ["gemini-2.5-pro"]  # Juez preferido
JUDGE_FALLBACKS = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro",
    "gemini-2.0-flash",               # Último recurso
    "gemini-1.5-flash-latest",
]

# ============ 1) IMPORTS ============
import os, re, sys, json, enum, string, functools, urllib.request
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.api_core import retry

# ============ 2) AUTENTICACIÓN ============
def _load_api_key() -> str:
    try:
        with open("YEK.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return os.environ.get("GOOGLE_API_KEY", "")

API_KEY = _load_api_key()
if not API_KEY:
    print("ERROR: No encontré tu API key. Ponla en YEK.txt o en GOOGLE_API_KEY.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# ============ 3) REINTENTOS 429/503 ============
is_retriable = lambda e: (isinstance(e, genai_errors.APIError) and getattr(e, "code", None) in {429, 503})
if not hasattr(genai.models.Models.generate_content, "__wrapped__"):
    genai.models.Models.generate_content = retry.Retry(
        predicate=is_retriable, deadline=60.0, initial=1.0, maximum=10.0
    )(genai.models.Models.generate_content)

# ============ 4) PDF: descarga → upload ============
PDF_URL   = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
LOCAL_PDF = "gemini.pdf"
if not os.path.exists(LOCAL_PDF):
    print("Descargando PDF...")
    urllib.request.urlretrieve(PDF_URL, LOCAL_PDF)

print("Subiendo PDF a Gemini Files...")
document_file = client.files.upload(file=LOCAL_PDF)

# =========================================================
# 5) FALLBACKS de MÓDULOS (si no están instalados/importables)
# =========================================================
# 5.1 schemas.qa_struct_schema()
try:
    from schemas import qa_struct_schema  # pragma: no cover
except Exception:
    def qa_struct_schema() -> Dict[str, Any]:
        """Esquema JSON para la respuesta del juez en QA/Resumen."""
        # Mantén nombres simples para robustez cross-model
        return {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "evidence_found": {"type": "boolean"},
                "evidence_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["citation", "numeric", "artifact", "baseline"]},
                },
                "justification": {"type": "string"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["score"],
            "additionalProperties": False,
        }

# 5.2 evidence_policy.enforce_cap_5star()
try:
    from evidence_policy import enforce_cap_5star  # pragma: no cover
except Exception:
    def _has_citation(txt: str) -> bool:
        return bool(re.search(r'(?i)\b(section|page|table|figure)\s+\d+', txt)) or bool(re.search(r'\(.*(table|figure)\s*\d+.*\)', txt, flags=re.I))
    def _has_numeric(txt: str) -> bool:
        return bool(re.search(r'\b\d+(\.\d+)?\b', txt)) or "%" in txt
    def _has_artifact(txt: str) -> bool:
        # dataset/benchmark/fig/table/acrónimos en mayúsculas
        return bool(re.search(r'\b(HumanEval|Natural2Code|MMLU|GSM8K|BIG-bench|Table\s*\d+|Figure\s*\d+)\b', txt, re.I)) \
               or bool(re.search(r'\b[A-Z]{3,}\b', txt))
    def _has_baseline(txt: str) -> bool:
        return bool(re.search(r'\b(vs\.?|compared to|relative to|over|surpass|outperform(s)?)\b', txt, re.I))

    def _evidence_types_from_text(*texts: str) -> List[str]:
        joined = " ".join(t for t in texts if t)
        kinds = []
        if _has_citation(joined): kinds.append("citation")
        if _has_numeric(joined):  kinds.append("numeric")
        if _has_artifact(joined): kinds.append("artifact")
        if _has_baseline(joined): kinds.append("baseline")
        return sorted(set(kinds))

    def enforce_cap_5star(
        rating, answer_text: str, judge_text: str, min_types_for_5: int = 2, cap_value: int = 4
    ):
        """
        Si la calificación es 5 pero NO hay al menos `min_types_for_5` tipos de evidencia,
        baja a `cap_value`. Evidencia: citation, numeric, artifact, baseline.
        """
        try:
            v = int(rating.value)
        except Exception:
            return rating
        if v < 5:
            return rating
        types_found = _evidence_types_from_text(answer_text, judge_text)
        if len(types_found) < min_types_for_5:
            v = min(v, cap_value)
            return rating.__class__(str(v))
        return rating

# 5.3 pairwise.eval_pairwise()
try:
    from pairwise import eval_pairwise  # pragma: no cover
except Exception:
    def eval_pairwise(prompt: str, response_a: str, response_b: str, judge_model: Optional[str] = None) -> Tuple[str, enum.Enum]:
        """Comparación A/B simple con el mismo juez."""
        _PAIRWISE_PROMPT = f"""
# Instrucción
Eres un evaluador experto. Elige qué respuesta responde MEJOR la pregunta del usuario usando SOLO el PDF adjunto como única fuente de verdad. Ignora conocimientos previos y no inventes.

# Criterios con prioridad (en este orden)
1) **Grounding y corrección**: toda afirmación material debe estar respaldada por el PDF. Prefiere respuestas con **citas o citas breves** ligadas a **Sección/Página/Tabla/Figura**. Penaliza afirmaciones sin evidencia.
2) **Métricas y baselines**: cuando haya métricas/porcentajes/puntajes, el **baseline comparativo** (p. ej., “comparado con X”) debe ser **explícito** y no mezclado.
3) **Relevancia (on-topic)**: limitarse estrictamente a la **pregunta**; excluir dominios o detalles no solicitados.
4) **Especificidad/Evidencia**: preferir números, benchmarks/datasets/artefactos con nombre, y baselines explícitos.
5) **No redundancia y claridad**: conciso pero **completo**; sin repetir la misma idea ni relleno.

# Descalificadores
- Si una respuesta contiene **afirmaciones materiales no respaldadas** por el PDF y la otra no, **prefiere la respaldada**, aunque sea más breve.
- Si **ambas** tienen afirmaciones no respaldadas, prefiere la que tenga **menos** y **menos graves**.

# Reglas de desempate (MUY IMPORTANTES)
- Si una respuesta incluye **citas/quotes correctas** y la otra no, **prefiere la que cita**, salvo que las citas sean materialmente erróneas.
- Si ambas citan, prefiere (a) la que tenga **más citas correctas** y (b) **más baselines explícitos** por cada métrica mencionada.
- Si siguen muy parejas, prefiere la que esté **más estrictamente on-topic** y sea **menos redundante**.
- Si realmente son equivalentes en todo, elige **"SAME"**.

# Tarea
Pregunta:
{prompt}

Respuesta A:
{response_a}

Respuesta B:
{response_b}

Primero proporciona un **análisis breve** de **ambas** respuestas mencionando explícitamente: grounding, relevancia y evidencia (citas/métricas/baselines). 
Luego, en una línea final separada, imprime **EXACTAMENTE** uno de estos:
Winner: A
Winner: B
Winner: SAME
""".strip()
        jm = judge_model or (JUDGE_MODELS + JUDGE_FALLBACKS)[0]
        cfg = types.GenerateContentConfig(temperature=0.0)
        rsp = client.models.generate_content(model=jm, config=cfg, contents=[_PAIRWISE_PROMPT, document_file])
        text = (rsp.text or "").strip()
        choice = "SAME"
        if re.search(r'(?i)\bwinner\s*[:\-]\s*A\b', text): choice = "A"
        elif re.search(r'(?i)\bwinner\s*[:\-]\s*B\b', text): choice = "B"
        # Enum mínima para consistencia
        class AnswerComparison(enum.Enum):
            A = "A"; SAME = "SAME"; B = "B"
        return text, AnswerComparison(choice)

# 5.4 vertex_batch_runner.submit_eval_batch()
try:
    from vertex_batch_runner import submit_eval_batch  # pragma: no cover
except Exception:
    def submit_eval_batch(project: str, location: str, items: List[Dict[str, str]], gcs_dest: str) -> str:
        """
        Fallback local: procesa items sin subir a Vertex. Devuelve un "job id".
        Cada item debe tener: { "pdf": path|uri, "summary_prompt": str, "qa_question": str }
        """
        print("[WARN] Usando stub local de submit_eval_batch (no Vertex).")
        # procesar local: generar respuestas con el generador actual
        outputs = []
        for it in items:
            req = it.get("summary_prompt", "")
            q   = it.get("qa_question", "")
            # Nota: ignoro it["pdf"] y uso document_file ya subido arriba para demo
            s = summarise_doc(req)
            a = answer_question_guided(q)  # guiado para mayor calidad en batch
            outputs.append({"summary_prompt": req, "summary": s, "qa_question": q, "answer": a})
        # guardar a disco como si fuese GCS
        os.makedirs(gcs_dest, exist_ok=True)
        out_path = os.path.join(gcs_dest, "batch_results.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return f"local-batch://{out_path}"

# =========================================================
# 6) PERFILES / QA focus / prompts del generador
# =========================================================
@dataclass
class CoverageAxis:
    name: str
    keywords: List[str]

@dataclass
class TaskProfile:
    name: str
    force_focus_from_question: bool = True
    focus_terms: List[str] = field(default_factory=list)
    coverage_axes: List[CoverageAxis] = field(default_factory=list)
    min_axes_for_ok: int = 0
    detail_keywords: List[str] = field(default_factory=list)
    require_citation_cap: int = 0
    penalize_vagueness: bool = True
    vague_phrases: List[str] = field(default_factory=lambda: [
        "a variety of", "variety of", "multiple", "various", "a host of",
        "across many different", "numerous", "several", "wide range of",
    ])
    strictness: int = 2

GENERIC_PROFILE = TaskProfile(name="generic")

def _pick_profile() -> TaskProfile:
    val = os.environ.get("EVAL_PROFILE", "").strip().lower()
    if val in {"ml_training_paper", "ml"}:
        return TaskProfile(
            name="ml_training_paper",
            coverage_axes=[
                CoverageAxis("data", ["multimodal","multilingual","web","code","image","audio","video"]),
                CoverageAxis("architecture", ["moe","mixture-of-expert","mixture of expert","transformer"]),
                CoverageAxis("infra", ["tpu","tpuv4","pathways"]),
                CoverageAxis("tuning", ["instruction","fine-tun","rlhf","human preference"]),
            ],
            min_axes_for_ok=3,
            detail_keywords=["tpuv4","pathways","moe","rlhf","jax"],
            require_citation_cap=0,
        )
    return GENERIC_PROFILE

CURRENT_PROFILE = _pick_profile()

# ====== Prompts del generador ======
# Resumen (sigue siendo conciso, con detalles y citas si hay)
GEN_SUMMARY_STYLE = (
    "Use ONLY the attached document. Produce a concise bullet list on the requested aspect. "
    "Avoid meta phrases like 'Based on the document you provided'. "
    "Include concrete details (numbers, acronyms, or proper names) and cite Section/Page/Table when available."
)

# QA policy alineada con la rúbrica del juez, sin limitar longitud a 1 frase:
QA_POLICY_ALIGNED = (
    "Use ONLY the provided document as the single source of truth. "
    "If the information needed to answer the question is NOT explicitly present, reply exactly 'Unknown'. "
    "Stay strictly on the user's question; do not drift. "
    "Prefer explicit evidence: include a short quote and/or cite Section/Page/Table/Figure when available. "
    "When you provide numbers or metrics, make baselines explicit (e.g., 'compared to X'). Do not conflate metrics. "
    "Prefer concrete details (metrics, named benchmarks/datasets/artifacts, and explicit baselines) over vague wording. "
    "Be concise but complete: include all directly relevant facts without repeating the same idea. "
    "Output only the answer (no preambles)."
)

# ====== Utilidades de QA (auto-focus) ======
_STOPWORDS = set("""
a an the and or for in on at to of with without by from about into over under again further
how what why where when which who whose is are was were be been being do does did doing
""".split())

def extract_focus_terms_from_question(q: str, k: int = 8) -> List[str]:
    q = q.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in q.split() if t not in _STOPWORDS and len(t) > 2]
    seen, terms = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t); terms.append(t)
    return terms[:k]

def make_dynamic_qa_guidance(question: str) -> Tuple[str, List[str]]:
    focus_terms = CURRENT_PROFILE.focus_terms[:]
    if CURRENT_PROFILE.force_focus_from_question:
        derived = extract_focus_terms_from_question(question)
        if not focus_terms:
            focus_terms = derived
    guidance = QA_POLICY_ALIGNED
    if focus_terms:
        guidance += " Focus ONLY on these aspects: " + ", ".join(focus_terms) + "."
    return guidance, focus_terms

# =========================================================
# 7) GENERADOR (Resumen / QA)
# =========================================================
def summarise_doc(request: str) -> str:
    cfg = types.GenerateContentConfig(temperature=0.0, system_instruction=GEN_SUMMARY_STYLE)
    rsp = client.models.generate_content(model=GENERATOR_MODEL, config=cfg, contents=[request, document_file])
    return (rsp.text or "").strip()

@functools.cache
def answer_question_guided(question: str) -> str:
    guidance, focus_terms = make_dynamic_qa_guidance(question)
    cfg = types.GenerateContentConfig(temperature=0.0, system_instruction=guidance)
    rsp = client.models.generate_content(model=GENERATOR_MODEL, config=cfg, contents=[question, document_file])
    ans = (rsp.text or "").strip()

    # autocorrección simple: si no toca ningún término foco y no es Unknown, reintenta
    if focus_terms and ans != "Unknown":
        low = ans.lower()
        if not any(t in low for t in focus_terms):
            cfg2 = types.GenerateContentConfig(temperature=0.0, system_instruction=guidance + " Keep ONLY content directly tied to the focus terms.")
            rsp2 = client.models.generate_content(model=GENERATOR_MODEL, config=cfg2, contents=[question, document_file])
            ans = (rsp2.text or "").strip()
    return ans

@functools.cache
def answer_question_raw(question: str) -> str:
    """Mismo modelo, sin system_instruction: para la Suite RAW."""
    cfg = types.GenerateContentConfig(temperature=0.0)
    rsp = client.models.generate_content(model=GENERATOR_MODEL, config=cfg, contents=[question, document_file])
    return (rsp.text or "").strip()

# =========================================================
# 8) JUEZ (prompts y esquema estructurado)
# =========================================================
SUMMARY_PROMPT = """
# Instruction
You are an expert evaluator. Evaluate the quality of the summary using ONLY the attached PDF.

Judge STRICTLY on:
- Grounding: cites Section/Page/Table/Figure when possible.
- Specificity/Evidence: uses concrete details (numbers, acronyms, named artifacts, explicit baselines).
- Coverage: covers the key aspects relevant to the request (if the topic is 'training process', consider data/architecture/infra/tuning).
- Clarity: concise bullets, no filler or redundancy.

If any of the above is missing, do NOT award 5/5.
Provide a short critique and an overall score 1..5.
""".strip()

QA_PROMPT = """
# Instruction
You are an expert evaluator. Evaluate the answer using ONLY the attached PDF.

Judge STRICTLY on:
- Grounding: do not credit content not supported by the document; prefer short quotes/citations when available.
- Relevance: stays strictly on the user's question; no drift.
- Precision: do not conflate metrics; make baselines explicit.
- Specificity/Evidence: numbers, named artifacts (benchmarks/datasets/tables/figures), and explicit baselines.
- Non-redundancy: concise but complete; avoid repeating the same idea.

If the document does not contain the information and the model replied exactly 'Unknown', treat this as correct behavior (do not penalize).
If any of the above is missing, do NOT award 5/5.
Provide a short critique and an overall score 1..5.
""".strip()

class SummaryRating(enum.Enum):
    VERY_GOOD = "5"; GOOD = "4"; OK = "3"; BAD = "2"; VERY_BAD = "1"

class AnswerRating(enum.Enum):
    VERY_GOOD = "5"; GOOD = "4"; OK = "3"; BAD = "2"; VERY_BAD = "1"

def _extract_score_from_text(text: str) -> Optional[str]:
    if not text: return None
    pats = [
        r'(?i)\boverall\s*score\s*[:\-]?\s*([1-5])\s*/\s*5\b',
        r'(?i)\bscore\s*[:\-]?\s*([1-5])\s*/\s*5\b',
        r'(?i)\boverall\s*score\s*[:\-]?\s*([1-5])\b',
        r'(?i)\bscore\s*[:\-]?\s*([1-5])\b',
        r'(?<!\d)([1-5])\s*/\s*5\b',
    ]
    for p in pats:
        m = re.search(p, text)
        if m: return m.group(1)
    return None

def _judge_once(base_prompt: str, user_prompt: str, ai_response: str, judge_model: str) -> Tuple[str, Dict[str, Any]]:
    """Una pasada de juez: narrativa + JSON estructurado (schema) con score/evidencia/…"""
    cfg_text = types.GenerateContentConfig(temperature=0.0)
    contents = [base_prompt, "### Prompt\n" + user_prompt, "## AI-generated Response\n" + ai_response, document_file]
    rsp = client.models.generate_content(model=judge_model, config=cfg_text, contents=contents)
    narrative = (rsp.text or "").strip()

    # 2º turno: objeto JSON rico
    cfg_json = types.GenerateContentConfig(response_mime_type="application/json", response_schema=qa_struct_schema())
    json_instruction = "Output ONLY a strict JSON with fields: score, evidence_found, evidence_types, justification, issues."
    try:
        rsp2 = client.models.generate_content(model=judge_model, config=cfg_json, contents=contents + [json_instruction])
        obj = rsp2.parsed  # dict si el modelo respeta schema
        if isinstance(obj, enum.Enum):  # por si un modelo responde enum
            obj = {"score": int(obj.value)}
    except Exception:
        # Fallback: pide JSON libre y parsea
        cfg_json2 = types.GenerateContentConfig(response_mime_type="application/json")
        rsp3 = client.models.generate_content(model=judge_model, config=cfg_json2, contents=contents + [json_instruction])
        try:
            obj = json.loads(rsp3.text or "{}")
        except Exception:
            # Último recurso: extrae score de la narrativa
            hint = _extract_score_from_text(narrative) or "3"
            obj = {"score": int(hint), "evidence_found": False, "evidence_types": [], "justification": "", "issues": []}
    return narrative, obj

def _score_lock(enum_cls, narrative: str, structured_obj: Dict[str, Any]):
    """Alinea score final si narrativa y JSON discrepan y narrativa trae número claro."""
    try:
        hint = _extract_score_from_text(narrative)
        if hint and int(structured_obj.get("score", 0)) != int(hint):
            structured_obj["score"] = int(hint)
    except Exception:
        pass
    # Devolver Enum
    val = str(structured_obj.get("score", 3))
    return enum_cls(val), structured_obj

# ====== Guardrails (resumen/qa) con cap de evidencia ======
def _adjust_rating(enum_rating, delta: int):
    v = max(1, min(5, int(enum_rating.value) + delta))
    return enum_rating.__class__(str(v))

def apply_summary_guardrails(generator_summary: str, judge_text: str, judge_rating: SummaryRating) -> Tuple[SummaryRating, List[str]]:
    notes, adj = [], 0
    # Vaguedad sin detalle/cita → -1
    def _has_number_or_percent(t): return bool(re.search(r'\b\d+(\.\d+)?\b', t)) or "%" in t
    def _has_acronym_or_proper(t): return bool(re.search(r'\b[A-Z]{2,}\b', t)) or bool(re.search(r'\b[A-Z][a-z]{2,}\b', t))
    def _has_citation(t): return bool(re.search(r'(?i)\b(section|page|table|figure)\s+\d+', t)) or bool(re.search(r'\(.*(table|figure)\s*\d+.*\)', t, flags=re.I))
    low = generator_summary.lower()
    vague_hit = any(p in low for p in CURRENT_PROFILE.vague_phrases)
    if vague_hit and not (_has_number_or_percent(generator_summary) or _has_acronym_or_proper(generator_summary) or _has_citation(generator_summary)):
        adj -= 1; notes.append("Lenguaje vago sin detalle o cita.")
    new_rating = judge_rating if adj == 0 else _adjust_rating(judge_rating, adj)
    # Cap 5★ si falta evidencia múltiple
    new_rating = enforce_cap_5star(new_rating, generator_summary, judge_text, min_types_for_5=2, cap_value=4)
    return new_rating, notes

def apply_qa_guardrails(generator_answer: str, judge_text: str, judge_rating: AnswerRating, require_evidence_for_5:int=2) -> Tuple[AnswerRating, List[str]]:
    notes, adj = [], 0
    # Métrica mezclada en un mismo paréntesis: -1
    if re.search(r'\(\s*[^)]*\+?\d+(\.\d+)?\s*%[^)]*\+?\d+(\.\d+)?\s*%[^)]*\)', generator_answer):
        adj -= 1; notes.append("Posible mezcla de métricas distintas en la misma cita.")
    new_rating = judge_rating if adj == 0 else _adjust_rating(judge_rating, adj)
    new_rating = enforce_cap_5star(new_rating, generator_answer, judge_text, min_types_for_5=require_evidence_for_5, cap_value=4)
    return new_rating, notes

# ====== Wrappers de evaluación con fallback de modelo juez ======
def eval_summary_single(prompt: str, ai_response: str, judge_model: Optional[str] = None):
    seen, candidates = set(), []
    for m in ([judge_model] if judge_model else []) + JUDGE_MODELS + JUDGE_FALLBACKS:
        if m and m not in seen:
            seen.add(m); candidates.append(m)
    last_err = None
    for jm in candidates:
        try:
            print(f"[INFO] Trying judge model: {jm}")
            narrative, obj = _judge_once(SUMMARY_PROMPT, prompt, ai_response, jm)
            enum_rating, obj = _score_lock(SummaryRating, narrative, obj)
            final_rating, notes = apply_summary_guardrails(ai_response, narrative, enum_rating)
            return narrative, enum_rating, final_rating, notes, jm, obj
        except genai_errors.ClientError as e:
            msg = str(e)
            if getattr(e, "code", None) == 404 or "NOT_FOUND" in msg or "not supported" in msg:
                print(f"[WARN] Judge model '{jm}' not available/supported. Trying next...")
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("No judge model worked for summary.")

def eval_answer_single(prompt: str, ai_response: str, judge_model: Optional[str] = None, require_evidence_for_5:int=2):
    seen, candidates = set(), []
    for m in ([judge_model] if judge_model else []) + JUDGE_MODELS + JUDGE_FALLBACKS:
        if m and m not in seen:
            seen.add(m); candidates.append(m)
    last_err = None
    for jm in candidates:
        try:
            print(f"[INFO] Trying judge model (QA): {jm}")
            narrative, obj = _judge_once(QA_PROMPT, prompt, ai_response, jm)
            enum_rating, obj = _score_lock(AnswerRating, narrative, obj)
            final_rating, notes = apply_qa_guardrails(ai_response, narrative, enum_rating, require_evidence_for_5=require_evidence_for_5)
            return narrative, enum_rating, final_rating, notes, jm, obj
        except genai_errors.ClientError as e:
            msg = str(e)
            if getattr(e, "code", None) == 404 or "NOT_FOUND" in msg or "not supported" in msg:
                print(f"[WARN] Judge model '{jm}' not available/supported. Trying next...")
                last_err = e
                continue
            raise
    raise last_err if last_err else RuntimeError("No judge model worked for QA.")

# =========================================================
# 9) SUITES: RAW y GUIDED, + pairwise
# =========================================================
def run_suite_raw():
    print("\n" + "="*80)
    print("### SUITE RAW (solo request) — MODO: RAW")
    print("="*80)
    # Resumen
    request = "Tell me about the training process used here."
    print("\n[PROMPTS · RESUMEN]\nrequest =", request, "\nsystem_instruction (summary) = (none)")
    print(f"\n=== RESUMEN (RAW (solo request): {GENERATOR_MODEL}) ===")
    cfg = types.GenerateContentConfig(temperature=0.0)
    summary_rsp = client.models.generate_content(model=GENERATOR_MODEL, config=cfg, contents=[request, document_file])
    raw_summary = (summary_rsp.text or "").strip()
    print(raw_summary)
    # Eval resumen
    text_eval, struct_eval, final_eval, reasons, used_judge, obj = eval_summary_single(prompt=request, ai_response=raw_summary)
    print(f"\n=== EVALUACIÓN DEL RESUMEN (JUEZ: {used_judge}) ===")
    print(text_eval)
    print(f"\nCalificación (resumen, juez): {int(struct_eval.value)}/5 ({struct_eval.name})")
    if reasons: print("Calificación final (resumen, tras guardrails):", f"{int(final_eval.value)}/5 ({final_eval.name})")
    # QA
    question = "How does the model perform on code tasks?"
    print("\n[PROMPTS · QA]\nquestion =", question, "\nsystem_instruction (qa) = (none)")
    print(f"\n=== RESPUESTA QA (RAW (solo request): {GENERATOR_MODEL}) ===")
    raw_ans = answer_question_raw(question)
    print(raw_ans)
    # Eval QA
    text_eval_q, struct_eval_q, final_eval_q, reasons_q, used_judge_q, obj_q = eval_answer_single(prompt=question, ai_response=raw_ans)
    print(f"\n=== EVALUACIÓN QA (JUEZ: {used_judge_q}) ===")
    print(text_eval_q)
    print(f"\nCalificación QA (juez): {int(struct_eval_q.value)}/5 ({struct_eval_q.name})")
    if reasons_q: print("Calificación final QA (tras guardrails):", f"{int(final_eval_q.value)}/5 ({final_eval_q.name})")
    return {"summary": raw_summary, "answer": raw_ans, "question": question}

def run_suite_guided():
    print("\n" + "="*80)
    print("### SUITE GUIDED (request + system) — MODO: GUIDED")
    print("="*80)
    # Resumen
    request = "Tell me about the training process used here."
    print("\n[PROMPTS · RESUMEN]\nrequest =", request, "\nsystem_instruction (summary) =", GEN_SUMMARY_STYLE)
    print(f"\n=== RESUMEN (GUIDED (request + system): {GENERATOR_MODEL}) ===")
    guided_summary = summarise_doc(request)
    print(guided_summary)
    text_eval, struct_eval, final_eval, reasons, used_judge, obj = eval_summary_single(prompt=request, ai_response=guided_summary)
    print(f"\n=== EVALUACIÓN DEL RESUMEN (JUEZ: {used_judge}) ===")
    print(text_eval)
    print(f"\nCalificación (resumen, juez): {int(struct_eval.value)}/5 ({struct_eval.name})")
    if reasons: print("Calificación final (resumen, tras guardrails):", f"{int(final_eval.value)}/5 ({final_eval.name})")

    # QA
    question = "How does the model perform on code tasks?"
    qa_guidance, focus_terms = make_dynamic_qa_guidance(question)
    print("\n[PROMPTS · QA]\nquestion =", question, "\nsystem_instruction (qa) =\n" + qa_guidance)
    print(f"\n=== RESPUESTA QA (GUIDED (request + system): {GENERATOR_MODEL}) ===")
    guided_ans = answer_question_guided(question)
    print(guided_ans)

    text_eval_q, struct_eval_q, final_eval_q, reasons_q, used_judge_q, obj_q = eval_answer_single(
        prompt=question, ai_response=guided_ans, require_evidence_for_5=2
    )
    print(f"\n=== EVALUACIÓN QA (JUEZ: {used_judge_q}) ===")
    print(text_eval_q)
    print(f"\nCalificación QA (juez): {int(struct_eval_q.value)}/5 ({struct_eval_q.name})")
    if reasons_q: print("Calificación final QA (tras guardrails):", f"{int(final_eval_q.value)}/5 ({final_eval_q.name})")
    return {"summary": guided_summary, "answer": guided_ans, "question": question}

def run_pairwise(question: str, raw_answer: str, guided_answer: str):
    text, choice = eval_pairwise(prompt=question, response_a=raw_answer, response_b=guided_answer)
    print("\n=== PAIRWISE RAW vs GUIDED ===")
    print(text)
    print("Winner:", choice.value)

# =========================================================
# 10) MAIN (muestra profile, modelos y corre suites)
# =========================================================
def main():
    print(f"\n[INFO] PROFILE: {CURRENT_PROFILE.name}")
    print(f"[INFO] GENERATOR_MODEL: {GENERATOR_MODEL}")
    print(f"[INFO] JUDGE_MODELS (preferred → fallbacks): {', '.join(JUDGE_MODELS + JUDGE_FALLBACKS)}")

    print("\n\n>>> SUITE A: SOLO PROMPTS INICIALES (RAW)")
    raw = run_suite_raw()

    print("\n\n>>> SUITE B: PROMPTS + ADICIONES (GUIDED)")
    guided = run_suite_guided()

    # Pairwise: comparar respuestas QA RAW vs GUIDED
    run_pairwise(question=raw["question"], raw_answer=raw["answer"], guided_answer=guided["answer"])

    # (Opcional) Ejemplo de “batch” con stub local
    if os.environ.get("RUN_LOCAL_BATCH", "0") == "1":
        items = [{"pdf": LOCAL_PDF, "summary_prompt": "Tell me about the training process used here.", "qa_question": raw["question"]}]
        job = submit_eval_batch(project="local", location="local", items=items, gcs_dest="./_batch_out")
        print("[INFO] Batch job:", job)
        # Leer resultados
        out_path = "./_batch_out/batch_results.jsonl"
        if os.path.exists(out_path):
            print("[INFO] Batch outputs:")
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    print(line.rstrip())

    print("\nHecho.")

# =========================================================
# 11) EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    main()
