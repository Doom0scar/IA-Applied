#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script demostrativo para trabajar con la API de Gemini (google-genai) en Python.

Incluye:
- Configuración de modelos preferidos.
- Detección de entorno (notebook vs. consola) y salida en Markdown cuando aplica.
- Cliente Gemini y carga de API key.
- Ayudantes robustos: retries (tenacity), extracción de texto, streaming, y uso de herramientas (code-exec).
- Secciones de ejemplo: primera llamada, chat multivuelta, inspección de modelos, control de longitud de salida,
  temperatura/top-p, zero-shot/enum/few-shot/JSON, chain-of-thought, streaming thinking, ReAct, code-gen y code-exec,
  y explicación de un archivo descargado.

Este archivo está fuertemente documentado línea por línea / bloque por bloque.
"""

# =======================================
# IMPORTS y utilidades generales
# =======================================

from __future__ import annotations  # Habilita anotaciones de tipos "postponed" (útil para forward refs en Python <3.11)

import os, sys, io, enum, json, re, subprocess  # Módulos estándar para OS/IO, expresiones regulares y subprocesos
from typing import Optional                         # Tipado opcional (no imprescindible aquí)
from pprint import pprint                           # Pretty-print para estructuras complejas

# ---------------------------
# Modelos preferidos (tuyos)
# ---------------------------
MODEL_FAST = "gemini-2.0-flash"                 # Modelo rápido / costo menor
MODEL_QUALITY = "gemini-2.5-pro"                # Modelo de mayor calidad
MODEL_THINKING = "gemini-2.0-flash-thinking-exp"  # Modelo experimental con "thinking"

# --- Detección de notebook y Markdown robusto ---
def _in_notebook() -> bool:
    """
    Devuelve True si se está ejecutando en un Jupyter/Colab (ZMQInteractiveShell).
    """
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

def _md(text: str):
    """
    Si hay notebook, imprime en Markdown; si no, usa print estándar.
    """
    if _in_notebook():
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text)

# ----------------------------------------------------------------
# Importar SDK de Gemini + helpers del notebook (como el original)
# ----------------------------------------------------------------
from google import genai                 # SDK principal
from google.genai import types          # Tipos auxiliares (GenerateContentConfig, Tool, etc.)

# Intento cargar utilidades de IPython (no son críticas en consola)
try:
    from IPython.display import HTML, Markdown, display
except Exception:
    HTML = Markdown = display = None

# -----------------------------------------------------------
# Retry helper (google.api_core.retry) aplicado a generate_content
# -----------------------------------------------------------
try:
    from google.api_core import retry

    # Función para decidir si un error es retirable (429/503 típicos de rate limiting o servicio)
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

    # Monkeypatch: envuelve generate_content con retry de google.api_core.retry
    genai.models.Models.generate_content = retry.Retry(
        predicate=is_retriable
    )(genai.models.Models.generate_content)

    # Nota: generate_content_stream NO se envuelve aquí por compatibilidad
except Exception as _e:
    print("[WARN] No se pudo activar google.api_core.retry:", _e)

# -----------------------------------------------------------
# Carga de API key (YEK.txt / env var)
# -----------------------------------------------------------
def load_api_key() -> str:
    """
    Intenta leer la API key desde YEK.txt, si no, de la variable de entorno GOOGLE_API_KEY.
    Si falla, aborta con mensaje de error.
    """
    try:
        with open("VYEK.txt", "r", encoding="utf-8") as f:
            k = f.read().strip()
            if k:
                return k
    except FileNotFoundError:
        pass
    k = os.environ.get("GOOGLE_API_KEY")
    if k:
        return k
    print("ERROR: No encontré tu API key. Ponla en 'YEK.txt' o define GOOGLE_API_KEY.", file=sys.stderr)
    sys.exit(1)

GOOGLE_API_KEY = load_api_key()              # Lee la key
client = genai.Client(api_key=GOOGLE_API_KEY)  # Crea cliente Gemini

# ===========================================================
# BLOQUE 0 — Helpers (config + backoff + streaming + tools)
# ===========================================================
try:
    # Tenacity: reintentos exponenciales a nivel de nuestras funciones helper
    from tenacity import retry as _retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    _HAS_TENACITY = True
except Exception:
    _HAS_TENACITY = False

def _retryable(fn):
    """
    Decorador que añade reintentos con tenacity (si está instalado).
    Reintenta hasta 3 veces con backoff exponencial ante cualquier Exception.
    """
    if not _HAS_TENACITY:
        return fn
    @_retry(wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(Exception),
            reraise=True)
    def _wrapped(*a, **kw): return fn(*a, **kw)
    return _wrapped

class GenCfg:
    """
    Contenedor simple para los parámetros de generación.
    Facilita construir un types.GenerateContentConfig y pasar 'model' + 'config' juntos.
    """
    def __init__(self, *, model=MODEL_FAST, temperature=None, top_p=None,
                 max_output_tokens=None, stop_sequences=None,
                 system_instruction=None, response_mime_type=None,
                 response_schema=None, response_modalities=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.stop_sequences = stop_sequences
        self.system_instruction = system_instruction
        self.response_mime_type = response_mime_type
        self.response_schema = response_schema
        self.response_modalities = response_modalities

    def to_kwargs(self):
        """
        Construye el dict de kwargs que espera client.models.generate_content:
        {'model': str, 'config': GenerateContentConfig}
        Solo incluye campos no None.
        """
        cfg = types.GenerateContentConfig(**{
            k: v for k, v in {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_output_tokens": self.max_output_tokens,
                "stop_sequences": self.stop_sequences,
                "system_instruction": self.system_instruction,
                "response_mime_type": self.response_mime_type,
                "response_schema": self.response_schema,
                "response_modalities": self.response_modalities,
            }.items() if v is not None
        })
        return {"model": self.model, "config": cfg}

def extract_text(resp) -> str:
    """
    Extrae texto de una respuesta de generate_content.
    - Si resp.text viene poblado, lo usa.
    - Si no, concatena los parts con 'text' en candidates[*].content.parts[*].
    - Devuelve string vacío ante cualquier excepción.
    """
    try:
        if getattr(resp, "text", None) and resp.text.strip():
            return resp.text
        texts = []
        for cand in getattr(resp, "candidates", []):
            parts = getattr(cand.content, "parts", []) if getattr(cand, "content", None) else []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    texts.append(t)
        return "\n".join(texts).strip()
    except Exception:
        return ""

@_retryable
def safe_generate(client, contents, *, cfg: GenCfg):
    """
    Envoltura segura de generate_content con:
    - Verificación de tipo de cfg.
    - Default de response_modalities=["TEXT"] si no se estableció.
    - Reintentos (vía tenacity) si está disponible.
    """
    if not isinstance(cfg, GenCfg):
        raise TypeError(f"safe_generate: 'cfg' debe ser GenCfg, recibí {type(cfg).__name__}")
    if not cfg.response_modalities:
        cfg.response_modalities = ["TEXT"]
    return client.models.generate_content(contents=contents, **cfg.to_kwargs())

def safe_stream(client, contents, *, cfg: GenCfg):
    """
    Versión streaming: devuelve chunks de texto (strings).
    Ajusta response_modalities a TEXT si no viene.
    """
    if not cfg.response_modalities:
        cfg.response_modalities = ["TEXT"]
    stream = client.models.generate_content_stream(contents=contents, **cfg.to_kwargs())
    for chunk in stream:
        txt = getattr(chunk, "text", "")
        if txt:
            yield txt

def _retryable_tools(fn):
    """
    Igual que _retryable, pero separado para funciones con tools.
    """
    if not _HAS_TENACITY:
        return fn
    @_retry(wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(Exception),
            reraise=True)
    def _wrap(*a, **k): return fn(*a, **k)
    return _wrap

@_retryable_tools
def safe_generate_with_tools(client, contents, *, cfg: GenCfg, tools: list):
    """
    Llama generate_content con tools (p.ej. code_execution).
    Construye un GenerateContentConfig base a partir de cfg y le asigna 'tools'.
    """
    base = types.GenerateContentConfig(**{
        k: v for k, v in {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_output_tokens": cfg.max_output_tokens,
            "stop_sequences": cfg.stop_sequences,
            "system_instruction": cfg.system_instruction,
            "response_mime_type": cfg.response_mime_type,
            "response_schema": cfg.response_schema,
            "response_modalities": cfg.response_modalities,
        }.items() if v is not None
    })
    base.tools = tools
    return client.models.generate_content(model=cfg.model, contents=contents, config=base)

# ===========================================================
# SECCIÓN 1 — Primera llamada (role/system prompting)
# ===========================================================
"""
Primera llamada (role/system prompting)

Qué es: defines la “persona” del asistente con una system instruction (role prompting) 
y le das un prompt del usuario.

Qué hace: “Eres un tutor claro y breve…” + “Explain AI to me like I’m a kid.”

Por qué importa: separa rol (tono, estilo, objetivos) del contenido (petición concreta). 
Es la base de un comportamiento consistente sin reescribir el estilo en cada turno.
"""
print("\n=== Sección 1 — Primera llamada (mejorada) ===")
cfg = GenCfg(
    model=MODEL_FAST,
    system_instruction="Eres un tutor claro y breve. Responde en lenguaje sencillo para un niño de 8 años.",
    temperature=0.3,
    top_p=0.9,
)
PROMPT = "Explain AI to me like I'm a kid."
print("=== System instruction ===")
print(cfg.system_instruction, "\n")
print("\n=== Prompt ===")
print(PROMPT, "\n")
print("=== Model Response === \n")
resp = safe_generate(client, PROMPT, cfg=cfg)      # Llamada con nuestro helper seguro
_md(extract_text(resp) or "[Sin texto]")           # Imprime en Markdown si hay notebook, o en texto plano

# ===========================================================
# SECCIÓN 2 — Chat multivuelta (conversational context)
# ===========================================================
"""
Chat multivuelta (conversational context)

Qué es: diálogo con memoria de contexto dentro de la misma sesión/chat.

Qué hace: el modelo “recuerda” que te llamas Alex y responde sobre dinosaurios, 
luego verifica si recuerda tu nombre.

Por qué importa: demuestra estado conversacional en la ventana de contexto 
(no es aprendizaje/entrenamiento ni memoria a largo plazo). 

Sirve para asistentes que deben hilar turnos y retener detalles breves.
"""
print("\n=== Sección 2 — Chat multivuelta (mejorado) ===")
chat_cfg = types.GenerateContentConfig(
    system_instruction="Actúa como guía amable y breve; recuerda el nombre si te lo dan."
)
chat = client.chats.create(model=MODEL_FAST, history=[], config=chat_cfg)  # Crea un chat con config
print("=== System instruction ===")
print(getattr(chat_cfg, "system_instruction", "[Sin instrucción]"))

def send_and_print(ch, msg: str):
    """
    Utilidad para enviar un mensaje al chat, imprimir entrada/salida y
    devolver la respuesta nativa del SDK.
    """
    print("\n=== Usuario ===")
    print(msg)
    resp = ch.send_message(msg)
    print("=== Asistente ===")
    print(extract_text(resp) or "[Sin texto]")
    return resp

# Tres turnos de ejemplo
send_and_print(chat, "Hello! My name is Alex.\n")
send_and_print(chat, "Can you tell me something interesting about dinosaurs?\n")
send_and_print(chat, "Do you remember what my name is?\n")

# ===========================================================
# SECCIÓN 3 — Modelos disponibles / inspección
# ===========================================================
"""
Modelos disponibles / inspección (model capabilities)

Qué es: introspección de modelos a través del SDK (nombres, límites, métodos soportados).

Qué hace: lista modelos y muestra input_token_limit, output_token_limit, acciones, etc.

Por qué importa: te ayuda a elegir el modelo correcto para cada tarea 
(p.ej., contexto grande, latencia, costo, calidad).
"""
print("\n=== Sección 3 — Modelos disponibles (resumen) ===")
models = list(client.models.list())      # Itera modelos (puede paginar en el SDK)
for m in models[:10]:                    # Muestra solo 10 por brevedad
    print(m.name)
print("\n[Info de", f"models/{MODEL_FAST}", "]:")
# Busca el objeto de ese modelo entre los listados
info = next((m for m in models if m.name.endswith(MODEL_FAST)), None) or next((m for m in models if m.name == MODEL_FAST), None)
if info:
    pprint(info.to_json_dict())          # Pretty-print del detalle del modelo (si el SDK lo expone)

# ===========================================================
# SECCIÓN 4 — Output length
# ===========================================================
"""
Output length (max_output_tokens & prompt shaping)

Qué es: control explícito del largo de salida con max_output_tokens y, 
desde el prompt, del formato (ensayo vs poema).

Qué hace: compara una respuesta larga vs corta forzada.

Por qué importa: gestiona coste, truncado y formato. 
Crítico cuando integras el LLM en pipelines que esperan salidas acotadas.
"""
print("\n=== Sección 4 — Output length ===")

short_config = types.GenerateContentConfig(max_output_tokens=200)  # Limita tokens de salida

def run_and_print(prompt: str):
    """
    Helper para ejecutar una prompt con max_output_tokens=200 y mostrar entrada/salida.
    """
    print("\n=== Prompt (contents) ===")
    print(prompt)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        config=short_config,
        contents=prompt
    )
    print("\n=== Respuesta ===")
    print(extract_text(resp) or "[Sin texto]")

# Dos ejemplos con distinto tamaño de salida esperado
run_and_print("Write a 1000 word essay on the importance of olives in modern society.")
run_and_print("Write a short poem on the importance of olives in modern society.")

# ===========================================================
# SECCIÓN 5 — Temperature
# ===========================================================
"""
Temperature (creatividad vs. determinismo)

Qué es: temperature controla la aleatoriedad de muestreo; top_p delimita la masa de probabilidad.

Qué hace: con T alta aparecen respuestas variadas; con T=0 tienden a ser estables/repetibles.

Por qué importa: ajusta creatividad vs consistencia según el uso (p.ej., ideación vs. clasificación).
"""
print("\n=== Sección 5 — Temperature ===")
hiT = GenCfg(model=MODEL_FAST, temperature=2.0, top_p=1.0)  # Alta creatividad/diversidad
prompt_diverse = "Pick a random uncommon colour name (e.g., chartreuse, vermilion, cerulean). Respond in a single word."
print("\n=== Prompt ===")
print(prompt_diverse, "\n")
print("[High T]")
for _ in range(5):
    print(extract_text(safe_generate(client, prompt_diverse, cfg=hiT)) or "[Sin texto]", "-"*25)

loT = GenCfg(model=MODEL_FAST, temperature=0.0, top_p=0.9)  # Determinista
prompt_low = "Pick a colour. Respond in a single word."
print("\n=== Prompt ===")
print(prompt_low, "\n")
print("\n[Low T]")
for _ in range(5):
    print(extract_text(safe_generate(client, prompt_low, cfg=loT)) or "[Sin texto]", "-"*25)

# ===========================================================
# SECCIÓN 6 — Top-P + story
# ===========================================================
"""
Top-P + story (nucleus sampling)

Qué es: top-p (nucleus sampling) limita a las palabras más probables que acumulan p≈0.95, etc.

Qué hace: cuenta corta con top_p=0.95 (junto a temperature).

Por qué importa: afina la diversidad de salida de forma más “semántica” que solo temperature; 
útil para mantener calidad sin divagar.
"""
print("\n=== Sección 6 — Top-P + story ===")
story_cfg = GenCfg(model=MODEL_FAST, temperature=1.0, top_p=0.95)  # Ajuste de muestreo
PROMPT = "You are a creative writer. Write a short story about a cat who goes on an adventure. \n \n"
print("\n=== Prompt ===")
print(PROMPT, "\n")
print("=== Model Response === \n")
print(extract_text(safe_generate(client, PROMPT, cfg=story_cfg)) or "[Sin texto]")

# ===========================================================
# SECCIÓN 7 — Zero-shot / Enum / Few-shot / JSON
# ===========================================================
"""
Zero-shot / Enum / Few-shot / JSON (prompting patterns)

Zero-shot (clasificación de sentimiento): sin ejemplos, el modelo infiere 
POSITIVE/NEUTRAL/NEGATIVE por el lenguaje global.
• Importancia: baseline rápido; útil cuando no quieres diseñar ejemplos.

Enum estricto: response_mime_type="text/x.enum" + response_schema (Enum) 
restringe la salida a valores válidos.
• Importancia: contratos de salida sólidos (ideal para integraciones).

Few-shot: das pocos ejemplos (one/few) y luego un caso nuevo; el modelo imita el formato.
• Importancia: sube precisión y consistencia de formato sin finetune.

JSON mode (TypedDict): response_mime_type="application/json" + response_schema 
para estructura tipada.
• Importancia: evita parseos frágiles; listo para consumir en código.
"""
print("\n=== Sección 7 — Zero-shot / Enum / Few-shot / JSON ===")

# ---------- Zero-shot ----------
print("\n=== Zero-shot ===")
zero_shot_prompt = (
    "Decide exactly whether the overall sentiment of the following movie review is: POSITIVE, NEUTRAL or NEGATIVE. Base your answer on the reviewer’s overall opinion, not on isolated phrases.\n"
    "Review: \"Her\" is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, "
    "unchecked. I wish there were more movies like this masterpiece.\n"
    "Sentiment:"
)
print("\n=== ZERO-SHOT PROMPT ===")
print(zero_shot_prompt, "\n")

zs_cfg = types.GenerateContentConfig(
    temperature=0.0,   # determinista
    top_p=1,
    max_output_tokens=16
)
zs_resp = client.models.generate_content(
    model=MODEL_FAST,
    config=zs_cfg,
    contents=zero_shot_prompt
)
print("=== ZERO-SHOT RESPONSE ===")
print(extract_text(zs_resp) or "[Sin texto]")

# ---------- Enum estricto ----------
print("\nEnum")
class Sentiment(enum.Enum):
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"

enum_cfg = GenCfg(
    model=MODEL_FAST,
    temperature=0.0,
    top_p=1,
    max_output_tokens=8,
    response_mime_type="text/x.enum",  # Modo enum del SDK
    response_schema=Sentiment
)
enum_resp = safe_generate(client, zero_shot_prompt, cfg=enum_cfg)
print("\n[Enum mode texto]:", extract_text(enum_resp) or "[Sin texto]")
print("[Enum parsed]:", getattr(enum_resp, "parsed", None))  # Si el SDK parsea al Enum

# ---------- Few-shot ----------
print("\n === Few-shot ===")
# Few-shot pizza (texto)
few = """Parse a customer's pizza order into valid JSON:

EXAMPLE:
I want a small pizza with cheese, tomato sauce, and pepperoni.
JSON Response:
```
{"size":"small","type":"normal","ingredients":["cheese","tomato sauce","pepperoni"]}
```

EXAMPLE:
Can I get a large pizza with tomato sauce, basil and mozzarella
JSON Response:
```
{"size":"large","type":"normal","ingredients":["tomato sauce","basil","mozzarella"]}
```


ORDER:
"""
print("Here is the example that tells AI how to answer: \n \n", few)
customer = "Give me a large with cheese & pineapple"
print(" === Here is the customer order (Input from user): ===\n", customer)
print("=== Model Response ===\n")
fs_cfg = GenCfg(model=MODEL_FAST, temperature=0.1, top_p=1, max_output_tokens=250)
print("\n[Few-shot pizza (texto)]:\n", extract_text(safe_generate(client, [few, customer], cfg=fs_cfg)) or "[Sin texto]")

# ---------- JSON mode ----------
print("\n === JSON mode ===")
import typing_extensions as typing
class PizzaOrder(typing.TypedDict):
    size: str
    ingredients: list[str]
    type: str

json_cfg = GenCfg(model=MODEL_FAST,
                  temperature=0.1,
                  response_mime_type="application/json",  # Pide JSON estructurado
                  response_schema=PizzaOrder)
PROMPT_JSON = "Can I have a large dessert pizza with apple and chocolate"
print("\n=== Here is the customer order (Input from user): ===")
print(PROMPT_JSON, "\n")
print("=== Model Response JSON===\n")
print("\n[JSON mode (TypedDict)]:\n", extract_text(safe_generate(client, PROMPT_JSON, cfg=json_cfg)) or "[Sin texto]")

# ===========================================================
# SECCIÓN 8 — Chain-of-Thought (no usar CoT privado; solo demostración de prompts)
# ===========================================================
"""
Chain-of-Thought (direct vs step-by-step)

Qué es: cambiar el estilo de razonamiento desde el prompt (“direct” vs “let’s think step by step”).

Qué hace: compara respuesta directa vs. razonamiento explícito.

Por qué importa: el razonamiento paso a paso puede mejorar tareas de lógica; 
también hace auditables los pasos 
(ojo: no es Chain of Thought “privado”; aquí pides pasos explícitos en la salida).
"""
print("\n=== Sección 8 — Chain-of-Thought (directo vs paso a paso) ===")
print("\nChain-of-Thought (direct)\n")
direct = """When I was 4 years old, my partner was 3 times my age. 
Now, I am 20 years old. How old is my partner? Return the answer directly."""
print("=== Problem to solve ===\n")
print(direct, "\n")
print("=== Model Response ===\n")
print("[Direct]:", extract_text(safe_generate(client, direct, cfg=GenCfg(model=MODEL_FAST))) or "[Sin texto]")

print("\nChain-of-Thought (step by step)\n")
cot = """When I was 4 years old, my partner was 3 times my age. 
Now, I am 20 years old. How old is my partner? Let's think step by step."""
print("=== Problem to solve ===\n")
print(cot, "\n")
print("=== Model Response ===\n")
print("[Step by step]:", extract_text(safe_generate(client, cot, cfg=GenCfg(model=MODEL_FAST))) or "[Sin texto]")

# ===========================================================
# SECCIÓN 9 — Thinking (streaming) + Fallback
# ===========================================================
"""
Thinking (streaming) + fallback (latencia percibida)

Qué es: uso de un modelo “thinking/exp” y streaming de tokens.

Qué hace: imprime chunks conforme llegan y acumula un buffer por si no hay entorno notebook.

Por qué importa: mejor UX (respuesta temprana), útil en interfaces en tiempo real.
"""
print("\n=== Sección 9 — Thinking (streaming) ===")
think_cfg = GenCfg(
    model=MODEL_THINKING,  # Modelo experimental con "thinking"
    system_instruction="Piensa paso a paso pero entrega una respuesta final clara y concisa.",
    temperature=0.3,
    max_output_tokens=500,
)
print("=== System instruction ===")
print(think_cfg.system_instruction, "\n")
prompt_think = (
    "Si cuando tenía 4 años mi pareja tenía el triple, y hoy tengo 20, "
    "¿cuántos años tiene mi pareja? Explica en 1-2 frases y da solo el número al final."
)
print("\n=== Prompt ===")
print(prompt_think, "\n")
buf = io.StringIO()  # Buffer donde acumulamos lo que llega del stream

# Importante: iteramos sobre safe_stream (que emite strings) y vamos imprimiendo y guardando
for chunk in safe_stream(client, prompt_think, cfg=think_cfg):
    buf.write(chunk)
    print(chunk, end="")

# Render “bonito” si hay notebook; si no, texto plano
try:
    from IPython.display import clear_output, Markdown as _MD, display as _display
    clear_output()
    _display(_MD(buf.getvalue()))
except Exception:
    print("\n=== Model Response (streaming) + Fallback ===")
    print("\n\n[STREAM RESULT]\n" + buf.getvalue())

# ===========================================================
# SECCIÓN 10 — ReAct (Reason + Act) con stop_sequences
# ===========================================================
"""
ReAct (Reason+Act con stop_sequences)

Qué es: patrón ReAct: alternar Thought → Action → Observation → Finish, 
y cortar con stop_sequences antes de “Observation”.

Qué hace: muestra dos ejemplos guía, formula una pregunta, 
simula una “Observation” y el modelo continúa.

Por qué importa: base de agentes con herramientas (búsqueda, lookup, ejecución), 
controlando el flujo para evitar desbordes y alinear el comportamiento.
"""
print("\n=== Sección 10 — ReAct (stop_sequences) ===")
model_instructions = """
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation,
Observation is understanding relevant information from an Action's output and Action can be one of three types:
 (1) <search>entity</search>, which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it
     will return some similar entities to search and you can try to search the information from those topics.
 (2) <lookup>keyword</lookup>, which returns the next sentence containing keyword in the current context. This only does exact matches,
     so keep your searches short.
 (3) <finish>answer</finish>, which returns the answer and finishes the task.
"""
print("=== Model Instructions ===\n")
print(model_instructions, "\n \n")

# Dos ejemplos que definen el formato ReAct
example1 = """Question
Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?

Thought 1
The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.

Action 1
<search>Milhouse</search>

Observation 1
Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Thought 2
The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".

Action 2
<lookup>named after</lookup>

Observation 2
Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.

Thought 3
Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.

Action 3
<finish>Richard Nixon</finish>
"""
print("\n === This is the first example that guides the model to the desired expected response: ===\n",example1, "\n \n")

example2 = """Question
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1
I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.

Action 1
<search>Colorado orogeny</search>

Observation 1
The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought 2
It does not mention the eastern sector. So I need to look up eastern sector.

Action 2
<lookup>eastern sector</lookup>

Observation 2
The eastern sector extends into the High Plains and is called the Central Plains orogeny.

Thought 3
The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.

Action 3
<search>High Plains</search>

Observation 3
High Plains refers to one of two distinct land regions

Thought 4
I need to instead search High Plains (United States).

Action 4
<search>High Plains (United States)</search>

Observation 4
The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m).

Thought 5
High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.

Action 5
<finish>1,800 to 7,000 ft</finish>
"""
print("\n === This is the second example that guides the model to the desired expected response: === \n",example2, "\n \n")

# Configuración con stop_sequences para cortar antes de "Observation"
react_cfg = types.GenerateContentConfig(
    stop_sequences=["\nObservation"],
    system_instruction=model_instructions + example1 + example2,
)
print("=== Final system instruction: model instructions + example1 + example2 ===")
print(react_cfg.system_instruction, "\n")

react_chat = client.chats.create(model=MODEL_FAST, config=react_cfg)  # Chat con las reglas ReAct
question = """Question
Who was the youngest author listed on the transformers NLP paper?
"""
print("=== Question asked by user ===\n")
print(question, "\n \n")
print("=== Model Response ===\n")
print(extract_text(react_chat.send_message(question)) or "[Sin texto]")  # Primer turno: modelo razona y sugiere acción

# Simulación de una "Observation" (en un sistema real la obtendrías de una herramienta de búsqueda)
obs = """Observation 1
[1706.03762] Attention Is All You Need
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
"""
print("=== Model Response ===\n")
print(extract_text(react_chat.send_message(obs)) or "[Sin texto]")  # Segundo turno: modelo usa la observación

# ===========================================================
# SECCIÓN 11 — Code-gen + Code-exec
# ===========================================================
"""
Code-gen + Code-exec (tools)

Code-gen: genera solo código (sin explicación) 
con una system instruction restrictiva y limpia el bloque.

Code-exec: usa la tool de ejecución del SDK para correr el código generado y 
parsear parts (texto, código, resultado).

Por qué importa: patrón de programación asistida end-to-end 
(generar → ejecutar → leer resultados) con seguridad y formato consistente.
"""
print("\n=== Sección 11 — Code-gen ===")
code_cfg = GenCfg(
    model=MODEL_FAST,
    system_instruction=(
        """ Devuelve solo código Python válido.
        No incluyas comentarios, docstrings, explicaciones ni markdown.
        Responde únicamente con el código."""
    ),
    temperature=0.2,
    top_p=0.9,
    max_output_tokens=512,
)
print("=== System instruction ===")
print(code_cfg.system_instruction, "\n")
print("\n=== Prompt TO CODE===")
PROMPT_CODE = "Write a Python function to calculate the factorial of a number."
print(PROMPT_CODE, "\n")
resp = safe_generate(client, PROMPT_CODE, cfg=code_cfg)  # Genera SOLO código Python
code_text_full = extract_text(resp)
# Extrae bloque de código si vino con ```; si no, usa el texto crudo
m = re.search(r"```(?:python)?\s*(.*?)```", code_text_full, flags=re.S)
final_code = m.group(1).strip() if m else (code_text_full.strip() if code_text_full else "")
if not final_code:
    # Respaldo mínimo si el modelo no devolvió nada
    final_code = "def factorial(n):\n    return 1 if n<=1 else n*factorial(n-1)"
print("=== Model Response ===\n")
print(final_code)

print("\n=== Sección 11 — Code-exec (herramienta de ejecución) ===")
exec_cfg = GenCfg(
    model=MODEL_FAST,
    system_instruction="Escribe el mínimo código Python necesario. Usa print para mostrar el resultado.",
    temperature=0.2,
    max_output_tokens=512,
)
print("=== System instruction ===")
print(exec_cfg.system_instruction, "\n")
PROMPT_EXEC = "Generate the first 14 odd prime numbers, then calculate their sum."
print("\n=== Prompt EXEC===")
print(PROMPT_EXEC, "\n")
# Usa herramientas del SDK: code_execution para ejecutar el código generado en un sandbox
resp_exec = safe_generate_with_tools(
    client,
    PROMPT_EXEC,
    cfg=exec_cfg,
    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
)

# Parse de parts: texto, código ejecutable y resultados de ejecución
parts = resp_exec.candidates[0].content.parts
result = {"text": [], "code": [], "exec": []}
for p in parts:
    if getattr(p, "text", None):
        result["text"].append(p.text)
    elif getattr(p, "executable_code", None):
        result["code"].append(p.executable_code.code)
    elif getattr(p, "code_execution_result", None):
        result["exec"].append({
            "status": p.code_execution_result.outcome,
            "output": p.code_execution_result.output
        })
print("=== Model Response ===\n")
print(result)

# ===========================================================
# SECCIÓN 12 — Explicar código de un archivo remoto
# ===========================================================
"""
Explicar código (archivo remoto)

Qué es: “ingesta” de un script (curl/urllib), 
prompt para resumen técnico y extracción robusta de texto.

Qué hace: descarga gitprompt.sh, arma un prompt y resume en lenguaje claro.

Por qué importa: mini-flujo de code review/asistencia: útil para onboarding, 
auditorías rápidas, documentación viva.
"""
print("\n=== Sección 12 — Explicar código de archivo ===")
import subprocess, urllib.request

URL = "https://raw.githubusercontent.com/magicmonty/bash-git-prompt/refs/heads/master/gitprompt.sh"

# --- Descargar archivo (curl -> urllib fallback) ---
res = subprocess.run(["curl", "-sL", URL], capture_output=True, text=True)
file_contents = res.stdout
if not file_contents:
    # Fallback a urllib en caso de que curl no esté disponible/permitido
    with urllib.request.urlopen(URL, timeout=15) as resp:
        file_contents = resp.read().decode("utf-8", errors="ignore")

# --- Prompt simple para explicación de archivo ---
explain_prompt = f"""
Please explain what this file does at a very high level. What is it, and why would I use it?

```bash
{file_contents}
```
""" 
print("\n=== Prompt ===")
print(explain_prompt[:500] + "\n...\n")  # Print inicio del prompt para referencia
# --- Llamada al modelo ---
resp = client.models.generate_content(
    model=MODEL_FAST,
    contents=explain_prompt
)

# --- Extraer texto de forma robusta y PRINT (no Markdown) ---
def _extract_text(r):
    if getattr(r, "text", None):
        t = r.text.strip()
        if t:
            return t
    parts = []
    for cand in getattr(r, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content:
            for p in getattr(content, "parts", []) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
    return "\n".join(parts).strip()

print("=== Model Response ===\n")
print(_extract_text(resp) or "[Sin texto]")