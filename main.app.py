"""
Laboratorio de PLN con Groq
---------------------------
Plataforma didáctica en Streamlit para analizar un texto escrito a mano:
  1. Tokenización con varios esquemas (IDs y tokens coloreados)
  2. Bolsa de palabras (Bag of Words) y TF-IDF
  3. Similitud de coseno entre frases
  4. Esquema generativo (modelo de bigramas entrenado con descenso de gradiente,
     donde SÍ existe la tasa de aprendizaje / learning rate)
  5. Generación de respuestas con modelos de Groq (sin Llama, prioridad GPT)
  6. Catálogo de modelos disponibles en Groq

Ejecutar:  streamlit run main_app.py
"""

import html
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tiktoken
except ImportError:  # la app sigue funcionando sin BPE
    tiktoken = None

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================================
# Configuración general
# ============================================================================
st.set_page_config(page_title="Laboratorio de PLN con Groq", page_icon="🧪", layout="wide")

TEXTO_EJEMPLO = (
    "El gato come pescado en la cocina.\n"
    "El perro come carne en el patio.\n"
    "Los modelos de lenguaje predicen la siguiente palabra.\n"
    "Un modelo de lenguaje aprende patrones a partir de muchos textos.\n"
    "El gato duerme en el patio."
)

# Colores pastel para pintar los tokens (se reciclan en orden)
PALETA = [
    "#FFD6A5", "#CAFFBF", "#9BF6FF", "#BDB2FF", "#FFC6FF",
    "#FDFFB6", "#FFADAD", "#A0C4FF", "#D0F4DE", "#E4C1F9",
]

# sklearn solo trae stopwords en inglés; esta es una lista corta en español
STOPWORDS_ES = sorted({
    "a", "al", "algo", "como", "con", "de", "del", "desde", "donde", "e", "el", "ella",
    "ellos", "en", "entre", "era", "es", "esa", "ese", "eso", "esta", "este", "esto",
    "fue", "ha", "hay", "la", "las", "le", "les", "lo", "los", "mas", "más", "me", "mi",
    "muy", "no", "nos", "o", "para", "pero", "por", "porque", "que", "qué", "se", "si",
    "sí", "sin", "sobre", "son", "su", "sus", "también", "te", "tu", "un", "una", "uno",
    "unos", "unas", "y", "ya", "yo",
})

# Palabras en el id que indican modelos que NO son de chat (audio, voz, etc.)
NO_CHAT = ("whisper", "tts", "orpheus", "playai", "distil")
MODELOS_RESPALDO = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


# ============================================================================
# Utilidades de Groq
# ============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def listar_modelos(api_key: str) -> list[dict]:
    """Consulta el catálogo real de Groq. Sirve también para validar la API key."""
    cliente = Groq(api_key=api_key)
    respuesta = cliente.models.list()
    filas = []
    for m in respuesta.data:
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        filas.append({
            "id": d.get("id"),
            "propietario": d.get("owned_by"),
            "ventana_contexto": d.get("context_window"),
            "activo": d.get("active"),
        })
    return sorted(filas, key=lambda f: f["id"] or "")


def es_llama(model_id: str) -> bool:
    return "llama" in model_id.lower()


def filtrar_modelos(filas: list[dict], solo_gpt: bool) -> list[dict]:
    """Quita Llama (requisito) y modelos que no son de chat; opcionalmente deja solo GPT."""
    salida = [
        f for f in filas
        if f["id"] and not es_llama(f["id"]) and not any(x in f["id"].lower() for x in NO_CHAT)
    ]
    if solo_gpt:
        salida = [f for f in salida if "gpt" in f["id"].lower()]
    return salida


def llamar_groq(api_key, modelo, mensajes, params, extra):
    cliente = Groq(api_key=api_key)
    t0 = time.time()
    resp = cliente.chat.completions.create(
        model=modelo, messages=mensajes, extra_body=extra or None, **params
    )
    duracion = time.time() - t0
    msg = resp.choices[0].message
    return msg.content, getattr(msg, "reasoning", None), resp.usage, duracion


# ============================================================================
# Utilidades de tokenización
# ============================================================================
ESQUEMAS = [
    "Espacios en blanco",
    "Palabras y puntuación (regex)",
    "Caracteres",
    "Subpalabras BPE (tiktoken)",
]

DESCRIPCION_ESQUEMAS = {
    "Espacios en blanco": "Corta solo donde hay espacios. La puntuación queda pegada: `cocina.` y `cocina` serían tokens distintos.",
    "Palabras y puntuación (regex)": "Usa la expresión regular `\\w+|[^\\w\\s]`: separa palabras y signos de puntuación en tokens propios.",
    "Caracteres": "Cada carácter es un token. Vocabulario diminuto, pero secuencias muy largas.",
    "Subpalabras BPE (tiktoken)": "Byte Pair Encoding: el esquema real de los modelos GPT. Las palabras frecuentes son un token; las raras se parten en trozos.",
}

DESCRIPCION_BPE = {
    "o200k_harmony": "Tokenizador de los modelos gpt-oss (los GPT que ofrece Groq).",
    "o200k_base": "Tokenizador de GPT-4o (≈200 mil tokens de vocabulario).",
    "cl100k_base": "Tokenizador de GPT-4 y GPT-3.5 (≈100 mil tokens).",
    "p50k_base": "Tokenizador de Codex y text-davinci (≈50 mil tokens).",
    "r50k_base": "Tokenizador de GPT-3 original (≈50 mil tokens).",
}


@st.cache_resource(show_spinner="Descargando el tokenizador BPE (solo la primera vez)…")
def cargar_encoding(nombre: str):
    return tiktoken.get_encoding(nombre)


def encodings_disponibles() -> list[str]:
    if tiktoken is None:
        return []
    disponibles = set(tiktoken.list_encoding_names())
    return [e for e in DESCRIPCION_BPE if e in disponibles]


def tokenizar(texto: str, esquema: str, encoding: str | None = None):
    """Devuelve (lista_de_tokens_como_texto, lista_de_ids, nota_sobre_los_ids)."""
    if esquema == "Subpalabras BPE (tiktoken)":
        enc = cargar_encoding(encoding)
        ids = enc.encode(texto, disallowed_special=())
        piezas = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
        nota = (f"IDs reales del vocabulario de `{encoding}` ({enc.n_vocab:,} tokens). "
                "Si ves `�`, ese token es un fragmento de bytes de una letra con tilde o ñ: "
                "BPE trabaja sobre bytes UTF-8, no sobre letras.")
        return piezas, ids, nota

    if esquema == "Espacios en blanco":
        piezas = texto.split()
    elif esquema == "Palabras y puntuación (regex)":
        piezas = re.findall(r"\w+|[^\w\s]", texto)
    else:  # Caracteres
        piezas = list(texto)

    vocab = {tok: i for i, tok in enumerate(sorted(set(piezas)))}
    ids = [vocab[p] for p in piezas]
    nota = (f"IDs de un vocabulario construido SOLO con tu texto ({len(vocab)} tokens únicos, "
            "ordenados alfabéticamente). Un modelo real usa un vocabulario fijo aprendido de millones de textos.")
    return piezas, ids, nota


def html_tokens(piezas, ids, mostrar_ids: bool) -> str:
    """Pinta cada token con un color; los espacios se ven como ␣ y los saltos como ↵."""
    spans = []
    for k, (p, i) in enumerate(zip(piezas, ids)):
        color = PALETA[k % len(PALETA)]
        visible = html.escape(p).replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")
        sub = (f"<sub style='font-size:0.65em;color:#444;margin-left:2px'>{i}</sub>"
               if mostrar_ids else "")
        spans.append(
            f"<span title='id {i}' style='background:{color};color:#111;border-radius:4px;"
            f"padding:2px 4px;margin:2px;display:inline-block;font-family:monospace;"
            f"white-space:pre'>{visible}{sub}</span>"
        )
    return f"<div style='line-height:2.2'>{''.join(spans)}</div>"


def dividir_frases(texto: str, modo: str) -> list[str]:
    if modo == "Una por línea":
        partes = texto.splitlines()
    else:
        partes = re.split(r"(?<=[.!?])\s+|\n+", texto)
    return [p.strip() for p in partes if p and p.strip()]


def tokens_simples(frase: str, minusculas: bool) -> list[str]:
    frase = frase.lower() if minusculas else frase
    return re.findall(r"\w+", frase)


# ============================================================================
# Barra lateral: API key y texto de trabajo
# ============================================================================
with st.sidebar:
    st.header("🔑 Acceso a Groq")
    if Groq is None:
        st.error("Falta la librería `groq`. Instala las dependencias con `pip install -r requirements.txt`.")
        st.stop()

    clave = st.text_input("API Key de Groq", type="password",
                          value=st.session_state.get("api_key", ""),
                          help="Se obtiene en console.groq.com → API Keys. No se guarda en disco.")
    if st.button("Conectar", type="primary", use_container_width=True):
        try:
            with st.spinner("Validando la clave…"):
                listar_modelos(clave.strip())
            st.session_state["api_key"] = clave.strip()
            st.success("Conectado ✅")
        except Exception as e:
            st.session_state.pop("api_key", None)
            st.error(f"No se pudo conectar: {e}")

    if st.session_state.get("api_key"):
        if st.button("Desconectar", use_container_width=True):
            st.session_state.pop("api_key", None)
            listar_modelos.clear()
            st.rerun()

    st.divider()
    st.header("⚙️ Frases")
    modo_frases = st.radio("¿Cómo se separan las frases?",
                           ["Una por línea", "Por signos de puntuación"],
                           help="Se usa en Bolsa de palabras, Similitud y Esquema generativo.")

st.title("🧪 Laboratorio de PLN con Groq")

if not st.session_state.get("api_key"):
    st.info("👈 Para empezar, ingresa tu **API Key de Groq** en la barra lateral y pulsa **Conectar**.")
    st.stop()

API_KEY = st.session_state["api_key"]

# ----- Texto de trabajo (lo escribe el usuario a mano) -----
if "texto" not in st.session_state:
    st.session_state["texto"] = ""

col_t1, col_t2 = st.columns([5, 1])
with col_t2:
    st.write("")
    if st.button("Cargar ejemplo", use_container_width=True):
        st.session_state["texto"] = TEXTO_EJEMPLO
    if st.button("Limpiar", use_container_width=True):
        st.session_state["texto"] = ""
with col_t1:
    texto = st.text_area("✍️ Texto de trabajo (todo el análisis se hace sobre este texto)",
                         key="texto", height=160,
                         placeholder="Escribe aquí tus frases, idealmente una por línea…")

if not texto.strip():
    st.warning("Escribe un texto arriba (o pulsa **Cargar ejemplo**) para activar el análisis.")
    st.stop()

frases = dividir_frases(texto, modo_frases)

tabs = st.tabs([
    "🔤 Tokenización",
    "🎒 Bolsa de palabras",
    "📐 Similitud coseno",
    "🧬 Esquema generativo",
    "🤖 Generación con Groq",
    "📚 Catálogo de modelos",
])


# ============================================================================
# 1. Tokenización
# ============================================================================
with tabs[0]:
    st.subheader("Esquemas de tokenización")
    st.caption("Un modelo de lenguaje no ve letras ni palabras: ve una secuencia de números (IDs). "
               "La tokenización decide cómo se corta el texto antes de convertirlo en números.")

    c1, c2 = st.columns([2, 2])
    with c1:
        opciones = ESQUEMAS if tiktoken else ESQUEMAS[:-1]
        esquema = st.selectbox("Esquema", opciones, index=len(opciones) - 1)
        st.markdown(f"ℹ️ {DESCRIPCION_ESQUEMAS[esquema]}")
    encoding = None
    with c2:
        if esquema == "Subpalabras BPE (tiktoken)":
            encs = encodings_disponibles()
            encoding = st.selectbox("Vocabulario BPE", encs,
                                    format_func=lambda e: f"{e} — {DESCRIPCION_BPE[e]}")
        mostrar_ids = st.toggle("Mostrar el ID debajo de cada token", value=True)

    try:
        piezas, ids, nota = tokenizar(texto, esquema, encoding)
    except Exception as e:
        st.error(f"No se pudo cargar el vocabulario BPE (¿sin internet?): {e}. "
                 "Se muestra el esquema de palabras y puntuación en su lugar.")
        esquema = "Palabras y puntuación (regex)"
        piezas, ids, nota = tokenizar(texto, esquema)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Caracteres", len(texto))
    m2.metric("Tokens", len(ids))
    m3.metric("Tokens únicos", len(set(ids)))
    m4.metric("Caracteres por token", f"{len(texto) / max(len(ids), 1):.2f}")

    st.markdown("**Tokens coloreados** (pasa el cursor sobre un token para ver su ID)")
    st.markdown(html_tokens(piezas, ids, mostrar_ids), unsafe_allow_html=True)
    st.caption(nota)

    with st.expander("🔢 Secuencia de IDs (lo que realmente recibe el modelo)"):
        st.code(str(ids), language="python")

    with st.expander("📋 Tabla token por token"):
        st.dataframe(pd.DataFrame({
            "posición": range(len(ids)),
            "token": [repr(p) for p in piezas],
            "id": ids,
            "bytes UTF-8": [len(p.encode("utf-8")) for p in piezas],
        }), hide_index=True)

    st.markdown("#### Comparación de esquemas sobre el mismo texto")
    comparacion = []
    for esq in opciones:
        if esq == "Subpalabras BPE (tiktoken)":
            for e in encodings_disponibles():
                try:
                    p, i, _ = tokenizar(texto, esq, e)
                    comparacion.append({"esquema": f"BPE {e}", "tokens": len(i), "únicos": len(set(i))})
                except Exception:
                    pass
        else:
            p, i, _ = tokenizar(texto, esq)
            comparacion.append({"esquema": esq, "tokens": len(i), "únicos": len(set(i))})
    df_comp = pd.DataFrame(comparacion)
    st.plotly_chart(px.bar(df_comp, x="esquema", y="tokens", text="tokens",
                           title="Número de tokens según el esquema"))
    st.caption("Menos tokens = el modelo procesa el mismo texto más barato y más rápido "
               "(Groq cobra por token). Por eso los vocabularios BPE grandes son más eficientes.")


# ============================================================================
# 2. Bolsa de palabras
# ============================================================================
with tabs[1]:
    st.subheader("Bolsa de palabras (Bag of Words)")
    st.caption("Cada frase se convierte en un vector que cuenta cuántas veces aparece cada palabra "
               "del vocabulario. Se pierde el orden: «el perro muerde al hombre» y «el hombre muerde "
               "al perro» producen el mismo vector.")

    o1, o2, o3, o4 = st.columns(4)
    bow_min = o1.checkbox("Minúsculas", value=True)
    bow_stop = o2.checkbox("Quitar stopwords (español)", value=False)
    bow_bin = o3.checkbox("Binario (0/1)", value=False,
                          help="Solo marca presencia o ausencia, no cuántas veces.")
    bow_ngram = o4.selectbox("N-gramas", ["Unigramas (1)", "Uni + bigramas (1-2)"])
    rango = (1, 1) if bow_ngram == "Unigramas (1)" else (1, 2)
    ponderacion = st.radio("Ponderación", ["Conteo", "TF-IDF"], horizontal=True,
                           help="TF-IDF baja el peso de palabras que aparecen en casi todas las frases.")

    kwargs = dict(lowercase=bow_min, ngram_range=rango, token_pattern=r"(?u)\b\w+\b",
                  stop_words=STOPWORDS_ES if bow_stop else None)
    try:
        if ponderacion == "Conteo":
            vec = CountVectorizer(binary=bow_bin, **kwargs)
        else:
            vec = TfidfVectorizer(binary=bow_bin, **kwargs)
        X = vec.fit_transform(frases)
        vocab = vec.get_feature_names_out()
        etiquetas = [f"F{i + 1}" for i in range(len(frases))]
        df_bow = pd.DataFrame(X.toarray(), index=etiquetas, columns=vocab)

        st.markdown(f"**Vocabulario:** {len(vocab)} términos · **Frases (documentos):** {len(frases)}")
        with st.expander("Frases numeradas", expanded=True):
            for et, f in zip(etiquetas, frases):
                st.markdown(f"**{et}:** {f}")

        st.markdown("**Matriz documento-término** (filas = frases, columnas = palabras)")
        st.dataframe(df_bow.round(3) if ponderacion == "TF-IDF" else df_bow)

        with st.expander("🔢 Vocabulario con su índice de columna"):
            st.dataframe(pd.DataFrame(sorted(vec.vocabulary_.items(), key=lambda x: x[1]),
                                      columns=["término", "índice"]), hide_index=True)

        totales = df_bow.sum(axis=0).sort_values(ascending=False)
        top_n = st.slider("Términos a mostrar en la gráfica", 5, max(5, min(50, len(totales))),
                          min(15, max(5, len(totales))))
        df_top = totales.head(top_n).reset_index()
        df_top.columns = ["término", "peso"]
        st.plotly_chart(px.bar(df_top, x="término", y="peso",
                               title=f"Términos con mayor {ponderacion.lower()} en todo el texto"))
    except ValueError as e:
        st.error(f"No quedó vocabulario (quizá todas las palabras eran stopwords): {e}")


# ============================================================================
# 3. Similitud de coseno
# ============================================================================
with tabs[2]:
    st.subheader("Similitud de coseno entre frases")
    st.caption("cos(θ) = (A · B) / (‖A‖ · ‖B‖). Mide el ángulo entre dos vectores: 1 = misma dirección "
               "(mismas palabras en proporción), 0 = ninguna palabra en común.")

    if len(frases) < 2:
        st.warning("Necesitas al menos 2 frases. Escribe varias líneas o cambia el modo de separación.")
    else:
        rep = st.radio("Representación de las frases",
                       ["Conteo (BoW)", "Binario (BoW 0/1)", "TF-IDF"], horizontal=True)
        s_min = st.checkbox("Minúsculas ", value=True, key="cos_min")
        s_stop = st.checkbox("Quitar stopwords (español) ", value=False, key="cos_stop")
        kw = dict(lowercase=s_min, token_pattern=r"(?u)\b\w+\b",
                  stop_words=STOPWORDS_ES if s_stop else None)
        try:
            if rep == "TF-IDF":
                v = TfidfVectorizer(**kw)
            else:
                v = CountVectorizer(binary=(rep == "Binario (BoW 0/1)"), **kw)
            M = v.fit_transform(frases).toarray()
            terminos = v.get_feature_names_out()
            S = cosine_similarity(M)
            etiquetas = [f"F{i + 1}" for i in range(len(frases))]

            fig = px.imshow(S, x=etiquetas, y=etiquetas, text_auto=".2f", zmin=0, zmax=1,
                            color_continuous_scale="Blues", title="Matriz de similitud de coseno")
            st.plotly_chart(fig)

            with st.expander("Frases numeradas"):
                for et, f in zip(etiquetas, frases):
                    st.markdown(f"**{et}:** {f}")

            pares = [(S[i, j], i, j) for i in range(len(frases)) for j in range(i + 1, len(frases))]
            pares.sort(reverse=True)
            st.markdown("**Ranking de pares más parecidos**")
            st.dataframe(pd.DataFrame([{
                "par": f"F{i + 1} – F{j + 1}", "coseno": round(s, 4),
                "ángulo (°)": round(float(np.degrees(np.arccos(np.clip(s, -1, 1)))), 1),
                "frase A": frases[i], "frase B": frases[j]} for s, i, j in pares]),
                hide_index=True)

            st.markdown("#### 🔍 Cálculo paso a paso entre dos frases")
            a1, a2 = st.columns(2)
            ia = a1.selectbox("Frase A", range(len(frases)), format_func=lambda k: f"F{k + 1}: {frases[k]}")
            ib = a2.selectbox("Frase B", range(len(frases)), index=1,
                              format_func=lambda k: f"F{k + 1}: {frases[k]}")
            A, B = M[ia], M[ib]
            activos = (A != 0) | (B != 0)
            st.dataframe(pd.DataFrame({"término": terminos[activos], "A": A[activos], "B": B[activos],
                                       "A×B": (A * B)[activos]}).round(4), hide_index=True)
            punto = float(A @ B)
            nA, nB = float(np.linalg.norm(A)), float(np.linalg.norm(B))
            cos = punto / (nA * nB) if nA and nB else 0.0
            st.latex(rf"\cos(\theta)=\frac{{A\cdot B}}{{\|A\|\,\|B\|}}="
                     rf"\frac{{{punto:.4f}}}{{{nA:.4f}\times{nB:.4f}}}={cos:.4f}")
            st.caption("Solo se muestran los términos presentes en alguna de las dos frases: los demás "
                       "valen 0 en ambas y no aportan al producto punto. Limitación: con BoW, «perro» y "
                       "«can» cuentan como palabras totalmente distintas (no captura sinónimos).")
        except ValueError as e:
            st.error(f"No quedó vocabulario: {e}")


# ============================================================================
# 4. Esquema generativo (modelo de bigramas entrenado con gradiente)
# ============================================================================
def softmax(z, temperatura=1.0):
    z = z / max(temperatura, 1e-8)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entrenar_bigramas(secuencias, lr, epocas):
    """Aprende una matriz W (V×V) de logits con descenso de gradiente sobre la entropía cruzada."""
    vocab = sorted({t for s in secuencias for t in s})
    idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V, V))  # C[i, j] = veces que j sigue a i
    for s in secuencias:
        for a, b in zip(s, s[1:]):
            C[idx[a], idx[b]] += 1
    N = C.sum()
    n = C.sum(axis=1, keepdims=True)

    W = np.zeros((V, V))
    perdidas = []
    for _ in range(epocas):
        Z = W - W.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        perdidas.append(float(-(C * np.log(P + 1e-12)).sum() / N))
        grad = (n * P - C) / N          # derivada de la entropía cruzada respecto a W
        W -= lr * grad                  # ← aquí actúa el learning rate

    # Pérdida mínima teórica: la del modelo por conteo (máxima verosimilitud)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_mle = np.where(n > 0, C / n, 0)
        minimo = float(-(C[C > 0] * np.log(P_mle[C > 0])).sum() / N)
    return {"vocab": vocab, "idx": idx, "W": W, "perdidas": perdidas, "minimo": minimo}


with tabs[3]:
    st.subheader("Esquema generativo: predecir el siguiente token")
    st.markdown(
        "Un modelo generativo produce texto **un token a la vez**: mira el contexto, calcula una "
        "distribución de probabilidad sobre el vocabulario (logits → *softmax*), **muestrea** un token "
        "y lo agrega al contexto. Aquí entrenamos un modelo mínimo (bigramas: el contexto es solo el "
        "token anterior) con **tu texto**, usando descenso de gradiente, igual que se entrena un LLM "
        "pero a escala de juguete."
    )
    st.code("contexto → [modelo] → logits → softmax(logits / T) → muestreo → nuevo token → contexto + token → …",
            language="text")

    g1, g2, g3 = st.columns(3)
    unidad = g1.radio("Unidad de token", ["Palabras", "Caracteres"], horizontal=True)
    lr = g2.select_slider("Learning rate (tasa de aprendizaje)",
                          options=[0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], value=50)
    epocas = g3.slider("Épocas de entrenamiento", 10, 2000, 300, step=10)

    if unidad == "Palabras":
        secuencias = [["<ini>"] + tokens_simples(f, True) + ["<fin>"] for f in frases]
    else:
        secuencias = [["<ini>"] + list(f.lower()) + ["<fin>"] for f in frases]

    if st.button("🏋️ Entrenar modelo", type="primary"):
        st.session_state["bigramas"] = entrenar_bigramas(secuencias, lr, epocas)
        st.session_state["bigramas_cfg"] = (unidad, lr, epocas, texto, modo_frases)

    modelo_bg = st.session_state.get("bigramas")
    if modelo_bg is None:
        st.info("Pulsa **Entrenar modelo**. Prueba después con learning rates muy bajos y muy altos "
                "y compara las curvas de pérdida.")
    else:
        cfg = st.session_state["bigramas_cfg"]
        if cfg[0] != unidad or cfg[3] != texto or cfg[4] != modo_frases:
            st.warning("El texto o la unidad cambiaron desde el último entrenamiento: vuelve a entrenar.")
        st.caption(f"Entrenado con: unidad = {cfg[0]}, learning rate = {cfg[1]}, épocas = {cfg[2]}")

        perd = modelo_bg["perdidas"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Vocabulario", len(modelo_bg["vocab"]))
        k2.metric("Pérdida inicial", f"{perd[0]:.3f}")
        k3.metric("Pérdida final", f"{perd[-1]:.3f}", delta=f"{perd[-1] - perd[0]:.3f}", delta_color="inverse")
        k4.metric("Perplejidad final", f"{np.exp(perd[-1]):.2f}",
                  help="≈ entre cuántos tokens 'duda' el modelo en promedio. Menor es mejor.")
        df_loss = pd.DataFrame({"época": range(1, len(perd) + 1), "pérdida": perd,
                                "mínimo teórico": modelo_bg["minimo"]})
        st.plotly_chart(px.line(df_loss, x="época", y=["pérdida", "mínimo teórico"],
                                title="Curva de aprendizaje (entropía cruzada)"))
        st.caption("Learning rate bajo → la pérdida baja muy lento. Learning rate adecuado → se acerca al "
                   "mínimo teórico. Learning rate demasiado alto → cada paso se pasa del mínimo y la pérdida "
                   "oscila o incluso sube (prueba con 1000).")

        st.markdown("#### Distribución del siguiente token")
        d1, d2 = st.columns(2)
        vocab_bg, idx_bg, W = modelo_bg["vocab"], modelo_bg["idx"], modelo_bg["W"]
        previo = d1.selectbox("Token anterior (contexto)", vocab_bg,
                              index=vocab_bg.index("<ini>") if "<ini>" in vocab_bg else 0)
        temp_bg = d2.slider("Temperatura (T)", 0.05, 3.0, 1.0, 0.05,
                            help="T<1 afila la distribución (más predecible); T>1 la aplana (más aleatoria).")
        probs = softmax(W[idx_bg[previo]], temp_bg)
        orden = np.argsort(probs)[::-1][:15]
        df_p = pd.DataFrame({"token": [repr(vocab_bg[i]) for i in orden], "probabilidad": probs[orden]})
        st.plotly_chart(px.bar(df_p, x="token", y="probabilidad",
                               title=f"P(siguiente | «{previo}») con T = {temp_bg}"))

        st.markdown("#### Generar una secuencia")
        e1, e2, e3 = st.columns(3)
        max_len = e1.slider("Longitud máxima", 5, 100, 20)
        top_k = e2.slider("Top-k (0 = sin límite)", 0, 20, 0,
                          help="Solo se muestrea entre los k tokens más probables.")
        semilla = e3.number_input("Semilla", 0, 9999, 42)
        if st.button("✨ Generar"):
            rng = np.random.default_rng(int(semilla))
            actual, pasos, salida = "<ini>" if "<ini>" in idx_bg else vocab_bg[0], [], []
            for _ in range(max_len):
                p = softmax(W[idx_bg[actual]], temp_bg)
                if top_k > 0:
                    corte = np.argsort(p)[::-1][top_k:]
                    p[corte] = 0
                    p /= p.sum()
                j = int(rng.choice(len(p), p=p))
                siguiente = vocab_bg[j]
                pasos.append({"contexto": actual, "elegido": siguiente, "probabilidad": round(float(p[j]), 4)})
                if siguiente == "<fin>":
                    break
                salida.append(siguiente)
                actual = siguiente
            sep = " " if unidad == "Palabras" else ""
            st.success(sep.join(salida) or "(el modelo terminó de inmediato)")
            st.dataframe(pd.DataFrame(pasos), hide_index=True)


# ============================================================================
# 5. Generación con Groq
# ============================================================================
with tabs[4]:
    st.subheader("Generación de respuestas con modelos de Groq")

    try:
        catalogo = listar_modelos(API_KEY)
    except Exception as e:
        catalogo = []
        st.warning(f"No se pudo leer el catálogo ({e}). Se usan modelos de respaldo.")

    solo_gpt_gen = st.toggle("Solo modelos GPT", value=True, key="gen_solo_gpt")
    ids_modelos = [f["id"] for f in filtrar_modelos(catalogo, solo_gpt_gen)] or MODELOS_RESPALDO
    modelo = st.selectbox("Modelo", ids_modelos)

    with st.expander("⚙️ Parámetros de generación", expanded=True):
        p1, p2, p3 = st.columns(3)
        temperatura = p1.slider("Temperatura", 0.0, 2.0, 0.7, 0.05,
                                help="Divide los logits antes del softmax (igual que en la pestaña anterior).")
        top_p = p2.slider("Top-p (nucleus sampling)", 0.0, 1.0, 1.0, 0.05,
                          help="Solo muestrea entre los tokens cuya probabilidad acumulada llega a p.")
        max_tokens = p3.number_input("Máx. tokens de salida", 16, 32768, 1024, step=64)

        q1, q2, q3 = st.columns(3)
        usar_semilla = q1.checkbox("Fijar semilla", value=False,
                                   help="Con la misma semilla y parámetros, la respuesta tiende a repetirse.")
        seed = q1.number_input("Semilla ", 0, 2**31 - 1, 42, disabled=not usar_semilla)
        stops_txt = q2.text_input("Secuencias de parada (separadas por |)", "",
                                  help="El modelo deja de generar al producir alguna. Máximo 4.")
        es_gpt_oss = "gpt-oss" in modelo.lower()
        esfuerzo = q3.selectbox("Esfuerzo de razonamiento", ["low", "medium", "high"], index=1,
                                disabled=not es_gpt_oss,
                                help="Solo en gpt-oss: cuánto 'piensa' antes de responder.")
        ver_razonamiento = q3.checkbox("Mostrar razonamiento", value=False, disabled=not es_gpt_oss)

        st.info("**¿Y el learning rate?** No aparece aquí porque no es un parámetro de *generación*: "
                "es un hiperparámetro de *entrenamiento*, y los modelos de Groq ya vienen entrenados "
                "(sus pesos están fijos). Puedes experimentar con él en la pestaña **🧬 Esquema generativo**.")

    sistema = st.text_area("Mensaje de sistema (rol del asistente)",
                           "Eres un asistente experto en procesamiento de lenguaje natural. Responde en español.",
                           height=80)
    usar_contexto = st.checkbox("Incluir el texto de trabajo como contexto", value=True)
    prompt = st.text_area("Prompt", placeholder="Ej.: Resume el texto y explica qué temas comparten las frases.",
                          height=100)

    modo_gen = st.radio("Modo", ["Una respuesta", "Comparar temperaturas"], horizontal=True)
    temps_comp = []
    if modo_gen == "Comparar temperaturas":
        temps_comp = st.multiselect("Temperaturas a comparar", [0.0, 0.3, 0.7, 1.0, 1.3, 1.6, 2.0],
                                    default=[0.0, 0.7, 1.3])

    if prompt.strip() and tiktoken:
        try:
            n_prompt = len(cargar_encoding("o200k_base").encode(
                sistema + (texto if usar_contexto else "") + prompt, disallowed_special=()))
            st.caption(f"Tokens aproximados de entrada: ~{n_prompt} (o200k_base)")
        except Exception:
            pass

    if st.button("🚀 Generar respuesta", type="primary", disabled=not prompt.strip()):
        contenido = (f"Texto de referencia:\n\"\"\"\n{texto}\n\"\"\"\n\n{prompt}" if usar_contexto else prompt)
        mensajes = [{"role": "system", "content": sistema}, {"role": "user", "content": contenido}]

        base = {"top_p": top_p, "max_completion_tokens": int(max_tokens)}
        if usar_semilla:
            base["seed"] = int(seed)
        stops = [s for s in (x.strip() for x in stops_txt.split("|")) if s][:4]
        if stops:
            base["stop"] = stops
        extra = {"reasoning_effort": esfuerzo, "include_reasoning": ver_razonamiento} if es_gpt_oss else {}

        lista_temps = temps_comp if modo_gen == "Comparar temperaturas" else [temperatura]
        if not lista_temps:
            st.warning("Elige al menos una temperatura.")
        columnas = st.columns(len(lista_temps)) if lista_temps else []
        for col, t in zip(columnas, lista_temps):
            with col:
                st.markdown(f"**Temperatura = {t}**")
                try:
                    with st.spinner("Generando…"):
                        texto_resp, razon, uso, dur = llamar_groq(
                            API_KEY, modelo, mensajes, {**base, "temperature": t}, extra)
                    if razon:
                        with st.expander("🧠 Razonamiento del modelo"):
                            st.write(razon)
                    st.markdown(texto_resp or "_(respuesta vacía: prueba subir el máximo de tokens)_")
                    if uso:
                        st.caption(f"⏱️ {dur:.2f} s · entrada {uso.prompt_tokens} · salida "
                                   f"{uso.completion_tokens} · total {uso.total_tokens} tokens")
                except Exception as e:
                    st.error(f"Error de Groq: {e}")


# ============================================================================
# 6. Catálogo de modelos
# ============================================================================
with tabs[5]:
    st.subheader("Catálogo de modelos de Groq (sin Llama)")
    try:
        catalogo = listar_modelos(API_KEY)
        solo_gpt_cat = st.toggle("Solo modelos GPT", value=True, key="cat_solo_gpt")
        visibles = filtrar_modelos(catalogo, solo_gpt_cat)
        ocultos_llama = sum(es_llama(f["id"] or "") for f in catalogo)
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelos en tu cuenta", len(catalogo))
        c2.metric("Mostrados", len(visibles))
        c3.metric("Llama excluidos", ocultos_llama)
        if visibles:
            df_cat = pd.DataFrame(visibles)
            df_cat["ventana_contexto"] = df_cat["ventana_contexto"].map(
                lambda x: f"{x:,}" if isinstance(x, (int, float)) and x else "—")
            st.dataframe(df_cat, hide_index=True)
        else:
            st.info("No hay modelos que cumplan el filtro.")
        st.caption("El catálogo se consulta en vivo a la API (`client.models.list()`), así que refleja los "
                   "modelos que Groq ofrece hoy. Se excluyen Llama y los modelos de audio/voz.")
        if st.button("🔄 Actualizar catálogo"):
            listar_modelos.clear()
            st.rerun()
    except Exception as e:
        st.error(f"No se pudo consultar el catálogo: {e}")
