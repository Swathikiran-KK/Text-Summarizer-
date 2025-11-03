import os
import io
import time
import json
import hashlib
import random
import datetime as dt
import requests
import streamlit as st
import pdfplumber
import numpy as np
import faiss

# Optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Token chunking
try:
    import tiktoken
except Exception:
    tiktoken = None

# TF‑IDF (pure CPU)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib

APP_VERSION = "1.5.0"

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Summarizer", page_icon="📝", layout="wide")
st.markdown("## 📝 Summarizer")
st.caption("Create high-quality summaries from PDFs or pasted text, then optionally translate the result into selected languages. Your results are saved to History for easy review and export.")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PRIMARY_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")  # preferred
FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "qwen/qwen2.5-7b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "openrouter/auto",
]

FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_store")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "history.index")
FAISS_META_PATH  = os.path.join(FAISS_DIR, "history_meta.json")
TFIDF_PATH       = os.path.join(FAISS_DIR, "tfidf_vectorizer.joblib")
FAISS_TEXTS_PATH = os.path.join(FAISS_DIR, "history_texts.json")
os.makedirs(FAISS_DIR, exist_ok=True)

SESSION = requests.Session()

# ---------------------------
# Helpers
# ---------------------------
def token_enc():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None  

def est_tokens(text: str):
    tok = token_enc()
    return len(tok.encode(text)) if tok else max(1, len(text)//4) 
def chunk_text(text: str, target_tokens=1600, overlap_tokens=200):
    tok = token_enc()
    if tok is None:
        avg_char = 4
        size = target_tokens * avg_char
        overlap = overlap_tokens * avg_char
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i+size])
            i += max(1, size - overlap)
        return out
    toks = tok.encode(text)
    out, start = [], 0
    step = max(1, target_tokens - overlap_tokens)
    while start < len(toks):
        end = min(start + target_tokens, len(toks))
        out.append(tok.decode(toks[start:end]))
        start += step
    return out  
@st.cache_data
def extract_text_from_pdf(file_bytes: bytes):
    buf = io.BytesIO(file_bytes)
    pages = []
    page_count = 0
    with pdfplumber.open(buf) as pdf:
        page_count = len(pdf.pages)
        for p in pdf.pages:
            t = p.extract_text() or ""
            pages.append(t)
    return "\n".join(pages), page_count

def _post_chat(messages, model, max_tokens=1200, temperature=0.2, retries=6, base_sleep=1.0):
    if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.startswith("sk-or-"):
        raise RuntimeError("OPENROUTER_API_KEY missing/invalid.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Summarizer",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "routing": {"strategy": "speed"},
        "extra_body": {"provider": {"allow_fallbacks": True}},
    }
    backoff = base_sleep
    last_err = None
    for _ in range(retries):
        resp = SESSION.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            wait = float(ra) if ra else backoff + random.uniform(0, 0.75)
            time.sleep(wait)
            backoff = min(backoff * 2, 16.0)
            last_err = resp.text
            continue
        last_err = f"{resp.status_code}: {resp.text}"
        break
    raise RuntimeError(f"Chat failed for model={model}. Last error: {last_err}") 

def chat_with_fallback(messages, max_tokens=1200, temperature=0.2):
    try:
        return _post_chat(messages, PRIMARY_MODEL, max_tokens=max_tokens, temperature=temperature)
    except RuntimeError:
        pass
    for fm in FALLBACK_MODELS:
        try:
            return _post_chat(messages, fm, max_tokens=max_tokens, temperature=temperature)
        except RuntimeError:
            continue
    raise RuntimeError("All models failed after fallbacks.")  

def summarize_once(text: str, style: str, max_tokens=900):
    targets = {
        "Brief": "3-5 bullet points",
        "Medium": "1-2 short paragraphs",
        "Detailed": "3-5 organized paragraphs",
    }
    sys = {"role": "system", "content": "You are a careful, faithful summarizer. Do not invent facts."}
    user = {"role": "user", "content": f"Summarize the text in {targets.get(style,'1-2 short paragraphs')}, focusing on key points.\n\nText:\n{text}"}
    return chat_with_fallback([sys, user], max_tokens=max_tokens, temperature=0.1) 

def reduce_summaries(mapped, style: str):
    target = {
        "Brief": "3-4 concise bullet points",
        "Medium": "1-2 cohesive paragraphs",
        "Detailed": "3-5 organized paragraphs",
    }[style]
    joined = "\n\n---\n\n".join(mapped)
    sys = {"role": "system", "content": "You combine partial summaries into a single, non-redundant final summary."}
    user = {"role": "user", "content": f"Produce {target} based on the partial summaries below.\n\n{joined}"}
    return chat_with_fallback([sys, user], max_tokens=900, temperature=0.1) 

def translate_batch(summary: str, languages: list[str]):
    if not languages:
        return ""
    lang_list = ", ".join(languages)
    sys = {"role": "system", "content": "You are a precise translator. Preserve meaning and tone."}
    user = {"role": "user", "content": f"Translate the following into these languages: {lang_list}.\nReturn as sections labeled with the language name.\n\n{summary}"}
    return chat_with_fallback([sys, user], max_tokens=1600, temperature=0.2)  
# -------- FAISS + TF‑IDF (cosine via IP on L2-normalized dense vectors) --------
def load_faiss_index(path: str):
    if os.path.exists(path):
        return faiss.read_index(path)
    return None  

def save_faiss_index(index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path) 

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_or_init_tfidf():
    if os.path.exists(TFIDF_PATH):
        try:
            return joblib.load(TFIDF_PATH)
        except Exception:
            pass
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5),
        lowercase=True,
        max_features=30000,
        strip_accents="unicode",
        sublinear_tf=True,
    )  

def ensure_index(dim: int):
    index = load_faiss_index(FAISS_INDEX_PATH)
    if index is None:
        index = faiss.IndexFlatIP(dim)
        save_faiss_index(index, FAISS_INDEX_PATH)
    meta = load_json(FAISS_META_PATH, {"items": []})
    texts = load_json(FAISS_TEXTS_PATH, {"items": []})
    return index, meta, texts  
def tfidf_embed(texts: list[str], vec: TfidfVectorizer, fit: bool = False) -> np.ndarray:
    if fit:
        X = vec.fit_transform(texts)
        joblib.dump(vec, TFIDF_PATH)
    else:
        try:
            X = vec.transform(texts)
        except Exception:
            X = vec.fit_transform(texts)
            joblib.dump(vec, TFIDF_PATH)
    arr = X.toarray().astype(np.float32)
    arr = normalize(arr, norm="l2", copy=False)
    return arr 

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

# ---------------------------
# UI
# ---------------------------
hdr_l, hdr_r = st.columns([3,1])
with hdr_l:
    st.markdown("### 📄 Input and Options")
with hdr_r:
    st.page_link("pages/history.py", label="Open History 🗂️", icon="🗂️", use_container_width=True)  

left, right = st.columns([2,1], gap="large")
with left:
    up = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], help="Upload a PDF or a plain text file.")
    pasted = st.text_area("Or paste text", height=220, placeholder="Paste content here...")
with right:
    style = st.radio("Summary length", ["Brief", "Medium", "Detailed"], index=1, horizontal=True)
    langs = st.multiselect("Translate to", ["Hindi", "Malayalam", "Tamil", "Telugu", "Kannada", "French", "Spanish"])
    run = st.button("Summarize", type="primary", use_container_width=True)

status = st.container()
sum_box = st.container()
trans_box = st.container()

if run:
    if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.startswith("sk-or-"):
        st.error("OPENROUTER_API_KEY missing/invalid.")
        st.stop()

    with status:
        st.info("Reading input...")
    raw, title, filetype, file_bytes, page_count, source = "", "", "", 0, 0, "pasted"
    if up is not None:
        file_bytes = len(up.getvalue())
        if up.type == "application/pdf":
            text, page_count = extract_text_from_pdf(up.getvalue())
            raw = text
            title = up.name
            filetype = "pdf"
            source = "upload"
        elif up.type == "text/plain":
            raw = up.getvalue().decode("utf-8", errors="ignore")
            title = up.name
            filetype = "txt"
            source = "upload"
    if not raw and pasted and pasted.strip():
        raw = pasted.strip()
        title = "pasted_text"
        filetype = "text"
        source = "pasted"
    if not raw:
        with status:
            st.warning("Provide a PDF/TXT or paste text.")
        st.stop()

    start_time = time.time()
    total_tokens = est_tokens(raw)
    with status:
        st.info(f"Estimated tokens: ~{total_tokens}")  
    # Single pass for short inputs; simple map-reduce for long inputs
    if total_tokens <= 2800:
        final_summary = summarize_once(raw, style, max_tokens=900)
        chunk_count = 1
    else:
        chunks = chunk_text(raw, target_tokens=1600, overlap_tokens=200)
        mapped = [summarize_once(ch, style, max_tokens=450) for ch in chunks]
        final_summary = reduce_summaries(mapped, style)
        chunk_count = len(chunks)

    duration_ms = int((time.time() - start_time) * 1000)

    with sum_box:
        st.markdown("### ✅ Final Summary")
        st.write(final_summary)

    translations_done = []
    if langs:
        with trans_box:
            st.markdown("### 🌐 Translations")
            try:
                batched = translate_batch(final_summary, langs)
                st.write(batched)
                translations_done = list(langs)
            except Exception as e:
                st.warning(f"Translation error: {e}")

    # Index only the FINAL SUMMARY (one history row per run)
    try:
        vec = load_or_init_tfidf()
        fit_needed = not os.path.exists(TFIDF_PATH)
        arr = tfidf_embed([final_summary], vec, fit=fit_needed)
        dim = arr.shape[1]
        index, meta_store, texts_store = ensure_index(dim)
        start_id = index.ntotal
        index.add(arr)
        now_iso = dt.datetime.utcnow().isoformat()
        ts = int(time.time())
        doc_hash = sha256_text(raw)
        summary_chars = len(final_summary)
        summary_words = len(final_summary.split())
        meta = {
            "type": "summary",
            "title": title,
            "unix_ts": ts,
            "iso_time": now_iso,
            "local_time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": title,
            "filetype": filetype,
            "file_bytes": file_bytes,
            "page_count": page_count,
            "source": source,
            "style": style,
            "chunk_count": chunk_count,
            "translated_to": translations_done,
            "model": PRIMARY_MODEL,
            "embed_kind": "tfidf",
            "vector_dim": dim,
            "doc_hash": doc_hash,
            "text_chars": summary_chars,
            "text_words": summary_words,
            "duration_ms": duration_ms,
            "app_version": APP_VERSION,
        }
        meta_store["items"].append({"id": start_id, "meta": meta})
        # Keep both texts in sidecar for preview/download if needed
        texts_store["items"].append({"id": start_id, "type": "document", "text": raw})
        texts_store["items"].append({"id": start_id, "type": "summary", "text": final_summary})
        save_faiss_index(index, FAISS_INDEX_PATH)
        save_json(meta_store, FAISS_META_PATH)
        save_json(texts_store, FAISS_TEXTS_PATH)
        with status:
            st.success("Saved to History. You can review, filter, and export results anytime")
    except Exception as e:
        with status:
            st.warning(f"History save/index error: {e}") 
