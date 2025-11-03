import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="History", page_icon="🗂️", layout="wide")
st.markdown("## 🗂️ History")
st.caption("Browse all your summaries in one place, use optional filters to narrow results.")

FAISS_DIR = os.getenv("FAISS_DIR", "./faiss_store")
FAISS_META_PATH  = os.path.join(FAISS_DIR, "history_meta.json")
FAISS_TEXTS_PATH = os.path.join(FAISS_DIR, "history_texts.json")

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

meta = load_json(FAISS_META_PATH, {"items": []})

if not meta["items"]:
    st.info("No history found yet. Run a summary on the main page first.")
    st.stop()

# Build DataFrame: only summary rows are stored now, so one row per run
rows = []
for rec in meta["items"]:
    m = rec.get("meta", {})
    rows.append({
        "id": rec.get("id"),
        "time": m.get("local_time") or m.get("iso_time") or "",
        "unix_ts": int(m.get("unix_ts", 0)),
        "title": m.get("title",""),
        "source": m.get("source",""),           # upload / pasted
        "style": m.get("style",""),             # Brief / Medium / Detailed
        "translated_to": ", ".join(m.get("translated_to", [])) if isinstance(m.get("translated_to"), list) else (m.get("translated_to","") or ""),
        "page_count": m.get("page_count",""),
        "chunk_count": m.get("chunk_count",""),
        "model": m.get("model",""),
        "duration_ms": m.get("duration_ms",""),
        "filetype": m.get("filetype",""),
        "file_bytes": m.get("file_bytes",""),
    })

df_all = pd.DataFrame(rows).sort_values("unix_ts", ascending=False)

with st.sidebar:
    st.markdown("### Filters (optional)")
    type_sel = st.multiselect("Source", ["upload","pasted"], default=[])
    style_sel = st.multiselect("Style", ["Brief","Medium","Detailed"], default=[])
    lang_text = st.text_input("Translated contains (comma-separated)", value="")
    min_ts = st.number_input("Min unix ts (0=ignore)", min_value=0, value=0, step=1)
    max_ts = st.number_input("Max unix ts (0=ignore)", min_value=0, value=0, step=1)

# Apply filters only if user selected something
df = df_all.copy()
if type_sel:
    df = df[df["source"].isin(type_sel)]
if style_sel:
    df = df[df["style"].isin(style_sel)]
if lang_text.strip():
    langs = [x.strip().lower() for x in lang_text.split(",") if x.strip()]
    if langs:
        df = df[df["translated_to"].str.lower().apply(lambda s: any(l in s for l in langs))]
if min_ts:
    df = df[df["unix_ts"] >= int(min_ts)]
if max_ts:
    df = df[df["unix_ts"] <= int(max_ts)]

column_order = ["time","title","source","style","translated_to","page_count","chunk_count","model","duration_ms","filetype","file_bytes","id","unix_ts"]

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_order=column_order,
)
