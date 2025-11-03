# Text Summarizer

Create high‑quality summaries from PDFs or pasted text, optionally translate them, and keep a clean, filterable history of results. This app features fast, robust LLM routing with automatic fallbacks as well as a lightweight vector store to make summaries easy to review and export.

## Features

- Summarize PDFs or pasted text instantly (long files are handled automatically).
- Translate summaries into multiple languages in one click.
- View your entire summary history in a sortable/filterable table.
- Lightweight local vector database for instant history search (no heavy dependencies).

## How It Works

- **Fast LLM Summarization:** All requests are routed to your preferred language model with smart fallbacks—retrying with alternative models if rate-limited or unavailable. Exponential backoff and efficient batching minimize delays.
- **Map-Reduce when Needed:** For very long documents, text is chunked, summarized in parallel, then recombined to produce a final summary that preserves structure and meaning.
- **One-Row History:** Each run stores just a single, clean row (the final summary), linked to original text for preview and download.
- **FAISS Vector Database:** Summaries are encoded using TF-IDF and stored in FAISS, making history searches fast and hardware-friendly (CPU-only, no Torch/ONNX required).

## Technologies Used

- **Streamlit:** Rapid UI/app development with interactive widgets and live previews.
- **Prompt Engineering:** Carefully designed prompts guide the LLM to produce clear, reliable, and concise results, minimizing hallucination.
- **LLM Routing:** OpenRouter’s fallback system directs requests to the fastest working model, reducing API errors and wait time.
- **FAISS:** High-performance vector database stores summary features locally for rapid history filtering and future semantic search.

## Quickstart

1. **Install requirements**
    ```
    pip install -r requirements.txt
    ```

2. **Configure environment**
    - Create a `.env` file with your OpenRouter API key like:
      ```
      OPENROUTER_API_KEY=sk-or-xxxxx
      ```

3. **Run**
    ```
    streamlit run streamlit_app.py
    ```

4. **Use**
    - Upload a PDF or paste text, optionally set translations, and click "Generate summary". Visit the "History" page to review all results and filter as needed.

## Repository Layout

- `streamlit_app.py` — Page navigation (Summarizer & History)
- `pages/text_app.py` — Main summarizer UI and logic
- `pages/history.py` — Full history table view, filtering, and exports
- `faiss_store/` — (auto-created) stores vector DB/index, metadata, and summary texts

## Why This Stack?

This project combines smart LLM orchestration, prompt design, and a minimal open-source vector database for speed, robustness, and excellent UX. It avoids all heavy GPU/torch/onnx dependencies, works reliably across platforms, and keeps all summaries easily accessible for future reference or export.

---
