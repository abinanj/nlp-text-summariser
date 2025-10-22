# Text Summarizer (FastAPI + Streamlit)

Minimal summarizer using Hugging Face Transformers (BART). Includes a FastAPI endpoint and a Streamlit UI.

## Quickstart

```bash
pip install -r requirements.txt

# Run API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run UI
python -m streamlit run frontend.py
```

## API
- POST `/summarize`
```json
{
  "text": "long text here ...",
  "max_length": 130,
  "min_length": 30
}
```

## Notes
- If GPU is available and torch is installed with CUDA, the pipeline will leverage it.
- If the transformer pipeline cannot initialize (no internet or incompatible env), a naive sentence-based fallback is used.


