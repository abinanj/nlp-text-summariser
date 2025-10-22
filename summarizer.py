from typing import Optional

try:
	from transformers import pipeline
	summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
	summarizer_pipeline = None


def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
	text = (text or "").strip()
	if not text:
		return ""
	if summarizer_pipeline is None:
		# naive fallback: return first N sentences clipped
		sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
		if not sentences:
			return text[:max_length]
		return ". ".join(sentences[:3])[: max_length + 50]
	result = summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
	return result[0]["summary_text"].strip()


