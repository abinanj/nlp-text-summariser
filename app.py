from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from summarizer import summarize_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Text Summarizer API")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class SummarizeRequest(BaseModel):
	text: str
	max_length: int = 130
	min_length: int = 30


@app.post("/summarize")
async def summarize(req: SummarizeRequest) -> Dict[str, str]:
	summary = summarize_text(req.text, max_length=req.max_length, min_length=req.min_length)
	return {"summary": summary}


@app.get("/")
async def root() -> Dict[str, str]:
	return {"message": "Summarizer running. POST /summarize"}


