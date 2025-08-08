git add main.py && git commit -m "health routes + lazy llm + POST auth" && git push origin main


from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests, tempfile, hashlib, threading
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# CONFIG
# =========================
BEARER_TOKEN = "ac96cb4db56939ddd84c8b78c7ac5eb9f288404f64af12fdf4d5aed51d1e3218"

app = FastAPI(title="HackRx Intelligent Query–Retrieval API", version="1.2")

# =========================
# Health / Info (GET)
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "hint": "POST JSON to /api/v1/hackrx/run with Authorization: Bearer <token>"
    }

@app.get("/hackrx/run")
@app.get("/api/v1/hackrx/run")
def info():
    return {
        "message": "Use POST with Authorization: Bearer <token>",
        "schema": {"documents": "<url>", "questions": ["q1","q2","..."]}
    }

# =========================
# Auth (POST only)
# =========================
@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.method == "POST" and request.url.path.endswith("/hackrx/run"):
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer Token")
        if auth.split(" ")[1] != BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid Token")
    return await call_next(request)

# =========================
# Schema
# =========================
class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

# =========================
# Utils: download + retriever cache
# =========================
def _download_pdf(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(tmp.name, "wb") as f:
        f.write(r.content)
    return tmp.name

_RETRIEVER_CACHE: Dict[str, any] = {}

def _build_retriever_from_url(pdf_url: str):
    key = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]
    path = _download_pdf(pdf_url)
    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=160)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    _RETRIEVER_CACHE[key] = retriever
    return retriever

# =========================
# Lazy LLM (loads on first use)
# =========================
_LLM = None
_LLM_LOCK = threading.Lock()

def _load_llm():
    global _LLM
    with _LLM_LOCK:
        if _LLM is not None:
            return _LLM
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline("text-generation", model=mdl, tokenizer=tok, max_length=512)
        _LLM = HuggingFacePipeline(pipeline=gen)
        return _LLM

# =========================
# Prompt + QA helper
# =========================
PROMPT = PromptTemplate(
    template=(
        "You are an expert in insurance policies. Answer ONLY from context. "
        "If not in context, say \"Not found in policy context.\" "
        "Question: {question}\nContext: {context}\nFinal Answer (one sentence):"
    ),
    input_variables=["question", "context"],
)

def _answer_batch(retriever, llm, questions: List[str]) -> List[str]:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
    )
    out = []
    for q in questions:
        res = qa({"query": q})
        out.append(res["result"].strip())
    return out

# =========================
# Endpoint
# =========================
@app.post("/hackrx/run")
@app.post("/api/v1/hackrx/run")
def run_hackrx(req: HackRxRequest):
    retriever = _build_retriever_from_url(req.documents)
    llm = _load_llm()
    answers = _answer_batch(retriever, llm, req.questions)
    return {"answers": answers}
