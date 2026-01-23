from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
import shutil
import os
import uuid
from ingestion import ingest_cv
from retrieve import answer_with_context
from semantic_cache import get_semantic_cache, set_semantic_cache
from embedding import embeddings
from fastapi.middleware.cors import CORSMiddleware
from semantic_cache import clear_session_cache

app = FastAPI(title="Sanjib Shah Portfolio Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sanjibshah.com.np"],
    allow_methods=["*"],
    allow_headers=["*"],
)



UPLOAD_DIR = "dataset"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    question: str



@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print(file_path)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_cv(file_path)

    return {"status": "success", "message": f"{file.filename} ingested successfully"}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    session_id = request.cookies.get("session_id")

    
    cached = get_semantic_cache(session_id, req.question, embeddings)
    if cached:
        return {"answer": cached, "cached": True}

   
    answer = answer_with_context(req.question)


    set_semantic_cache(session_id, req.question, answer, embeddings)

    return {"answer": answer, "cached": False}



@app.post("/clear-session")
def clear_session(req: dict):
    clear_session_cache(req["session_id"])
    return {"status": "cleared"}


