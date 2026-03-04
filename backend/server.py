# server.py
"""
FastAPI wrapper around SokratesApp.

Run:
  pip install fastapi uvicorn
  uvicorn server:app --host 0.0.0.0 --port 8000

POST /api/chat
  { "user_text": "Hallo" }
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from sokrates_app import SokratesApp

app = FastAPI()

# Adjust origins to your website domain later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SokratesApp(model="deepseek-r1:8b")


class ChatBody(BaseModel):
    user_text: str
    settings: Optional[Dict[str, Any]] = None  # reserved for later


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatBody):
    return engine.handle_turn(body.user_text)
