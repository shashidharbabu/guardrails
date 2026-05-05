"""
Enterprise Guardrails — App Backend
Run: uvicorn app.backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Make sure multi_agent_debate imports work
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "multi_agent_debate"))

from dotenv import load_dotenv
load_dotenv(_repo_root / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend import db
from app.backend.routers import gateway, sessions, analytics, feedback
from app.backend.routers.sessions import query_router

app = FastAPI(
    title="Enterprise Guardrails — App Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(gateway.router)
app.include_router(sessions.router)
app.include_router(query_router)
app.include_router(analytics.router)
app.include_router(feedback.router)


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok", "service": "enterprise-guardrails-app"}
