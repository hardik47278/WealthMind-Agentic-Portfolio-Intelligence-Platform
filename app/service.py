"""FastAPI service exposing the contract the harness/server expects:

    GET  /health   -> 200 once ready
    GET  /agents   -> your roster (schema/agents.schema.json)
    POST /answer   -> one question envelope in, one answer object out

Reads BOOK_PATH, MARKET_PATH, LLM_BASE_URL, LLM_API_KEY, PORT from the
environment. Loads the book and market once at startup and holds them in
memory -- they do not change during a run. The compliance classifier and
router agents are also built once at startup (see app/router.py and
app/compliance.py docstrings for why -- neither depends on client-specific
data, so building them per-question would waste latency/tokens for no
benefit).

Run locally:
    uvicorn app.service:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import Question, Roster, AgentDecl, abstain
from app.team import answer_question
from app.compilance_agent import build_ambiguity_classifier
from app.router import build_router_agent
from app.router import build_router_agent, build_strong_router_agent
from app.verifier import build_abstention_judge, build_call_relevance_classifier

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOOK_PATH = os.environ.get("BOOK_PATH", "data/client_book.json")
MARKET_PATH = os.environ.get("MARKET_PATH", "data/market_data.json")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8600")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "dummy")
PORT = int(os.environ.get("PORT", "8080"))

# Model ids used for every agent build in this service. Default to the
# real gateway's two tiers ("valura-fast" / "valura-deep") so a clean
# environment (no overrides) targets the grading gateway correctly.
# For local testing against OpenAI directly, override these via env:
#   FAST_MODEL=gpt-4.1-mini  DEEP_MODEL=gpt-4.1
FAST_MODEL = os.environ.get("FAST_MODEL", "valura-fast")
DEEP_MODEL = os.environ.get("DEEP_MODEL", "valura-deep")


def _llm_v1_url() -> str:
    """LLM_BASE_URL as given by the environment may or may not already
    end in /v1 (compose.grading.yml sets it to .../v1 directly; local
    dev might set it without). Normalise once, here, instead of
    repeating the same ternary at every call site."""
    base = LLM_BASE_URL.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


app = FastAPI(title="valura-ai-arena-service")

STATE: dict = {
    "book": None,
    "market": None,
    "clients_by_id": {},
    "classifier_agent": None,
    "router_agent": None,
    "strong_router_agent":None,
    "abstention_judge":None,
    "relevance_classifier":None,
    
}


@app.on_event("startup")
def _load_data() -> None:
    book = json.loads(Path(BOOK_PATH).read_text(encoding="utf-8"))
    market = json.loads(Path(MARKET_PATH).read_text(encoding="utf-8"))
    STATE["book"] = book
    STATE["market"] = market
    STATE["clients_by_id"] = {c["id"]: c for c in book["clients"]}

    # Built once, reused for every question in this service's lifetime.
    # If agent construction itself fails (bad LLM_BASE_URL, agno not
    # installed correctly), fail loudly at startup rather than silently
    # per-question -- you want to know immediately, not 40 questions in.
    STATE["classifier_agent"] = build_ambiguity_classifier(
        base_url=_llm_v1_url(), api_key=LLM_API_KEY, model_id=FAST_MODEL)
    STATE["router_agent"] = build_router_agent(
        base_url=_llm_v1_url(), api_key=LLM_API_KEY, model_id=FAST_MODEL)
    STATE["strong_router_agent"] = build_strong_router_agent(
        base_url=_llm_v1_url(), api_key=LLM_API_KEY, model_id=DEEP_MODEL)
    STATE["abstention_judge"] = build_abstention_judge(
        base_url=_llm_v1_url(), api_key=LLM_API_KEY, model_id=FAST_MODEL)
    STATE["relevance_classifier"] = build_call_relevance_classifier(
        base_url=_llm_v1_url(), api_key=LLM_API_KEY, model_id=FAST_MODEL)
    

# ---------------------------------------------------------------------------
# The roster. Update model ids/names to match what you actually wire up.
# All 6 required roles must be present and must each actually run at least
# once across your answers -- a roster is a claim, checked against behaviour.
# ---------------------------------------------------------------------------
def build_roster() -> Roster:
    return Roster(
        framework="agno",
        framework_version=os.environ.get("AGNO_VERSION", "2.6.9"),
        agents=[
            AgentDecl(role="router", name="Router", model="valura-fast"),
            AgentDecl(role="book_qa", name="BookQA", model="valura-fast"),
            AgentDecl(role="kyc_profile", name="KYCProfile", model="valura-fast"),
            AgentDecl(role="notes_desk", name="NotesDesk", model="valura-deep"),
            AgentDecl(role="market_desk", name="MarketDesk", model="valura-fast"),
            AgentDecl(role="compliance", name="Compliance", model="valura-fast"),
            
            AgentDecl(role="verifier", name="Verifier", model="valura-fast"),
        ],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/agents")
def agents():
    return build_roster().model_dump()


@app.post("/answer")
async def answer(request: Request):
    body = await request.json()
    try:
        q = Question.model_validate(body)
    except Exception as e:
        # Even a malformed envelope must get a well-formed reply back --
        # there is no question_id to echo here, so this is the one place a
        # non-Answer response is unavoidable. Keep it minimal.
        return JSONResponse(status_code=400, content={"error": str(e)})

    client = STATE["clients_by_id"].get(q.client_id)
    if client is None:
        # Should not happen against a well-formed run, but never fabricate:
        # abstain honestly rather than guess at a client that isn't in the
        # book you were given.
        result = abstain(q.question_id,
                          f"No client record found for {q.client_id}.")
        return JSONResponse(content=result.model_dump())

    try:
        result = route_and_answer(q, client)
    except Exception as e:  # never let an exception kill the run
        result = abstain(
            q.question_id,
            f"internal error while answering: {type(e).__name__}: {e}",
        )
        result = result.model_dump()
    return JSONResponse(content=result)


def route_and_answer(q: Question, client: dict) -> dict:
    return answer_question(
        question_id=q.question_id,
        client_id=q.client_id,
        prompt=q.prompt,
        client=client,
        market=STATE["market"],
        base_url=_llm_v1_url(),
        api_key=LLM_API_KEY,
        classifier_agent=STATE["classifier_agent"],
        router_agent=STATE["router_agent"],
        strong_router_agent=STATE["strong_router_agent"],
        relevance_classifier=STATE["relevance_classifier"],
        fast_model=FAST_MODEL,
        deep_model=DEEP_MODEL,
        
        abstention_judge=STATE["abstention_judge"],
        
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.service:app", host="0.0.0.0", port=PORT)