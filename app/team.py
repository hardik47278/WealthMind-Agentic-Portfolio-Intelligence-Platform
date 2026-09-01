"""app/team.py -- the orchestrator."""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from app.contract import AnswerContract
from app.compilance_agent import (
    compliance_check,
    output_scope_scan,
    apply_final_masking,
)
from app.agent_book import build_book_qa_agent
from app.agent_kyc import build_kyc_profile_agent
from app.agent_nodes_desk import build_notes_desk_agent
from app.market_agent import build_market_desk_agent
from app.verifier import (
    CITATION_LIST_MAX,
    DraftAnswer,
    verify_draft,
    classify_primary_call,
)

from app.query_decomposer import build_query_decomposer, decompose_question
import random
import time

import re
from decimal import Decimal
from typing import Optional


class UpstreamIssue(Exception):
    """Raised when an LLM call itself fails (quota/429, timeout, proxy
    blackout) -- distinct from a normal processing error, so the outer
    except block in answer_question can flag it correctly."""
    pass

def _run_with_backoff(agent, prompt: str, role: str,
                       max_retries: int = 3, base_delay: float = 1.0) -> object:
    """Wraps agent.run(prompt) with exponential backoff + jitter."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return agent.run(prompt)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

    raise UpstreamIssue(f"{role} agent call failed after {max_retries} attempts: {last_exc}") from last_exc

def _is_misroute(piece: dict) -> bool:
    if piece.get("tool_name") == "not_my_scope":
        return True
    mc = piece.get("market_content")
    if mc and str(mc.get("reason") or "").startswith("NOT_MY_SCOPE:"):
        return True
    return False

def _extract_symbol_drift(
    blob: str,
    symbol: str,
) -> Optional[Decimal]:
    m = re.search(
        rf"{re.escape(symbol.upper())}:.*?drift ([+-]?[\d.]+)pp",
        blob,
    )
    return Decimal(m.group(1)) if m else None

def _sum_drift_by_sector(
    blob: str,
    sector: str,
    market: dict,
) -> Optional[Decimal]:
    sector_symbols = {
        i["symbol"]
        for i in market.get("instruments", [])
        if i.get("sector", "").lower() == sector.lower()
    }

    if not sector_symbols:
        return None

    total = Decimal("0")
    found_any = False

    for sym in sector_symbols:
        value = _extract_symbol_drift(blob, sym)

        if value is not None:
            total += value
            found_any = True

    return total if found_any else None


def _resolve_capture(
    calls: list,
    client_id: str,
    question: str = "",
    relevance_classifier=None,
    allow_llm_pick=False,
) -> dict:
    """Looks at every tool call a specialist made this run, not just the
    last one, and picks the one that actually answers the question."""

    if not calls:
        return {
            "tool_name": None,
            "value": None,
            "citations": [],
            "note": "specialist made no supporting tool call for this question",
            "conflict": False,
            "tool_args": {},
            "status": "not_found",
        }

    if len(calls) == 1:
        tool_name, result, tool_args = calls[0]

        if tool_name == "not_my_scope":
            status = "not_my_scope"
        elif result.value is not None:
            status = "found"
        else:
            status = "not_found"

        return {
            "tool_name": tool_name,
            "value": None if result.value is None else str(result.value),
            "citations": result.citations(client_id),
            "note": result.note,
            "conflict": getattr(result, "conflict", False),
            "tool_args": tool_args,
            "status": status,
        }

    with_data = [
        (name, r, a)
        for name, r, a in calls
        if r.value is not None
    ]

    if not with_data:
        tool_name, result, tool_args = calls[-1]

        status = (
            "not_my_scope"
            if tool_name == "not_my_scope"
            else "not_found"
        )

        return {
            "tool_name": tool_name,
            "value": None,
            "citations": [],
            "note": result.note or "no matching record found across attempts",
            "conflict": False,
            "tool_args": tool_args,
            "status": status,
        }

    distinct_values = {
        str(r.value)
        for _, r, _ in with_data
    }

    if len(distinct_values) == 1:
        tool_name, result, tool_args = with_data[-1]

        return {
            "tool_name": tool_name,
            "value": str(result.value),
            "citations": result.citations(client_id),
            "note": result.note,
            "conflict": getattr(result, "conflict", False),
            "tool_args": tool_args,
            "status": "found",
        }

    if not allow_llm_pick:
        return {
            "tool_name": None,
            "value": None,
            "citations": [],
            "note": (
                "multi-evidence question -- resolved via plan, "
                "not primary-call pick"
            ),
            "conflict": False,
            "tool_args": {},
            "status": "unresolved",
        }

    primary_idx = classify_primary_call(
        question,
        calls,
        relevance_classifier,
    )

    if primary_idx is not None:
        tool_name, result, tool_args = calls[primary_idx]

        if tool_name == "not_my_scope":
            status = "not_my_scope"
        elif result.value is not None:
            status = "found"
        else:
            status = "not_found"

        return {
            "tool_name": tool_name,
            "value": (
                None
                if result.value is None
                else str(result.value)
            ),
            "citations": result.citations(client_id),
            "note": result.note,
            "conflict": getattr(result, "conflict", False),
            "tool_args": tool_args,
            "status": status,
        }

    tried = ", ".join(
        name for name, _, _ in calls
    )

    return {
        "tool_name": None,
        "value": None,
        "citations": [],
        "note": (
            f"specialist explored multiple unrelated fields ({tried}) "
            "and found different values across them, with no way to "
            "confirm which one answers the question asked -- treating "
            "as not on record rather than guessing"
        ),
        "conflict": False,
        "tool_args": {},
        "status": "unresolved",
    }


def _run_capture_agent(builder, client: dict, prompt: str, base_url: str,
                        api_key: str, model_id: str, role: str,
                        relevance_classifier=None, allow_llm_pick=False) -> dict:
    agent, capture = builder(client, base_url, api_key, model_id)
    run = _run_with_backoff(agent, prompt, role)
    text = getattr(run, "content", None) or str(run)

    resolved = _resolve_capture(capture["calls"], client["id"], prompt,
                                 relevance_classifier,allow_llm_pick)

    return {
        "role": role,
        "text": text,
        "value": resolved["value"],
        "citations": resolved["citations"],
        "note": resolved["note"],
        "tool_name": resolved["tool_name"],
        "conflict": resolved["conflict"],
        "tool_args": resolved["tool_args"],
        "status": resolved["status"],
        "all_calls": capture["calls"],
        "source": {
    "kyc_profile": "kyc",
    "notes_desk": "notes",
}.get(role, "book"),
    }


def _run_market_desk(client: Optional[dict], market: dict, prompt: str,
                      base_url: str, api_key: str, model_id: str,
                      relevance_classifier=None,allow_llm_pick=False) -> dict:
    agent, capture = build_market_desk_agent(market, client, base_url, api_key, model_id)
    run = _run_with_backoff(agent, prompt, "market_desk")
    text = getattr(run, "content", None) or str(run)

    calls_as_tuples = [
        (c["tool"], c["result"], {**c.get("args", {}), **c.get("kwargs", {})})
        for c in capture["calls"]
    ]

    resolved = _resolve_capture(calls_as_tuples, client["id"] if client else "",
                                 prompt, relevance_classifier, allow_llm_pick)

    return {
        "role": "market_desk",
        "text": text,
        "value": resolved["value"],
        "citations": resolved["citations"],
        "note": resolved["note"],
        "tool_name": resolved["tool_name"],
        "conflict": resolved["conflict"],
        "tool_args": resolved["tool_args"],
        "status": resolved["status"],
        "all_calls": calls_as_tuples,
        "source": "market",
    }


def _merge(question_id: str, client_id: str, pieces: list[dict]) -> dict:

    print("[MERGE DEBUG]", [
        {
            "role": p.get("role"),
            "value": p.get("value"),
            "status": p.get("status"),
            "note": p.get("note"),
        }
        for p in pieces
    ])
    texts: list[str] = []
    values: list[str] = []
    citations: list[str] = []
    agents_used: list[str] = ["router"]
    any_gap = False
    any_conflict = False

    for p in pieces:
        if p.get("conflict"):
            any_conflict = True

        if p["value"] is not None:
            values.append(p["value"])
            texts.append(p["text"])

        elif p.get("status") == "not_my_scope":
            # This specialist was not responsible for this part.
            # Do NOT treat it as missing evidence.
            pass

        elif p.get("status") in ("not_found", "unresolved"):
            # Relevant specialist could not provide verified evidence.
            any_gap = True
            texts.append(f"({p['role']}: {p['note']})")

        else:
            # Unknown state: remain conservative.
            any_gap = True
            texts.append(f"({p['role']}: {p['note']})")

        citations.extend(p["citations"])
        agents_used.append(p["role"])

    seen: set[str] = set()
    citations = [
        c for c in citations
        if not (c in seen or seen.add(c))
    ]

    if len(citations) > CITATION_LIST_MAX:
        citations = [client_id]

    complete = not any_gap and len(values) > 0

    return {
        "question_id": question_id,
        "answer": " ".join(t for t in texts if t).strip(),
        "answer_value": (
            values[-1]
            if complete and len(pieces) == 1
            else None
        ),
        "abstained": (not complete) and not any_conflict,
        "refused": False,
        "reason": (
            None
            if (complete or any_conflict)
            else (
                "This question needs more than one part of the record, "
                "and at least one part could not be found."
            )
        ),
        "citations": (
            citations
            if (complete or any_conflict)
            else []
        ),
        "confidence": 0.75 if len(pieces) > 1 else 0.85,
        "flags": ["conflict"] if any_conflict else [],
        "agents": list(dict.fromkeys(agents_used)),
    }





def _dispatch_role(role: str, client: dict, market: dict, prompt: str,
                    base_url: str, api_key: str, fast_model: str,
                    deep_model: str, relevance_classifier=None,allow_llm_pick=True)-> Optional[dict]:
    if role == "book_qa":
        return _run_capture_agent(
            build_book_qa_agent, client, prompt, base_url, api_key,
            fast_model, "book_qa", relevance_classifier,allow_llm_pick)
    if role == "kyc_profile":
        return _run_capture_agent(
            build_kyc_profile_agent, client, prompt, base_url, api_key,
            fast_model, "kyc_profile", relevance_classifier,allow_llm_pick)
    if role == "notes_desk":
        return _run_capture_agent(
            build_notes_desk_agent, client, prompt, base_url, api_key,
            deep_model, "notes_desk", relevance_classifier,allow_llm_pick)
    if role == "market_desk":
        return _run_market_desk(
            client, market, prompt, base_url, api_key, fast_model,
            relevance_classifier,allow_llm_pick)
    return None



def answer_question(question_id: str, client_id: str, prompt: str, 
                    client: dict, market: dict, base_url: str, api_key: str, 
                    classifier_agent=None, 
                    router_agent=None, 
                    strong_router_agent=None, 
                    relevance_classifier=None, 
                    abstention_judge=None, 
                    fast_model: str = "gpt-4.1-mini", 
                    deep_model: str = "gpt-4.1-mini") -> dict: 
    try: 
        refusal = compliance_check( 
            question_id, 
            prompt, 
            client_id, 
            classifier_agent, 
            client.get("name"), 
        ) 
 
        if refusal is not None: 
            return refusal 


        decomposer_agent = build_query_decomposer(base_url, api_key, model_id=fast_model)
        intents = decompose_question(decomposer_agent,prompt)

        
            
 
        print( 
            f"[DECOMPOSER] question={question_id} " 
            f"intent_count={len(intents)} "

        ) 

        pieces: list[dict] = []

        for i, intent in enumerate(intents, start=1):
            role = intent["role"]
            sub_question = intent["question"]

            print(
                f"[DECOMPOSER] INTENT {i}: ",
                f"role={role} ",
                f"sub_question={sub_question}",

            )

            piece = _dispatch_role(
                role,client,market,sub_question,base_url,api_key,fast_model,deep_model,relevance_classifier,True)

            if piece is not None:
                pieces.append(piece)

            
 
        
 
        
 
        print("\n=== DEBUG PIECES ===") 
 
        if question_id == "q_060": 
            print("\n========== Q60 FULL TOOL TRACE ==========") 
            print("QUESTION:", prompt) 
            print("CLIENT ID:", client_id) 
            print("ROLES:", [p["role"] for p in pieces]) 
            print() 
 
        for p in pieces: 
            print("ROLE:", p["role"]) 
            print("SELECTED TOOL:", p["tool_name"]) 
            print("SOURCE:", p["source"]) 
            print("VALUE:", p["value"])
            print("LLM TEXT:", p["text"]) 
            print("TOOL:", p["tool_name"]) 
            print("TOOL ARGS:", p["tool_args"]) 
            print("ALL CALLS:") 
 
            for tool_name, result, tool_args in p["all_calls"]: 
                print( 
                    "  ", 
                    tool_name, 
                    "ARGS=", 
                    tool_args, 
                    "VALUE=", 
                    result.value, 
                ) 
 
                if question_id == "q_060": 
                    print( 
                        "     NOTE=", 
                        result.note, 
                    ) 
                    print( 
                        "     RECORD_IDS=", 
                        result.record_ids, 
                    ) 
 
        if question_id == "q_060": 
            print("\n========== END Q60 TOOL TRACE ==========\n") 
 
        verified_pieces: list[dict] = [] 
 
        for p in pieces: 
            draft = DraftAnswer( 
                question_id=question_id, 
                client_id=client_id, 
                answer=p["text"], 
                answer_value=p["value"], 
                citations=p["citations"], 
                confidence=0.85, 
                flags=[], 
                agents=[p["role"]], 
                source=p["source"], 
                tool_name=p["tool_name"], 
                tool_args=p["tool_args"], 
                question=prompt, 
            ) 
 
            checked = verify_draft( 
                draft, 
                client, 
                market, 
                abstention_judge=abstention_judge, 
            ) 

            print(f"[VERIFIER] question={question_id} ",
                  f"role={p['role']} "
                  f"tool={p['tool_name']} "
                  f"abstained={checked.abstained} "
                  f"refused={checked.refused} "
                  f"reason={checked.reason} "
                  f"answer_value={checked.answer_value} "
                  f"citations={checked.citations}")


 
            p["value"] = checked.answer_value 
            p["citations"] = checked.citations 
 
            if checked.abstained: 
                p["value"] = None 
                p["note"] = checked.reason or p["note"] 
 
                if p.get("status") != "not_my_scope": 
                    p["status"] = "unresolved" 
 
            verified_pieces.append(p) 
 
        pieces = verified_pieces 
 
        merged = _merge( 
            question_id, 
            client_id, 
            pieces, 
        )
         
 
        leak = output_scope_scan( 
            merged.get("answer", ""), 
            merged.get("citations", []), 
            client_id, 
        ) 
 
        if leak: 
            return { 
                "question_id": question_id, 
                "answer": "", 
                "answer_value": None, 
                "abstained": False, 
                "refused": True, 
                "reason": f"Response withheld: {leak}", 
                "citations": [], 
                "confidence": 1.0, 
                "flags": [], 
                "agents": list(dict.fromkeys( 
                    merged.get("agents", ["router"]) 
                )), 
            } 
 
        merged["answer"] = apply_final_masking( 
            merged.get("answer", "") 
        ) 
 
        return AnswerContract.model_validate( 
            merged 
        ).model_dump() 
 
    except UpstreamIssue as exc: 
        return { 
            "question_id": question_id, 
            "answer": "", 
            "answer_value": None, 
            "abstained": True, 
            "refused": False, 
            "reason": f"Model service unavailable: {exc}", 
            "citations": [], 
            "confidence": 0.0, 
            "flags": ["upstream_issue"], 
            "agents": ["router"], 
        } 
 
    except Exception as exc: 
        return { 
            "question_id": question_id, 
            "answer": "", 
            "answer_value": None, 
            "abstained": True, 
            "refused": False, 
            "reason": ( 
                f"internal error while answering: " 
                f"{type(exc).__name__}: {exc}" 
            ), 
            "confidence": 0.0, 
            "flags": [], 
            "agents": ["router"], 
        }




