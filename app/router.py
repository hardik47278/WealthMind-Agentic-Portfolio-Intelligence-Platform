"""app/router.py -- the routing decision, as an LLM call with structured
output, per the DAG:

    QUESTION -> ROUTER -> compliance? -> REFUSE
                        -> legitimate -> specialist(s)
"""
from __future__ import annotations

from typing import Optional

try:
    from agno.agent import Agent
    from agno.models.openai.like import OpenAILike
    from pydantic import BaseModel, Field
    _AGNO_AVAILABLE = True
except ImportError:
    _AGNO_AVAILABLE = False

VALID_ROLES = {"book_qa", "kyc_profile", "notes_desk", "market_desk"}

ROUTER_FEW_SHOT = """\
Examples (question -> roles):

Q: "What is the current cash balance on the account?"
-> roles=["book_qa"]. Pure transaction/cash arithmetic.

Q: "What sector is NVDA in?"
-> roles=["market_desk"]. Pure instrument/market lookup, no client data needed.

Q: "What is the value of the client's NVDA position?"
-> roles=["book_qa","market_desk"]. Needs quantity held (book) x price (market).

Q: "What was the client's NVDA position worth on 2025-06-01?"
-> roles=["book_qa","market_desk"]. Same as above, with an as-of date -- the
   date does not change which specialists are needed, only their arguments.

Q: "How much did the client's technology exposure deviate from the mandate
    last quarter?"
-> roles=["book_qa","market_desk"]. "Deviate from the mandate" is allocation
   drift: current exposure needs positions (book) and sector classification
   (market) to compute against the recorded target.

Q: "Is the client's KYC verification current?"
-> roles=["kyc_profile"]. KYC status lookup.

Q: "What is the client's risk profile on file?"
-> roles=["kyc_profile"]. Recorded fact, not a recommendation.

Q: "Summarise the most recent note on this client's file."
-> roles=["notes_desk"]. Free-text note retrieval.

Q: "Did any note mention a settlement delay, and what's the client's
    current cash balance?"
-> roles=["notes_desk","book_qa"]. Two distinct asks in one question, each
   owned by a different specialist.

Q: "What is the PAN on file, and has the client received any dividends
    from it appearing in a note?"
-> roles=["kyc_profile","notes_desk"]. Identity lookup plus note search.

Q: "Should the client buy more NVDA?"
-> roles=["market_desk"]. Route to market_desk as the closest owning
   specialist for the subject matter; compliance's own classifier (run
   separately, before this router) is what actually decides this gets
   refused as advice, not this router. Routing and refusing are separate
   decisions.

Q: "What AAPL coverage do we hold dated on or before 1 April 2026,
    including count and substance?"
-> roles=["market_desk"]. "Coverage" here means market/news coverage.
   Because AAPL is a named ticker and the question asks about coverage
   before a date, use market_desk, specifically its symbol-scoped news
   retrieval. Do NOT route this to notes_desk just because "hold" or
   "coverage" sounds like client-record language.   
"""

ROUTER_INSTRUCTIONS = f"""\
You are the routing classifier for a client-records question-answering
service. Given one question (already scoped to a single client), decide
which specialist role(s), from this fixed set, are needed to answer it:

  book_qa      -- transactions, cash, deposits, withdrawals, fees,
                  dividends, trade counts/quantities, holdings.
  kyc_profile  -- identity, KYC status, employment, risk profile, bank
                  details.
  notes_desk   -- free-text operations notes and transaction memos.
  market_desk  -- instrument sector/industry, price history, news,
                  allocation drift against a recorded target.

Rules:
- Return every role genuinely needed, not just the first one that seems
  plausible. Some questions need two roles working together -- e.g. the
  USD value of a holding needs both book_qa (quantity) and market_desk
  (price); allocation drift needs both book_qa (current positions) and
  market_desk (sector weights) to compare against the target on file.
- A question can ask two separate things at once, each owned by a
  different specialist -- route both.
- Do not add a role "just in case". If a question is answerable by one
  specialist alone, return only that one -- an unnecessary specialist call
  costs latency and tokens for no benefit.
- Route by what the question needs to be ANSWERED, not by whether it
  should ultimately be refused -- refusal is decided elsewhere, separately,
  before your output is used. If a question solicits advice, still route
  it to whichever specialist owns that subject matter.
- Never invent a role outside the four listed. Never return an empty list.
- Also return a confidence score in [0, 1]. This is not a formality: if the
  question is ambiguous, vaguely worded, or you are choosing between two
  plausible role sets, say so with a LOWER number rather than picking one
  confidently. Reserve 0.85+ for cases that clearly match the pattern of
  one of the examples below. A wrong guess dressed up as confident is
  worse than an honest low score -- a low score gets a second, stronger
  look before anything commits to it.

- NEWS/COVERAGE RULE:
  If the question asks about a named company/ticker's news, coverage,
  headlines, announcements, events, or market developments, route to
  market_desk.

  "Coverage" in this context means MARKET/NEWS COVERAGE, not
  transaction-memo coverage.

  Examples:
    "What AAPL coverage do we hold?"
      -> ["market_desk"]

    "What AAPL coverage do we hold up to 1 April 2026?"
      -> ["market_desk"]

    "How many AAPL news items are on file?"
      -> ["market_desk"]

    "What did the AAPL coverage discuss?"
      -> ["market_desk"]

  Do NOT route these to notes_desk merely because the question uses
  words such as "hold", "on file", or "coverage".

  notes_desk is appropriate only when the question explicitly asks
  about client notes, transaction memos, buy memos, fee descriptions,
  or other client-record free text.  

{ROUTER_FEW_SHOT}
"""


class RouteDecision(BaseModel if _AGNO_AVAILABLE else object):
    if _AGNO_AVAILABLE:
        roles: list[str] = Field(
            description="One or more of: book_qa, kyc_profile, notes_desk, market_desk"
        )
        confidence: float = Field(
            default=0.5,
            description=(
                "0.0-1.0. How sure the model is that these are the right "
                "role(s) for this question. Ambiguous/vague questions should "
                "score low rather than guess confidently."
            ),
        )
        rationale: str = ""


CONFIDENCE_ESCALATION_THRESHOLD = 0.6


def build_router_agent(base_url: str, api_key: str, model_id: str = "gpt-4.1-mini"):
    if not _AGNO_AVAILABLE:
        raise RuntimeError("agno is not installed in this environment")

    model = OpenAILike(id=model_id, api_key=api_key, base_url=base_url, temperature=0)
    return Agent(
        name="router",
        role="router",
        model=model,
        instructions=ROUTER_INSTRUCTIONS,
        output_schema=RouteDecision,
        markdown=False,
    )


def build_strong_router_agent(base_url: str, api_key: str, model_id: str = "gpt-4.1"):
    if not _AGNO_AVAILABLE:
        raise RuntimeError("agno is not installed in this environment")

    model = OpenAILike(id=model_id, api_key=api_key, base_url=base_url, temperature=0)
    return Agent(
        name="router_strong",
        role="router",
        model=model,
        instructions=ROUTER_INSTRUCTIONS,
        output_schema=RouteDecision,
        markdown=False,
    )


def route(question: str, router_agent) -> list[str]:
    result = router_agent.run(question)
    decision = result.content
    roles = [r for r in getattr(decision, "roles", []) if r in VALID_ROLES]
    if not roles:
        roles = ["book_qa"]
    return roles


def route_with_confidence(
    question: str,
    fast_agent,
    strong_agent,
    threshold: float = CONFIDENCE_ESCALATION_THRESHOLD,
) -> tuple[list[str], str, float]:
    """Cascade: try the fast router first; if ITS OWN reported confidence
    is below `threshold`, re-run the identical question against the
    strong router and use that result instead. The two results are never
    blended and the low-confidence guess is never kept as a fallback --
    if the strong router is being asked at all, its answer is the one
    that ships.

    Returns (roles, model_used, confidence) where model_used is "fast" or
    "strong" and confidence is the deciding agent's own reported score --
    so team.py can log which tier decided this question AND how sure it
    was, for later threshold tuning (the whole point of collecting this
    in the first place -- the previous version of this function reported
    model_used but silently dropped the confidence value itself, which is
    the actual bug: any caller written to also capture confidence would
    unpack 3 values against a 2-value return and crash)."""
    result = fast_agent.run(question)
    decision = result.content
    roles = [r for r in getattr(decision, "roles", []) if r in VALID_ROLES]
    confidence = float(getattr(decision, "confidence", 0.0))
    model_used = "fast"

    if confidence < threshold:
        result = strong_agent.run(question)
        decision = result.content
        roles = [r for r in getattr(decision, "roles", []) if r in VALID_ROLES]
        confidence = float(getattr(decision, "confidence", 0.0))
        model_used = "strong"

    if not roles:
        roles = ["book_qa"]
    return roles, model_used, confidence


def route_with_fallback(
    question: str,
    router_agent: Optional[object],
    strong_router_agent: Optional[object] = None,
) -> tuple[list[str], str, float]:
    """Route a question using the LLM router first.

    If both fast and strong routers are available, use confidence-gated
    routing. If the router call fails, fall back to deterministic keyword
    routing so the question can still be answered.

    Returns:
        (roles, model_used, confidence)

    model_used is one of:
        "fast"
        "strong"
        "keyword_fallback"
    """

    # ---------------------------------------------------------
    # 1. Try the LLM router
    # ---------------------------------------------------------
    if router_agent is not None:
        try:
            # Use confidence-gated fast -> strong routing when available.
            if strong_router_agent is not None:
                return route_with_confidence(
                    question,
                    router_agent,
                    strong_router_agent,
                )

            # Otherwise use the fast router directly.
            result = router_agent.run(question)
            decision = result.content

            roles = [
                r
                for r in getattr(decision, "roles", [])
                if r in VALID_ROLES
            ]

            confidence = float(
                getattr(decision, "confidence", 0.0)
            )

            if not roles:
                roles = ["book_qa"]

            return roles, "fast", confidence

        except Exception:
            # Router failure should not kill the question.
            # Continue to deterministic fallback below.
            pass

    # ---------------------------------------------------------
    # 2. Deterministic fallback routing
    # ---------------------------------------------------------
    p = question.lower()
    roles: set[str] = set()

    # Market-related questions
    if any(
        k in p
        for k in (
            "price",
            "sector",
            "industry",
            "news",
            "market",
            "listed",
            "stock",
            "share price",
        )
    ):
        roles.add("market_desk")

    # KYC-related questions
    if any(
        k in p
        for k in (
            "pan",
            "kyc",
            "risk profile",
            "bank account",
            "employer",
            "employment",
            "occupation",
            "address",
            "date of birth",
            "dob",
        )
    ):
        roles.add("kyc_profile")

    # Notes-related questions
    if any(
        k in p
        for k in (
            "note",
            "notes",
            "memo",
            "mentioned",
            "comment",
            "remark",
        )
    ):
        roles.add("notes_desk")

    # Book / transaction questions
    if any(
        k in p
        for k in (
            "cash",
            "balance",
            "deposit",
            "withdrawal",
            "transaction",
            "transactions",
            "purchase",
            "purchases",
            "buy",
            "bought",
            "sell",
            "sold",
            "dividend",
            "fee",
            "fees",
            "holding",
            "hold",
            "shares",
            "position",
            "portfolio",
        )
    ):
        roles.add("book_qa")

    # ---------------------------------------------------------
    # 3. Multi-specialist cases
    # ---------------------------------------------------------
    # If the question asks for value/worth of a holding,
    # both quantity (book) and market price are required.
    if any(k in p for k in ("value", "worth")):
        roles.add("book_qa")
        roles.add("market_desk")

    # Explicit "as of" price/value questions often need market data
    # together with the client's holdings.
    if (
        ("as at" in p or "as of" in p)
        and any(k in p for k in ("value", "worth", "shares", "position"))
    ):
        roles.add("book_qa")

  
    if not roles:
        roles.add("book_qa")

    return sorted(roles), "keyword_fallback", 1.0