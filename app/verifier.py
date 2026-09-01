
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from app import tools_book as book
from app import tools_kyc as kyc
from app import tools_market as mkt
from app import tool_nodes as notes
from app.tools_book import Result as BookResult, first_symbol_buy_date
try:
    from agno.agent import Agent
    from agno.models.openai.like import OpenAILike
    from pydantic import BaseModel
    _AGNO_AVAILABLE = True
except ImportError:
    _AGNO_AVAILABLE = False
@dataclass
class VerificationClaim:
    field: str
    source: str
    tool_name: str
    tool_args: dict
    expected_value: Optional[str]
@dataclass
class DraftAnswer:
    question_id: str
    client_id: str
    answer: str
    answer_value: Optional[str]
    citations: list[str]
    confidence: float
    flags: list[str]
    agents: list[str]
    source: str  # "book", "kyc", or "market" -- which registry to look the tool up in
    tool_name: Optional[str] = None  # function name in tools_book.py / tools_market.py
    tool_args: dict = field(default_factory=dict)  # kwargs to re-call it with (as_of as ISO str, not date)
    abstained: bool = False
    refused: bool = False
    reason: Optional[str] = None
    question: str = ""
    verification_claims: list[VerificationClaim] = field(default_factory=list)

@dataclass
class VerificationResult:
    passed: bool
    detail: str
    forced_abstain_reason: Optional[str] = None
    corrected_citations: Optional[list[str]] = None

_BOOK_REGISTRY = {
    "calculate_cash_balance": book.calculate_cash_balance,
    "total_deposits": book.total_deposits,
    "largest_deposit": book.largest_deposit,
    "total_withdrawals": book.total_withdrawals,
    "withdrawal_count": book.withdrawal_count,
    "total_fees": book.total_fees,
    "total_dividends": book.total_dividends,
    "last_dividend_date": book.last_dividend_date,
    "total_dividend_withholding_tax": book.total_dividend_withholding_tax,
    "dividend_symbols": book.dividend_symbols,
    "symbol_purchase_count": book.symbol_purchase_count,
    "symbol_sale_count": book.symbol_sale_count,
    "total_quantity_bought": book.total_quantity_bought,
    "total_quantity_sold": book.total_quantity_sold,
    "largest_buy": book.largest_buy,
    "largest_sell": book.largest_sell,
    "first_transaction_date": book.first_transaction_date,
    "first_symbol_buy_date": book.first_symbol_buy_date,
    "latest_transaction_date": book.latest_transaction_date,
    "distinct_holding_count": book.distinct_holding_count,
    "current_holding_from_snapshot": book.current_holding_from_snapshot,
    "holding_as_of": book.holding_as_of,
    "transactions_in_range": book.transactions_in_range,
    "sum_in_range": book.sum_in_range,
}
_MARKET_REGISTRY = {
    "is_covered": mkt.is_covered,
    "get_instrument": mkt.get_instrument,
    "get_latest_price": mkt.get_latest_price,
    "get_price_as_of": mkt.get_price_as_of,
    "get_latest_news": mkt.get_latest_news,
    "search_news_by_symbol": mkt.search_news_by_symbol,
    "search_news_global": mkt.search_news_global,
    "get_sector_allocation": mkt.get_sector_allocation,
    "get_allocation_drift": mkt.get_allocation_drift,
    "get_percentage_return_as_of": mkt.get_percentage_return_as_of,
}
_KYC_REGISTRY = {
    "identity_lookup": kyc.identity_lookup,
    "kyc_lookup": kyc.kyc_lookup,
    "employment_lookup": kyc.employment_lookup,
    "bank_lookup": kyc.bank_lookup,
}

_NOTES_REGISTRY = {
    "get_client_notes": notes.get_client_notes,
    "outstanding_actions": notes.outstanding_actions,
    "transaction_memos_containing": notes.transaction_memos_containing,
}
_DATE_ARGS = {"as_of"}

def _coerce_args(tool_args: dict) -> dict:
    coerced = dict(tool_args)
    for key in _DATE_ARGS:
        if key in coerced and coerced[key] is not None and isinstance(coerced[key], str):
            coerced[key] = datetime.strptime(coerced[key], "%Y-%m-%d").date()
    return coerced

def _recompute(draft: DraftAnswer, client: dict, market: Optional[dict]) -> Optional[BookResult]:
    """Re-runs the exact function+args recorded in provenance. Returns
    None if there's nothing to recompute (no tool_name -- e.g. a pure
    prose/notes answer with no single figure)."""
    if draft.tool_name is None:
        return None
    args = _coerce_args(draft.tool_args)
    if draft.source == "book":
        fn = _BOOK_REGISTRY.get(draft.tool_name)
        if fn is None:
            raise KeyError(f"unknown book tool in provenance: {draft.tool_name}")
        return fn(client, **args)
    if draft.source == "kyc":
        fn = _KYC_REGISTRY.get(draft.tool_name)
        if fn is None:
            raise KeyError(f"unknown kyc tool in provenance: {draft.tool_name}")
        return fn(client, **args)

    if draft.source == "notes":
        fn = _NOTES_REGISTRY.get(draft.tool_name)
        if fn is None:
            raise KeyError(f"unknown notes tool in provenance: {draft.tool_name}")
        return fn(client, **args)


    if draft.source == "market":
        fn = _MARKET_REGISTRY.get(draft.tool_name)
        if fn is None:
            raise KeyError(f"unknown market tool in provenance: {draft.tool_name}")
        if draft.tool_name in {"get_sector_allocation", "get_allocation_drift"}:
            return fn(client, market)
        return fn(market, **args)

    raise ValueError(f"unknown provenance source: {draft.source!r}")

def _values_match(recomputed: Any, stated: Optional[str]) -> bool:
    """Compares the freshly recomputed value against the string the
    specialist put in answer_value. Decimal-aware (2dp quantisation, same
    as the contract's money formatting) so '71.880' vs '71.88' isn't a
    false mismatch; falls back to plain string equality for dates/counts/
    text values."""
    if stated is None:
        return recomputed is None
    if isinstance(recomputed, Decimal):
        try:
            stated_d = Decimal(stated)
        except InvalidOperation:
            return False
        return recomputed.quantize(Decimal("0.01")) == stated_d.quantize(Decimal("0.01"))
    if isinstance(recomputed, bool):
        return str(recomputed).lower() == stated.strip().lower()
    return str(recomputed) == stated.strip()

def verify_claims(
    draft: DraftAnswer,
    client: dict,
    market: Optional[dict] = None,
) -> VerificationResult:
    """Verify every independent factual claim in a multi-source answer."""

    if not draft.verification_claims:
        return VerificationResult(
            passed=True,
            detail="no independent verification claims recorded",
        )

    for claim in draft.verification_claims:
        claim_draft = DraftAnswer(
            question_id=draft.question_id,
            client_id=draft.client_id,
            answer=draft.answer,
            answer_value=claim.expected_value,
            citations=draft.citations,
            confidence=draft.confidence,
            flags=draft.flags,
            agents=draft.agents,
            source=claim.source,
            tool_name=claim.tool_name,
            tool_args=claim.tool_args,
        )

        result = verify_numeric(claim_draft, client, market)

        if not result.passed:
            return VerificationResult(
                passed=False,
                detail=f"claim '{claim.field}' failed: {result.detail}",
                forced_abstain_reason=result.forced_abstain_reason,
            )

    return VerificationResult(
        passed=True,
        detail=f"all {len(draft.verification_claims)} claims re-verified",
    )



def verify_numeric(draft: DraftAnswer, client: dict, market: Optional[dict] = None) -> VerificationResult:
    if draft.abstained or draft.refused:
        # Nothing to verify numerically -- answer_value must be null on
        # these anyway (contract.py already enforces that separately).
        return VerificationResult(passed=True, detail="abstained/refused draft -- no figure to verify")

    if draft.tool_name is None:
        if draft.answer_value is not None:
            return VerificationResult(
                passed=False,
                detail="answer_value is set but no provenance (tool_name) was recorded -- "
                       "cannot confirm this figure came from a deterministic tool call",
                forced_abstain_reason="Unable to verify this figure against the underlying records.",
            )
        return VerificationResult(passed=True, detail="no numeric value to verify")

    try:
        result = _recompute(draft, client, market)
    except Exception as exc:  # noqa: BLE001 -- any recompute failure fails closed
        return VerificationResult(
            passed=False,
            detail=f"recompute raised: {exc}",
            forced_abstain_reason="Unable to verify this figure against the underlying records.",
        )
    if result is None:
        return VerificationResult(passed=True, detail="no numeric value to verify")

    if _values_match(result.value, draft.answer_value):
        return VerificationResult(passed=True, detail=f"recomputed value matches: {result.value}")

    return VerificationResult(
        passed=False,
        detail=f"MISMATCH: draft answer_value={draft.answer_value!r}, "
               f"recomputed via {draft.tool_name}({draft.tool_args})={result.value!r}",
        forced_abstain_reason=(
            "This service computed a figure for this question that did not match on "
            "re-verification, so it is withholding the number rather than risk stating "
            "an incorrect one."
        ),
    )

def _all_known_record_ids(client: dict) -> set[str]:
    ids = set()
    for t in client.get("transactions", []):
        ids.add(t["id"])
    for p in client.get("positions_snapshot", []):
        if "id" in p:
            ids.add(p["id"])
    for n in client.get("notes", []):
        ids.add(n["id"])
    for r in client.get("suitability_reviews", []):
        if "id" in r:
            ids.add(r["id"])
    if "kyc" in client and "id" in client["kyc"]:
        ids.add(client["kyc"]["id"])
    return ids

def verify_citations(draft: DraftAnswer, client: dict, market: Optional[dict] = None) -> VerificationResult:
    if not draft.citations:
        return VerificationResult(passed=True, detail="no citations to check")

    known_client_ids = _all_known_record_ids(client)
    scoped_client_id = draft.client_id

    market_symbols = set(market.get("meta", {}).get("covered_symbols", [])) if market else set()
    market_news_ids = {n["id"] for n in market.get("news", [])} if market else set()

    bad: list[str] = []
    for cid in draft.citations:
        if cid == scoped_client_id:
            continue 
        if cid in known_client_ids:
            continue
        if ":" in cid:  
            sym = cid.split(":", 1)[0]
            if sym in market_symbols:
                continue
        if cid in market_news_ids:
            continue
        if cid in market_symbols:  # instrument-level citation (get_instrument)
            continue
        bad.append(cid)

    if bad:
        return VerificationResult(
            passed=False,
            detail=f"citations not found in this client's book or the market file: {bad}",
            forced_abstain_reason="This answer cited records that could not be confirmed on re-check.",
        )
    return VerificationResult(passed=True, detail="all citations verified")



CITATION_LIST_MAX = 6

def verify_citation_count_rule(draft: DraftAnswer) -> VerificationResult:
    n = len(draft.citations)
    if n <= CITATION_LIST_MAX:
        return VerificationResult(passed=True, detail=f"{n} citations, within the list-them-individually limit")
    if draft.citations == [draft.client_id]:
        return VerificationResult(passed=True, detail="citations correctly collapsed to client_id")
    return VerificationResult(
        passed=False,
        detail=f"{n} citations listed individually, over the {CITATION_LIST_MAX} limit -- should cite client_id instead",
        corrected_citations=[draft.client_id],
    )


ABSTENTION_JUDGE_INSTRUCTIONS = """\
You are a strict pre-answer judge for a financial client-servicing agent. You
will be given the user's ORIGINAL QUESTION and the DRAFT ANSWER the agent
produced for it, with no recomputable figure behind it (no tool call backs
this draft -- it is prose/advice/opinion).

Before letting this draft stand, interrogate the question the way a careful
analyst would, by asking yourself tough questions such as:
  - Does this question rest on a false premise, or assume a fact not in
    evidence?
  - Is this actually asking for investment advice, a recommendation, or a
    prediction (buy/sell/hold, "should I", "is now a good time", target
    allocation) rather than a factual lookup? This service must not give
    financial advice.
  - Is the question ambiguous or missing context needed to answer it safely
    (unclear client, unclear date range, unclear which record)?
  - Does the draft answer actually address what was asked, or does it dodge,
    guess, or state something as fact that the question doesn't establish?
  - Would a careful, liability-conscious analyst answer this as confidently
    as the draft does, or would they qualify, clarify, or decline?

Decide exactly one of:
  ANSWER            -- the draft is safe, factual, and appropriately scoped.
  ASK_CLARIFICATION -- the question is missing/ambiguous information needed
                        to answer safely; do not guess.
  ABSTAIN           -- the question calls for advice/opinion/prediction this
                        service should not give, or rests on a false premise,
                        or the draft cannot be trusted as-is.

Do not answer the underlying question yourself. Do not soften your decision
to be helpful -- an honest ABSTAIN or ASK_CLARIFICATION is preferred over a
confident answer that oversteps.
"""

def build_abstention_judge(base_url: str, api_key: str, model_id: str = "gpt-4o-mini"):
    """Builds the LLM judge used by verify_abstention_worthy. Kept as its
    own small agent (default model_id="gpt-4o-mini") rather than reusing the
    numeric-path model -- this is a judgment call, not arithmetic, and is
    cheap/fast by design so it can run on every no-provenance draft without
    materially affecting latency."""
    if not _AGNO_AVAILABLE:
        raise RuntimeError("agno is not installed in this environment")

    class AbstentionJudgment(BaseModel):
        decision: str  # "ANSWER" | "ASK_CLARIFICATION" | "ABSTAIN"
        tough_questions: list[str] = []
        rationale: str = ""

    model = OpenAILike(id=model_id, api_key=api_key, base_url=base_url, temperature=0)
    return Agent(
        name="verifier_abstention_judge",
        role="verifier",
        model=model,
        instructions=ABSTENTION_JUDGE_INSTRUCTIONS,
        output_schema=AbstentionJudgment,
        markdown=False,
 
    )

def verify_abstention_worthy(draft: DraftAnswer, judge_agent=None) -> VerificationResult:
    """Only meaningful for drafts with no tool-backed figure (draft.tool_name
    is None) that aren't already abstained/refused -- i.e. exactly the class
    verify_numeric can't grade. Fails open if no judge is configured or the
    call errors, matching verify_prose_support's posture."""
    if draft.abstained or draft.refused:
        return VerificationResult(passed=True, detail="already abstained/refused -- nothing to judge")
    if draft.tool_name is not None:
        return VerificationResult(passed=True, detail="tool-backed draft -- numeric check already covers this")
    if judge_agent is None:
        return VerificationResult(passed=True, detail="no abstention judge configured -- skipped")

    try:
        result = judge_agent.run(
            f"ORIGINAL QUESTION:\n{draft.question}\n\n"
            f"DRAFT ANSWER:\n{draft.answer}"
        )
        verdict = result.content
        decision = getattr(verdict, "decision", "ANSWER").strip().upper()
        questions = getattr(verdict, "tough_questions", [])
        rationale = getattr(verdict, "rationale", "")

        if decision == "ANSWER":
            return VerificationResult(passed=True, detail=f"judge: ANSWER -- {rationale}")

        if decision == "ASK_CLARIFICATION":
            return VerificationResult(
                passed=False,
                detail=f"judge: ASK_CLARIFICATION -- {rationale} (tough questions: {questions})",
                forced_abstain_reason=(
                    "This question needs clarification before it can be answered safely: "
                    + ("; ".join(questions) if questions else rationale)
                ),
            )

        
        return VerificationResult(
            passed=False,
            detail=f"judge: ABSTAIN -- {rationale} (tough questions: {questions})",
            forced_abstain_reason=(
                "This request calls for judgment, advice, or a premise this service "
                "cannot safely confirm, so it is declining rather than guessing: " + rationale
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- optional layer, fail open
        return VerificationResult(passed=True, detail=f"abstention judge unavailable ({exc}) -- skipped, failing open")

VERIFIER_LLM_INSTRUCTIONS = """\
You are a verification checker. You will be given a drafted answer's prose and a
small set of source record excerpts it claims to be based on. Decide only:
  is_supported: does the prose accurately reflect what the excerpts say, with no
    claim the excerpts don't actually support?
Do not answer the underlying question yourself. Do not add information. If the
prose says something the excerpts don't confirm (e.g. claims "no fees" while a
fee record is present in the excerpts), is_supported=false.
"""

def build_llm_fallback_checker(base_url: str, api_key: str, model_id: str = "valura-fast"):
    if not _AGNO_AVAILABLE:
        raise RuntimeError("agno is not installed in this environment")

    class SupportSchema(BaseModel):
        is_supported: bool
        rationale: str = ""

    model = OpenAILike(id=model_id, api_key=api_key, base_url=base_url,temperature=0
                       )
    return Agent(
        name="verifier_llm_fallback",
        role="verifier",
        model=model,
        instructions=VERIFIER_LLM_INSTRUCTIONS,
        output_schema=SupportSchema,
        markdown=False,
          # deterministic classification, not creative writing
    )


def verify_prose_support(draft: DraftAnswer, record_excerpts: str, checker_agent=None) -> VerificationResult:
    """Only call this for prose-heavy answers with no single figure to
    recompute (draft.tool_name is None but draft.answer is non-trivial,
    e.g. notes_desk summaries). Fails open (passed=True) if no checker is
    given or the call errors -- this is the optional, unscored, last-resort
    layer, not something that should block a submission on its own during
    a blackout."""
    if checker_agent is None:
        return VerificationResult(passed=True, detail="no LLM fallback configured -- skipped")
    try:
        result = checker_agent.run(f"DRAFT ANSWER:\n{draft.answer}\n\nSOURCE EXCERPTS:\n{record_excerpts}")
        verdict = result.content
        if getattr(verdict, "is_supported", True):
            return VerificationResult(passed=True, detail="prose supported by cited excerpts")
        return VerificationResult(
            passed=False,
            detail=f"prose not supported: {getattr(verdict, 'rationale', '')}",
            forced_abstain_reason="This answer's wording could not be confirmed against the cited records.",
        )
    except Exception as exc:  # noqa: BLE001
        return VerificationResult(passed=True, detail=f"LLM fallback unavailable ({exc}) -- skipped, failing open")


def verify_draft(
    draft: DraftAnswer,
    client: dict,
    market: Optional[dict] = None,
    abstention_judge=None,
) -> DraftAnswer:
    """Runs the deterministic checks in order and returns a (possibly
    modified) draft: on a numeric or citation-existence failure, forces
    abstained=true with a reason and clears answer_value/citations, per
    'a confident wrong number is far more expensive than an honest the
    data does not say'. On a citation-count-rule failure, just corrects
    the citations list in place -- that's a formatting fix, not a trust
    problem, so it doesn't need to become an abstain.

    abstention_judge is optional (default None -> skipped, same posture as
    verify_prose_support's checker_agent) so existing callers that don't
    pass one behave exactly as before. When provided, it only fires on
    drafts with no tool-backed figure (see verify_abstention_worthy) -- the
    class of subjective/advisory/ambiguous questions the deterministic
    checks above have no figure to grade."""
    claims = verify_claims(draft, client, market)

    if not claims.passed:
        draft.abstained = True
        draft.answer = ""
        draft.answer_value = None
        draft.citations = []
        draft.reason = claims.forced_abstain_reason
        draft.confidence = min(draft.confidence, 0.3)
        return draft
    numeric = verify_numeric(draft, client, market)
    if not numeric.passed:
        draft.abstained = True
        draft.answer = ""
        draft.answer_value = None
        draft.citations = []
        draft.reason = numeric.forced_abstain_reason
        draft.confidence = min(draft.confidence, 0.3)
        return draft

    citations = verify_citations(draft, client, market)
    if not citations.passed:
        draft.abstained = True
        draft.answer = ""
        draft.answer_value = None
        draft.citations = []
        draft.reason = citations.forced_abstain_reason
        draft.confidence = min(draft.confidence, 0.3)
        return draft

    count_rule = verify_citation_count_rule(draft)
    if not count_rule.passed and count_rule.corrected_citations is not None:
        draft.citations = count_rule.corrected_citations

    abstention = verify_abstention_worthy(draft, abstention_judge)
    if not abstention.passed:
        draft.abstained = True
        draft.answer = ""
        draft.answer_value = None
        draft.citations = []
        draft.reason = abstention.forced_abstain_reason
        draft.confidence = min(draft.confidence, 0.3)
        return draft

    return draft


CALL_RELEVANCE_INSTRUCTIONS = """\
You are given a question and a list of tool calls a specialist agent made
while trying to answer it, each with its name, arguments, and the value it
returned. Some of these calls answer the question directly. Others are
supporting/incidental lookups the agent made along the way and do not
themselves answer what was asked.

Pick the SINGLE call whose returned value is the actual answer to the
question. Do not judge whether that value is correct -- only identify
which call, if any, is the one being asked for.

If two or more calls look equally relevant, or none of them actually
answers the question as asked, return primary_call_index = -1 and explain
why in rationale -- do not guess.
"""

def build_call_relevance_classifier(base_url: str, api_key: str, model_id: str = "gpt-4.1-mini"):
    if not _AGNO_AVAILABLE:
        raise RuntimeError("agno is not installed in this environment")

    class CallRelevanceJudgment(BaseModel):
        primary_call_index: int
        rationale: str = ""

    model = OpenAILike(id=model_id, api_key=api_key, base_url=base_url, temperature=0)
    return Agent(
        name="verifier_call_relevance",
        role="verifier",
        model=model,
        instructions=CALL_RELEVANCE_INSTRUCTIONS,
        output_schema=CallRelevanceJudgment,
        markdown=False,
          # deterministic classification, not creative writing
    )

def classify_primary_call(
    question: str,
    calls: list[tuple[str, Any, dict]],
    classifier_agent=None,
) -> Optional[int]:
    """Returns the index into calls of the one call that answers
    question, or None if no classifier is configured, the call errors,
    or the model itself could not pick one (primary_call_index == -1 or
    out of range) -- in every one of those cases the caller should fall
    back to its own existing logic (e.g. value-equality) rather than
    trust a bad index. This function never raises and never returns an
    index it has not range-checked itself."""
    if classifier_agent is None or not calls:
        return None
    if len(calls) == 1:
        return 0  # nothing to disambiguate

    listing = "\n".join(
        f"[{i}] {name}(args={args}) -> value={getattr(result, 'value', None)!r}, "
        f"note={getattr(result, 'note', None)!r}"
        for i, (name, result, args) in enumerate(calls)
    )
    try:
        run_result = classifier_agent.run(f"QUESTION:\n{question}\n\nTOOL CALLS:\n{listing}")
        verdict = run_result.content
        idx = getattr(verdict, "primary_call_index", -1)
        if isinstance(idx, int) and 0 <= idx < len(calls):
            return idx
        return None
    except Exception:  # noqa: BLE001 -- optional layer, fail open to caller's own logic
        return None