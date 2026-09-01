"""app/planner.py -- the Evidence Planner.

Sits between router and specialist dispatch:

    QUESTION -> ROUTER (which role(s)?) -> PLANNER (what evidence, within
    those roles, does answering actually require, and how does it
    combine?) -> specialist(s) (which tool actually gets it) -> resolver
    (checks the plan against what was actually returned) -> verifier ->
    AnswerContract

Why this exists: _resolve_capture()/_merge() previously had to INFER
intent from whatever tool calls happened to occur, after the fact. That
made legitimate multi-call synthesis (e.g. price(Jan) + price(Mar), meant
to be combined into a % return) look identical to a genuine exploratory
mismatch (kyc_profile trying the wrong field twice) -- both are "multiple
different values from one run", and value-equality alone cannot tell them
apart. The planner removes the guessing by deciding, BEFORE any tool is
called, exactly what evidence is needed and whether/how it combines.

IMPORTANT -- the planner does NOT choose tools. Tool selection is the
specialist's job, not the planner's:

    ROUTER      -- WHO answers this? (which role(s))
    PLANNER     -- WHAT evidence does answering need? (semantics, not
                   implementation)
    SPECIALIST  -- HOW do I get that evidence? (chooses its own real
                   tool, e.g. get_price_as_of, current_holding, ...)
    RESOLVER    -- did we actually obtain every required piece?
    VERIFIER    -- is the combined result correct?

This keeps the planner decoupled from tool implementation details. If a
special

ist renames or adds a tool, nothing here has to change, because
the planner never names a tool -- only what evidence (field/source/
symbol/date) is required. Coupling the planner to tool names would just
relocate the old guessing problem into the planner instead of removing
it.

Every claim this module makes is checked, never trusted blindly:
  - every requirement's `source` must correspond to a role the router
    actually dispatched to -- a plan that reaches for kyc when the router
    never routed kyc_profile invalidates the plan.
  - every `formula` must exist in KNOWN_FORMULAS, AND the shape/naming of
    `required` must be semantically plausible for that formula (a
    "product" plan needs a quantity-shaped field and a price-shaped
    field, not two unrelated numbers) -- see _validate_formula_semantics.
  - `rationale` is for humans/logs only. Nothing downstream may use it to
    decide how to compute anything -- only `formula` and `output_field`
    are ever read programmatically.

Fail-closed, not fail-open, for anything beyond the simplest case: if the
router sent a question to more than one role (a genuinely compound
question) and the planner cannot produce a valid plan, the caller must
NOT silently fall back to the old ambiguous multi-call resolver -- that
reintroduces exactly the guessing this module exists to remove. See
plan_is_required() below; team.py should abstain honestly on that
combination rather than guess.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

try:
    from agno.agent import Agent
    from agno.models.openai.like import OpenAILike
    from pydantic import BaseModel, Field
    _AGNO_AVAILABLE = True
except ImportError:  # keeps this module importable/testable without agno installed
    _AGNO_AVAILABLE = False

AnswerType = Literal["SINGLE_FACT", "MULTI_STEP", "MULTI_SOURCE"]
Source = Literal["book", "kyc", "market", "notes"]

# Which specialist role owns each source -- used to check a plan's
# requirements against what the router actually dispatched to. Update
# this alongside app/router.py's VALID_ROLES if either changes.
SOURCE_TO_ROLE = {
    "book": "book_qa",
    "kyc": "kyc_profile",
    "market": "market_desk",
    "notes": "notes_desk",
}

# Every formula name the planner is allowed to pick MUST exist in
# verifier.py's FORMULA_REGISTRY (if/when that's wired in) -- kept here as
# a plain set so the planner's own schema and downstream capability can be
# checked against each other.
KNOWN_FORMULAS = {
    "pct_return",       # (end_value - start_value) / start_value * 100
    "difference",       # a - b
    "product",          # a * b
    "sum",              # a + b
    "drift_pp",         # current_pct - target_pct, only for the rare case
                         # the two percentages were fetched as SEPARATE
                         # evidence -- NOT for the allocation-drift blob,
                         # which is SINGLE_FACT with extract_symbol.
}

# Loose, substring-based field-name expectations per formula, used for
# semantic (not just syntactic) validation. None means "generic, no
# specific naming expected, just the right item count" -- difference and
# sum are genuinely generic operations with no fixed vocabulary.
_FORMULA_FIELD_PATTERNS: dict[str, Optional[list[set[str]]]] = {
    "pct_return": [{"start", "begin", "initial", "open"},
                   {"end", "final", "close", "latest"}],
    "product": [{"quantity", "qty", "shares", "units"},
                {"price", "value", "close"}],
    "difference": None,
    "sum": None,
    "drift_pp": [{"current", "actual"}, {"target", "mandate", "recorded"}],
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


PLANNER_FEW_SHOT = """\
Examples (question -> plan):

Q: "What did AAPL close at on 1 July 2026?"
-> answer_type=SINGLE_FACT
   required=[{field:"price", source:"market",
              symbol:"AAPL", date:"2026-07-01"}]
   formula=null
   output_field="price"

Q: "What was AMD's percentage return between 1 July 2025 and 1 July 2026?"
-> answer_type=MULTI_STEP
   required=[
     {field:"start_price", source:"market",
      symbol:"AMD", date:"2025-07-01"},
     {field:"end_price",   source:"market",
      symbol:"AMD", date:"2026-07-01"}
   ]
   formula="pct_return"
   output_field="return_pct"

Q: "By how many percentage points is the client's JPM holding away from
    the agreed target?"
-> answer_type=SINGLE_FACT
   required=[{field:"drift", source:"market",
              extract_symbol:"JPM"}]
   formula=null
   output_field="drift"
   (this is the allocation-drift-per-symbol evidence -- still ONE
   required piece, just one that needs the JPM line extracted after the
   specialist retrieves it. Do not ask for current_pct and target_pct as
   two separate requirements unless you actually mean two independently
   fetched percentages.)

Q: "What is the USD value of the client's current NVDA position?"
-> answer_type=MULTI_SOURCE
   required=[
     {field:"quantity", source:"book",   symbol:"NVDA"},
     {field:"price",    source:"market", symbol:"NVDA"}
   ]
   formula="product"
   output_field="position_value"

Q: "What is the PAN on file, and when did the client first buy AAPL?"
-> answer_type=MULTI_SOURCE
   required=[
     {field:"pan", source:"kyc"},
     {field:"first_buy_date", source:"book", symbol:"AAPL"}
   ]
   formula=null
   output_field=null
   (two genuinely separate asks, not a computation -- state both, do not
   combine them into one number. MULTI_SOURCE with formula=null means
   "report multiple independent facts", not "compute one result".)

Q: "What sector is NVDA in?"
-> answer_type=SINGLE_FACT
   required=[{field:"sector", source:"market", symbol:"NVDA"}]
   formula=null
   output_field="sector"


Q: "How much did technology exposure deviate from the mandate?"

-> answer_type=SINGLE_FACT
   required=[{
       field:"drift",
       source:"market",
       sector:"Information Technology"
   }]
   formula=null
   output_field="drift"   
"""

PLANNER_INSTRUCTIONS = f"""\
You are the evidence planner for a client-records question-answering
service. You are given one question, already scoped to a single client,
and the specialist role(s) the router already decided this question
needs (book_qa, kyc_profile, and/or market_desk). Your only job: decide
what EVIDENCE answering this question actually requires -- not which
tool retrieves it.

You do NOT answer the question. You do NOT compute a number. You do NOT
name a tool. Naming a tool is the specialist's job -- it owns its tools
and knows how to satisfy an evidence requirement; you only own what
evidence is needed, and, if two or more pieces must be combined into one
figure, which formula combines them.

answer_type:
  SINGLE_FACT  -- exactly one piece of evidence answers this, full stop.
  MULTI_STEP   -- two or more pieces of evidence from the SAME source,
                  combined with a formula into the one figure asked for.
  MULTI_SOURCE -- two or more pieces of evidence from DIFFERENT sources.
                  May or may not need a formula -- some multi-source
                  questions just want both facts reported side by side
                  (formula=null), others compute one combined figure
                  (formula set).

Each required-evidence item needs:
  field   -- a short name for what this piece of evidence is. Use
             standard names where a formula is involved so they can be
             matched: start_price/end_price for pct_return, quantity/
             price for product, current_pct/target_pct for drift_pp.
  source  -- one of "book", "kyc", "market". MUST be a source whose owning
             role (book->book_qa, kyc->kyc_profile, market->market_desk)
             is in the routed role list you were given -- never reach
             into a source the router did not send this question to.
  symbol  -- instrument ticker, if relevant. Omit otherwise.
  date    -- YYYY-MM-DD if this is a dated lookup, or "latest" for
             current/now questions. Omit if not date-scoped.
  extract_symbol -- ONLY for allocation-drift-style questions: the one
             symbol whose line should be pulled out of a combined,
             multi-symbol result the specialist retrieves.
             sector -- ONLY for sector-exposure/concentration questions
("technology exposure", "how concentrated in Communication Services").
NEVER put a sector name in extract_symbol.
extract_symbol is for pulling ONE ticker's line out of a drift
result; sector aggregates every symbol whose instrument sector
matches.
A requirement sets at most one of extract_symbol or sector.

Do NOT include a tool name anywhere in your output. The specialist that
owns the relevant source decides which of its own tools satisfies each
requirement -- that is implementation detail you are not given and must
not guess at.

formula: null, or one of {sorted(KNOWN_FORMULAS)} -- the ONLY formulas
that actually exist downstream. Never invent one. If a computation is
needed that none of these cover, still list the required evidence, leave
formula null, and say so in rationale -- the resolver will report the raw
pieces rather than guess at an unsupported computation.

output_field: a short name for the ONE thing this question ultimately
wants reported (e.g. "return_pct", "position_value", "price"). Null only
when answer_type is MULTI_SOURCE with formula=null (reporting several
independent facts, not one final figure).

rationale is for a human reading logs. Nothing in your rationale is ever
used to decide how a value is computed -- only formula and output_field
control that, so state them correctly rather than relying on rationale to
clarify intent.

Never ask for evidence a question doesn't need. Never combine two
genuinely separate asks under one formula just because they appear in the
same sentence -- only set a formula when the question wants ONE resulting
figure derived from more than one piece of evidence.

{PLANNER_FEW_SHOT}
"""


class EvidenceRequirement(BaseModel if _AGNO_AVAILABLE else object):
    if _AGNO_AVAILABLE:
        field: str
        source: Source
        symbol: Optional[str] = None
        date: Optional[str] = None
        extract_symbol: Optional[str] = None
        sector: Optional[str] = None

class EvidencePlan(BaseModel if _AGNO_AVAILABLE else object):
    if _AGNO_AVAILABLE:
        answer_type: AnswerType
        required: list[EvidenceRequirement] = Field(default_factory=list)
        formula: Optional[str] = None
        output_field: Optional[str] = None
        rationale: str = ""


def build_planner_agent(base_url: str, api_key: str, model_id: str = "valura-fast"):
    """Built once at startup, same as router/compliance -- no
    client-specific state. valura-fast by default: this is a structured
    classification call, not deep reasoning, matching the brief's own
    guidance not to spend the capable tier on a mechanical decision."""
    if not _AGNO_AVAILABLE:
        raise RuntimeError("agno is not installed in this environment")

    model = OpenAILike(id=model_id, api_key=api_key, base_url=base_url, temperature=0)
    return Agent(
        name="evidence_planner",
        role="planner",
        model=model,
        instructions=PLANNER_INSTRUCTIONS,
        output_schema=EvidencePlan,
        markdown=False,
    )


def _normalise_dates(plan: "EvidencePlan") -> None:
    """Mutates plan in place: any date field that isn't exactly
    YYYY-MM-DD or the literal 'latest' is dropped (set to None) rather
    than trusted. This is deliberately a normalisation, not a hard
    rejection of the whole plan -- a cosmetically malformed date on one
    requirement shouldn't discard an otherwise-valid plan; the affected
    specialist will simply be asked for that evidence without a date
    argument and decides what that means (usually 'no data')."""
    for req in plan.required:
        if req.date is not None and req.date != "latest" and not _DATE_RE.match(req.date):
            req.date = None


def _validate_sources_against_routed_roles(plan: "EvidencePlan", routed_roles) -> bool:
    routed = set(routed_roles or [])
    for req in plan.required:
        owning_role = SOURCE_TO_ROLE.get(req.source)
        if owning_role not in routed:
            return False
    return True


def _validate_formula_semantics(plan: "EvidencePlan") -> bool:
    """Syntactic check (formula in KNOWN_FORMULAS) is done by the caller
    before this runs. This is the SEMANTIC check: does the shape/naming
    of `required` actually make sense for the chosen formula. Loose,
    substring-based matching on `field` names -- planner's field names
    are free text, not a fixed enum, so exact matching would be brittle;
    this catches the clearly-wrong cases (e.g. 'product' requested over
    two price points with no quantity-shaped field) without being
    unreasonably strict about exact wording."""
    if plan.formula is None:
        return True
    patterns = _FORMULA_FIELD_PATTERNS.get(plan.formula)
    if patterns is None:
        # generic operations (difference, sum): just need >= 2 items
        return len(plan.required) >= 2
    if len(plan.required) != len(patterns):
        return False
    field_names = [r.field.lower() for r in plan.required]
    used_indices: set[int] = set()
    for pattern in patterns:
        match_idx = None
        for i, fn in enumerate(field_names):
            if i in used_indices:
                continue
            if any(keyword in fn for keyword in pattern):
                match_idx = i
                break
        if match_idx is None:
            return False
        used_indices.add(match_idx)
    return True


def build_plan_or_none(question: str, routed_roles, planner_agent) -> Optional[EvidencePlan]:
    """Runs the planner and validates its output against what the rest of
    the system can actually do. Returns None on ANY failure -- schema
    failure, a source the router didn't route to, an internally
    inconsistent answer_type/required-count, an unknown formula, or a
    formula whose required-evidence shape doesn't match what that formula
    needs.

    None does NOT mean "safe to fall back to guessing". See
    plan_is_required() below -- the caller (team.py) must decide what
    None means based on how many roles were routed, not treat every
    None the same way."""
    if planner_agent is None:
        return None
    try:
        result = planner_agent.run(f"routed_roles: {sorted(routed_roles or [])}\nquestion: {question}")
        plan: EvidencePlan = result.content
    except Exception as e:
        print(f"[PLANNER ERROR] {type(e).__name__}: {e}")
        return None

    if plan is None or not getattr(plan, "required", None):
        return None

    if plan.answer_type == "SINGLE_FACT" and len(plan.required) != 1:
        return None

    if plan.answer_type in ("MULTI_STEP", "MULTI_SOURCE") and len(plan.required) < 2:
        return None

    if not _validate_sources_against_routed_roles(plan, routed_roles):
        return None

    if plan.formula is not None and plan.formula not in KNOWN_FORMULAS:
        return None

    if not _validate_formula_semantics(plan):
        return None

    if plan.answer_type == "MULTI_SOURCE" and plan.formula is None and plan.output_field is not None:
        # MULTI_SOURCE + formula=None means "report several independent
        # facts" -- output_field implies a single combined result, which
        # contradicts that. Not fatal on its own, but don't trust an
        # internally contradictory plan.
        return None

    _normalise_dates(plan)
    return plan


def plan_is_required(routed_roles) -> bool:
    """Always True.

    Role count is NOT a reliable signal for whether a question needs the
    planner. A single routed role can still be MULTI_STEP (e.g. "AMD
    return Jan->Mar" routes to market_desk alone but needs start_price +
    end_price + pct_return) -- that is one of the main cases the planner
    exists to handle, so gating on len(routed_roles) > 1 would skip the
    planner for exactly the questions it matters most for.

    Since we can't know in advance whether a question is a trivial
    SINGLE_FACT or a same-source MULTI_STEP without running the planner,
    the planner must always run, and a failed/missing plan must always
    abstain rather than fall back to the old ambiguous multi-call
    resolver -- for one role or many. There is no case where falling
    back to guessing is safe."""
    return True