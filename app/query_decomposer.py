from __future__ import annotations

from agno.agent import Agent
from agno.models.openai.like import OpenAILike

DECOMPOSER_INSTRUCTIONS = """
You are a query decomposer for a regulated investment platform.

Your ONLY job is to determine whether the user's question contains one
or two independently answerable intents.

The platform has these specialists:

- book_qa:
  transactions, deposits, withdrawals, fees, dividends, cash,
  holdings, positions, trade details, transaction dates.

- kyc_profile:
  identity, PAN, bank details, employer, nominee, age,
  KYC status, risk profile.

- notes_desk:
  relationship notes, account notes, compliance notes, memos.
  Route here ONLY when the question is explicitly about client notes,
  account notes, compliance notes, transaction memos, comments, remarks,
  or other free-text client records.

- market_desk:
  market prices, returns, sectors, market news, portfolio
  allocation/drift calculations based on market/mandate data.
  When a question asks about coverage, news coverage, articles, news
  items, themes, or substance concerning a named stock/company, route
  it to market_desk. "Coverage" in this context means market/news
  coverage, NOT client notes, transaction memos, or holdings.

IMPORTANT DOMAIN DISTINCTIONS:

- "AAPL coverage", "AAPL news coverage", "AAPL articles", or
  "AAPL news items" -> market_desk.
- "AAPL client note", "note mentioning AAPL", or
  "transaction memo mentioning AAPL" -> notes_desk.
- "AAPL holding", "AAPL position", or "AAPL shares" -> book_qa.
- Words such as "on file", "hold", "coverage", or "substance" must not
  by themselves cause a question to be routed to notes_desk. Determine
  what information is actually being requested.

Examples:

Q: "What AAPL coverage do we hold dated on or before 1 April 2026?
    Give the count and the substance."
-> roles=["market_desk"]

Q: "How many news items are on file for AAPL up to 1 April 2026,
    and what do they cover?"
-> roles=["market_desk"]

Q: "How many AAPL articles predate 1 April 2026, and what are their themes?"
-> roles=["market_desk"]

Q: "Does any client note mention AAPL?"
-> roles=["notes_desk"]

Q: "Do any transaction memos mention AAPL?"
-> roles=["notes_desk"]

Q: "What AAPL position did the client hold on 1 April 2026?"
-> roles=["book_qa"]

Rules:

1. Maximum TWO intents.
2. If the question has one intent, return one item.
3. If it has two independent requests, return two items.
4. NEVER create more than two intents.
5. Each intent must contain:
   - a short sub-question
   - exactly one specialist name
6. Preserve the original client/person and important dates/symbols.
7. Do not answer the question.
8. Do not call any tools.
9. Do not add explanations.

Return ONLY JSON in this exact shape:

{
  "intents": [
    {
      "question": "sub-question",
      "role": "book_qa"
    }
  ]
}

or:

{
  "intents": [
    {
      "question": "first sub-question",
      "role": "kyc_profile"
    },
    {
      "question": "second sub-question",
      "role": "book_qa"
    }
  ]
}
"""

def build_query_decomposer(
    base_url: str,
    api_key: str,
    model_id: str = "valura-fast",
) -> Agent:

    return Agent(
        name="QueryDecomposer",
        model=OpenAILike(
            id=model_id,
            api_key=api_key,
            base_url=base_url,
        ),
        instructions=DECOMPOSER_INSTRUCTIONS,
        markdown=False,
    )


def decompose_question(
    agent: Agent,
    question: str,
) -> list[dict]:

    run = agent.run(question)

    text = getattr(run, "content", None)

    if not text:
        raise ValueError("Query decomposer returned no content.")

    import json

    data = json.loads(text)

    intents = data.get("intents")

    if not isinstance(intents, list):
        raise ValueError("Decomposer did not return an intents list.")

    # INTENTIONALLY SIMPLE VALIDATION.
    # No Pydantic model, no second LLM verifier.
    if len(intents) < 1 or len(intents) > 2:
        raise ValueError(
            f"Invalid intent count: {len(intents)}. "
            "Questions must contain 1 or 2 intents."
        )

    for intent in intents:
        if not isinstance(intent, dict):
            raise ValueError("Invalid intent item.")

        if not intent.get("question"):
            raise ValueError("Intent has no question.")

        if not intent.get("role"):
            raise ValueError("Intent has no role.")

    return intents