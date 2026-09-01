
from __future__ import annotations
from dataclasses import field
from dataclasses import field
from datetime import datetime
from typing import Callable
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from app import tools_kyc as tk
from app.tools_book import Result

KYC_PROFILE_INSTRUCTIONS = """
You are the kyc_profile specialist for a regulated investment platform.

You answer questions about ONE client's identity, KYC status,
employment profile, risk profile, and bank account details.

You MUST answer by calling exactly one tool.

Available tools:

- identity_lookup(field=...)
    Supported fields:
    - pan
    - dob
    - address

- kyc_lookup(field=...)
    Supported fields:
    - status
    - risk

- employment_lookup(field=...)
    Supported fields:
    - income
    - employer

- bank_lookup(field=...)
    Supported fields:
    - bank_name
    - account_number
    - ifsc

Rules:

- Call exactly one tool.
- Choose the correct field argument based on what the user asked.
- Never substitute one field for another.
  Example:
    - If asked for PAN, do not return address.
    - If asked for account number, do not return bank name.
    - If asked for employer, do not return income band.

- Whatever the tool returns is final.
- Some fields (PAN and account number) are already masked.
  Never attempt to reveal, reconstruct, or infer hidden digits.

- If a tool reports that a field is not recorded, say so plainly.
  Do not guess and do not search for a nearby field.

- Never mention information from another client.

- Keep answers short (one or two sentences).

- Do not restate record ids in the response.

- If the question is not about identity, KYC status,
  employment, risk profile, or bank details,
  call not_my_scope(reason=...) and stop.

- If a tool's result note describes a disagreement between two records
  (e.g. KYC status vs. a suitability review), state both values plainly
  in your answer and note that they conflict -- do not pick one silently.
"""
def _wrap(fn: Callable[..., Result], capture: dict) -> Callable:
    """Wrap a tools_kyc function so every call's Result is recorded in
    `capture`, and the LLM sees a short string instead of a Result object
    (which it can't use directly as a tool return value). Identical
    pattern to agents_book_qa.py's _wrap, kept local rather than imported
    since it's a two-line adapter, not shared logic.
    """

    def wrapped(*args, **kwargs) -> str:
        result: Result = fn(*args, **kwargs)
        tool_args = dict(zip(fn.__code__.co_varnames[1:len(args)], args[1:])) | kwargs
        capture["calls"].append((fn.__name__, result, tool_args))

        if result.value is None:
            return result.note or "No matching record."

        if isinstance(result.value, str) and _looks_like_date(result.value):
            return f"{result.value}" + (
                f" ({result.note})" if result.note else ""
            )

        return f"{result.value}" + (
            f" ({result.note})" if result.note else ""
        )

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__

    return wrapped


def _looks_like_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def build_kyc_profile_agent(
    client: dict,
    base_url: str,
    api_key: str,
    model_id: str = "valura-fast",
) -> tuple[Agent, dict]:
    """Returns (agent, capture). `capture["calls"]` accumulates every
    (tool_name, Result) pair made during the agent's run -- read
    capture["calls"][-1] after agent.run() for the authoritative figure,
    exactly as agents_book_qa.py does.
    """

    capture: dict = {"calls": []}
    c = client  # bound via closure -- no other client is ever reachable

    def pan(_: str = "") -> str:
        """The client's PAN, masked. Use for any question about PAN,
        identity number, or 'tax ID'.
        """
        return _wrap(tk.get_pan, capture)(c)

    def date_of_birth(_: str = "") -> str:
        """The client's date of birth."""
        return _wrap(tk.get_date_of_birth, capture)(c)

    def address(_: str = "") -> str:
        """The client's recorded address."""
        return _wrap(tk.get_address, capture)(c)

    def kyc_status(_: str = "") -> str:
        """The client's KYC verification status."""
        return _wrap(tk.get_kyc_status, capture)(c)

    def risk_profile(_: str = "") -> str:
        """The client's declared risk profile/appetite on file. This is a
        recorded fact, not a recommendation -- reporting it is not
        investment advice.
        """
        return _wrap(tk.get_risk_profile, capture)(c)

    def annual_income_band(_: str = "") -> str:
        """The client's declared annual income band."""
        return _wrap(tk.get_annual_income_band, capture)(c)

    def employer(_: str = "") -> str:
        """The client's employer or occupation, if recorded. Not every
        client has this field on file.
        """
        return _wrap(tk.get_employer, capture)(c)

    def bank_name(_: str = "") -> str:
        """The name of the bank holding the client's account."""
        return _wrap(tk.get_bank_name, capture)(c)

    def bank_account_number(_: str = "") -> str:
        """The client's bank account number, masked. Use for any question
        about the account number itself.
        """
        return _wrap(tk.get_bank_account_number, capture)(c)

    def bank_ifsc(_: str = "") -> str:
        """The IFSC code of the client's bank account."""
        return _wrap(tk.get_bank_ifsc, capture)(c)

    def not_my_scope(reason: str = "") -> str:
        """Call this INSTEAD of any other tool, and call nothing else, if
        this question is not actually about identity/KYC/employment/risk/bank
        details -- e.g. it's asking about transactions, notes, or market data
        with nothing for kyc_profile to do. `reason` should say what the
        question actually seems to need instead.
        """

        if any(
        tool_name != "not_my_scope"
        for tool_name, _, _ in capture["calls"]
        ):
            return (
                "A KYC tool has already been called for this question"
                "Do not call not-my_scope() after calling other tools."
            )
        
        capture["calls"].append(
            (
                "not_my_scope",
                Result(
                    value="NOT_MY_SCOPE",
                    record_ids=[],
                    note=reason,
                ),
                {"reason":reason},
            )
        )

        return "Noted -- this will be routed elsewhere."

    def identity_lookup(field: str) -> str:
      return _wrap(tk.identity_lookup, capture)(c, field)


    def kyc_lookup(field: str) -> str:
      return _wrap(tk.kyc_lookup, capture)(c, field)


    def employment_lookup(field: str) -> str:
      return _wrap(tk.employment_lookup, capture)(c, field)


    def bank_lookup(field: str) -> str:
      return _wrap(tk.bank_lookup, capture)(c, field)

    


    agent = Agent(
        name="KYCProfile",
        model=OpenAILike(
            id=model_id,
            api_key=api_key,
            base_url=base_url,
            
        ),
        tools=[identity_lookup,kyc_lookup,employment_lookup,bank_lookup,not_my_scope],
        instructions=KYC_PROFILE_INSTRUCTIONS,
        markdown=False,
    )

    return agent, capture

