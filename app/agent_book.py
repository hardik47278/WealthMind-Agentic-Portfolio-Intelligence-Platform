
from __future__ import annotations
from datetime import date, datetime
from typing import Callable
from agno import agent
from agno import agent
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from app import tools_book as tb
from app.tools_book import Result, first_symbol_buy_date

BOOK_QA_INSTRUCTIONS = """\
You are the book_qa specialist for a regulated investment platform.
You answer questions about ONE client's transactions, cash and holdings by
calling exactly one of the tools available to you. You never compute totals
yourself -- the tools do exact arithmetic; you only pick the right tool and
the right arguments, then phrase a short, natural answer sentence using the
number the tool returned.

Rules:
- Call exactly one tool per question. Once you have called a tool, do not
  call any other tool for that question.

- If at least one part of the question concerns transactions, cash, or
  holdings and can be answered by one of your available tools, call the
  appropriate tool for that Book-related part. Do NOT call not_my_scope
  merely because another part of the question is outside your scope.

- not_my_scope is mutually exclusive with all other tools. Call
  not_my_scope(reason=...) ONLY when NONE of the requested parts can be
  answered by any of your available Book tools. If you call not_my_scope,
  it must be the ONLY tool call for the entire question.

- If a Book tool can answer any requested Book-related part, use that tool
  and STOP. Do not subsequently call not_my_scope because another part of
  the question belongs to KYC, notes, market data, or another specialist.

- If a date is mentioned (e.g. "as at 31 March 2025", "during 2024"), pass it
  through to the tool's as_of/year argument -- do not filter dates yourself.

- If the tool result has a note saying no matching records exist, say so
  plainly in your answer; do not invent a number.

- Never mention any client other than the one you were given data for.

- Keep the answer to one or two sentences. Do not restate raw record ids in
  the prose -- citations are handled separately.

- For a PERIOD ("during July 2024", "between 27 January 2025 and 27 July
  2026", "in Q1 2025"), use count_in_range or sum_in_range with
  start_date/end_date -- as_of only accepts a single exact YYYY-MM-DD
  cutoff and cannot express a range on its own.


Examples:

Example 1 — Book part + another specialist's part:
Question: "What is the cash balance and what notes are on file?"

Correct Book behavior:
1. Call calculate_cash_balance.
2. Use the returned cash balance in the answer.
3. STOP.

Do NOT call not_my_scope just because "notes" belongs to notes_desk.

Incorrect behavior:
calculate_cash_balance -> not_my_scope

Correct behavior:
calculate_cash_balance -> STOP


Example 2 — Book-only question:
Question: "What is the client's current cash balance?"

Correct Book behavior:
1. Call calculate_cash_balance.
2. Use the returned value.
3. STOP.

Do NOT call not_my_scope after receiving the result.


Example 3 — Entirely outside Book scope:
Question: "What notes are on file for this client?"

Correct Book behavior:
1. Call not_my_scope(reason=...).
2. STOP.

Do NOT call calculate_cash_balance or any other Book tool.


Example 4 — Another specialist's question:
Question: "What is the client's latest market price for AAPL?"

Correct Book behavior:
1. Call not_my_scope(reason=...).
2. STOP.

Do NOT call a Book tool just because the question mentions the client.


Example 5 — Book question with irrelevant additional part:
Question: "What were the total deposits, and what KYC information is on file?"

Correct Book behavior:
1. Call total_deposits.
2. Use the returned value.
3. STOP.

Do NOT call not_my_scope because the KYC portion belongs to another specialist.


Example 6 — No Book-supported part:
Question: "What are the client's risk notes and latest market news?"

Correct Book behavior:
1. Call not_my_scope(reason=...).
2. STOP.

Never call a Book transaction, cash, or holdings tool for an unrelated
question.
"""

def _wrap(fn: Callable[..., Result], capture: dict) -> Callable:
    """Wrap a tools_book function so every call's Result is recorded in
    `capture`, and the LLM sees a short string instead of a Result object
    (which it can't use directly as a tool return value)."""

    def wrapped(*args, **kwargs) -> str:
        result: Result = fn(*args, **kwargs)
        tool_args = dict(zip(fn.__code__.co_varnames[1:len(args)], args[1:])) | kwargs
        capture["calls"].append((fn.__name__, result, tool_args))
        
        if result.value is None:
            return result.note or "No matching records."
        if isinstance(result.value, str) and _looks_like_date(result.value):
            return f"{result.value}" + (f" ({result.note})" if result.note else "")
        return f"{result.value}" + (f" ({result.note})" if result.note else "")

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _looks_like_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _parse_as_of(as_of: str | None) -> date | None:
    return datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else None


def build_book_qa_agent(client: dict, base_url: str, api_key: str,
                         model_id: str = "valura-fast") -> tuple[Agent, dict]:
    """Returns (agent, capture). `capture["calls"]` accumulates every
    (tool_name, Result) pair made during the agent's run -- read
    capture["calls"][-1] after agent.run() for the authoritative figure."""
    capture: dict = {"calls": []}
    c = client  # bound via closure -- no other client is ever reachable

    def cash_balance(as_of: str | None = None) -> str:
        """Current cash balance, or as at a given date (YYYY-MM-DD)."""
        return _wrap(tb.calculate_cash_balance, capture)(c, _parse_as_of(as_of))

    def total_deposits(as_of: str | None = None) -> str:
        """Total USD deposited into the account, optionally as at a date."""
        return _wrap(tb.total_deposits, capture)(c, _parse_as_of(as_of))

    def largest_deposit(as_of: str | None = None) -> str:
        """The single largest deposit made, in USD."""
        return _wrap(tb.largest_deposit, capture)(c, _parse_as_of(as_of))

    def total_withdrawals(as_of: str | None = None) -> str:
        """Total USD withdrawn from the account."""
        return _wrap(tb.total_withdrawals, capture)(c, _parse_as_of(as_of))

    def withdrawal_count(as_of: str | None = None) -> str:
        """How many withdrawals the client has made."""
        return _wrap(tb.withdrawal_count, capture)(c, _parse_as_of(as_of))

    def total_fees(as_of: str | None = None) -> str:
        """Total USD in platform fees charged."""
        return _wrap(tb.total_fees, capture)(c, _parse_as_of(as_of))

    def total_dividends(symbol: str | None = None, year: int | None = None,
                         as_of: str | None = None) -> str:
        """Net dividend income received (after withholding tax), optionally
        filtered by symbol and/or calendar year."""
        return _wrap(tb.total_dividends, capture)(c, _parse_as_of(as_of), symbol, year)

    def total_dividend_withholding_tax(symbol: str | None = None,
                                        year: int | None = None,
                                        as_of: str | None = None) -> str:
        """Total withholding tax deducted from dividends -- use this ONLY
        when the question asks about tax withheld, not net income received."""
        return _wrap(tb.total_dividend_withholding_tax, capture)(
            c, _parse_as_of(as_of), symbol, year)

    def last_dividend_date(as_of: str | None = None) -> str:
        """The date of the most recent dividend payment."""
        return _wrap(tb.last_dividend_date, capture)(c, _parse_as_of(as_of))

    def dividend_symbols(as_of: str | None = None) -> str:
        """Which symbol(s) generated dividend income."""
        return _wrap(tb.dividend_symbols, capture)(c, _parse_as_of(as_of))

    def symbol_purchase_count(symbol: str, as_of: str | None = None) -> str:
        """How many separate buy transactions were made for a given symbol."""
        return _wrap(tb.symbol_purchase_count, capture)(c, symbol, _parse_as_of(as_of))

    def symbol_sale_count(symbol: str, as_of: str | None = None) -> str:
        """How many separate sell transactions were made for a given symbol."""
        return _wrap(tb.symbol_sale_count, capture)(c, symbol, _parse_as_of(as_of))

    def total_quantity_bought(symbol: str, as_of: str | None = None) -> str:
        """Total quantity/shares bought of a given symbol, all time."""
        return _wrap(tb.total_quantity_bought, capture)(c, symbol, _parse_as_of(as_of))

    def total_quantity_sold(symbol: str, as_of: str | None = None) -> str:
        """Total quantity/shares sold of a given symbol, all time."""
        return _wrap(tb.total_quantity_sold, capture)(c, symbol, _parse_as_of(as_of))

    def largest_buy(symbol: str | None = None, as_of: str | None = None) -> str:
        """The largest single buy trade by gross USD value, optionally for
        one symbol only."""
        return _wrap(tb.largest_buy, capture)(c, symbol, _parse_as_of(as_of))

    def largest_sell(symbol: str | None = None, as_of: str | None = None) -> str:
        """The largest single sell trade by gross USD value, optionally for
        one symbol only."""
        return _wrap(tb.largest_sell, capture)(c, symbol, _parse_as_of(as_of))

    def first_transaction_date(as_of: str | None = None) -> str:
        """The date of the client's very first transaction."""
        return _wrap(tb.first_transaction_date, capture)(c, _parse_as_of(as_of))

    def first_buy_date(symbol: str) -> str:
        """Return the date of the client's first BUY transaction for a symbol.
    Use this for questions such as:
    "When did the client first buy KO?"
    "When was AAPL first purchased?"
    """ 
        return _wrap(tb.first_symbol_buy_date, capture)(c, symbol)

    def latest_transaction_date(as_of: str | None = None) -> str:
        """The date of the client's most recent transaction."""
        return _wrap(tb.latest_transaction_date, capture)(c, _parse_as_of(as_of))

    def current_holding(symbol: str) -> str:
        """Current quantity held of a given symbol, from the latest
        positions snapshot. Use this for 'current'/'now' holding questions."""
        return _wrap(tb.current_holding_from_snapshot, capture)(c, symbol)

    def distinct_holding_count() -> str:
        """
        Return the number of distinct current holdings.

        USE THIS WHEN:
        - The user asks how many holdings they currently have.
        - The user asks for the number of current positions.
        - The user asks how many different securities they hold.
        - The user asks for the count of distinct current holdings.

        DO NOT USE THIS WHEN:
        - The user asks for the quantity of a specific symbol.
          Use current_holding().
        - The user asks which securities they currently hold.
          Use a holdings-symbols/list tool if available.
        - The user asks about holdings at a past date.
          Use holding_as_of_date().
        - The user asks how many buy/sell transactions occurred.
          Use symbol_purchase_count() or symbol_sale_count().
        - The user asks which symbols generated dividends.
          Use dividend_symbols().

        IMPORTANT:
        Count only distinct symbols with non-zero current quantity
        in the positions snapshot. Do not infer holdings from
        dividend, buy, or sell transactions.
        """
        return _wrap(tb.distinct_holding_count, capture)(c)


    

    def holding_as_of_date(symbol: str, as_of: str) -> str:
        """Quantity held of a given symbol as at a PAST date (YYYY-MM-DD).
        Use this, not current_holding, whenever a specific past date is
        named -- the snapshot is only valid as at the book's as_of date."""
        parsed = _parse_as_of(as_of)
        assert parsed is not None
        return _wrap(tb.holding_as_of, capture)(c, symbol, parsed)


    def count_in_range(txn_type: str, start_date: str, end_date: str,
                        symbol: str | None = None) -> str:
        """Count transactions of a given type between two dates
        (inclusive). Use for period questions like 'how many buys during
        July 2024' -- as_of-based tools only support a single cutoff, not
        a range.

        Args:
            txn_type: one of buy, sell, deposit, withdrawal, fee, dividend.
            start_date: YYYY-MM-DD, inclusive.
            end_date: YYYY-MM-DD, inclusive.
            symbol: optional, restrict to one instrument.
        """
        return _wrap(tb.transactions_in_range, capture)(c, txn_type, start_date, end_date, symbol)

    def sum_in_range(txn_type: str, start_date: str, end_date: str,
                      symbol: str | None = None) -> str:
        """Sum USD amount of transactions of a given type between two
        dates (inclusive). Use for 'deposited between X and Y', 'fees
        during Q3'.

        Args:
            txn_type: one of buy, sell, deposit, withdrawal, fee, dividend.
            start_date: YYYY-MM-DD, inclusive.
            end_date: YYYY-MM-DD, inclusive.
            symbol: optional, restrict to one instrument.
        """
        return _wrap(tb.sum_in_range, capture)(c, txn_type, start_date, end_date, symbol)



    def not_my_scope(reason: str = "") -> str:
        """call this only when no book tool an answer any part of the question"""

        if any(
            tool_name != "not_my_scope"
            for tool_name, _, _ in capture["calls"]
        ):
            return (
                "A Book tool has already been called for this question. "
                "Do not call not_my_scope."
            )


        capture["calls"].append(
            (
            "not_my_scope",
            Result(
                value="NOT_MY_SCOPE",
                record_ids=[],
                note=reason,
            ),
            {"reason": reason},
            )
        )
        

        return "Noted -- this will be routed elsewhere."
        
    
    tools = [
        cash_balance,
        total_deposits,
        largest_deposit,
        total_withdrawals,
        withdrawal_count,
        total_fees,
        total_dividends,
        total_dividend_withholding_tax,
        last_dividend_date,
        dividend_symbols,
        symbol_purchase_count,
        symbol_sale_count,
        total_quantity_bought,
        total_quantity_sold,
        largest_buy,
        largest_sell,
        first_transaction_date,
        first_buy_date,
        latest_transaction_date,
        current_holding,
        distinct_holding_count,
        holding_as_of_date,
        count_in_range,
        sum_in_range,
        not_my_scope,
    ]

    agent = Agent(
    name="BookQA",
    model=OpenAILike(
        id=model_id,
        api_key=api_key,
        base_url=base_url,
        
    ),
    tools=tools,
    instructions=BOOK_QA_INSTRUCTIONS,
    markdown=False,
    
)
    return agent, capture