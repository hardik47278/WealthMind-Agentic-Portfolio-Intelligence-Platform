"""Deterministic query functions over a single client's book data.

These do the arithmetic the brief says the model must never be trusted with:
"language models cannot reliably total a thousand rows and every numeric
answer here is checked exactly." Every function here takes a `client` dict
(already loaded in memory -- see app/service.py) and returns a `Result`:
the exact value (as a Decimal or date), plus the record ids that back it.

No file or network I/O happens in this module. The client dict is the only
input, which is what keeps scope airtight: a function bound to one client's
dict physically cannot reach another client's records.

All money fields in the book arrive as strings ("179.32") specifically so
callers use Decimal, not float, and avoid drift across thousands of rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP


CITATION_LIST_MAX = 6  # brief: >6 records -> cite the client id instead


@dataclass
class Result:
    value: Decimal | str | int | None
    record_ids: list[str] = field(default_factory=list)
    note: str | None = None  # e.g. "no dividend transactions found"
    conflict: bool = False

    def citations(self, client_id: str) -> list[str]:
        """Apply the citation rule: <=6 -> list the records, >6 -> the
        client id instead. A figure derived from four hundred transactions
        is not made more auditable by four hundred citations."""
        if not self.record_ids:
            return []
        if len(self.record_ids) <= CITATION_LIST_MAX:
            return list(self.record_ids)
        return [client_id]

    def money_str(self) -> str:
        """USD, no symbol, no thousands separator, 2dp, exact."""
        assert isinstance(self.value, Decimal)
        q = self.value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return f"{q}"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _d(x: str | None) -> Decimal:
    return Decimal(x) if x is not None else Decimal("0")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def get_transactions(client: dict, as_of: date | None = None) -> list[dict]:
    """All transactions, optionally filtered to on-or-before a given date.
    'As at a past date' means as at the end of that date -- records after it
    must be ignored, which is exactly what this filter does."""
    txns = client.get("transactions", [])
    if as_of is None:
        return txns
    return [t for t in txns if _parse_date(t["date"]) <= as_of]


def _by_type(client: dict, ttype: str, as_of: date | None = None) -> list[dict]:
    return [t for t in get_transactions(client, as_of) if t["type"] == ttype]


def get_deposits(client: dict, as_of: date | None = None) -> list[dict]:
    return _by_type(client, "deposit", as_of)


def get_withdrawals(client: dict, as_of: date | None = None) -> list[dict]:
    return _by_type(client, "withdrawal", as_of)


def get_fees(client: dict, as_of: date | None = None) -> list[dict]:
    return _by_type(client, "fee", as_of)


def get_dividends(client: dict, as_of: date | None = None) -> list[dict]:
    return _by_type(client, "dividend", as_of)


def get_buys(client: dict, as_of: date | None = None) -> list[dict]:
    return _by_type(client, "buy", as_of)


def get_sells(client: dict, as_of: date | None = None) -> list[dict]:
    return _by_type(client, "sell", as_of)


def get_symbol_buys(client: dict, symbol: str, as_of: date | None = None) -> list[dict]:
    return [t for t in get_buys(client, as_of) if t.get("symbol") == symbol]


def get_symbol_sells(client: dict, symbol: str, as_of: date | None = None) -> list[dict]:
    return [t for t in get_sells(client, as_of) if t.get("symbol") == symbol]


def _flow_amount(t: dict) -> Decimal:
    """The USD amount that hits cash for a single transaction, unsigned.
    deposit/withdrawal/fee carry amount_usd; buy/sell/dividend carry
    net_usd (fees already applied). Falls back gracefully if a field is
    named differently than expected -- check against your real book and
    adjust the .get() chain if any type doesn't match."""
    return _d(t.get("amount_usd") or t.get("net_usd"))


# ---------------------------------------------------------------------------
# Cash
# ---------------------------------------------------------------------------
def calculate_cash_balance(client: dict, as_of: date | None = None) -> Result:
    
    """

    Return the client's cash balance as calculated from transactions,
    optionally as of a specified date.

    USE THIS WHEN:
    - The user asks for cash balance or cash available.
    - The user asks "What was my cash balance as of <date>?"

    DO NOT USE THIS WHEN:
    - The user asks for total deposits only. Use total_deposits().
    - The user asks for total withdrawals only. Use total_withdrawals().
    - The user asks for a current securities holding. Use
      current_holding_from_snapshot().

    `as_of` is an inclusive calendar-date cutoff.

    """
    
    txns = get_transactions(client, as_of)
    total = Decimal("0")
    ids: list[str] = []
    for t in txns:
        amt = _flow_amount(t)
        if t["type"] in ("deposit", "sell", "dividend"):
            total += amt
        elif t["type"] in ("withdrawal", "fee", "buy"):
            total -= amt
        ids.append(t["id"])
    return Result(value=total, record_ids=ids)


def total_deposits(client: dict, as_of: date | None = None) -> Result:
    """
    Return the total amount deposited into the client's account.

    USE THIS WHEN
    - The user asks "How much have I deposited?"
    - The user asks for deposits up to a specific date.

    DO NOT USE THIS WHEN:
    - The user asks for withdrawals. Use total_withdrawals().
    - The user asks for deposits during a start/end period. Use the
      deposit range tool.
    """
    txns = get_deposits(client, as_of)
    total = sum((_flow_amount(t) for t in txns), Decimal("0"))
    return Result(value=total, record_ids=[t["id"] for t in txns])


def largest_deposit(client: dict, as_of: date | None = None) -> Result:
    """
    Return the largest single deposit transaction.

    USE THIS WHEN:
    - The user asks for the biggest/largest deposit.

    DO NOT USE THIS WHEN:
    - The user asks for total deposited amount. Use total_deposits().
    - The user asks for deposit count. Use the deposit-count tool.
    """
    txns = get_deposits(client, as_of)
    if not txns:
        return Result(value=None, record_ids=[], note="no deposits on record")
    best = max(txns, key=lambda t: _flow_amount(t))
    return Result(value=_flow_amount(best), record_ids=[best["id"]])


def total_withdrawals(client: dict, as_of: date | None = None) -> Result:
    """
    Return the total amount withdrawn from the client's account.

    USE THIS WHEN:
    - The user asks how much they withdrew, optionally as of a date.

    DO NOT USE THIS WHEN:
    - The user asks for deposits. Use total_deposits().
    - The user asks for withdrawals during a specific date range.
      Use the withdrawal range tool.
    """
    txns = get_withdrawals(client, as_of)
    total = sum((_flow_amount(t) for t in txns), Decimal("0"))
    return Result(value=total, record_ids=[t["id"] for t in txns])


def withdrawal_count(client: dict, as_of: date | None = None) -> Result:
    txns = get_withdrawals(client, as_of)
    return Result(value=len(txns), record_ids=[t["id"] for t in txns])


def total_fees(client: dict, as_of: date | None = None) -> Result:
    txns = get_fees(client, as_of)
    total = sum((_flow_amount(t) for t in txns), Decimal("0"))
    return Result(value=total, record_ids=[t["id"] for t in txns])


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------
def total_dividends(client: dict, as_of: date | None = None,
                     symbol: str | None = None,
                     year: int | None = None) -> Result:
    txns = get_dividends(client, as_of)
    if symbol:
        txns = [t for t in txns if t.get("symbol") == symbol]
    if year:
        txns = [t for t in txns if _parse_date(t["date"]).year == year]
    total = sum((_flow_amount(t) for t in txns), Decimal("0"))
    if not txns:
        return Result(value=Decimal("0"), record_ids=[],
                       note="no dividend transactions match")
    return Result(value=total, record_ids=[t["id"] for t in txns])


def last_dividend_date(client: dict, as_of: date | None = None) -> Result:
    txns = get_dividends(client, as_of)
    if not txns:
        return Result(value=None, record_ids=[], note="no dividends on record")
    latest = max(txns, key=lambda t: _parse_date(t["date"]))
    return Result(value=latest["date"], record_ids=[latest["id"]])


def total_dividend_withholding_tax(client: dict, as_of: date | None = None,
                                    symbol: str | None = None,
                                    year: int | None = None) -> Result:
    """Dividends carry gross_usd, withholding_tax_usd and net_usd
    separately -- 'net dividend income' means net_usd (see
    total_dividends), but a question about tax withheld wants this."""
    txns = get_dividends(client, as_of)
    if symbol:
        txns = [t for t in txns if t.get("symbol") == symbol]
    if year:
        txns = [t for t in txns if _parse_date(t["date"]).year == year]
    total = sum((_d(t.get("withholding_tax_usd")) for t in txns), Decimal("0"))
    if not txns:
        return Result(value=Decimal("0"), record_ids=[],
                       note="no dividend transactions match")
    return Result(value=total, record_ids=[t["id"] for t in txns])


def dividend_symbols(client: dict, as_of: date | None = None) -> Result:
    txns = get_dividends(client, as_of)
    symbols = sorted({t["symbol"] for t in txns if t.get("symbol")})
    return Result(value=", ".join(symbols) if symbols else None,
                   record_ids=[t["id"] for t in txns],
                   note=None if symbols else "no dividends on record")


# ---------------------------------------------------------------------------
# Trading activity
# ---------------------------------------------------------------------------
def symbol_purchase_count(client: dict, symbol: str,
                           as_of: date | None = None) -> Result:
    txns = get_symbol_buys(client, symbol, as_of)
    return Result(value=len(txns), record_ids=[t["id"] for t in txns])


def symbol_sale_count(client: dict, symbol: str,
                       as_of: date | None = None) -> Result:
    txns = get_symbol_sells(client, symbol, as_of)
    return Result(value=len(txns), record_ids=[t["id"] for t in txns])


def total_quantity_bought(client: dict, symbol: str,
                           as_of: date | None = None) -> Result:
    txns = get_symbol_buys(client, symbol, as_of)
    total = sum((_d(t.get("quantity")) for t in txns), Decimal("0"))
    return Result(value=total, record_ids=[t["id"] for t in txns])


def total_quantity_sold(client: dict, symbol: str,
                         as_of: date | None = None) -> Result:
    txns = get_symbol_sells(client, symbol, as_of)
    total = sum((_d(t.get("quantity")) for t in txns), Decimal("0"))
    return Result(value=total, record_ids=[t["id"] for t in txns])


def largest_buy(client: dict, symbol: str | None = None,
                 as_of: date | None = None) -> Result:
    txns = get_symbol_buys(client, symbol, as_of) if symbol else get_buys(client, as_of)
    if not txns:
        return Result(value=None, record_ids=[], note="no buy transactions match")
    best = max(txns, key=lambda t: _d(t.get("gross_usd")))
    return Result(value=_d(best.get("gross_usd")), record_ids=[best["id"]])


def largest_sell(client: dict, symbol: str | None = None,
                  as_of: date | None = None) -> Result:
    
    txns = get_symbol_sells(client, symbol, as_of) if symbol else get_sells(client, as_of)
    if not txns:
        return Result(value=None, record_ids=[], note="no sell transactions match")
    best = max(txns, key=lambda t: _d(t.get("gross_usd")))
    return Result(value=_d(best.get("gross_usd")), record_ids=[best["id"]])


def first_transaction_date(client: dict, as_of: date | None = None) -> Result:
    """
    Return the earliest transaction date across the client's entire account.

    USE THIS TOOL WHEN:
    - The user asks for the first transaction in the account.
    - Examples:
      * "When was my first transaction?"
      * "What is the earliest transaction on my account?"

    DO NOT USE THIS TOOL WHEN:
    - The user asks for the first BUY of a specific symbol.
      Use first_symbol_buy_date(symbol).
    - The user asks for the first SELL of a specific symbol.
      Use the first-symbol-sale tool.
    """
    txns = get_transactions(client, as_of)
    if not txns:
        return Result(value=None, record_ids=[], note="no transactions on record")
    earliest = min(txns, key=lambda t: _parse_date(t["date"]))
    return Result(value=earliest["date"], record_ids=[earliest["id"]])


def first_symbol_buy_date(client: dict, symbol: str) -> Result:
    """Return the earliest BUY transaction date for the specified symbol.

    Use this for questions such as:
    - "When did I first buy KO?"
    - "When was AAPL first purchased?"
    - "What was the first date I bought MSFT?"

    Does NOT:
    - return the account's earliest transaction date;
    - answer first SELL/disposal questions;
    - answer purchase-count or quantity questions.

    For the account's first transaction, use first_transaction_date().
    """
    buys = [
        t for t in _by_type(client, "buy")
        if t.get("symbol") == symbol
    ]

    if not buys:
        return Result(
            value=None,
            record_ids=[],
            note=f"no buy transactions on record for {symbol}",
        )

    earliest = min(
        buys,
        key=lambda t: _parse_date(t["date"]),
    )

    return Result(
        value=earliest["date"],
        record_ids=[earliest["id"]],
    )




def latest_transaction_date(client: dict, as_of: date | None = None) -> Result:
    txns = get_transactions(client, as_of)
    if not txns:
        return Result(value=None, record_ids=[], note="no transactions on record")
    latest = max(txns, key=lambda t: _parse_date(t["date"]))
    return Result(value=latest["date"], record_ids=[latest["id"]])


# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------
def current_holding_from_snapshot(client: dict, symbol: str) -> Result:
    """Prefer positions_snapshot for *current* holdings -- it's already
    computed and dated (meta.as_of), so recomputing from years of trades
    would be redundant and slower."""
    snap = client.get("positions_snapshot", [])
    for p in snap:
        if p.get("symbol") == symbol:
            return Result(value=_d(p["quantity"]), record_ids=[p["id"]])
    return Result(value=Decimal("0"), record_ids=[],
                   note=f"no current {symbol} position in snapshot")

def distinct_holding_count(client: dict) -> Result:
    """
    Return the number of distinct current holdings in the client's
    positions snapshot.

    USE THIS TOOL WHEN:
    - The user asks how many distinct holdings they currently have.
    - The user asks for the number of current holdings or positions.
    - The user asks "How many holdings do I have?"
    - The user asks "How many different securities do I currently hold?"
    - The user asks for the count of distinct symbols in the current
      portfolio.

    DO NOT USE THIS TOOL WHEN:
    - The user asks for the quantity of a specific holding.
      Use current_holding_from_snapshot(symbol).
    - The user asks which securities they currently hold.
      Use current_holding_symbols().
    - The user asks about holdings as of a historical date.
      Use holding_as_of(symbol, as_of).
    - The user asks how many times they purchased a security.
      Use symbol_purchase_count(symbol).
    - The user asks how many times they sold a security.
      Use symbol_sale_count(symbol).
    - The user asks which symbols paid dividends.
      Use dividend_symbols().
    - The user asks for the total quantity bought or sold.
      Use total_quantity_bought() or total_quantity_sold().

    IMPORTANT:
    - Count distinct symbols from positions_snapshot.
    - Count only positions with non-zero quantity.
    - Do not infer holdings from dividend transactions, buy transactions,
      or sell transactions.
    - Do not use dividend_symbols() to answer a holdings-count question.

    Returns:
        Result containing the exact integer count and the supporting
        position record IDs.
    """

    positions = client.get("positions_snapshot", [])

    symbols = {
        p["symbol"]
        for p in positions
        if p.get("symbol") and _d(p.get("quantity")) != 0
    }

    return Result(
        value=len(symbols),
        record_ids=[
            p["id"]
            for p in positions
            if p.get("symbol") in symbols
        ],
    )


def holding_as_of(client: dict, symbol: str, as_of: date) -> Result:
    """For a historical/as-at-date holding, the snapshot (which is only
    valid as at meta.as_of) cannot be used -- reconstruct from transactions
    up to and including that date instead."""
    buys = get_symbol_buys(client, symbol, as_of)
    sells = get_symbol_sells(client, symbol, as_of)
    qty = sum((_d(t.get("quantity")) for t in buys), Decimal("0")) \
        - sum((_d(t.get("quantity")) for t in sells), Decimal("0"))
    ids = [t["id"] for t in buys] + [t["id"] for t in sells]
    return Result(value=qty, record_ids=ids)


def _in_range(client: dict, ttype: str, start_date: str, end_date: str,
              symbol: str | None = None) -> list[dict]:
    start, end = _parse_date(start_date), _parse_date(end_date)
    txns = _by_type(client, ttype)
    if symbol:
        txns = [t for t in txns if t.get("symbol") == symbol]
    return [t for t in txns if start <= _parse_date(t["date"]) <= end]


def _out_of_range_note(client: dict, start_date: str, end_date: str) -> str | None:
    all_txns = get_transactions(client)
    if not all_txns:
        return "no transactions on record for this account at all"
    earliest = min(t["date"] for t in all_txns)
    latest = max(t["date"] for t in all_txns)
    if end_date < earliest:
        return f"requested period ends before the account's earliest record ({earliest})"
    if start_date > latest:
        return f"requested period starts after the account's latest record ({latest})"
    return None


def transactions_in_range(client: dict, txn_type: str, start_date: str,
                           end_date: str, symbol: str | None = None) -> Result:
    txns = _in_range(client, txn_type, start_date, end_date, symbol)
    if not txns:
        note = _out_of_range_note(client, start_date, end_date) \
            or f"no {txn_type} transactions between {start_date} and {end_date}"
        return Result(value=0, record_ids=[], note=note)
    return Result(value=len(txns), record_ids=[t["id"] for t in txns])


def sum_in_range(client: dict, txn_type: str, start_date: str,
                  end_date: str, symbol: str | None = None) -> Result:
    txns = _in_range(client, txn_type, start_date, end_date, symbol)
    if not txns:
        note = _out_of_range_note(client, start_date, end_date) \
            or f"no {txn_type} transactions between {start_date} and {end_date}"
        return Result(value=Decimal("0"), record_ids=[], note=note)
    total = sum((_flow_amount(t) for t in txns), Decimal("0"))
    return Result(value=total, record_ids=[t["id"] for t in txns])    