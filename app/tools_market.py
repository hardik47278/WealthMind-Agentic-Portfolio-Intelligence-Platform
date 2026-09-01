
from __future__ import annotations
from decimal import Decimal
from app.tools_book import Result
ABSTAIN = "ABSTAIN:"
MAX_CITED_IDS = 6
def _covered(market: dict) -> set[str]:
    return set(market.get("meta", {}).get("covered_symbols", []))

def _instrument(market: dict, symbol: str) -> dict | None:
    for row in market.get("instruments", []):
        if row.get("symbol") == symbol:
            return row
    return None

def _cap_ids(ids: list[str]) -> tuple[list[str], int]:
    """Returns (ids_to_cite, total_count). If total_count > len(ids_to_cite),
    the caller must say so in `note` -- the cap is a citation-list-size
    control, never a silent drop of evidence."""
    total = len(ids)
    if total <= MAX_CITED_IDS:
        return ids, total
    return ids[:MAX_CITED_IDS], total

def is_covered(market: dict, symbol: str) -> Result:
    """Whether `symbol` is inside this dataset's coverage at all. Call
    this first for any symbol you're not already sure about -- a 'no'
    means there is no price, sector, or news for it here, regardless of
    how well-known the name is."""
    covered = symbol in _covered(market)
    return Result(value=covered, record_ids=[],
                  note=None if covered else f"{ABSTAIN} {symbol} is outside this dataset's coverage")

def get_instrument(market: dict, symbol: str) -> Result:
    """Sector, industry, currency and listing venue for `symbol`. Returns
    no value if the symbol is not covered -- do not guess a sector for an
    uncovered symbol from general knowledge."""
    if symbol not in _covered(market):
        return Result(value=None, record_ids=[],
                       note=f"{ABSTAIN} {symbol} is outside this dataset's coverage -- no sector/instrument data exists here")
    row = _instrument(market, symbol)
    if row is None:
        return Result(value=None, record_ids=[], note=f"{ABSTAIN} {symbol} listed as covered but no instrument row found (data gap)")
    summary = f"{row.get('sector')} / {row.get('industry')}, {row.get('currency')}, listed on {row.get('listed_on')}"
    return Result(value=summary, record_ids=[symbol])

def get_latest_price(market: dict, symbol: str) -> Result:
    """The most recent monthly close on file for `symbol`, and the date
    it's from. Returns no value if the symbol is not covered."""
    if symbol not in _covered(market):
        return Result(value=None, record_ids=[],
                       note=f"{ABSTAIN} {symbol} is outside this dataset's coverage -- no price exists here")
    series = market.get("prices", {}).get(symbol, [])
    if not series:
        return Result(value=None, record_ids=[], note=f"{ABSTAIN} {symbol} is covered but has no price points on file")
    latest = series[-1]  # confirmed ascending-sorted for every symbol
    return Result(value=latest["close"], record_ids=[f"{symbol}:{latest['date']}"],
                  note=f"close as of {latest['date']}")

def get_price_as_of(market: dict, symbol: str, date: str) -> Result:
    """The close for `symbol` as of `date` (YYYY-MM-DD): since prices are
    MONTHLY closes, this is the most recent close on or before `date`,
    never the exact date unless it happens to be a month-start. The
    returned note always states which date the close is actually from --
    every as-of answer must say so.

    Args:
        symbol: instrument ticker.
        date: the date being asked about, YYYY-MM-DD.
    """
    if symbol not in _covered(market):
        return Result(value=None, record_ids=[],
                       note=f"{symbol} is outside this dataset's coverage -- no price exists here")
    series = market.get("prices", {}).get(symbol, [])
    candidates = [p for p in series if p["date"] <= date]
    if not candidates:
        earliest = series[0]["date"] if series else "unknown"
        return Result(value=None, record_ids=[],
                       note=f"no {symbol} close on or before {date} -- earliest point on file is {earliest}")
    point = candidates[-1]  # series confirmed ascending-sorted, so last match is most recent <= date
    return Result(value=point["close"], record_ids=[f"{symbol}:{point['date']}"],
                  note=f"most recent close on or before {date} is dated {point['date']}")

def get_percentage_return_as_of(
    market: dict,
    symbol: str,
    start_date: str,
    end_date: str,
) -> Result:
    """Calculate the percentage return between two dates using the
    deterministic monthly market-price records.

    Use this tool when the user asks for:
      - percentage return between two dates
      - percentage gain/loss between two dates
      - stock return over a specified period
      - change in price expressed as a percentage

    The function uses get_price_as_of() for both dates, meaning it uses
    the most recent monthly close on or before each requested date.

    Do NOT use this tool when:
      - the user asks only for the price on a date
        -> use get_price_as_of()
      - the user asks for the latest price
        -> use get_latest_price()
      - the user asks about sector, industry, currency, or exchange
        -> use get_instrument()
      - the user asks about news
        -> use the appropriate news tool
      - the user asks for investment advice or what they should buy/sell
        -> do not calculate a recommendation; compliance handles scope

    The tool performs the arithmetic itself. The LLM must not calculate
    the percentage return from two separately retrieved prices.

    Return:
      Result.value = deterministic percentage return.
      Result.record_ids = the price records used for both dates.
      Result.note = the actual dates of the monthly closes used.
    """
    if symbol not in _covered(market):
        return Result(
            value=None,
            record_ids=[],
            note=f"{ABSTAIN} {symbol} is outside this dataset's coverage -- no price exists here",
        )

    start = get_price_as_of(market, symbol, start_date)
    if start.value is None:
        return Result(
            value=None,
            record_ids=start.record_ids,
            note=f"{ABSTAIN} cannot calculate return: start date {start_date}: {start.note}",
        )

    end = get_price_as_of(market, symbol, end_date)
    if end.value is None:
        return Result(
            value=None,
            record_ids=end.record_ids,
            note=f"{ABSTAIN} cannot calculate return: end date {end_date}: {end.note}",
        )

    start_price = Decimal(str(start.value))
    end_price = Decimal(str(end.value))

    if start_price == 0:
        return Result(
            value=None,
            record_ids=start.record_ids + end.record_ids,
            note=f"{ABSTAIN} cannot calculate percentage return because the starting price is zero",
        )

    percentage_return = (end_price - start_price) / start_price * Decimal("100")

    return Result(
        value=percentage_return,
        record_ids=start.record_ids + end.record_ids,
        note=(
            f"return from {start.note}; "
            f"end price from {end.note}"
        ),
    )



def get_latest_news(market: dict, symbol: str, cutoff_date: str | None = None) -> Result:
    """The most recent news item on file for `symbol` (headline + body),
    by date. Returns no value if the symbol is not covered or has no
    news on file -- absence of news is a real, reportable fact, not a
    reason to invent a headline."""
    if symbol not in _covered(market):
        return Result(value=None, record_ids=[],
                       note=f"{symbol} is outside this dataset's coverage -- no news exists here")
    items = [n for n in market.get("news", []) if n.get("symbol") == symbol]
    if not items:
        return Result(value=None, record_ids=[], note=f"no news on file for {symbol}")
    latest = max(items, key=lambda n: n.get("date", ""))
    if cutoff_date and latest.get("date") > cutoff_date:
        return Result(value=None, record_ids=[], note=f"no news on file for {symbol} before {cutoff_date}")
    return Result(value=f"{latest['headline']} -- {latest['body']}",
                  record_ids=[latest.get("id")],
                  note=f"dated {latest.get('date')}, source {latest.get('source')}")



def search_news(market: dict, keyword: str) -> Result:
    """News items (any symbol) whose headline or body contains `keyword`
    (case-insensitive substring). Match count is the deterministic value;
    verbatim matches with symbol/date/id are in `note`.

    Args:
        keyword: substring to search for, case-insensitive.
    """
    kw = (keyword or "").lower()
    matches = [n for n in market.get("news", [])
               if kw in n.get("headline", "").lower() or kw in n.get("body", "").lower()]
    if not matches:
        return Result(value=0, record_ids=[], note=f"no news matched '{keyword}'")
    detail = " | ".join(
        f"[{n.get('id')}] {n.get('date')} {n.get('symbol')}: {n.get('headline')} -- {n.get('body')}"
        for n in matches
    )
    return Result(value=len(matches), record_ids=[n.get("id") for n in matches], note=detail)    

def search_news(market: dict, keyword: str) -> Result:
    """Search MARKET NEWS across all covered companies/symbols.

    PURPOSE:
    Use this tool when the question asks about news, coverage, headlines,
    announcements, events, developments, or other MARKET NEWS and does
    NOT name a specific company/ticker.

    IMPORTANT:
    - This searches market["news"] only.
    - It does NOT search client notes.
    - It does NOT search transaction memos.
    - It does NOT search buy memos or fee descriptions.
    - If the question names a specific company/ticker such as AAPL,
      use search_news_by_symbol() instead.

    Examples:
        "Any news about mergers?"
            -> keyword="merger"

        "What news is on file about CFO appointments?"
            -> keyword="CFO"

    DO NOT use this tool when:
    - The question names a specific ticker/company.
      -> use search_news_by_symbol()
    - The question asks about a client's transaction memo.
      -> use transaction_memos_containing()
    - The question asks about client notes.
      -> use the appropriate notes tool.

    Args:
        keyword:
            Substring to search for in the market-news headline or body,
            case-insensitive.
    """

    kw = (keyword or "").lower()

    matches = [
        n
        for n in market.get("news", [])
        if (
            kw in n.get("headline", "").lower()
            or kw in n.get("body", "").lower()
        )
    ]

    if not matches:
        return Result(
            value=0,
            record_ids=[],
            note=f"no market news matched '{keyword}'",
        )

    detail = " | ".join(
        f"[{n.get('id')}] {n.get('date')} {n.get('symbol')}: "
        f"{n.get('headline')} -- {n.get('body')}"
        for n in matches
    )

    return Result(
        value=len(matches),
        record_ids=[n.get("id") for n in matches],
        note=detail,
    )


def search_news_by_symbol(
    market: dict,
    symbol: str,
    keyword: str,
    cutoff_date: str | None = None,
) -> Result:
    """Search MARKET NEWS for ONE specific company/ticker.

    PURPOSE:
    Use this tool whenever the question asks about NEWS or COVERAGE
    for a named company, stock, or ticker.

    Examples:
        "What news do we have on AAPL?"
            -> symbol="AAPL", keyword=""

        "Any AAPL news about earnings?"
            -> symbol="AAPL", keyword="earnings"

        "How many AAPL news items were on file up to 1 April 2026?"
            -> symbol="AAPL",
               keyword="",
               cutoff_date="2026-04-01"

        "What AAPL coverage do we hold dated on or before 1 April 2026?"
            -> symbol="AAPL",
               keyword="",
               cutoff_date="2026-04-01"

    IMPORTANT DOMAIN RULE:
    This tool searches MARKET NEWS, not client notes or transaction
    memos.

    If a question uses words such as:
    - news
    - coverage
    - headlines
    - announcement
    - market developments
    - company events
    - press/analyst coverage

    and names a company/ticker, use THIS tool.

    DO NOT confuse "coverage" with client transaction-memo text.

    For example:

        "What AAPL coverage do we hold?"
            -> THIS TOOL

        "How many AAPL news items are on file?"
            -> THIS TOOL

        "What does the AAPL coverage say?"
            -> THIS TOOL

        "Does the client's transaction memo mention AAPL?"
            -> transaction_memos_containing()

    DATE RULE:
    If the question contains a date boundary such as:
        - up to 1 April 2026
        - before 1 April 2026
        - on or before 1 April 2026
        - through 1 April 2026

    convert it to cutoff_date="2026-04-01".

    When cutoff_date is provided, ONLY news dated on or before that
    date is included.

    When cutoff_date is None, all matching news for the symbol is
    included.

    KEYWORD RULE:
    If the user asks about the company's news generally, use:
        keyword=""

    Do not invent a topic keyword merely because the question contains
    words such as "coverage", "news", or "on file".

    SYMBOL RULE:
    This tool is for ONE named symbol/company.

    Do NOT use search_news_global() when the question names a specific
    company/ticker. The global tool searches across unrelated symbols.

    DATA SOURCE:
    Searches market["news"] only.

    It does NOT search:
    - client["notes"]
    - transaction memos
    - buy.memo
    - fee.description

    Args:
        symbol:
            Instrument ticker, e.g. "AAPL".

        keyword:
            Optional substring to search for in the news headline/body,
            case-insensitive.

            Use an empty string when the question asks for all news
            about the company rather than a specific topic.

        cutoff_date:
            Optional YYYY-MM-DD date.

            When provided, only news dated on or before this date
            is included.
    """

    if symbol not in _covered(market):
        return Result(
            value=None,
            record_ids=[],
            note=(
                f"{ABSTAIN} {symbol} is outside this dataset's coverage "
                f"-- no market news exists here"
            ),
        )

    kw = (keyword or "").lower()

    matches = [
        n
        for n in market.get("news", [])
        if (
            n.get("symbol") == symbol
            and (
                kw in n.get("headline", "").lower()
                or kw in n.get("body", "").lower()
            )
            and (
                cutoff_date is None
                or n.get("date", "") <= cutoff_date
            )
        )
    ]

    if not matches:
        if cutoff_date:
            return Result(
                value=0,
                record_ids=[],
                note=(
                    f"no {symbol} market news matched '{keyword}' "
                    f"on or before {cutoff_date}"
                ),
            )

        return Result(
            value=0,
            record_ids=[],
            note=f"no {symbol} market news matched '{keyword}'",
        )

    ids, total = _cap_ids(
        [n.get("id") for n in matches]
    )

    detail = " | ".join(
        f"[{n.get('id')}] {n.get('date')}: "
        f"{n.get('headline')} -- {n.get('body')}"
        for n in matches
        if n.get("id") in ids
    )

    if total > len(ids):
        detail += (
            f" (+{total - len(ids)} more matches not shown)"
        )

    if cutoff_date:
        detail = (
            f"market news on or before {cutoff_date}: "
            + detail
        )

    return Result(
        value=len(matches),
        record_ids=ids,
        note=detail,
    )


def search_news_global(market: dict, keyword: str) -> Result:
    """News across ALL covered symbols, filtered by keyword, with NO
    symbol scoping. Use ONLY when the question does not name a specific
    company -- e.g. 'any news about a merger' with no ticker mentioned.
    If the question names a symbol, use search_news_by_symbol instead --
    this tool will return matches from unrelated companies too.

    Args:
        keyword: substring to search for, case-insensitive.
    """
    kw = (keyword or "").lower()
    matches = [n for n in market.get("news", [])
               if kw in n.get("headline", "").lower() or kw in n.get("body", "").lower()]
    if not matches:
        return Result(value=0, record_ids=[], note=f"no news matched '{keyword}'")
    ids, total = _cap_ids([n.get("id") for n in matches])
    detail = " | ".join(
        f"[{n.get('id')}] {n.get('date')} {n.get('symbol')}: {n.get('headline')} -- {n.get('body')}"
        for n in matches if n.get("id") in ids
    )
    if total > len(ids):
        detail += f" (+{total - len(ids)} more matches not shown)"
    return Result(value=len(matches), record_ids=ids, note=detail)

def get_sector_allocation(
    client: dict,
    market: dict,
    sector: str,
) -> Result:
    """
    Return the client's CURRENT portfolio allocation percentage
    for one specific market sector.

    WHEN TO USE:
    - Use when the user asks for the CURRENT percentage/allocation
      of the portfolio in a specific sector.
    - Examples:
        "What percentage of Shreya Kapoor's portfolio is in
         Communication Services?"
        "How much of the portfolio is in Technology?"
        "What is the current allocation to Healthcare?"

    WHEN NOT TO USE:
    - Do NOT use for allocation DRIFT or difference from target.
      Use get_allocation_drift instead.
    - Do NOT use for TARGET allocation.
    - Do NOT use for a specific stock's allocation.
    - Do NOT use for stock returns or price performance.
    - Do NOT use for historical portfolio allocation.
    - Do NOT use for transaction questions.
    - Do NOT use when no sector is specified.

    LOGIC:
    1. Find all covered instruments belonging to the requested sector.
    2. Calculate each holding's current portfolio percentage using
       market_value_usd / total covered portfolio market value * 100.
    3. Sum the current percentages of all holdings in the requested sector.
    4. Return ONLY that summed percentage in Result.value.

    RETURN CONTRACT:
    Result.value must be the single numeric sector allocation.
    Do NOT return the complete allocation table.
    Do NOT return target percentages.
    Do NOT return drift values.
    """

    if not sector or not sector.strip():
        return Result(
            value=None,
            record_ids=[],
            note=f"{ABSTAIN} sector was not specified",
        )

    requested_sector = sector.strip().lower()

    positions = client.get("positions_snapshot", [])

    if not positions:
        return Result(
            value=None,
            record_ids=[],
            note=f"{ABSTAIN} no positions snapshot on record",
        )

    covered = _covered(market)

    # ---------------------------------------------------------
    # Find symbols belonging to the requested sector
    # ---------------------------------------------------------
    sector_symbols = {
        row.get("symbol")
        for row in market.get("instruments", [])
        if (
            row.get("symbol") in covered
            and str(row.get("sector", "")).strip().lower()
            == requested_sector
        )
    }

    if not sector_symbols:
        return Result(
            value=None,
            record_ids=[],
            note=(
                f"{ABSTAIN} no covered instruments found for "
                f"sector '{sector}'"
            ),
        )

    # ---------------------------------------------------------
    # Same current-allocation calculation used by
    # get_allocation_drift()
    # ---------------------------------------------------------
    total_value = sum(
        Decimal(str(p.get("market_value_usd", 0)))
        for p in positions
        if p.get("symbol") in covered
    )

    if total_value <= 0:
        return Result(
            value=None,
            record_ids=[],
            note=(
                f"{ABSTAIN} total covered portfolio market value "
                f"is zero or unavailable"
            ),
        )

    sector_value = Decimal("0")
    matched_symbols = []
    record_ids = []

    for position in positions:
        symbol = position.get("symbol")

        if symbol not in sector_symbols:
            continue

        market_value = Decimal(
            str(position.get("market_value_usd", 0))
        )

        sector_value += market_value

        if symbol not in matched_symbols:
            matched_symbols.append(symbol)

        if position.get("id"):
            record_ids.append(position["id"])

    if not matched_symbols:
        return Result(
            value=None,
            record_ids=[],
            note=(
                f"{ABSTAIN} client has no current holdings in "
                f"sector '{sector}'"
            ),
        )

    allocation = (
        sector_value / total_value
    ) * Decimal("100")

    return Result(
        value=allocation,
        record_ids=record_ids,
        note=(
            f"current allocation for {sector}: "
            f"{', '.join(sorted(matched_symbols))} = "
            f"{allocation:.2f}%"
        ),
    )    


def get_allocation_drift(client: dict, market: dict) -> Result:
    """Current vs. target allocation, per symbol, for this client.

    'Current %' is computed from positions_snapshot[].market_value_usd
    (already valued in the book as of its as_of date) as a share of the
    total covered market value. 'Target %' comes from the client's most
    recent suitability_reviews[].target_allocation_pct entry -- a fact on
    file, not something this function or the agent may originate or
    adjust. Drift = current% - target%, per symbol.

    This function ONLY computes drift against the recorded target -- it
    never says what the target should be. 'What should the target be' is
    advice and belongs to the compliance agent's refusal, not here.

    If a symbol appears in positions or the target but is not in this
    dataset's coverage, that symbol's drift cannot be computed and is
    reported as a gap in `note` rather than silently omitted or
    estimated -- do not average it away.
    """
    positions = client.get("positions_snapshot", [])
    reviews = client.get("suitability_reviews", [])
    if not reviews:
        return Result(value=None, record_ids=[], note="no suitability review / target allocation on record")
    latest_review = max(reviews, key=lambda r: r.get("date", ""))
    target = latest_review.get("target_allocation_pct", {})
    if not target:
        return Result(value=None, record_ids=[latest_review.get("id")],
                       note="suitability review on file has no target_allocation_pct")

    covered = _covered(market)
    total_value = sum(float(p.get("market_value_usd", 0)) for p in positions)
    current_pct: dict[str, float] = {}
    if total_value > 0:
        for p in positions:
            sym = p.get("symbol")
            current_pct[sym] = current_pct.get(sym, 0.0) + float(p.get("market_value_usd", 0)) / total_value * 100

    all_symbols = sorted(set(target.keys()) | set(current_pct.keys()))
    lines = []
    uncovered = []
    record_ids = [latest_review.get("id")] + [p.get("id") for p in positions]
    for sym in all_symbols:
        if sym not in covered:
            uncovered.append(sym)
            lines.append(f"{sym}: not covered by market data -- drift not computable")
            continue
        cur = current_pct.get(sym, 0.0)
        tgt = float(target.get(sym, 0.0))
        drift = cur - tgt
        lines.append(f"{sym}: current {cur:.2f}%, target {tgt:.2f}%, drift {drift:+.2f}pp")

    note = "; ".join(lines)
    if uncovered:
        note += f" -- uncovered symbols excluded from any total: {', '.join(uncovered)}"
    return Result(value=note, record_ids=record_ids, note=None)