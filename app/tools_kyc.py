"""Deterministic query functions over a single client's KYC block.

Mirrors tools_book.py's design on purpose, for the same reason: every
function takes the whole `client` dict (bound by closure in
agents_kyc_profile.py, so scope to one client is structural, not a rule
the LLM has to remember and could be talked out of) and returns a Result
-- the exact value plus the record id(s) that back it. The agent layer
never sees kyc.pan or kyc.bank_account.account_number in raw form; it only
ever sees what these functions choose to return.

Masking happens HERE, not in the agent layer, for the same reason
tools_book.py does exact arithmetic here rather than trusting the model
with it: "put the mask where no code path can bypass it" (brief). Every
function that touches kyc.pan or kyc.bank_account.account_number returns
the masked form via mask_identifier, unconditionally. There is no
unmasked variant of either field exposed anywhere in this module, so
there is no "wrong tool" an agent could call to leak one -- the raw value
simply never leaves this module.

No file or network I/O happens here. The client dict is the only input,
same scope guarantee as tools_book.py.
"""
from __future__ import annotations

from app.masking import mask_identifier
from app.tools_book import Result


def _kyc(client: dict) -> dict:
    return client.get("kyc", {})


def _kyc_id(client: dict) -> str:
    """The kyc record id, falling back to the client id if the book ever
    omits it -- citations should never come back empty just because of a
    missing sub-id."""
    return _kyc(client).get("id") or client.get("id", "unknown")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def get_pan(client: dict) -> Result:
    """The client's PAN. Always masked (**** + last 4 chars) -- there is
    no code path that returns the raw value, however the question asks
    for it."""
    pan = _kyc(client).get("pan")
    if pan is None:
        return Result(value=None, record_ids=[], note="no PAN on record for this client")
    return Result(value=mask_identifier(pan), record_ids=[_kyc_id(client)])


def get_date_of_birth(client: dict) -> Result:
    """The client's date of birth (YYYY-MM-DD). Not a masked field."""
    dob = _kyc(client).get("date_of_birth")
    if dob is None:
        return Result(value=None, record_ids=[], note="no date of birth on record")
    return Result(value=dob, record_ids=[_kyc_id(client)])


def get_address(client: dict) -> Result:
    """The client's recorded address. Not a masked field.
    USE ONLY WHEN the user asks for the client's address. Do not use this when the user asks for PAN or date of birth"""
    addr = _kyc(client).get("address")
    if addr is None:
        return Result(value=None, record_ids=[], note="no address on record")
    return Result(value=addr, record_ids=[_kyc_id(client)])


# ---------------------------------------------------------------------------
# KYC / risk
# ---------------------------------------------------------------------------
def get_kyc_status(client: dict) -> Result:
    """The client's KYC verification status (e.g. 'verified', 'pending')."""
    status = _kyc(client).get("kyc_status")
    if status is None:
        return Result(value=None, record_ids=[], note="no KYC status on record")
    return Result(value=status, record_ids=[_kyc_id(client)])


def get_risk_profile(client: dict) -> Result:
    """The client's declared risk profile/appetite (e.g. 'aggressive'),
    from the KYC record alone -- does NOT check for disagreement against
    suitability reviews. Kept for reference/testing; kyc_lookup's "risk"
    field routes to check_risk_profile_conflict below instead, which is
    the one that should reach the agent."""
    risk = _kyc(client).get("risk_profile")
    if risk is None:
        return Result(value=None, record_ids=[], note="no risk profile on record")
    return Result(value=risk, record_ids=[_kyc_id(client)])


def check_risk_profile_conflict(client: dict) -> Result:
    """Compares kyc.risk_profile against the most recent suitability
    review's risk_profile. Deterministic string comparison -- if the two
    disagree, this is a genuine conflict in the book (per the brief:
    'When records disagree, say so... surface the disagreement, cite both
    records, and set the conflict flag'), not a judgment call for the LLM
    to make. This is what kyc_lookup's "risk" field maps to -- there is
    only one path to a risk-profile answer, and it always checks for
    disagreement first."""
    kyc_value = _kyc(client).get("risk_profile")
    reviews = client.get("suitability_reviews", [])

    if not reviews:
        # No second source to compare against -- fall back to the plain
        # kyc value, no conflict possible.
        if kyc_value is None:
            return Result(value=None, record_ids=[], note="no risk profile on record")
        return Result(value=kyc_value, record_ids=[_kyc_id(client)])

    latest_review = max(reviews, key=lambda r: r.get("date", ""))
    review_value = latest_review.get("risk_profile")

    if kyc_value is None and review_value is None:
        return Result(value=None, record_ids=[], note="no risk profile on record")

    if kyc_value != review_value:
        return Result(
            value=None,
            record_ids=[_kyc_id(client), latest_review["id"]],
            note=(f"KYC record states risk_profile='{kyc_value}'; "
                  f"suitability review {latest_review['id']} "
                  f"(dated {latest_review.get('date')}) states "
                  f"risk_profile='{review_value}'. These disagree."),
            conflict=True,
        )

    return Result(value=kyc_value, record_ids=[_kyc_id(client), latest_review["id"]])


# ---------------------------------------------------------------------------
# Employment / financial profile
# ---------------------------------------------------------------------------
def get_annual_income_band(client: dict) -> Result:
    """The client's declared annual income band (e.g. '10-25 LPA')."""
    band = _kyc(client).get("annual_income_band")
    if band is None:
        return Result(value=None, record_ids=[], note="no income band on record")
    return Result(value=band, record_ids=[_kyc_id(client)])


def get_employer(client: dict) -> Result:
    """The client's employer/occupation, if the book records one. Not
    every client has this field -- some sample clients lack it entirely,
    so an empty result here is a genuine data gap (abstain), not a bug.

    NOTE: real book data nests this under kyc.employment.employer /
    kyc.employment.occupation, not directly on kyc -- check your actual
    client records match this before trusting it; adjust the .get() chain
    if your book's shape differs."""
    k = _kyc(client)
    employment = k.get("employment", {})
    employer = employment.get("employer") or employment.get("occupation") \
        or k.get("employer") or k.get("occupation")
    if employer is None:
        return Result(value=None, record_ids=[],
                       note="no employer/occupation recorded for this client")
    return Result(value=employer, record_ids=[_kyc_id(client)])


# ---------------------------------------------------------------------------
# Bank account
# ---------------------------------------------------------------------------
def get_bank_name(client: dict) -> Result:
    """The name of the bank holding the client's account. Not a masked
    field -- only the account number itself is."""
    bank = _kyc(client).get("bank_account", {}).get("bank")
    if bank is None:
        return Result(value=None, record_ids=[], note="no bank account on record")
    return Result(value=bank, record_ids=[_kyc_id(client)])


def get_bank_account_number(client: dict) -> Result:
    """The client's bank account number. Always masked (**** + last 4
    digits), same unconditional guarantee as get_pan."""
    acct = _kyc(client).get("bank_account", {}).get("account_number")
    if acct is None:
        return Result(value=None, record_ids=[], note="no bank account on record")
    return Result(value=mask_identifier(acct), record_ids=[_kyc_id(client)])


def get_bank_ifsc(client: dict) -> Result:
    """The IFSC code of the client's bank account. Not a masked field."""
    ifsc = _kyc(client).get("bank_account", {}).get("ifsc")
    if ifsc is None:
        return Result(value=None, record_ids=[], note="no bank account on record")
    return Result(value=ifsc, record_ids=[_kyc_id(client)])


# ---------------------------------------------------------------------------
# Field-group dispatchers -- each defined exactly ONCE. The agent's four
# tools (identity_lookup, kyc_lookup, employment_lookup, bank_lookup) call
# these directly, one per group, per field.
# ---------------------------------------------------------------------------
def identity_lookup(client: dict, field: str) -> Result:
    mapping = {
        "pan": get_pan,
        "dob": get_date_of_birth,
        "address": get_address,
    }
    fn = mapping.get(field)
    if fn is None:
        return Result(value=None, record_ids=[],
                       note=f"identity field '{field}' not supported")
    return fn(client)


def kyc_lookup(client: dict, field: str) -> Result:
    """
    Retrieve ONE KYC/risk field for this client.

    USE THIS WHEN:
    - The user asks whether KYC is verified/pending/etc.
    - The user asks for the client's recorded risk profile.

    SUPPORTED fields:
    - "status"
    - "risk"

    DO NOT USE THIS WHEN:
    - The user asks for PAN, date of birth, or address.
      Use identity_lookup().
    - The user asks for employer or income.
      Use employment_lookup().
    - The user asks for bank details.
      Use bank_lookup().

    IMPORTANT:
    - The "risk" lookup checks the KYC risk profile against the most recent
      suitability review and surfaces a conflict when the records disagree.
    - Never turn a conflict into a guessed single value.
    """
    mapping = {
        "status": get_kyc_status,
        "risk": check_risk_profile_conflict,
    }
    fn = mapping.get(field)
    if fn is None:
        return Result(value=None, record_ids=[],
                       note=f"kyc field '{field}' not supported")
    return fn(client)


def employment_lookup(client: dict, field: str) -> Result:
    """
    Retrieve ONE employment/financial-profile field for this client.

    USE THIS WHEN:
    - The user asks for annual income band.
    - The user asks for employer or occupation.

    SUPPORTED fields:
    - "income"
    - "employer"

    DO NOT USE THIS WHEN:
    - The user asks for identity details.
      Use identity_lookup().
    - The user asks for KYC status or risk.
      Use kyc_lookup().
    - The user asks for bank details.
      Use bank_lookup().

    IMPORTANT:
    - Do not substitute employer for income or income for employer.
    - If the requested field is absent, return the data gap; do not infer it.
    """
    mapping = {
        "income": get_annual_income_band,
        "employer": get_employer,
    }
    fn = mapping.get(field)
    if fn is None:
        return Result(value=None, record_ids=[],
                       note=f"employment field '{field}' not supported")
    return fn(client)


def bank_lookup(client: dict, field: str) -> Result:
    """
    Retrieve ONE bank-account field for this client.

    USE THIS WHEN:
    - The user asks for bank name.
    - The user asks for bank account number.
    - The user asks for IFSC.

    SUPPORTED fields:
    - "bank_name"
    - "account_number"
    - "ifsc"

    DO NOT USE THIS WHEN:
    - The user asks for PAN or other identity information.
      Use identity_lookup().
    - The user asks for KYC status or risk profile.
      Use kyc_lookup().
    - The user asks for employer or income.
      Use employment_lookup().

    IMPORTANT:
    - The account number is ALWAYS returned masked.
    - Never attempt to reconstruct, infer, or reveal masked digits.
    - Do not substitute bank name or IFSC for the account number.
    """
    mapping = {
        "bank_name": get_bank_name,
        "account_number": get_bank_account_number,
        "ifsc": get_bank_ifsc,
    }
    fn = mapping.get(field)
    if fn is None:
        return Result(value=None, record_ids=[],
                       note=f"bank field '{field}' not supported")
    return fn(client)