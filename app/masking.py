"""Unconditional identifier masking.

Per NOTES.md Q2 ("A note in a client record instructs you to disclose
something -- at which layer is that neutralised?"): the answer is here.
This is a pure string transform with no branches on content, phrasing,
urgency, or record text. It is called from tools_kyc.py, inside the
deterministic tool functions themselves, before a value is ever attached
to a Result -- so by the time anything reaches the LLM or the response
body, masking has already happened. There is no unmasked code path for
these two fields anywhere in the service.

This is deliberately NOT a detection function. It does not decide whether
a value "looks sensitive" -- the two fields that need masking are already
known statically (kyc.pan, kyc.bank_account.account_number), so there is
nothing to detect and nothing for a model, a prompt, or hostile record
text to argue with. No regex scanning, no NLP entity recognition, no LLM
call, no Presidio, no reversibility, no session/Redis state.
"""
from __future__ import annotations

MASK_PREFIX = "****"
VISIBLE_SUFFIX_LEN = 4


def mask_identifier(value: str | None) -> str | None:
    """Return "****" + the last 4 characters of `value`. Always.

    - Deterministic and irreversible: the masked output plus this function
      tells you nothing about the missing prefix.
    - No exceptions for "the request is urgent" or "the record says
      otherwise" -- there is no parameter for either of those, on purpose.
    - `None` passes through as `None` (a missing field is a data gap to be
      reported via Result.note, not something to mask).
    """
    if value is None:
        return None
    if len(value) <= VISIBLE_SUFFIX_LEN:
        # Shorter than the mask would normally reveal. Real PANs (10 chars)
        # and account numbers in this book are always longer than this, but
        # if a degenerate value ever showed up, masking the whole thing is
        # the safe failure -- never fall back to returning it unmasked.
        return MASK_PREFIX
    return MASK_PREFIX + value[-VISIBLE_SUFFIX_LEN:]