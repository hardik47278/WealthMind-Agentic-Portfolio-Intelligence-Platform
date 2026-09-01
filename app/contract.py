"""app/contract.py -- alias for app/schemas.py.

AnswerContract and Answer are the SAME schema (matches
schema/answer.schema.json exactly). Some modules in this codebase
(compliance.py, market_desk.py, verifier.py, team.py) import
`AnswerContract` from here; app/service.py and app/schemas.py itself use
`Answer`. Rather than maintain two separate Pydantic models that could
drift out of sync, this file just re-exports the one real definition.

If you'd rather have a single canonical name everywhere, it's safe to
delete this file and do a project-wide rename of AnswerContract -> Answer
(or the reverse) instead -- this alias exists only to unblock the modules
that already reference the other name.
"""
from app.schemas import Answer as AnswerContract  # noqa: F401
from app.schemas import Question, Roster, AgentDecl, abstain, refuse  # noqa: F401