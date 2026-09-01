"""Pydantic models mirroring schema/answer.schema.json and
schema/agents.schema.json exactly. Keep these in lockstep with the JSON
schema files in your kit -- if the kit files change, update here too.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Role = Literal[
    "router",
    "book_qa",
    "kyc_profile",
    "notes_desk",
    "market_desk",
    "compliance",
    "evidence_planner",
    "verifier",
]
Flag = Literal["conflict", "upstream_issue", "stale_data"]
ModelId = Literal["valura-fast", "valura-deep"]


# ---------------------------------------------------------------------------
# Incoming question envelope (POST /answer body, from the harness/server)
# ---------------------------------------------------------------------------
class Question(BaseModel):
    question_id: str
    client_id: str
    prompt: str
    deadline_seconds: Optional[int] = None
    chaos: Optional[str] = None  # present in practice_questions.jsonl


# ---------------------------------------------------------------------------
# Outgoing answer (POST /answer response, and what you POST to /v1/answer)
# ---------------------------------------------------------------------------
class Answer(BaseModel):
    question_id: str
    answer: str = ""
    answer_value: Optional[str] = None
    abstained: bool = False
    refused: bool = False
    reason: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    flags: list[Flag] = Field(default_factory=list)
    agents: list[Role]

    @model_validator(mode="after")
    def _check_contract(self) -> "Answer":
        # router must appear somewhere in the path
        if "router" not in self.agents:
            raise ValueError("agents must include 'router'")

        # reason required + non-empty whenever abstained or refused
        if (self.abstained or self.refused) and not (self.reason and self.reason.strip()):
            raise ValueError("reason is required and non-empty when abstained/refused")

        # answer_value must be null when abstaining or refusing
        if (self.abstained or self.refused) and self.answer_value is not None:
            raise ValueError("answer_value must be null when abstained or refused")

        # can't be both -- they are separate limits (epistemic vs policy)
        if self.abstained and self.refused:
            raise ValueError("abstained and refused are mutually exclusive")

        return self


# ---------------------------------------------------------------------------
# Roster (GET /agents response, and what you POST to /v1/roster)
# ---------------------------------------------------------------------------
class AgentDecl(BaseModel):
    role: Role
    name: Optional[str] = None
    model: Optional[ModelId] = None
    tools: list[str] = Field(default_factory=list)


class Roster(BaseModel):
    framework: Literal["agno"] = "agno"
    framework_version: Optional[str] = None
    agents: list[AgentDecl] = Field(min_length=6)

    @model_validator(mode="after")
    def _check_required_roles(self) -> "Roster":
        required = {"router", "book_qa", "kyc_profile",
                    "notes_desk", "market_desk", "compliance"}
        present = {a.role for a in self.agents}
        missing = required - present
        if missing:
            raise ValueError(f"roster missing required roles: {sorted(missing)}")
        return self


# ---------------------------------------------------------------------------
# Convenience: build a valid abstain answer. Used as the safety-net fallback
# whenever anything downstream fails -- a well-formed abstain beats a crash
# or a fabricated number every time.
# ---------------------------------------------------------------------------
def abstain(question_id: str, reason: str,
            agents: list[Role] | None = None,
            flags: list[Flag] | None = None) -> Answer:
    return Answer(
        question_id=question_id,
        answer="",
        answer_value=None,
        abstained=True,
        refused=False,
        reason=reason,
        citations=[],
        confidence=0.0,
        flags=flags or [],
        agents=agents or ["router"],
    )


def refuse(question_id: str, reason: str,
           agents: list[Role] | None = None) -> Answer:
    return Answer(
        question_id=question_id,
        answer="",
        answer_value=None,
        abstained=False,
        refused=True,
        reason=reason,
        citations=[],
        confidence=1.0,
        flags=[],
        agents=agents or ["router", "compliance"],
    )