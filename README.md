in# Challenges & Engineering Solutions

## 1. Preventing Fabricated Financial Values

LLMs can generate plausible but unsupported numerical values. We addressed this by grounding answers in retrieved financial evidence and adding an independent verification layer.

- Attached provenance to generated answers
- Recomputed numerical results from source data
- Used numeric verification before accepting an answer
- Validated citation support
- Added abstention when sufficient evidence was unavailable

## 2. Multi-Agent / Specialist Orchestration

Financial questions can require multiple specialists such as market, portfolio, and news data. A major challenge was correctly combining outputs without one specialist overwriting another.

We introduced structured intermediate results and explicit provenance so evidence from multiple specialists could be preserved and combined correctly.

## 3. Complex Financial Queries

Many questions are not single-retrieval problems. They require identifying multiple information requirements, retrieving the relevant evidence, performing calculations, and then synthesizing the result.

We addressed this through agentic planning and query decomposition rather than relying on a single retrieval step.

## 4. Financial Date Semantics

Financial datasets contain different date concepts such as trade date, settlement date, month-end close, and reporting period.

Incorrect date interpretation can produce numerically valid but semantically incorrect answers. We therefore treated date selection as part of the retrieval and verification process.

## 5. Numerical Accuracy

Rounding, percentages, signs, aggregation, and derived metrics introduced another failure mode.

Instead of relying on LLM arithmetic, calculations were independently recomputed from the underlying evidence whenever possible.

## 6. Evidence and Citation Reliability

Retrieving a relevant document does not necessarily mean that it supports the generated claim.

The verification layer therefore checked whether citations actually supported the claims being made, rather than treating the presence of a citation as sufficient.

## 7. Agent Abstention

A financial agent should not guess when required data is unavailable.

The system includes an abstention path that allows the agent to reject an answer when evidence is insufficient or verification fails.

## 8. Stateful Agent Execution

Multi-step agent workflows require state to survive across execution steps and failures. Process-local memory was insufficient for production-style execution.

The system therefore separates workflow state from transient data and is designed around persistent state and recoverable execution.

## 9. Evaluation Under Constraints

Agent quality cannot be measured only by whether an answer sounds reasonable.

We evaluated the system against predefined questions while considering correctness, evidence, citations, and execution deadlines, allowing individual failures to be traced back to retrieval, orchestration, calculation, or verification.

## Challenge: Scope-Aware Routing

A key challenge was preventing the agent from attempting to answer questions outside the capabilities of the available financial tools.

A naive router may try to route every query to the closest specialist, causing the system to produce unsupported or fabricated answers when the required data is unavailable.

We introduced a scope-aware routing decision:

```
User Query
    ↓
Router
    ↓
Is this within the system's supported scope?
    ├── No → NOT_MY_SCOPE / Abstain
    │
    └── Yes
          ↓
     Query Decomposition
          ↓
     Specialist Routing
          ↓
     Evidence Retrieval
          ↓
     Verification
          ↓
     Final Answer
```

![Router agent — recoverable routing diagram showing the router classifying a request, splitting by confidence, dispatching to a specialist, and recovering via a not_my_scope reroute back to the router](router_agent_dark.png)

For example, if the system only has access to financial market, portfolio, and news data, a question requiring unrelated information should not be forced into one of those specialists.

Instead, the router returns a controlled `NOT_MY_SCOPE` outcome.

This prevents a common agentic failure mode:

> «Forcing every query through an available tool and allowing the LLM to fill missing information with a guess.»

The scope check therefore acts as an early guardrail before decomposition, tool execution, and answer generation.



What Each Verification Check Does

1. Numerical / Mathematical Verification

Validates all numerical claims and derived financial calculations.

- Recomputes values from source evidence.
- Checks totals, percentages, ratios, averages, and other derived metrics.
- Handles rounding and numerical tolerance.
- Detects incorrect signs or arithmetic.
- Prevents the LLM from inventing or modifying financial values.

2. Semantic Verification

Checks whether the generated answer actually answers the user's question.

It validates:

- Correct company / instrument / entity.
- Correct financial metric.
- Correct date or reporting period.
- Correct interpretation of the question.
- Correct relationship between the evidence and the generated claim.

For example, if the question asks for 2025 revenue, evidence for 2024 revenue should not be accepted simply because the metric name matches.

3. Citation Verification

Checks whether citations attached to the answer are valid and actually support the associated claims.

It verifies:

- Citation exists when required.
- Citation points to the relevant evidence.
- Claim is supported by the cited source.
- Required citation-count rules are satisfied.

4. Provenance Verification

Tracks where every important claim came from.

Source Data
    ↓
Retrieved Evidence
    ↓
Claim
    ↓
DraftAnswer
    ↓
Final Answer

This makes it possible to trace a financial value back to the underlying source rather than trusting the LLM's generated number.

5. Consistency Verification

Checks for contradictions between different pieces of evidence or generated claims.

Examples:

- Two specialists return different values.
- The answer uses a different date from the retrieved evidence.
- A percentage does not match the underlying amount.
- A final conclusion contradicts an earlier claim.

This is particularly important for multi-step financial questions.

6. Abstention Judge

The system should not always produce an answer.

When evidence is incomplete, conflicting, unavailable, or insufficient to verify the claim:

Evidence Insufficient
        ↓
   Abstention Judge
        ↓
      ABSTAIN

The objective is to prefer abstention over fabricated information.

---

Numerical Verification Code Logic

The important principle is that the LLM's calculation is treated as untrusted.

def verify_numeric(draft_answer):

    agent_value = draft_answer.value

    verified_value = recompute(
        evidence=draft_answer.evidence,
        operation=draft_answer.operation
    )

    if approximately_equal(agent_value, verified_value):
        return PASS

    return FAIL

For example:

LLM Generated Value
        ↓
      27.40%
        │
        │ compare
        ↓
Independent Recalculation
        ↓
      27.38%
        │
        ↓
Tolerance Check
        │
   ┌────┴────┐
   ↓         ↓
 PASS       FAIL

The tolerance is important because financial calculations may involve legitimate rounding differences. However, a material mismatch should cause the claim to fail verification.

---

Verification Outcome

The verifier ultimately produces one of three outcomes:

                VERIFIER
                    │
        ┌───────────┼───────────┐
        ↓           ↓           ↓
     VERIFIED    REJECTED    ABSTAIN
        │           │           │
 Evidence +     Verification   Evidence
 checks pass     failure        insufficient
        │           │           │
        ↓           ↓           ↓
  Accept Answer  Reject Answer  Don't Guess

VERIFIED → Evidence and verification checks support the answer.

REJECTED → One or more verification checks fail.

ABSTAIN → The system cannot establish sufficient evidence to safely answer.

Key Design Principle

«Generation is not acceptance.»

The LLM is responsible for generating a candidate answer, while the verification layer independently determines whether that answer can be trusted.

This separation is especially important in financial applications, where a fluent but incorrect numerical answer can be more dangerous than an explicit abstention.