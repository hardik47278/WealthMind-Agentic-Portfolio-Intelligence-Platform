# Challenges & Engineering Solutions

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
