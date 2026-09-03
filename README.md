
Challenges & Engineering Solutions

1. Preventing Fabricated Financial Values

LLMs can generate plausible but unsupported numerical values. We addressed this by grounding answers in retrieved financial evidence and adding an independent verification layer.

- Attached provenance to generated answers
- Recomputed numerical results from source data
- Used numeric verification before accepting an answer
- Validated citation support
- Added abstention when sufficient evidence was unavailable

2. Multi-Agent / Specialist Orchestration

Financial questions can require multiple specialists such as market, portfolio, and news data. A major challenge was correctly combining outputs without one specialist overwriting another.

We introduced structured intermediate results and explicit provenance so evidence from multiple specialists could be preserved and combined correctly.

3. Complex Financial Queries

Many questions are not single-retrieval problems. They require identifying multiple information requirements, retrieving the relevant evidence, performing calculations, and then synthesizing the result.

We addressed this through agentic planning and query decomposition rather than relying on a single retrieval step.

4. Financial Date Semantics

Financial datasets contain different date concepts such as trade date, settlement date, month-end close, and reporting period.

Incorrect date interpretation can produce numerically valid but semantically incorrect answers. We therefore treated date selection as part of the retrieval and verification process.

5. Numerical Accuracy

Rounding, percentages, signs, aggregation, and derived metrics introduced another failure mode.

Instead of relying on LLM arithmetic, calculations were independently recomputed from the underlying evidence whenever possible.

6. Evidence and Citation Reliability

Retrieving a relevant document does not necessarily mean that it supports the generated claim.

The verification layer therefore checked whether citations actually supported the claims being made, rather than treating the presence of a citation as sufficient.

7. Agent Abstention

A financial agent should not guess when required data is unavailable.

The system includes an abstention path that allows the agent to reject an answer when evidence is insufficient or verification fails.

8. Stateful Agent Execution

Multi-step agent workflows require state to survive across execution steps and failures. Process-local memory was insufficient for production-style execution.

The system therefore separates workflow state from transient data and is designed around persistent state and recoverable execution.

9. Evaluation Under Constraints

Agent quality cannot be measured only by whether an answer sounds reasonable.

We evaluated the system against predefined questions while considering correctness, evidence, citations, and execution deadlines, allowing individual failures to be traced back to retrieval, orchestration, calculation, or verification.