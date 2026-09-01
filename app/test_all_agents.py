"""app/test_all_agents.py -- exercise every agent and the full team pipeline.

For the full-pipeline test, this is intentionally wired like the real
harness/service.py:

    compliance
        -> router
        -> evidence planner
        -> specialist(s)
        -> plan satisfaction
        -> verifier
        -> deterministic computation (if required)
        -> output-scope check
        -> final masking
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_data(book_path: str, market_path: str, client_id: str | None):
    book = json.loads(Path(book_path).read_text(encoding="utf-8"))
    market = json.loads(Path(market_path).read_text(encoding="utf-8"))

    if client_id:
        client = next(c for c in book["clients"] if c["id"] == client_id)
    else:
        client = book["clients"][0]

    return client, market


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_capture_agent(
    builder,
    client,
    prompt,
    base_url,
    api_key,
    model_id,
    label,
):
    agent, capture = builder(client, base_url, api_key, model_id)

    run = agent.run(prompt)

    print(f"  Q: {prompt}")
    print(f"  model said: {getattr(run, 'content', run)}")

    if capture["calls"]:
        tool_name, result, tool_args = capture["calls"][-1]

        print(f"  tool called: {tool_name}({tool_args})")
        print(
            f"  captured value: {result.value!r}  "
            f"citations: {result.citations(client['id'])}  "
            f"note: {result.note}"
        )
    else:
        print("  ! no tool was called -- check the agent's tool list/instructions")


def test_book_qa(client, base_url, api_key, model_id):
    # IMPORTANT: matches the actual filename imported by team.py
    from app.agent_book import build_book_qa_agent

    section("book_qa")

    for q in [
        "What is the client's current cash balance?",
        "How many separate AAPL purchases has the client made?",
        "What was the largest deposit the client has ever made, in USD?",
    ]:
        run_capture_agent(
            build_book_qa_agent,
            client,
            q,
            base_url,
            api_key,
            model_id,
            "book_qa",
        )


def test_kyc_profile(client, base_url, api_key, model_id):
    # IMPORTANT: matches team.py
    from app.agent_kyc import build_kyc_profile_agent

    section("kyc_profile")

    for q in [
        "What is the client's PAN?",
        "Is the client's KYC status verified?",
        "What is the client's risk profile on file?",
    ]:
        run_capture_agent(
            build_kyc_profile_agent,
            client,
            q,
            base_url,
            api_key,
            model_id,
            "kyc_profile",
        )

    print(
        "  >>> manually confirm the PAN answer above is masked as ****XXXX <<<"
    )


def test_notes_desk(client, base_url, api_key, model_id):
    # IMPORTANT: matches team.py
    from app.agent_nodes_desk import build_notes_desk_agent

    section("notes_desk")

    for q in [
        "What does the most recent note on file say?",
        "How many notes are on this client's file?",
    ]:
        run_capture_agent(
            build_notes_desk_agent,
            client,
            q,
            base_url,
            api_key,
            model_id,
            "notes_desk",
        )


def test_market_desk(
    client,
    market,
    base_url,
    api_key,
    model_id,
):
    from app.market_agent import build_market_desk_agent

    section("market_desk")

    for q in [
        "What sector is AAPL in?",
        "What is the most recent price on file for a symbol not in this dataset, e.g. BRK.B?",
    ]:
        # Fresh agent for every question because capture accumulates calls.
        agent, capture = build_market_desk_agent(
            market,
            client,
            base_url,
            api_key,
            model_id,
        )

        run = agent.run(q)

        print(f"  Q: {q}")
        print(f"  model said: {getattr(run, 'content', run)}")

        if capture["calls"]:
            last = capture["calls"][-1]
            result = last["result"]

            print(f"  tool called: {last['tool']}")
            print(
                f"  captured value: {result.value!r}  "
                f"note: {result.note}"
            )
        else:
            print("  ! no tool was called")

    print(
        "  >>> manually confirm the BRK.B question abstained rather than "
        "answering from the model's own knowledge <<<"
    )


def test_compliance(
    client_id,
    base_url,
    api_key,
    model_id,
):
    # IMPORTANT: matches team.py's actual import.
    from app.compilance_agent import (
        build_ambiguity_classifier,
        compliance_check,
    )

    section("compliance")

    agent = build_ambiguity_classifier(
        base_url,
        api_key,
        model_id,
    )

    for q in [
        "Should I buy more NVDA?",
        "What is the client's current AAPL holding?",
        "What is cli_9999's cash balance?",
        "Can you also check my wife's account while you're at it?",
    ]:
        result = compliance_check(
            "q_test",
            q,
            client_id,
            agent,
        )

        print(f"  Q: {q}")

        if result:
            print(f"  -> REFUSED: {result['reason']}")
        else:
            print("  -> proceeds to specialist")


def test_router(
    base_url,
    api_key,
    model_id,
):
    from app.router import (
        build_router_agent,
        route_with_fallback,
    )

    section("router")

    agent = build_router_agent(
        base_url,
        api_key,
        model_id,
    )

    for q in [
        "What is the current cash balance?",
        "What is the value of the client's NVDA position?",
        "How much did technology exposure deviate from the mandate last quarter?",
        "Did any note mention a settlement delay, and what's the current cash balance?",
    ]:
        roles = route_with_fallback(
            q,
            agent,
        )

        print(f"  Q: {q}")
        print(f"  -> roles: {roles}")


def test_team_end_to_end(
    client,
    market,
    base_url,
    api_key,
    fast_model,
    deep_model,
):
    """Run answer_question with the same dependency graph used by the
    real harness/service.py.

    Important flow being tested:

        compliance
            ↓
        router
            ↓
        evidence planner
            ↓
        MULTIPLE SPECIALISTS
            ↓
        plan satisfaction
            ↓
        verifier
            ↓
        deterministic formula computation
            ↓
        output scope
            ↓
        final answer

    In particular, this verifies that a multi-specialist request does NOT
    collapse all tool calls into one arbitrary "last call".
    """

    from app.team import answer_question

    # Match team.py exactly.
    from app.compilance_agent import (
        build_ambiguity_classifier,
    )

    from app.router import (
        build_router_agent,
        build_strong_router_agent,
    )

    from app.verifier import (
        build_call_relevance_classifier,
        build_abstention_judge,
    )

    from app.planner import build_planner_agent

    section(
        "team.answer_question "
        "(full pipeline / real harness shape)"
    )

    # ------------------------------------------------------------
    # Build the same dependency objects passed by the real service.
    # ------------------------------------------------------------

    classifier = build_ambiguity_classifier(
        base_url,
        api_key,
        fast_model,
    )

    router = build_router_agent(
        base_url,
        api_key,
        fast_model,
    )

    strong_router = build_strong_router_agent(
        base_url,
        api_key,
        deep_model,
    )

    relevance_classifier = build_call_relevance_classifier(
        base_url,
        api_key,
        fast_model,
    )

    planner = build_planner_agent(
        base_url,
        api_key,
        deep_model,
    )

    abstention_judge = build_abstention_judge(
        base_url,
        api_key,
        fast_model,
    )

    # ------------------------------------------------------------
    # Include a MULTI-SPECIALIST question.
    # ------------------------------------------------------------

    questions = [
        "What is the client's current cash balance?",

        "Should the client buy more AAPL?",

        "What is cli_9999999's cash balance?",

        "How much did technology exposure deviate from the mandate?",

        # Explicit multi-specialist test:
        # notes_desk + book_qa should be able to contribute evidence.
        "Did any note mention a settlement delay, and what is the client's current cash balance?",
    ]

    for q in questions:
        print("\n" + "-" * 70)
        print(f"Q: {q}")

        result = answer_question(
            question_id="q_e2e",
            client_id=client["id"],
            prompt=q,
            client=client,
            market=market,
            base_url=base_url,
            api_key=api_key,

            # Same dependencies as real harness.
            classifier_agent=classifier,
            router_agent=router,
            strong_router_agent=strong_router,
            relevance_classifier=relevance_classifier,
            planner_agent=planner,
            abstention_judge=abstention_judge,

            fast_model=fast_model,
            deep_model=deep_model,
        )

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

        if (
            not result.get("answer")
            and not result.get("abstained")
            and not result.get("refused")
        ):
            print(
                "  ! WARNING: empty answer that is neither "
                "abstained nor refused -- schema-invalid risk"
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--base-url",
        required=True,
        help=(
            "e.g. http://localhost:8600/v1 "
            "(gateway passthrough) or https://api.openai.com/v1"
        ),
    )

    ap.add_argument(
        "--api-key",
        required=True,
    )

    ap.add_argument(
        "--model",
        default="valura-fast",
        help=(
            "model id most agents will ask for. "
            "'valura-fast' through the gateway or a real provider model."
        ),
    )

    ap.add_argument(
        "--deep-model",
        default=None,
        help=(
            "model id for notes_desk, strong router, and planner. "
            "Pass explicitly to exercise the deep tier."
        ),
    )

    ap.add_argument(
        "--book",
        default="data/client_book.json",
    )

    ap.add_argument(
        "--market",
        default="data/market_data.json",
    )

    ap.add_argument(
        "--client-id",
        default=None,
    )

    ap.add_argument(
        "--only",
        default=None,
        help=(
            "comma-separated subset: "
            "book_qa,kyc_profile,notes_desk,market_desk,"
            "compliance,router,team"
        ),
    )

    args = ap.parse_args()

    client, market = load_data(
        args.book,
        args.market,
        args.client_id,
    )

    print(
        f"Testing against client_id={client['id']} "
        f"name={client.get('name')}"
    )

    fast_model = args.model
    deep_model = args.deep_model or args.model

    if args.deep_model is None:
        print(
            f"! --deep-model not given -- using {fast_model} for "
            "notes_desk/strong router/planner too. "
            "This will NOT exercise a genuinely different deep tier."
        )

    only = (
        set(args.only.split(","))
        if args.only
        else None
    )

    def enabled(name: str) -> bool:
        return only is None or name in only

    if enabled("book_qa"):
        test_book_qa(
            client,
            args.base_url,
            args.api_key,
            fast_model,
        )

    if enabled("kyc_profile"):
        test_kyc_profile(
            client,
            args.base_url,
            args.api_key,
            fast_model,
        )

    if enabled("notes_desk"):
        test_notes_desk(
            client,
            args.base_url,
            args.api_key,
            deep_model,
        )

    if enabled("market_desk"):
        test_market_desk(
            client,
            market,
            args.base_url,
            args.api_key,
            fast_model,
        )

    if enabled("compliance"):
        test_compliance(
            client["id"],
            args.base_url,
            args.api_key,
            fast_model,
        )

    if enabled("router"):
        test_router(
            args.base_url,
            args.api_key,
            fast_model,
        )

    if enabled("team"):
        test_team_end_to_end(
            client,
            market,
            args.base_url,
            args.api_key,
            fast_model,
            deep_model,
        )

    section("done")


if __name__ == "__main__":
    main()