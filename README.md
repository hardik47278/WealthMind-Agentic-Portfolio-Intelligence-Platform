[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/3X1iDIdp)
# AI Engineering Take-Home: candidate kit

Everything you need to build, run and score your submission locally. Read
`TAKE_HOME_BRIEF.pdf` first; this file is just the quickstart.

All data here is synthetic. Every client, identity number, bank account,
holding and note was fabricated by a generator. Nothing in this kit came from a
real customer or a production system.

## What is in here

```
data/client_book.json              the practice book. Read it end to end.
data/market_data.json              instruments, sectors, prices, news feed
questions/practice_questions.jsonl the practice question stream
schema/answer.schema.json          the response contract, as JSON Schema
schema/agents.schema.json          the roster contract for GET /agents
gateway/llm_gateway.py             the LLM gateway, including its failure modes
harness/run_assessment.py          delivers questions to your service
harness/score.py                   the scorer. Byte-for-byte what we grade with.
harness/judge.py                   the judged dimension, rubric included
harness/practice_key.json          the answer key for the practice questions
harness/practice_leakmap.json      what the scorer scans responses for
docker/Dockerfile.example          a working example of the packaging contract
docker/requirements.example.txt    pin agno; we build the file as it stands
docker/compose.grading.yml         the exact topology grading uses
```

## What you build

A multi-agent ecosystem, in **Agno**, behind an HTTP service with three
endpoints:

```
GET  /health     200 once you are ready
GET  /agents     your agent roster
POST /answer     one question envelope in, one answer object out
```

It reads five environment variables: `BOOK_PATH`, `MARKET_PATH`,
`LLM_BASE_URL`, `LLM_API_KEY`, `PORT`. It must be built and run by a
`Dockerfile` at the root of your repository, on a network whose only route out
is the gateway. Point Agno's model client at `LLM_BASE_URL`; there is no other
way out.

Six agent roles are required, and each agent reports one of them so routing
can be scored without us knowing your naming:

| Role | Owns |
| --- | --- |
| `router` | Classifies and dispatches. In the path on every answer. |
| `book_qa` | Figures from transactions and positions. |
| `kyc_profile` | Identity, KYC, employment, risk. Owns masking. |
| `notes_desk` | Free-text notes and transaction memos. |
| `market_desk` | Instruments, sectors, prices, news. Owns what is covered. |
| `compliance` | Refusals: out-of-scope accounts, personalised advice. |
| `verifier` | Optional. Checks a draft against its citations before it ships. |

Every answer carries the role path that produced it, in `agents`. Some
questions span two specialists and must be answered by both, with the
`client_id` scope intact across the handoff. See `schema/agents.schema.json`
and `schema/answer.schema.json`.

## The second dataset

`data/market_data.json` is the market you are allowed to talk about. Three
parts: `instruments` (symbol, sector, industry, currency, listing), `prices` (a
**monthly** close series per symbol, `{"date", "close"}`, close as a decimal
string) and `news` (dated headlines with a body, one symbol each).

Prices are month-start closes, not daily. For a date between two points, use the
most recent close on or before it and say which date that was.

`meta.covered_symbols` is exactly what the dataset covers, and it is
deliberately incomplete. Some instruments clients hold, and some a prompt names
outright, are not in it.

Those are household names, which is the whole difficulty: a model will answer
about them from memory, fluently, with no source and no date, and the reply
will not look any different from a sourced one. Uncovered means no price, no
sector and no news, and the only right answer is to say so.

Keep drift and advice apart. Every client has an agreed target allocation on
file, so drift against it is arithmetic and we want the number. What the target
*should be* is advice, and that is a refusal. Refusing both scores the same as
answering both.

## The normal way: against the server

Your invitation gives you a URL and, once you enrol, a key. Everything runs
there: `GET /v1/book` and `GET /v1/market` for your two datasets, `GET /v1/next`
for a question, `POST /v1/answer` to submit, `POST /llm/v1/chat/completions` for
the model. Practice is unlimited and tells you, after every answer, what was
expected and how you scored.

**Which model answers depends on the tier.** Your three qualifying attempts and
your final run reach a real reasoning model, supplied and paid for by us, so
the graded comparison is like for like. Practice answers from the stub: it
exercises the protocol, the retries, the deadlines, the token meter and both
chaos bands, and it still gives you full per-question feedback, but it does not
reason. If you want a reasoning model while you iterate, use the offline path
below with your own provider key. That is optional and costs you only what you
choose to spend; nothing about your scored runs depends on it.

Fetch the book and the market once at startup and hold them. They do not change
inside a run, and re-fetching per question wastes your latency budget.

## The two models

There are exactly two model ids, and you ask for them by name:

| Model | Use it for |
| --- | --- |
| `valura-fast` | The default. Routing, lookups, anything mechanical. |
| `valura-deep` | Genuinely hard reasoning. Billed at 4x `valura-fast`. |

Both go to `POST /llm/v1/chat/completions` (or `LLM_BASE_URL` offline), which is
OpenAI-compatible and the only route out. Some questions are scored on getting
the right answer *without* a `valura-deep` call: spending the capable tier on a
trivial lookup is a defect we are looking for, not a safe default.

`harness/reference_client.py` is the whole protocol loop in about eighty lines,
including the retry and resume behaviour. Read it first; it will save you an
hour.

```bash
python harness/reference_client.py \
  --url https://<your-assessment-domain> --key vlr_… --mode practice
```

## The offline way: same book, no server

The book and questions in this kit are byte-identical to what the server calls
practice, and `harness/score.py` is byte-identical to what marks you. So you can
iterate locally without spending anything or touching the network.

Start the bundled gateway in stub mode. It needs no key and returns a canned
string, which is enough to exercise your plumbing, your retries and your failure
handling, including both chaos bands:

```bash
python gateway/llm_gateway.py
```

Point it at any OpenAI-compatible upstream when you want real responses:

```bash
UPSTREAM_MODE=passthrough UPSTREAM_BASE_URL=https://api.openai.com/v1 \
UPSTREAM_API_KEY=sk-... MODEL_MAP_FAST=gpt-4.1-mini MODEL_MAP_DEEP=gpt-4.1 \
python gateway/llm_gateway.py
```

For the offline path your ecosystem exposes `POST /answer` and `GET /agents`,
and the local runner drives it exactly as the server would:

```bash
python harness/run_assessment.py --service http://localhost:8080 \
  --gateway http://localhost:8600 \
  --questions questions/practice_questions.jsonl --out runs/latest

python harness/score.py --key harness/practice_key.json \
  --leakmap harness/practice_leakmap.json \
  --transcript runs/latest/transcript.jsonl \
  --usage runs/latest/gateway_usage.json --roster runs/latest/roster.json
```

The judged dimension is 4 marks of 100 and needs a real upstream. Its rubric is
published in `harness/judge.py`:

```bash
python harness/judge.py --key harness/practice_key.json \
  --transcript runs/latest/transcript.jsonl --gateway http://localhost:8600
```

Only your scored attempts and your final run count, and those happen on the
server. The offline path exists so you are not paying attention to a network
while you are still finding bugs.

## Two numbers, never combined

The scorer prints availability and quality separately, on purpose.

**Availability** is the share of questions that got a schema-valid answer inside
the deadline. It says nothing about whether the answers were right. A service
that replies to everything with a well-formed "I cannot determine that" scores
100% availability and close to zero quality.

**Quality** is the weighted score. That is the one that ranks you.

## The gateway will fail on you

Two bands in the question stream degrade the upstream, and the grading run
includes both. The runner drives them automatically, so a local run rehearses
them exactly.

- **Transient**: the first call for each question is rejected with `429` and a
  `Retry-After` header. Later calls for that question succeed. Retry with
  backoff and you are through.
- **Blackout**: every call fails with a quota-exhausted error for the whole
  band. Nothing gets you through it. The questions still have to be handled.

## Grading uses a different book and different questions

Same generator, same categories, same shapes, same contract. Different clients,
different values, different questions, differently worded. Treat the practice
key as a specification to satisfy, not a target to fit: anything tuned to these
particular answers scores near zero on the day.

## Please do not publish

Not this kit, not the data, not your solution, to any public repository.
