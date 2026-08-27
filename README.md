# Agentic Copilots — read-only AI agents on one safe pattern

Two hands-on agentic AI projects that share **one design**. The domain changes; the discipline doesn't.

> **The pattern:** read-only data → **invariant enforced in code** → **eval gate with negative tests** → a human approves. Each agent **proposes**; it never acts.

## The two projects
| | [`finops-copilot/`](./finops-copilot) | [`ops-triage/`](./ops-triage) |
|---|---|---|
| Domain | Cloud cost governance | Incident / observability |
| Reads (read-only) | costs · waste · budget | alerts · metrics · changes |
| Ranks | waste by $ / month | causes by severity × change-correlation |
| Proposes | a savings action for approval | a runbook step for approval |
| Invariant in code | `top_action` = highest-cost item **in the source report** | `top_cause` = highest-severity service |
| Eval checks | contract · total · top · no-hallucination · policy · brief-numbers · derivation | top · no-hallucination · evidence · policy |

## The five invariants (kept in both)
1. **Read-only tools only** — the agent can see everything and change nothing. Safety is structural, not prompted.
2. **No invented data** — the eval checks every figure/id in the plan exists in the source, and every dollar figure quoted in the executive brief too.
3. **Grounded** — every claim cites real evidence.
4. **Propose, never execute** — a policy check forbids "done / released / rolled back / restarted" language.
5. **The fragile invariant lives in code, not the prompt** — the single most important field (the "top" item) is derived in code **from the source data, not from the model's output**. The code owns the figures and the ordering; the model owns the language. That makes the result *correct against the source* — not merely self-consistent with whatever the model happened to submit.

## The differentiator: an eval gate you can prove goes red
Each project has an `evals.py` that scores every run against golden fixtures and only reports **GREEN** if all hard checks pass (a go/no-go gate, CI-ready). A `--selftest` mode validates the checker itself with a good/bad plan and **no API calls** — and, in `finops-copilot/`, with a deliberately corrupted plan the code has to correct. A gate that only ever passes is worthless — these prove they go **RED** on bad input.

## Run either one
```bash
cd finops-copilot           # or: cd ops-triage
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install anthropic
setx ANTHROPIC_API_KEY "sk-..."                       # key in an env var, never in code
python evals.py --selftest   # free: validates the checker (no API)
python <agent>.py            # live: copilot.py  /  triage.py
python evals.py              # live: the GREEN/RED gate against the fixtures
```

## Consumption control

An agent loop is a `while` that makes a billed call each turn and resends the
whole history every time, so **the cost per turn grows with the turn count**.
Until IA-29 these four loops had no cap, and the `usage` field the API returns
on every call was read by nobody.

The irony was worth fixing: this project builds a watchdog for someone else's
AWS bill, and that watchdog was not watching its own.

`agent_budget.py` is shared by both agents and gives three guarantees:

| Guarantee | What it does |
|---|---|
| **Iteration cap** | A loop whose stop condition never arrives ends anyway, naming how many turns it took and which tools it called. |
| **Accounting** | Input and output tokens per call, plus a trace of every tool call and the turn it happened on. |
| **Budget cap** | Checked **before** the request. A cap consulted afterwards is not a cap, it is a receipt. |

```
consumption: 4 iterations, 4 calls, 10000 in + 800 out = 10800 tokens, ~$0.0420 USD | tools: get_costs x4
```

### The honest limit of a pre-call cap

The size of a call cannot be known before making it. What *can* be bounded is
the output, because `max_tokens` is a ceiling we set ourselves, and the input,
because after the first turn the history only grows — so the largest input seen
so far is a lower bound on the next. The projection is conservative from turn
two onward, and **the first call is always allowed**: refusing to start because
the unknown might be too large would make the agent useless.

That limit is why the report says `estimated` next to the money.

### Both caps are proven by tripping

The negative tests replace the model with a fake that always asks for another
tool and never submits — the exact failure the caps exist for, and one that
cannot be reproduced against the real API on demand.

* the iteration cap fires after exactly N turns and N billed calls, no more;
* the budget cap fires **without making the call that would cross it**, and the
  test asserts the tokens actually spent stayed under the ceiling;
* and a third case checks the caps do **not** fire when there is room — a limit
  that always trips is as useless as one that never does.

All of it runs with **no API key and at no cost**, so it belongs in the free
gate rather than in the paid one.

### On the cost figure

`model-pricing.json` carries its source, its date and a `verified` flag, the
same discipline the sibling repository applies to EC2 prices. It currently
declares `verified: false`: the numbers were read off the pricing page rather
than recalled, but nobody has confirmed the transcription a second time. Every
cost printed here is an **estimate** — the invoice is the only authority.

## Honest notes
- Read-only by design; the agents propose, a human approves.
- Evals run on fixtures (known answers), not live data — intentional, to keep the gate deterministic.
- Built on the AWS free tier; each agent consumes a sample snapshot and is architected to consume the real read-only source with no change to the plan/eval code.

**The transferable skill isn't "I made a bot." It's designing the pattern once and knowing exactly where it fits — read data, rank, propose to a human, and test against a reference answer.**
