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
| Invariant in code | `top_action` = highest-cost item | `top_cause` = highest-severity service |
| Eval checks | total · top · no-hallucination · policy | top · no-hallucination · evidence · policy |

## The five invariants (kept in both)
1. **Read-only tools only** — the agent can see everything and change nothing. Safety is structural, not prompted.
2. **No invented data** — the eval checks every figure/id in the plan exists in the source.
3. **Grounded** — every claim cites real evidence.
4. **Propose, never execute** — a policy check forbids "done / released / rolled back / restarted" language.
5. **The fragile invariant lives in code, not the prompt** — the single most important field (the "top" item) is derived in code, so the output is self-consistent even when the model isn't.

## The differentiator: an eval gate you can prove goes red
Each project has an `evals.py` that scores every run against golden fixtures and only reports **GREEN** if all hard checks pass (a go/no-go gate, CI-ready). A `--selftest` mode validates the checker itself with a good/bad plan and **no API calls**. A gate that only ever passes is worthless — these prove they go **RED** on bad input.

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

## Honest notes
- Read-only by design; the agents propose, a human approves.
- Evals run on fixtures (known answers), not live data — intentional, to keep the gate deterministic.
- Built on the AWS free tier; each agent consumes a sample snapshot and is architected to consume the real read-only source with no change to the plan/eval code.

**The transferable skill isn't "I made a bot." It's designing the pattern once and knowing exactly where it fits — read data, rank, propose to a human, and test against a reference answer.**
