# Ops Triage Copilot — read-only agentic incident triage

A second agentic AI project that reuses the **same safe pattern** as my FinOps Copilot, ported to a new domain — proving the pattern is portable, not a one-off.

> Read-only signals → tool-use LLM → **code-forced structured plan** → **deterministic invariant in code** → **eval gate with negative tests** → a human approves. The agent **proposes**; it never acts.

## What it does
Given a read-only observability snapshot (`signals.json`: alerts, metrics vs. baseline, recent changes, runbook), the agent:
- reads the signals with read-only tools (no command execution, no writes),
- returns a **structured triage plan**: ranked probable causes (each grounded in real evidence ids), and a **proposed** next runbook step,
- writes `triage_plan.json` + `incident_brief.md`. Nothing in any system is changed.

## The guardrails (why it's safe)
1. **Read-only tools only** — `get_alerts`, `get_context`, `get_runbook`. No tool can restart, roll back, or scale anything.
2. **No invented signals** — the eval checks every service and evidence id in the plan exists in the snapshot.
3. **Evidence-grounded** — every ranked cause must cite ≥1 real evidence id (alert `AL-*`, change `CH-*`, or metric `M-*`); evidence is IDs only, never free text.
4. **Propose, never execute** — a policy check forbids "restarted / rolled back / scaled / resolved" language.
5. **Deterministic invariant in code** — the `top_cause` is *derived in code* from a severity score (severity × change-correlation), never trusted from the model. Self-consistent by construction.

## The eval harness (the differentiator)
`evals.py` scores every run against golden fixtures on 4 hard checks: **top-correct, no-hallucination, evidence-grounded, policy-ok**. A run is GREEN only if all pass — a go/no-go gate. A `--selftest` mode validates the checker itself with a good/bad plan and **no API calls** (free, deterministic).

Fixtures:
- `case1_deploy` — error spike right after a deploy → top = the deployed service (change-correlated), DB CPU is a red herring.
- `case2_saturation` — critical CPU with no deploy → top = the saturated service (change-correlation must not over-fire).
- `case3_allclear` — no alerts → no incident, empty causes (tests the "nothing's wrong" path).

## Run it
```
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install anthropic
setx ANTHROPIC_API_KEY "sk-..."                       # your key in an env var, never in code

python evals.py --selftest     # free: validates the checker (no API)
python triage.py               # live: reads signals.json, prints + writes the triage plan
python evals.py                # live: scores the agent against the fixtures (GREEN/RED gate)
```
Model is env-configurable: `set TRIAGE_MODEL=claude-...`.

## Honest limits
- Read-only by design; it proposes, a human approves. That's the feature.
- Evals run on fixtures (known answers), not live telemetry — intentional, to keep the gate deterministic.
- `signals.json` is a sample snapshot; the architecture is ready to consume a real read-only export (e.g., CloudWatch alarms + Logs Insights + deploy history) with no change to `triage()` or `evals.py`.

## Same pattern as FinOps — that's the point
| Slot | FinOps Copilot | Ops Triage Copilot |
|---|---|---|
| Data → snapshot | AWS → `report.json` | Observability → `signals.json` |
| Read-only tools | costs / waste / budget | alerts / context / runbook |
| Structured plan | ranked actions + brief | ranked causes + proposed step |
| Invariant in code | top_action = highest cost | top_cause = highest severity |
| Eval checks | total/top/halluc/policy | top/halluc/evidence/policy |

Two domains, one discipline. The transferable skill is moving the fragile invariant out of the LLM and proving the gate can fail.
