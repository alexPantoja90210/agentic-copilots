# FinOps Copilot — read-only agentic layer over the FinOps Guardian

An LLM agent (Anthropic Claude, tool-use) that reasons over AWS cost data and produces a **prioritized, structured savings plan** — safely. It can read the whole bill and change nothing.

> Read-only tools → tool-use loop → **`submit_plan`** (structured output) → **deterministic guardrail in code** → an eval gate. The agent **proposes**; a human approves.

## What it does
Given a cost/waste snapshot (`report.json`, produced read-only by the FinOps Guardian via boto3: Cost Explorer, EC2, CloudWatch), the agent:
- reads costs, waste and budget with **read-only tools** (no tool can touch a resource),
- returns a **structured plan**: `total_monthly_waste_usd`, `ranked_actions[]`, an `exec_brief`, and a `top_action`,
- writes `action_plan.json` + `exec_brief.md`. Nothing in AWS is changed.

## Guardrails
1. **Read-only tools only** — `get_costs`, `list_waste`, `get_budget`. No destructive function exists.
2. **No invented numbers** — the agent may only use figures the tools return; the eval verifies every dollar figure exists in the source.
3. **Propose, never execute** — a policy check forbids "released / deleted / done" claims.
4. **Deterministic invariant in code** — `ranked_actions` and `top_action` are *derived in code from the source report*, never from the model's output. The code owns the resources, the figures and the ordering; the model owns the wording. An item the model invents cannot enter the ranking, and cannot be crowned — the same plausibility check `ops-triage/` applies to `top_cause`.

## The eval harness (the differentiator)
`evals.py` scores every run against golden fixtures with three families of hard checks, and is GREEN only if all pass:

- **contract** — `contract-ok`: does the input report still match what the collector emits.
- **plan** — `total-correct`, `top-correct`, `no-hallucination`, `policy-ok`: is the submitted plan sound.
- **derivation** — `derivation-from-source`, `total-from-source`, `no-invented-resource`, `model-wording-kept`: does the code still win when the model hands over a deliberately corrupted plan.

`--selftest` validates the checker with **no API calls**, including the derivation family. It caught three real defects before shipping: a forced `top_action` with no waste; model non-determinism, fixed with the code guardrail above; and a schema drift between the Guardian and this consumer that the gate could not see until a contract check was added.

The derivation family exists because the earlier guardrail sorted *the model's own list*. That guaranteed self-consistency, not correctness: a model that omitted the costliest item, or mistyped a figure, still produced a confident winner. The negative test hands over exactly that kind of plan and asserts the source wins — and asserts that the previous guardrail **fails** on it, because a check that cannot go red proves nothing.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install anthropic
setx ANTHROPIC_API_KEY "sk-..."                       # env var, never in code
python evals.py --selftest     # free: validates the checker
python copilot.py              # live: reads report.json, writes the plan
python evals.py                # live: GREEN/RED gate against the fixtures
```
Model is env-configurable: `set COPILOT_MODEL=claude-...`.

## Relationship to the Guardian
The upstream **AWS FinOps Guardian** reads AWS read-only and writes `report.json`. This Copilot consumes that output (or a sample). It's the sibling of **`ops-triage/`** — same pattern, different domain (see the repo root README).
