"""
Ops Triage Copilot - READ-ONLY agentic layer over an observability snapshot.
Same safe pattern as the FinOps Copilot, ported to a new domain:
  read-only tools -> tool-use loop -> STRUCTURED triage plan -> deterministic
  guardrail in code -> a human approves. The agent PROPOSES; it never acts.

Phase 1: ask()   - free-form Q&A over the signals (read-only).
Phase 2: triage() - agent reads signals and submits a structured triage plan
                    (incident_brief.md + triage_plan.json). Changes nothing.

Requires: pip install anthropic  and  ANTHROPIC_API_KEY in your environment.
"""
import json
import os
import sys
from anthropic import Anthropic

from signals_contract import (
    SignalsContractError,
    describe_problems,
    validate_signals,
)

# Pick a current model from https://docs.claude.com/en/docs/about-claude/models
MODEL = os.environ.get("TRIAGE_MODEL", "claude-sonnet-4-5")

# The per-call output ceiling, passed to the API and to the budget alike, so the
# cap projects against the number the request is actually bounded by.
MAX_OUTPUT_TOKENS = 1024
# The client is built on first use, never on import. Instantiating it at module
# scope makes `import triage` require ANTHROPIC_API_KEY, which would drag a
# credential into anything that merely imports this module — the free selftest
# and the CI gate included. IA-27 fixed the same defect in copilot.py; here it
# had not bitten yet only because evals.py imports triage inside run_live().
# Working by accident is not the same as working by design.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_budget import (  # noqa: E402  (import after the path shim, by necessity)
    BudgetExceeded,
    IterationCapExceeded,
    RunBudget,
)

# The budget of the most recent run, readable by a caller that did not pass one.
LAST_BUDGET = None

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _CLIENT

SNAP = {}
SEV = {"critical": 3, "major": 2, "minor": 1}

def load_snapshot(path="signals.json"):
    """
    Load the signals snapshot, refusing anything that does not meet the contract.

    It fails loudly and early on purpose. The alternative — loading whatever is
    in the file and letting the agent narrate around the gaps — is exactly what
    IA-26 found in the sibling agent: the model was fluent enough to hide a
    broken schema, and nothing looked wrong until someone checked by hand.
    """
    global SNAP
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    problems = validate_signals(data)
    if problems:
        raise SignalsContractError(describe_problems(path, problems))
    SNAP = data
    return SNAP

# ---------- read-only tools (none of these change anything) ----------
def get_alerts(_a):
    return SNAP.get("alerts", [])

def get_context(_a):
    return {
        "window": SNAP.get("window"),
        "services": SNAP.get("services", []),
        "metrics": SNAP.get("metrics", []),
        "recent_changes": SNAP.get("recent_changes", []),
    }

def get_runbook(_a):
    return SNAP.get("runbook", [])

TOOL_FUNCS = {"get_alerts": get_alerts, "get_context": get_context, "get_runbook": get_runbook}

READ_TOOLS = [
    {"name": "get_alerts", "description": "List firing alerts, each with service, signal, severity, value, threshold, first_seen.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_context", "description": "Get the window, service list, current metrics vs. baseline (each with an id like M-1, citable as evidence), and recent changes (deploys/config).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_runbook", "description": "Get the runbook: for each cause_type, the recommended (propose-only) next step.",
     "input_schema": {"type": "object", "properties": {}}},
]

# Phase 2: the tool the model MUST call to submit its structured triage (structured output).
SUBMIT_TRIAGE_TOOL = {
    "name": "submit_triage",
    "description": "Submit the final incident triage. Call this once, after reading the signals.",
    "input_schema": {
        "type": "object",
        "properties": {
            "incident_summary": {"type": "string"},
            "ranked_causes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "evidence": {"type": "array", "items": {"type": "string"},
                                      "description": "A list of evidence IDs only (alert AL-*, change CH-*, or metric M-*). No free text."},
                    },
                    "required": ["service", "hypothesis", "confidence", "evidence"],
                },
            },
            "proposed_next_step": {"type": "string"},
        },
        # top_cause is derived in code (see guardrail), so it is NOT required here.
        "required": ["incident_summary", "ranked_causes", "proposed_next_step"],
    },
}

SYSTEM = (
    "You are Ops Triage Copilot, a READ-ONLY incident assistant. "
    "You can call tools to read alerts, metrics, recent changes and the runbook, "
    "but you CANNOT change any system or run any command. "
    "Rules: (1) Use ONLY services, ids and figures returned by the tools; never invent them. "
    "(2) Every cause you rank must cite at least one real evidence id. Evidence entries MUST be IDs only - "
    "an alert id (AL-*), a change id (CH-*), or a metric id (M-*). Never put descriptive text in the evidence "
    "list; put your reasoning in the hypothesis field instead. "
    "(3) Weigh a firing alert that correlates with a recent change to the same service as the strongest signal. "
    "(4) Never claim an action was performed (no 'restarted', 'rolled back', 'scaled', 'resolved'); always PROPOSE a next step for a human. "
    "(5) If there are no alerts, say so: return an empty ranked_causes and a proposed_next_step of 'No incident - continue monitoring.'"
)

def _run(messages, tools, budget):
    """
    The single choke point every billed call passes through.

    The cap is consulted BEFORE the request. Checking afterwards would only
    produce a receipt, which is not a cap.
    """
    budget.before_call()
    resp = _get_client().messages.create(model=MODEL, max_tokens=MAX_OUTPUT_TOKENS,
                                         system=SYSTEM, tools=tools, messages=messages)
    budget.record_response(resp)
    return resp

def ask(question, verbose=True, budget=None):
    """Phase 1: free-form Q&A over the signals (read-only)."""
    global LAST_BUDGET
    budget = budget or RunBudget(MODEL, max_output_per_call=MAX_OUTPUT_TOKENS)
    LAST_BUDGET = budget
    messages = [{"role": "user", "content": question}]
    while True:
        budget.begin_iteration()
        resp = _run(messages, READ_TOOLS, budget)
        if resp.stop_reason != "tool_use":
            if verbose:
                print("  [" + budget.summary() + "]")
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                budget.record_tool(block.name)
                if verbose:
                    print("  [tool call] " + block.name)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(TOOL_FUNCS[block.name](block.input))})
        messages.append({"role": "user", "content": results})

def _severity_score(service):
    """Deterministic: max alert severity for a service (+1 if a recent change touched it)."""
    alerts = [a for a in SNAP.get("alerts", []) if a.get("service") == service]
    if not alerts:
        return 0
    sev = max(SEV.get(a.get("severity"), 0) for a in alerts)
    changed = {c.get("service") for c in SNAP.get("recent_changes", [])}
    return sev * 10 + (1 if service in changed else 0)

def triage(write_files=True, budget=None, verbose=True):
    """Phase 2: agent reads the signals and submits a structured triage plan."""
    global LAST_BUDGET
    budget = budget or RunBudget(MODEL, max_output_per_call=MAX_OUTPUT_TOKENS)
    LAST_BUDGET = budget
    tools = READ_TOOLS + [SUBMIT_TRIAGE_TOOL]
    messages = [{"role": "user", "content":
                 "Read the alerts, context and runbook with the tools, then call submit_triage "
                 "with the ranked probable causes and a proposed next step."}]
    while True:
        budget.begin_iteration()
        resp = _run(messages, tools, budget)
        messages.append({"role": "assistant", "content": resp.content})
        submitted = None
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                budget.record_tool(block.name)
                if block.name == "submit_triage":
                    submitted = dict(block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": "ok"})
                else:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(TOOL_FUNCS[block.name](block.input))})
        if submitted is not None:
            # Guardrail in code: derive top_cause from the ranked list by a deterministic
            # severity score, so it is always self-consistent (never trust the model for this).
            ranked = sorted(submitted.get("ranked_causes", []) or [],
                            key=lambda c: _severity_score(c.get("service", "")), reverse=True)
            submitted["ranked_causes"] = ranked
            if ranked and _severity_score(ranked[0].get("service", "")) > 0:
                submitted["top_cause"] = ranked[0]
            else:
                submitted.pop("top_cause", None)
            if verbose:
                print("  [" + budget.summary() + "]")
            if write_files:
                with open("triage_plan.json", "w") as f:
                    json.dump(submitted, f, indent=2)
                with open("incident_brief.md", "w") as f:
                    f.write("# Ops Triage - Incident Brief\n\n" + submitted.get("incident_summary", "") +
                            "\n\n**Proposed next step (for human approval):** " +
                            submitted.get("proposed_next_step", "") + "\n")
            return submitted
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    load_snapshot("signals.json")
    print("== Phase 1: ask ==")
    print(ask("What is on fire, and what is the single most likely cause? Use the tools.") + "\n")
    print("== Phase 2: triage ==")
    p = triage()
    print(json.dumps(p, indent=2))
    print("\nWrote triage_plan.json and incident_brief.md (read-only: nothing was changed).")
