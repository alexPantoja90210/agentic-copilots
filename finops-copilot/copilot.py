"""
FinOps Copilot - READ-ONLY agentic layer over the FinOps Guardian.
Phase 0/1: the agent can call tools to READ cost/waste data (changes nothing).
Phase 2: plan() makes the agent produce a STRUCTURED action plan (action_plan.json + exec_brief.md).
Requires: pip install anthropic  and  ANTHROPIC_API_KEY in your environment.
"""
import json
import os
from anthropic import Anthropic

from report_contract import ReportContractError, describe_problems, validate_report

# Pick a current model from https://docs.claude.com/en/docs/about-claude/models
MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-5")
client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

REPORT = {}

def load_report(path="report.json"):
    """
    Carga el reporte y EXIGE que cumpla el contrato.

    Antes de IA-26 esta funcion aceptaba cualquier JSON. Si el reporte traia
    otra forma, las tools devolvian None en silencio y el modelo rellenaba el
    hueco por su cuenta: el resultado salia bien por iniciativa del modelo, no
    porque el codigo funcionara. Ahora falla ruidosamente y de inmediato.
    """
    global REPORT
    with open(path) as f:
        data = json.load(f)
    problems = validate_report(data)
    if problems:
        raise ReportContractError(describe_problems(path, problems))
    REPORT = data
    return REPORT

# ---------- read-only tools (none of these change anything) ----------
def get_costs(_a):
    forecast = REPORT.get("forecast", {})
    return {
        "projected_month_end_usd": forecast.get("projected_eom"),
        "month_to_date_usd": forecast.get("mtd"),
        "budget_usd": forecast.get("budget"),
        "budget_status": forecast.get("status"),
        "health_score": REPORT.get("health_score"),
    }

def list_waste(_a):
    """
    Traduce el vocabulario del REPORTE al vocabulario del PLAN, en codigo.

    El reporte dice est_monthly_usd/action; el plan dice monthly_cost_usd/fix.
    Son dos contratos distintos y esta bien que lo sean — lo que no esta bien
    es que la traduccion la hiciera el modelo, como ocurria antes de IA-26.
    Traducir aqui la vuelve explicita, unica y verificable.
    """
    return [
        {
            "resource": w.get("resource"),
            "type": w.get("type"),
            "detail": w.get("detail"),
            "monthly_cost_usd": w.get("est_monthly_usd"),
            "fix": w.get("action"),
        }
        for w in REPORT.get("waste", [])
    ]

def get_budget(_a):
    forecast = REPORT.get("forecast", {})
    return {"budget_usd": forecast.get("budget"), "status": forecast.get("status")}

TOOL_FUNCS = {"get_costs": get_costs, "list_waste": list_waste, "get_budget": get_budget}

READ_TOOLS = [
    {"name": "get_costs", "description": "Get month-to-date spend, projected month-end spend, budget, status and health score.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_waste", "description": "List detected waste items, each with monthly_cost_usd (USD) and the recommended fix.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_budget", "description": "Get the budget guardrail and current status.",
     "input_schema": {"type": "object", "properties": {}}},
]

# Phase 2: a tool the model MUST call to submit its structured plan (structured output).
SUBMIT_PLAN_TOOL = {
    "name": "submit_plan",
    "description": "Submit the final prioritized action plan. Call this once, after reading the data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "total_monthly_waste_usd": {"type": "number"},
            "top_action": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "monthly_cost_usd": {"type": "number"},
                    "fix": {"type": "string"},
                },
                "required": ["resource", "monthly_cost_usd", "fix"],
            },
            "ranked_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "resource": {"type": "string"},
                        "monthly_cost_usd": {"type": "number"},
                        "fix": {"type": "string"},
                    },
                    "required": ["resource", "monthly_cost_usd", "fix"],
                },
            },
            "exec_brief": {"type": "string"},
        },
        "required": ["total_monthly_waste_usd", "ranked_actions", "exec_brief"],
    },
}

SYSTEM = (
    "You are FinOps Copilot, a READ-ONLY cloud cost assistant. "
    "You can call tools to read cost and waste data, but you CANNOT change any cloud resource. "
    "Rules: (1) Use ONLY numbers returned by the tools; never invent figures. "
    "(2) Never claim an action was performed; always PROPOSE actions for human approval. "
    "(3) Rank actions by dollars saved per month, highest first. "
    "(4) In exec_brief, write 2-4 plain sentences a manager can act on; propose, do not assert completion. "
    "(5) If there is no waste, set total_monthly_waste_usd to 0, ranked_actions to an empty list, and do NOT include top_action."
)

def _run(messages, tools):
    return client.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM, tools=tools, messages=messages)

def ask(question, verbose=True):
    """Phase 0/1: free-form Q&A over the data (read-only)."""
    messages = [{"role": "user", "content": question}]
    while True:
        resp = _run(messages, READ_TOOLS)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if verbose:
                    print("  [tool call] " + block.name)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(TOOL_FUNCS[block.name](block.input))})
        messages.append({"role": "user", "content": results})

def plan(write_files=True):
    """Phase 2: agent reads the data and submits a structured action plan."""
    tools = READ_TOOLS + [SUBMIT_PLAN_TOOL]
    messages = [{"role": "user", "content":
                 "Read the cost and waste data with the tools, then call submit_plan with the prioritized plan."}]
    while True:
        resp = _run(messages, tools)
        messages.append({"role": "assistant", "content": resp.content})
        submitted = None
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                if block.name == "submit_plan":
                    submitted = dict(block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": "ok"})
                else:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(TOOL_FUNCS[block.name](block.input))})
        if submitted is not None:
            # Guardrail in code: derive top_action from the ranked list so it is always self-consistent.
            # NOTA (IA-27): esto ordena la salida del modelo, no el dato de origen.
            # Es mas debil de lo que la narrativa del proyecto afirma. Se corrige
            # en IA-27; aqui se deja intacto a proposito para no mezclar alcances.
            ranked = sorted(submitted.get("ranked_actions", []) or [],
                            key=lambda a: a.get("monthly_cost_usd", 0), reverse=True)
            submitted["ranked_actions"] = ranked
            if ranked:
                submitted["top_action"] = ranked[0]
            else:
                submitted.pop("top_action", None)
            if write_files:
                with open("action_plan.json", "w") as f:
                    json.dump(submitted, f, indent=2)
                with open("exec_brief.md", "w") as f:
                    f.write("# FinOps Copilot - Executive Brief\n\n" + submitted.get("exec_brief", "") + "\n")
            return submitted
        messages.append({"role": "user", "content": results})

if __name__ == "__main__":
    load_report("report.json")
    print("== Phase 1: ask ==")
    print(ask("Where is my cloud money going, and what is the single highest-value action? Use the tools.") + "\n")
    print("== Phase 2: plan ==")
    p = plan()
    print(json.dumps(p, indent=2))
    print("\nWrote action_plan.json and exec_brief.md (read-only: nothing in AWS changed).")