"""
FinOps Copilot - READ-ONLY agentic layer over the FinOps Guardian.
Phase 0/1: the agent can call tools to READ cost/waste data (changes nothing).
Phase 2: plan() makes the agent produce a STRUCTURED action plan (action_plan.json + exec_brief.md).
Requires: pip install anthropic  and  ANTHROPIC_API_KEY in your environment.
"""
import json
import os
import sys
from anthropic import Anthropic

from report_contract import ReportContractError, describe_problems, validate_report

# Pick a current model from https://docs.claude.com/en/docs/about-claude/models
MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-5")

# The per-call output ceiling. It is passed to the API AND to the budget, so the
# cap projects against the same number the request is actually bounded by.
# Two copies of this value would drift, and the drift would make the cap lie.
MAX_OUTPUT_TOKENS = 1024

# Cliente perezoso (IA-27): antes se instanciaba al importar, lo que exigia
# ANTHROPIC_API_KEY solo para poder importar el modulo. El test negativo de
# IA-27 no llama a la API; debe poder correr gratis y en CI (IA-1).
_CLIENT = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_budget import (  # noqa: E402  (import after the path shim, by necessity)
    BudgetExceeded,
    IterationCapExceeded,
    RunBudget,
)

# The budget of the most recent run, so a caller that did not pass its own can
# still read what the run cost. IA-29: consumption that is not reported is
# consumption nobody controls.
LAST_BUDGET = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _CLIENT

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

def _cost(action):
    """Coerciona el costo a float. Lo que no es numero vale 0 y nunca corona."""
    try:
        return float(action.get("monthly_cost_usd") or 0)
    except (TypeError, ValueError):
        return 0.0

def _waste_as_actions(report, include_context=False):
    """
    Traduccion unica REPORTE -> PLAN. La usan la tool list_waste y la
    derivacion en codigo de IA-27, para que no puedan divergir entre si.
    """
    out = []
    for w in report.get("waste", []) or []:
        action = {
            "resource": w.get("resource"),
            "monthly_cost_usd": w.get("est_monthly_usd"),
            "fix": w.get("action"),
        }
        if include_context:
            action["type"] = w.get("type")
            action["detail"] = w.get("detail")
        out.append(action)
    return out

def list_waste(_a):
    """
    Traduce el vocabulario del REPORTE al vocabulario del PLAN, en codigo.

    El reporte dice est_monthly_usd/action; el plan dice monthly_cost_usd/fix.
    Son dos contratos distintos y esta bien que lo sean — lo que no esta bien
    es que la traduccion la hiciera el modelo, como ocurria antes de IA-26.
    Traducir aqui la vuelve explicita, unica y verificable.
    """
    return _waste_as_actions(REPORT, include_context=True)

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

def _run(messages, tools, budget):
    """
    The single choke point through which every billed call passes.

    The order matters and is the whole point of IA-29: the cap is consulted
    BEFORE the request. Checking afterwards would only ever produce a receipt.
    """
    budget.before_call()
    resp = _get_client().messages.create(model=MODEL, max_tokens=MAX_OUTPUT_TOKENS,
                                         system=SYSTEM, tools=tools, messages=messages)
    budget.record_response(resp)
    return resp

def ask(question, verbose=True, budget=None):
    """Phase 0/1: free-form Q&A over the data (read-only)."""
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


# ---------- IA-27: el codigo es dueno de los numeros; el modelo, del lenguaje ----------
def build_ranked_actions(report, submitted=None):
    """
    Construye ranked_actions DESDE report["waste"], no desde la salida del modelo.

    Antes de IA-27 se ordenaba submitted["ranked_actions"], que es lo que el
    modelo entrego. Eso garantizaba auto-consistencia (el top era el maximo de
    su propia lista) pero no correccion: si el modelo omitia un item o copiaba
    mal una cifra, el codigo ordenaba una lista corrompida con total seguridad.

    Ahora:
      - resource, monthly_cost_usd y el ORDEN salen del reporte. Punto.
      - la REDACCION del fix se injerta del modelo cuando escribio algo para
        ese mismo resource; si no, cae al "action" del reporte.
      - un resource que no existe en el reporte no entra en la lista, y por lo
        tanto no puede coronarse. Es la verificacion de plausibilidad que
        ops-triage/triage.py ya implementaba con _severity_score(...) > 0.
    """
    model_wording = {}
    for a in (submitted or {}).get("ranked_actions") or []:
        if not isinstance(a, dict):
            continue
        resource, fix = a.get("resource"), a.get("fix")
        if resource and isinstance(fix, str) and fix.strip():
            model_wording.setdefault(resource, fix.strip())

    ranked = []
    for item in _waste_as_actions(report):
        action = dict(item)
        wording = model_wording.get(action.get("resource"))
        if wording:
            action["fix"] = wording
        ranked.append(action)

    # Desempate por resource: mismo reporte -> mismo orden, siempre.
    ranked.sort(key=lambda a: (-_cost(a), str(a.get("resource") or "")))
    return ranked

def enforce_source_of_truth(submitted, report=None):
    """Aplica la derivacion en codigo sobre el plan que entrego el modelo."""
    report = REPORT if report is None else report
    ranked = build_ranked_actions(report, submitted)
    submitted["ranked_actions"] = ranked
    submitted["total_monthly_waste_usd"] = round(sum(_cost(a) for a in ranked), 2)
    if ranked and _cost(ranked[0]) > 0:
        submitted["top_action"] = ranked[0]
    else:
        submitted.pop("top_action", None)
    return submitted

def legacy_guardrail(submitted, report=None):
    """
    El guardrail ANTERIOR a IA-27, conservado a proposito.

    No se usa en produccion. Existe para que el test negativo pueda demostrar
    en ROJO lo que este ordenamiento no protege, y para poder explicar el antes
    y el despues sin recurrir al historial de git.
    """
    ranked = sorted(submitted.get("ranked_actions", []) or [],
                    key=lambda a: a.get("monthly_cost_usd", 0), reverse=True)
    submitted["ranked_actions"] = ranked
    if ranked:
        submitted["top_action"] = ranked[0]
    else:
        submitted.pop("top_action", None)
    return submitted

def plan(write_files=True, enforce=True, budget=None, verbose=True):
    """Phase 2: agent reads the data and submits a structured action plan.

    enforce=False desactiva la re-derivacion de IA-27 y vuelve al guardrail
    anterior. Existe para que el test negativo pueda salir en ROJO.
    """
    global LAST_BUDGET
    budget = budget or RunBudget(MODEL, max_output_per_call=MAX_OUTPUT_TOKENS)
    LAST_BUDGET = budget
    tools = READ_TOOLS + [SUBMIT_PLAN_TOOL]
    messages = [{"role": "user", "content":
                 "Read the cost and waste data with the tools, then call submit_plan with the prioritized plan."}]
    while True:
        budget.begin_iteration()
        resp = _run(messages, tools, budget)
        messages.append({"role": "assistant", "content": resp.content})
        submitted = None
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                budget.record_tool(block.name)
                if block.name == "submit_plan":
                    submitted = dict(block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": "ok"})
                else:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(TOOL_FUNCS[block.name](block.input))})
        if submitted is not None:
            # Guardrail in code (IA-27): ranked_actions, el total y el top_action
            # se re-derivan desde REPORT["waste"]. El modelo no es dueno de
            # ninguna cifra ni del orden; solo de como se redacta cada cosa.
            if enforce:
                enforce_source_of_truth(submitted)
            else:
                legacy_guardrail(submitted)  # solo para el test negativo
            if verbose:
                print("  [" + budget.summary() + "]")
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