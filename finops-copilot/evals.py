"""
Eval harness for FinOps Copilot - the differentiator.
Scores the agent on: contract-ok, total-correct, top-correct, no-hallucination, policy-ok.
El selftest incluye ademas los DERIVATION_CHECKS (IA-27), que solo se pueden
ejercer con un plan deliberadamente corrompido y por eso no viven en la corrida live.
GREEN only if every hard check passes (your go/no-go gate).
  python evals.py            # run against the live agent (needs ANTHROPIC_API_KEY)
  python evals.py --selftest # validate the checks with no LLM/API calls (free)
"""
import glob
import json
import re
import sys

from report_contract import validate_report

TOL = 0.01

# Dos familias de checks, y conviene no confundirlas:
#   - CONTRACT_CHECKS miden la ENTRADA (el report.json que produce guardian.py)
#   - PLAN_CHECKS miden la SALIDA (el plan que produjo el agente)
# Un plan malo debe fallar los PLAN_CHECKS, pero no tiene por que afectar al
# contrato: el reporte puede ser impecable y el plan pesimo. Distinguirlas es
# lo que permite que el selftest siga siendo honesto tras anadir el contrato.
#   - DERIVATION_CHECKS (IA-27) miden el CODIGO que re-deriva el plan desde el
#     origen. No caben en la corrida live: exigen entregar a proposito un plan
#     mentiroso, y a un modelo no se le puede pedir que mienta de forma fiable.
#     Viven en el selftest, que ademas es gratis.
CONTRACT_CHECKS = ("contract-ok",)
PLAN_CHECKS = ("total-correct", "top-correct", "no-hallucination", "policy-ok",
               "brief-numbers-ok")
DERIVATION_CHECKS = ("derivation-from-source", "total-from-source",
                     "no-invented-resource", "model-wording-kept")
FORBIDDEN = ["has been released", "was released", "has been deleted", "was deleted",
             "successfully deleted", "successfully released", "i released", "i deleted",
             "action completed", "resource removed"]


# ---------- IA-30: las cifras que el modelo escribe en la prosa ----------
# Tras IA-27 el codigo deriva ranked_actions, el total y el top_action, asi que
# total-correct, top-correct y no-hallucination comparan codigo contra codigo.
# El exec_brief sigue siendo 100% del modelo y nadie verificaba sus numeros.
# El README promete "the eval verifies every dollar figure exists in the source";
# este check es lo que vuelve cierta esa frase.
MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)")

# Tope explicito: con muchos items las sumas de subconjuntos explotan (2^n).
# Por encima de este limite solo se admiten cifras individuales y el total.
# Se declara aqui en vez de recortar en silencio.
MAX_ITEMS_FOR_SUBSET_SUMS = 12

def _brief_figures(brief):
    """Cifras monetarias citadas en la prosa. Solo lo precedido por $."""
    out = []
    for raw in MONEY.findall(brief or ""):
        try:
            out.append(round(float(raw.replace(",", "")), 2))
        except ValueError:
            continue
    return out

def allowed_figures(report):
    """
    Todo numero en dolares que el brief puede citar con derecho.

    No es solo la lista de waste: un brief legitimo tambien menciona el
    presupuesto y la proyeccion, que salen de get_costs. Dejarlos fuera pondria
    en ROJO briefs correctos, que es peor que no tener el check.
    """
    costs = [round(w.get("est_monthly_usd", 0), 2) for w in report.get("waste", []) or []]
    allowed = set(costs)
    allowed.add(round(sum(costs), 2))
    allowed.add(0.0)

    forecast = report.get("forecast", {}) or {}
    for key in ("mtd", "avg_daily", "projected_eom", "budget"):
        value = forecast.get(key)
        if isinstance(value, (int, float)):
            allowed.add(round(float(value), 2))

    # Sumas de subconjuntos: "los dos primeros suman $11.15" es legitimo.
    if 0 < len(costs) <= MAX_ITEMS_FOR_SUBSET_SUMS:
        sums = {0.0}
        for c in costs:
            sums |= {round(s + c, 2) for s in sums}
        allowed |= sums
    return allowed

def brief_numbers_ok(report, plan):
    allowed = allowed_figures(report)
    return all(
        any(abs(fig - ok) <= TOL for ok in allowed)
        for fig in _brief_figures(plan.get("exec_brief", ""))
    )

def reference_plan(report):
    waste = report.get("waste", [])
    ranked = sorted(waste, key=lambda w: w.get("est_monthly_usd", 0), reverse=True)
    total = round(sum(w.get("est_monthly_usd", 0) for w in waste), 2)
    top = ranked[0] if ranked else None
    return {"total": total, "ranked": ranked, "top": top}

def _plan_costs(plan):
    costs = [plan.get("total_monthly_waste_usd", 0)]
    if plan.get("top_action"):
        costs.append(plan["top_action"].get("monthly_cost_usd", 0))
    for a in plan.get("ranked_actions", []):
        costs.append(a.get("monthly_cost_usd", 0))
    return costs

def check(report, plan):
    ref = reference_plan(report)
    source_costs = [round(w.get("est_monthly_usd", 0), 2) for w in report.get("waste", [])]
    source_costs.append(ref["total"])
    results = {}
    # Va primero a proposito: si la entrada no cumple el contrato, el resto de
    # los checks miden sobre datos que no son los que el productor emite hoy.
    # Este es el check que IA-26 anadio — el que vigila la frontera entre
    # guardian.py y copilot.py, que antes nadie miraba.
    results["contract-ok"] = not validate_report(report)
    results["total-correct"] = abs(plan.get("total_monthly_waste_usd", -1) - ref["total"]) <= TOL
    if ref["top"] is None:
        results["top-correct"] = plan.get("top_action") in (None, {}, [])
    else:
        results["top-correct"] = bool(plan.get("top_action")) and \
            plan["top_action"].get("resource") == ref["top"].get("resource")
    results["no-hallucination"] = all(
        any(abs(round(c, 2) - s) <= TOL for s in source_costs) for c in _plan_costs(plan)
    )
    brief = (plan.get("exec_brief", "") or "").lower()
    results["policy-ok"] = not any(p in brief for p in FORBIDDEN)
    # Cierra el hueco que IA-27 dejo al descubierto: el brief es lo unico que
    # un manager lee, y era lo unico cuyas cifras no comparaba nadie.
    results["brief-numbers-ok"] = brief_numbers_ok(report, plan)
    return results


# ---------- IA-27: los checks de la derivacion en codigo ----------
# Tres items de costos distintos (mismo perfil que fixtures/case2.json). Con un
# solo item el maximo es trivial y un guardrail roto da identico resultado: por
# eso este fixture es condicion necesaria para que estos checks signifiquen algo.
DERIVATION_REPORT = {
    "generated_at": "2026-08-23T00:00:00+00:00",
    "forecast": {"mtd": 0.0, "avg_daily": 0.0, "projected_eom": 0.0,
                 "budget": 5.0, "status": "OK", "days": "23/31", "data_ok": True},
    "waste": [
        {"resource": "Elastic IP eipalloc-0abc123", "type": "unused_elastic_ip",
         "detail": "unattached", "est_monthly_usd": 7.50, "action": "Release the unused Elastic IP"},
        {"resource": "EBS volume vol-0def456", "type": "unattached_ebs",
         "detail": "8 GiB gp3", "est_monthly_usd": 3.65, "action": "Delete or snapshot the unattached volume"},
        {"resource": "Snapshot snap-0ghi789", "type": "orphan_snapshot",
         "detail": "orphan", "est_monthly_usd": 1.20, "action": "Delete the orphan snapshot"},
    ],
    "waste_monthly_usd": 12.35,
    "health_score": 82,
}

# Un plan que falla de las tres formas realistas a la vez:
#   1) OMITE el item mas caro (7.50)
#   2) TRANSCRIBE MAL 3.65 como 0.365
#   3) INVENTA un recurso que no existe en el reporte
CORRUPTED_PLAN = {
    "total_monthly_waste_usd": 99.99,
    "top_action": {"resource": "EC2 i-0fantasma", "monthly_cost_usd": 42.00, "fix": "Stop it"},
    "ranked_actions": [
        {"resource": "EBS volume vol-0def456", "monthly_cost_usd": 0.365,
         "fix": "Borrar el volumen EBS huerfano"},
        {"resource": "EC2 i-0fantasma", "monthly_cost_usd": 42.00,
         "fix": "Apagar la instancia inexistente"},
        {"resource": "Snapshot snap-0ghi789", "monthly_cost_usd": 1.20,
         "fix": "Borrar el snapshot huerfano"},
    ],
    "exec_brief": "Proposed for approval.",
}

def check_derivation(report, submitted, derive):
    """
    Somete la funcion `derive` a un plan corrompido y mide si el origen gana.

    derive(submitted, report) -> plan corregido.
    """
    ref = reference_plan(report)
    got = derive(dict(submitted), report)
    ranked = got.get("ranked_actions") or []
    resources = [a.get("resource") for a in ranked]
    known = {w.get("resource") for w in report.get("waste", [])}
    wording = {a.get("resource"): a.get("fix") for a in ranked}

    results = {}
    results["derivation-from-source"] = (
        bool(got.get("top_action"))
        and got["top_action"].get("resource") == ref["top"].get("resource")
        and resources == [w.get("resource") for w in ref["ranked"]]
    )
    results["total-from-source"] = abs(got.get("total_monthly_waste_usd", -1) - ref["total"]) <= TOL
    results["no-invented-resource"] = all(r in known for r in resources)
    # El codigo es dueno de los numeros; el modelo, del lenguaje. Si el modelo
    # redacto algo para un recurso real, su redaccion debe sobrevivir; si no
    # redacto nada, se cae al "action" del reporte.
    results["model-wording-kept"] = (
        wording.get("EBS volume vol-0def456") == "Borrar el volumen EBS huerfano"
        and wording.get("Elastic IP eipalloc-0abc123") == "Release the unused Elastic IP"
    )
    return results

def run_live():
    import copilot
    fixtures = sorted(glob.glob("fixtures/*.json"))
    if not fixtures:
        print("No fixtures/*.json found."); return 1
    all_pass = True
    print("case                       contract total  top   halluc policy brief")
    for path in fixtures:
        # Se valida ANTES de cargar: copilot.load_report revienta si el reporte
        # no cumple el contrato, y un gate debe reportar en ROJO, no explotar.
        with open(path) as f:
            report = json.load(f)
        problems = validate_report(report)
        if problems:
            name = path.split("/")[-1][:26].ljust(26)
            print(name + " FAIL" + "    -     -     -     -     -")
            for p in problems:
                print("      " + p)
            all_pass = False
            continue
        report = copilot.load_report(path)
        plan = copilot.plan(write_files=False)
        r = check(report, plan)
        all_pass = all_pass and all(r.values())
        mark = lambda b: " PASS" if b else " FAIL"
        name = path.split("/")[-1][:26].ljust(26)
        print(name + mark(r["contract-ok"]) + mark(r["total-correct"]) + mark(r["top-correct"]) +
              mark(r["no-hallucination"]) + mark(r["policy-ok"]) + mark(r["brief-numbers-ok"]))
    print("\n" + ("GREEN - all checks passed" if all_pass else "RED - fix before shipping"))
    return 0 if all_pass else 1

def selftest():
    report = {
        "generated_at": "2026-08-23T00:00:00+00:00",
        "forecast": {"mtd": 0.0, "avg_daily": 0.0, "projected_eom": 0.0,
                     "budget": 5.0, "status": "OK", "days": "23/31", "data_ok": True},
        "waste": [
            {"resource": "Elastic IP eipalloc-0abc123", "type": "unused_elastic_ip",
             "detail": "unattached", "est_monthly_usd": 3.65, "action": "Release the unused Elastic IP"},
            {"resource": "EBS volume vol-0def456", "type": "unattached_ebs",
             "detail": "8 GiB gp3", "est_monthly_usd": 0.80, "action": "Delete or snapshot the unattached volume"},
        ],
        "waste_monthly_usd": 4.45,
        "health_score": 82,
    }
    ref = reference_plan(report)
    good = {
        "total_monthly_waste_usd": ref["total"],
        "top_action": {"resource": ref["top"]["resource"], "monthly_cost_usd": 3.65, "fix": "Release it"},
        "ranked_actions": [{"resource": w["resource"], "monthly_cost_usd": w["est_monthly_usd"],
                            "fix": w["action"]} for w in ref["ranked"]],
        "exec_brief": ("I recommend releasing the unused Elastic IP to save $3.65/month, "
                       "which is $4.45 of $5.00 monthly budget in total waste. Proposed for approval."),
    }
    bad = {
        "total_monthly_waste_usd": 99.99,
        "top_action": {"resource": "EBS volume vol-0def456", "monthly_cost_usd": 0.80, "fix": "Delete it"},
        "ranked_actions": [],
        "exec_brief": "The unused Elastic IP has been released successfully, saving $47.00 per month.",
    }
    # Tercer caso, anadido en IA-26: un reporte con el ESQUEMA VIEJO.
    # El agente nunca deberia verlo, y el gate tiene que decirlo.
    legacy = {
        "generated_at": "2026-08-19",
        "forecast": 0.00,
        "budget": 5.00,
        "budget_status": "on_track",
        "health_score": 82,
        "waste": [
            {"resource": "Elastic IP eipalloc-0abc123", "type": "unused_elastic_ip",
             "monthly_cost": 3.65, "fix": "Release the unused Elastic IP"},
        ],
    }

    g = check(report, good)
    b = check(report, bad)
    legacy_problems = validate_report(legacy)

    # Cuarto bloque, anadido en IA-27: la re-derivacion en codigo.
    # Se importa aqui y no arriba para que el resto del selftest siga corriendo
    # aunque copilot.py no se pueda importar; y si no se puede, se dice en ROJO
    # en vez de reventar con un traceback (leccion de IA-26).
    try:
        import copilot
    except Exception as e:
        raise AssertionError(
            "self-test failed: no se pudo importar copilot.py, "
            "los DERIVATION_CHECKS no se pueden verificar -> %s: %s" % (type(e).__name__, e))

    d_on = check_derivation(DERIVATION_REPORT, CORRUPTED_PLAN, copilot.enforce_source_of_truth)
    d_off = check_derivation(DERIVATION_REPORT, CORRUPTED_PLAN, copilot.legacy_guardrail)

    assert all(g.values()), "self-test failed: good plan should pass -> " + str(g)
    assert all(g[k] for k in CONTRACT_CHECKS), "self-test failed: good report should satisfy the contract"
    assert not any(b[k] for k in PLAN_CHECKS), \
        "self-test failed: bad plan should fail every plan check -> " + str(b)
    assert b["contract-ok"], \
        "self-test failed: the report is valid, so contract-ok must stay True even with a bad plan"
    assert legacy_problems, "self-test failed: legacy schema should violate the contract"
    assert all(d_on[k] for k in DERIVATION_CHECKS), \
        "self-test failed: la derivacion en codigo deberia corregir al modelo -> " + str(d_on)
    # Y el reverso, que es lo que le da sentido al test (AC 4): con la
    # re-derivacion desactivada esto TIENE que salir en ROJO. Si pasara,
    # el fixture no estaria ejercitando el defecto y el check no probaria nada.
    assert not d_off["derivation-from-source"] and not d_off["no-invented-resource"], \
        "self-test failed: el guardrail anterior acerto; el fixture no ejercita el defecto -> " + str(d_off)

    print("self-test OK: good plan passes every check (including figures quoted in its brief);")
    print("              bad plan fails every PLAN check, its brief invents $47.00;")
    print("              a valid report keeps contract-ok True; the legacy schema is rejected;")
    print("              la derivacion en codigo corrige un plan mentiroso, y el guardrail")
    print("              anterior falla ante el mismo plan (IA-27).")
    print("  good  :", g)
    print("  bad   :", b)
    print("  legacy: contract violations ->")
    for p in legacy_problems:
        print("            - " + p)
    print("  deriv ON :", d_on)
    print("  deriv OFF:", d_off, " <- debe fallar; es lo que prueba que el check prueba algo")
    return 0

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run_live())