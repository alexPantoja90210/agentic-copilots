"""
El contrato del report.json — definido UNA sola vez.

guardian.py es el productor y es dueño de la forma del reporte. Este módulo
escribe esa forma de manera explícita para que todo consumidor la exija en vez
de suponerla.

Vive aparte de copilot.py a propósito: evals.py necesita validar sin importar
el agente, porque importar copilot.py exigiría ANTHROPIC_API_KEY y el
--selftest dejaría de ser gratis.

Origen: IA-26. El colector reestructuró su salida, el agente y los evals se
quedaron leyendo la forma anterior, y nada lo detectó porque los fixtures
también eran viejos. Un contrato implícito no puede romperse ruidosamente.
"""

# Campos que el reporte debe traer. Si guardian.py cambia su salida, este es el
# archivo que hay que actualizar — y el eval se pondrá ROJO hasta que se haga.
REQUIRED_FORECAST_FIELDS = ("projected_eom", "budget", "status")
REQUIRED_WASTE_FIELDS = ("resource", "est_monthly_usd", "action")


class ReportContractError(ValueError):
    """El reporte no cumple el contrato. Se falla ruidosamente, no en silencio."""


def validate_report(report):
    """
    Devuelve la lista de incumplimientos del contrato.
    Lista vacía = el reporte es válido.

    Devuelve una lista en vez de lanzar, para que cada consumidor decida cómo
    reaccionar: copilot.py revienta al cargar, evals.py reporta ROJO.
    """
    problems = []

    if not isinstance(report, dict):
        return ["el reporte no es un objeto JSON"]

    forecast = report.get("forecast")
    if forecast is None:
        problems.append("falta 'forecast'")
    elif not isinstance(forecast, dict):
        # El esquema viejo tenia 'forecast' como numero suelto. Este es el
        # sintoma exacto de la deriva que documenta IA-26.
        problems.append(
            "'forecast' no es un objeto (el esquema anterior lo tenia como numero)"
        )
    else:
        for field in REQUIRED_FORECAST_FIELDS:
            if field not in forecast:
                problems.append("'forecast' no trae '%s'" % field)

    if "health_score" not in report:
        problems.append("falta 'health_score'")

    waste = report.get("waste")
    if waste is None:
        problems.append("falta la lista 'waste'")
    elif not isinstance(waste, list):
        problems.append("'waste' no es una lista")
    else:
        for i, item in enumerate(waste):
            if not isinstance(item, dict):
                problems.append("waste[%d] no es un objeto" % i)
                continue
            for field in REQUIRED_WASTE_FIELDS:
                if field not in item:
                    problems.append("waste[%d] no trae '%s'" % (i, field))

    return problems


def describe_problems(path, problems):
    """Mensaje de error legible, con la ruta del archivo y cada incumplimiento."""
    return "%s no cumple el contrato del reporte:\n  - %s" % (
        path,
        "\n  - ".join(problems),
    )