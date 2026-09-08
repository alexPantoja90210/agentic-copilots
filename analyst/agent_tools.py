"""
The two tools the agent is given, and the sandbox around the dangerous one.

The agent must COMPUTE, not recall. It gets a description of the corpus and a
pandas expression evaluator over it, and nothing else: no file access, no
network, no imports, no way out of the namespace. Every call it makes is
recorded, and an answer that arrives with no call behind it is scored
`unsupported` however right the number looks.

IA-68, under IA-64.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

MAX_RESULT_CHARS = 4000

# Names the expression may reference at the top level. Everything else is
# rejected by name before any evaluation happens.
ALLOWED_NAMES = frozenset({
    "tickets", "agents", "pd", "np",
    "len", "sum", "min", "max", "round", "sorted", "abs", "list", "dict",
    "set", "tuple", "int", "float", "str", "bool", "range", "zip",
    "enumerate", "any", "all",
})


class ToolError(Exception):
    pass


def check_expression(expression: str) -> None:
    """
    Reject anything that could leave the namespace, by parsing the expression
    rather than searching its text.

    Substring matching was the obvious implementation and it is the wrong one.
    A rule that rejects any expression containing "import" also rejects
    `tickets["Import Date"]`, and a rule loose enough to allow that column name
    admits the keyword. It is the defect this project keeps finding -- a check
    that cannot distinguish two situations it treats as one -- so the check is
    made on the parse tree, where a name and a keyword are different objects and
    cannot be confused.

    What is refused:
      * anything that fails to parse as a single expression;
      * any identifier or attribute beginning with an underscore, which is how
        every escape from a restricted namespace begins;
      * any top-level name outside ALLOWED_NAMES.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("not a single valid Python expression: %s" % exc)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ToolError(
                "attribute %r begins with an underscore. Private and dunder "
                "attributes are how a restricted namespace is escaped."
                % node.attr)
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ToolError("name %r begins with an underscore" % node.id)
            if node.id not in ALLOWED_NAMES:
                raise ToolError(
                    "name %r is not available. You have: tickets, agents, pd, "
                    "np, and the common builtins." % node.id)
    return None


def describe_corpus(tickets: pd.DataFrame, agents: pd.DataFrame) -> str:
    """
    The shape of the data, so the agent need not guess column names.

    Deliberately shows dtypes, row counts and the distinct values of the small
    categorical columns -- which is schema, not answer. It does NOT show any
    aggregate: no mean, no count per category, no correlation. Those are the
    questions.
    """
    lines = []
    for name, frame in (("tickets", tickets), ("agents", agents)):
        lines.append("%s: %d rows x %d columns" % (name, len(frame), len(frame.columns)))
        for column in frame.columns:
            series = frame[column]
            note = ""
            if series.dtype == object and series.nunique() <= 8:
                note = "  values: %s" % sorted(series.dropna().unique().tolist())
            lines.append("  %-24s %s%s" % (column, series.dtype, note))
        lines.append("")
    return "\n".join(lines)


def run_query(expression: str, tickets: pd.DataFrame, agents: pd.DataFrame) -> str:
    """Evaluate one pandas expression against the corpus and return its repr."""
    check_expression(expression)
    namespace = {
        "__builtins__": {name: __builtins__[name] if isinstance(__builtins__, dict)
                         else getattr(__builtins__, name)
                         for name in ALLOWED_NAMES
                         if name not in ("tickets", "agents", "pd", "np")},
        "tickets": tickets, "agents": agents, "pd": pd, "np": np,
    }
    try:
        value = eval(compile(ast.parse(expression, mode="eval"), "<query>", "eval"),
                     namespace, {})
    except Exception as exc:                      # the agent's error, not ours
        raise ToolError("%s: %s" % (type(exc).__name__, exc))

    text = repr(value) if not isinstance(value, (pd.DataFrame, pd.Series)) \
        else value.to_string()
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "\n... truncated at %d characters" % MAX_RESULT_CHARS
    return text


TOOLS = [
    {
        "name": "describe_corpus",
        "description": "Column names, dtypes and row counts for the two tables, "
                       "plus the distinct values of small categorical columns. "
                       "Returns no aggregates.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_query",
        "description": "Evaluate a single pandas expression against the corpus. "
                       "Two DataFrames are in scope: `tickets` and `agents`. "
                       "`pd` and `np` are available. Example: "
                       "tickets.groupby('Request Category').size(). "
                       "One expression per call, no statements, no imports.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {
                "type": "string",
                "description": "A single Python expression over `tickets` and "
                               "`agents`."}},
            "required": ["expression"],
        },
    },
]


def dispatch(name: str, arguments: dict, tickets, agents) -> tuple[str, bool]:
    """Returns (result text, is_error). An error is data for the agent, not a crash."""
    try:
        if name == "describe_corpus":
            return describe_corpus(tickets, agents), False
        if name == "run_query":
            return run_query(arguments.get("expression", ""), tickets, agents), False
        return "no tool named %r" % name, True
    except ToolError as exc:
        return str(exc), True
