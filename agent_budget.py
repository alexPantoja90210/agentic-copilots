"""
Consumption control for both agents.

This project built a watchdog for someone else's AWS bill. For weeks that
watchdog did not watch its own: four `while True:` loops with no cap, and a
`response.usage` field that arrived on every call and was thrown away.

Three guarantees, moved out of intention and into code:

  1. ITERATION CAP  — a loop that never gets its stop condition ends anyway,
                      loudly, saying how many turns it took.
  2. ACCOUNTING     — every call's input and output tokens are recorded, along
                      with which tool was called and on which turn.
  3. BUDGET CAP     — checked BEFORE a call, not after. Stopping once the money
                      is already spent is not a cap, it is a receipt.

Guarantee 3 has an honest limit worth stating in the open: the size of a call
cannot be known before making it. What CAN be bounded is the output, because
`max_tokens` is a hard ceiling we set ourselves, and the input, because after
the first turn the history only grows -- so the largest input seen so far is a
lower bound on the next one. The estimate is therefore conservative from turn
two onward, and the FIRST call is always allowed: refusing to start on the
grounds that the unknown might be too large would make the agent useless.

That limit is the reason `estimated` appears in the report next to `spent`.
"""
import json
import os

DEFAULT_MAX_ITERATIONS = 8
_PRICING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "model-pricing.json")


class ConsumptionError(Exception):
    """Base for every reason a run is stopped on consumption grounds."""


class IterationCapExceeded(ConsumptionError):
    """The loop ran more turns than it was allowed."""


class BudgetExceeded(ConsumptionError):
    """The next call would cross the budget cap, so it was never made."""


def load_pricing(path=_PRICING_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class RunBudget:
    """
    One instance per run. Not reusable on purpose: a budget that survives across
    runs is not a per-run budget.
    """

    def __init__(self, model, max_iterations=DEFAULT_MAX_ITERATIONS,
                 max_tokens=None, max_usd=None, max_output_per_call=1024,
                 pricing=None):
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.max_output_per_call = max_output_per_call
        self.pricing = pricing if pricing is not None else load_pricing()

        self.iterations = 0
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls = []
        self._largest_input = 0

    # ---- rates -----------------------------------------------------------

    def _rates(self):
        model_prices = self.pricing.get("models", {}).get(self.model)
        if model_prices is None:
            return None
        return model_prices["input"], model_prices["output"]

    def cost_usd(self, input_tokens=None, output_tokens=None):
        """
        Estimated cost. Returns None when the model is not in the price file --
        an unknown price is reported as unknown, never as zero. A zero would
        silently disable the cap.
        """
        rates = self._rates()
        if rates is None:
            return None
        rate_in, rate_out = rates
        used_in = self.input_tokens if input_tokens is None else input_tokens
        used_out = self.output_tokens if output_tokens is None else output_tokens
        return (used_in * rate_in + used_out * rate_out) / 1_000_000

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    # ---- the three guarantees -------------------------------------------

    def begin_iteration(self):
        """Call at the top of the loop, before anything else."""
        if self.max_iterations is not None and self.iterations >= self.max_iterations:
            raise IterationCapExceeded(
                "stopped after %d iterations without the model reaching a final "
                "answer (cap: %d). Consumed %d input + %d output tokens across "
                "%d calls. Tools called: %s"
                % (self.iterations, self.max_iterations, self.input_tokens,
                   self.output_tokens, self.calls,
                   ", ".join(self.tool_names()) or "none"))
        self.iterations += 1

    def before_call(self):
        """
        Call immediately before every API request. Raises instead of returning
        a flag so that a caller cannot ignore it by accident.
        """
        if self.calls == 0:
            return  # nothing observed yet; the first call is always allowed
        projected_in = self.input_tokens + self._largest_input
        projected_out = self.output_tokens + self.max_output_per_call
        projected_total = projected_in + projected_out

        if self.max_tokens is not None and projected_total > self.max_tokens:
            raise BudgetExceeded(
                "the next call was NOT made: it would take the run to about "
                "%d tokens, over the cap of %d. Spent so far: %d input + %d "
                "output = %d tokens across %d calls."
                % (projected_total, self.max_tokens, self.input_tokens,
                   self.output_tokens, self.total_tokens, self.calls))

        if self.max_usd is not None:
            projected_cost = self.cost_usd(projected_in, projected_out)
            if projected_cost is None:
                raise BudgetExceeded(
                    "a USD cap of %.4f was set but model '%s' is not in the "
                    "price file, so the cost cannot be estimated and the cap "
                    "cannot be enforced. Refusing to continue blind."
                    % (self.max_usd, self.model))
            if projected_cost > self.max_usd:
                raise BudgetExceeded(
                    "the next call was NOT made: it would take the run to about "
                    "$%.4f, over the cap of $%.4f. Spent so far: $%.4f over %d "
                    "calls (%d input + %d output tokens)."
                    % (projected_cost, self.max_usd, self.cost_usd() or 0.0,
                       self.calls, self.input_tokens, self.output_tokens))

    def record_response(self, response):
        """Call right after every API request, whatever it returned."""
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        used_in = int(getattr(usage, "input_tokens", 0) or 0)
        used_out = int(getattr(usage, "output_tokens", 0) or 0)
        self.input_tokens += used_in
        self.output_tokens += used_out
        self._largest_input = max(self._largest_input, used_in)

    def record_tool(self, name):
        self.tool_calls.append({"iteration": self.iterations, "tool": name})

    # ---- reporting -------------------------------------------------------

    def tool_names(self):
        seen = []
        for entry in self.tool_calls:
            if entry["tool"] not in seen:
                seen.append(entry["tool"])
        return seen

    def report(self):
        return {
            "model": self.model,
            "iterations": self.iterations,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.cost_usd(),
            "prices_verified": bool(self.pricing.get("verified")),
            "tool_calls": list(self.tool_calls),
            "caps": {"iterations": self.max_iterations,
                     "tokens": self.max_tokens,
                     "usd": self.max_usd},
        }

    def summary(self):
        cost = self.cost_usd()
        if cost is None:
            money = "cost unknown (model '%s' not in the price file)" % self.model
        else:
            money = "~$%.4f USD" % cost
            if not self.pricing.get("verified"):
                money += " (estimate; prices unverified)"
        tools = ", ".join("%s x%d" % (n, sum(1 for t in self.tool_calls
                                             if t["tool"] == n))
                          for n in self.tool_names()) or "none"
        return ("consumption: %d iterations, %d calls, %d in + %d out = %d "
                "tokens, %s | tools: %s"
                % (self.iterations, self.calls, self.input_tokens,
                   self.output_tokens, self.total_tokens, money, tools))


def merge_reports(reports):
    """Aggregate several run reports. Used by the live gate across fixtures."""
    total = {"runs": 0, "iterations": 0, "calls": 0, "input_tokens": 0,
             "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0,
             "cost_known": True}
    for report in reports:
        total["runs"] += 1
        for key in ("iterations", "calls", "input_tokens", "output_tokens",
                    "total_tokens"):
            total[key] += report[key]
        if report["estimated_cost_usd"] is None:
            total["cost_known"] = False
        else:
            total["estimated_cost_usd"] += report["estimated_cost_usd"]
    if not total["cost_known"]:
        total["estimated_cost_usd"] = None
    return total
