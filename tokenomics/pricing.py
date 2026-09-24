"""What a conversation cost: tokens times a price table (`prices.toml`, or your own via --prices).

A chat API is sent the whole conversation on every turn, so a reply's input is everything said
before it. Where a log records what the provider billed (an assistant turn's `usage`:
`input_tokens`, `output_tokens`, `cached_tokens`), those counts are used as they are and cached
tokens are priced at the cache rate. Where it does not, output tokens come from the turn's
`tokens` (WildChat records them) or its length, input tokens are the history before the reply,
nothing is cached, and the counts are marked as estimated.

The history resent on each turn is exactly what a provider's prompt cache serves: `resent_tokens`
are those sent again at full price, which `Price.cache_savings` prices at the cache rate instead.
Counting (`cost`) and billing (`bill`) are separate, so saved counts can be billed again under a
new table without the conversation.
"""

from __future__ import annotations

import functools
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

CHARS_PER_TOKEN = 4
DEFAULT_PRICES = Path(__file__).with_name("prices.toml")


@dataclass(frozen=True)
class Price:
    input: float  # per million tokens
    output: float
    cached_input: float | None = None  # None: no prompt cache for this model
    discount: float = 0.0  # contracted discount off list, 0 to 1

    @property
    def cached(self) -> float:
        """What a cached input token bills at: the cache price, or full price without a cache."""
        return self.input if self.cached_input is None else self.cached_input

    def cache_savings(self, tokens: int) -> float:
        """What serving `tokens` of input from a prompt cache would take off the list price."""
        return tokens * (self.input - self.cached) / 1e6


@dataclass
class PriceTable:
    models: dict[str, Price]
    currency: str = "USD"
    small_model: str = "gpt-4o-mini"
    _hits: dict[str, Price] = field(default_factory=dict, init=False, repr=False, compare=False)

    def of(self, model: str) -> Price:
        """The price of the longest key `model` starts with."""
        if model not in self._hits:
            prefix = max((k for k in self.models if model.startswith(k)), key=len, default=None)
            if prefix is None:
                raise KeyError(f"no price for model {model!r}; add it to the price table")
            self._hits[model] = self.models[prefix]
        return self._hits[model]


def load_prices(path: str | Path | None = None) -> PriceTable:
    raw = tomllib.loads(Path(path or DEFAULT_PRICES).read_text())
    settings = raw.get("settings", {})
    models = {}
    for name, spec in raw.get("models", {}).items():
        p = Price(float(spec["input"]), float(spec["output"]), float(spec["cached_input"]) if "cached_input" in spec else None, float(spec.get("discount", 0.0)))
        if not 0 <= p.discount < 1:
            raise ValueError(f"model {name!r}: discount must be from 0 to 1")
        models[name] = p
    table = PriceTable(models, settings.get("currency", "USD"), settings.get("small_model", "gpt-4o-mini"))
    table.of(table.small_model)  # the downgrade target has to be priced
    return table


@functools.cache
def default_prices() -> PriceTable:
    return load_prices()


def tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Cost:
    model: str
    input_tokens: int  # all input, cached included
    output_tokens: int
    output_measured: int = 0  # of output_tokens, how many the log recorded rather than estimated
    cached_tokens: int = 0  # of input, read from the provider's cache
    input_measured: bool = False  # the log recorded input and cache counts; otherwise estimated
    resent_tokens: int = 0  # history sent again at full price: what a prompt cache would serve
    usd: float = 0.0  # at list prices
    contracted_usd: float = 0.0  # after the contract discount
    usd_on_small_model: float = 0.0  # the same tokens at the small model's prices
    resent_usd: float = 0.0  # what resent_tokens cost
    input_usd: float = 0.0  # usd split by charge, as FOCUS bills it: full-price input,
    cached_usd: float = 0.0  # cached input,
    output_usd: float = 0.0  # and output
    currency: str = "USD"


def bill(c: Cost, table: PriceTable) -> Cost:
    """`c`'s token counts at `table`'s prices: every dollar field again, the counts as they are."""
    p, s = table.of(c.model), table.of(table.small_model)
    full_in = c.input_tokens - c.cached_tokens
    input_usd, cached_usd, output_usd = full_in * p.input / 1e6, c.cached_tokens * p.cached / 1e6, c.output_tokens * p.output / 1e6
    usd = input_usd + cached_usd + output_usd
    return replace(
        c,
        usd=usd,
        contracted_usd=usd * (1 - p.discount),
        usd_on_small_model=(full_in * s.input + c.cached_tokens * s.cached + c.output_tokens * s.output) / 1e6,
        resent_usd=c.resent_tokens * p.input / 1e6,
        input_usd=input_usd,
        cached_usd=cached_usd,
        output_usd=output_usd,
        currency=table.currency,
    )


def cost(model: str, turns: list[dict], prices: PriceTable | None = None) -> Cost:
    """`turns`: [{"role", "content", "tokens"?, "usage"?, "prompt"?}], in order. An assistant turn's
    `usage` ({"input_tokens", "output_tokens", "cached_tokens"}) is the provider's own count; one
    marked `prompt` was sent as input (a few-shot example), not generated, and bills as input."""
    history = sent = 0  # sent: tokens already sent once as input
    total_in = cached = total_out = measured_out = resent = replies = 0
    measured_in = True
    for t in turns:
        n = t.get("tokens") or tokens(t["content"])
        if t["role"] == "assistant" and not t.get("prompt"):  # a reply generated on this bill
            replies += 1
            u = t.get("usage") or {}
            if "input_tokens" in u:
                total_in += int(u["input_tokens"])
                cached += int(u.get("cached_tokens", 0))
            else:
                measured_in = False
                total_in += history  # this reply was generated from everything before it
            resent += sent  # of which this much went out on an earlier turn already
            sent = history
            out = int(u.get("output_tokens", n))
            total_out += out
            measured_out += out if ("output_tokens" in u or t.get("tokens")) else 0
            n = out
        history += n
    counts = Cost(model, total_in, total_out, measured_out, cached, measured_in and replies > 0, max(0, resent - cached))
    return bill(counts, prices or default_prices())
