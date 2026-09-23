"""What a conversation cost: tokens times a price table (`prices.toml`, or your own via --prices).

A chat API is sent the whole conversation on every turn, so a reply's input is everything said
before it. Where a log records what the provider billed (an assistant turn's `usage`:
`input_tokens`, `output_tokens`, `cached_tokens`), those counts are used as they are and cached
tokens are priced at the cache rate. Where it does not, output tokens come from the turn's
`tokens` (WildChat records them) or its length, input tokens are the history before the reply,
nothing is cached, and the counts are marked as estimated.

The history resent on each turn is exactly what a provider's prompt cache serves, so the
tokens resent at full price are also what turning caching on would bill at the cached rate:
`caching_savings_usd` is that difference, for models that have a cache price.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CHARS_PER_TOKEN = 4
DEFAULT_PRICES = Path(__file__).with_name("prices.toml")


@dataclass
class Price:
    input: float  # per million tokens
    output: float
    cached_input: float | None = None  # None: no prompt cache for this model
    discount: float = 0.0  # contracted discount off list, 0 to 1


@dataclass
class PriceTable:
    models: dict[str, Price]
    currency: str = "USD"
    small_model: str = "gpt-4o-mini"

    def of(self, model: str) -> Price:
        for prefix in sorted(self.models, key=len, reverse=True):
            if model.startswith(prefix):
                return self.models[prefix]
        raise KeyError(f"no price for model {model!r}; add it to the price table")


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


_DEFAULT: PriceTable | None = None


def default_prices() -> PriceTable:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_prices()
    return _DEFAULT


def price_of(model: str, prices: PriceTable | None = None) -> tuple[float, float]:
    """(input, output) list price per million tokens."""
    p = (prices or default_prices()).of(model)
    return p.input, p.output


def tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Cost:
    model: str
    input_tokens: int  # all input, cached included
    cached_tokens: int  # of input, read from the provider's cache
    output_tokens: int
    input_measured: bool  # the log recorded input and cache counts; otherwise estimated
    output_measured: int  # of output_tokens, how many the log recorded rather than estimated
    usd: float  # at list prices
    contracted_usd: float  # after the contract discount
    usd_on_small_model: float  # the same tokens at the small model's prices
    resent_usd: float  # what sending earlier turns again cost at full price
    resent_tokens: int  # those tokens: history resent and not served from a cache
    caching_savings_usd: float  # what prompt caching would take off resent_usd (0 without a cache price)
    currency: str = "USD"
    input_usd: float = field(default=0.0)
    cached_usd: float = field(default=0.0)
    output_usd: float = field(default=0.0)


def cost(model: str, turns: list[dict], prices: PriceTable | None = None) -> Cost:
    """`turns`: [{"role", "content", "tokens"?, "usage"?}], in order. An assistant turn's
    `usage` ({"input_tokens", "output_tokens", "cached_tokens"}) is the provider's own count."""
    table = prices or default_prices()
    p, s = table.of(model), table.of(table.small_model)
    history = sent = 0  # sent: tokens already sent once as input
    total_in = cached = total_out = measured_out = resent = 0
    measured_in = True
    replies = 0
    for t in turns:
        n = t.get("tokens") or tokens(t["content"])
        if t["role"] == "assistant":
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
    full_in = total_in - cached
    cached_price = p.cached_input if p.cached_input is not None else p.input
    input_usd, cached_usd, output_usd = full_in * p.input / 1e6, cached * cached_price / 1e6, total_out * p.output / 1e6
    usd = input_usd + cached_usd + output_usd
    s_cached = s.cached_input if s.cached_input is not None else s.input
    small = (full_in * s.input + cached * s_cached + total_out * s.output) / 1e6
    uncached_resent = max(0, resent - cached)  # history the cache could have served but did not
    savings = uncached_resent * (p.input - p.cached_input) / 1e6 if p.cached_input is not None else 0.0
    return Cost(
        model,
        total_in,
        cached,
        total_out,
        measured_in and replies > 0,
        measured_out,
        usd,
        usd * (1 - p.discount),
        small,
        uncached_resent * p.input / 1e6,
        uncached_resent,
        savings,
        table.currency,
        input_usd,
        cached_usd,
        output_usd,
    )
