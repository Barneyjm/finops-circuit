"""What a conversation cost: tokens times a price table.

A chat API is sent the whole conversation on every turn, so turn k's input is everything said
before it plus the new message. Output tokens come from the data where it records them
(WildChat's `token_counter` on assistant turns) and are estimated otherwise; input tokens are
always estimated, at four characters a token. Estimates are marked, so a report can say how
much of its total is measured.

Prices are list prices per million tokens at each model's release, in USD. Edit PRICES, or
pass your own table, for your contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN = 4

# (input, output) USD per million tokens. Matched by prefix, longest first.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-3.5-turbo": (1.50, 2.00),
    "gpt-4-0314": (30.00, 60.00),
    "gpt-4-0613": (30.00, 60.00),
    "gpt-4-1106": (10.00, 30.00),
    "gpt-4-0125": (10.00, 30.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o": (5.00, 15.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o1-mini": (3.00, 12.00),
    "o1": (15.00, 60.00),
    "gpt-4": (30.00, 60.00),
}
# What a conversation a small model could have handled would cost on it instead.
SMALL_MODEL = "gpt-4o-mini"


def price_of(model: str, prices: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    table = prices or PRICES
    for prefix in sorted(table, key=len, reverse=True):
        if model.startswith(prefix):
            return table[prefix]
    raise KeyError(f"no price for model {model!r}; add it to PRICES")


def tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Cost:
    model: str
    input_tokens: int
    output_tokens: int
    output_measured: int  # of output_tokens, how many the data recorded rather than estimated
    usd: float
    usd_on_small_model: float  # the same tokens at SMALL_MODEL's prices
    resent_usd: float  # of usd, what sending earlier turns again cost: the price of a long conversation


def cost(model: str, turns: list[dict], prices: dict[str, tuple[float, float]] | None = None) -> Cost:
    """`turns`: [{"role": "user"|"assistant", "content": str, "tokens": int | None}], in order."""
    p_in, p_out = price_of(model, prices)
    s_in, s_out = price_of(SMALL_MODEL, prices)
    history = sent = 0  # sent: tokens already sent once as input
    total_in = total_out = measured = resent = 0
    for t in turns:
        n = t.get("tokens") or tokens(t["content"])
        if t["role"] == "assistant":
            total_in += history  # this reply was generated from everything before it
            resent += sent  # of which this much went out on an earlier turn already
            sent = history
            total_out += n
            measured += n if t.get("tokens") else 0
        history += n
    usd = total_in * p_in / 1e6 + total_out * p_out / 1e6
    small = total_in * s_in / 1e6 + total_out * s_out / 1e6
    return Cost(model, total_in, total_out, measured, usd, small, resent * p_in / 1e6)
