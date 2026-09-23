"""Findings as a FOCUS 1.4 Cost and Usage dataset (https://focus.finops.org), in CSV.

Each conversation is a usage charge per kind of token: input, cached input (when any was
read from a prompt cache) and output, each priced separately, so each is its own SKU and row.
The mapping:

    ServiceCategory / ServiceSubcategory   AI and Machine Learning / Generative AI
    ServiceProviderName, HostProviderName, InvoiceIssuerName   the model's vendor (OpenAI for gpt-*, o1-*)
    ChargeCategory "Usage", ChargeClass null, ChargeFrequency "Usage-Based", PricingCategory "Standard"
    ConsumedQuantity / ConsumedUnit        tokens / "Tokens"
    PricingQuantity / PricingUnit          tokens / 1e6 / "1000000 Tokens"
    ListUnitPrice, ContractedUnitPrice     the price table's list price, and after its contract discount
    ListCost; ContractedCost = EffectiveCost = BilledCost    list, then after the discount (prices.toml)
    ChargePeriodStart/End                  the hour of the conversation; BillingPeriod its month
    ResourceId / ResourceName / ResourceType   the conversation / its app / "Conversation"
    SkuId, SkuPriceId, SkuMeter            "<model>/input-tokens" and so on
    Tags                                   declared tags as given; inferred and code tags under the
                                           "finops-circuit/" prefix, their own tag scheme
    x_ columns                             tag sources, estimated quantities, actions, potential savings

FOCUS asks for one prefix-free user tag scheme and a prefix for every other scheme, so what the
conversation declared keeps its keys and what this tool inferred carries `finops-circuit/`.
Untagged and n/a tags are left out: Tags holds values, not the absence of one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .circuit import UNTAGGED
from .pricing import PriceTable, default_prices

TAG_PREFIX = "finops-circuit/"
NOT_APPLICABLE = "n/a"

COLUMNS = [
    # mandatory in FOCUS 1.4
    "BilledCost", "BillingAccountId", "BillingAccountName", "BillingCurrency", "BillingPeriodEnd", "BillingPeriodStart",
    "ChargeCategory", "ChargeClass", "ChargeDescription", "ChargePeriodEnd", "ChargePeriodStart", "ContractedCost",
    "EffectiveCost", "HostProviderName", "InvoiceIssuerName", "ListCost", "PricingQuantity", "PricingUnit",
    "ServiceCategory", "ServiceName", "ServiceProviderName",
    # conditional and recommended, all applicable here
    "ChargeFrequency", "ConsumedQuantity", "ConsumedUnit", "ContractedUnitPrice", "ListUnitPrice", "PricingCategory",
    "PricingCurrency", "ResourceId", "ResourceName", "ResourceType", "ServiceSubcategory", "SkuId", "SkuMeter", "SkuPriceId", "Tags",
    # custom
    "x_ConversationModel", "x_QuantityEstimated", "x_TagSources", "x_Actions", "x_PotentialSavings", "x_BusinessUse",
]  # fmt: skip

VENDORS = (("gpt-", "OpenAI"), ("o1", "OpenAI"), ("o3", "OpenAI"), ("o4", "OpenAI"), ("claude", "Anthropic"), ("gemini", "Google"), ("mistral", "Mistral AI"))


def vendor(model: str) -> str:
    return next((v for prefix, v in VENDORS if model.startswith(prefix)), "Unknown")


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _periods(timestamp: str | None) -> tuple[str, str, str, str]:
    """(charge start, charge end, billing start, billing end): the hour and the month, UTC."""
    t = datetime.fromisoformat(timestamp).replace(tzinfo=UTC) if timestamp else datetime.now(UTC)
    hour = t.replace(minute=0, second=0, microsecond=0)
    month = hour.replace(day=1, hour=0)
    next_month = (month + timedelta(days=32)).replace(day=1)
    return _iso(hour), _iso(hour + timedelta(hours=1)), _iso(month), _iso(next_month)


def _tags(tags: dict[str, str], source: dict[str, str]) -> str:
    out = {}
    for k, v in tags.items():
        if v in (UNTAGGED, NOT_APPLICABLE):
            continue
        out[k if source.get(k) == "declared" else TAG_PREFIX + k] = v
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


def rows(findings: list[Any], *, account_id: str = "llm-usage", account_name: str = "LLM usage", prices: PriceTable | None = None) -> list[dict[str, Any]]:
    """FOCUS rows per finding: input tokens, cached input tokens (when there are any), output tokens."""
    table = prices or default_prices()
    out = []
    for f in findings:
        d = asdict(f) if not isinstance(f, dict) else f
        c, model = d["cost"], d["model"]
        who = vendor(model)
        price = table.of(model)
        c_start, c_end, b_start, b_end = _periods(d.get("timestamp"))
        source = d.get("tag_source") or {k: ("code" if k == "app" else "inferred") for k in d["tags"]}
        currency = c.get("currency", table.currency)
        shared = {
            "BillingAccountId": account_id,
            "BillingAccountName": account_name,
            "BillingCurrency": currency,
            "PricingCurrency": currency,
            "BillingPeriodStart": b_start,
            "BillingPeriodEnd": b_end,
            "ChargePeriodStart": c_start,
            "ChargePeriodEnd": c_end,
            "ChargeCategory": "Usage",
            "ChargeClass": "",
            "ChargeFrequency": "Usage-Based",
            "PricingCategory": "Standard",
            "ServiceCategory": "AI and Machine Learning",
            "ServiceSubcategory": "Generative AI",
            "ServiceName": f"{who} API",
            "ServiceProviderName": who,
            "HostProviderName": who,
            "InvoiceIssuerName": who,
            "ResourceId": d["id"],
            "ResourceName": d["tags"].get("app", ""),
            "ResourceType": "Conversation",
            "PricingUnit": "1000000 Tokens",
            "ConsumedUnit": "Tokens",
            "Tags": _tags(d["tags"], source),
            "x_ConversationModel": model,
            "x_TagSources": json.dumps(source, sort_keys=True),
            "x_Actions": json.dumps(d["actions"], ensure_ascii=False),
            "x_BusinessUse": "" if d["business"] is None else str(bool(d["business"])).lower(),
        }
        cached = c.get("cached_tokens", 0)
        input_measured = c.get("input_measured", False)
        charges = [("input", c["input_tokens"] - cached, price.input, not input_measured)]
        if cached:
            charges.append(("cached-input", cached, price.cached_input if price.cached_input is not None else price.input, not input_measured))
        charges.append(("output", c["output_tokens"], price.output, c["output_measured"] < c["output_tokens"]))
        for i, (kind, tokens, unit, estimated) in enumerate(charges):
            list_cost = tokens * unit / 1e6
            contracted = list_cost * (1 - price.discount)
            last = i == len(charges) - 1
            out.append(
                shared
                | {
                    "ChargeDescription": f"{model} {kind.replace('-', ' ')} tokens",
                    "ConsumedQuantity": tokens,
                    "PricingQuantity": tokens / 1e6,
                    "ListUnitPrice": unit,
                    "ContractedUnitPrice": unit * (1 - price.discount),
                    "ListCost": list_cost,
                    "ContractedCost": contracted,
                    "EffectiveCost": contracted,
                    "BilledCost": contracted,
                    "SkuId": f"{model}/{kind}-tokens",
                    "SkuPriceId": f"{model}/{kind}-tokens/list-{unit:g}" + (f"/discount-{price.discount:g}" if price.discount else ""),
                    "SkuMeter": f"{kind.replace('-', ' ').title()} Tokens",
                    "x_QuantityEstimated": str(estimated).lower(),
                    # the conversation's potential savings sit on its last row, so they sum once
                    "x_PotentialSavings": d["savings_usd"] if last else 0.0,
                }
            )
    return out


def write_csv(findings: list[Any], path: str | Path, **kw: Any) -> int:
    data = rows(findings, **kw)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(data)
    return len(data)
