"""Findings as a FOCUS 1.4 Cost and Usage dataset (https://focus.finops.org), in CSV.

Each conversation is a usage charge per kind of token: input, cached input (when any was
read from a prompt cache) and output, each priced separately, so each is its own SKU and row.
The mapping:

    ServiceCategory / ServiceSubcategory   AI and Machine Learning / Generative AI
    ServiceProviderName, HostProviderName, InvoiceIssuerName   the model's vendor (OpenAI for gpt-*, o1-*)
    ChargeCategory "Usage", ChargeClass null, ChargeFrequency "Usage-Based", PricingCategory "Standard"
    ConsumedQuantity / ConsumedUnit        tokens / "Tokens"
    PricingQuantity / PricingUnit          tokens / 1e6 / "1000000 Tokens"
    ListUnitPrice, ContractedUnitPrice     the saved cost's price per million, list and after the discount
    ListCost; ContractedCost = EffectiveCost = BilledCost    list, then after the discount, as the
                                           finding was billed (`finops reprice` to change prices)
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
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .agent import NOT_APPLICABLE, Finding
from .circuit import UNTAGGED

TAG_PREFIX = "finops-circuit/"

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


def rows(findings: list[Any], *, account_id: str = "llm-usage", account_name: str = "LLM usage") -> Iterator[dict[str, Any]]:
    """FOCUS rows per finding: input tokens, cached input tokens (when there are any), output
    tokens, billed as the finding's cost line split them, so the rows add up to its spend."""
    for f in findings:
        f = f if isinstance(f, Finding) else Finding.from_dict(f)
        c, who = f.cost, vendor(f.model)
        c_start, c_end, b_start, b_end = _periods(f.timestamp)
        discount = round(1 - c.contracted_usd / c.usd, 6) if c.usd else 0.0
        shared = {
            "BillingAccountId": account_id,
            "BillingAccountName": account_name,
            "BillingCurrency": c.currency,
            "PricingCurrency": c.currency,
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
            "ResourceId": f.id,
            "ResourceName": f.tags.get("app", ""),
            "ResourceType": "Conversation",
            "PricingUnit": "1000000 Tokens",
            "ConsumedUnit": "Tokens",
            "Tags": _tags(f.tags, f.tag_source),
            "x_ConversationModel": f.model,
            "x_TagSources": json.dumps(f.tag_source, sort_keys=True),
            "x_Actions": json.dumps([a.text for a in f.actions], ensure_ascii=False),
            "x_BusinessUse": "" if f.business is None else str(bool(f.business)).lower(),
        }
        charges = [
            ("input", c.input_tokens - c.cached_tokens, c.input_usd, not c.input_measured),
            ("cached-input", c.cached_tokens, c.cached_usd, not c.input_measured),
            ("output", c.output_tokens, c.output_usd, c.output_measured < c.output_tokens),
        ]
        charges = [ch for ch in charges if ch[1] > 0]  # no row for a kind of token the conversation did not use
        for i, (kind, tokens, list_cost, estimated) in enumerate(charges):
            unit = round(list_cost / tokens * 1e6, 6)  # per million, as the table had it
            contracted = list_cost * (1 - discount)
            yield shared | {
                "ChargeDescription": f"{f.model} {kind.replace('-', ' ')} tokens",
                "ConsumedQuantity": tokens,
                "PricingQuantity": tokens / 1e6,
                "ListUnitPrice": unit,
                "ContractedUnitPrice": unit * (1 - discount),
                "ListCost": list_cost,
                "ContractedCost": contracted,
                "EffectiveCost": contracted,
                "BilledCost": contracted,
                "SkuId": f"{f.model}/{kind}-tokens",
                "SkuPriceId": f"{f.model}/{kind}-tokens/list-{unit:g}" + (f"/discount-{discount:g}" if discount else ""),
                "SkuMeter": f"{kind.replace('-', ' ').title()} Tokens",
                "x_QuantityEstimated": str(estimated).lower(),
                # the conversation's potential savings sit on its last row, so they sum once
                "x_PotentialSavings": f.savings_usd if i == len(charges) - 1 else 0.0,
            }


def write_csv(findings: list[Any], path: str | Path, **kw: Any) -> int:
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows(findings, **kw):
            w.writerow(row)
            n += 1
    return n
