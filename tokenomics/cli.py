"""tokenomics fetch|analyze|report|diagram

tokenomics fetch --n 200                                   # a WildChat-4.8M sample into data/wildchat.jsonl
tokenomics import litellm-logs.jsonl --out data/mine.jsonl # gateway logs (LiteLLM, Helicone, OpenAI pairs) as conversations
tokenomics analyze samples/02_sql_debug.json                # one conversation: tags, cost, actions, reasons
tokenomics report data/wildchat.jsonl --by task,subtask     # spend grouped by tags, coverage, savings
tokenomics report data/wildchat.jsonl --save data/findings.json
tokenomics report data/big.jsonl --sample 400 --save f.json  # tag 400 cost-weighted draws; shares with 90% intervals
tokenomics html data/findings.json --out report.html        # the dashboard, one self-contained file
tokenomics focus data/findings.json --out focus.csv          # the same spend as a FOCUS 1.4 dataset
tokenomics reprice data/findings.json --prices my-prices.toml  # the saved findings under new prices, no model calls
                                                        # (--prices on focus or html reprices on the fly)
tokenomics diagram                                          # the circuit as Mermaid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .agent import analyze, report, reprice, sample, tag_keys
from .backends import BACKENDS, V1_MODELS, V2_BACKENDS, pick_backend, speaks_v2
from .circuit import build_circuit, load_taxonomy
from .conversations import fetch_wildchat, load
from .focus import write_csv
from .html import document, render
from .logs import read as read_logs
from .pricing import cost, load_prices
from .tags import app_ids, app_labels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tokenomics", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "import", "analyze", "report", "html", "focus", "reprice", "diagram"])
    ap.add_argument("path", nargs="?", default="samples")
    ap.add_argument("--backend", default="circuits", help=", ".join(BACKENDS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=200, help="fetch: how many conversations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="fetch: data/wildchat.jsonl; import: data/imported.jsonl; html: report.html; reprice: the findings file itself")
    ap.add_argument("--save", default=None, help="report: also write the findings (audit included) to this JSON file, for `tokenomics html`")
    ap.add_argument("--with-text", action="store_true", help="html: include each conversation's first message (left out by default)")
    ap.add_argument("--workers", type=int, default=4, help="conversations analyzed at once")
    ap.add_argument("--account", default="llm-usage", help="focus: BillingAccountId (BillingAccountName is the same unless --account-name)")
    ap.add_argument("--account-name", default=None)
    ap.add_argument("--json", action="store_true", help="print full findings, audit included")
    ap.add_argument("--by", default=None, help="report: comma-separated tag keys to group spend by (any taxonomy tag, app, or a declared tag)")
    ap.add_argument("--format", default="auto", help="import: litellm, helicone, openai, or auto (from the first row's keys)")
    ap.add_argument("--sample", type=float, default=None, help="report: tag only N cost-weighted draws (or a fraction, 0.02), with 90%% intervals; spend is still counted in full")
    ap.add_argument("--prices", default=None, help="a price table TOML (default: tokenomics/prices.toml)")
    ap.add_argument("--taxonomy", default=None, help="a tag taxonomy TOML (default: tokenomics/taxonomy.toml)")
    ap.add_argument("--v2", action=argparse.BooleanOptionalAction, default=None, help=f"ask the circuit v2 questions (default: on for {', '.join(V2_BACKENDS)}, not for {', '.join(V1_MODELS)})")
    args = ap.parse_args(argv)
    v2 = args.v2 if args.v2 is not None else speaks_v2(args.backend, args.model)
    taxonomy = load_taxonomy(args.taxonomy)
    prices = load_prices(args.prices)

    if args.command == "diagram":
        print(build_circuit(taxonomy, v2).to_mermaid())
        return
    if args.command == "fetch":
        out = Path(args.out or "data/wildchat.jsonl")
        convs = fetch_wildchat(args.n, args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs))
        print(f"wrote {len(convs)} conversations to {out} (WildChat-4.8M, ODC-BY: attribute AI2 when you publish from it)")
        return
    if args.command == "import":
        convs = read_logs(args.path, args.format)
        out = Path(args.out or "data/imported.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs))
        calls = sum(1 for c in convs for t in c["turns"] if t["role"] == "assistant" and not t.get("prompt"))
        print(f"wrote {len(convs)} conversations ({calls} calls) to {out}")
        return
    if args.command in ("focus", "reprice", "html"):
        saved = json.loads(Path(args.path).read_text())
        findings = saved["findings"]
        if args.prices or args.command == "reprice":  # bill the saved counts again; the model is not asked
            findings = [reprice(r, taxonomy=taxonomy, prices=prices) for r in findings]
        n = len(findings)
        if args.command == "focus":
            out = Path(args.out or "focus.csv")
            rows = write_csv(findings, out, account_id=args.account, account_name=args.account_name or args.account)
            print(f"wrote {rows} FOCUS 1.4 rows ({n} conversations: input, cached input when any, and output tokens) to {out}")
        elif args.command == "reprice":
            by = _by(args.by, taxonomy, {k for f in findings for k in f.tags})
            print(json.dumps(report(findings, by, saved.get("apps")), indent=1))
            out = Path(args.out or args.path)
            rows = [_saved(f, r.get("first_message", "")) for f, r in zip(findings, saved["findings"], strict=True)]
            out.write_text(json.dumps(saved | {"findings": rows}, ensure_ascii=False, default=str))
            print(f"repriced {n} findings into {out}", file=sys.stderr)
        else:
            texts = {r["id"]: r["first_message"] for r in saved["findings"] if r.get("first_message")} if args.with_text else None
            body = render(findings, apps=saved.get("apps"), backend=saved.get("backend", ""), source=saved.get("source", ""), texts=texts)
            out = Path(args.out or "report.html")
            out.write_text(document(body))
            print(f"wrote {out} ({n} conversations{', with their first messages' if texts else ''})")
        return

    backend = pick_backend(args.backend, args.model)
    circuit = build_circuit(taxonomy, v2)
    convs = load(args.path)
    by = _by(args.by, taxonomy, {str(k) for c in convs for k in (c.get("tags") or {})})
    apps = app_ids(convs)  # a template shows only across conversations, so the whole set is fingerprinted first
    labels = app_labels(convs)
    draws: Counter[int] = Counter()
    if args.sample:  # tag a cost-weighted sample; costs need no model, so they are known for all first
        costs = [cost(c["model"], c["turns"], prices).usd for c in convs]
        n = int(args.sample) if args.sample >= 1 else max(1, math.ceil(args.sample * len(convs)))
        draws = sample(costs, n, args.seed)
        per_draw = sum(costs) / n
        print(f"   sample: {n} cost-weighted draws, {len(draws)} distinct of {len(convs)} conversations, ${sum(costs):.4f} in all", file=sys.stderr)
        convs = [convs[i] for i in sorted(draws)]
        draws = Counter({convs[j]["id"]: draws[i] for j, i in enumerate(sorted(draws))})

    def one(conv):
        try:
            return analyze(conv, backend, model=args.model, circuit=circuit, taxonomy=taxonomy, app=apps[conv["id"]], prices=prices)
        except Exception as e:
            print(f"   {conv.get('file', conv['id'])}: skipped ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            return None

    findings = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (conv, f) in enumerate(zip(convs, pool.map(one, convs), strict=True), 1):  # map keeps the input order
            if f is None:
                continue
            if draws:
                f.draws, f.weight_usd = draws[f.id], draws[f.id] * per_draw
            findings.append(f)
            if args.command == "analyze":
                print(f.to_json() if args.json else _line(conv, f))
            elif i % 100 == 0:
                print(f"   {i}/{len(convs)}", file=sys.stderr, flush=True)
    if args.command == "report":
        print(json.dumps(report(findings, by, labels), indent=1))
        if args.save:
            first = {c["id"]: next((t["content"] for t in c["turns"] if t["role"] == "user"), "")[:300] for c in convs}
            rows = [_saved(f, first.get(f.id, "")) for f in findings]
            payload = {"backend": getattr(backend, "model", args.backend), "source": args.path, "apps": labels, "findings": rows}
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(json.dumps(payload, ensure_ascii=False, default=str))
            print(f"saved {len(rows)} findings to {args.save}", file=sys.stderr)


def _by(arg: str | None, taxonomy, extra: set[str]) -> tuple[str, ...]:
    """--by as tag keys, checked against the taxonomy's and any declared ones."""
    by = tuple(k.strip() for k in (arg or taxonomy.top[0].name).split(",") if k.strip())
    known = set(tag_keys(taxonomy)) | extra
    if unknown := [k for k in by if k not in known]:
        raise SystemExit(f"--by: unknown tag keys {unknown}; use {', '.join(sorted(known))}")
    return by


def _saved(f, first_message: str) -> dict:
    """A finding as --save writes it: every field, the total savings for readers of the JSON, the first message."""
    return asdict(f) | {"savings_usd": f.savings_usd, "first_message": first_message}


def _line(conv, f) -> str:
    first = next((t["content"] for t in conv["turns"] if t["role"] == "user"), "")
    head = " ".join(first.split())[:110]
    a = f.audit["answers"]
    lines = [
        f"\n== {conv.get('file', f.id)}  {f.model}, {f.turns} replies, ${f.cost.usd:.4f}  ({f.cost.input_tokens} in / {f.cost.output_tokens} out)",
        f"   {head}",
        "-> " + " ".join(f"{k}={v}" for k, v in f.tags.items()),
        f"   business={f.business} saves ${f.savings_usd:.4f} | small model ok {a['small_model_ok']:.2f} | repeatable {a['repeatable']:.2f}",
    ]
    lines += [f"   * {x.text}" for x in f.actions] or ["   * no action"]
    for gid in [*(k for k in f.tags if k != "app"), "downgrade"]:
        g = f.audit["gates"].get(gid)
        if g:
            lines.append(f"   {gid:12s} {g['outcome']:9s} {'; '.join(g['trace'] or [])[:140]}")
    lines.append(f"   {f.audit['model']} in {f.audit['latency_ms']:.0f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
