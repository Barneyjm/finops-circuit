"""finops fetch|analyze|report|diagram

finops fetch --n 200                                   # a WildChat-4.8M sample into data/wildchat.jsonl
finops analyze samples/03_sql_debug.json --backend jev  # one conversation: cost, value, actions, reasons
finops report data/wildchat.jsonl --backend circuits    # the review: spend by value, savings by action
finops diagram                                          # the circuit as Mermaid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import analyze, report
from .backends import BACKENDS, V2_BACKENDS, pick_backend
from .circuit import build_circuit
from .conversations import fetch_wildchat, load


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "analyze", "report", "diagram"])
    ap.add_argument("path", nargs="?", default="samples")
    ap.add_argument("--backend", default="jev", help=", ".join(BACKENDS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=200, help="fetch: how many conversations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/wildchat.jsonl")
    ap.add_argument("--json", action="store_true", help="print full findings, audit included")
    ap.add_argument("--v2", action=argparse.BooleanOptionalAction, default=None, help=f"ask the circuit v2 questions (default: on for {', '.join(V2_BACKENDS)})")
    args = ap.parse_args(argv)
    v2 = args.v2 if args.v2 is not None else args.backend in V2_BACKENDS

    if args.command == "diagram":
        print(build_circuit(v2).to_mermaid())
        return
    if args.command == "fetch":
        convs = fetch_wildchat(args.n, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs))
        print(f"wrote {len(convs)} conversations to {args.out} (WildChat-4.8M, ODC-BY: attribute AI2 when you publish from it)")
        return

    backend = pick_backend(args.backend, args.model)
    circuit = build_circuit(v2)
    convs = load(args.path)
    findings = []
    for conv in convs:
        try:
            f = analyze(conv, backend, model=args.model, circuit=circuit)
        except Exception as e:
            print(f"   {conv.get('file', conv['id'])}: skipped ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            continue
        findings.append(f)
        if args.command == "analyze":
            print(f.to_json() if args.json else _line(conv, f))
    if args.command == "report":
        print(json.dumps(report(findings), indent=1))


def _line(conv, f) -> str:
    first = next((t["content"] for t in conv["turns"] if t["role"] == "user"), "")
    head = " ".join(first.split())[:110]
    a = f.audit["answers"]
    top = max(a["value_type"], key=a["value_type"].get)
    lines = [
        f"\n== {conv.get('file', f.id)}  {f.model}, {f.turns} replies, ${f.cost.usd:.4f}  ({f.cost.input_tokens} in / {f.cost.output_tokens} out)",
        f"   {head}",
        f"-> value={f.value} business={f.business} saves ${f.savings_usd:.4f}",
        f"   value_type {top} {a['value_type'][top]:.2f} | work {a['work']:.2f} | small model ok {a['small_model_ok']:.2f} | repeatable {a['repeatable']:.2f} | sensitive {a['sensitive']:.2f}",
    ]
    lines += [f"   * {x}" for x in f.actions] or ["   * no action"]
    for gid in ("value", "business", "downgrade", "policy"):
        g = f.audit["gates"][gid]
        lines.append(f"   {gid:10s} {g['outcome']:9s} {'; '.join(g['trace'] or [])[:150]}")
    lines.append(f"   {f.audit['model']} in {f.audit['latency_ms']:.0f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
