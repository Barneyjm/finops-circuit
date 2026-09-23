"""finops fetch|analyze|report|diagram

finops fetch --n 200                                   # a WildChat-4.8M sample into data/wildchat.jsonl
finops analyze samples/02_sql_debug.json --backend jev          # one conversation: tags, cost, actions, reasons
finops report data/wildchat.jsonl --backend jev --by task,subtask  # spend grouped by tags, coverage, savings
finops report data/wildchat.jsonl --backend jev --save data/findings.json
finops html data/findings.json --out report.html        # the dashboard, one self-contained file
finops diagram                                          # the circuit as Mermaid
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .agent import TAG_KEYS, analyze, report
from .backends import BACKENDS, V2_BACKENDS, pick_backend
from .circuit import build_circuit
from .conversations import fetch_wildchat, load
from .html import document, render
from .tags import app_ids, app_labels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="finops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "analyze", "report", "html", "diagram"])
    ap.add_argument("path", nargs="?", default="samples")
    ap.add_argument("--backend", default="jev", help=", ".join(BACKENDS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=200, help="fetch: how many conversations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="fetch: data/wildchat.jsonl; html: report.html")
    ap.add_argument("--save", default=None, help="report: also write the findings (audit included) to this JSON file, for `finops html`")
    ap.add_argument("--with-text", action="store_true", help="html: include each conversation's first message (left out by default)")
    ap.add_argument("--workers", type=int, default=4, help="conversations analyzed at once")
    ap.add_argument("--json", action="store_true", help="print full findings, audit included")
    ap.add_argument("--by", default="task", help=f"report: comma-separated tag keys to group spend by ({', '.join(TAG_KEYS)})")
    ap.add_argument("--v2", action=argparse.BooleanOptionalAction, default=None, help=f"ask the circuit v2 questions (default: on for {', '.join(V2_BACKENDS)})")
    args = ap.parse_args(argv)
    v2 = args.v2 if args.v2 is not None else args.backend in V2_BACKENDS

    if args.command == "diagram":
        print(build_circuit(v2).to_mermaid())
        return
    if args.command == "fetch":
        out = Path(args.out or "data/wildchat.jsonl")
        convs = fetch_wildchat(args.n, args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in convs))
        print(f"wrote {len(convs)} conversations to {out} (WildChat-4.8M, ODC-BY: attribute AI2 when you publish from it)")
        return
    if args.command == "html":
        saved = json.loads(Path(args.path).read_text())
        texts = {r["id"]: r["first_message"] for r in saved["findings"] if r.get("first_message")} if args.with_text else None
        body = render(saved["findings"], apps=saved.get("apps"), backend=saved.get("backend", ""), source=saved.get("source", ""), texts=texts)
        out = Path(args.out or "report.html")
        out.write_text(document(body))
        print(f"wrote {out} ({len(saved['findings'])} conversations{', with their first messages' if texts else ''})")
        return

    backend = pick_backend(args.backend, args.model)
    circuit = build_circuit(v2)
    convs = load(args.path)
    by = tuple(k.strip() for k in args.by.split(",") if k.strip())
    if unknown := [k for k in by if k not in TAG_KEYS]:
        raise SystemExit(f"--by: unknown tag keys {unknown}; use {', '.join(TAG_KEYS)}")
    apps = app_ids(convs)  # a template shows only across conversations, so the whole set is fingerprinted first

    def one(conv):
        try:
            return analyze(conv, backend, model=args.model, circuit=circuit, app=apps[conv["id"]])
        except Exception as e:
            print(f"   {conv.get('file', conv['id'])}: skipped ({type(e).__name__}: {str(e)[:120]})", file=sys.stderr)
            return None

    findings = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (conv, f) in enumerate(zip(convs, pool.map(one, convs), strict=True), 1):  # map keeps the input order
            if f is None:
                continue
            findings.append(f)
            if args.command == "analyze":
                print(f.to_json() if args.json else _line(conv, f))
            elif i % 100 == 0:
                print(f"   {i}/{len(convs)}", file=sys.stderr, flush=True)
    if args.command == "report":
        print(json.dumps(report(findings, by, app_labels(convs)), indent=1))
        if args.save:
            first = {c["id"]: next((t["content"] for t in c["turns"] if t["role"] == "user"), "")[:300] for c in convs}
            rows = [asdict(f) | {"first_message": first.get(f.id, "")} for f in findings]
            payload = {"backend": getattr(backend, "model", args.backend), "source": args.path, "apps": app_labels(convs), "findings": rows}
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(json.dumps(payload, ensure_ascii=False, default=str))
            print(f"saved {len(rows)} findings to {args.save}", file=sys.stderr)


def _line(conv, f) -> str:
    first = next((t["content"] for t in conv["turns"] if t["role"] == "user"), "")
    head = " ".join(first.split())[:110]
    a = f.audit["answers"]
    lines = [
        f"\n== {conv.get('file', f.id)}  {f.model}, {f.turns} replies, ${f.cost.usd:.4f}  ({f.cost.input_tokens} in / {f.cost.output_tokens} out)",
        f"   {head}",
        "-> " + " ".join(f"{k}={f.tags[k]}" for k in TAG_KEYS),
        f"   business={f.business} saves ${f.savings_usd:.4f} | small model ok {a['small_model_ok']:.2f} | repeatable {a['repeatable']:.2f}",
    ]
    lines += [f"   * {x}" for x in f.actions] or ["   * no action"]
    for gid in ("task", "subtask", "environment", "data_class", "downgrade"):
        g = f.audit["gates"].get(gid)
        if g:
            lines.append(f"   {gid:12s} {g['outcome']:9s} {'; '.join(g['trace'] or [])[:140]}")
    lines.append(f"   {f.audit['model']} in {f.audit['latency_ms']:.0f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
