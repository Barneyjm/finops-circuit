"""Tag questions as held-out eval rows for training circuit models (s1-proto's choice-row format).

    uv run python scripts/export_circuit_eval.py --out ../s1-proto/data/open_tax_tagger_eval.jsonl

Two families: `tagger_sample`, this repo's 13 hand-labelled samples; `tagger_ref_<tag>`, WildChat
conversations with Jev's saved tags (data/findings1000.json) as a proxy reference. Eval only:
Jev is a reference, never a training target. Kept here, not in the public model repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tokenomics.circuit import load_taxonomy
from tokenomics.conversations import load, transcript

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("task", "domain", "environment", "workload", "data_class")


def row(family: str, state, instructions: str, criteria: dict, gold: str) -> dict:
    ref = {k: float(k == gold) for k in criteria}
    q = {"type": "choice", "instructions": instructions, "criteria": criteria}
    key = json.dumps([state, q], sort_keys=True, ensure_ascii=False)
    return {
        "id": f"{family}-{hashlib.sha1(key.encode()).hexdigest()[:10]}",
        "family": family,
        "kind": "choice",
        "heldout": True,
        "state": state,
        "question": q,
        "refs": {"jev": ref, "gemini": ref},  # the format's slots; the one reference stands in for both
        "ref": ref,
        "source": "open_taxonomy",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", type=int, default=200, help="WildChat conversations with the reference tags")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tx = load_taxonomy()
    rows: list[dict] = []

    def ask(family, conv, tag, gold, parent=None):
        t = tx.tags[tag]
        if parent is None:
            crit, q = dict(t.options), t.question
        else:
            crit = dict(t.by_parent.get(parent, {}))
            if len(crit) < 2:
                return
            q = t.question.format(parent=parent.replace("_", " "))
        if gold in crit:
            rows.append(row(family, {"transcript": transcript(conv)}, q, crit, gold))

    for c in load(ROOT / "samples"):
        exp = c.get("expected_answers", {})
        for tag in TAGS:
            if tag in exp:
                ask("tagger_sample", c, tag, max(exp[tag], key=exp[tag].get))
        if "subtask" in exp and "task" in exp:
            ask("tagger_sample", c, "subtask", max(exp["subtask"], key=exp["subtask"].get), parent=max(exp["task"], key=exp["task"].get))
    saved = json.loads((ROOT / "data/findings1000.json").read_text())
    convs = {c["id"]: c for c in load(ROOT / saved["source"])}
    for f in rng.sample(saved["findings"], min(args.ref, len(saved["findings"]))):
        c = convs.get(f["id"])
        if not c:
            continue
        for tag in TAGS:
            if f["tags"].get(tag) not in (None, "untagged"):
                ask(f"tagger_ref_{tag}", c, tag, f["tags"][tag])
        if f["tags"].get("subtask") not in (None, "untagged", "n/a"):
            ask("tagger_ref_subtask", c, "subtask", f["tags"]["subtask"], parent=f["tags"]["task"])
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
