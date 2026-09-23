"""A self-contained HTML dashboard of a set of findings: spend by any tags, coverage, actions,
and the conversation ledger with each one's gate traces.

    finops report data/wildchat.jsonl --backend jev --save data/findings.json
    finops html data/findings.json --out report.html

The page carries its data inline and loads nothing but two web fonts, so it opens from disk.
Conversation text is left out unless `with_text` is set: a cost dashboard gets passed around,
and what people typed should not travel with it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from typing import Any

from .agent import TAG_KEYS

GATES_SHOWN = ("task", "subtask", "workload", "environment", "domain", "data_class", "business", "downgrade", "cache", "hold", "dev_test")


def _row(f: Any, text: str | None) -> dict[str, Any]:
    d = asdict(f) if not isinstance(f, dict) else f
    c = d["cost"]
    gates = d.get("audit", {}).get("gates", {})
    return {
        "id": d["id"][:12],
        "model": d["model"],
        "replies": d["turns"],
        "usd": c["usd"],
        "in": c["input_tokens"],
        "out": c["output_tokens"],
        "tags": d["tags"],
        "source": d.get("tag_source") or {k: ("code" if k == "app" else "inferred") for k in TAG_KEYS},
        "business": d["business"],
        "actions": d["actions"],
        "savings": d["savings_usd"],
        "gates": {k: {"value": gates[k]["value"], "outcome": gates[k]["outcome"], "trace": "; ".join(gates[k].get("trace") or [])} for k in GATES_SHOWN if k in gates},
        **({"text": text} if text else {}),
    }


def render(findings: list[Any], *, apps: dict[str, str] | None = None, backend: str = "", source: str = "", texts: dict[str, str] | None = None) -> str:
    """The page body (title, style, markup, data, script), without the html/head/body wrapper."""
    data = {
        "rows": [_row(f, (texts or {}).get(f["id"] if isinstance(f, dict) else f.id)) for f in findings],
        "keys": list(TAG_KEYS),
        "apps": apps or {},
        "backend": backend,
        "source": source,
        "date": date.today().isoformat(),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload)


def document(body: str) -> str:
    """The body wrapped as a standalone file."""
    return f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n{body}\n</body>\n</html>\n'


PAGE = r"""<title>LLM Spend Ledger</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600;700&display=swap">
<style>
:root {
  --ground: #F6F7F5; --panel: #FFFFFF; --ink: #1B2124; --muted: #5E686D; --rule: #DDE2E0; --faint: #EEF1EF;
  --accent: #2F4FB5; --accent-soft: #E3E8F8; --good: #2E7D4F; --good-soft: #E1F1E7; --warn: #B7791F; --warn-soft: #F7ECD9;
  --crit: #B83A3A; --crit-soft: #F6E1E1; --focus: #2F4FB5;
  --t0: #2F4FB5; --t1: #4A68C8; --t2: #6A84D3; --t3: #8BA0DD; --t4: #AABBE6; --t5: #C8D3EF; --t6: #DCE3F5;
  --on0: #FFFFFF; --on1: #FFFFFF; --on2: #FFFFFF; --on3: #1B2124; --on4: #1B2124; --on5: #1B2124; --on6: #1B2124;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #12161A; --panel: #192026; --ink: #E4E8EB; --muted: #97A2AA; --rule: #2B343B; --faint: #20282F;
    --accent: #8EA6FF; --accent-soft: #232C4A; --good: #6CC592; --good-soft: #1D3226; --warn: #E3B061; --warn-soft: #3A2E1B;
    --crit: #EC8A8A; --crit-soft: #3C2224; --focus: #8EA6FF;
    --t0: #8EA6FF; --t1: #7590F0; --t2: #5E7AD8; --t3: #4B66BE; --t4: #3B53A0; --t5: #2F4383; --t6: #263668;
    --on0: #12161A; --on1: #12161A; --on2: #FFFFFF; --on3: #FFFFFF; --on4: #FFFFFF; --on5: #E4E8EB; --on6: #E4E8EB;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #12161A; --panel: #192026; --ink: #E4E8EB; --muted: #97A2AA; --rule: #2B343B; --faint: #20282F;
  --accent: #8EA6FF; --accent-soft: #232C4A; --good: #6CC592; --good-soft: #1D3226; --warn: #E3B061; --warn-soft: #3A2E1B;
  --crit: #EC8A8A; --crit-soft: #3C2224; --focus: #8EA6FF;
  --t0: #8EA6FF; --t1: #7590F0; --t2: #5E7AD8; --t3: #4B66BE; --t4: #3B53A0; --t5: #2F4383; --t6: #263668;
  --on0: #12161A; --on1: #12161A; --on2: #FFFFFF; --on3: #FFFFFF; --on4: #FFFFFF; --on5: #E4E8EB; --on6: #E4E8EB;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink); font: 15px/1.5 "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; display: grid; gap: 28px; }
.num, td.n, .fig { font-family: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace; font-variant-numeric: tabular-nums; }
h1 { font-size: 30px; line-height: 1.15; margin: 0; font-weight: 700; letter-spacing: -0.01em; text-wrap: balance; }
h2 { font-size: 13px; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.sub { color: var(--muted); margin: 6px 0 0; max-width: 68ch; }
header { display: grid; gap: 18px; }
.statement { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border-top: 2px solid var(--ink); border-bottom: 1px solid var(--rule); }
.statement div { padding: 12px 14px 12px 0; }
.statement dt { font-size: 12px; color: var(--muted); }
.statement dd { margin: 2px 0 0; font-size: 22px; font-weight: 500; }
.statement .good { color: var(--good); }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 18px; display: grid; gap: 14px; min-width: 0; }
.grid2 { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 860px) { .grid2 { grid-template-columns: minmax(0, 1fr); } }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.controls label { font-size: 13px; color: var(--muted); display: flex; gap: 8px; align-items: center; }
select { font: inherit; font-size: 14px; color: var(--ink); background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 5px 8px; }
select:focus-visible, button:focus-visible, tr:focus-visible, .tile:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.treemap { position: relative; width: 100%; aspect-ratio: 16 / 7; max-width: 100%; background: var(--faint); border-radius: 4px; overflow: hidden; }
.tile { position: absolute; border: 1px solid var(--panel); padding: 6px 8px; overflow: hidden; color: #fff; cursor: pointer; display: flex; flex-direction: column; justify-content: space-between; }
.tile .lab { font-size: 12.5px; font-weight: 600; line-height: 1.2; overflow-wrap: anywhere; }
.tile .val { font-size: 12px; opacity: 0.9; }
.tile.small .lab, .tile.small .val { display: none; }
.tile.dim { opacity: 0.35; }
.tablewrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; font-weight: 600; font-size: 12px; color: var(--muted); border-bottom: 1px solid var(--rule); padding: 6px 8px; white-space: nowrap; }
td { padding: 7px 8px; border-bottom: 1px solid var(--faint); vertical-align: top; }
td.n, th.n { text-align: right; white-space: nowrap; }
tr.grp { cursor: pointer; }
tr.grp:hover td, tr.sel td { background: var(--accent-soft); }
.bar { height: 6px; background: var(--faint); border-radius: 3px; margin-top: 5px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--accent); }
.cov { display: grid; gap: 10px; }
.cov .row { display: grid; grid-template-columns: 96px minmax(0, 1fr) 44px; gap: 10px; align-items: center; font-size: 13.5px; }
.stack { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--warn-soft); }
.stack .d { background: var(--good); } .stack .i { background: var(--accent); } .stack .c { background: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; background: var(--c); }
.acts { display: grid; gap: 8px; }
.act { display: grid; grid-template-columns: 10px minmax(0, 1fr) auto; gap: 10px; align-items: baseline; font-size: 14px; }
.act .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--c); align-self: center; }
.chip { display: inline-block; font-size: 12px; padding: 1px 7px; border-radius: 10px; background: var(--faint); color: var(--ink); margin: 0 4px 4px 0; white-space: nowrap; }
.chip.untagged { background: var(--warn-soft); color: var(--warn); }
.chip.declared { box-shadow: inset 0 0 0 1px var(--good); }
.pill { font-size: 12px; padding: 1px 8px; border-radius: 10px; white-space: nowrap; }
.pill.policy { background: var(--crit-soft); color: var(--crit); }
.pill.hold { background: var(--crit-soft); color: var(--crit); }
.pill.save { background: var(--good-soft); color: var(--good); }
.pill.review { background: var(--warn-soft); color: var(--warn); }
.pill.other { background: var(--accent-soft); color: var(--accent); }
tr.conv { cursor: pointer; }
tr.conv:hover td { background: var(--faint); }
tr.detail td { background: var(--faint); font-size: 13px; }
.trace { display: grid; grid-template-columns: 110px 90px minmax(0, 1fr); gap: 4px 10px; }
.trace .o { color: var(--muted); }
.trace .t { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; overflow-wrap: anywhere; }
.filter { font-size: 13px; color: var(--muted); display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button { font: inherit; font-size: 13px; background: none; border: 1px solid var(--rule); color: var(--ink); border-radius: 4px; padding: 3px 9px; cursor: pointer; }
footer { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 14px; }
@media (prefers-reduced-motion: no-preference) { .tile { transition: opacity 0.15s; } }
</style>

<div class="wrap">
  <header>
    <div>
      <h1>LLM spend, allocated by tag</h1>
      <p class="sub" id="lede"></p>
    </div>
    <dl class="statement" id="statement"></dl>
  </header>

  <section class="panel" aria-labelledby="h-alloc">
    <div class="controls">
      <h2 id="h-alloc" style="margin-right:auto">Allocation</h2>
      <label for="by1">Group by <select id="by1"></select></label>
      <label for="by2">then <select id="by2"></select></label>
    </div>
    <div class="treemap" id="treemap" role="img" aria-label="Spend by group"></div>
    <div class="tablewrap"><table id="groups"><thead><tr><th>Group</th><th class="n">Conversations</th><th class="n">Spend</th><th class="n">Share</th></tr></thead><tbody></tbody></table></div>
  </section>

  <div class="grid2">
    <section class="panel" aria-labelledby="h-cov">
      <h2 id="h-cov">Tag coverage, share of spend</h2>
      <div class="cov" id="coverage"></div>
      <div class="legend"><span style="--c: var(--good)">declared</span><span style="--c: var(--accent)">inferred</span><span style="--c: var(--muted)">code</span><span style="--c: var(--warn-soft)">untagged</span></div>
    </section>
    <section class="panel" aria-labelledby="h-act">
      <h2 id="h-act">Actions</h2>
      <div class="acts" id="actions"></div>
    </section>
  </div>

  <section class="panel" aria-labelledby="h-ledger">
    <div class="controls">
      <h2 id="h-ledger" style="margin-right:auto">Conversations</h2>
      <div class="filter" id="filter"></div>
    </div>
    <div class="tablewrap"><table id="ledger"><thead><tr><th>Conversation</th><th>Model</th><th class="n">Replies</th><th class="n">Spend</th><th>Tags</th><th>Actions</th></tr></thead><tbody></tbody></table></div>
  </section>

  <footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;
const rows = DATA.rows;
const total = rows.reduce((s, r) => s + r.usd, 0);
const fmt = (x) => x >= 1 ? "$" + x.toFixed(2) : x >= 0.01 ? "$" + x.toFixed(3) : "$" + x.toFixed(4);
const pct = (x) => (total ? (100 * x / total) : 0).toFixed(x / total < 0.1 ? 1 : 0) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const actHead = (a) => a.split(":")[0].split(" to ")[0];
let filter = null;

function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

// ---- statement
(function () {
  const saved = rows.reduce((s, r) => s + r.savings, 0);
  const tagged = rows.filter((r) => !Object.values(r.tags).includes("untagged")).reduce((s, r) => s + r.usd, 0);
  const biz = rows.filter((r) => r.business === true).reduce((s, r) => s + r.usd, 0);
  const items = [["Spend", fmt(total)], ["Conversations", rows.length], ["Fully tagged", pct(tagged)], ["Business use", pct(biz)], ["Savings found", fmt(saved) + " (" + pct(saved) + ")", "good"]];
  document.getElementById("statement").innerHTML = items.map(([k, v, c]) => `<div><dt>${k}</dt><dd class="num ${c || ""}">${v}</dd></div>`).join("");
  const models = new Set(rows.map((r) => r.model)).size;
  document.getElementById("lede").textContent = `${rows.length} conversations across ${models} models, tagged by ${DATA.backend || "a decision model"}${DATA.source ? " from " + DATA.source : ""}. Pick the tags to group by; click a group to see its conversations.`;
})();

// ---- group-by
const by1 = document.getElementById("by1"), by2 = document.getElementById("by2");
by1.innerHTML = DATA.keys.map((k) => `<option value="${k}">${k}</option>`).join("");
by2.innerHTML = `<option value="">nothing</option>` + DATA.keys.map((k) => `<option value="${k}">${k}</option>`).join("");
by1.value = load("finops.by1") || "task";
by2.value = load("finops.by2") ?? "";
by1.onchange = by2.onchange = () => { store("finops.by1", by1.value); store("finops.by2", by2.value); filter = null; draw(); };

function keyOf(r) { return by2.value && by2.value !== by1.value ? r.tags[by1.value] + " / " + r.tags[by2.value] : r.tags[by1.value]; }

function groups() {
  const g = new Map();
  for (const r of rows) { const k = keyOf(r); const x = g.get(k) || { key: k, n: 0, usd: 0 }; x.n++; x.usd += r.usd; g.set(k, x); }
  return [...g.values()].sort((a, b) => b.usd - a.usd);
}

// squarified treemap over the unit rectangle
function squarify(items, x, y, w, h, out) {
  if (!items.length) return;
  const sum = items.reduce((s, i) => s + i.v, 0);
  if (items.length === 1) { out.push({ ...items[0], x, y, w, h }); return; }
  const short = Math.min(w, h), area = w * h;
  let row = [], best = Infinity, i = 0;
  const worst = (r) => { const s = r.reduce((a, b) => a + b.a, 0); const mx = Math.max(...r.map((q) => q.a)), mn = Math.min(...r.map((q) => q.a)); return Math.max((short * short * mx) / (s * s), (s * s) / (short * short * mn)); };
  const scaled = items.map((it) => ({ ...it, a: (it.v / sum) * area }));
  while (i < scaled.length) { const trial = [...row, scaled[i]]; const w2 = worst(trial); if (w2 <= best) { row = trial; best = w2; i++; } else break; }
  const rs = row.reduce((s, q) => s + q.a, 0);
  if (w >= h) { const cw = rs / h; let cy = y; for (const q of row) { const qh = q.a / cw; out.push({ ...q, x, y: cy, w: cw, h: qh }); cy += qh; } squarify(scaled.slice(i).map(({ a, ...q }) => q), x + cw, y, w - cw, h, out); }
  else { const ch = rs / w; let cx = x; for (const q of row) { const qw = q.a / ch; out.push({ ...q, x: cx, y, w: qw, h: ch }); cx += qw; } squarify(scaled.slice(i).map(({ a, ...q }) => q), x, y + ch, w, h - ch, out); }
}

function draw() {
  const gs = groups();
  // treemap: aspect 16:7, computed in those units then set as percentages
  const tm = document.getElementById("treemap");
  const cells = []; squarify(gs.filter((g) => g.usd > 0).map((g, i) => ({ ...g, v: g.usd, rank: i })), 0, 0, 16, 7, cells);
  tm.innerHTML = cells.map((c) => {
    const small = c.w * c.h < 0.9 || c.w < 1.4;
    const tone = Math.min(c.rank, 6);
    return `<div class="tile ${small ? "small" : ""} ${filter && filter !== c.key ? "dim" : ""}" tabindex="0" role="button" data-k="${esc(c.key)}"
      title="${esc(c.key)}: ${fmt(c.usd)} (${pct(c.usd)})" style="left:${(c.x / 16) * 100}%;top:${(c.y / 7) * 100}%;width:${(c.w / 16) * 100}%;height:${(c.h / 7) * 100}%;background:var(--t${tone});color:var(--on${tone})">
      <span class="lab">${esc(c.key)}</span><span class="val fig">${fmt(c.usd)} (${pct(c.usd)})</span></div>`;
  }).join("");
  tm.querySelectorAll(".tile").forEach((t) => { t.onclick = () => pick(t.dataset.k); t.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(t.dataset.k); } }; });
  const max = gs[0] ? gs[0].usd : 1;
  document.querySelector("#groups tbody").innerHTML = gs.map((g) => `<tr class="grp ${filter === g.key ? "sel" : ""}" tabindex="0" data-k="${esc(g.key)}">
    <td>${esc(g.key)}${DATA.apps[g.key] ? `<div class="sub" style="margin:0;font-size:12px">${esc(DATA.apps[g.key])}…</div>` : ""}<div class="bar"><i style="width:${(100 * g.usd) / max}%"></i></div></td>
    <td class="n">${g.n}</td><td class="n">${fmt(g.usd)}</td><td class="n">${pct(g.usd)}</td></tr>`).join("");
  document.querySelectorAll("#groups tr.grp").forEach((t) => { t.onclick = () => pick(t.dataset.k); t.onkeydown = (e) => { if (e.key === "Enter") pick(t.dataset.k); }; });
  ledger();
}

function pick(k) { filter = filter === k ? null : k; draw(); document.getElementById("h-ledger").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }); }

// ---- coverage
(function () {
  document.getElementById("coverage").innerHTML = DATA.keys.map((k) => {
    const part = (f) => rows.filter(f).reduce((s, r) => s + r.usd, 0) / (total || 1);
    const d = part((r) => r.tags[k] !== "untagged" && r.source[k] === "declared");
    const i = part((r) => r.tags[k] !== "untagged" && r.source[k] === "inferred");
    const c = part((r) => r.tags[k] !== "untagged" && r.source[k] === "code");
    const cov = d + i + c;
    return `<div class="row"><span>${k}</span><div class="stack" title="declared ${(100 * d).toFixed(0)}%, inferred ${(100 * i).toFixed(0)}%, code ${(100 * c).toFixed(0)}%, untagged ${(100 * (1 - cov)).toFixed(0)}%">
      <span class="d" style="width:${100 * d}%"></span><span class="i" style="width:${100 * i}%"></span><span class="c" style="width:${100 * c}%"></span></div><span class="num" style="text-align:right">${(100 * cov).toFixed(0)}%</span></div>`;
  }).join("");
})();

// ---- actions
const ACT = { "policy": ["var(--crit)", "Spend with no business use"], "hold": ["var(--crit)", "Sensitive data: a person decides first"], "downgrade": ["var(--good)", "A small model would do"],
  "dev/test on a premium model": ["var(--good)", "Testing on a premium model"], "cache or template": ["var(--accent)", "A common request: cache or template it"],
  "trim context": ["var(--accent)", "Long conversation resending its history"], "review": ["var(--warn)", "Could not be tagged: a person reviews"] };
(function () {
  const m = new Map();
  for (const r of rows) r.actions.forEach((a, j) => { const h = actHead(a); const x = m.get(h) || { n: 0, save: 0 }; x.n++; if (j === 0) x.save += r.savings; m.set(h, x); });
  document.getElementById("actions").innerHTML = [...m.entries()].sort((a, b) => b[1].save - a[1].save || b[1].n - a[1].n).map(([h, x]) => {
    const [c, what] = ACT[h] || ["var(--muted)", h];
    return `<div class="act" style="--c:${c}"><span class="dot"></span><span><b>${esc(h)}</b> <span style="color:var(--muted)">${what}</span></span><span class="num">${x.n}${x.save ? ", " + fmt(x.save) : ""}</span></div>`;
  }).join("") || `<p class="sub">No actions.</p>`;
})();

// ---- ledger
function pillClass(h) { return h === "policy" || h === "hold" ? h : h === "review" ? "review" : h === "downgrade" || h.startsWith("dev/test") ? "save" : "other"; }
function ledger() {
  const list = rows.filter((r) => !filter || keyOf(r) === filter).sort((a, b) => b.usd - a.usd);
  document.getElementById("filter").innerHTML = filter ? `<span>Showing <b>${esc(filter)}</b>: ${list.length} conversations</span><button id="clear">Show all</button>` : `<span>All ${list.length}, most expensive first</span>`;
  const clear = document.getElementById("clear"); if (clear) clear.onclick = () => { filter = null; draw(); };
  document.querySelector("#ledger tbody").innerHTML = list.slice(0, 200).map((r, i) => `<tr class="conv" tabindex="0" data-i="${rows.indexOf(r)}">
    <td class="fig">${esc(r.id)}</td><td>${esc(r.model)}</td><td class="n">${r.replies}</td><td class="n">${fmt(r.usd)}</td>
    <td>${DATA.keys.filter((k) => k !== "app" || r.tags.app !== "adhoc").map((k) => `<span class="chip ${r.tags[k] === "untagged" ? "untagged" : ""} ${r.source[k] === "declared" ? "declared" : ""}" title="${k} (${r.source[k]})">${esc(r.tags[k])}</span>`).join("")}</td>
    <td>${r.actions.map((a) => `<span class="pill ${pillClass(actHead(a))}" title="${esc(a)}">${esc(actHead(a))}</span>`).join(" ")}</td></tr>`).join("");
  document.querySelectorAll("#ledger tr.conv").forEach((t) => { const open = () => toggle(t); t.onclick = open; t.onkeydown = (e) => { if (e.key === "Enter") open(); }; });
}
function toggle(tr) {
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("detail")) { next.remove(); return; }
  const r = rows[+tr.dataset.i];
  const gates = Object.entries(r.gates).map(([k, g]) => `<span>${esc(k)}</span><span class="o">${esc(g.outcome)}</span><span class="t">${esc(g.trace)}</span>`).join("");
  const d = document.createElement("tr"); d.className = "detail";
  d.innerHTML = `<td colspan="6"><div style="display:grid;gap:10px">
    ${r.text ? `<div>${esc(r.text)}</div>` : ""}
    <div class="fig">${r.in.toLocaleString()} tokens in, ${r.out.toLocaleString()} out. Business use: ${r.business === null ? "unsure" : r.business ? "yes" : "no"}.${r.savings ? " Saves " + fmt(r.savings) + "." : ""}</div>
    <div>${r.actions.map((a) => esc(a)).join("<br>") || "No action."}</div>
    <div class="trace">${gates}</div></div></td>`;
  tr.after(d);
}

document.getElementById("foot").innerHTML = `Tagged ${DATA.date}${DATA.backend ? " by " + esc(DATA.backend) : ""}. Spend is tokens times list prices; input tokens are estimated at four characters a token. ` +
  `A tag below the circuit's confidence floor is <i>untagged</i>, never guessed. Conversation text is not included. Built with finops-circuit on decision circuits.`;
draw();
</script>
"""
