# ABOUTME: Builds the self-contained HTML comparison page for an apply result: reference, input
# ABOUTME: and candidate columns, sentence-aligned diffs, and the checks pinned to paragraphs.
# ruff: noqa: E501  (inline CSS and JS)
from __future__ import annotations

import difflib
import html
import json
import re
from dataclasses import dataclass, field
from typing import Any

from deslop.nlp import WORD_RE, analyze

MATCH_RATIO = 0.45  # content-word overlap below which a sentence counts as new, not changed
SAME_RATIO = 0.98


@dataclass
class AlignedSentence:
    text: str
    status: str  # same | changed | new | moved
    source: int | None = None  # index of the matched input sentence
    words: list[tuple[str, str]] = field(default_factory=list)  # (op, text) word-level diff
    paragraph: int = 0


def _content(sentence: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(sentence)]


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _content(a), _content(b), autojunk=False).ratio()


def word_diff(before: str, after: str) -> list[tuple[str, str]]:
    """Word-level diff as (op, text) runs: equal, insert (in after only), delete (before)."""
    ta = re.findall(r"\S+|\s+", before)
    tb = re.findall(r"\S+|\s+", after)
    out: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(("equal", "".join(tb[j1:j2])))
        elif op == "insert":
            out.append(("insert", "".join(tb[j1:j2])))
        elif op == "delete":
            out.append(("delete", "".join(ta[i1:i2])))
        else:
            out.append(("delete", "".join(ta[i1:i2])))
            out.append(("insert", "".join(tb[j1:j2])))
    return out


def sentences_of(text: str) -> list[tuple[str, int]]:
    """(sentence, paragraph index) for every prose sentence of `text`."""
    a = analyze(text)
    return [(s.text.strip(), p.index) for p in a.paragraphs for s in p.sentences]


def align_sentences(source: str, candidate: str) -> list[AlignedSentence]:
    """Each candidate sentence matched by content to its best unused input sentence: same,
    changed (with a word diff), moved (matched out of order) or new."""
    src = sentences_of(source)
    cand = sentences_of(candidate)
    used: set[int] = set()
    out: list[AlignedSentence] = []
    last_source = -1
    for text, para in cand:
        best, best_ratio = None, 0.0
        for i, (s, _) in enumerate(src):
            if i in used:
                continue
            r = _ratio(text, s)
            if r > best_ratio:
                best, best_ratio = i, r
        if best is None or best_ratio < MATCH_RATIO:
            out.append(AlignedSentence(text=text, status="new", paragraph=para))
            continue
        used.add(best)
        status = "same" if best_ratio >= SAME_RATIO and text == src[best][0] else "changed"
        if best < last_source:
            status = "moved"
        last_source = max(last_source, best)
        out.append(
            AlignedSentence(
                text=text,
                status=status,
                source=best,
                words=word_diff(src[best][0], text) if status != "same" else [],
                paragraph=para,
            )
        )
    return out


def _find_paragraph(needle: str, paragraphs: list[str]) -> int | None:
    key = " ".join(_content(needle)[:8])
    if not key:
        return None
    for i, p in enumerate(paragraphs):
        if key in " ".join(_content(p)):
            return i
    return None


def candidate_notes(cand: dict[str, Any], paragraphs: list[str]) -> list[dict[str, Any]]:
    """Every check result pinned to a paragraph of the candidate (-1 = whole document)."""
    notes: list[dict[str, Any]] = []
    if cand.get("lexical_detail") and not cand.get("lexical_ok", True):
        notes.append({"kind": "lock", "paragraph": -1, "text": f"lock: {cand['lexical_detail']}"})
    sem = cand.get("semantic") or {}
    for c in sem.get("checks", []):
        if not c.get("kept"):
            m = re.match(r"¶(\d+)\s", c["proposition"])
            para = int(m.group(1)) - 1 if m else -1
            notes.append(
                {"kind": "dropped", "paragraph": para, "text": f"dropped: {c['proposition']}"}
            )
    for a in sem.get("added", []):
        if a.get("kind") != "claim":
            continue
        found = _find_paragraph(a.get("sentence") or a.get("text", ""), paragraphs)
        notes.append(
            {
                "kind": "added",
                "paragraph": -1 if found is None else found,
                "text": f"added ({a.get('category') or 'claim'}): {a.get('text')}",
            }
        )
    for a in sem.get("cleared", []):
        found = _find_paragraph(a.get("sentence") or a.get("text", ""), paragraphs)
        notes.append(
            {
                "kind": "cleared",
                "paragraph": -1 if found is None else found,
                "text": f"cleared on review ({a.get('category')}): {a.get('text')} "
                f"— {a.get('note', '')}",
            }
        )
    coh = cand.get("coherence") or {}
    for d in coh.get("tense", []):
        notes.append(
            {
                "kind": "coherence",
                "paragraph": d["paragraph"] - 1,
                "text": f"tense {d['before']} -> {d['after']}: {d['example'][:90]}",
            }
        )
    for f in coh.get("antecedents", []):
        notes.append(
            {
                "kind": "coherence",
                "paragraph": f["paragraph"] - 1,
                "text": f"antecedent: {f['text'][:90]} ({f['word']})",
            }
        )
    conn = coh.get("connectives") or {}
    if conn and not conn.get("ok", True):
        notes.append(
            {"kind": "coherence", "paragraph": -1, "text": f"connectives: {conn.get('note')}"}
        )
    if coh and coh.get("paragraphs_before") != coh.get("paragraphs_after"):
        notes.append(
            {
                "kind": "coherence",
                "paragraph": -1,
                "text": f"paragraphs {coh.get('paragraphs_before')} -> {coh.get('paragraphs_after')}",
            }
        )
    for v in cand.get("verdicts", []):
        items = ", ".join(f"{k} {x}" for k, x in v.get("items", {}).items())
        notes.append(
            {
                "kind": "judge",
                "paragraph": -1,
                "text": f"judge vs {v.get('against')}: {items}; confidence "
                f"{v.get('confidence', 0):.2f}; {v.get('note', '')}",
            }
        )
    return notes


def page_data(result: dict[str, Any], reference: str | None) -> dict[str, Any]:
    """Everything the page needs, computed in Python: columns, per-candidate alignment,
    notes and header facts."""
    unit = result["units"][0] if result.get("units") else None
    source = unit["original"] if unit else result["text"]
    candidates = unit["candidates"] if unit else []
    chosen = unit.get("chosen") if unit else None
    if not result.get("report_all"):
        candidates = [candidates[chosen]] if chosen is not None else []
    src_paragraphs = [p.text for p in analyze(source).paragraphs]
    cands = []
    for c in candidates:
        if not c.get("text"):
            continue
        paragraphs = [p.text for p in analyze(c["text"]).paragraphs]
        aligned = align_sentences(source, c["text"])
        cands.append(
            {
                "attempt": c["attempt"],
                "model": c.get("model", ""),
                "accepted": c.get("accepted", False),
                "winner": unit is not None
                and chosen is not None
                and unit["candidates"][chosen] is c,
                "reason": c.get("reason", ""),
                "paragraphs": paragraphs,
                "sentences": [
                    {
                        "text": s.text,
                        "status": s.status,
                        "source": s.source,
                        "paragraph": s.paragraph,
                        "words": s.words,
                    }
                    for s in aligned
                ],
                "notes": candidate_notes(c, paragraphs),
                "judge": next(
                    (
                        v.get("items", {})
                        for v in c.get("verdicts", [])
                        if v.get("against") == "original"
                    ),
                    {},
                ),
                "confidence": next(
                    (
                        v.get("confidence")
                        for v in c.get("verdicts", [])
                        if v.get("against") == "original"
                    ),
                    None,
                ),
            }
        )
    before, after = result["report_before"], result["report_after"]
    return {
        "title": f"deslop {result.get('mode', 'apply')}: {len(cands)} candidate(s)",
        "reference": [p.text for p in analyze(reference).paragraphs] if reference else [],
        "source": src_paragraphs,
        "candidates": cands,
        "header": {
            "distance_before": before["distance"],
            "distance_after": after["distance"],
            "verdict_before": before["verdict"],
            "verdict_after": after["verdict"],
            "stopped": result.get("stopped", ""),
            "judge_items": result.get("judge_items", {}),
            "judge_gate": result.get("judge_gate", ""),
            "calls": result.get("cost", {}).get("calls", 0),
            "cached": result.get("cost", {}).get("cached", 0),
            "selected": (chosen + 1) if chosen is not None else None,
            "kept_original": bool(unit and unit.get("kept_original")),
        },
    }


CSS = """
:root{--bg:#f7f5f0;--fg:#23211c;--muted:#6c675c;--rule:#d9d4c7;--panel:#fffdf8;--ins:#dcecd8;--insfg:#1e4d22;--del:#f2d9d2;--delfg:#7a2e1e;--chg:#fff3c4;--mov:#e3e8f7;--new:#f6e1f5;--accent:#3c5a78}
@media (prefers-color-scheme: dark){:root{--bg:#1b1a17;--fg:#e8e3d8;--muted:#a39d90;--rule:#3a372f;--panel:#23221e;--ins:#1f3a22;--insfg:#b9e0b4;--del:#4a2219;--delfg:#f0b8a8;--chg:#4a3f14;--mov:#2a3350;--new:#43284a;--accent:#9cbce0}}
body{margin:0;padding:20px;background:var(--bg);color:var(--fg);font:16px/1.5 Georgia,serif}
header{max-width:1400px;margin:0 auto 18px}h1{font-size:1.4rem;margin:0 0 6px}
.facts{font:14px/1.5 system-ui,sans-serif;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 22px}
.controls{font:14px/1.5 system-ui,sans-serif;display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:10px 0}
main{max-width:1400px;margin:0 auto;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
@media (max-width:1000px){main{grid-template-columns:1fr}}
.col{min-width:0;background:var(--panel);border:1px solid var(--rule);padding:12px 14px}
.col h2{font:600 12px/1.4 system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.col p{margin:0 0 1em}
.s{border-radius:2px;padding:0 1px}.changed{background:var(--chg)}.moved{background:var(--mov);border-bottom:2px dotted var(--accent)}.new{background:var(--new)}
ins{background:var(--ins);color:var(--insfg);text-decoration:none}del{background:var(--del);color:var(--delfg)}
.diff{display:none}body.words .s.changed .plain,body.words .s.moved .plain{display:none}body.words .diff{display:inline}
.s.changed:hover .plain,.s.moved:hover .plain{display:none}.s.changed:hover .diff,.s.moved:hover .diff{display:inline}
body.clean .s{background:none;border:none}body.clean .diff{display:none!important}body.clean .plain{display:inline!important}
.notes{font:13px/1.45 system-ui,sans-serif;border-left:3px solid var(--rule);padding:4px 8px;margin:0 0 1em;color:var(--muted)}
.notes .lock{color:var(--delfg)}.notes .added{color:var(--delfg)}.notes .dropped{color:var(--delfg)}.notes .cleared{color:var(--accent)}.notes .coherence{color:var(--delfg)}.notes .judge{color:var(--fg)}
.legend span{padding:0 6px;border-radius:2px;margin-right:6px}
select{font:inherit}
"""

JS = """
const DATA = __DATA__;
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function renderParas(el, paras){ el.innerHTML = paras.map(p => '<p>' + esc(p) + '</p>').join(''); }
function renderCandidate(i){
  const c = DATA.candidates[i]; const el = document.getElementById('cand'); if(!c){ el.innerHTML = '<p>(no candidate)</p>'; return; }
  const byPara = {}; c.sentences.forEach(s => { (byPara[s.paragraph] = byPara[s.paragraph] || []).push(s); });
  const notesBy = {}; c.notes.forEach(n => { (notesBy[n.paragraph] = notesBy[n.paragraph] || []).push(n); });
  let out = '';
  const docNotes = notesBy[-1] || [];
  if (docNotes.length) out += '<div class="notes">' + docNotes.map(n => '<div class="' + n.kind + '">' + esc(n.text) + '</div>').join('') + '</div>';
  c.paragraphs.forEach((p, pi) => {
    const sents = byPara[pi] || [{text: p, status: 'same', words: []}];
    out += '<p>' + sents.map(s => {
      const plain = '<span class="plain">' + esc(s.text) + '</span>';
      const diff = s.words && s.words.length ? '<span class="diff">' + s.words.map(([op, t]) => op === 'equal' ? esc(t) : op === 'insert' ? '<ins>' + esc(t) + '</ins>' : '<del>' + esc(t) + '</del>').join('') + '</span>' : '';
      return '<span class="s ' + s.status + '" title="' + s.status + '">' + plain + diff + '</span>';
    }).join(' ') + '</p>';
    const pn = notesBy[pi] || [];
    if (pn.length) out += '<div class="notes">' + pn.map(n => '<div class="' + n.kind + '">' + esc(n.text) + '</div>').join('') + '</div>';
  });
  el.innerHTML = out;
  const j = c.judge || {}; const items = Object.keys(j).map(k => k + ' ' + j[k]).join(', ');
  document.getElementById('candfacts').textContent = 'candidate ' + c.attempt + ' (' + c.model + '): ' + (c.winner ? 'winner' : c.accepted ? 'accepted' : 'rejected') + (items ? '; judge: ' + items : '') + (c.confidence ? ' (' + c.confidence.toFixed(2) + ')' : '') + '; ' + c.reason;
}
document.addEventListener('DOMContentLoaded', () => {
  renderParas(document.getElementById('ref'), DATA.reference.length ? DATA.reference : ['(no reference given)']);
  renderParas(document.getElementById('src'), DATA.source);
  const sel = document.getElementById('select');
  DATA.candidates.forEach((c, i) => { const o = document.createElement('option'); o.value = i; o.textContent = 'candidate ' + c.attempt + (c.winner ? ' (winner)' : c.accepted ? ' (accepted)' : ' (rejected)'); sel.appendChild(o); });
  let start = DATA.candidates.findIndex(c => c.winner); if (start < 0) start = DATA.candidates.findIndex(c => c.accepted); if (start < 0) start = 0;
  sel.value = start; renderCandidate(start);
  sel.addEventListener('change', e => renderCandidate(parseInt(e.target.value, 10)));
  document.getElementById('words').addEventListener('change', e => document.body.classList.toggle('words', e.target.checked));
  document.getElementById('clean').addEventListener('change', e => document.body.classList.toggle('clean', e.target.checked));
});
"""


def build_page(result: dict[str, Any], *, reference: str | None = None) -> str:
    """The comparison page as one HTML document with everything inline: no external
    scripts, styles or fonts."""
    data = page_data(result, reference)
    h = data["header"]
    judge = ", ".join(f"{k} {v}" for k, v in h["judge_items"].items()) or "none"
    selected = (
        "original kept"
        if h["kept_original"]
        else (f"candidate {h['selected']}" if h["selected"] else "none")
    )
    facts = [
        f"distance {h['distance_before']:.0f} → {h['distance_after']:.0f}",
        f"verdict {h['verdict_before']} → {h['verdict_after']}",
        f"judge items: {judge}",
        f"calls {h['calls']} ({h['cached']} cached)",
        f"selected: {selected}",
        f"stopped: {h['stopped']}",
    ]
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(data['title'])}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f"<header>\n<h1>{html.escape(data['title'])}</h1>\n"
        f'<div class="facts">{"".join(f"<span>{html.escape(f)}</span>" for f in facts)}</div>\n'
        f'<div class="facts">{html.escape(h["judge_gate"])}</div>\n'
        '<div class="controls"><label>candidate <select id="select"></select></label>'
        '<label><input type="checkbox" id="words"> word diff always on (else on hover)</label>'
        '<label><input type="checkbox" id="clean"> read clean</label>'
        '<span class="legend"><span class="changed">changed</span><span class="moved">moved</span><span class="new">new</span></span></div>\n'
        '<div class="facts" id="candfacts"></div>\n</header>\n<main>\n'
        '<section class="col"><h2>Human reference</h2><div id="ref"></div></section>\n'
        '<section class="col"><h2>Input</h2><div id="src"></div></section>\n'
        '<section class="col"><h2>Candidate</h2><div id="cand"></div></section>\n'
        f"</main>\n<script>{JS.replace('__DATA__', payload)}</script>\n</body>\n</html>\n"
    )
