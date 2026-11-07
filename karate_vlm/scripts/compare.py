"""Step 5: one page comparing the models on the same match, window by window.

Top: per model, a timeline per player and the counts; how often the models
agree. Below: every window in order, its image and the three models' labels
side by side (agreement highlighted). Click a label to see that model's
reasoning steps.

usage: python compare.py <name> [model keys ...]   -> compare_<name>.html
"""
import collections
import html
import json
import sys
from pathlib import Path

from report import MATCH, SHORT, events, img_src, kind

HERE = Path(__file__).parent
TITLES = {"qwen3vl": "Qwen3-VL-4B", "qwen35": "Qwen3.5-4B", "gemma3": "Gemma 3 4B"}


def main() -> int:
    name = sys.argv[1]
    keys = sys.argv[2:] or list(TITLES)
    R = {k: [r for r in json.loads((HERE / f"results_{k}.json").read_text(encoding="utf-8"))["results"]
             if r["video"] == name] for k in keys}
    n = len(R[keys[0]])
    dur = max(r["end_s"] for r in R[keys[0]])
    esc = html.escape

    def lane(k, p):
        bars = "".join(
            f'<a class="ev {kind(v["label"])}" href="#w{i_of(v["start_s"])}" title="{esc(SHORT[v["label"]])} '
            f'{v["start_s"]:.1f}-{v["end_s"]:.1f}s" style="left:{100 * v["start_s"] / dur:.2f}%;'
            f'width:{max(0.6, 100 * (v["end_s"] - v["start_s"]) / dur):.2f}%"></a>' for v in events(R[k], p))
        return f'<div class="lane">{bars}</div>'

    starts = [r["start_s"] for r in R[keys[0]]]

    def i_of(s):
        return min(range(n), key=lambda i: abs(starts[i] - s))

    timelines = "".join(
        f'<div class="tlrow"><span class="m">{esc(TITLES[k])}</span>'
        f'<div class="pl"><span class="who p1">P1</span>{lane(k, 1)}</div>'
        f'<div class="pl"><span class="who p2">P2</span>{lane(k, 2)}</div></div>' for k in keys)

    # stats
    stat_rows = ""
    for k in keys:
        rs = R[k]
        cells = []
        for p in ("player1", "player2"):
            c = collections.Counter(r[p]["label"] for r in rs)
            cells.append(n - c["MOVEMENT"])
        inc = sum(1 for r in rs for p in ("player1", "player2") if r[p].get("consistent") is False)
        secs = sum(r["seconds"] for r in rs) / n
        stat_rows += (f'<tr><th>{esc(TITLES[k])}</th><td class="n">{cells[0]}</td><td class="n">{cells[1]}</td>'
                      f'<td class="n">{inc}</td><td class="n">{secs:.1f}s</td></tr>')
    same_label = same_atk = same_attack_label = 0
    for i in range(n):
        for p in ("player1", "player2"):
            labs = [R[k][i][p]["label"] for k in keys]
            same_label += len(set(labs)) == 1
            same_atk += len({l != "MOVEMENT" for l in labs}) == 1
            same_attack_label += len(set(labs)) == 1 and labs[0] != "MOVEMENT"
    any_attack = sum(1 for i in range(n) for p in ("player1", "player2")
                     if any(R[k][i][p]["label"] != "MOVEMENT" for k in keys))

    count_rows = ""
    labels = sorted({r[p]["label"] for k in keys for r in R[k] for p in ("player1", "player2")} - {"MOVEMENT", None},
                    key=lambda l: (kind(l), l))
    for l in labels:
        tds = "".join(f'<td class="n">{sum(1 for v in events(R[k], 1) if v["label"] == l)} / '
                      f'{sum(1 for v in events(R[k], 2) if v["label"] == l)}</td>' for k in keys)
        count_rows += f'<tr><td><span class="dot {kind(l)}"></span>{esc(SHORT[l])}</td>{tds}</tr>'

    def cell(k, a):
        lab = a.get("label") or "ERROR"
        bad = '<span class="warn" title="الليبل مش ماشي مع الخطوات اللي الموديل قالها">!</span>' if a.get("consistent") is False else ""
        steps = " · ".join(f"{x}: {esc(str(a.get(x)))}" for x in ("attack", "side", "kick_type", "height"))
        return (f'<details class="c {kind(lab)}"><summary><span class="m">{esc(TITLES[k])}</span>'
                f'<b>{esc(SHORT.get(lab, lab))}</b> <span class="conf">{a.get("confidence")}</span>{bad}</summary>'
                f'<div class="steps">{steps}</div><p>{esc(a.get("what_happened") or "")}</p></details>')

    cards = ""
    for i in range(n):
        r0 = R[keys[0]][i]
        rows, atk = "", False
        for p, pn in (("player1", 1), ("player2", 2)):
            labs = [R[k][i][p]["label"] for k in keys]
            atk |= any(l != "MOVEMENT" for l in labs)
            agree = "agree" if len(set(labs)) == 1 else ("part" if len(set(labs)) < len(labs) else "split")
            rows += (f'<div class="prow {agree}"><span class="who p{pn}">Player {pn}</span>'
                     f'<div class="cells">{"".join(cell(k, R[k][i][p]) for k in keys)}</div></div>')
        cards += (f'<article id="w{i}" class="win {"atk" if atk else "quiet"}"><div class="t">{r0["start_s"]:.1f}–{r0["end_s"]:.1f}s</div>'
                  f'<img loading="lazy" src="{img_src(r0["image"])}" alt="frames {r0["start_s"]:.1f}s"><div class="rows">{rows}</div></article>')

    head = "".join(f"<th>{esc(TITLES[k])}<br><small>P1 / P2</small></th>" for k in keys)
    page = f'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>مقارنة الـ VLM · {esc(MATCH.get(name, name))}</title>
<style>
:root{{--bg:#f4f6f9;--paper:#fff;--ink:#17202b;--muted:#5d6878;--line:#dbe0e8;--p1:#2563c9;--p2:#d9771f;
--kick:#7a3fc4;--punch:#0f7b6c;--feint:#b08600;--move:#9aa3b2;--warn:#b23a32;--ok:#e3f3e8;--mid:#fbf3dc;--no:#f9e5e3}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Segoe UI",Tahoma,sans-serif}}
.page{{max-width:1240px;margin:0 auto;padding:24px 16px 60px;display:grid;gap:20px}}
h1{{margin:0;font-size:24px}} h2{{margin:0 0 10px;font-size:17px}} .sub{{color:var(--muted)}}
.box{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}}
.tlrow{{display:grid;grid-template-columns:120px 1fr;gap:4px 12px;align-items:center;margin-bottom:12px}}
.tlrow .m{{grid-row:span 2;font-weight:600}} .pl{{display:grid;grid-template-columns:28px 1fr;gap:6px;align-items:center}}
.lane{{position:relative;height:20px;background:#eef1f5;border-radius:4px;direction:ltr}}
.ev{{position:absolute;top:2px;bottom:2px;border-radius:2px;background:var(--c)}}
.kick{{--c:var(--kick)}} .punch{{--c:var(--punch)}} .feint{{--c:var(--feint)}} .move{{--c:var(--move)}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-inline-end:6px;background:var(--c)}}
.axis{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-inline-start:166px;direction:ltr}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:6px}}
table{{border-collapse:collapse;font-size:14px}} td,th{{padding:5px 12px;border-bottom:1px solid var(--line);text-align:right}}
.n{{font-variant-numeric:tabular-nums;text-align:center;direction:ltr}}
.who{{font-weight:700}} .who.p1{{color:var(--p1)}} .who.p2{{color:var(--p2)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}
.kpis{{display:flex;gap:24px;flex-wrap:wrap}} .kpis div{{display:grid}} .kpis b{{font-size:22px;font-variant-numeric:tabular-nums}} .kpis span{{color:var(--muted);font-size:13px}}
.bar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
button{{font:inherit;border:1px solid var(--line);background:var(--paper);border-radius:6px;padding:4px 12px;cursor:pointer}}
button[aria-pressed=true]{{background:var(--ink);color:var(--paper)}}
.win{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px;display:grid;
grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:14px;align-items:start}}
.win img{{width:100%;border-radius:6px}} .t{{grid-column:1/-1;font-weight:700;direction:ltr;text-align:right}}
.rows{{display:grid;gap:12px}} .prow{{border-radius:8px;padding:8px;display:grid;gap:6px}}
.prow.agree{{background:var(--ok)}} .prow.part{{background:var(--mid)}} .prow.split{{background:var(--no)}}
.cells{{display:grid;gap:6px}}
.c{{background:var(--paper);border-inline-start:4px solid var(--c);border-radius:4px;padding:4px 8px}}
.c summary{{cursor:pointer;display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}} .c .m{{color:var(--muted);font-size:12.5px;min-width:92px}}
.conf{{color:var(--muted);font-size:12px}} .warn{{color:#fff;background:var(--warn);border-radius:50%;width:16px;height:16px;display:inline-grid;place-items:center;font-size:11px;font-weight:700}}
.steps{{font-size:12px;color:var(--muted);direction:ltr;text-align:right}} .c p{{margin:2px 0 4px;font-size:13px;direction:ltr;text-align:left}}
body.only-atk .quiet{{display:none}}
@media (max-width:820px){{.win{{grid-template-columns:1fr}} .tlrow{{grid-template-columns:1fr}} .tlrow .m{{grid-row:auto}} .axis{{margin-inline-start:34px}}}}
</style></head><body class="only-atk"><div class="page">
<header><h1>مقارنة 3 موديلات VLM · {esc(MATCH.get(name, name))}</h1>
<div class="sub">الفيديو الأصلي: {esc(MATCH.get(name, name))} · {n} شباك من أول الماتش لآخره (1.2 ثانية، كل 0.8 ثانية) · نفس البرومبت ونفس السكيما · من غير تدريب · مفيش تسمية صح، فالحكم بعينك</div></header>
<section class="box"><h2>مدى اتفاق الموديلات</h2><div class="kpis">
<div><b>{same_atk}/{2 * n}</b><span>اتفقوا هل فيه هجوم ولا لأ</span></div>
<div><b>{same_label}/{2 * n}</b><span>اتفقوا على نفس الليبل بالظبط (أغلبها MOVEMENT)</span></div>
<div><b>{same_attack_label}/{any_attack}</b><span>اتفقوا على نفس الهجوم، من الحالات اللي ولو موديل واحد شاف فيها هجوم</span></div>
</div></section>
<section class="box"><h2>الـ timeline</h2>{timelines}
<div class="axis"><span>0s</span><span>{dur / 2:.0f}s</span><span>{dur:.0f}s</span></div>
<div class="legend"><span><i class="dot kick"></i>كيك</span><span><i class="dot punch"></i>لكمة</span><span><i class="dot feint"></i>Feint</span><span>دوس على أي حدث يوديك لصورته</span></div></section>
<div class="grid2">
<section class="box"><h2>لكل موديل</h2><table><tr><th></th><th>شبابيك فيها هجوم<br><small class="who p1">P1</small></th><th><br><small class="who p2">P2</small></th><th>ليبل مش ماشي<br>مع خطواته</th><th>وقت الشباك</th></tr>{stat_rows}</table></section>
<section class="box"><h2>عدد الحركات (أحداث بعد الدمج)</h2><table><tr><th>الحركة</th>{head}</tr>{count_rows}</table></section>
</div>
<div class="bar"><span>اعرض:</span><button id="f-atk" aria-pressed="true">الشبابيك اللي أي موديل شاف فيها هجوم</button>
<button id="f-all" aria-pressed="false">كل الشبابيك بالترتيب</button>
<span class="sub">الخلفية: <span style="background:var(--ok);padding:0 6px">التلاتة متفقين</span> <span style="background:var(--mid);padding:0 6px">اتنين متفقين</span> <span style="background:var(--no);padding:0 6px">كلهم مختلفين</span> · دوس على أي إجابة تشوف خطواتها</span></div>
{cards}
</div><script>
const b=document.body,a=document.getElementById("f-atk"),l=document.getElementById("f-all");
a.onclick=()=>{{b.classList.add("only-atk");a.ariaPressed="true";l.ariaPressed="false"}};
l.onclick=()=>{{b.classList.remove("only-atk");l.ariaPressed="true";a.ariaPressed="false"}};
document.querySelectorAll(".ev").forEach(e=>e.addEventListener("click",()=>l.click()));
</script></body></html>'''
    out = HERE / f"compare_{name}.html"
    out.write_text(page, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
