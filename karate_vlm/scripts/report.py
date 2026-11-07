"""Step 4: one HTML page per model + match to judge the answers by eye.

Top: a timeline per player (attacks only, neighbouring windows with the same
label merged into one event). Below: every window in order, its image and
both players' answers. Filters: attacks only / everything.

usage: python report.py <model key> <name>   -> report_<key>_<name>.html
"""
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

# the original video each window set was cut from
MATCH = {"match1": "Match 1", "match2": "Match 2", "match3": "Match 3 (1)", "balkas": "ماتش 1 مع بلقاس"}


def img_src(rel: str, width: int = 880) -> str:
    """The window image embedded in the page (data: URI), so the page shows
    its images wherever it is opened or sent, without the clips folder."""
    import base64
    import io

    from PIL import Image

    im = Image.open(HERE / rel).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=74, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

SHORT = {
    "FEINT": "Feint", "GYAKU_ZUKI": "Gyaku zuki", "KIZAMI_GYAKU_COMBINATION": "Kizami + gyaku",
    "KIZAMI_MAWASHI_GERI_CHUDAN": "Kizami mawashi chudan", "KIZAMI_MAWASHI_GERI_JODAN": "Kizami mawashi jodan",
    "KIZAMI_URA_MAWASHI_GERI_JODAN": "Kizami ura mawashi jodan", "KIZAMI_ZUKI": "Kizami zuki",
    "MAWASHI_GERI_CHUDAN": "Mawashi chudan", "MAWASHI_GERI_JODAN": "Mawashi jodan", "MOVEMENT": "Movement",
    "URA_MAWASHI_GERI_JODAN": "Ura mawashi jodan", "YOKO_GERI_CHUDAN": "Yoko geri chudan",
}
KIND = {"FEINT": "feint", "MOVEMENT": "move"}  # everything else: punch or kick


def kind(label):
    if label in KIND:
        return KIND[label]
    return "punch" if "ZUKI" in (label or "") or "COMBINATION" in (label or "") else "kick"


def events(results, k):
    """Merge neighbouring windows with the same attack label into one event."""
    out = []
    for r in results:
        a = r[f"player{k}"]
        lab = a.get("label")
        if not lab or lab == "MOVEMENT":
            continue
        if out and out[-1]["label"] == lab and r["start_s"] <= out[-1]["end_s"]:
            out[-1]["end_s"] = r["end_s"]
            out[-1]["conf"] = max(out[-1]["conf"], a.get("confidence") or 0)
        else:
            out.append({"label": lab, "start_s": r["start_s"], "end_s": r["end_s"], "conf": a.get("confidence") or 0})
    return out


def main() -> int:
    key, name = sys.argv[1], sys.argv[2]
    data = json.loads((HERE / f"results_{key}.json").read_text(encoding="utf-8"))
    res = [r for r in data["results"] if r["video"] == name]
    dur = max(r["end_s"] for r in res)
    e = esc = html.escape

    def tl(k):
        bars = "".join(
            f'<a class="ev {kind(v["label"])}" href="#w{v["start_s"]}" title="{esc(SHORT[v["label"]])} '
            f'{v["start_s"]:.1f}-{v["end_s"]:.1f}s" style="left:{100 * v["start_s"] / dur:.2f}%;'
            f'width:{max(0.6, 100 * (v["end_s"] - v["start_s"]) / dur):.2f}%"></a>' for v in events(res, k))
        return f'<div class="track"><span class="who p{k}">Player {k}</span><div class="lane">{bars}</div></div>'

    counts = {}
    for k in (1, 2):
        for v in events(res, k):
            counts.setdefault(v["label"], [0, 0])[k - 1] += 1
    count_rows = "".join(f'<tr><td><span class="dot {kind(l)}"></span>{esc(SHORT[l])}</td>'
                         f'<td class="n">{c[0]}</td><td class="n">{c[1]}</td></tr>'
                         for l, c in sorted(counts.items(), key=lambda x: -sum(x[1])))

    def ans(a, k):
        lab = a.get("label") or "ERROR"
        flag = "" if a.get("consistent") in (True, None) else '<span class="warn">الخطوات مش ماشية مع الليبل</span>'
        steps = " · ".join(f"{x}: {esc(str(a.get(x)))}" for x in ("attack", "side", "kick_type", "height"))
        return (f'<div class="ans p{k} {kind(lab)}"><div class="lab"><span class="who p{k}">P{k}</span>'
                f'<b>{esc(SHORT.get(lab, lab))}</b><span class="conf">{a.get("confidence")}</span>{flag}</div>'
                f'<div class="steps">{steps}</div><p>{esc(a.get("what_happened") or "")}</p></div>')

    cards = "".join(
        f'<article id="w{r["start_s"]}" class="win {"atk" if any(r[f"player{k}"].get("label") not in (None, "MOVEMENT") for k in (1, 2)) else "quiet"}">'
        f'<div class="t">{r["start_s"]:.1f}–{r["end_s"]:.1f}s</div>'
        f'<img loading="lazy" src="{img_src(r["image"])}" alt="frames {r["start_s"]:.1f}s">'
        f'<div class="answers">{ans(r["player1"], 1)}{ans(r["player2"], 2)}</div></article>' for r in res)

    page = f'''<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(data["model"])} · {esc(MATCH.get(name, name))}</title>
<style>
:root{{--bg:#f4f6f9;--paper:#fff;--ink:#17202b;--muted:#5d6878;--line:#dbe0e8;--p1:#2563c9;--p2:#d9771f;
--kick:#7a3fc4;--punch:#0f7b6c;--feint:#b08600;--move:#9aa3b2;--warn:#b23a32}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Segoe UI",Tahoma,sans-serif}}
.page{{max-width:1180px;margin:0 auto;padding:24px 16px 60px;display:grid;gap:22px}}
h1{{margin:0;font-size:24px}} .sub{{color:var(--muted)}}
.box{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:16px}}
.track{{display:grid;grid-template-columns:80px 1fr;gap:10px;align-items:center;margin:8px 0}}
.lane{{position:relative;height:26px;background:#eef1f5;border-radius:5px;direction:ltr}}
.ev{{position:absolute;top:3px;bottom:3px;border-radius:3px}}
.kick{{--c:var(--kick)}} .punch{{--c:var(--punch)}} .feint{{--c:var(--feint)}} .move{{--c:var(--move)}}
.ev,.dot{{background:var(--c)}} .dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-inline-end:6px}}
.axis{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-inline-start:90px;direction:ltr}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted)}}
table{{border-collapse:collapse;font-size:14px}} td,th{{padding:4px 12px;border-bottom:1px solid var(--line);text-align:right}}
.n{{font-variant-numeric:tabular-nums;text-align:center}}
.who{{font-weight:700}} .who.p1{{color:var(--p1)}} .who.p2{{color:var(--p2)}}
.bar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
button{{font:inherit;border:1px solid var(--line);background:var(--paper);border-radius:6px;padding:4px 12px;cursor:pointer}}
button[aria-pressed=true]{{background:var(--ink);color:var(--paper)}}
.win{{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:12px;display:grid;
grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:14px;align-items:start}}
.win img{{width:100%;border-radius:6px}} .t{{grid-column:1/-1;font-weight:700;direction:ltr;text-align:right}}
.answers{{display:grid;gap:10px}} .ans{{border-inline-start:4px solid var(--c);padding:6px 10px;background:#fafbfc;border-radius:4px}}
.lab{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}} .conf{{color:var(--muted);font-size:13px}}
.steps{{font-size:12.5px;color:var(--muted);direction:ltr;text-align:right}} .ans p{{margin:4px 0 0;font-size:13.5px;direction:ltr;text-align:left}}
.warn{{color:var(--warn);font-size:12px;font-weight:600}}
body.only-atk .quiet{{display:none}}
@media (max-width:760px){{.win{{grid-template-columns:1fr}}}}
</style></head><body class="only-atk"><div class="page">
<header><h1>{esc(data["model"])} · {esc(MATCH.get(name, name))}</h1>
<div class="sub">{len(res)} شباك (1.2 ثانية، كل 0.8 ثانية) · من غير أي تدريب · كل إجابة متشافة جنب صورتها</div></header>
<section class="box"><h2 style="margin:0 0 8px;font-size:17px">الـ timeline</h2>{tl(1)}{tl(2)}
<div class="axis"><span>0s</span><span>{dur / 2:.0f}s</span><span>{dur:.0f}s</span></div>
<div class="legend"><span><i class="dot kick"></i>كيك</span><span><i class="dot punch"></i>لكمة</span><span><i class="dot feint"></i>Feint</span>
<span>الشبابيك المتجاورة بنفس الحركة اتدمجت في حدث واحد. دوس على أي حدث يوديك لصورته.</span></div></section>
<section class="box"><h2 style="margin:0 0 8px;font-size:17px">عدد الحركات</h2>
<table><tr><th>الحركة</th><th class="who p1">Player 1</th><th class="who p2">Player 2</th></tr>{count_rows}</table></section>
<div class="bar"><span>اعرض:</span><button id="f-atk" aria-pressed="true">الشبابيك اللي فيها هجوم بس</button>
<button id="f-all" aria-pressed="false">كل الشبابيك بالترتيب</button></div>
{cards}
</div><script>
const b=document.body,a=document.getElementById("f-atk"),l=document.getElementById("f-all");
a.onclick=()=>{{b.classList.add("only-atk");a.ariaPressed="true";l.ariaPressed="false"}};
l.onclick=()=>{{b.classList.remove("only-atk");l.ariaPressed="true";a.ariaPressed="false"}};
document.querySelectorAll(".ev").forEach(e=>e.addEventListener("click",()=>l.click()));
</script></body></html>'''
    out = HERE / f"report_{key}_{name}.html"
    out.write_text(page, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
