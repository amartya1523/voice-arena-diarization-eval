"""Build the submission PDFs from the HTML sources and the pilot results.

- docs/src/main.html      -> Amartya_Singh_Q1_Main.pdf   (pilot section generated here)
- docs/src/onepager.html  -> Amartya_Singh_Q1_OnePager.pdf
- leaderboard/index.html  -> Amartya_Singh_Q1_Leaderboard_Sketch.pdf / .png

Charts are drawn with matplotlib from results/*.csv, so every number in the
PDFs comes straight from the scoring output.
"""
import json, pathlib, subprocess, html
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, BUILD, OUT = ROOT / "docs/src", ROOT / "docs/build", ROOT / "deliverables"
FIG = BUILD / "figs"
for d in (BUILD, FIG, OUT):
    d.mkdir(parents=True, exist_ok=True)
import time
START = time.time()
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

S = pd.read_csv(ROOT / "results/summary.csv")
F = pd.read_csv(ROOT / "results/per_file.csv")
P = pd.read_csv(ROOT / "results/pairwise.csv")
X = pd.read_csv(ROOT / "results/crosscheck.csv")
MAN = json.loads((ROOT / "data/testset/manifest.json").read_text())
TUNE = {s: json.loads((ROOT / f"results/devset/{s}/tuning.json").read_text()) for s in ("titanet", "eres2net", "wespeaker")}
SHORT = {"titanet": "NeMo TitaNet-small", "eres2net": "3D-Speaker ERes2Net-base", "wespeaker": "WeSpeaker ResNet34-LM"}
COND = {"close_talk": "Close-talk", "far_field": "Far-field", "phone_line": "Phone line"}

plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#333", "ytick.color": "#333"})
COLORS = {"titanet": "#1d4ed8", "eres2net": "#e0a526", "wespeaker": "#d1495b"}

main_sys = S[~S.oracle_count & S.system.isin(SHORT)].set_index("system")
orc = S[S.oracle_count].assign(system=lambda d: d.system.str.replace("_oracle", "")).set_index("system")
best = main_sys.der.idxmin()
pct = lambda v: f"{100 * v:.1f}"
dscore_jer = X.groupby("system").dscore_jer.mean()

def ami_der(s, c):
    sub = F[(F.system == s) & (F.condition == c) & F.uri.isin(["IS1009a", "TS3003a"])]
    return (sub.missed.sum() + sub.fa.sum() + sub.confusion.sum()) / sub.total.sum()


# ---------- figure 1: DER by condition (AMI files only, same conversations) ----------
fig, ax = plt.subplots(figsize=(6.6, 2.6))
conds = ["close_talk", "far_field", "phone_line"]
w = 0.26
for i, s in enumerate(SHORT):
    vals = []
    for c in conds:
        sub = F[(F.system == s) & (F.condition == c) & F.uri.isin(["IS1009a", "TS3003a"])]
        vals.append(100 * (sub.missed.sum() + sub.fa.sum() + sub.confusion.sum()) / sub.total.sum())
    xs = [k + (i - 1) * w for k in range(3)]
    ax.bar(xs, vals, w * 0.92, color=COLORS[s], label=SHORT[s])
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=7.5, color="#333")
ax.set_xticks(range(3), [COND[c] for c in conds])
ax.set_ylabel("DER % (collar 0, overlap scored)")
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16))
fig.tight_layout(); fig.savefig(FIG / "der_by_condition.png", dpi=220); plt.close(fig)

# ---------- figure 2: dev tuning curves ----------
fig, ax = plt.subplots(figsize=(6.6, 2.3))
for s in SHORT:
    g = pd.DataFrame(TUNE[s]["grid"])
    ax.plot(g.threshold, 100 * g.der, marker="o", ms=3.5, color=COLORS[s], label=SHORT[s])
    ch = TUNE[s]["chosen"]
    ax.plot([ch], [100 * g.set_index("threshold").der[ch]], marker="o", ms=8, mfc="none", color=COLORS[s])
ax.set_xlabel("clustering threshold (larger = merge more voices)")
ax.set_ylabel("dev DER %")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "dev_tuning.png", dpi=220); plt.close(fig)

# ---------- figure 3: what the scoring settings do to the same output ----------
b = main_sys.loc[best]
settings = [("Primary\ncollar 0, overlap", b.der), ("Collar 0.25 s,\noverlap scored", b.der_collar25),
            ("Collar 0,\noverlap skipped", b.der_skip_overlap), ("Legacy: 0.25 s,\noverlap skipped", b.der_legacy),
            ("Speaker count\ngiven (oracle)", orc.loc[best].der if best in orc.index else float("nan"))]
fig, ax = plt.subplots(figsize=(6.6, 2.3))
ax.bar(range(len(settings)), [100 * v for _, v in settings], 0.6, color=["#15171c"] + ["#9aa3b5"] * 4)
for i, (_, v) in enumerate(settings):
    ax.text(i, 100 * v + 0.8, f"{100 * v:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(settings)), [n for n, _ in settings], fontsize=7.5)
ax.set_ylabel("DER %")
fig.tight_layout(); fig.savefig(FIG / "settings_effect.png", dpi=220); plt.close(fig)

# ---------- pilot section HTML ----------
def row(cells, cls=""):
    return f"<tr class='{cls}'>" + "".join(f"<td class='{'r' if i else ''}'>{c}</td>" for i, c in enumerate(cells)) + "</tr>"

data_rows = []
for m in MAN:
    f = F[(F.system == best) & (F.uri == m["uri"]) & (F.condition == m["condition"])].iloc[0]
    data_rows.append(f"<tr><td>{m['uri']}</td><td>{m['source']}</td><td>{COND[m['condition']]}</td>"
                     f"<td class='r'>{(m['end'] - m['start']) / 60:.1f}</td><td class='r'>{m['speakers']}</td>"
                     f"<td class='r'>{100 * f.overlap_share:.1f}%</td><td class='r'>{int(f.short_turns)}</td></tr>")

sys_rows = []
for s in SHORT:
    sys_rows.append(f"<tr><td>{html.escape(main_sys.loc[s].display)}</td><td class='code'>{json.loads((ROOT / f'results/testset/{s}/run.json').read_text())['system']}</td>"
                    f"<td class='r'>{TUNE[s]['chosen']}</td><td class='r'>{', '.join(str(g['threshold']) for g in TUNE[s]['grid'])}</td></tr>")

res_rows = []
for s in main_sys.sort_values("der").index:
    r = main_sys.loc[s]
    res_rows.append(row([f"<b>{SHORT[s]}</b>", f"<b>{pct(r.der)}</b>", f"{pct(r.ci_lo)}–{pct(r.ci_hi)}", pct(r.missed), pct(r.fa),
                         pct(r.confusion), f"{dscore_jer[s]:.1f}", f"{100 * r.count_exact:.0f}%", f"{r.mean_abs_count_error:.1f}",
                         pct(r.der_overlap_regions), f"{100 * r.short_turn_recall:.0f}%", f"{r.rtf:.2f}"], "best" if s == best else ""))

pair_rows = []
names = list(main_sys.sort_values("der").index)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        p = P[(P.a == names[i]) & (P.b == names[j])].iloc[0]
        sig = "yes" if p.diff_hi < 0 and abs(p["diff"]) >= 0.005 else "no"
        pair_rows.append(row([f"{SHORT[names[i]]} vs {SHORT[names[j]]}", f"{100 * p['diff']:+.1f}",
                              f"{100 * p.diff_lo:+.1f} to {100 * p.diff_hi:+.1f}", f"{100 * p.p_a_better:.1f}%", sig]))

orc_rows = []
for s in SHORT:
    if s in orc.index:
        orc_rows.append(row([SHORT[s], pct(main_sys.loc[s].der), pct(orc.loc[s].der), f"{100 * (orc.loc[s].der - main_sys.loc[s].der):+.1f}",
                             pct(main_sys.loc[s].confusion), pct(orc.loc[s].confusion)]))

default = S[S.system == "wespeaker_default_thr0.5"]
ctx = dict(
    best=SHORT[best], data_rows="\n".join(data_rows), sys_rows="\n".join(sys_rows), res_rows="\n".join(res_rows),
    pair_rows="\n".join(pair_rows), orc_rows="\n".join(orc_rows),
    xcheck_max=f"{X.abs_diff.max():.2f}", xcheck_n=len(X),
    total_min=f"{sum(m['end'] - m['start'] for m in MAN) / 60:.0f}",
    default_der=pct(default.der.iloc[0]) if len(default) else "n/a",
    default_count=f"{default.mean_abs_count_error.iloc[0]:.1f}" if len(default) else "n/a",
    tuned_wes=pct(main_sys.loc["wespeaker"].der), tuned_wes_count=f"{main_sys.loc['wespeaker'].mean_abs_count_error:.1f}",
    wes_thr=TUNE["wespeaker"]["chosen"],
    b_der=pct(b.der), b_c25=pct(b.der_collar25), b_skip=pct(b.der_skip_overlap), b_legacy=pct(b.der_legacy),
    b_far=f"{100 * b.delta_far_field:+.1f}", b_phone=f"{100 * b.delta_phone_line:+.1f}",
    b_jer=f"{dscore_jer[best]:.1f}",
    b_far_abs=f"{100 * ami_der(best, 'far_field'):.1f}", b_close_ami=f"{100 * ami_der(best, 'close_talk'):.1f}",
    tit_count=f"{main_sys.loc['titanet'].mean_abs_count_error:.1f}",
    orc_titanet=pct(orc.loc["titanet"].der), orc_wespeaker=pct(orc.loc["wespeaker"].der), der_wespeaker=pct(main_sys.loc["wespeaker"].der),
    jer_titanet=f"{dscore_jer['titanet']:.1f}%", jer_eres2net=f"{dscore_jer['eres2net']:.1f}%",
)
ctx["notes"] = (SRC / "pilot_notes.html").read_text().format(**ctx)
pilot = (SRC / "pilot.html").read_text().format(**ctx)
main = (SRC / "main.html").read_text().replace("<!--PILOT-->", pilot)
(SRC / "_main.html").write_text(main)

# ---------- one-pager fill ----------
op = (SRC / "onepager.html").read_text()
rows = []
for s in main_sys.sort_values("der").index:
    r = main_sys.loc[s]
    rows.append(f"<tr class='{'best' if s == best else ''}'><td>{SHORT[s]}</td><td class='r'><b>{pct(r.der)}</b></td><td class='r'>{pct(r.ci_lo)}–{pct(r.ci_hi)}</td>"
                f"<td class='r'>{pct(r.der_close_talk)}</td><td class='r'>{pct(r.der_far_field)}</td><td class='r'>{pct(r.der_phone_line)}</td></tr>")
fill = json.loads((SRC / "onepager_fill.json").read_text())
op = op.replace("{{pilot_rows}}", "\n".join(rows))
for k, v in fill.items():
    op = op.replace("{{" + k + "}}", v.format(**ctx) if isinstance(v, str) else "\n".join(f"<li>{x.format(**ctx)}</li>" for x in v))
(SRC / "_onepager.html").write_text(op)


def pdf(src, dst):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={BUILD / 'chrome-profile'}", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={dst}", f"file://{src}"], capture_output=True)
    assert dst.exists() and dst.stat().st_mtime > START, f"Chrome did not write {dst}"
    print("wrote", dst.name)


pdf(SRC / "_main.html", OUT / "Amartya_Singh_Q1_Main.pdf")
pdf(SRC / "_onepager.html", OUT / "Amartya_Singh_Q1_OnePager.pdf")
lb = ROOT / "leaderboard"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={BUILD / 'chrome-profile'}", "--hide-scrollbars", "--window-size=1760,1060",
                "--force-device-scale-factor=2", "--virtual-time-budget=5000",
                f"--screenshot={OUT / 'Amartya_Singh_Q1_Leaderboard_Sketch.png'}", f"file://{lb / 'index.html'}"], capture_output=True)
(lb / "leaderboard_sketch.png").write_bytes((OUT / "Amartya_Singh_Q1_Leaderboard_Sketch.png").read_bytes())
print("wrote leaderboard sketch")
