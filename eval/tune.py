"""Pick each system's clustering threshold on the dev split only.

The threshold decides when two voices are "the same speaker". The value that
ships as a default is tuned for some other data, so every system gets the same
small grid on dev audio and keeps the value with the lowest pooled DER. Test
files are never touched here.

usage: python eval/tune.py --system wespeaker
"""
import argparse, json, pathlib
import soundfile as sf
from pyannote.metrics.diarization import DiarizationErrorRate
from systems import SYSTEMS, build, ROOT
from score import load_rttm, load_uem

ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=SYSTEMS)
ap.add_argument("--grid", default="0.5,0.7,0.9,1.1,1.3",
                help="thresholds to try; extend the grid if the best value sits on its edge")
a = ap.parse_args()
GRID = [float(x) for x in a.grid.split(",")]
log = ROOT / "results/devset" / a.system / "grid.jsonl"   # every point ever tried, so the grid can be extended
log.parent.mkdir(parents=True, exist_ok=True)

dev = ROOT / "data/devset"
files = sorted(dev.glob("*/*.wav"))
audio = {f: sf.read(f, dtype="float32")[0] for f in files}
table = []
for thr in GRID:
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    sd = build(a.system, thr)
    counts = []
    for f in files:
        res = sd.process(audio[f]).sort_by_start_time()
        hyp_lines = [f"SPEAKER {f.stem} 1 {r.start:.3f} {r.end - r.start:.3f} <NA> <NA> spk{r.speaker:02d} <NA> <NA>" for r in res]
        tmp = ROOT / "results/devset" / a.system / f"thr{thr}" / f.parent.name
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / f"{f.stem}.rttm").write_text("\n".join(hyp_lines) + "\n")
        metric(load_rttm(f.with_suffix(".rttm"), f.stem), load_rttm(tmp / f"{f.stem}.rttm", f.stem), uem=load_uem(f.with_suffix(".uem")))
        counts.append(len({r.speaker for r in res}))
    table.append(dict(threshold=thr, der=abs(metric), speakers=counts))
    with log.open("a") as fh:
        fh.write(json.dumps(table[-1]) + "\n")
    print(a.system, thr, round(abs(metric), 4), counts, flush=True)

table = sorted({r["threshold"]: r for r in map(json.loads, log.read_text().splitlines())}.values(), key=lambda r: r["threshold"])
best = min(table, key=lambda r: r["der"])
out = ROOT / "results/devset" / a.system / "tuning.json"
out.write_text(json.dumps(dict(system=a.system, grid=table, chosen=best["threshold"]), indent=2))
print("chosen", best)
