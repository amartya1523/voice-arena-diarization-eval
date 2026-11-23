"""Run one system on every file of a split and write RTTM hypotheses.

usage: python eval/run.py --system wespeaker --split testset [--threshold 0.5] [--oracle-count]
"""
import argparse, json, time, pathlib, platform
import numpy as np, soundfile as sf, sherpa_onnx
from systems import SYSTEMS, build, ROOT


def count_speakers(rttm):
    return len({l.split()[7] for l in rttm.read_text().splitlines() if l.strip()})


ap = argparse.ArgumentParser()
ap.add_argument("--system", required=True, choices=SYSTEMS)
ap.add_argument("--split", default="testset")
ap.add_argument("--threshold", default="auto", help="float, or 'auto' = value picked on dev by tune.py")
ap.add_argument("--oracle-count", action="store_true")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

if a.threshold == "auto":
    a.threshold = json.loads((ROOT / "results/devset" / a.system / "tuning.json").read_text())["chosen"]
a.threshold = float(a.threshold)
tag = a.tag or (a.system + ("_oracle" if a.oracle_count else ""))
base = ROOT / "data" / a.split
out = ROOT / "results" / a.split / tag
log = []
for wav in sorted(base.glob("*/*.wav")):
    cond, uri = wav.parent.name, wav.stem
    audio, sr = sf.read(wav, dtype="float32")
    n = count_speakers(wav.with_suffix(".rttm")) if a.oracle_count else -1
    sd = build(a.system, a.threshold, n)
    assert sr == sd.sample_rate
    t = time.perf_counter()
    res = sd.process(audio).sort_by_start_time()
    dt = time.perf_counter() - t
    (out / cond).mkdir(parents=True, exist_ok=True)
    lines = [f"SPEAKER {uri} 1 {r.start:.3f} {r.end - r.start:.3f} <NA> <NA> spk{r.speaker:02d} <NA> <NA>" for r in res]
    (out / cond / f"{uri}.rttm").write_text("\n".join(lines) + "\n")
    dur = len(audio) / sr
    log.append(dict(uri=uri, condition=cond, seconds=round(dur, 1), runtime=round(dt, 2), rtf=round(dt / dur, 4),
                    hyp_speakers=len({r.speaker for r in res})))
    print(tag, cond, uri, f"rtf={dt/dur:.3f}", "speakers", log[-1]["hyp_speakers"], flush=True)

(out / "run.json").write_text(json.dumps(dict(
    system=a.system, display=SYSTEMS[a.system][0], threshold=a.threshold, oracle_count=a.oracle_count,
    sherpa_onnx=sherpa_onnx.__version__, machine=f"{platform.machine()} {platform.system()} {platform.release()}",
    threads=4, min_duration_on=0.3, min_duration_off=0.5, files=log), indent=2))
