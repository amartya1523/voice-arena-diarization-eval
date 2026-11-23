"""Cross-check pyannote.metrics DER against dscore (the DIHARD scorer).

For every system and file, run dscore with collar 0 and overlap scored, then
compare to the DER in results/per_file.csv. dscore's JER (the original DIHARD
definition) is also saved, since pyannote.metrics computes JER slightly
differently.
"""
import subprocess, sys, pathlib, re
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "data/testset"
per_file = pd.read_csv(ROOT / "results/per_file.csv")
rows = []
for sysdir in sorted((ROOT / "results/testset").iterdir()):
    if not (sysdir / "run.json").exists():
        continue
    for hyp in sorted(sysdir.glob("*/*.rttm")):
        cond, uri = hyp.parent.name, hyp.stem
        out = subprocess.run([sys.executable, str(ROOT / "tools/dscore/score.py"), "--collar", "0",
                              "-u", str(REF / cond / f"{uri}.uem"), "-r", str(REF / cond / f"{uri}.rttm"),
                              "-s", str(hyp)], capture_output=True, text=True, check=True).stdout
        line = next(l for l in out.splitlines() if l.startswith(uri))
        der, jer = map(float, line.split()[1:3])
        mine = per_file[(per_file.system == sysdir.name) & (per_file.condition == cond) & (per_file.uri == uri)]
        rows.append(dict(system=sysdir.name, condition=cond, uri=uri, dscore_der=der, dscore_jer=jer,
                         pyannote_der=round(100 * mine.der.iloc[0], 2) if len(mine) else None))
df = pd.DataFrame(rows)
df["abs_diff"] = (df.dscore_der - df.pyannote_der).abs()
df.to_csv(ROOT / "results/crosscheck.csv", index=False)
print(df.to_string(index=False))
print("max |DER difference| =", df.abs_diff.max())
