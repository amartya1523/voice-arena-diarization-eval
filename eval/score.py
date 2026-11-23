"""Score every hypothesis folder in results/<split>/ against the references.

Primary metric : DER, collar 0 s, overlap scored, unknown speaker count.
Also reported  : legacy DER (0.25 s collar, overlap skipped), JER, DER parts
                 (missed / false alarm / confusion), speaker-count error,
                 DER restricted to overlap regions, real-time factor.
Uncertainty    : 95% bootstrap CIs over 60-second blocks, plus paired
                 bootstrap win-rates between systems on the same blocks.
"""
import json, pathlib, itertools
import numpy as np, pandas as pd
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.metrics.identification import IdentificationErrorRate

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT = "testset"
REF = ROOT / "data" / SPLIT
RES = ROOT / "results" / SPLIT
BLOCK = 60.0
RNG = np.random.default_rng(1523)
N_BOOT = 2000


def load_rttm(path, uri):
    ann = Annotation(uri=uri)
    for line in path.read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            if d > 0:
                ann[Segment(s, s + d), len(ann)] = p[7]
    return ann


def load_uem(path):
    p = path.read_text().split()
    return Timeline([Segment(float(p[2]), float(p[3]))])


def overlap_timeline(ref):
    tl = Timeline()
    segs = list(ref.itertracks())
    for (s1, _), (s2, _) in itertools.combinations(segs, 2):
        if s1 & s2:
            tl.add(s1 & s2)
    return tl.support()


def components(metric, ref, hyp, uem):
    d = metric(ref, hyp, uem=uem, detailed=True)
    return d


def score_system(sysdir):
    rows, blocks = [], []
    der0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der_legacy = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    der_c25 = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    der_skip = DiarizationErrorRate(collar=0.0, skip_overlap=True)
    for hyp_path in sorted(sysdir.glob("*/*.rttm")):
        cond, uri = hyp_path.parent.name, hyp_path.stem
        ref = load_rttm(REF / cond / f"{uri}.rttm", uri)
        hyp = load_rttm(hyp_path, uri)
        uem = load_uem(REF / cond / f"{uri}.uem")
        d = der0(ref, hyp, uem=uem, detailed=True)
        legacy = der_legacy(ref, hyp, uem=uem)
        c25 = der_c25(ref, hyp, uem=uem)
        skip = der_skip(ref, hyp, uem=uem)
        j = jer(ref, hyp, uem=uem)
        # Fix the speaker mapping once per file (the one DER chose), then measure
        # sub-regions with that mapping. Re-mapping inside every block or inside
        # overlap only would let the system pick a friendlier mapping per region.
        mapping = der0.optimal_mapping(ref, hyp, uem=uem)
        mapped = hyp.rename_labels(mapping={k: mapping.get(k, f"_extra_{k}") for k in hyp.labels()})
        ier = IdentificationErrorRate(collar=0.0, skip_overlap=False)
        ov = overlap_timeline(ref).crop(uem)
        d_ov = ier(ref, mapped, uem=ov, detailed=True) if ov.duration() > 0 else None
        n_ref, n_hyp = len(ref.crop(uem).labels()), len(hyp.crop(uem).labels())
        short = [s for s, _, _ in ref.crop(uem).itertracks(yield_label=True) if s.duration < 1.0]
        hit = 0
        for (seg, _, lab) in ref.crop(uem).itertracks(yield_label=True):
            if seg.duration >= 1.0:
                continue
            got = mapped.crop(seg).label_timeline(lab).duration() if lab in mapped.labels() else 0.0
            hit += got >= 0.5 * seg.duration
        rows.append(dict(condition=cond, uri=uri, total=d["total"], missed=d["missed detection"],
                         fa=d["false alarm"], confusion=d["confusion"], der=d["diarization error rate"],
                         der_legacy=legacy, der_collar25=c25, der_skip_overlap=skip, jer=j, overlap_share=ov.duration() / uem.duration(),
                         der_overlap=None if d_ov is None else d_ov["identification error rate"],
                         overlap_total=0.0 if d_ov is None else d_ov["total"],
                         short_turns=len(short), short_turn_recall=hit / max(len(short), 1),
                         ref_speakers=n_ref, hyp_speakers=n_hyp, count_error=n_hyp - n_ref))
        end = uem.extent().end
        for b0 in np.arange(0, end, BLOCK):
            seg = Timeline([Segment(b0, min(b0 + BLOCK, end))])
            db = ier(ref, mapped, uem=seg, detailed=True)
            err = db["missed detection"] + db["false alarm"] + db["confusion"]
            blocks.append(dict(condition=cond, uri=uri, block=int(b0 // BLOCK), err=err, total=db["total"]))
    return pd.DataFrame(rows), pd.DataFrame(blocks)


def pooled(df):
    return (df.missed.sum() + df.fa.sum() + df.confusion.sum()) / df.total.sum()


def boot_ci(blocks):
    e, t = blocks.err.values, blocks.total.values
    idx = RNG.integers(0, len(e), (N_BOOT, len(e)))
    s = e[idx].sum(1) / t[idx].sum(1)
    return np.percentile(s, [2.5, 97.5])


def main():
    systems = {}
    for sysdir in sorted(p for p in RES.iterdir() if p.is_dir() and (p / "run.json").exists()):
        meta = json.loads((sysdir / "run.json").read_text())
        if len(list(sysdir.glob("*/*.rttm"))) < len(list(REF.glob("*/*.rttm"))):
            print("skip incomplete", sysdir.name); continue
        rows, blocks = score_system(sysdir)
        systems[sysdir.name] = (meta, rows, blocks)

    summary, per_file = [], []
    for name, (meta, rows, blocks) in systems.items():
        lo, hi = boot_ci(blocks)
        rtf = sum(f["runtime"] for f in meta["files"]) / sum(f["seconds"] for f in meta["files"])
        rec = dict(system=name, display=meta["display"], oracle_count=meta["oracle_count"],
                   der=pooled(rows), ci_lo=lo, ci_hi=hi,
                   missed=rows.missed.sum() / rows.total.sum(), fa=rows.fa.sum() / rows.total.sum(),
                   confusion=rows.confusion.sum() / rows.total.sum(),
                   der_legacy=float(np.average(rows.der_legacy, weights=rows.total)),
               der_collar25=float(np.average(rows.der_collar25, weights=rows.total)),
               der_skip_overlap=float(np.average(rows.der_skip_overlap, weights=rows.total)),
                   jer=float(rows.jer.mean()), mean_abs_count_error=float(rows.count_error.abs().mean()), rtf=rtf)
        for cond in ["close_talk", "far_field", "phone_line"]:
            sub = rows[rows.condition == cond]
            rec[f"der_{cond}"] = pooled(sub)
            if cond != "close_talk":  # same two AMI meetings in all three conditions
                ami_close = rows[(rows.condition == "close_talk") & rows.uri.isin(sub.uri)]
                rec[f"delta_{cond}"] = pooled(sub) - pooled(ami_close)
        ov = rows.dropna(subset=["der_overlap"])
        rec["der_overlap_regions"] = float((ov.der_overlap * ov.overlap_total).sum() / ov.overlap_total.sum())
        rec["short_turn_recall"] = float((rows.short_turn_recall * rows.short_turns).sum() / rows.short_turns.sum())
        rec["count_exact"] = float((rows.count_error == 0).mean())
        summary.append(rec)
        per_file.append(rows.assign(system=name))

    summary = pd.DataFrame(summary).sort_values("der")
    per_file = pd.concat(per_file)

    # paired bootstrap: same resampled blocks for both systems
    pairs = []
    names = list(systems)
    for a, b in itertools.permutations(names, 2):
        A, B = systems[a][2], systems[b][2]
        key = ["condition", "uri", "block"]
        m = A.merge(B, on=key, suffixes=("_a", "_b"))
        idx = RNG.integers(0, len(m), (N_BOOT, len(m)))
        da = m.err_a.values[idx].sum(1) / m.total_a.values[idx].sum(1)
        db = m.err_b.values[idx].sum(1) / m.total_b.values[idx].sum(1)
        pairs.append(dict(a=a, b=b, p_a_better=float((da < db).mean()),
                          diff=float(np.median(da - db)), diff_lo=float(np.percentile(da - db, 2.5)),
                          diff_hi=float(np.percentile(da - db, 97.5))))
    pairs = pd.DataFrame(pairs)

    out = ROOT / "results"
    summary.to_csv(out / "summary.csv", index=False)
    per_file.to_csv(out / "per_file.csv", index=False)
    pairs.to_csv(out / "pairwise.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(pairs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
