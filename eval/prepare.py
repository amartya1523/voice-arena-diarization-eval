"""Build the mini test set used in the pilot run.

Takes public recordings (AMI test meetings, VoxConverse test clips) from the
diarizers-community Hugging Face parquet mirrors, cuts fixed windows, writes
16 kHz mono WAVs, matching reference RTTMs and UEMs, and creates the
"phone line" condition by passing the close-talk mix through a G.711 mu-law
8 kHz channel with a 300-3400 Hz band-pass.
"""
import io, json, subprocess, pathlib
import pyarrow.parquet as pq
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "testset"
RAW, REF, OUT = ROOT / "data/raw", ROOT / "data/ref", ROOT / "data" / SPLIT

# (source parquet, file inside it, uri, window start, window end, reference rttm)
TEST_ITEMS = [
    ("ami_ihm_test0", "IS1009a.Mix-Headset.wav", "IS1009a", 0.0, 838.0, "IS1009a.rttm"),
    ("ami_ihm_test0", "TS3003a.Mix-Headset.wav", "TS3003a", 300.0, 900.0, "TS3003a.rttm"),
    ("vox_test0", "fzwtp.wav", "fzwtp", 0.0, 285.0, "fzwtp.rttm"),
    ("vox_test0", "bjruf.wav", "bjruf", 0.0, 360.0, "bjruf.rttm"),
]
# Dev split: different meetings / clips, used only to pick each system's clustering threshold.
DEV_ITEMS = [
    ("ami_ihm_test0", "ES2004b.Mix-Headset.wav", "ES2004b", 600.0, 840.0, "ES2004b.rttm"),
    ("vox_test0", "qwepo.wav", "qwepo", 0.0, 240.0, "qwepo.rttm"),
    ("vox_test0", "vgaez.wav", "vgaez", 0.0, 177.0, "vgaez.rttm"),
]
ITEMS = DEV_ITEMS if SPLIT == "devset" else TEST_ITEMS
FARFIELD = {"IS1009a": "IS1009a.Array1-01.wav", "TS3003a": "TS3003a.Array1-01.wav", "ES2004b": "ES2004b.Array1-01.wav"}


def load(parquet, name):
    t = pq.read_table(RAW / f"{parquet}.parquet")
    for i in range(t.num_rows):
        a = t.column("audio")[i].as_py()
        if a["path"] == name:
            spk = t.column("speakers")[i].as_py()
            st = t.column("timestamps_start")[i].as_py()
            en = t.column("timestamps_end")[i].as_py()
            return sf.read(io.BytesIO(a["bytes"])), list(zip(st, en, spk))
    raise KeyError(name)


def read_rttm(path, segs_fallback):
    if path.exists() and path.stat().st_size > 0:
        rows = []
        for line in path.read_text().splitlines():
            p = line.split()
            if p and p[0] == "SPEAKER":
                s, d = float(p[3]), float(p[4])
                rows.append((s, s + d, p[7]))
        return rows
    return segs_fallback


def write_window(uri, cond, audio, sr, segs, t0, t1):
    d = OUT / cond
    d.mkdir(parents=True, exist_ok=True)
    clip = audio[int(t0 * sr): int(t1 * sr)]
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    sf.write(d / f"{uri}.wav", clip, sr, subtype="PCM_16")
    lines = []
    for s, e, spk in segs:
        s, e = max(s, t0), min(e, t1)
        if e - s > 0:
            lines.append(f"SPEAKER {uri} 1 {s - t0:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>")
    (d / f"{uri}.rttm").write_text("\n".join(lines) + "\n")
    (d / f"{uri}.uem").write_text(f"{uri} 1 0.000 {t1 - t0:.3f}\n")


def phone(src, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af",
                    "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-c:a", "pcm_mulaw",
                    "-f", "wav", "/tmp/_mulaw.wav"], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "/tmp/_mulaw.wav",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(dst)], check=True)


manifest = []
for parquet, name, uri, t0, t1, rttm in ITEMS:
    (audio, sr), segs = load(parquet, name)
    segs = read_rttm(REF / rttm, segs)
    src = "AMI" if parquet.startswith("ami") else "VoxConverse"
    write_window(uri, "close_talk", audio, sr, segs, t0, t1)
    manifest.append(dict(uri=uri, source=src, condition="close_talk", start=t0, end=t1,
                         speakers=len({k for a, b, k in segs if a < t1 and b > t0})))
    if uri in FARFIELD and SPLIT == "devset":
        (far, sr2), _ = load(parquet.replace("ihm", "sdm"), FARFIELD[uri])
        write_window(uri, "far_field", far, sr2, segs, t0, t1)
        manifest.append(dict(manifest[-1], condition="far_field"))
    elif uri in FARFIELD:
        (far, sr2), _ = load(parquet.replace("ihm", "sdm"), FARFIELD[uri])
        write_window(uri, "far_field", far, sr2, segs, t0, t1)
        manifest.append(dict(manifest[-1], condition="far_field"))
        d = OUT / "phone_line"; d.mkdir(parents=True, exist_ok=True)
        phone(OUT / "close_talk" / f"{uri}.wav", d / f"{uri}.wav")
        for ext in ("rttm", "uem"):
            (d / f"{uri}.{ext}").write_text((OUT / "close_talk" / f"{uri}.{ext}").read_text())
        manifest.append(dict(manifest[-2], condition="phone_line"))

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
