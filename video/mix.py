"""Mix the voice-over clips onto the timeline with a quiet synth bed, loudness-normalise, mux."""
import json, subprocess, numpy as np, soundfile as sf, pathlib
here = pathlib.Path(__file__).parent
pl = json.load(open(here / "vo/placement.json"))
SR = 44100
order = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
n = int((pl["total"] + 0.5) * SR)
voice = np.zeros(n)
for k, st in zip(order, pl["starts"]):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"vo/{k}.mp3", "-filter:a", "atempo=1.04", "-ac", "1", "-ar", str(SR), f"vo/{k}.wav"], check=True)
    x, _ = sf.read(f"vo/{k}.wav")
    i = int(st * SR); voice[i:i + len(x)] += x[: n - i]

# soft pad: slow chord changes, gentle attack, low-passed by using few harmonics
t = np.arange(n) / SR
chords = [[220.0, 277.18, 329.63], [196.0, 246.94, 293.66], [174.61, 220.0, 261.63], [196.0, 246.94, 329.63]]
bed = np.zeros(n); seg = 8.0
for ci in range(int(np.ceil(pl["total"] / seg))):
    a, b = int(ci * seg * SR), min(int((ci + 1) * seg * SR) + int(SR), n)
    tt = t[a:b] - t[a]
    env = np.minimum(1, tt / 2.0) * np.minimum(1, np.maximum(0, (b - a) / SR - tt) / 2.0)
    for f in chords[ci % 4]:
        bed[a:b] += env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt + 0.5)) / 3
bed *= 0.05 * np.minimum(1, t / 1.5) * np.minimum(1, (pl["total"] - t).clip(0) / 2.5)
# duck the bed under speech
env = np.convolve(np.abs(voice), np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same")
bed *= 1 - 0.55 * np.clip(env / (env.max() * 0.15), 0, 1)
mix = voice + bed
sf.write("frames/mix.wav", mix / max(1e-9, np.abs(mix).max()) * 0.9, SR)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", "frames/video_noaudio.mp4", "-i", "frames/mix.wav",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest",
                "../deliverables/Amartya_Singh_Q1_Video_Walkthrough.mp4"], check=True)
print("muxed")
