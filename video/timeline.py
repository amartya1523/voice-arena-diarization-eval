"""Turn the voice-over clips into a timeline the scenes page can follow.

Reads vo/<scene>.json (ElevenLabs character alignment), groups characters into
words with start times, lays scenes back to back with a short gap, and writes
the result into scenes.html as window data, plus a concat list for the audio.
"""
import json, re, pathlib, subprocess
ORDER = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
GAP, LEAD, TAIL, TEMPO = 0.4, 0.6, 1.5, 1.04
here = pathlib.Path(__file__).parent
scenes, t = [], LEAD
for k in ORDER:
    al = json.loads((here / f"vo/{k}.json").read_text())
    chars, st = al["characters"], al["character_start_times_seconds"]
    words, cur, cur_t = [], "", None
    for c, s in zip(chars, st):
        if c.isspace():
            if cur: words.append([cur, round(cur_t / TEMPO, 3)])
            cur, cur_t = "", None
        else:
            if not cur: cur_t = s
            cur += c
    if cur: words.append([cur, round(cur_t / TEMPO, 3)])
    dur = al["character_end_times_seconds"][-1] / TEMPO
    scenes.append(dict(id=k, start=round(t, 3), voice=round(dur, 3), dur=round(dur + GAP, 3), words=words))
    t += dur + GAP
scenes[0]["start"] = 0.0; scenes[0]["dur"] += LEAD
scenes[-1]["dur"] += TAIL; scenes[-1]["last"] = True
total = round(t + TAIL, 3)
tl = dict(scenes=scenes, total=total, lead=LEAD)
src = (here / "scenes.html").read_text()
(here / "render.html").write_text(src.replace("/*TIMELINE*/null", json.dumps(tl)))
# audio: each clip placed at its scene's voice start
starts = [LEAD] + [s["start"] for s in scenes[1:]]
json.dump(dict(starts=starts, total=total), open(here / "vo/placement.json", "w"))
print("total", total)
