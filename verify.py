"""End-to-end check: video -> Whisper (GPU) -> Sortformer (GPU, auto speakers) -> report.

python verify.py "C:\\path\\meeting.mp4" [--ref previous_transcript.txt] [--expect 2] [--engine sortformer|resnet]
Everything temporary/cached stays on D: (see transcribe.py).
"""
import argparse
import gc
import re
import sys
import time
from collections import Counter
from pathlib import Path

import diarize
from transcribe import TMP_DIR, add_cuda_dll_paths, fmt_ts, load_model


def hms(t):
    return fmt_ts(t).split(",")[0]


def toks(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ref", help="earlier transcript .txt of the same recording (text sanity check)")
    ap.add_argument("--expect", type=int, default=2, help="expected number of speakers (checked, not forced)")
    ap.add_argument("--engine", choices=("sortformer", "resnet"), default="sortformer")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "output"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    stem = Path(a.video).stem
    wav = str(Path(TMP_DIR) / f"{stem}.verify.wav")

    t = time.time()
    print("[1/4] extracting audio ->", wav, flush=True)
    diarize.extract_audio(a.video, wav)
    print(f"      {time.time() - t:.0f}s", flush=True)

    add_cuda_dll_paths()
    t = time.time()
    print("[2/4] Whisper on GPU …", flush=True)
    model, mode = load_model(a.model, "cuda")
    gen, info = model.transcribe(wav, beam_size=5, vad_filter=True,
                                 vad_parameters={"min_silence_duration_ms": 500}, word_timestamps=True)
    words, text = [], []
    for s in gen:
        text.append(s.text.strip())
        words += [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]
        print(f"\r      {hms(s.end)} / {hms(info.duration)}", end="", flush=True)
    print(f"\n      {len(words)} words, lang={info.language}, {time.time() - t:.0f}s on {mode}", flush=True)
    del model
    gc.collect()

    t = time.time()
    print(f"[3/4] speakers via {a.engine} (auto) …", flush=True)
    if a.engine == "sortformer":
        import sortformer
        segs, secs, turns = sortformer.run(wav, words, "cuda")
    else:
        samples = diarize.load_wav(wav)
        segs, secs, turns = diarize.run(samples, words, 0)
        del samples
    print(f"      {time.time() - t:.0f}s", flush=True)

    # ---- report
    print("\n[4/4] RESULT")
    total = sum(secs.values()) or 1
    print(f"  speakers detected (auto): {len(secs)}   expected: {a.expect}   "
          f"-> {'PASS' if len(secs) == a.expect else 'FAIL'}")
    for k in sorted(secs):
        nw = sum(len(s['text'].split()) for s in segs if s['spk'] == k)
        print(f"  Speaker {k + 1}: {hms(secs[k])} talk time ({secs[k] / total:5.1%}), {nw} words")
    sw = Counter(s["spk"] for s in segs)
    print("  segments per speaker:", dict(sorted(sw.items())))
    changes = sum(1 for x, y in zip(segs, segs[1:]) if x["spk"] != y["spk"])
    print(f"  speaker changes: {changes} over {hms(info.duration)}")
    # presence over time: both speakers should appear throughout, not just in one part
    thirds = [Counter() for _ in range(3)]
    for s in segs:
        thirds[min(int(3 * s["start"] / max(info.duration, 1)), 2)][s["spk"]] += s["end"] - s["start"]
    for i, c in enumerate(thirds, 1):
        print(f"  part {i}/3:", {f"S{k + 1}": hms(v) for k, v in sorted(c.items())})

    if a.ref and Path(a.ref).is_file():
        ref = Counter(toks(re.sub(r"^\[[0-9:]+\]\s*", "", Path(a.ref).read_text(encoding="utf-8"), flags=re.M)))
        new = Counter(toks(" ".join(text)))
        ov = sum((ref & new).values()) / max(sum(ref.values()), 1)
        print(f"  text vs reference: {ov:.1%} of reference words present in new transcript")

    lines = []
    for s in segs:
        lines.append(f"[{hms(s['start'])}] Speaker {s['spk'] + 1}: {s['text']}")
    dest = out / f"{stem}.speakers.txt"
    dest.write_text("\n\n".join(lines), encoding="utf-8")
    print("  saved:", dest)

    try:
        Path(wav).unlink()
    except OSError:
        pass
    sys.exit(0 if len(secs) == a.expect else 1)


if __name__ == "__main__":
    main()
