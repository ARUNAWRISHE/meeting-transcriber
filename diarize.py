"""Speaker diarization (who spoke when) with sherpa-onnx. Token-free, CPU, models live on D:.

Pipeline:
  1. pyannote-seg-3.0 splits audio into speaker turns, voice embeddings are clustered -> speakers.
  2. Each speaker gets a voice profile (centroid of their clean, non-overlapping turns).
  3. Every turn is re-scored against all profiles and moved if it matches another speaker better.
  4. Whisper words are assigned to turns; where two speakers overlap, the word goes to the
     speaker whose profile matches that overlapped audio with the highest confidence (cosine).

Models (downloaded once into models/diarization, from github.com/k2-fsa/sherpa-onnx releases):
  - pyannote segmentation 3.0                 ~7 MB
  - WeSpeaker ResNet293-LM                 ~114 MB
"""
import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

from transcribe import MODEL_DIR

DIA_DIR = Path(MODEL_DIR) / "diarization"
REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
SEG_URL = REL + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEG_MODEL = DIA_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
# WeSpeaker ResNet293-LM: most accurate English voice model in sherpa-onnx (VoxCeleb EER ~0.45%).
# CAM++ was tested and rejected: it mixed up speakers at every clustering threshold.
EMB_MODELS = {"best": "wespeaker_en_voxceleb_resnet293_LM.onnx"}
SR = 16000
THREADS = max(1, (os.cpu_count() or 4) // 2)
MIN_SPEAKER_SECONDS = 3.0  # speakers with less total speech are treated as noise
MIN_EMB_SECONDS = 1.0      # shorter audio gives unreliable voice embeddings
REASSIGN_MARGIN = 0.05     # move a turn only if another profile matches clearly better


def _download(url: str, dest: Path):
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part)
    part.replace(dest)


def ensure_models(model: str = "best") -> Path:
    DIA_DIR.mkdir(parents=True, exist_ok=True)
    if not SEG_MODEL.is_file():
        tmp = DIA_DIR / "seg.tar.bz2"
        _download(SEG_URL, tmp)
        with tarfile.open(tmp, "r:bz2") as t:
            t.extractall(DIA_DIR, filter="data")
        tmp.unlink()
    emb = DIA_DIR / EMB_MODELS[model]
    if not emb.is_file():
        _download(REL + "speaker-recongition-models/" + emb.name, emb)
    return emb


def extract_audio(media: str, wav_path: str):
    """Any video/audio -> 16 kHz mono PCM wav."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", media, "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav_path],
        capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-400:])


def load_wav(path: str) -> np.ndarray:
    """int16 samples memory-mapped from the wav on D: (no RAM copy; slice + to_f32() what you need)."""
    with wave.open(path, "rb") as w:
        n, ch, width = w.getnframes(), w.getnchannels(), w.getsampwidth()
    assert ch == 1 and width == 2, "expected 16-bit mono wav"
    with open(path, "rb") as f:
        head = f.read(4096)
    off = head.index(b"data") + 8  # start of PCM data
    return np.memmap(path, dtype=np.int16, mode="r", offset=off, shape=(n,))


def to_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 32768.0 if x.dtype == np.int16 else np.asarray(x, dtype=np.float32)


# ------------------------------------------------------------------ step 1: segmentation + clustering
def _cluster(samples, emb_model: Path, num_speakers: int, threshold: float, progress):
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise RuntimeError("Invalid diarization config / model files")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def cb(done, total, *_):
        if progress:
            progress(0.7 * done / max(total, 1))
        return 0

    res = sd.process(to_f32(samples), callback=cb).sort_by_start_time()
    return [[r.start, r.end, r.speaker] for r in res]


# ------------------------------------------------------------------ step 2-3: voice profiles + re-check
class _Embedder:
    def __init__(self, samples, emb_model: Path):
        import sherpa_onnx

        self.samples = samples
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=THREADS))

    def __call__(self, start: float, end: float, max_seconds: float = 20.0, pad: bool = True):
        """L2-normalised embedding of [start, end] (padded to MIN_EMB_SECONDS unless pad=False; centre-cropped if long)."""
        mid, dur = (start + end) / 2, min(max(end - start, MIN_EMB_SECONDS if pad else 0.4), max_seconds)
        a = max(0, int((mid - dur / 2) * SR))
        b = min(len(self.samples), int((mid + dur / 2) * SR))
        if b - a < int(0.5 * SR):
            return None
        st = self.ext.create_stream()
        st.accept_waveform(sample_rate=SR, waveform=to_f32(self.samples[a:b]))
        st.input_finished()
        if not self.ext.is_ready(st):
            return None
        v = np.asarray(self.ext.compute(st), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else None


def _overlaps(turns, i):
    s, e = turns[i][0], turns[i][1]
    return any(j != i and t[0] < e and t[1] > s for j, t in enumerate(turns))


def _profiles(turns, embs):
    """Speaker -> centroid of clean (non-overlapping, >= MIN_EMB_SECONDS) turns, weighted by duration."""
    acc = {}
    for i, (s, e, k) in enumerate(turns):
        v = embs[i]
        if v is None:
            continue
        clean = e - s >= MIN_EMB_SECONDS and not _overlaps(turns, i)
        acc.setdefault(k, {"clean": [], "all": []})
        acc[k]["all"].append(v * (e - s))
        if clean:
            acc[k]["clean"].append(v * (e - s))
    prof = {}
    for k, d in acc.items():
        c = np.sum(d["clean"] or d["all"], axis=0)
        prof[k] = c / (np.linalg.norm(c) or 1)
    return prof


def _refine(turns, embs, prof):
    """Re-score every turn against all voice profiles; return [s, e, k, confidence]."""
    keys = list(prof)
    P = np.stack([prof[k] for k in keys])
    out = []
    for (s, e, k), v in zip(turns, embs):
        if v is None:
            out.append([s, e, k, 0.0])
            continue
        sims = P @ v
        best = int(np.argmax(sims))
        own = sims[keys.index(k)] if k in prof else -1.0
        if keys[best] != k and sims[best] - own > REASSIGN_MARGIN and e - s >= MIN_EMB_SECONDS:
            k, own = keys[best], sims[best]
        out.append([s, e, k, float(own)])
    return out


def _clean_speakers(turns):
    """Drop near-silent 'speakers', renumber by talk time (most talkative = 0)."""
    tot = {}
    for t in turns:
        tot[t[2]] = tot.get(t[2], 0) + (t[1] - t[0])
    keep = {k for k, v in tot.items() if v >= MIN_SPEAKER_SECONDS} or {max(tot, key=tot.get)}
    remap = {k: i for i, k in enumerate(sorted(keep, key=lambda k: -tot[k]))}
    turns = [[t[0], t[1], remap[t[2]], *t[3:]] for t in turns if t[2] in remap]
    return turns, {remap[k]: tot[k] for k in keep}, remap


# ------------------------------------------------------------------ step 4: words -> speakers
def _assign_words(words, turns, prof, embed):
    """Give each word a speaker. Overlap -> speaker whose voice profile best matches that audio."""
    cache, last, j0 = {}, 0, 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        while j0 < len(turns) and turns[j0][1] < mid - 30:  # turns sorted by start; skip far past ones
            j0 += 1
        cands = []
        for t in turns[j0:]:
            if t[0] > mid:
                break
            if mid <= t[1]:
                cands.append(t)
        ks = sorted({t[2] for t in cands})
        if len(ks) == 1:
            last = ks[0]
            w["conf"] = max(t[3] for t in cands)
        elif len(ks) > 1:
            s, e = max(t[0] for t in cands), min(t[1] for t in cands)  # overlapped region
            key = (round(s, 2), round(e, 2))
            if key not in cache:
                v = embed(s, e)
                sims = {k: float(prof[k] @ v) if v is not None and k in prof else -1.0 for k in ks}
                cache[key] = max(sims.items(), key=lambda x: x[1])
            last, w["conf"] = cache[key]
        else:  # word falls in a gap between turns -> nearest turn within 1 s, else previous speaker
            near = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)), default=None)
            if near and min(abs(near[0] - mid), abs(near[1] - mid)) < 1.0:
                last = near[2]
            w["conf"] = 0.0
        w["spk"] = last
    return words


def words_to_segments(words, max_chars: int = 120, gap: float = 1.0):
    """Group consecutive same-speaker words into subtitle-sized segments."""
    segs, cur = [], None
    for w in words:
        txt = w["word"]
        new = (cur is None or w["spk"] != cur["spk"] or w["start"] - cur["end"] > gap
               or len(cur["text"]) > max_chars
               or (cur["text"].rstrip()[-1:] in ".?!" and len(cur["text"]) > 40))
        if new:
            cur = {"start": w["start"], "end": w["end"], "text": txt.strip(), "spk": w["spk"], "_c": [w["conf"]]}
            segs.append(cur)
        else:
            cur["text"] += txt
            cur["end"] = w["end"]
            cur["_c"].append(w["conf"])
    for s in segs:
        c = s.pop("_c")
        s["conf"] = round(sum(c) / len(c), 3)
        s["text"] = s["text"].strip()
    return segs


# ------------------------------------------------------------------ public entry point
def run(samples, words, num_speakers: int = 0, model: str = "best", threshold: float = 0.5, progress=None):
    """words: [{'start','end','word'}] from Whisper. Returns (segments, {spk: seconds}, turns)."""
    emb_model = ensure_models(model)
    turns = _cluster(samples, emb_model, num_speakers, threshold, progress)
    if not turns:
        return [], {}, []

    embed = _Embedder(samples, emb_model)
    embs = []
    for i, (s, e, _) in enumerate(turns):
        embs.append(embed(s, e))
        if progress and i % 10 == 0:
            progress(0.7 + 0.25 * i / len(turns))
    prof = _profiles(turns, embs)
    turns = [t + [v] for t, v in zip(_refine(turns, embs, prof), embs)]  # [s, e, k, conf, emb]
    turns, secs, _ = _clean_speakers(turns)
    prof = _profiles([t[:3] for t in turns], [t[4] for t in turns])  # profiles after re-check
    turns = [t[:4] for t in turns]

    words = _assign_words([dict(w) for w in words], turns, prof, embed)
    if progress:
        progress(1.0)
    return words_to_segments(words), secs, turns


def speaker_samples(samples: np.ndarray, turns, max_seconds: float = 12.0) -> dict:
    """WAV bytes per speaker: their most confident, clean turns joined, up to max_seconds."""
    by = {}
    for i, t in enumerate(turns):
        s, e, k = t[:3]
        conf = t[3] if len(t) > 3 else 0.0
        clean = not _overlaps(turns, i)
        by.setdefault(k, []).append((clean, conf * min(e - s, 6), s, e))
    out = {}
    for k, lst in by.items():
        parts, total = [], 0.0
        for _, _, s, e in sorted(lst, reverse=True):
            if total >= max_seconds:
                break
            d = min(e - s, max_seconds - total)
            parts.append(to_f32(samples[int(s * SR): int((s + d) * SR)]))
            parts.append(np.zeros(int(0.3 * SR), dtype=np.float32))
            total += d
        pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        out[k] = buf.getvalue()
    return out
