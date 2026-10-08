"""Speaker diarization with NVIDIA NeMo Streaming Sortformer (GPU, max 4 speakers).

Model: nvidia/diar_streaming_sortformer_4spk-v2.1 (NVIDIA Open Model License), cached on D: under models/.
Output is a (frames x 4) matrix of speaker-activity probabilities, one frame = 80 ms.
Each Whisper word goes to the speaker with the highest mean probability over the word's frames,
so overlapped speech is resolved by confidence.
"""
import os

import numpy as np

import diarize
from diarize import SR, _clean_speakers, _overlaps, words_to_segments
from transcribe import MODEL_DIR

os.environ.setdefault("NEMO_CACHE_DIR", os.path.join(MODEL_DIR, "nemo"))

MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
MAX_SPEAKERS = 4
FRAME = 0.08         # seconds per output frame
ACTIVE = 0.5         # probability above which a speaker is "talking"
MIN_ON, MIN_OFF = 0.3, 0.5  # drop turns shorter than MIN_ON, bridge gaps shorter than MIN_OFF

_model = None


def load(device: str = "cuda"):
    global _model
    if _model is None:
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        m = SortformerEncLabelModel.from_pretrained(MODEL_ID)
        # "very high latency" config from the model card = best accuracy for offline files
        sm = m.sortformer_modules
        sm.chunk_len, sm.chunk_right_context, sm.fifo_len = 340, 40, 40
        sm.spkcache_update_period, sm.spkcache_len = 300, 188
        m.eval()
        if device == "cuda" and torch.cuda.is_available():
            m = m.cuda()
        _model = m
    return _model


def probabilities(wav_path: str, device: str = "cuda") -> np.ndarray:
    """(frames, 4) speaker-activity probabilities. Reads the 16 kHz mono wav itself (no numpy copy of the audio)."""
    import torch

    m = load(device)
    with torch.inference_mode():
        _, probs = m.diarize(audio=[wav_path], batch_size=1, include_tensor_outputs=True)
    p = probs[0] if isinstance(probs, (list, tuple)) else probs
    p = p.detach().float().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return p.reshape(-1, p.shape[-1])


def _turns(p: np.ndarray):
    """Per-speaker contiguous active regions -> [start, end, spk, mean_prob]."""
    turns = []
    for k in range(p.shape[1]):
        on = p[:, k] > ACTIVE
        runs, i, n = [], 0, len(on)
        while i < n:
            if on[i]:
                j = i
                while j < n and on[j]:
                    j += 1
                runs.append([i, j])
                i = j
            else:
                i += 1
        merged = []
        for a, b in runs:
            if merged and (a - merged[-1][1]) * FRAME < MIN_OFF:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if (b - a) * FRAME >= MIN_ON:
                turns.append([a * FRAME, b * FRAME, k, float(p[a:b, k].mean())])
    return sorted(turns)


# ---------------------------------------------------------------- voice-based identity
# Sortformer's 4 output slots drift on long recordings (one person ends up spread over several
# slots, and a slot can contain two people). Who is who is therefore decided by voice:
#   reliable turns -> mean-centred WeSpeaker embeddings -> k-means, k chosen automatically
#   -> every turn re-labelled by its nearest voice (slot is only a fallback for ambiguous ones).
VOICE_MIN_S = 1.5      # a turn must be this long (and not overlapped) to define a voice
VOICE_MIN_SHORT = 0.8  # slots with few such turns (people who mostly say "okay"/"yeah") also use turns this long
VOICE_FEW = 20         # "few" = fewer than this many reliable turns in a slot
VOICE_CAP_S = 8.0      # embed at most this much of a turn (centre crop)
VOICE_PER_SLOT = 80    # reliable turns per slot, spread across the recording
TURN_CAP_S = 3.0       # audio per turn when re-labelling every turn (cost grows with length)
KMAX = 4
SIL_MIN = 0.15         # best silhouette below this -> a single voice
SIL_TOL = 0.02         # prefer fewer speakers unless more is clearly better
MIN_MASS = 0.03        # a voice must hold >= 3% of the reliable talk time
MARGIN = 0.08          # nearest-voice must beat the runner-up by this to override the slot


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def _choose_voices(V, durs):
    """V: (n, d) unit embeddings. Returns (mu, centroids (k, d), labels) with k picked automatically."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    mu = V.mean(0)
    C = _unit(V - mu)
    cands = []
    for k in range(2, KMAX + 1):
        lab = KMeans(k, n_init=20, random_state=0).fit_predict(C)
        mass = np.array([durs[lab == j].sum() for j in range(k)]) / durs.sum()
        if len(set(lab)) == k and mass.min() >= MIN_MASS:
            cands.append((silhouette_score(C, lab), k, lab))
    if not cands or max(c[0] for c in cands) < SIL_MIN:
        return mu, _unit(C.mean(0, keepdims=True)), np.zeros(len(C), int), 0.0
    best = max(c[0] for c in cands)
    sc, k, lab = min((c for c in cands if c[0] >= best - SIL_TOL), key=lambda c: c[1])
    cen = _unit(np.stack([C[lab == j].mean(0) for j in range(k)]))
    return mu, cen, lab, sc


def _relabel_by_voice(samples, turns, p, progress=None):
    """turns: [s, e, slot, conf]. Returns (turns with voice id as slot, prob matrix per voice id, silhouette)."""
    ext = diarize._Embedder(samples, diarize.ensure_models())
    ov = [_overlaps(turns, i) for i in range(len(turns))]

    # 1) reliable turns per slot, spread across the whole recording
    rel, V, D, S = [], [], [], []
    for k in sorted({t[2] for t in turns}):
        idx = [i for i, t in enumerate(turns) if t[2] == k and t[1] - t[0] >= VOICE_MIN_S and not ov[i]]
        if len(idx) < VOICE_FEW:
            idx = [i for i, t in enumerate(turns) if t[2] == k and t[1] - t[0] >= VOICE_MIN_SHORT and not ov[i]]
        for i in idx[:: max(1, len(idx) // VOICE_PER_SLOT)][:VOICE_PER_SLOT]:
            v = ext(turns[i][0], turns[i][1], VOICE_CAP_S, pad=False)
            if v is not None:
                V.append(v), D.append(min(turns[i][1] - turns[i][0], VOICE_CAP_S)), S.append(k)
    if len(V) < 4:
        return turns, p, 0.0
    V, D, S = np.stack(V), np.array(D), np.array(S)
    mu, cen, lab, sil = _choose_voices(V, D)
    if len(cen) == 1:
        # Voices are inconclusive (e.g. a short clip where one person barely speaks). Over a short window
        # Sortformer's own slots are reliable, so keep them rather than collapsing everyone into one voice.
        return turns, p, sil
    if progress:
        progress(0.4)

    # slot -> voice (duration-weighted majority) as fallback for ambiguous / too-short turns
    slot2v = {k: int(np.bincount(lab[S == k], weights=D[S == k], minlength=len(cen)).argmax()) for k in set(S)}

    # 2) every turn -> nearest voice
    out, n = [], len(turns)
    for i, (s, e, k, conf) in enumerate(turns):
        v = ext(s, e, TURN_CAP_S, pad=False) if e - s >= 0.4 and not ov[i] else None
        vid, c = slot2v.get(k, 0), 0.0
        if v is not None:
            sims = cen @ _unit(v - mu)
            o = np.argsort(-sims)
            c = float(sims[o[0]] - (sims[o[1]] if len(o) > 1 else 0))
            if c >= MARGIN or len(cen) == 1:
                vid = int(o[0])
        out.append([s, e, vid, c, k])
        if progress and i % 50 == 0:
            progress(0.4 + 0.5 * i / n)

    # 3) per-frame probability per voice: strongest of the slots currently assigned to that voice
    p2 = np.zeros((p.shape[0], len(cen)), dtype=np.float32)
    for s, e, vid, _, k in out:
        a, b = int(s / FRAME), int(np.ceil(e / FRAME))
        p2[a:b, vid] = np.maximum(p2[a:b, vid], p[a:b, k])
    return [t[:4] for t in out], p2, sil


def _assign_words(words, p):
    last, out = 0, []
    for w in words:
        a = int(w["start"] / FRAME)
        b = max(a + 1, int(np.ceil(w["end"] / FRAME)))
        seg, w = p[a:b], dict(w)
        w["conf"] = 0.0
        if len(seg):
            mean = seg.mean(axis=0)
            k = int(mean.argmax())
            if mean[k] >= 0.2:  # someone is audibly talking -> most confident speaker (overlap rule)
                last, w["conf"] = k, float(mean[k])
        w["spk"] = last
        out.append(w)
    return out


def run(wav_path: str, words, device: str = "cuda", progress=None, voices: bool = True):
    """words: [{'start','end','word'}] from Whisper. Returns (segments, {spk: seconds}, turns)."""
    if progress:
        progress(0.05)
    p = probabilities(wav_path, device)
    if progress:
        progress(0.3)
    turns = _turns(p)
    if not turns:
        return [], {}, []
    sil = 0.0
    if voices:
        samples = diarize.load_wav(wav_path)
        turns, p, sil = _relabel_by_voice(samples, turns, p, lambda f: progress and progress(0.3 + 0.65 * f))
        del samples
    turns, secs, remap = _clean_speakers(turns)
    keep = sorted(remap, key=remap.get)
    p = p[:, keep]                       # columns in new speaker order (0 = most talkative)
    out = _assign_words(words, p)
    if progress:
        progress(1.0)
    return words_to_segments(out), secs, turns
