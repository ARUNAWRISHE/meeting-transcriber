"""GPU video -> transcript using faster-whisper (CTranslate2 + CUDA).

Usage:
    python transcribe.py video.mp4 [-m large-v3-turbo] [-l en] [--task translate]
Outputs <video>.txt, <video>.srt, <video>.json next to the video (or in --out).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Keep all model files on D: (next to this script) instead of C:\Users\...\.cache
_ROOT = Path(__file__).resolve().parent
MODEL_DIR = str(_ROOT / "models")
TMP_DIR = str(_ROOT / ".tmp")
Path(TMP_DIR).mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", MODEL_DIR)
os.environ.setdefault("TORCH_HOME", str(Path(MODEL_DIR) / "torch"))
os.environ.setdefault("NEMO_CACHE_DIR", str(Path(MODEL_DIR) / "nemo"))
os.environ["TMP"] = os.environ["TEMP"] = os.environ["TMPDIR"] = TMP_DIR  # nothing temp goes to C:
import tempfile  # noqa: E402

tempfile.tempdir = TMP_DIR

def add_cuda_dll_paths():
    """Windows: make pip-installed CUDA libs (nvidia-cublas-cu12, nvidia-cudnn-cu12) findable."""
    if sys.platform != "win32":
        return
    import importlib.util
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ModuleNotFoundError, ValueError):
            continue
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                for sub in ("bin", "lib"):
                    p = Path(loc) / sub
                    if p.is_dir():
                        os.add_dll_directory(str(p))
                        os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def load_model(name: str, device: str):
    from faster_whisper import WhisperModel

    if device == "cpu":
        return WhisperModel(name, device="cpu", compute_type="int8", download_root=MODEL_DIR), "cpu/int8"
    # float16 is fastest; int8_float16 uses less VRAM (8 GB card).
    for ct in ("float16", "int8_float16"):
        try:
            return WhisperModel(name, device="cuda", compute_type=ct, download_root=MODEL_DIR), f"cuda/{ct}"
        except Exception as e:  # CUDA libs missing / unsupported arch / OOM
            print(f"[warn] cuda/{ct} failed: {e}", file=sys.stderr)
    raise RuntimeError("GPU init failed: install CUDA libs / update ctranslate2, or use --device cpu.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-m", "--model", default="large-v3-turbo",
                    help="tiny/base/small/medium/large-v3/large-v3-turbo/distil-large-v3 (default: large-v3-turbo)")
    ap.add_argument("-l", "--language", default=None, help="e.g. en, ta, hi (default: auto-detect)")
    ap.add_argument("--task", choices=("transcribe", "translate"), default="transcribe",
                    help="translate = output English")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--no-vad", action="store_true", help="disable silence filtering")
    ap.add_argument("-o", "--out", default=None, help="output folder (default: video folder)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        sys.exit(f"File not found: {video}")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not on PATH")

    out_dir = Path(args.out) if args.out else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / video.stem

    add_cuda_dll_paths()
    try:
        model, mode = load_model(args.model, args.device)
    except RuntimeError as e:
        sys.exit(str(e))
    print(f"[info] model={args.model} mode={mode}")

    t0 = time.time()
    # faster-whisper decodes via bundled PyAV, so video files work directly.
    segments, info = model.transcribe(
        str(video),
        language=args.language,
        task=args.task,
        beam_size=5,
        vad_filter=not args.no_vad,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"[info] language={info.language} (p={info.language_probability:.2f}) duration={info.duration:.0f}s")

    rows = []
    with open(f"{stem}.txt", "w", encoding="utf-8") as ftxt, \
         open(f"{stem}.srt", "w", encoding="utf-8") as fsrt:
        for i, seg in enumerate(segments, 1):  # lazy generator: decoding happens here
            text = seg.text.strip()
            ftxt.write(text + "\n")
            fsrt.write(f"{i}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{text}\n\n")
            rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
            print(f"[{fmt_ts(seg.start)}] {text}")
            if seg.end and info.duration:
                print(f"\r  {seg.end / info.duration:5.1%}", end="\r", file=sys.stderr)

    with open(f"{stem}.json", "w", encoding="utf-8") as fj:
        json.dump({"language": info.language, "segments": rows}, fj, ensure_ascii=False, indent=2)

    el = time.time() - t0
    print(f"\n[done] {el:.1f}s ({info.duration / max(el, 1e-6):.1f}x realtime) -> {stem}.txt/.srt/.json")


if __name__ == "__main__":
    main()
