# meeting-transcriber

Local, GPU-accelerated **speech-to-text with speaker separation**. Works on any recording with one or more voices — meetings, interviews, calls, lectures, podcasts. Upload a video or audio file and get a readable transcript (`.txt`, `.srt`, `.docx`) where each speaker's speech is separated and named by you after listening to a short voice clip.

Everything runs on your own machine — no audio or text leaves it.

## How it works

```
media ──ffmpeg──▶ 16 kHz mono wav
                     │
                     ├─▶ faster-whisper (large-v3-turbo, GPU) ──▶ words + timestamps
                     │
                     └─▶ NVIDIA Streaming Sortformer v2.1 (GPU, ≤4 speakers) ──▶ speaker turns
                              │
                              └─▶ WeSpeaker ResNet293 voice re-labelling (CPU, sherpa-onnx)
                                  clusters mean-centred embeddings to correct slot drift
                                                   │
        words assigned to speakers by confidence ◀─┘ ──▶ you name each speaker ──▶ transcript
```

| Stage | Model | Runs on |
|---|---|---|
| Speech-to-text | `faster-whisper` large-v3-turbo (other sizes selectable) | GPU (CUDA) or CPU |
| Diarization (default) | NVIDIA `diar_streaming_sortformer_4spk-v2.1` via NeMo | GPU |
| Voice re-labelling | WeSpeaker ResNet293-LM via `sherpa-onnx` | CPU |
| Diarization (alternative) | pyannote-seg-3.0 + ResNet293 clustering, any number of speakers | CPU |

Sortformer alone can drift across its 4 speaker slots on long recordings; the voice re-labelling pass fixes this. When voice evidence is inconclusive (very short or quiet speech) it falls back to the Sortformer slots.

## Requirements

- Windows or Linux, Python 3.11
- [FFmpeg](https://ffmpeg.org/) on `PATH`
- NVIDIA GPU with ~8 GB VRAM recommended (CPU mode works but is much slower; Sortformer needs a GPU)
- Disk: ~2 GB for models, downloaded on first use

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch with CUDA (pick the build matching your driver: https://pytorch.org/get-started/locally/)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

All model weights, caches and temporary files are kept inside the project folder (`models/`, `.tmp/`) rather than the user profile. Both are git-ignored.

## Usage

### Web app

```bash
streamlit run app.py
```

1. Upload a file (up to 5 GB) or paste the path of a file already on disk.
2. Choose model, language (blank = auto-detect), task (transcribe / translate to English) and speaker engine.
3. Click **Transcribe**, then listen to each speaker's clip and enter a name.
4. Download the transcript as `.txt`, `.srt` or `.docx`.

### Command line

```bash
# Transcript only → .txt / .srt / .json
python transcribe.py recording.mp4 -m large-v3-turbo -l en -o out/

# End-to-end check with speaker detection and a report
python verify.py recording.mp4 --expect 2 --engine sortformer
```

`transcribe.py` options: `-m/--model`, `-l/--language`, `--task {transcribe,translate}`, `--device {cuda,cpu}`, `--no-vad`, `-o/--out`.
`verify.py` options: `--ref` (earlier transcript for a text sanity check), `--expect` (expected speaker count), `--engine {sortformer,resnet}`, `--model`, `--out`.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload → transcribe → name speakers → export |
| `transcribe.py` | Whisper loading, CUDA DLL setup, CLI |
| `sortformer.py` | Sortformer diarization + voice re-labelling + word assignment |
| `diarize.py` | ffmpeg audio extraction, ResNet clustering engine, shared helpers |
| `verify.py` | End-to-end check and report |

## Notes and limitations

- Sortformer supports at most 4 speakers per run; use the ResNet engine for recordings with more speakers.
- Speaker accuracy has been checked on real recordings but there is no labelled ground-truth benchmark; review output before relying on it.
- Recordings, transcripts and model files are never committed (see `.gitignore`).

## Licenses

Check the licenses of the models you download: faster-whisper / Whisper (MIT), NVIDIA Sortformer (NVIDIA Open Model License), pyannote segmentation 3.0 and WeSpeaker (see their model cards).
