# FamilyFriend

**FamilyFriend** is a desktop application that automatically censors toxic speech in
audio. You load an audio file (or record from your microphone), and it transcribes the
speech, detects toxic/abusive/hateful content, and produces a censored version of the
audio using one of three strategies — from a simple beep up to a full AI rewrite that
re-synthesizes the speaker's own voice.

> **Scope (current version):** English-language audio only, and an NVIDIA GPU with CUDA
> is required. See [Requirements](#requirements).

---

## How it works

The pipeline runs entirely offline on your machine:

1. **Transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   (`large-v3`) transcribes the audio with word-level timestamps. A 16 kHz mono copy is
   made for Whisper while the working audio stays at its native sample rate, preserving
   output quality. A VAD (Voice Activity Detection) filter can be toggled in the UI.
2. **Toxicity detection** — [`unitary/unbiased-toxic-roberta`](https://huggingface.co/unitary/unbiased-toxic-roberta)
   scores the text across several labels (toxicity, identity attack, severe toxicity,
   obscenity, insult, threat, sexual content). Each label has its own sensitivity slider.
3. **Censoring** — one of three modes (below) edits only the toxic spans of the audio,
   splicing the result back into the original with equal-power crossfades so the
   surrounding audio and ambience are untouched.

### The three censor modes

| # | Mode | What it does | Engine |
|---|------|--------------|--------|
| 1 | **Beep** | Overlays a beep tone on each profane word found by a hard dictionary match. | Dictionary + DSP |
| 2 | **Synonym clone** | Replaces each profane word with a cleaner synonym, re-spoken in the speaker's own voice. | Dictionary + F5-TTS |
| 3 | **Contextual rewrite** | Detects hateful/abusive *sentences* with RoBERTa, then uses an LLM to rewrite them, neutralizing the hate, and re-synthesizes the speaker's voice. | RoBERTa + Qwen + F5-TTS |

Mode 4 ("all") produces all three outputs in one run.

### How the contextual rewrite (mode 3) works

This is the most involved mode:

- **Phrase grouping** — Whisper words are grouped into phrases at pauses and sentence
  punctuation.
- **Windowed detection** — each phrase is scored both as a whole *and* in overlapping
  ~10-word windows, so a short hateful span buried in a long sentence isn't diluted below
  the detection threshold.
- **Region coalescing** — contiguous toxic phrases are merged into a single region so
  the rewrite reads as one coherent passage.
- **Context-aware rewrite** — the LLM ([`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct))
  rewrites each region with its neighboring (non-toxic) sentences supplied as read-only
  context, so the rewrite flows into the surrounding narrative.
- **Verify loop** — the rewrite is cleaned, run through the dictionary again, and
  re-scored by RoBERTa. The first attempt is deterministic (greedy); if it's still toxic,
  it retries with sampling (up to 3 attempts), then falls back to a safe neutral sentence.
- **Voice reference** — F5-TTS clones the speaker using a short (~6 s) clip of the
  region's own audio as reference. Keeping the reference short avoids an F5 failure mode
  where long references get echoed/repeated in the output.

> Mode 3 is the newest and least battle-tested feature. It can occasionally leave a small
> artifact or, on very long rewrites, wander. Modes 1 and 2 are the most robust.

---

## Requirements

- **NVIDIA GPU with CUDA.** The app loads Whisper `large-v3`, RoBERTa, Qwen-3B and F5-TTS.
  Models are loaded on demand and the NLP models are offloaded to system RAM before
  F5-TTS runs to fit in VRAM, but a GPU with ~16 GB is recommended. **CPU is not
  supported** — the app shows an error if no CUDA device is found.
- **Python 3.12** (see `.python-version`).
- **First run downloads several GB** of model weights from Hugging Face — an internet
  connection and free disk space are required the first time.
- **Linux only:** install PortAudio before running:
  ```bash
  sudo apt update
  sudo apt install libportaudio2 portaudio19-dev
  ```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies into a local virtual environment
uv sync
```

## Running

```bash
uv run python main.py
```

This opens the FamilyFriend window. From there you can:

- **Carregar arquivo** — load an audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`).
- **Gravar microfone** — record from the microphone, then stop to process.

### Options in the UI

- **Modo de censura** — choose mode 1–4 (see table above).
- **Filtros Avançados** (sidebar) — per-label sensitivity sliders. A slider at its
  maximum (shown as "Desativado") turns that label off.
- **Salvar cópia do áudio original** — also save the untouched recording (mic only).
- **Salvar .txt com transcrições** — save original and censored transcripts.
- **Ativar VAD Filter** — voice-activity detection; disable it for noisy music.

### Output

Results are written to a subfolder (named after your chosen output file) containing the
censored audio, and optionally the original copy and the transcripts. Output preserves
the source sample rate and, where possible, bit depth.

---

## Project layout

| File | Responsibility |
|------|----------------|
| `main.py` | CustomTkinter GUI, audio capture/loading, and the processing orchestration. |
| `toxic_classifier.py` | `ToxicCensor` — RoBERTa detection (incl. windowing), the Qwen rewrite + verify loop, and the dictionary matcher. |
| `audio_censor.py` | `AudioCensor` — all DSP: beep generation, phrase grouping, boundary detection, crossfade splicing, and the offline voice-replacement pipeline. |
| `voice_cloner.py` | `VoiceCloner` — thin wrapper around F5-TTS for zero-shot voice synthesis. |
| `toxic_synonyms.py` | The profanity → synonym dictionary used by modes 1, 2 and the rewrite post-pass. |

---

## Limitations & notes

- **English only** by design — Whisper is pinned to English and the toxicity model and
  synonym dictionary are English.
- **NVIDIA/CUDA only** — no CPU fallback.
- The RoBERTa model is pinned to a specific PR revision (`refs/pr/4`) on Hugging Face for
  compatibility; this is intentional.
- Mode 3 (contextual rewrite) is experimental — see the note above.
