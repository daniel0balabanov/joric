# joric

Interactive voice chat pipeline: speak → LLM replies → voice output.

## Pipeline

```
Microphone (PTT)
    → faster-whisper large-v3-turbo  (STT)
    → Ollama gemma3:12b  OR  Claude  (LLM)
    → GPT-SoVITS v2 @ localhost:9880 (TTS)
    → Speaker
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point — main PTT loop |
| `audio.py` | `record_ptt()` + `play_audio()` |
| `stt.py` | `Transcriber` — faster-whisper wrapper |
| `llm.py` | `LLMClient` — Ollama or Claude |
| `tts.py` | `TTS` — GPT-SoVITS HTTP client |
| `config.py` | `AppConfig`, `STTConfig`, `LLMConfig`, `TTSConfig` |
| `train_voice.py` | Voice fine-tuning script for GPT-SoVITS |
| `pyproject.toml` | Dependencies |
| `Makefile` | Convenience targets |

## Running

```bash
# 1. Install deps
make venv

# 2. Start TTS server (separate terminal)
make tts-server

# 3. Start voice chat
make run

# Text-only (no TTS)
make run ARGS='--no-tts'
```

Push-to-talk: press **Enter** to start recording, **Enter** again to stop.

## LLM backend

The backend is selected automatically at startup (priority order):

| Condition | Backend |
|---|---|
| `GOOGLE_API_KEY` is set | Gemini (`gemini-2.0-flash`) |
| `ANTHROPIC_API_KEY` is set | Claude (`claude-sonnet-4-6`) |
| Neither set (default) | Ollama `gemma3:5b` at `localhost:11434` |

Copy `.env.example` → `.env` and set your key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
# or GOOGLE_API_KEY=... for Gemini
```

## GPT-SoVITS TTS

**Default voice**: Darkest Dungeon narrator (trained in the `arena` project).

Server lives at `~/GPT-SoVITS`. Start it with:

```bash
make tts-server
# or manually:
cd ~/GPT-SoVITS && .venv/bin/python api_v2.py -a 0.0.0.0 -p 9880
```

To use a custom trained voice, pass weight paths:

```bash
cd ~/GPT-SoVITS && .venv/bin/python api_v2.py -a 0.0.0.0 -p 9880 \
  -s SoVITS_weights_v2/myvoice.pth \
  -g GPT_weights_v2/myvoice-e12.ckpt
```

Then update `TTSConfig.ref_audio_path` and `TTSConfig.prompt_text` in `config.py`
to point at a reference clip from that voice.

**API reference** — POST `/tts`:
```json
{
  "text": "Hello world",
  "text_lang": "en",
  "ref_audio_path": "/abs/path/to/ref.wav",
  "prompt_lang": "en",
  "prompt_text": "Transcript of ref clip",
  "media_type": "wav",
  "streaming_mode": false
}
```

## Training a new voice

Place voice clips (WAV/OGG/MP3) in a directory, then:

```bash
make train-voice VOICE_DIR=./voices/myvoice VOICE_NAME=myvoice VOICE_LANG=en
# or directly:
python train_voice.py --name myvoice --voice-dir ./training/samples/... --language en
```

### Training pipeline

`train_voice.py` runs five stages inside `~/GPT-SoVITS`:

| Stage | Script | Output |
|---|---|---|
| 0 — Convert | ffmpeg | `processed_voices/<name>/wav32k/*.wav` (32kHz mono) |
| 1 — ASR | faster-whisper large-v3 | `<name>_list.txt` (path\|lang\|text per clip) |
| 2 — BERT | `1-get-text.py` | `2-name2text.txt` (phoneme features) |
| 3 — HuBERT | `2-get-hubert-wav32k.py` | HuBERT + wav32k acoustic features |
| 3b — SV | `2-get-sv.py` | Speaker-verification embeddings (v2ProPlus only) |
| 4 — Semantic | `3-get-semantic.py` | `6-name2semantic.tsv` (audio tokens) |
| 5a — SoVITS | `s2_train.py` | `SoVITS_weights_v2/<name>.pth` (voice timbre) |
| 5b — GPT | `s1_train.py` | `GPT_weights_v2/<name>-e*.ckpt` (prosody/rhythm) |

### Options

| Flag | Default | Description |
|---|---|---|
| `--sovits-epochs N` | 8 | SoVITS decoder training epochs |
| `--gpt-epochs N` | 15 | GPT model training epochs |
| `--batch-size N` | 4 | Reduce to 2 if VRAM OOM |
| `--save-every N` | 4 | Save checkpoint every N epochs |
| `--skip-asr` | — | Reuse existing transcript file |
| `--skip-preprocess` | — | Skip all preprocessing (stages 1–4) |
| `--skip-sovits` | — | Skip SoVITS training |
| `--skip-gpt` | — | Skip GPT training |

Weights are saved to `~/GPT-SoVITS/SoVITS_weights_v2/` and `~/GPT-SoVITS/GPT_weights_v2/`.
