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

The backend is selected automatically at startup:

| Condition | Backend |
|---|---|
| `ANTHROPIC_API_KEY` is set | Claude (`claude-sonnet-4-6`) |
| Not set (default) | Ollama `gemma3:12b` at `localhost:11434` |

Copy `.env.example` → `.env` and set your key:

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
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
python train_voice.py --name myvoice --voice-dir ./voices/myvoice --language en
```

Options:
- `--skip-asr` — reuse existing transcripts (if ASR already ran)
- `--sovits-epochs N` — default 8 (enough for small datasets)
- `--gpt-epochs N` — default 15
- `--batch-size N` — reduce to 2 if VRAM OOM

Weights are saved to `~/GPT-SoVITS/SoVITS_weights_v2/` and `~/GPT-SoVITS/GPT_weights_v2/`.
