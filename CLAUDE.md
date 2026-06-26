# joric

Interactive voice chat pipeline: speak → LLM replies → voice output.

## Pipeline

```
Microphone (PTT)
    → faster-whisper large-v3-turbo  (STT)
    → Gemini 2.0 Flash  OR  Claude  OR  Ollama gemma4:e4b  (LLM)
    → GPT-SoVITS v2  OR  OmniVoice  (TTS)
    → Speaker
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point — PTT loop + interactive setup menu |
| `audio.py` | `record_ptt()` + `play_audio()` |
| `stt.py` | `Transcriber` — faster-whisper wrapper |
| `llm.py` | `LLMClient` — Gemini, Claude, or Ollama |
| `tts.py` | `TTS` — GPT-SoVITS HTTP client or OmniVoice |
| `config.py` | `AppConfig`, `STTConfig`, `LLMConfig`, `TTSConfig`, `VOICE_PROFILES` |
| `prompts.py` | `PERSONAS` dict — narrator, warhammer, default |
| `train_voice.py` | Voice fine-tuning script for GPT-SoVITS |
| `pyproject.toml` | Dependencies |
| `Makefile` | Convenience targets |

## Running

```bash
# 1. Install deps
make venv

# 2. Start TTS server (separate terminal, only for gptsovits backend)
make tts-server

# 3. Start voice chat (interactive setup menu at launch)
make run

# Skip setup menu, use defaults
make run ARGS='--no-menu'

# Force a specific Ollama model
make run MODEL=gemma3:12b

# Text-only (no TTS)
make run ARGS='--no-tts'
```

Push-to-talk: press **Enter** to start recording, **Enter** again to stop.

### Setup menu

At startup an interactive menu lets you pick:
- **LLM backend** — Gemini / Claude / Ollama (only shows backends whose API key is set)
- **Ollama model** — auto-fetched from running Ollama instance
- **Persona** — narrator, warhammer, default
- **TTS backend** — gptsovits / omnivoice
- **TTS voice** — from `VOICE_PROFILES` in `config.py`

Skip it with `--no-menu`.

## LLM backend

Priority order (first matching key wins):

| Condition | Backend |
|---|---|
| `GOOGLE_API_KEY` is set | Gemini (`gemini-2.0-flash`) |
| `ANTHROPIC_API_KEY` is set | Claude (`claude-sonnet-4-6`) |
| Neither set (default) | Ollama `gemma4:e4b` at `localhost:11434` |

Copy `.env.example` → `.env` and set your key:

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY=... or ANTHROPIC_API_KEY=sk-ant-...
```

### Ollama GPU management

Two Makefile targets control which GPU Ollama uses (useful when GPT-SoVITS occupies GPU 0):

```bash
make ollama-gpu1   # pin Ollama to GPU 1 (RTX 5070), leaves GPU 0 for TTS
make ollama-cpu    # run Ollama on CPU, frees both GPUs for TTS
```

## Personas

Selectable via `PERSONA=` env var or the setup menu:

| Persona | Description |
|---|---|
| `narrator` | Darkest Dungeon narrator — Wayne June cadence, grim wisdom |
| `warhammer` | Warhammer 40K — same cadence, Imperial/grimdark flavour |
| `default` | Plain helpful assistant |

## Voice profiles

Defined in `VOICE_PROFILES` in `config.py`. Each entry maps a name to a `(ref_audio_path, transcript)` pair:

```python
VOICE_PROFILES = {
    "narrator": ("/path/to/ref.wav", "Great heroes can be found even here..."),
    "goblin":   ("/path/to/clip_0074.wav", "Кэнди из Дэнди, батликер из Квикер..."),
}
```

Active voice is selected at startup via the setup menu (or defaults to the first entry).

## GPT-SoVITS TTS

**Default voice**: Darkest Dungeon narrator (trained in the `arena` project).

Server lives at `~/GPT-SoVITS`. Start it with:

```bash
make tts-server
# or manually:
cd ~/GPT-SoVITS && .venv/bin/python api_v2.py -a 0.0.0.0 -p 9880
```

To use a custom trained voice, edit `~/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml` `custom` section before starting the server:

```yaml
custom:
  vits_weights_path: SoVITS_weights_v2/myvoice.pth
  t2s_weights_path: GPT_weights_v2/myvoice-e12.ckpt
  version: v2ProPlus
  device: cuda
  is_half: true
```

Then add an entry to `VOICE_PROFILES` in `config.py` pointing at a reference clip from that voice.

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

## OmniVoice TTS

Alternative TTS backend — runs locally, no server needed:

```bash
TTS_BACKEND=omnivoice make run
```

Configure via `.env`:
```
TTS_BACKEND=omnivoice
OMNIVOICE_MODEL=k2-fsa/OmniVoice
OMNIVOICE_DEVICE=cuda:0
```

## Training a new voice

Extract audio from a video and split into 15s clips:

```bash
ffmpeg -i myvideo.mp4 -vn -ar 32000 -ac 1 -c:a pcm_s16le \
  -f segment -segment_time 15 -reset_timestamps 1 \
  voices/myvoice/clip_%04d.wav
```

Then transcribe the first clip and add it to `VOICE_PROFILES` in `config.py`.

To fine-tune GPT-SoVITS weights from your clips:

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
