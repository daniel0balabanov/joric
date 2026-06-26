#!/usr/bin/env python3
"""joric — interactive voice chat pipeline.

Pipeline:
  Microphone (PTT) → Whisper STT → LLM → GPT-SoVITS TTS → Speaker

Usage:
    python main.py
    python main.py --no-tts        # text-only output
    python main.py --debug         # verbose logging
"""

import argparse
import logging
import os
import sys

from audio import record_ptt
from config import AppConfig, VOICE_PROFILES
from llm import LLMClient
from prompts import PERSONAS
from stt import Transcriber
from tts import TTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="joric voice chat")
    p.add_argument("--no-tts", action="store_true", help="Disable TTS (text output only)")
    p.add_argument("--no-menu", action="store_true", help="Skip setup menu, use defaults")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--model", help="Ollama model to use (e.g. gemma4:e4b, gemma3:12b)")
    return p.parse_args()


def _pick(label: str, options: list[str], default: int = 0) -> int:
    """Numbered menu — returns chosen index. Enter accepts default."""
    print(f"\n{label}")
    for i, opt in enumerate(options, 1):
        marker = "  *" if i - 1 == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        raw = input(f"  [1-{len(options)}, Enter={default + 1}] > ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"  Enter a number between 1 and {len(options)}.")


def _ollama_models(base_url: str) -> list[str]:
    """Return list of locally available Ollama model names."""
    try:
        import httpx
        base = base_url.replace("/v1", "").rstrip("/")
        resp = httpx.get(f"{base}/api/tags", timeout=3.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def setup_menu(cfg: AppConfig) -> None:
    """Interactive pre-launch configuration."""
    print("\n=== joric setup ===")

    # LLM backend
    backends: list[tuple[str, str]] = []
    if os.getenv("GOOGLE_API_KEY"):
        backends.append(("gemini", cfg.llm.gemini_model))
    if os.getenv("ANTHROPIC_API_KEY"):
        backends.append(("claude", cfg.llm.claude_model))
    backends.append(("ollama", cfg.llm.ollama_model))

    default_llm = next((i for i, (b, _) in enumerate(backends) if b == cfg.llm.backend), 0)
    idx = _pick("LLM backend:", [f"{b}  ({m})" for b, m in backends], default=default_llm)
    cfg.llm.backend = backends[idx][0]

    # Ollama model selection
    if cfg.llm.backend == "ollama":
        models = _ollama_models(cfg.llm.ollama_base_url)
        if models:
            default_model = models.index(cfg.llm.ollama_model) if cfg.llm.ollama_model in models else 0
            midx = _pick("Ollama model:", models, default=default_model)
            cfg.llm.ollama_model = models[midx]

    # Persona
    persona_names = list(PERSONAS.keys())
    current = os.getenv("PERSONA", "narrator")
    default_persona = persona_names.index(current) if current in persona_names else 0
    idx = _pick("Persona:", persona_names, default=default_persona)
    cfg.llm.system_prompt = PERSONAS[persona_names[idx]]

    # TTS backend
    if cfg.tts.enabled:
        tts_backends = ["gptsovits", "omnivoice"]
        default_tts = tts_backends.index(cfg.tts.backend) if cfg.tts.backend in tts_backends else 0
        idx = _pick("TTS backend:", tts_backends, default=default_tts)
        cfg.tts.backend = tts_backends[idx]

    # TTS voice (only if multiple profiles are defined)
    if cfg.tts.enabled and len(VOICE_PROFILES) > 1:
        voice_names = list(VOICE_PROFILES.keys())
        idx = _pick("TTS voice:", voice_names, default=0)
        cfg.tts.ref_audio_path, cfg.tts.prompt_text = VOICE_PROFILES[voice_names[idx]]

    print()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    cfg = AppConfig()
    if args.no_tts:
        cfg.tts.enabled = False
    if args.model:
        cfg.llm.ollama_model = args.model
        cfg.llm.backend = "ollama"

    if not args.no_menu:
        setup_menu(cfg)

    print("joric — voice chat")
    print(f"  STT : Whisper {cfg.stt.model_size}")
    print(f"  LLM : {cfg.llm.backend}  ({cfg.llm._active_model if hasattr(cfg.llm, '_active_model') else ''})")
    if cfg.tts.enabled:
        tts_label = f"omnivoice ({cfg.tts.omnivoice_model})" if cfg.tts.backend == "omnivoice" else f"gptsovits @ {cfg.tts.api_url}"
    else:
        tts_label = "disabled"
    print(f"  TTS : {tts_label}")
    print("  Type 'quit' or press Ctrl-C to exit.\n")

    print("Loading Whisper model...", end=" ", flush=True)
    transcriber = Transcriber(cfg.stt)
    transcriber._load()
    print("ready.")

    llm = LLMClient(cfg.llm)
    tts = TTS(cfg.tts)

    print("\nStarting conversation. Press Enter to speak.\n")

    try:
        while True:
            # 1. Record
            audio = record_ptt()
            if len(audio) == 0:
                print("  (no audio captured, try again)")
                continue

            # 2. Transcribe
            print("  [transcribing...]", end=" ", flush=True)
            text = transcriber.transcribe(audio)
            if not text:
                print("(silence detected, skipped)")
                continue
            print(f"\nYou: {text}")

            # Check for quit commands
            if text.strip().lower() in {"quit", "exit", "bye"}:
                print("Goodbye.")
                break

            # 3. LLM
            print("  [thinking...]", end=" ", flush=True)
            reply = llm.chat(text)
            print(f"\nAssistant: {reply}\n")

            # 4. TTS
            if cfg.tts.enabled:
                print("  [speaking...]", end=" ", flush=True)
                tts.speak(reply)
                print()

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
