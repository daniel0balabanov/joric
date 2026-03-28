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
import sys

from audio import record_ptt
from config import AppConfig
from llm import LLMClient
from stt import Transcriber
from tts import TTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="joric voice chat")
    p.add_argument("--no-tts", action="store_true", help="Disable TTS (text output only)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    cfg = AppConfig()
    if args.no_tts:
        cfg.tts.enabled = False

    print("joric — voice chat")
    print(f"  STT : Whisper {cfg.stt.model_size}")
    print(f"  LLM : {cfg.llm.backend}  ({cfg.llm._active_model if hasattr(cfg.llm, '_active_model') else ''})")
    print(f"  TTS : {'GPT-SoVITS @ ' + cfg.tts.api_url if cfg.tts.enabled else 'disabled'}")
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
