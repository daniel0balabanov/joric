#!/usr/bin/env python3
"""TTS test: type text → speak. No LLM, no microphone.

Usage:
    python test_chat.py
    python test_chat.py --debug
"""

import argparse
import logging
import sys

from config import TTSConfig, VOICE_PROFILES
from tts import TTS


def _pick(label: str, options: list[str], default: int = 0) -> int:
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


def main() -> None:
    p = argparse.ArgumentParser(description="joric TTS test")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    cfg = TTSConfig()

    print("\n=== joric TTS test ===")

    tts_backends = ["gptsovits", "omnivoice"]
    default_tts = tts_backends.index(cfg.backend) if cfg.backend in tts_backends else 0
    idx = _pick("TTS backend:", tts_backends, default=default_tts)
    cfg.backend = tts_backends[idx]

    voice_names = list(VOICE_PROFILES.keys())
    idx = _pick("Voice:", voice_names, default=0)
    cfg.ref_audio_path, cfg.prompt_text = VOICE_PROFILES[voice_names[idx]]

    tts = TTS(cfg)
    label = f"omnivoice ({cfg.omnivoice_model})" if cfg.backend == "omnivoice" else f"gptsovits @ {cfg.api_url}"
    print(f"\n  TTS : {label}")
    print(f"  Voice: {voice_names[idx]}")
    print("  Type text to speak. 'quit' or Ctrl-C to exit.\n")

    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break

            if not text:
                continue
            if text.lower() in {"quit", "exit"}:
                break

            print("  [speaking...]", end=" ", flush=True)
            tts.speak(text)
            print()

    except KeyboardInterrupt:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
