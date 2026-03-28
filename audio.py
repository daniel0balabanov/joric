"""Microphone recording (push-to-talk) and audio playback."""

import logging
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # Hz — Whisper native sample rate
CHANNELS = 1


def record_ptt() -> np.ndarray:
    """Record audio using push-to-talk.

    Press Enter to start recording, press Enter again to stop.
    Returns a float32 numpy array at 16 kHz.
    """
    chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            log.debug("sounddevice status: %s", status)
        chunks.append(indata.copy())

    print("  [Press Enter to start recording]", flush=True)
    input()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=_callback,
    )

    with stream:
        print("  [Recording... Press Enter to stop]", flush=True)
        input()

    if not chunks:
        return np.zeros(0, dtype="float32")

    audio = np.concatenate(chunks, axis=0).squeeze()
    log.debug("Recorded %.2f seconds", len(audio) / SAMPLE_RATE)
    return audio


def play_audio(audio: np.ndarray, samplerate: int) -> None:
    """Play audio array, blocking until playback finishes."""
    try:
        sd.play(audio, samplerate=samplerate)
        sd.wait()
    except Exception as exc:
        log.warning("Playback error: %s", exc)
