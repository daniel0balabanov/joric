"""Split a long audio file into fixed-length segments for TTS training.

Usage:
    python split_audio.py training/samples/gob/sherginHD.mp3
    python split_audio.py training/samples/gob/sherginHD.mp3 --segment 45 --out ./voices/gob
    python split_audio.py training/samples/gob/sherginHD.mp3 --min-silence 0.5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def split(src: Path, out_dir: Path, segment: int, min_silence: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())
    expected = int(duration / segment) + 1
    print(f"  Source  : {src.name}  ({duration/60:.1f} min)")
    print(f"  Segment : {segment}s  →  ~{expected} clips")
    print(f"  Output  : {out_dir}\n")

    out_pattern = str(out_dir / "clip_%04d.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", "32000",        # 32kHz mono — ready for GPT-SoVITS
        "-ac", "1",
        "-f", "segment",
        "-segment_time", str(segment),
        "-reset_timestamps", "1",
    ]

    if min_silence > 0:
        # Prefer cutting at silence boundaries near segment boundaries
        cmd += [
            "-segment_time_delta", str(min_silence),
        ]

    cmd.append(out_pattern)

    subprocess.run(cmd, check=True)

    clips = sorted(out_dir.glob("clip_*.wav"))
    # Remove clips shorter than 3s (usually the tail fragment)
    removed = 0
    for clip in clips:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(clip)],
            capture_output=True, text=True,
        )
        if probe.returncode == 0 and float(probe.stdout.strip()) < 3.0:
            clip.unlink()
            removed += 1

    final = sorted(out_dir.glob("clip_*.wav"))
    print(f"\nDone: {len(final)} clips written to {out_dir}  ({removed} short tail clips removed)")
    print(f"\nTo train:")
    name = out_dir.name
    print(f"  python train_voice.py --name {name} --voice-dir {out_dir} --language en")


def main() -> None:
    p = argparse.ArgumentParser(description="Split audio for GPT-SoVITS training")
    p.add_argument("src", help="Source audio file")
    p.add_argument("--segment", type=int, default=60, help="Segment length in seconds (default: 60)")
    p.add_argument("--out", help="Output directory (default: voices/<stem>)")
    p.add_argument("--min-silence", type=float, default=0.3,
                   help="Allow cut to drift this many seconds to find a silence boundary")
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path("voices") / src.stem
    split(src, out_dir, args.segment, args.min_silence)


if __name__ == "__main__":
    main()
