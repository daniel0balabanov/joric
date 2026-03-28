"""Fine-tune GPT-SoVITS v2 on a new speaker's voice clips.

Full pipeline:
  1. Convert audio clips to WAV 32kHz mono
  2. Run faster-whisper ASR to generate transcripts
  3. GPT-SoVITS preprocessing (BERT → HuBERT/wav32k → semantic tokens)
  4. Train SoVITS decoder
  5. Train GPT model

Trained weights land in:
  ~/GPT-SoVITS/SoVITS_weights_v2/<name>.pth
  ~/GPT-SoVITS/GPT_weights_v2/<name>-e*.ckpt

Usage:
    python train_voice.py --name myvoice --voice-dir ./voices/myvoice
    python train_voice.py --name myvoice --skip-asr    # reuse existing transcripts
    python train_voice.py --name myvoice --language ru

To use the trained voice, update TTSConfig in config.py:
    ref_audio_path = "/path/to/your/reference/clip.wav"
    prompt_text    = "Transcript of that clip"

Or pass --sovits-weights / --gpt-weights to api_v2.py:
    cd ~/GPT-SoVITS
    .venv/bin/python api_v2.py -a 0.0.0.0 -p 9880 \\
      -s SoVITS_weights_v2/<name>.pth \\
      -g GPT_weights_v2/<name>-e*.ckpt
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GPTSOVITS_DIR = Path.home() / "GPT-SoVITS"
PYTHON = str(GPTSOVITS_DIR / ".venv/bin/python")

PRETRAINED = GPTSOVITS_DIR / "GPT_SoVITS/pretrained_models"
BERT_DIR = str(PRETRAINED / "chinese-roberta-wwm-ext-large")
HUBERT_DIR = str(PRETRAINED / "chinese-hubert-base")
SV_PATH = str(PRETRAINED / "sv/pretrained_eres2netv2w24s4ep4.ckpt")
PRETRAINED_S2G = str(PRETRAINED / "gsv-v2final-pretrained/s2G2333k.pth")
PRETRAINED_S2D = str(PRETRAINED / "gsv-v2final-pretrained/s2D2333k.pth")
PRETRAINED_S1 = str(PRETRAINED / "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")

VERSION = "v2"
EXP_ROOT = str(GPTSOVITS_DIR / "logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: str = str(GPTSOVITS_DIR), env: dict | None = None) -> None:
    full_env = {**os.environ, **(env or {})}
    log.info("$ %s", cmd)
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=full_env)
    if result.returncode != 0:
        log.error("Command failed (exit %d)", result.returncode)
        sys.exit(result.returncode)


def _convert_to_wav32k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "32000", "-ac", "1", str(dst)],
        check=True, capture_output=True,
    )


# ---------------------------------------------------------------------------
# Step 0: prepare WAV files
# ---------------------------------------------------------------------------

def step_prepare_wavs(voice_dir: Path, wav_dir: Path) -> list[Path]:
    log.info("[0/5] Converting clips to WAV 32kHz …")
    wav_dir.mkdir(parents=True, exist_ok=True)
    audio_exts = {".ogg", ".oga", ".wav", ".mp3", ".flac", ".m4a"}
    clips = [p for p in sorted(voice_dir.iterdir()) if p.suffix.lower() in audio_exts]
    if not clips:
        log.error("No audio files found in %s", voice_dir)
        sys.exit(1)
    wavs = []
    for clip in clips:
        dst = wav_dir / (clip.stem + ".wav")
        if not dst.exists():
            _convert_to_wav32k(clip, dst)
            log.info("  converted %s", clip.name)
        wavs.append(dst)
    log.info("  %d clips ready in %s", len(wavs), wav_dir)
    return wavs


# ---------------------------------------------------------------------------
# Step 1: ASR → transcript list
# ---------------------------------------------------------------------------

def step_asr(wav_dir: Path, transcript_path: Path, language: str) -> None:
    log.info("[1/5] Running faster-whisper ASR …")
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    asr_script = GPTSOVITS_DIR / "tools/asr/fasterwhisper_asr.py"
    asr_output = transcript_path.parent / "asr_output"
    asr_output.mkdir(exist_ok=True)

    env = {
        "PYTHONPATH": f"{GPTSOVITS_DIR}:{GPTSOVITS_DIR}/GPT_SoVITS",
        "_CUDA_VISIBLE_DEVICES": "0",
    }
    _run(
        f'"{PYTHON}" "{asr_script}"'
        f' -i "{wav_dir}"'
        f' -o "{asr_output}"'
        f' -s large-v3'
        f' -l {language}'
        f' -p float16',
        env=env,
    )

    list_files = list(asr_output.glob("*.list"))
    if not list_files:
        log.error("ASR produced no .list files in %s", asr_output)
        sys.exit(1)

    lines = []
    for lf in sorted(list_files):
        lines += lf.read_text(encoding="utf-8").splitlines()

    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("  transcripts → %s  (%d lines)", transcript_path, len(lines))


# ---------------------------------------------------------------------------
# Steps 2–4: GPT-SoVITS preprocessing
# ---------------------------------------------------------------------------

def step_get_text(exp_name: str, transcript_path: Path, wav_dir: Path) -> None:
    log.info("[2/5] Extracting BERT features …")
    opt_dir = f"{EXP_ROOT}/{exp_name}"
    env = {
        "inp_text": str(transcript_path),
        "inp_wav_dir": str(wav_dir),
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "bert_pretrained_dir": BERT_DIR,
        "is_half": "True",
        "version": VERSION,
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": f"{GPTSOVITS_DIR}:{GPTSOVITS_DIR}/GPT_SoVITS",
    }
    _run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/1-get-text.py', env=env)
    part = Path(f"{opt_dir}/2-name2text-0.txt")
    merged = Path(f"{opt_dir}/2-name2text.txt")
    if part.exists():
        merged.write_text(part.read_text(encoding="utf-8"), encoding="utf-8")
        part.unlink()


def step_get_hubert(exp_name: str, transcript_path: Path, wav_dir: Path) -> None:
    log.info("[3/5] Extracting HuBERT + wav32k …")
    env = {
        "inp_text": str(transcript_path),
        "inp_wav_dir": str(wav_dir),
        "exp_name": exp_name,
        "opt_dir": f"{EXP_ROOT}/{exp_name}",
        "cnhubert_base_dir": HUBERT_DIR,
        "sv_path": SV_PATH,
        "is_half": "True",
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": f"{GPTSOVITS_DIR}:{GPTSOVITS_DIR}/GPT_SoVITS",
    }
    _run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', env=env)


def step_get_semantic(exp_name: str, transcript_path: Path) -> None:
    log.info("[4/5] Extracting semantic tokens …")
    opt_dir = f"{EXP_ROOT}/{exp_name}"
    env = {
        "inp_text": str(transcript_path),
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "pretrained_s2G": PRETRAINED_S2G,
        "s2config_path": str(GPTSOVITS_DIR / "GPT_SoVITS/configs/s2.json"),
        "is_half": "True",
        "i_part": "0",
        "all_parts": "1",
        "_CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": f"{GPTSOVITS_DIR}:{GPTSOVITS_DIR}/GPT_SoVITS",
    }
    _run(f'"{PYTHON}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py', env=env)
    part = Path(f"{opt_dir}/6-name2semantic-0.tsv")
    merged = Path(f"{opt_dir}/6-name2semantic.tsv")
    if part.exists():
        merged.write_text(part.read_text(encoding="utf-8"), encoding="utf-8")
        part.unlink()


# ---------------------------------------------------------------------------
# Steps 5a/5b: Training
# ---------------------------------------------------------------------------

def step_train_sovits(exp_name: str, batch_size: int, epochs: int, save_every: int) -> None:
    log.info("[5a/5] Training SoVITS decoder (%d epochs) …", epochs)
    exp_dir = f"{EXP_ROOT}/{exp_name}"
    os.makedirs(f"{exp_dir}/logs_s2_{VERSION}", exist_ok=True)
    os.makedirs(str(GPTSOVITS_DIR / "SoVITS_weights_v2"), exist_ok=True)

    with open(GPTSOVITS_DIR / "GPT_SoVITS/configs/s2.json") as f:
        cfg = json.load(f)

    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["epochs"] = epochs
    cfg["train"]["pretrained_s2G"] = PRETRAINED_S2G
    cfg["train"]["pretrained_s2D"] = PRETRAINED_S2D
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["save_every_epoch"] = save_every
    cfg["train"]["gpu_numbers"] = "0"
    cfg["train"]["grad_ckpt"] = True
    cfg["train"]["lora_rank"] = 128
    cfg["model"]["version"] = VERSION
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = exp_dir
    cfg["save_weight_dir"] = "SoVITS_weights_v2"
    cfg["name"] = exp_name
    cfg["version"] = VERSION

    tmp_cfg = GPTSOVITS_DIR / "tmp_s2.json"
    tmp_cfg.write_text(json.dumps(cfg))

    env = {"_CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(GPTSOVITS_DIR)}
    _run(f'"{PYTHON}" -s GPT_SoVITS/s2_train.py --config "{tmp_cfg}"', env=env)

    raw_ckpts = sorted(
        (GPTSOVITS_DIR / f"logs/{exp_name}/logs_s2_{VERSION}").glob("G_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if raw_ckpts:
        raw = raw_ckpts[-1]
        cfg_path = GPTSOVITS_DIR / f"logs/{exp_name}/config.json"
        hps = json.loads(cfg_path.read_text())
        ckpt = torch.load(str(raw), map_location="cpu", weights_only=False)
        opt: dict = OrderedDict()
        opt["weight"] = {k: v.half() for k, v in ckpt["model"].items() if "enc_q" not in k}
        opt["config"] = hps
        opt["info"] = f"{exp_name}_converted"
        out = GPTSOVITS_DIR / f"SoVITS_weights_v2/{exp_name}.pth"
        torch.save(opt, str(out))
        log.info("  SoVITS weight → %s", out)


def step_train_gpt(exp_name: str, batch_size: int, epochs: int, save_every: int) -> None:
    log.info("[5b/5] Training GPT model (%d epochs) …", epochs)
    exp_dir = f"{EXP_ROOT}/{exp_name}"
    os.makedirs(f"{exp_dir}/logs_s1_{VERSION}", exist_ok=True)
    os.makedirs(str(GPTSOVITS_DIR / "GPT_weights_v2"), exist_ok=True)

    with open(GPTSOVITS_DIR / "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["epochs"] = epochs
    cfg["pretrained_s1"] = PRETRAINED_S1
    cfg["train"]["save_every_n_epoch"] = save_every
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_dpo"] = False
    cfg["train"]["half_weights_save_dir"] = "GPT_weights_v2"
    cfg["train"]["exp_name"] = exp_name
    cfg["train_semantic_path"] = f"{exp_dir}/6-name2semantic.tsv"
    cfg["train_phoneme_path"] = f"{exp_dir}/2-name2text.txt"
    cfg["output_dir"] = f"{exp_dir}/logs_s1_{VERSION}"

    tmp_cfg = GPTSOVITS_DIR / "tmp_s1.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    env = {
        "_CUDA_VISIBLE_DEVICES": "0",
        "hz": "25hz",
        "PYTHONPATH": f"{GPTSOVITS_DIR}:{GPTSOVITS_DIR}/GPT_SoVITS",
    }
    _run(f'"{PYTHON}" -s GPT_SoVITS/s1_train.py --config_file "{tmp_cfg}"', env=env)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fine-tune GPT-SoVITS v2 on a new speaker voice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", required=True, help="Speaker / experiment name (e.g. myvoice)")
    p.add_argument("--voice-dir", required=True,
                   help="Directory containing voice clips (.wav/.ogg/.mp3/…)")
    p.add_argument("--language", default="en",
                   help="Language code for ASR transcription (en, ru, zh, …)")
    p.add_argument("--sovits-epochs", type=int, default=8)
    p.add_argument("--gpt-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Reduce to 2 if VRAM OOM")
    p.add_argument("--save-every", type=int, default=4,
                   help="Save checkpoint every N epochs")
    p.add_argument("--skip-asr", action="store_true",
                   help="Skip ASR — reuse existing transcript file")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip all preprocessing (assumes it already ran)")
    p.add_argument("--skip-sovits", action="store_true")
    p.add_argument("--skip-gpt", action="store_true")
    args = p.parse_args()

    voice_dir = Path(args.voice_dir)
    if not voice_dir.exists():
        log.error("Voice directory not found: %s", voice_dir)
        sys.exit(1)

    exp_name = args.name
    processed_dir = Path(__file__).parent / "processed_voices" / exp_name
    wav_dir = processed_dir / "wav32k"
    transcript_path = processed_dir / f"{exp_name}_list.txt"

    log.info("Experiment : %s", exp_name)
    log.info("Voice dir  : %s", voice_dir)
    log.info("WAV dir    : %s", wav_dir)

    step_prepare_wavs(voice_dir, wav_dir)

    if not args.skip_preprocess:
        if not args.skip_asr:
            step_asr(wav_dir, transcript_path, args.language)
        else:
            if not transcript_path.exists():
                log.error("--skip-asr set but transcript not found: %s", transcript_path)
                sys.exit(1)
            log.info("[1/5] Skipping ASR, using %s", transcript_path)

        step_get_text(exp_name, transcript_path, wav_dir)
        step_get_hubert(exp_name, transcript_path, wav_dir)
        step_get_semantic(exp_name, transcript_path)
    else:
        log.info("[1–4/5] Skipping preprocessing.")

    if not args.skip_sovits:
        step_train_sovits(exp_name, args.batch_size, args.sovits_epochs, args.save_every)
    if not args.skip_gpt:
        step_train_gpt(exp_name, args.batch_size, args.gpt_epochs, args.save_every)

    log.info("")
    log.info("Done! Trained weights:")
    log.info("  %s/SoVITS_weights_v2/%s.pth", GPTSOVITS_DIR, exp_name)
    log.info("  %s/GPT_weights_v2/%s-e*.ckpt", GPTSOVITS_DIR, exp_name)
    log.info("")
    log.info("To use the new voice, update TTSConfig.ref_audio_path in config.py")
    log.info("and start the TTS server with:")
    log.info("  cd ~/GPT-SoVITS && .venv/bin/python api_v2.py -a 0.0.0.0 -p 9880 \\")
    log.info("    -s SoVITS_weights_v2/%s.pth \\", exp_name)
    log.info("    -g GPT_weights_v2/%s-e*.ckpt", exp_name)


if __name__ == "__main__":
    main()
