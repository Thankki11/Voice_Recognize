"""
Feature extraction pipeline — 1 file WAV = 1 vector 23 chieu.

Columns (metadata + 23 features):
  Metadata : file_path, audio_group, filename, sentence_id, variant
  MFCC     : mfcc_mean_0..12                          (13 chieu)
  Pitch    : f0_mean, f0_std, voiced_ratio             ( 3 chieu)
  Quality  : jitter, shimmer, hnr                      ( 3 chieu)
  Time     : zcr_mean, zcr_std, rms_mean, rms_std      ( 4 chieu)
  Total    : 23 chieu
"""

import re
import sys
import csv
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.mfcc import extract_mfcc
from features.pitch import extract_pitch
from features.time_domain import extract_time_domain
from features.voice_quality import extract_voice_quality

DATA_STORE = PROJECT_ROOT / "data" / "raw" / "data_store"
OUT_CSV    = PROJECT_ROOT / "features" / "segments_all.csv"
N_MFCC     = 13
HOP_LENGTH = 512
FRAME_LEN  = 2048

_PAT = re.compile(r"^arctic_a(\d+)(?:\((\d+)\))?\.wav$", re.IGNORECASE)

FIELDNAMES = (
    ["file_path", "audio_group", "filename", "sentence_id", "variant"]
    + [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    + ["f0_mean", "f0_std", "voiced_ratio",
       "jitter", "shimmer", "hnr",
       "zcr_mean", "zcr_std", "rms_mean", "rms_std"]
)


def _parse_name(filename: str):
    m = _PAT.match(filename)
    if not m:
        return None, None
    return m.group(1), m.group(2) or "0"


def _fmt(v, decimals=6):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    return round(float(v), decimals)


def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Nhan vao mang bien do va sample rate cua 1 file da trim.
    Tra ve dict 23 chieu dac trung.
    """
    mfcc   = extract_mfcc(y, sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    pitch  = extract_pitch(y, sr, hop_length=HOP_LENGTH)
    td     = extract_time_domain(y, sr, hop_length=HOP_LENGTH)
    vq     = extract_voice_quality(y, sr, hop_length=HOP_LENGTH)

    row = {}

    # MFCC mean x13
    for i in range(N_MFCC):
        row[f"mfcc_mean_{i}"] = _fmt(mfcc[i].mean())

    # Pitch x3
    row["f0_mean"]      = _fmt(pitch["f0_mean"], 3)
    row["f0_std"]       = _fmt(pitch["f0_std"],  3)
    row["voiced_ratio"] = _fmt(pitch["voiced_ratio"], 4)

    # Voice quality x3
    row["jitter"]  = _fmt(vq["jitter"],  5)
    row["shimmer"] = _fmt(vq["shimmer"], 5)
    row["hnr"]     = _fmt(vq["hnr"],     3)

    # Time domain x4
    row["zcr_mean"] = _fmt(td["zcr_mean"])
    row["zcr_std"]  = _fmt(td["zcr_std"])
    row["rms_mean"] = _fmt(td["rms_mean"])
    row["rms_std"]  = _fmt(td["rms_std"])

    return row


def run(data_dir: Path = DATA_STORE, out_csv: Path = OUT_CSV):
    wav_files = sorted(data_dir.rglob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {data_dir}")
        return

    total = 0
    skipped = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for wav_path in tqdm(wav_files, desc="Extracting", unit="file"):
            filename    = wav_path.name
            sentence_id, variant = _parse_name(filename)
            if sentence_id is None:
                skipped += 1
                continue

            try:
                y, sr = librosa.load(str(wav_path), sr=None)
            except Exception as e:
                tqdm.write(f"SKIP {filename}: {e}")
                skipped += 1
                continue

            # Trim padding dau/cuoi, giu nguyen pause giua cac tu
            y, _ = librosa.effects.trim(y, top_db=20)

            if len(y) < HOP_LENGTH:
                tqdm.write(f"SKIP {filename}: too short after trim")
                skipped += 1
                continue

            feats = extract_features(y, sr)

            row = {
                "file_path":   str(wav_path.relative_to(PROJECT_ROOT)),
                "audio_group": wav_path.parent.name,
                "filename":    filename,
                "sentence_id": sentence_id,
                "variant":     variant,
                **feats,
            }
            writer.writerow(row)
            total += 1

    print(f"\nDone. {total} files -> {out_csv}")
    if skipped:
        print(f"Skipped: {skipped} files")


if __name__ == "__main__":
    run()
