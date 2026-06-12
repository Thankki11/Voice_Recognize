"""
save_clean_audio.py
-------------------
Chi lam sach (load + trim silence) va luu cac file WAV sang thu muc data_clean/
KHONG trich xuat lai dac trung — chay nhanh hon nhieu.

Cach chay:
    python voice_search_project/features/save_clean_audio.py
"""

import sys
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_STORE    = PROJECT_ROOT / "data" / "raw" / "data_store"
PROCESSED_DIR = PROJECT_ROOT / "data" / "data_clean" / "data_store"
HOP_LENGTH    = 512


def main():
    wav_files = sorted(DATA_STORE.rglob("*.wav"))
    if not wav_files:
        print(f"Khong tim thay file WAV trong {DATA_STORE}")
        return

    print(f"Tong file       : {len(wav_files)}")
    print(f"Luu vao         : {PROCESSED_DIR}")
    print(f"Trim tieu chi   : top_db=20 (RMS < 0.1 x max_RMS)\n")

    saved = 0
    skipped = 0

    for wav_path in tqdm(wav_files, desc="Cleaning", unit="file"):
        try:
            y, sr = librosa.load(str(wav_path), sr=None)
            y, _  = librosa.effects.trim(y, top_db=20)

            if len(y) < HOP_LENGTH:
                tqdm.write(f"SKIP {wav_path.name}: too short after trim")
                skipped += 1
                continue

            rel_path = wav_path.relative_to(DATA_STORE)
            out_wav  = PROCESSED_DIR / rel_path
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav), y, sr)
            saved += 1

        except Exception as e:
            tqdm.write(f"SKIP {wav_path.name}: {e}")
            skipped += 1

    print(f"\nHoan tat.")
    print(f"  Da luu  : {saved} file")
    print(f"  Bo qua  : {skipped} file")
    print(f"  Thu muc : {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
