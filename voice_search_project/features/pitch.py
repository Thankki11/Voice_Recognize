

import numpy as np
import librosa


def extract_pitch(
    y: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 600.0,
    hop_length: int = 512,
) -> dict:
    
    if len(y) < 2048:
        return _empty_result(np.array([]))

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
        )
    except Exception:
        return _empty_result(np.array([]))

    # pyin returns NaN for unvoiced frames; replace with 0 for storage
    f0_clean = np.where(np.isnan(f0), 0.0, f0)
    voiced_f0 = f0[voiced_flag]

    if voiced_f0.size == 0 or np.all(np.isnan(voiced_f0)):
        return _empty_result(f0_clean)

    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    voiced_ratio = float(voiced_flag.sum()) / len(voiced_flag)

    return {
        "f0_mean": float(voiced_f0.mean()),
        "f0_std": float(voiced_f0.std()),
        "f0_min": float(voiced_f0.min()),
        "f0_max": float(voiced_f0.max()),
        "voiced_ratio": voiced_ratio,
        "f0_frames": f0_clean,
    }


def _empty_result(f0_frames: np.ndarray) -> dict:
    return {
        "f0_mean": float("nan"),
        "f0_std": float("nan"),
        "f0_min": float("nan"),
        "f0_max": float("nan"),
        "voiced_ratio": 0.0,
        "f0_frames": f0_frames,
    }


def extract_pitch_file(path: str, **kwargs) -> dict:
    y, sr = librosa.load(path, sr=None)
    return extract_pitch(y, sr, **kwargs)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "voice_search_project/data/raw/data_store/audio1/arctic_a0001.wav"
    )
    p = extract_pitch_file(path)
    print(f"File         : {path}")
    print(f"F0 mean/std  : {p['f0_mean']:.2f} Hz / {p['f0_std']:.2f} Hz")
    print(f"F0 min/max   : {p['f0_min']:.2f} Hz / {p['f0_max']:.2f} Hz")
    print(f"Voiced ratio : {p['voiced_ratio']:.3f}  ({p['voiced_ratio']*100:.1f}%)")
