"""
Time-domain features: Zero Crossing Rate, RMS Energy.

ZCR cong thuc: ZCR = (1/2N) * sum(|sgn(x[n]) - sgn(x[n-1])|)
RMS cong thuc: RMS = sqrt( mean(x[n]^2) )
"""

import numpy as np
import librosa


def _zcr_frame(frame: np.ndarray) -> float:
    """
    ZCR = (1/2N) * Sigma |sgn(x[n]) - sgn(x[n-1])|

    |sgn(+1) - sgn(-1)| = 2  khi doi dau  → dong gop 2
    |sgn(+1) - sgn(+1)| = 0  khi cung dau → dong gop 0
    Chia 2N de triet tieu he so 2, ket qua = so_crossings / N
    """
    N = len(frame)
    signs = np.sign(frame)
    signs[signs == 0] = 1                        # xu ly sample = 0
    total = np.sum(np.abs(np.diff(signs)))       # Sigma |sgn[n] - sgn[n-1]|
    return float(total / (2 * N))


def _rms_frame(frame: np.ndarray) -> float:
    """RMS = sqrt( mean(x[n]^2) )"""
    return float(np.sqrt(np.mean(frame ** 2)))


def extract_time_domain(
    y: np.ndarray,
    sr: int,  # noqa: ARG001 — kept for API consistency with other feature functions
    frame_length: int = 2048,
    hop_length: int = 512,
) -> dict:
    """
    Chia y thanh cac frame (frame_length, hop_length),
    tinh ZCR va RMS theo cong thuc cho tung frame,
    tra ve mean va std qua tat ca frames.
    """
    n_frames = 1 + (len(y) - frame_length) // hop_length

    zcr_frames = np.zeros(n_frames)
    rms_frames = np.zeros(n_frames)

    for i in range(n_frames):
        frame = y[i * hop_length : i * hop_length + frame_length]
        zcr_frames[i] = _zcr_frame(frame)
        rms_frames[i] = _rms_frame(frame)

    return {
        "zcr_mean":   float(zcr_frames.mean()),
        "zcr_std":    float(zcr_frames.std()),
        "rms_mean":   float(rms_frames.mean()),
        "rms_std":    float(rms_frames.std()),
        "zcr_frames": zcr_frames,
        "rms_frames": rms_frames,
    }


def extract_time_domain_file(path: str, **kwargs) -> dict:
    y, sr = librosa.load(path, sr=None)
    return extract_time_domain(y, sr, **kwargs)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "voice_search_project/data/raw/data_store/audio1/arctic_a0001.wav"
    )
    feats = extract_time_domain_file(path)
    print(f"File          : {path}")
    print(f"ZCR mean/std  : {feats['zcr_mean']:.4f} / {feats['zcr_std']:.4f}")
    print(f"RMS mean/std  : {feats['rms_mean']:.4f} / {feats['rms_std']:.4f}")
    print(f"Silence ratio : {feats['silence_ratio']:.3f}  "
          f"({feats['silence_ratio']*100:.1f}% of frames)")
