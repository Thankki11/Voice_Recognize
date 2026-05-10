"""
Voice quality features: Jitter, Shimmer, HNR (Harmonic-to-Noise Ratio).
Computed directly from audio signal using librosa + numpy (no parselmouth).
"""

import numpy as np
import librosa


def extract_voice_quality(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 75.0,
    fmax: float = 600.0,
) -> dict:
    """
    Returns:
      jitter  : relative average perturbation of pitch period (ratio, ~0.002–0.02)
      shimmer : relative average perturbation of amplitude (ratio, ~0.02–0.15)
      hnr     : harmonic-to-noise ratio in dB (higher = cleaner voice, ~10–30 dB)
    Returns NaN for each metric if audio is too short or no voiced frames found.
    """
    nan_result = {"jitter": float("nan"), "shimmer": float("nan"), "hnr": float("nan")}

    if len(y) < 2048:
        return nan_result

    # --- F0 via pyin ---
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
        )
    except Exception:
        return nan_result

    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if voiced_f0.size < 4:
        return nan_result

    # --- Jitter ---
    periods = 1.0 / voiced_f0
    diffs = np.abs(np.diff(periods))
    jitter = float(diffs.mean() / periods.mean())

    # --- Shimmer ---
    amplitudes = []
    voiced_indices = np.where(voiced_flag & ~np.isnan(f0))[0]
    for idx in voiced_indices:
        sample_center = idx * hop_length
        period_samples = max(1, int(sr / f0[idx]))
        start = max(0, sample_center - period_samples // 2)
        end = min(len(y), start + period_samples)
        segment = y[start:end]
        if len(segment) > 0:
            amplitudes.append(float(np.sqrt(np.mean(segment ** 2))))

    if len(amplitudes) < 4:
        shimmer = float("nan")
    else:
        amp = np.array(amplitudes)
        amp_diffs = np.abs(np.diff(amp))
        shimmer = float(amp_diffs.mean() / amp.mean()) if amp.mean() > 0 else float("nan")

    # --- HNR via autocorrelation ---
    mean_f0 = float(voiced_f0.mean())
    lag_samples = int(round(sr / mean_f0))
    if lag_samples >= len(y):
        hnr = float("nan")
    else:
        y_norm = y - y.mean()
        power = float(np.dot(y_norm, y_norm))
        if power == 0:
            hnr = float("nan")
        else:
            r_lag = float(np.dot(y_norm[:-lag_samples], y_norm[lag_samples:]))
            r = r_lag / power
            r = max(min(r, 0.9999), 0.0001)
            hnr = float(10 * np.log10(r / (1.0 - r)))

    return {"jitter": jitter, "shimmer": shimmer, "hnr": hnr}


def extract_voice_quality_file(path: str, **kwargs) -> dict:
    y, sr = librosa.load(path, sr=None)
    return extract_voice_quality(y, sr, **kwargs)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "voice_search_project/data/raw/data_store/audio1/arctic_a0001.wav"
    )
    vq = extract_voice_quality_file(path)
    print(f"File    : {path}")
    print(f"Jitter  : {vq['jitter']:.4f}  (typ. 0.002-0.02)")
    print(f"Shimmer : {vq['shimmer']:.4f}  (typ. 0.02-0.15)")
    print(f"HNR     : {vq['hnr']:.2f} dB  (typ. 10-30 dB)")
