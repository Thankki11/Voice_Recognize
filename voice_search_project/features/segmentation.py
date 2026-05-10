"""
Energy-based voice activity detection: splits a WAV file into voiced segments
by thresholding short-time RMS energy.
"""

import numpy as np
import librosa


def segment_audio(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    energy_threshold: float | None = None,
    min_silence_frames: int = 5,
) -> list[dict]:
    """
    Returns list of voiced segments:
      [{"start_s": float, "end_s": float, "start_sample": int, "end_sample": int}]
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = energy_threshold if energy_threshold is not None else rms.mean() * 0.3
    voiced = rms > threshold

    segments = []
    in_seg = False
    start_frame = 0
    silence_count = 0

    for i, v in enumerate(voiced):
        if v:
            if not in_seg:
                start_frame = i
                in_seg = True
            silence_count = 0
        else:
            if in_seg:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    end_frame = i - silence_count
                    segments.append(_to_time(start_frame, end_frame, hop_length, sr, len(y)))
                    in_seg = False
                    silence_count = 0

    if in_seg:
        segments.append(_to_time(start_frame, len(voiced) - 1, hop_length, sr, len(y)))

    return segments


def _to_time(sf: int, ef: int, hop: int, sr: int, n: int) -> dict:
    s = sf * hop
    e = min(ef * hop, n)
    return {"start_s": s / sr, "end_s": e / sr, "start_sample": s, "end_sample": e}


def segment_file(path: str, **kwargs) -> list[dict]:
    y, sr = librosa.load(path, sr=None)
    return segment_audio(y, sr, **kwargs)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "voice_search_project/data/raw/data_store/audio1/arctic_a0001.wav"
    )
    segs = segment_file(path)
    print(f"File     : {path}")
    print(f"Segments : {len(segs)}")
    for i, s in enumerate(segs):
        dur = s["end_s"] - s["start_s"]
        print(f"  [{i+1}] {s['start_s']:.3f}s - {s['end_s']:.3f}s  (dur={dur:.3f}s)")
