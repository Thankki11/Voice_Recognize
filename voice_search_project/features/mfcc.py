"""
MFCC feature extraction (13 coefficients) with optional delta and delta-delta.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_delta: bool = False,
) -> np.ndarray:
    """
    Returns shape (n_mfcc, T) — or (n_mfcc*3, T) when include_delta=True.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if not include_delta:
        return mfcc
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])


def extract_mfcc_file(path: str, **kwargs) -> tuple[np.ndarray, int]:
    """Returns (mfcc_matrix, sr)."""
    y, sr = librosa.load(path, sr=None)
    return extract_mfcc(y, sr, **kwargs), sr


def visualize_heatmap(mfcc: np.ndarray, sr: int, hop_length: int = 512,
                      title: str = "MFCC Heatmap", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mfcc, sr=sr, hop_length=hop_length,
        x_axis="time", ax=ax, cmap="viridis"
    )
    fig.colorbar(img, ax=ax, label="MFCC coefficient value")
    ax.set_title(title)
    ax.set_ylabel("MFCC coefficient")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "voice_search_project/data/raw/data_store/audio1/arctic_a0001.wav"
    )
    mfcc, sr = extract_mfcc_file(path)
    print(f"File  : {path}")
    print(f"Shape : {mfcc.shape}  (n_mfcc={mfcc.shape[0]}, frames={mfcc.shape[1]})")
    print(f"Mean per coeff:\n{mfcc.mean(axis=1).round(3)}")
    visualize_heatmap(mfcc, sr, title=f"MFCC — {path.split('/')[-1]}")
