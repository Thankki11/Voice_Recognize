"""
query.py
--------
Entry point của pipeline tìm kiếm giọng nói.

Luồng hoạt động:
    1. Đọc file WAV query → trim khoảng lặng (energy-based, top_db=20)
    2. Trích xuất vector 23 chiều (MFCC×13, Pitch×3, Quality×3, Time×4)
    3. Z-score normalize bằng scaler_params đã lưu trong DB
           z[i] = (raw[i] - mean[i]) / std[i]
    4. Tìm Top-K bằng cosine similarity (brute-force pgvector <=>)
           similarity = 1 - cosine_distance
           cosine_distance = 1 - (A·B)/(||A||·||B||)
    5. Trả về list kết quả kèm rank và similarity

Cách dùng (import):
    from search.query import search
    results = search("path/to/query.wav", top_k=5)

Cách chạy (terminal):
    python -m search.query <path_to_wav> [--top_k N]
"""

import sys
import time
import argparse
import numpy as np
import librosa
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.extract_all import extract_features
from database.db_search import search_top_k, FEATURE_COLS

HOP_LENGTH = 512


def _load_and_trim(wav_path: str) -> tuple[np.ndarray, int]:
    """
    Đọc file WAV và trim khoảng lặng đầu/cuối.

    Tiêu chí trim (energy-based):
        frame được coi là silence nếu:
        RMS_frame < 10^(-20/20) × max(RMS_file) = 0.1 × max_RMS

    Returns:
        y  : tín hiệu sau trim, shape (L,)
        sr : sample rate (Hz)
    """
    y, sr = librosa.load(wav_path, sr=None)
    y, _  = librosa.effects.trim(y, top_db=20)
    return y, sr


def search(wav_path: str, top_k: int = 5) -> list[dict]:
    """
    Tìm Top-K file có giọng giống nhất với file WAV query.

    Pipeline:
        WAV → trim → extract 23 features → Z-score normalize → cosine similarity → Top-K

    Args:
        wav_path : đường dẫn đến file WAV query
        top_k    : số kết quả trả về (mặc định 5)

    Returns:
        list[dict] sắp xếp theo similarity giảm dần, mỗi dict gồm:
            rank        : thứ hạng (1 = giống nhất)
            filename    : tên file kết quả
            audio_group : thư mục chứa file (audio2, audio50...)
            sentence_id : ID đoạn văn — ground truth để đánh giá
            variant     : số thứ tự người đọc (0=SPK2, 1=SPK1, 3=SPK3...)
            similarity  : cosine similarity ∈ [-1, 1], càng cao càng giống
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {wav_path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Chỉ hỗ trợ file WAV, nhận được: {path.suffix}")

    # Bước 1: Load + trim
    y, sr = _load_and_trim(wav_path)
    if len(y) < HOP_LENGTH:
        raise ValueError(f"File quá ngắn sau khi trim: {len(y)} samples")

    # Bước 2: Extract 23 đặc trưng
    feat_dict = extract_features(y, sr)
    raw_vec   = np.array([feat_dict[c] for c in FEATURE_COLS], dtype=np.float64)

    if np.any(np.isnan(raw_vec)):
        nan_cols = [FEATURE_COLS[i] for i, v in enumerate(raw_vec) if np.isnan(v)]
        raise ValueError(f"Feature extraction thất bại tại: {nan_cols}")

    # Bước 3 + 4: Normalize + tìm kiếm (gộp trong search_top_k)
    results = search_top_k(raw_vec, top_k=top_k, normalize=True)

    # Bước 5: Thêm rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


# ── Chạy trực tiếp từ terminal ────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tìm kiếm giọng nói tương đồng",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python -m search.query D:\\data\\audio2\\arctic_a0002.wav
  python -m search.query D:\\data\\audio2\\arctic_a0002.wav --top_k 10
        """
    )
    parser.add_argument("wav_path", help="Đường dẫn đến file WAV query")
    parser.add_argument("--top_k", type=int, default=5, help="Số kết quả (mặc định: 5)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  TÌM KIẾM GIỌNG NÓI TƯƠNG ĐỒNG")
    print("=" * 60)
    print(f"\n  Query : {args.wav_path}")
    print(f"  Top-K : {args.top_k}")

    t0 = time.time()
    try:
        results = search(args.wav_path, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)
    elapsed = (time.time() - t0) * 1000

    # Lấy sentence_id của query để đánh giá
    query_name = Path(args.wav_path).name
    print(f"\n  {'Rank':<5} {'Filename':<38} {'Group':<10} {'SentID':>6} {'Similarity':>10}")
    print(f"  {'-'*72}")
    for r in results:
        print(
            f"  [{r['rank']}]   "
            f"{r['filename']:<38} "
            f"{r['audio_group']:<10} "
            f"{str(r['sentence_id']):>6} "
            f"{r['similarity']:>10.4f}"
        )

    print(f"\n  Thời gian truy vấn : {elapsed:.1f} ms")
    print("=" * 60 + "\n")
