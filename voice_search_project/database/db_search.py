"""
db_search.py
------------
Tìm kiếm Top-K file giống nhất bằng Cosine Similarity (brute-force).

PostgreSQL sequential scan toàn bộ bảng segments (~1,248 dòng) với
toán tử <=> của pgvector — chính xác 100%, < 5ms.

Cách dùng:
    from database.db_search import search_top_k, normalize_query

    results = search_top_k(query_vec_raw, top_k=5)
    for r in results:
        print(r["filename"], r["similarity"])
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db_connect import get_cursor

FEATURE_COLS = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + ["f0_mean", "f0_std", "voiced_ratio",
       "jitter", "shimmer", "hnr",
       "zcr_mean", "zcr_std", "rms_mean", "rms_std"]
)


def load_scaler() -> tuple[np.ndarray, np.ndarray]:
    """
    Lấy mean và std từ bảng scaler_params.
    Dùng để normalize query vector trước khi tìm kiếm.

    Returns:
        means : np.ndarray shape (23,)
        stds  : np.ndarray shape (23,)
    """
    with get_cursor() as cur:
        cur.execute("SELECT feature_name, mean, std FROM scaler_params")
        rows = {r["feature_name"]: (r["mean"], r["std"]) for r in cur.fetchall()}

    means = np.array([rows[f][0] for f in FEATURE_COLS], dtype=np.float64)
    stds  = np.array([rows[f][1] for f in FEATURE_COLS], dtype=np.float64)
    return means, stds


def normalize_query(raw_vec: np.ndarray) -> np.ndarray:
    """
    Z-score normalize query vector bằng scaler đã fit lúc insert.

    Args:
        raw_vec : vector 23 chiều chưa normalize, shape (23,)

    Returns:
        vector đã normalize, shape (23,)
    """
    means, stds = load_scaler()
    return (raw_vec - means) / (stds + 1e-8)


def search_top_k(
    query_vec : np.ndarray,
    top_k     : int  = 5,
    normalize : bool = True,
) -> list[dict]:
    """
    Tìm Top-K file có giọng giống nhất với query.
    Dùng brute-force cosine similarity — PostgreSQL quét toàn bảng,
    chính xác 100%, < 5ms với ~1,248 dòng.

    Args:
        query_vec : vector đặc trưng 23 chiều (raw, chưa normalize)
        top_k     : số kết quả trả về (mặc định 5)
        normalize : True = tự Z-score normalize trước khi query

    Returns:
        list[dict] sắp xếp theo similarity giảm dần, mỗi dict gồm:
            filename, audio_group, sentence_id, variant, similarity
    """
    if query_vec.shape[0] != 23:
        raise ValueError(f"Query vector phải có 23 chiều, nhận {query_vec.shape[0]}")

    vec = normalize_query(query_vec) if normalize else query_vec
    vec_str = "[" + ",".join(f"{v:.8f}" for v in vec.tolist()) + "]"

    sql = """
        SELECT
            filename,
            file_path,
            audio_group,
            sentence_id,
            variant,
            1 - (feature_vector <=> %s::vector) AS similarity
        FROM segments
        ORDER BY feature_vector <=> %s::vector
        LIMIT %s
    """
    with get_cursor() as cur:
        cur.execute(sql, (vec_str, vec_str, top_k))
        rows = cur.fetchall()

    return [dict(r) for r in rows]


# ── Test khi chạy trực tiếp ───────────────────────────────────
if __name__ == "__main__":
    from database.db_connect import test_connection

    print("\n" + "=" * 50)
    print("  KIỂM TRA DB_SEARCH (Brute-force Cosine)")
    print("=" * 50)

    if not test_connection():
        sys.exit(1)

    # Kiểm tra số dòng trong DB
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM segments")
        n = cur.fetchone()["n"]
    print(f"\n  Segments trong DB : {n:,} dòng")

    if n == 0:
        print("  [INFO] Bảng segments rỗng — chạy db_insert.py trước.")
        sys.exit(0)

    # Test với random vector
    print(f"\n  Test với random vector 23 chiều (chưa normalize)...")
    np.random.seed(42)
    raw_vec = np.random.randn(23).astype(np.float64) * 50  # giả lập giá trị thô

    results = search_top_k(raw_vec, top_k=5)

    print(f"\n  Top-5 kết quả:")
    print(f"  {'Rank':<5} {'Filename':<35} {'Group':<10} {'Similarity':>10}")
    print(f"  {'-'*65}")
    for i, r in enumerate(results, 1):
        print(f"  [{i}]   {r['filename']:<35} {r['audio_group']:<10} {r['similarity']:>10.4f}")

    print("\n" + "=" * 50 + "\n")
