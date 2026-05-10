"""
Helpers để load dữ liệu từ PostgreSQL phục vụ tìm kiếm cosine similarity.
"""

import numpy as np
import psycopg2

FEATURE_COLS = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + ["f0_mean", "f0_std", "voiced_ratio",
       "jitter", "shimmer", "hnr",
       "zcr_mean", "zcr_std", "rms_mean", "rms_std"]
)


def load_matrix(conn: psycopg2.extensions.connection) -> tuple[np.ndarray, list[dict]]:
    """
    Trả về:
      matrix   : (N, 23) float64 — vectors đã chuẩn hóa
      metadata : list[dict] — file_path, audio_group, filename, sentence_id, variant
    """
    cols_sql = ", ".join(FEATURE_COLS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT file_path, audio_group, filename, sentence_id, variant, {cols_sql} "
            f"FROM segments ORDER BY id"
        )
        rows = cur.fetchall()

    n_meta = 5
    matrix = np.array([list(r[n_meta:]) for r in rows], dtype=np.float64)
    metadata = [
        {
            "file_path":   r[0],
            "audio_group": r[1],
            "filename":    r[2],
            "sentence_id": r[3],
            "variant":     r[4],
        }
        for r in rows
    ]
    return matrix, metadata


def load_scaler_params(conn: psycopg2.extensions.connection) -> tuple[np.ndarray, np.ndarray]:
    """
    Trả về means (23,) và stds (23,) theo thứ tự FEATURE_COLS.
    Dùng để chuẩn hóa query mới: z = (x - means) / stds
    """
    with conn.cursor() as cur:
        cur.execute("SELECT feature_name, mean, std FROM scaler_params")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    means = np.array([rows[f][0] for f in FEATURE_COLS], dtype=np.float64)
    stds  = np.array([rows[f][1] for f in FEATURE_COLS], dtype=np.float64)
    return means, stds
