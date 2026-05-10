"""
Đọc segments_all.csv → Fit StandardScaler → Insert vào PostgreSQL.

Bảng segments     : 1248 dòng × 23 chiều (đã chuẩn hóa Z-score)
Bảng scaler_params: 23 dòng  (feature_name, mean, std)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_connect import get_connection

CSV_PATH = PROJECT_ROOT / "features" / "segments_all.csv"

FEATURE_COLS = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + ["f0_mean", "f0_std", "voiced_ratio",
       "jitter", "shimmer", "hnr",
       "zcr_mean", "zcr_std", "rms_mean", "rms_std"]
)

META_COLS = ["file_path", "audio_group", "filename", "sentence_id", "variant"]

INSERT_SEGMENTS = """
    INSERT INTO segments
        (file_path, audio_group, filename, sentence_id, variant,
         mfcc_mean_0,  mfcc_mean_1,  mfcc_mean_2,  mfcc_mean_3,
         mfcc_mean_4,  mfcc_mean_5,  mfcc_mean_6,  mfcc_mean_7,
         mfcc_mean_8,  mfcc_mean_9,  mfcc_mean_10, mfcc_mean_11,
         mfcc_mean_12,
         f0_mean, f0_std, voiced_ratio,
         jitter, shimmer, hnr,
         zcr_mean, zcr_std, rms_mean, rms_std)
    VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s,%s)
"""

INSERT_SCALER = """
    INSERT INTO scaler_params (feature_name, mean, std)
    VALUES (%s, %s, %s)
    ON CONFLICT (feature_name) DO UPDATE SET mean=EXCLUDED.mean, std=EXCLUDED.std
"""


def run(csv_path: Path = CSV_PATH):
    # --- Bước 1: Đọc CSV ---
    df = pd.read_csv(csv_path)
    print(f"Đọc CSV: {len(df)} dòng × {len(df.columns)} cột")

    # --- Bước 2: Fit StandardScaler ---
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    means = scaler.mean_
    stds  = scaler.scale_

    print(f"Đã fit StandardScaler trên {len(FEATURE_COLS)} chiều")
    print(f"  mfcc_mean_0 : μ={means[0]:.4f}, σ={stds[0]:.4f}")
    print(f"  f0_mean     : μ={means[13]:.4f}, σ={stds[13]:.4f}")

    # --- Bước 3 & 4: Insert vào DB ---
    conn = get_connection()
    with conn.cursor() as cur:
        # Xóa dữ liệu cũ nếu chạy lại
        cur.execute("TRUNCATE TABLE segments RESTART IDENTITY CASCADE")
        cur.execute("DELETE FROM scaler_params")

        # Insert scaler params (23 dòng)
        scaler_rows = [
            (feat, float(means[i]), float(stds[i]))
            for i, feat in enumerate(FEATURE_COLS)
        ]
        cur.executemany(INSERT_SCALER, scaler_rows)

        # Insert segments (1248 dòng)
        segment_rows = []
        for _, row in df_scaled.iterrows():
            meta  = [row[c] for c in META_COLS]
            feats = [float(row[c]) for c in FEATURE_COLS]
            segment_rows.append(meta + feats)

        cur.executemany(INSERT_SEGMENTS, segment_rows)

    conn.commit()

    # Xác nhận
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM segments")
        n_seg = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scaler_params")
        n_scaler = cur.fetchone()[0]
        cur.execute("SELECT mfcc_mean_0 FROM segments")
        vals = np.array([r[0] for r in cur.fetchall()])

    print(f"\nĐã insert:")
    print(f"  segments     : {n_seg} dòng")
    print(f"  scaler_params: {n_scaler} dòng")
    print(f"\nKiểm tra Z-score mfcc_mean_0:")
    print(f"  mean = {vals.mean():.6f}  (kỳ vọng ≈ 0.0)")
    print(f"  std  = {vals.std():.6f}  (kỳ vọng ≈ 1.0)")

    conn.close()
    print(f"\nDB: PostgreSQL voice_search @ localhost:5432")


if __name__ == "__main__":
    run()
