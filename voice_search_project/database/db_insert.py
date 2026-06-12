"""
db_insert.py
------------
Đọc segments_all.csv → Z-score normalize → insert vào PostgreSQL.

Bảng segments     : 1 dòng/file, cột feature_vector vector(23)
Bảng scaler_params: 23 dòng (mean + std của từng chiều)

Cách chạy:
    python -m database.db_insert
    python -m database.db_insert --dry-run
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.db_connect import get_cursor, test_connection

_BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH  = _BASE_DIR / "features" / "segments_all.csv"

FEATURE_COLS = (
    [f"mfcc_mean_{i}" for i in range(13)]
    + ["f0_mean", "f0_std", "voiced_ratio",
       "jitter", "shimmer", "hnr",
       "zcr_mean", "zcr_std", "rms_mean", "rms_std"]
)

INSERT_SEGMENT = """
    INSERT INTO segments (file_path, audio_group, filename, sentence_id, variant, feature_vector)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
    ON CONFLICT (filename) DO UPDATE SET feature_vector = EXCLUDED.feature_vector
"""

INSERT_SCALER = """
    INSERT INTO scaler_params (feature_name, mean, std)
    VALUES (%s, %s, %s)
    ON CONFLICT (feature_name) DO UPDATE SET mean = EXCLUDED.mean, std = EXCLUDED.std
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default=str(CSV_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  NẠP FEATURE VECTOR VÀO POSTGRESQL")
    print("=" * 50)

    if not test_connection():
        sys.exit(1)

    # --- Bước 1: Đọc CSV ---
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] Không tìm thấy: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n  Đọc CSV    : {csv_path.name}")
    print(f"  Số file    : {len(df)}")
    print(f"  NaN        : {df[FEATURE_COLS].isnull().sum().sum()} ô trống")

    # --- Bước 2: Z-score normalize ---
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    print(f"  Z-score    : fit xong {len(FEATURE_COLS)} chiều")
    print(f"    f0_mean  : μ={scaler.mean_[13]:.2f} Hz, σ={scaler.scale_[13]:.2f}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Sẽ insert {len(df)} dòng vào segments")
        print(f"  [DRY RUN] Sẽ upsert {len(FEATURE_COLS)} dòng vào scaler_params")
        print("\n" + "=" * 50 + "\n")
        return

    # --- Bước 3: Insert scaler_params ---
    scaler_rows = [
        (feat, float(scaler.mean_[i]), float(scaler.scale_[i]))
        for i, feat in enumerate(FEATURE_COLS)
    ]
    with get_cursor() as cur:
        cur.executemany(INSERT_SCALER, scaler_rows)
    print(f"\n  scaler_params : {len(scaler_rows)} dòng ✅")

    # --- Bước 4: Insert segments ---
    print(f"  Đang insert {len(df_scaled)} dòng vào segments...")
    rows = []
    for _, row in df_scaled.iterrows():
        vec = [float(row[c]) for c in FEATURE_COLS]
        vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
        rows.append((
            row["file_path"],
            row["audio_group"],
            row["filename"],
            int(row["sentence_id"]),
            int(row["variant"]),
            vec_str,
        ))

    with get_cursor() as cur:
        cur.executemany(INSERT_SEGMENT, rows)

    # --- Bước 5: Xác nhận ---
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM segments")
        n_seg = cur.fetchone()["n"]
        cur.execute("SELECT COUNT(*) AS n FROM scaler_params")
        n_scaler = cur.fetchone()["n"]
    print(f"\n  Kết quả:")
    print(f"    segments      : {n_seg:,} dòng ✅")
    print(f"    scaler_params : {n_scaler} dòng ✅")

    print("\n" + "=" * 50)
    print("  HOÀN TẤT")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
