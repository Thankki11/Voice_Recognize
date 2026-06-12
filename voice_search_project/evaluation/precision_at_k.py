"""
evaluation/precision_at_k.py
-----------------------------
Đánh giá hệ thống tìm kiếm giọng nói bằng Precision@K và mAP@K.

Ground truth: file kết quả đúng = file có cùng VARIANT (cùng người đọc) với query.

Tối ưu: dùng feature_vector đã có sẵn trong DB thay vì trích xuất lại từ WAV
       → chạy chỉ vài giây.

Cách chạy:
    python voice_search_project/evaluation/precision_at_k.py
"""

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_connect import get_cursor

K_VALUES = [1, 3, 5, 10]


def average_precision(target_variant: int, ranked: list[dict], k: int, n_relevant: int) -> float:
    hits, score = 0, 0.0
    for i, r in enumerate(ranked[:k], 1):
        if r["variant"] == target_variant:
            hits += 1
            score += hits / i
    return score / min(n_relevant, k) if n_relevant > 0 else 0.0


def evaluate(k_values: list[int] = K_VALUES) -> dict:
    max_k = max(k_values) + 1

    with get_cursor() as cur:
        cur.execute("SELECT id, filename, variant, feature_vector FROM segments")
        all_rows = cur.fetchall()

    print(f"  Tổng file trong DB : {len(all_rows)}")

    variant_counts = Counter(r["variant"] for r in all_rows)
    print(f"  Phân bố variant    : {dict(variant_counts)}")

    precision = {k: [] for k in k_values}
    ap_scores = {k: [] for k in k_values}

    for i, query_row in enumerate(all_rows, 1):
        qid = query_row["id"]
        target_variant = query_row["variant"]
        n_relevant = variant_counts[target_variant] - 1

        if n_relevant <= 0:
            continue

        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, variant
                FROM   segments
                WHERE  id <> %s
                ORDER BY feature_vector <=> (
                    SELECT feature_vector FROM segments WHERE id = %s
                )
                LIMIT %s
                """,
                (qid, qid, max_k),
            )
            results = [dict(r) for r in cur.fetchall()]

        for k in k_values:
            top = results[:k]
            hits = sum(1 for r in top if r["variant"] == target_variant)
            precision[k].append(hits / k)
            ap_scores[k].append(
                average_precision(target_variant, results, k, n_relevant)
            )

        if i % 200 == 0:
            print(f"  Đã xử lý {i}/{len(all_rows)} file...")

    return {
        "n_queries" : len([1 for k in precision[K_VALUES[0]] for _ in [1]]),
        "precision" : {k: sum(v)/len(v) if v else 0 for k, v in precision.items()},
        "mAP"       : {k: sum(v)/len(v) if v else 0 for k, v in ap_scores.items()},
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ HỆ THỐNG TÌM KIẾM GIỌNG NÓI")
    print("  Ground truth: cùng VARIANT (cùng speaker)")
    print("=" * 60)

    results = evaluate()

    print(f"\n  Số query đã đánh giá : {results['n_queries']}")
    print(f"\n  {'K':<6} {'Precision@K':>12} {'mAP@K':>10}")
    print(f"  {'-'*30}")
    for k in K_VALUES:
        p = results['precision'][k]
        m = results['mAP'][k]
        print(f"  K={k:<4} {p:>12.4f} {m:>10.4f}")

    print("\n" + "=" * 60 + "\n")
