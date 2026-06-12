"""
db_connect.py
-------------
Quản lý kết nối đến PostgreSQL + pgvector.
Tất cả file khác trong dự án đều import từ đây.

Cấu hình kết nối đọc từ file .env:
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=csdldpt
    DB_USER=postgres
    DB_PASS=your_password

Cách dùng:
    from database.db_connect import get_connection, get_cursor

    conn = get_connection()
    conn.close()

    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM audio_files")
        print(cur.fetchone())
"""

import os
import sys
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_FILE = os.path.join(_BASE_DIR, ".env")
load_dotenv(_ENV_FILE)


def _get_config() -> dict:
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASS"]
    missing  = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"\n[ERROR] Thiếu biến môi trường trong .env: {missing}")
        print(f"  Kiểm tra file: {_ENV_FILE}")
        sys.exit(1)
    return {
        "host":     os.getenv("DB_HOST"),
        "port":     int(os.getenv("DB_PORT", "5432")),
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS"),
    }


def get_connection() -> psycopg2.extensions.connection:
    """Tạo và trả về 1 kết nối psycopg2. Caller có trách nhiệm đóng sau khi dùng."""
    config = _get_config()
    try:
        return psycopg2.connect(**config)
    except psycopg2.OperationalError as e:
        print(f"\n[ERROR] Không kết nối được PostgreSQL: {e}")
        print(f"  Host={config['host']}  Port={config['port']}")
        print(f"  DB={config['dbname']}  User={config['user']}")
        raise


@contextmanager
def get_cursor(autocommit: bool = False):
    """
    Context manager — tự động mở/đóng connection và cursor.
    Tự COMMIT khi không có lỗi, tự ROLLBACK khi có lỗi.

        with get_cursor() as cur:
            cur.execute("SELECT ...")
            rows = cur.fetchall()
    """
    conn = get_connection()
    try:
        if autocommit:
            conn.set_isolation_level(0)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            yield cur
            if not autocommit:
                conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """Kiểm tra kết nối và in thông tin database. Dùng để verify setup."""
    print("\n" + "=" * 50)
    print("  KIỂM TRA KẾT NỐI POSTGRESQL")
    print("=" * 50)

    config = _get_config()
    print(f"\n  Host     : {config['host']}:{config['port']}")
    print(f"  Database : {config['dbname']}")
    print(f"  User     : {config['user']}")

    try:
        with get_cursor() as cur:
            cur.execute("SELECT version()")
            print(f"\n  {cur.fetchone()[0].split(',')[0]}")

            cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            if row:
                print(f"  pgvector  : v{row['extversion']} ✅")
            else:
                print(f"  pgvector  : CHƯA CÀI ❌  →  CREATE EXTENSION vector;")

            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('audio_files', 'scaler_params')
                ORDER BY table_name
            """)
            tables = [r["table_name"] for r in cur.fetchall()]
            print(f"\n  Bảng tồn tại:")
            for t in ["audio_files", "scaler_params"]:
                mark = "✅" if t in tables else "❌ (chưa tạo — chạy schema.sql)"
                print(f"    {t:15s} {mark}")

            if "audio_files" in tables:
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE split = 'db')                   AS db_files,
                        COUNT(*) FILTER (WHERE split = 'query')                AS query_files,
                        COUNT(*) FILTER (WHERE feature_vector IS NOT NULL)     AS has_vector
                    FROM audio_files
                """)
                r = cur.fetchone()
                print(f"\n  Dữ liệu hiện tại:")
                print(f"    audio_files (db)    : {r['db_files']:,}")
                print(f"    audio_files (query) : {r['query_files']:,}")
                print(f"    có feature_vector   : {r['has_vector']:,}")

        print(f"\n  Kết nối thành công ✅")
        print("=" * 50 + "\n")
        return True

    except Exception as e:
        print(f"\n  Kết nối thất bại ❌: {e}")
        print("=" * 50 + "\n")
        return False


if __name__ == "__main__":
    test_connection()
