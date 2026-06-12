-- =================================================================
-- schema.sql — Hệ thống tìm kiếm giọng nói
-- PostgreSQL 18 + pgvector
--
-- Chạy:  psql -U postgres -d csdldpt -f schema.sql
-- Reset: bỏ comment 2 dòng DROP bên dưới rồi chạy lại
-- =================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- DROP TABLE IF EXISTS scaler_params CASCADE;
-- DROP TABLE IF EXISTS segments      CASCADE;

-- =================================================================
-- BẢNG 1: segments
-- 1 dòng = 1 file WAV
-- feature_vector: 23 chiều, đã Z-score normalize
--
-- Bố cục vector(23):
--   [0–12]  mfcc_mean_0..12  — hình dạng khoang cộng hưởng
--   [13]    f0_mean          — pitch trung bình (Hz)
--   [14]    f0_std           — biến thiên ngữ điệu
--   [15]    voiced_ratio     — tỉ lệ voiced frames
--   [16]    jitter           — độ run chu kỳ dây thanh
--   [17]    shimmer          — độ run biên độ
--   [18]    hnr              — Harmonic-to-Noise Ratio (dB)
--   [19]    zcr_mean         — Zero Crossing Rate trung bình
--   [20]    zcr_std          — biến thiên ZCR
--   [21]    rms_mean         — năng lượng trung bình
--   [22]    rms_std          — biến thiên năng lượng
-- =================================================================
CREATE TABLE IF NOT EXISTS segments (
    id              SERIAL       PRIMARY KEY,
    file_path       TEXT         NOT NULL,
    audio_group     TEXT,
    filename        TEXT         NOT NULL UNIQUE,
    sentence_id     INTEGER,
    variant         INTEGER,
    feature_vector  vector(23)   NOT NULL
);

-- =================================================================
-- BẢNG 2: scaler_params
-- Lưu mean + std của từng chiều để normalize query lúc tìm kiếm
-- =================================================================
CREATE TABLE IF NOT EXISTS scaler_params (
    feature_name    TEXT    PRIMARY KEY,
    mean            REAL    NOT NULL,
    std             REAL    NOT NULL
);

-- =================================================================
-- KHÔNG dùng ANN index (IVFFlat/HNSW)
-- Với ~1,248 file, brute-force sequential scan chính xác 100% và
-- nhanh hơn IVFFlat (< 5ms). Index vector chỉ cần khi > 50,000 file.
-- =================================================================

-- =================================================================
-- KIỂM TRA SAU KHI CHẠY
-- =================================================================
-- SELECT table_name FROM information_schema.tables
-- WHERE table_schema = 'public' ORDER BY table_name;

-- SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- =================================================================
-- TRUY VẤN MẪU (sau khi nạp data)
-- =================================================================
-- Top-5 file giống nhất với query vector:
-- SELECT id, filename, audio_group,
--        1 - (feature_vector <=> '[0.1,...]'::vector) AS similarity
-- FROM segments
-- ORDER BY feature_vector <=> '[0.1,...]'::vector
-- LIMIT 5;

-- Kiểm tra số dòng:
-- SELECT COUNT(*) FROM segments;
-- SELECT COUNT(*) FROM scaler_params;
