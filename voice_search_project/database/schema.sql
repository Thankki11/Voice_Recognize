CREATE TABLE IF NOT EXISTS segments (
    id           SERIAL PRIMARY KEY,
    file_path    TEXT NOT NULL,
    audio_group  TEXT,
    filename     TEXT NOT NULL,
    sentence_id  INTEGER,
    variant      INTEGER,
    mfcc_mean_0  REAL, mfcc_mean_1  REAL, mfcc_mean_2  REAL,
    mfcc_mean_3  REAL, mfcc_mean_4  REAL, mfcc_mean_5  REAL,
    mfcc_mean_6  REAL, mfcc_mean_7  REAL, mfcc_mean_8  REAL,
    mfcc_mean_9  REAL, mfcc_mean_10 REAL, mfcc_mean_11 REAL,
    mfcc_mean_12 REAL,
    f0_mean      REAL, f0_std       REAL, voiced_ratio REAL,
    jitter       REAL, shimmer      REAL, hnr          REAL,
    zcr_mean     REAL, zcr_std      REAL,
    rms_mean     REAL, rms_std      REAL
);

CREATE TABLE IF NOT EXISTS scaler_params (
    feature_name TEXT PRIMARY KEY,
    mean         REAL NOT NULL,
    std          REAL NOT NULL
);
