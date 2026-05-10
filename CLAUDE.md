# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice search system built on the ARCTIC speech corpus. The system extracts acoustic features from WAV files, indexes them in a database, and matches query audio against the stored repository using DTW or cosine similarity.

## Running Code

```bash
# Organize raw audio files into subdirectory groups
python voice_search_project/organize_audio.py <source_dir> [--dry-run]

# Run Jupyter notebooks
jupyter notebook voice_search_project/notebooks/

# Run demo app (once implemented)
python voice_search_project/demo/app.py
```

## Architecture

All implementation files under `voice_search_project/` are currently empty scaffolds. The intended pipeline is:

1. **Data** (`data/raw/`): 1,535 WAV files from the ARCTIC dataset, pre-organized into groups:
   - `data_store/audio1–audio178/`: tập dữ liệu để index/train hệ thống (~1,074 files)
   - `data_query/audio180–audio220/`: tập dữ liệu **chỉ dùng để đánh giá độ chính xác** sau khi hệ thống hoàn thiện — không dùng trong quá trình xây dựng (~246 files)
   - Naming convention: `arctic_aXXXX[(<variant>)].wav`

2. **Feature Extraction** (`features/`): Independent modules per feature type (`mfcc.py`, `pitch.py`, `spectral.py`, `time_domain.py`, `voice_quality.py`, `segmentation.py`). `extract_all.py` orchestrates them all, outputting to `segments_all.csv`.

3. **Database** (`database/`): `schema.sql` defines the schema; `db_connect.py` manages connections; `db_insert.py` indexes features; `db_search.py` retrieves candidates.

4. **Search** (`search/`): Two similarity strategies — `dtw.py` (Dynamic Time Warping for sequence alignment) and `cosine_sim.py` (vector comparison). `scoring.py` ranks results; `query.py` is the entry point.

5. **Evaluation** (`evaluation/`): `precision_at_k.py` computes retrieval quality; `plot_scores.py` and `plot_dtw.py` visualize results.

6. **Demo** (`demo/app.py`): Web interface (Dash or Flask) for interactive voice search.

## Dependencies

`requirements.txt` is currently empty. Expected dependencies when implementing:
- `librosa` — audio loading and feature extraction
- `numpy`, `scipy` — numerical computation
- `pandas` — feature CSV handling
- `sqlalchemy` or `sqlite3` — database layer
- `plotly` / `matplotlib` — evaluation plots
- `dash` or `flask` — demo web app

Add all dependencies to `voice_search_project/requirements.txt` as they are introduced.
