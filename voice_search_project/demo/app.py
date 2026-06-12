"""
demo/app.py
-----------
Gradio demo app cho hệ thống tìm kiếm giọng nói.

Cách chạy:
    python voice_search_project/demo/app.py
    Mở trình duyệt: http://localhost:7861
"""

import sys
import time
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from search.query import search

# Lưu file_path của kết quả hiện tại để dùng khi click hàng
_current_file_paths: list[str] = []


def run_search(wav_file: str, top_k: int) -> tuple[pd.DataFrame, str, None]:
    global _current_file_paths

    if wav_file is None:
        _current_file_paths = []
        return pd.DataFrame(), "⚠️ Vui lòng upload file WAV trước.", None

    t0 = time.time()
    try:
        results = search(wav_file, top_k=int(top_k))
    except Exception as e:
        _current_file_paths = []
        return pd.DataFrame(), f"❌ Lỗi: {e}", None
    elapsed = time.time() - t0

    _current_file_paths = [r["file_path"] for r in results]

    df = pd.DataFrame([
        {
            "Rank"       : r["rank"],
            "Filename"   : r["filename"],
            "Group"      : r["audio_group"],
            "Sentence ID": r["sentence_id"],
            "Similarity" : round(r["similarity"], 4),
        }
        for r in results
    ])

    status = (
        f"✅ Tìm kiếm hoàn tất  |  "
        f"⏱ Tổng thời gian: {elapsed:.1f}s  |  "
        f"📊 Trả về {len(results)} kết quả"
    )
    return df, status, None


def on_row_select(evt: gr.SelectData) -> tuple[str | None, str]:
    row_idx = evt.index[0]
    if row_idx >= len(_current_file_paths):
        return None, ""
    file_path = Path(_current_file_paths[row_idx])
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        return None, f"⚠️ Không tìm thấy file: {file_path}"
    return str(file_path), f"🔊 Đang phát: {file_path.name}"


# ── Giao diện Gradio ──────────────────────────────────────────
with gr.Blocks(title="Voice Search Demo") as demo:

    gr.Markdown("""
    # 🎤 Hệ Thống Tìm Kiếm Giọng Nói
    **ARCTIC Corpus — Cosine Similarity (Vector 23 chiều)**

    Upload một file WAV tiếng Anh giọng nữ, hệ thống sẽ tìm các file có giọng nói giống nhất
    trong cơ sở dữ liệu 1,248 file dựa trên đặc trưng âm học (MFCC, Pitch, Voice Quality, ZCR/RMS).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            wav_input = gr.Audio(
                label="📁 Upload file WAV query",
                type="filepath",
                sources=["upload"],
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="Số kết quả Top-K",
            )
            search_btn = gr.Button("🔍 Tìm kiếm", variant="primary", size="lg")

        with gr.Column(scale=2):
            result_table = gr.Dataframe(
                headers=["Rank", "Filename", "Group", "Sentence ID", "Similarity"],
                label="📊 Kết quả tìm kiếm — Click vào hàng để nghe",
                interactive=False,
                wrap=True,
            )
            status_text = gr.Textbox(
                interactive=False,
                show_label=False,
            )

    with gr.Row():
        with gr.Column():
            audio_player = gr.Audio(
                label="🔊 Nghe file được chọn",
                type="filepath",
                interactive=False,
            )
            audio_label = gr.Textbox(
                interactive=False,
                show_label=False,
            )

    gr.Markdown("""
    ---
    ### 📖 Hướng dẫn đọc kết quả
    | Cột | Ý nghĩa |
    |---|---|
    | **Rank** | Thứ hạng — 1 là giống nhất |
    | **Filename** | Tên file trong corpus |
    | **Group** | Thư mục chứa file (audio2 = đoạn văn số 2) |
    | **Sentence ID** | ID đoạn văn — các file cùng Sentence ID là cùng câu đọc bởi người khác |
    | **Similarity** | Cosine similarity ∈ [-1, 1] — càng gần 1.0 càng giống |

    > **Lưu ý:** Thời gian xử lý ~5–30 giây do bước trích xuất đặc trưng pYIN (pitch detection).
    > Click vào bất kỳ hàng nào trong bảng kết quả để nghe file âm thanh tương ứng.
    """)

    search_btn.click(
        fn=run_search,
        inputs=[wav_input, top_k_slider],
        outputs=[result_table, status_text, audio_player],
    )

    result_table.select(
        fn=on_row_select,
        outputs=[audio_player, audio_label],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
