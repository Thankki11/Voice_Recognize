import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Thay đổi backend matplotlib thành Agg để không cần giao diện đồ họa
import matplotlib
matplotlib.use('Agg')

from search.query import search

def main():
    out_dir = PROJECT_ROOT / "evaluation" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = str(PROJECT_ROOT / "data" / "raw" / "data_store" / "audio2" / "arctic_a0002(1).wav")
    print(f"Loading {sample_path}...")
    y, sr = librosa.load(sample_path, sr=None)
    y_trim, index = librosa.effects.trim(y, top_db=20)
    
    print("Plotting preprocessing...")
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title("1. Raw Waveform (Trước khi cắt khoảng lặng)")
    plt.axvline(x=index[0]/sr, color='r', linestyle='--', label='Trim Start')
    plt.axvline(x=index[1]/sr, color='r', linestyle='--', label='Trim End')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_trim, sr=sr, alpha=0.8, color='orange')
    plt.title("2. Trimmed Waveform (Sau khi cắt khoảng lặng)")
    plt.tight_layout()
    plt.savefig(out_dir / "1_preprocessing.png", dpi=150)
    plt.close()

    print("Plotting features...")
    mfcc = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=13, hop_length=512)
    f0, voiced_flag, voiced_probs = librosa.pyin(y_trim, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title("3. Trích xuất hệ số MFCC (Mel-frequency cepstral coefficients)")
    
    plt.subplot(2, 1, 2)
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    plt.plot(times, f0, label='Cao độ (F0)', color='blue', linewidth=2)
    plt.title("4. Theo dõi Cao độ (Pitch Tracking bằng pYIN)")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Tần số (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "2_features.png", dpi=150)
    plt.close()
    
    print("Running search and plotting results...")
    try:
        results = search(sample_path, top_k=5)
        filenames = [r["filename"] for r in results]
        similarities = [r["similarity"] for r in results]
        
        plt.figure(figsize=(10, 5))
        bars = plt.barh(filenames[::-1], similarities[::-1], color='mediumseagreen')
        plt.xlabel("Độ tương đồng Cosin (Cosine Similarity)")
        plt.title("5. Kết quả tìm kiếm (Top 5 file tương đồng nhất)")
        
        # Đặt giới hạn trục X phù hợp với giá trị
        min_sim = min(similarities)
        plt.xlim(max(0, min_sim - 0.05), 1.0)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(out_dir / "3_search_results.png", dpi=150)
        plt.close()
        print("Done! Plots saved to evaluation/plots/")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    main()
