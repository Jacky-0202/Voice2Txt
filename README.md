# 🎧 Whisper Dual-Language Transcriber

一個基於 `faster-whisper` (Large-v3-Turbo) 的高效能語音轉錄工具，專為 **N 卡 (NVIDIA GPU)** 環境優化，能全開 Tensor Cores 加速運算，並產出適合直接匯入 **Obsidian** 庫的純淨時間軸筆記。

## ✨ 核心特色
- **效能全開**：針對 N 卡架構優化，全線支援 FP16 精度與 CUDA 加速，拒絕 CPU 緩慢折磨。
- **Obsidian 友善**：自動產出帶有 `[MM:SS]` 時間戳記的格式，完美契合雙向鏈結與快速檢索美學。
- **純淨斷句**：內建 VAD 語音活動檢測，精準依文法結構斷句，徹底解決「重複字死循環」的墜機問題。
- **極簡維護**：精簡版環境設定，僅抓取核心框架，其餘百個底層相依套件由系統自動拉取，拒絕臃腫。

## 📁 專案架構
```text
.
├── bilingual_trader.py    # 核心轉錄與動態進度條腳本
├── requirements.txt       # 精簡版環境配置檔
├── inputs/                # 放置欲轉錄的音訊檔 (.wav, .mp3)
└── outputs/               # 自動產出的時間戳記文字檔 (.txt)
```
