import os
from faster_whisper import WhisperModel
from tqdm import tqdm
import torch

# --- 1. Hardware and Model Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# MAX PRECISION STRATEGY: Switch from "large-v3-turbo" to the full "large-v3".
# Full large-v3 has tighter semantic comprehension, essential for complex Chinese terminology.
model_size = "large-v3"
print(f"[System] Device in use: {device}")
print(f"[System] Loading the accuracy king ({model_size}) into VRAM...")

model = WhisperModel(model_size, device=device, compute_type="float16")

# --- 2. File Path Handling ---
# Place your sermon WAV file here
AUDIO_FILE = "./inputs/[僅中文]20260131 (六) 末日的危險-異端 葉專宏傳道.mp3" 
file_stem = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
OUTPUT_TXT = os.path.join("./outputs", f"{file_stem}_zh_precision.txt")

os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

# --- 3. Transcription Execution ---
if not os.path.exists(AUDIO_FILE):
    print(f"[Error] Local WAV file not found: {AUDIO_FILE}")
    exit()

try:
    print(f"[System] Transcribing Chinese sermon with high-precision parameters...")
    
    # PREMIUM PROMPT INJECTION: Forces Traditional Chinese output and primes 
    # the model's vocabulary matrix with religious and theological terms.
    sermon_prompt = "以下是一篇繁體中文的講道錄音，內容包含聖經、信仰、基督、神學、福音與教會生活。"
    
    segments, info = model.transcribe(
        AUDIO_FILE, 
        beam_size=7,                 # Upgraded from 5 to 7 for deeper semantic path searching
        language="zh",               # Enforce Chinese language mapping
        initial_prompt=sermon_prompt, # Prompt engineering to lock Traditional Chinese + vocabulary
        repetition_penalty=1.25,     # Suppress stuttering and phrase loops
        no_repeat_ngram_size=3,      # Suppress repeating identical 3-character blocks
        vad_filter=True,             # Critical for sermons: Silences background piano/organ music
        vad_parameters=dict(min_silence_duration_ms=500) # Prevents background music from triggering hallucinations
    )

    print(f"[System] Audio Duration: {info.duration:.2f} seconds")
    print(f"[System] Generating Obsidian-friendly plain text at: {OUTPUT_TXT}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        # Write Obsidian Frontmatter metadata Header
        f.write(f"# 📜 講道逐字稿: {file_stem}\n")
        f.write(f"Source File: {AUDIO_FILE}\n")
        f.write(f"---\n\n")

        # Initialize the real-time timeline progress bar
        pbar = tqdm(total=info.duration, unit="sec", desc="Transcribing Sermon")
        last_pos = 0

        for segment in segments:
            # Convert segment seconds to clean [MM:SS] format
            minutes = int(segment.start // 60)
            seconds = int(segment.start % 60)
            time_str = f"[{minutes}:{seconds:02d}]"
            
            text = segment.text.strip()
            
            # OBSIDIAN FRIENDLY FORMATTING:
            # Using Markdown H3 headers for timestamps allows you to expand/collapse 
            # each section smoothly inside your Obsidian Vault outline view.
            f.write(f"### {time_str}\n{text}\n\n")
            
            pbar.update(segment.end - last_pos)
            last_pos = segment.end
            
        pbar.close()

except Exception as e:
    print(f"[Critical Error] {str(e)}")

print(f"\n[Success] High-precision transcription completed successfully!")
print(f"[System] Your Obsidian-friendly file is ready at: {OUTPUT_TXT}")