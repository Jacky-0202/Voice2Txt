import os
from faster_whisper import WhisperModel
from tqdm import tqdm
import torch

# --- 1. Hardware and Model Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# MAX PRECISION STRATEGY: Use the full large-v3 for complex Chinese homophones.
model_size = "large-v3"
print(f"[System] Device in use: {device}")
print(f"[System] Loading the Chinese accuracy king ({model_size}) into VRAM...")

model = WhisperModel(model_size, device=device, compute_type="float16")

# --- 2. File Path Handling ---
AUDIO_FILE = "./inputs/test.m4a" 
file_stem = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
OUTPUT_TXT = os.path.join("./outputs", f"{file_stem}_zh_meeting.txt")

os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

# --- 3. Transcription Execution ---
if not os.path.exists(AUDIO_FILE):
    print(f"[Error] Local audio file not found: {AUDIO_FILE}")
    exit()

try:
    print(f"[System] Transcribing mainly Chinese meeting with specialized prompt...")
    
    # TAIWANESE MANDARIN PROMPT INJECTION: Forces Traditional Chinese output and 
    # primes the model to handle natural spoken Chinese with occasional corporate English buzzwords.
    taiwan_meeting_prompt = "這是一場主要是中文的基督教會議報告，對話內容包含專案進度、簡報討論、決議事項以及待辦清單，採用台灣繁體中文標點符號。"
    
    segments, info = model.transcribe(
        AUDIO_FILE, 
        beam_size=7,                 # Deeper path search to catch mumbled or fast Chinese speech
        language="zh",               # Lock the core decoder language to Chinese
        initial_prompt=taiwan_meeting_prompt, # Prevent simplified Chinese and optimize word choices
        repetition_penalty=1.25,     # Block verbal stuttering loops
        no_repeat_ngram_size=3,      # Suppress repeating identical phrase blocks
        vad_filter=True,             # Skip background noise, long yawns, or silence
        vad_parameters=dict(
            min_silence_duration_ms=700, # Prevents fragmentation during brief verbal pauses
            speech_pad_ms=200            # Keeps word-initial and word-final consonants intact
        )
    )

    print(f"[System] Audio Duration: {info.duration:.2f} seconds")
    print(f"[System] Generating Obsidian-friendly meeting text at: {OUTPUT_TXT}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        # Write Obsidian Frontmatter metadata Header
        f.write(f"# 👔 中文會議逐字稿: {file_stem}\n")
        f.write(f"Source File: {AUDIO_FILE}\n")
        f.write(f"---\n\n")

        # Initialize the real-time timeline progress bar
        pbar = tqdm(total=info.duration, unit="sec", desc="Transcribing Meeting")
        last_pos = 0

        for segment in segments:
            # Convert segment seconds to clean [MM:SS] format
            minutes = int(segment.start // 60)
            seconds = int(segment.start % 60)
            time_str = f"[{minutes}:{seconds:02d}]"
            
            text = segment.text.strip()
            
            # OBSIDIAN FRIENDLY FORMATTING:
            # Markdown H3 headers allow you to easily unfold/fold different sections 
            # of the meeting transcript inside your Obsidian Vault.
            f.write(f"### {time_str}\n{text}\n\n")
            
            pbar.update(segment.end - last_pos)
            last_pos = segment.end
            
        pbar.close()

except Exception as e:
    print(f"[Critical Error] {str(e)}")

print(f"\n[Success] Transcription completed successfully!")
print(f"[System] Your Obsidian-friendly Chinese file is ready at: {OUTPUT_TXT}")