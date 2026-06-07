import os
from faster_whisper import WhisperModel
from tqdm import tqdm
import torch

# --- 1. Hardware and Model Initialization ---
# Target Device: NVIDIA RTX 5070 Ti (Compute Capability 12.0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "large-v3-turbo"
print(f"[System] Device in use: {device}")
print(f"[System] Loading Whisper model ({model_size}) into VRAM...")

# Load model with float16 precision for maximum inference speed on Tensor Cores
model = WhisperModel(model_size, device=device, compute_type="float16")

# --- 2. File Path Handling ---
# Target local WAV file (e.g., your converted TED-Ed video audio)
AUDIO_FILE = "./inputs/audio.wav" 
file_stem = os.path.splitext(os.path.basename(AUDIO_FILE))[0]
OUTPUT_TXT = os.path.join("./outputs", f"{file_stem}.txt") # Enforced .txt output

os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

# --- 3. Transcription Execution ---
if not os.path.exists(AUDIO_FILE):
    print(f"[Error] Local WAV file not found: {AUDIO_FILE}")
    exit()

try:
    print(f"[System] Transcribing source audio...")
    
    # Enforce language="en" for English speech. 
    # Change to language="zh" if you use this script for Chinese meetings later.
    segments, info = model.transcribe(
        AUDIO_FILE, 
        beam_size=5,
        language="en", 
        repetition_penalty=1.2,     # Prevent hallucination/stuttering loops
        no_repeat_ngram_size=3,      # Prevent repeating identical 3-word clusters
        vad_filter=True              # Automatically skip blank audio spaces to save time
    )

    print(f"[System] Audio Duration: {info.duration:.2f} seconds")
    print(f"[System] Generating plain text file at: {OUTPUT_TXT}")

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        # File metadata header
        f.write(f"Source File: {AUDIO_FILE}\n")
        f.write(f"Detected Language: {info.language} ({info.language_probability:.2f})\n")
        f.write(f"---\n\n")

        # Initialize the real-time progress bar linked to the audio timeline
        pbar = tqdm(total=info.duration, unit="sec", desc="Transcribing Timeline")
        last_pos = 0

        for segment in segments:
            # Convert segment seconds to standard [MM:SS] format
            minutes = int(segment.start // 60)
            seconds = int(segment.start % 60)
            time_str = f"[{minutes}:{seconds:02d}]"
            
            text = segment.text.strip()
            
            # Write a clean, single line per segment into the .txt file
            f.write(f"{time_str} {text}\n")
            
            # Update the timeline progress bar dynamically
            pbar.update(segment.end - last_pos)
            last_pos = segment.end
            
        pbar.close()

except Exception as e:
    print(f"[Critical Error] {str(e)}")

print(f"\n[Success] Process completed successfully!")
print(f"[System] Your pure text note is ready at: {OUTPUT_TXT}")