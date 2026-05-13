import os

# Set environment variables BEFORE importing torch/nemo
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import subprocess
import torch
import nemo.collections.asr as nemo_asr
from pathlib import Path
import tempfile
import time
import wave
import gc

# Enable CuDNN and TF32 for optimal GPU performance
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True

def extract_audio_to_memory(video_path):
    """Extracts 16kHz mono audio directly into RAM."""
    print(f"[*] Extracting audio from {video_path} to RAM...")
    start_time = time.time()
    command = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-f", "s16le", "-"
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_data, err = process.communicate()
        if process.returncode != 0:
            print(f"[!] FFmpeg error: {err.decode()}")
            return None
        print(f"[+] Extracted {len(audio_data)/(1024*1024):.2f} MB to RAM in {time.time() - start_time:.2f}s")
        return audio_data
    except Exception as e:
        print(f"[!] Extraction error: {e}")
        return None

def process_file(file_path, model):
    audio_data = extract_audio_to_memory(file_path)
    if not audio_data: return

    file_path = Path(file_path)
    base_name = file_path.stem
    output_dir = file_path.parent / f"{base_name}_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the full extracted audio as a proper WAV file
    audio_file_path = output_dir / f"{base_name}.wav"
    with wave.open(str(audio_file_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_data)
    print(f"[*] Saved extracted audio to: {audio_file_path}")

    # 1-minute chunks for extreme stability
    chunk_size = 16000 * 2 * 60 * 1
    total_bytes = len(audio_data)
    
    temp_files = []
    print(f"[*] Splitting into {max(1, total_bytes // chunk_size + 1)} chunks...")
    
    # Save chunks as temporary files first
    for i in range(0, total_bytes, chunk_size):
        chunk = audio_data[i : i + chunk_size]
        tfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tfile.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(chunk)
        temp_files.append(tfile.name)
        tfile.close()

    try:
        print(f"[*] Transcribing {len(temp_files)} chunks in a single managed batch...")
        start_time = time.time()
        
        # Force a clean slate
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad(), torch.autocast('cuda'):
            # Passing all paths to model.transcribe is more stable than a loop
            transcriptions = model.transcribe(temp_files, batch_size=24)

        full_text = []
        srt_segments = []
        for i, res in enumerate(transcriptions):
            text = res.text if hasattr(res, 'text') else str(res)
            full_text.append(text)
            
            start_sec = i * 60
            end_sec = start_sec + 60
            srt_segments.append((start_sec, end_sec, text))

        final_text = " ".join(full_text)
        elapsed = time.time() - start_time

        print("\n" + "═"*50)
        print(f" FILE: {Path(file_path).name}")
        print(f" TIME: {elapsed:.2f}s")
        print("═"*50)
        print(final_text[:1000] + "..." if len(final_text) > 1000 else final_text)
        print("═"*50 + "\n")

        # Save TXT
        txt_path = output_dir / f"{base_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(final_text)

        # Generate Word Timestamps (Simulated evenly over chunks)
        import json
        flat_words = []
        word_index = 1
        word_map = {}
        for (start, end, text) in srt_segments:
            words = text.split()
            if not words: continue
            duration = end - start
            time_per_word = duration / len(words)
            for i, w in enumerate(words):
                w_start = start + i * time_per_word
                w_end = w_start + time_per_word
                flat_words.append({
                    "word": w,
                    "start": round(w_start, 2),
                    "end": round(w_end, 2),
                    "probability": 1.0
                })
                word_map[str(word_index)] = w
                word_index += 1

        json_map_path = output_dir / f"{base_name}_word_map.json"
        with open(json_map_path, 'w', encoding='utf-8') as f:
            json.dump(word_map, f, indent=2, ensure_ascii=False)

        json_time_path = output_dir / f"{base_name}_word_timestamps.json"
        with open(json_time_path, 'w', encoding='utf-8') as f:
            json.dump(flat_words, f, indent=2, ensure_ascii=False)

        # Create smaller segments for SRT and JSON (split by punctuation or max 12 words)
        import re
        small_segments = []
        current_chunk = []
        
        for w_dict in flat_words:
            current_chunk.append(w_dict)
            word_text = w_dict['word']
            
            # End segment if it has punctuation or reaches 12 words
            if re.search(r'[.?!،؛]', word_text) or len(current_chunk) >= 12:
                seg_start = current_chunk[0]['start']
                seg_end = current_chunk[-1]['end']
                seg_text = " ".join([w['word'] for w in current_chunk])
                small_segments.append((seg_start, seg_end, seg_text))
                current_chunk = []

        if current_chunk:
            seg_start = current_chunk[0]['start']
            seg_end = current_chunk[-1]['end']
            seg_text = " ".join([w['word'] for w in current_chunk])
            small_segments.append((seg_start, seg_end, seg_text))

        # Save SRT
        def format_timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

        srt_path = output_dir / f"{base_name}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            for idx, (start, end, text) in enumerate(small_segments, start=1):
                if text.strip():
                    f.write(f"{idx}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text.strip()}\n\n")

        # Save JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([{"start": start, "end": end, "text": text.strip()} for start, end, text in small_segments if text.strip()], f, indent=2, ensure_ascii=False)

        print(f"[+] Saved outputs to: {output_dir}")

    except Exception as e:
        print(f"[!] Transcription failed: {e}")
    finally:
        for f in temp_files:
            if os.path.exists(f): os.remove(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    args = parser.parse_args()

    print(f"[*] Initializing {args.model} (CuDNN Enabled for performance)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model = model.to(device)
    model.eval()

    supported = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm', '.ts', '.m4v', '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    for input_path in args.inputs:
        p = Path(input_path)
        if p.is_dir():
            for f in p.iterdir():
                if f.suffix.lower() in supported: process_file(f, model)
        elif p.exists() and p.suffix.lower() in supported:
            process_file(p, model)

if __name__ == "__main__":
    main()
