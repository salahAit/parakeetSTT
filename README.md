# NVIDIA Parakeet Transcription Tool

This tool uses the latest NVIDIA Parakeet models (via NeMo) to transcribe video and audio files. It automatically handles audio extraction from video files and leverages GPU acceleration.

## Features
- Transcribes both audio and video files.
- Supports batch processing (multiple files or entire directories).
- Uses the state-of-the-art **Parakeet-TDT 0.6B-v3** model by default.
- Automatically extracts audio from video using FFmpeg.
- Saves transcriptions to `.txt` files alongside the original media.

## Setup
The tool is already set up in a virtual environment.

```bash
source venv/bin/activate
```

## Usage

### Transcribe a single file
```bash
python transcribe.py path/to/video.mp4
```

### Transcribe multiple files
```bash
python transcribe.py file1.wav file2.mp4
```

### Transcribe an entire directory
```bash
python transcribe.py /path/to/media_folder/
```

### Use a different model
```bash
python transcribe.py video.mp4 --model nvidia/parakeet-rnnt-1.1b
```

## Requirements
- Python 3.12+
- FFmpeg
- NVIDIA GPU with CUDA drivers
- NeMo Toolkit (installed in `venv`)
