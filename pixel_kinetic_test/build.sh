#!/usr/bin/env bash
# One-click build: audio -> video (1080x1920, 30 fps, H.264 + AAC) -> keyframes + contact sheet + cover.
# Requirements: python3 with numpy, pillow, scipy; ffmpeg with libx264.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output/keyframes

echo "[1/4] synthesising audio track"
python3 src/audio.py output/audio_track.wav

echo "[2/4] rendering video"
python3 src/render.py output/pedestrian_ke_test_pixel.mp4 --audio output/audio_track.wav --preset slow --crf 10

echo "[3/4] keyframes + contact sheet + cover"
python3 tools/keyframes.py output/keyframes
python3 tools/cover.py output/cover.png

echo "[4/4] verify"
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels \
        -show_entries format=duration -of compact output/pedestrian_ke_test_pixel.mp4
echo "done -> output/"
