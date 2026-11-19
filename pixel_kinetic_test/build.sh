#!/usr/bin/env bash
# One-click build. Usage:
#   ./build.sh          # song cut (default): picture re-timed to 《路口淡然警告》 -> output/song_version/
#   ./build.sh synth    # original cut with the self-synthesised chiptune         -> output/
# Requirements: python3 with numpy, pillow, scipy; ffmpeg with libx264.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-song}"
export PKT_MODE="$MODE"
if [ "$MODE" = "song" ]; then
  OUT=output/song_version
  VIDEO="$OUT/pedestrian_ke_test_pixel_song.mp4"
  AUDIO="$OUT/audio_mix.wav"
else
  OUT=output
  VIDEO="$OUT/pedestrian_ke_test_pixel.mp4"
  AUDIO="$OUT/audio_track.wav"
fi
mkdir -p "$OUT/keyframes"

echo "[1/4] audio ($MODE)"
python3 src/audio.py "$AUDIO"

echo "[2/4] rendering video ($MODE)"
python3 src/render.py "$VIDEO" --audio "$AUDIO" --preset slow --crf 10

echo "[3/4] keyframes + contact sheet + cover"
python3 tools/keyframes.py "$OUT/keyframes"
python3 tools/cover.py "$OUT/cover.png"

echo "[4/4] verify"
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels \
        -show_entries format=duration -of compact "$VIDEO"
echo "done -> $OUT/"
