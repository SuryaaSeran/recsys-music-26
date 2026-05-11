#!/bin/bash
# ============================================================
# MusicRecSys — One-shot setup script
# Run this from inside ~/Desktop/ReccysMusic:
#   bash setup.sh
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧹 Cleaning up stale git folders..."
rm -rf music-crs-baselines music-crs-evaluator 2>/dev/null || true

echo "📦 Cloning official repos..."
git clone https://github.com/nlp4musa/music-crs-baselines.git
git clone https://github.com/nlp4musa/music-crs-evaluator.git

echo "🐍 Installing Python dependencies..."
pip install -r src/requirements.txt --break-system-packages

echo ""
echo "✅ Setup complete! Folder layout:"
echo "   music-crs-baselines/   — official baseline (reference)"
echo "   music-crs-evaluator/   — official evaluation harness"
echo "   src/                   — our implementation"
echo ""
echo "Next steps:"
echo "  1. Download TalkPlayData-2 → data/TalkPlayData-2/"
echo "  2. python src/quantize/build_semantic_ids.py"
echo "  3. python src/train/train.py --config config/train.yaml"
echo "  4. python src/infer/run_inference.py --split dev"
echo "  5. python music-crs-evaluator/evaluate_devset.py --predictions exp/predictions_dev.json"
