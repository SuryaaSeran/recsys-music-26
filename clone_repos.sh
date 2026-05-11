#!/bin/bash
# Run this once to clone the official baseline and evaluator repos
# Usage: bash clone_repos.sh  (from inside ReccysMusic folder)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Cloning music-crs-baselines..."
git clone https://github.com/nlp4musa/music-crs-baselines.git

echo "📦 Cloning music-crs-evaluator..."
git clone https://github.com/nlp4musa/music-crs-evaluator.git

echo ""
echo "✅ Done! Both repos are now in: $SCRIPT_DIR"
echo "   music-crs-baselines/   — baseline inference systems"
echo "   music-crs-evaluator/   — local evaluation harness"
