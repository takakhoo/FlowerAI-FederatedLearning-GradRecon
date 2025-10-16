#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/f004h1v/flower-ASR/flower/baselines/fedwav2vec2

echo "=== Baseline tree (depth 3) ==="
tree -L 3 .

echo "=== Files of interest ==="
echo "[entry] fedwav2vec2/main.py"
echo "[client] fedwav2vec2/client.py"
echo "[server] fedwav2vec2/server.py"
echo "[strategy] fedwav2vec2/strategy.py"
echo "[sb recipe] fedwav2vec2/sb_recipe.py"
echo "[model build] fedwav2vec2/models.py"
echo "[dataset] fedwav2vec2/dataset.py"
echo "[prep] fedwav2vec2/dataset_preparation.py"
echo "[config] fedwav2vec2/conf/base.yaml"
echo "[sb cfg] fedwav2vec2/conf/sb_config/w2v2.yaml"

echo "=== grep: ctc_lin locations ==="
nl -ba fedwav2vec2/sb_recipe.py | grep -n "ctc_lin" || true
nl -ba fedwav2vec2/conf/sb_config/w2v2.yaml | grep -n "ctc_lin" || true

echo "=== grep: loss.backward() ==="
grep -n "backward()" fedwav2vec2/*.py || true

echo "=== show hydra hooks block ==="
awk '/^hooks:/{flag=1} flag{print NR":"$0}' fedwav2vec2/conf/base.yaml || true


