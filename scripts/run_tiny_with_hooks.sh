#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/f004h1v/flower-ASR/flower/baselines/fedwav2vec2

# Data mapping (prefer existing tiny client; skip external clone if present)
mkdir -p data
if [ -f data/client_0/ted_train.csv ]; then
  echo "[run_tiny_with_hooks] Found data/client_0/ted_train.csv, skipping external mapping clone."
else
  if [ ! -d _temp ]; then
    git clone https://github.com/tuanct1997/Federated-Learning-ASR-based-on-wav2vec-2.0.git _temp || true
  fi
  mv -f _temp/data/* data/ 2>/dev/null || true
  rm -rf _temp || true
fi

# Prepare TED-LIUM 3 partitions only if tiny client not present
if [ -f data/client_0/ted_train.csv ]; then
  echo "[run_tiny_with_hooks] Tiny CSV present; skipping dataset_preparation."
else
  python -m fedwav2vec2.dataset_preparation || true
fi

# Print one CSV header for sanity
head -5 data/client_0/ted_train.csv || true

# Ensure a minimal server split exists to satisfy centralized evaluation
if [ ! -f data/server/ted_train.csv ]; then
  echo "[run_tiny_with_hooks] Creating minimal server split by copying client_0 CSVs."
  mkdir -p data/server
  cp -f data/client_0/ted_train.csv data/server/ted_train.csv || true
  cp -f data/client_0/ted_dev.csv data/server/ted_dev.csv || true
  cp -f data/client_0/ted_test.csv data/server/ted_test.csv || true
fi

# Run a SINGLE round, ONE client, minimal local work, with hooks on
# Force CPU for the smoke test to avoid GPU downloads/memory issues
CUDA_VISIBLE_DEVICES= \
python -m fedwav2vec2.main \
  rounds=1 local_epochs=1 total_clients=1 \
  strategy.min_fit_clients=1 strategy.fraction_fit=1.0 \
  client_resources.num_gpus=0 client_resources.num_cpus=4 \
  server_device=cpu data_path=data 'dataset.extract_subdirectory=audio' \
  hooks.save_updates=true hooks.save_grads=true

echo "=== Updates written to: $(realpath updates) ==="
ls -lh updates || true

#!/usr/bin/env bash
set -euo pipefail
cd /scratch2/f004h1v/flower-ASR/flower/baselines/fedwav2vec2

# Data mapping (safe to re-run)
mkdir -p data
if [ ! -d _temp ]; then
  git clone https://github.com/tuanct1997/Federated-Learning-ASR-based-on-wav2vec-2.0.git _temp
fi
mv -f _temp/data/* data/ 2>/dev/null || true
rm -rf _temp || true

# Prepare TED-LIUM 3 partitions (script also fixes paths)
python -m fedwav2vec2.dataset_preparation

# Print one CSV header for sanity
head -5 data/client_0/ted_train.csv || true

# Run a SINGLE round, ONE client, minimal local work, with hooks on
python -m fedwav2vec2.main   rounds=1 local_epochs=1 total_clients=1   strategy.min_fit_clients=1 strategy.fraction_fit=1.0   client_resources.num_gpus=1 client_resources.num_cpus=4   sb_config=fedwav2vec2/conf/sb_config/w2v2.yaml   data_path=data 'dataset.extract_subdirectory=audio'   +hooks.save_updates=true +hooks.save_grads=true

echo ===
