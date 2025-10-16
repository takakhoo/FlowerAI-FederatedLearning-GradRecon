# flower-ASR Workspace (lisplab-1)

## Overview

### Goal
Reproduce DS1-style gradient inversion for modern ASR in a federated-learning setting using Flower’s `fedwav2vec2` baseline. We will:
1) capture per-client **model deltas** and **final-CTC-layer gradients** during a federated round,
2) reconstruct a **synthetic input** whose gradients match, and
3) visualize its **mel-spectrogram** side-by-side with the original to measure leakage (mel-MSE/PSNR) while tracking utility (WER).

### What Flower is
Flower coordinates federated rounds: the server sends weights, clients train locally, and return updates. We use it mainly to **produce the same signals a server would see** (client deltas and grads) without centralizing raw audio.

### Model & layer we target
Baseline model: **wav2vec2.0** encoder + SpeechBrain **CTC head**.  
**Target layer for inversion:** the **final CTC projection** (`modules.ctc_lin.weight`), mirroring DS1 which targeted the last FC layer.

### What we reconstruct
Given a saved gradient \(G^\* = \nabla_W \mathcal{L}(x^\*, y^\*)\) at the final CTC layer weights, we optimize a synthetic input \(z\) so that its induced gradient matches \(G^\*\). We then plot mel-spectrograms of \(z\) (recon) vs the original.

### Experiments (quick)
- **E1 Norm sensitivity (DS1 link):** freeze vs train LayerNorm; compare WER & leakage.
- **E2 Adapters vs full fine-tuning:** trade off comms size & leakage vs WER.
- **E3 Local epochs:** 1/2/4 local steps; effect on leakage for similar WER.

### Minimal workflow
1. Run a **tiny round** (1 client, 1 epoch) with hooks → writes:
   - `updates/client_{cid}_round_{r}_delta.pt`
   - `updates/grads_client_{cid}_round_{r}.pt`
2. Run `scripts/reconstruct_from_grads.py --grads <file>` → images in `viz/`.
3. Iterate E1–E3 with small overrides; log results in this README.

## Quick Commands

**Inventory (lists key files and symbols)**
```bash
bash scripts/list_baseline_inventory.sh
```

**Tiny federated run (1 round, 1 client, hooks on)**
```bash
bash scripts/run_tiny_with_hooks.sh
```

**Reconstruction from saved grads**
```bash
python scripts/reconstruct_from_grads.py \
  --grads flower/baselines/fedwav2vec2/updates/grads_client_0_round_0.pt \
  --steps 500 --lr 0.05
```

## Changelog
- 2025-10-1: Added Overview & Quick Commands sections.
- 2025-10-5: Created workspace at , cloned , installed deps, verified CUDA/Torch.
- 2025-10-8: Edited  to add:
  - Hook A: save client pre/post state deltas under .
  - Hook B: dump grads of final CTC linear () for one minibatch.
- 2025-10-10: Added Hydra flags in : , .
- 2025-10-15: (You) are now adding runner scripts, reconstruction scaffold, and verification steps.

## Paths
- Baseline root: 
- Updates (client deltas & grads): 
- Quick scripts (top-level): 
- Visualizations: 

=== Workspace tree (depth 2) ===

## Scripts
- scripts/run_tiny_with_hooks.sh: prepares TED-LIUM mapping and runs 1 round with hooks enabled; writes updates under baseline .
- scripts/list_baseline_inventory.sh: prints a baseline tree and key grep hits (ctc_lin, backward, hooks block).

## Reconstruction quickstart
After running a tiny round with hooks, pick a grads file in , e.g.:



Images will be saved in .
