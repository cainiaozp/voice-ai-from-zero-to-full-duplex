#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Optional quick CPU smoke: does not run pretrained model downloads.
python scripts/verify_repo.py
python extras/phase/phase_lab.py --out outputs/smoke_phase
python extras/esc50-normalization/compare_normalization.py --out outputs/smoke_esc50.png
python extras/dense-vs-moe/moe_toy.py
python lessons/03-whisper-asr/code/cross_attention_toy.py
python lessons/10-realtime-system/code/lesson10_simulator.py --out outputs/smoke_lesson10
