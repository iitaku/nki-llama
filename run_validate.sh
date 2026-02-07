#!/bin/bash
# NKI LLaMA Validation Script
# Run on trn1/inf2 instance with Neuron SDK installed
#
# This script checks accuracy only (no benchmarking) by running in "validate" mode.
# It compares NKI-optimized model logits against the baseline model.
#
# Usage:
#   ./run_validate.sh
#   MODEL_PATH=/path/to/model SEQ_LEN=128 ./run_validate.sh
#   PROMPT="Once upon a time" ./run_validate.sh

# --- Configurable variables with sensible defaults ---
MODEL_PATH=${MODEL_PATH:-"/home/ubuntu/models/llama-3.2-1b/"}
COMPILED_PATH=${COMPILED_PATH:-"/home/ubuntu/traced_model/llama-3.2-1b/"}
SEQ_LEN=${SEQ_LEN:-64}
PROMPT=${PROMPT:-"I believe the meaning of life is"}
TP_DEGREE=${TP_DEGREE:-2}

set -e

echo "============================================"
echo " NKI LLaMA Validation (accuracy check)"
echo "============================================"
echo "Model path:    $MODEL_PATH"
echo "Compiled path: $COMPILED_PATH"
echo "Seq length:    $SEQ_LEN"
echo "TP degree:     $TP_DEGREE"
echo "Prompt:        $PROMPT"
echo "============================================"
echo ""

python main.py --llama llama --enable-nki \
    --mode validate \
    --model-path "$MODEL_PATH" \
    --compiled-model-path "${COMPILED_PATH}_nki" \
    --seq-len "$SEQ_LEN" \
    --prompt "$PROMPT" \
    --tp-degree "$TP_DEGREE"

echo ""
echo "============================================"
echo " Validation complete"
echo "============================================"
