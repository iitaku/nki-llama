#!/bin/bash
# NKI LLaMA Evaluation Script
# Run on trn1/inf2 instance with Neuron SDK installed
#
# This script runs both the NKI-optimized model and the baseline model,
# performing accuracy checks, benchmarking, and scoring via "evaluate_single" mode.
#
# Usage:
#   ./run_evaluate.sh
#   MODEL_PATH=/path/to/model SEQ_LEN=128 ./run_evaluate.sh
#   PROMPT="Once upon a time" ./run_evaluate.sh

# --- Configurable variables with sensible defaults ---
MODEL_PATH=${MODEL_PATH:-"/home/ubuntu/models/llama-3.2-1b/"}
COMPILED_PATH=${COMPILED_PATH:-"/home/ubuntu/traced_model/llama-3.2-1b/"}
SEQ_LEN=${SEQ_LEN:-64}
PROMPT=${PROMPT:-"I believe the meaning of life is"}
TP_DEGREE=${TP_DEGREE:-2}

set -e

echo "============================================"
echo " NKI LLaMA Evaluation"
echo "============================================"
echo "Model path:    $MODEL_PATH"
echo "Compiled path: $COMPILED_PATH"
echo "Seq length:    $SEQ_LEN"
echo "TP degree:     $TP_DEGREE"
echo "Prompt:        $PROMPT"
echo "============================================"
echo ""

echo "=== Running NKI-enabled model (evaluate_single) ==="
python main.py --llama llama --enable-nki \
    --mode evaluate_single \
    --model-path "$MODEL_PATH" \
    --compiled-model-path "${COMPILED_PATH}_nki" \
    --seq-len "$SEQ_LEN" \
    --prompt "$PROMPT" \
    --tp-degree "$TP_DEGREE"

echo ""
echo "=== Running baseline model (evaluate_single) ==="
python main.py --llama llama_baseline \
    --mode evaluate_single \
    --model-path "$MODEL_PATH" \
    --compiled-model-path "${COMPILED_PATH}_baseline" \
    --seq-len "$SEQ_LEN" \
    --prompt "$PROMPT" \
    --tp-degree "$TP_DEGREE"

echo ""
echo "============================================"
echo " Evaluation complete"
echo "============================================"
