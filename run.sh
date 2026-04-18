#!/bin/bash

echo "[INFO] Starting pretraining..."
python pretrain.py

if [ $? -eq 0 ]; then
    echo "[INFO] Pretraining finished successfully."
    echo "[INFO] Starting finetuning..."
    python finetuning.py
else
    echo "[ERROR] Pretraining failed. Skipping finetuning."
    exit 1
fi
