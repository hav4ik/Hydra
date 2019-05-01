#!/bin/bash

python src/run.py \
    configs/starter.yaml \
    --epochs 7 \
    --workers 3 \
    --resume
