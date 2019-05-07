#!/bin/bash

python src/run.py \
    "configs/toy_experiments/$1.yaml" \
    --epochs $2 \
    --workers 3 \
    --resume
