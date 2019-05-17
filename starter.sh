#!/bin/bash

python src/run.py \
    "configs/trinist/$1.yaml" \
    --epochs $2 \
    --workers 3 \
    --resume
