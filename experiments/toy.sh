#!/bin/bash


#######################################
#                                     #
#             NAIVE BLOCK             #
#                                     #
#######################################

python src/run.py \
    configs/toy_experiments/naive.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: naive"

python src/run.py \
    configs/toy_experiments/naive.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: naive"

python src/run.py \
    configs/toy_experiments/naive.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: naive"

python src/run.py \
    configs/toy_experiments/naive.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: naive"


#######################################
#                                     #
#          AVERAGING BLOCK            #
#                                     #
#######################################

python src/run.py \
    configs/toy_experiments/averaging.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: averaging"

python src/run.py \
    configs/toy_experiments/averaging.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: averaging"

python src/run.py \
    configs/toy_experiments/averaging.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: averaging"

python src/run.py \
    configs/toy_experiments/averaging.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: averaging"


#######################################
#                                     #
#             MGDA BLOCK              #
#                                     #
#######################################

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: mgda" \
    --update "trainer.kwargs.mode: phase_2"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: mgda" \
    --update "trainer.kwargs.mode: phase_2"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: mgda" \
    --update "trainer.kwargs.mode: phase_2"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 120 \
    --workers 4 \
    --update "experiment: mgda" \
    --update "trainer.kwargs.mode: phase_2"


#######################################
#                                     #
#           MGDA-NORM BLOCK           #
#                                     #
#######################################

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 80 \
    --workers 4 \
    --update "experiment: mgda-norm" \
    --update "trainer.kwargs.mode: phase_1" \
    --update "trainer.kwargs.patience: 10"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 40 \
    --workers 4 \
    --update "experiment: mgda-norm-000" \
    --update "trainer.kwargs.mode: phase_2" \
    --update "trainer.kwargs.normalize: null" \
    --resume


python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 80 \
    --workers 4 \
    --update "experiment: mgda-norm" \
    --update "trainer.kwargs.mode: phase_1" \
    --update "trainer.kwargs.patience: 10"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 40 \
    --workers 4 \
    --update "experiment: mgda-norm-001" \
    --update "trainer.kwargs.mode: phase_2" \
    --update "trainer.kwargs.normalize: null" \
    --resume


python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 80 \
    --workers 4 \
    --update "experiment: mgda-norm" \
    --update "trainer.kwargs.mode: phase_1" \
    --update "trainer.kwargs.patience: 10"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 40 \
    --workers 4 \
    --update "experiment: mgda-norm-002" \
    --update "trainer.kwargs.mode: phase_2" \
    --update "trainer.kwargs.normalize: null" \
    --resume


python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 80 \
    --workers 4 \
    --update "experiment: mgda-norm" \
    --update "trainer.kwargs.mode: phase_1" \
    --update "trainer.kwargs.patience: 10"

python src/run.py \
    configs/toy_experiments/mgda.yaml \
    --epochs 40 \
    --workers 4 \
    --update "experiment: mgda-norm-003" \
    --update "trainer.kwargs.mode: phase_2" \
    --update "trainer.kwargs.normalize: null" \
    --resume
