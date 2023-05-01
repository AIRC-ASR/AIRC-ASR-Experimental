#!/bin/sh
module load gcc cuda

# sbatch -vvvv -N 1 --gres=gpu:1 -t 180 -p npl -o ./slurm_outputs/generate-%J.out ./experiments/generate.sh
# 139371
python generate.py \
--model-path "out/test/model_latest.pt"  \
--add_input --learn_prior 