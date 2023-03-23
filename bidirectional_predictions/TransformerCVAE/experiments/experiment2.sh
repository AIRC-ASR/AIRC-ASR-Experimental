#!/bin/sh
module load gcc cuda
# sbatch -vvvv -N 1 --gres=gpu:1 -t 360 -p npl -o ./slurm_outputs/slurmoutput-%J.out ./experiments/experiment2.sh

# 1 2 0 0
python train_bidirectional.py \
  test --add_input --learn_prior --fp16 --switch-time 0.5 \
  --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 \
  --short_seq_len 1024 --long_seq_len 1024 \
  --fwd_loss_weight 1 --bkwd_loss_weight 2 --all_sentence_loss_weight 0 \
  --prompt_loss_weight 0