#!/bin/sh
module load gcc cuda
# sbatch -vvvv -N 1 --gres=gpu:1 -t 360 -p npl -o ./slurm_outputs/slurmoutput-%J.out ./experiments/experiment3.sh

# 0.5 0 0 0.5
python train_bidirectional.py \
  test --add_input --learn_prior --fp16 --switch-time 0.5 \
  --train_batch_size 3 --val_batch_size 3 --test_batch_size 3 \
  --short_seq_len 1024 --long_seq_len 1024 \
  --fwd_loss_weight 0.5 --bkwd_loss_weight 0 --all_sentence_loss_weight 0 \
  --prompt_loss_weight 0.5