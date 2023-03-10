#!/bin/sh
module load gcc cuda
# sbatch -vvvv -N 1 --gres=gpu:8 -t 360 -p npl -o ./slurm_outputs/slurmoutput-%J.out ./experiments/experiment4.sh

# 0.33 0.33 0 0.33
python train_bidirectional_dist.py \
  test --add_input --learn_prior --fp16 --switch-time 0.5 \
  --train_batch_size 1 --val_batch_size 1 --test_batch_size 1 \
  --short_seq_len 1024 --long_seq_len 1024 \
  --fwd_loss_weight 0.33 --bkwd_loss_weight 0.33 --all_sentence_loss_weight 0.0 \
  --prompt_loss_weight 0.33