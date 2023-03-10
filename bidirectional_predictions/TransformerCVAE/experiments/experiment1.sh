#!/bin/sh
module load gcc cuda
# sbatch -vvvv -N 1 --gres=gpu:8 -t 360 -p npl -o ./slurm_outputs/slurmoutput-%J.out ./experiments/experiment1.sh

# 1 0 0 0
# slurmoutput-104848
# slurmoutput-104945
python train_bidirectional_dist.py \
  test --add_input --learn_prior --fp16 --switch-time 0.5 \
  --train_batch_size 5 --val_batch_size 5 --test_batch_size 5 \
  --short_seq_len 1024 --long_seq_len 1024 \
  --fwd_loss_weight 1 --bkwd_loss_weight 0 --all_sentence_loss_weight 0 \
  --prompt_loss_weight 0 \
#  --reload_path "out/test/model_0040000_bidirectional_1.0_2.0_0.0_0.0.pt" \
#  --reload_epoch 0 --reload_iters 40000 --reload_batches 20