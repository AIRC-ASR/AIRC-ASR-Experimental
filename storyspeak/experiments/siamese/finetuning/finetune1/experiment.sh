#!/bin/sh
module load gcc cuda

source venv/bin/activate
python train_siamese_network.py