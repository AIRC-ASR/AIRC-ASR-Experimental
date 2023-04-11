#!/bin/sh
module load gcc cuda

source venv/bin/activate
python finetune_nsp_librispeech.py