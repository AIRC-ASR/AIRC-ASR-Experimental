#!/bin/sh
module load gcc cuda

source venv/bin/activate
python finetune_common_voice.py