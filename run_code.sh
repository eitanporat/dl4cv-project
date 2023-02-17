#!/bin/bash
# Run python train.py with CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2 python train-without-train-ddim.py --optimizer_time_steps=10 --batch_size=42 --save_every=20 --lr=0.01 --parallel