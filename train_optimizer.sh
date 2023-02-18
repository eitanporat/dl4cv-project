#!/bin/bash
# Run python train.py with CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=3,4,5,6 python train_optimizer.py --optimizer_time_steps=10 --batch_size=256 --save_every=20 --parallel