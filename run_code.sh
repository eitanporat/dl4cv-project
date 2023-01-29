#!/bin/bash
# Run python train.py with CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=2,3,4,5,6 python train.py --optimizer_time_steps=10 --batch_size=384 --save_every=20 --parallel