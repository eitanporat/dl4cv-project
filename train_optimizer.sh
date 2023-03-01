#!/bin/bash
# Run python train.py with CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=5,6 python train_optimizer.py --optimizer_time_steps=10 --batch_size=212 --save_every=20 --parallel \
--file_dir='./generated/momentum_without_time_embedding' --optimizer_type='adamw' \
--lr=0.001 \
--total_steps=10000 \
--train_time_embedding=False \
--sampler_checkpoint='/home/eitanpo/dl4cv-eitan/generated/momentum_without_time_embedding/20230222-213158/23.9916-sampler.ckpt'