#!/bin/bash
# Run python train.py with CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=1,2,7 python train_wgan.py --optimizer_time_steps=10 \
--batch_size=64 --save_every=20 --parallel \
--lr=0.001 \
--total_steps=10000 \
--train_time_embedding=True \
--sampler_checkpoint='/home/eitanpo/dl4cv-eitan/generated/momentum_with_time_embedding/20230222-161708/0.2208-sampler.ckpt' \
--time_embedding_checkpoint='/home/eitanpo/dl4cv-eitan/generated/momentum_with_time_embedding/20230222-161708/0.2208-time-embedding.ckpt'