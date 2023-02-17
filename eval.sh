# CUDA_VISIBLE_DEVICES=7 python eval.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --parallel \
# --sampler_checkpoint='./generated/embedding_fine_tuning/sampler-0.3732.ckpt' \
# --time_embedding_checkpoint='./generated/embedding_fine_tuning/time-embedding-0.3732.ckpt' \
# --sampler_type='optimizer'
CUDA_VISIBLE_DEVICES=7 python eval.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --parallel \
--sampler_checkpoint='./generated/linear_optimizer/model-5.3621.ckpt' \
--sampler_type='optimizer'