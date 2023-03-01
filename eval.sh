# CUDA_VISIBLE_DEVICES=7 python eval.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --parallel \
# --sampler_checkpoint='./generated/embedding_fine_tuning/sampler-0.3732.ckpt' \
# --time_embedding_checkpoint='./generated/embedding_fine_tuning/time-embedding-0.3732.ckpt' \
# --sampler_type='optimizer'
# CUDA_VISIBLE_DEVICES=3 python eval.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --parallel \
# --sampler_checkpoint='/home/eitanpo/dl4cv-eitan/generated/momentum_with_time_embedding/20230222-161708/0.3375-sampler.ckpt' \
# --time_embedding_checkpoint='/home/eitanpo/dl4cv-eitan/generated/momentum_with_time_embedding/20230222-161708/0.3375-time-embedding.ckpt' \
# --sampler_type='momentum'

CUDA_VISIBLE_DEVICES=3 python eval.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --parallel \
--sampler_type='ddim' \
--model_checkpoint='/home/eitanpo/dl4cv-eitan/logs/DDPM_CIFAR10_EPS/ckpt.pt'