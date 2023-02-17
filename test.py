import torch
weights = torch.load('/home/eitanpo/dl4cv-eitan/generated/linear_optimizer/model-5.3621.ckpt')

print('a', ' '.join('{0:.4f}'.format(weights[f'sampler.layers.{i}.a'].item()) for i in range(10)))
print('b', ' '.join('{0:.4f}'.format(weights[f'sampler.layers.{i}.b'].item()) for i in range(10)))
print('c', ' '.join('{0:.4f}'.format(weights[f'sampler.layers.{i}.c'].item()) for i in range(10)))
