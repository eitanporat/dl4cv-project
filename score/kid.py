import torch
from torch import nn
from .inception import InceptionV3

class KernelInceptionDistance(nn.Module):
    def __init__(self, parallel=True) -> None:
        super(KernelInceptionDistance, self).__init__()
        block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx1], normalize_input=False)
        self.model.eval()

    # are you kidding me?
    def forward(self, generated_images, real_images):
        if generated_images.shape != real_images.shape:
            print(generated_images.shape, real_images.shape)
            
        assert generated_images.shape == real_images.shape

        n = generated_images.size(0)
        chunks = n // 8
        
        gen_features = torch.cat([self.model(chunk)[0] for chunk in generated_images.chunk(chunks)])
        gen_features = gen_features.flatten(start_dim=1)

        real_features = torch.cat([self.model(chunk)[0] for chunk in real_images.chunk(chunks)])
        real_features = real_features.flatten(start_dim=1)

        cov1 = gen_features @ gen_features.transpose(0, 1)
        cov2 = real_features @ real_features.transpose(0, 1)

        score1 = 1./(n*(n-1)) * (cov1.sum() - cov1.trace())
        score2 = 1./(n*(n-1)) * (cov2.sum() - cov2.trace())
        score3 = 1./(n*n) * (gen_features @ real_features.transpose(0, 1)).sum() 
        # try other kernels

        return score1 + score2 - 2 * score3