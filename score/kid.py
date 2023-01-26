from torch import nn
from .inception import InceptionV3
import torch.nn.functional as F

class KernelInceptionDistance(nn.Module):
    def __init__(self, parallel=True) -> None:
        super(KernelInceptionDistance, self).__init__()
        block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx1])
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        if parallel:
            model = nn.DataParallel(model)

    # are you kidding me?
    def forward(self, generated_images, real_images):
        assert generated_images.shape == real_images.shape

        n = generated_images.size(0)

        gen_features = self.model(generated_images)[0]
        gen_features = gen_features.flatten(start_dim=1)
        # gen_features = gen_features - gen_features.mean(axis=0, keepdim=True)
        # gen_features = gen_features / gen_features.std(axis=0, keepdim=True)
        # gen_features = F.normalize(gen_features, dim=1)

        real_features = self.model(real_images)[0]
        real_features = real_features.flatten(start_dim=1)
        # real_features = real_features - real_features.mean(axis=0, keepdim=True)
        # real_features = real_features / real_features.std(axis=0, keepdim=True)
        # real_features = F.normalize(gen_features, dim=1)

        cov = gen_features @ gen_features.transpose(0, 1)
        
        score1 = 1./(n*(n-1)) * (cov.sum() - cov.trace())
        score2 = 1./(n*n) * (gen_features @ real_features.transpose(0, 1)).sum() 
        # try other kernels

        return score1 - score2