def to_np(x):
    return x.data.cpu().numpy()

SAVE_PER_TIMES = 100

def train(module):
    for p in module.parameters():
        p.requires_grad = False

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False
