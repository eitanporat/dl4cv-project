def to_np(x):
    return x.data.cpu().numpy()

def train(module):
    for p in module.parameters():
        p.requires_grad = True

    module.zero_grad()

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False