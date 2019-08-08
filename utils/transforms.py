import torch
from torchvision import transforms

def squeeze(x):
    return torch.squeeze(x, dim=0)

def clip_fn(min, max):
    def clip(x):
        x = torch.clamp(x, min, max)
        return x
    return clip

def scale_fn(min, max, a, b):
    def scale(x):
        x = ((b - a) * (x - min) / (max - min)) + a
        return x
    return scale

def get_transform():
    clip = clip_fn(-19, 21)
    scale = scale_fn(-19, 21, 0, 5)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(clip),
        transforms.Lambda(scale),
        transforms.Lambda(squeeze)
    ])

    return transform