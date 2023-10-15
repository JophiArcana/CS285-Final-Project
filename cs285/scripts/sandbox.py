import torch
import numpy as np

if __name__ == '__main__':
    M = torch.arange(1, 11)[None].repeat(10, 1)
    W = torch.cumsum(M, dim=1) / torch.sum(M, dim=1, keepdim=True)
    print(W)
    k = torch.searchsorted(W, torch.rand(10, 1))
    B = torch.rand(10, 10) > 0.5
    print(0.5 * B)

