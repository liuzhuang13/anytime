import torch
import torch.nn as nn
import pdb, time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class conv_mask_uniform(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, p=0.5, interpolate='none'):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.mask = None 
        self.mask_built = False 
        self.p = p

        self.interpolate = interpolate
        self.r = 7
        self.padding_interpolate = 3

        self.Lambda = nn.Parameter(torch.tensor(3.0))
        square_dis = np.zeros((self.r, self.r))
        center_point = (square_dis.shape[0]//2, square_dis.shape[1]//2)

        for i in range(square_dis.shape[0]):
            for j in range(square_dis.shape[1]):
                square_dis[i][j] = (i - center_point[0])**2 + (j - center_point[1])**2

        square_dis[center_point[0]][center_point[1]] = 100000.0

        self.square_dis = nn.Parameter(torch.Tensor(square_dis), requires_grad=False)

    def build_mask(self, x):
        mask_p = x.new(x.shape[2:]).fill_(self.p)
        mask = torch.bernoulli(mask_p)
        self.mask = mask[None, None, :, :].float()
        self.mask_built = True

        if self.in_channels == 3:
            print('Mask sum:', torch.sum(self.mask))

    def build_mask_random(self, x):
        mask_p = x.new(size=(x.shape[0], *x.shape[2:])).fill_(self.p)
        mask = torch.bernoulli(mask_p)
        self.mask = mask[:, None, :, :].float()
        self.mask_built = True

    def set_mask(self, mask):
        self.mask = mask[:, None, :, :]
        self.mask_built = True

    def forward(self, x):
        y = super().forward(x)
        self.out_h, self.out_w = y.size(-2), y.size(-1)
        if not self.mask_built:
            self.build_mask_random(y)

        kernel = (-(self.Lambda**2) * self.square_dis.detach()).exp()
        kernel = kernel / (kernel.sum() + 10**(-5)) 
        kernel = kernel.expand((self.out_channels, 1, kernel.size(0), kernel.size(1)))
        interpolated = F.conv2d(y * self.mask, kernel, stride=1, padding=self.padding_interpolate, groups=self.out_channels)

        out = y * self.mask + interpolated * (1 - self.mask)
        self.mask_built = False

        return out

if __name__ == '__main__':
    a = Smooth(n_channels=10, kernel_size=3, padding=1)

