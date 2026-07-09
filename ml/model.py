"""Small fully-convolutional residual UNet that corrects GFS REFC toward observed dBZ.

Fully convolutional => train on random 128x128 patches, run inference on the full
721x1440 frame. Predicts a residual (obs_dBZ - gfs_dBZ), so an untrained/zeroed
final layer is the identity.

Deliberately shallow and narrow (2 pooling levels, base=16): this task is learning
GFS's smooth *systematic* bias against observations, not fine texture, so a small net
generalizes well AND stays cheap enough to run on CPU inside GitHub Actions for every
frame (~1s/frame). Height/width must be multiples of 4 (two 2x pools); the inference
hook in pipeline/correct.py pads to that and crops back.
"""
import torch
import torch.nn as nn

POOL_DIVISOR = 4  # 2 pooling levels => input H,W padded to a multiple of 4


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
    )


class RefcUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)  # start as identity: residual = 0
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)  # residual added to input dBZ by the caller
