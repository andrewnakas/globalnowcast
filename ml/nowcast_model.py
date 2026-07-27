"""Sequence model: 4 hourly radar frames + HRRR forecast -> 6 hourly radar frames.

This is a different job from ml/model.py, which de-biases one GFS frame. Here the
model has to both extrapolate the radar it is given and correct the model forecast it
is shown, which is the DGMR/SEVIR formulation with HRRR supplied as a side input.

Design is set by where the skill actually comes from, measured against radar truth:

  HRRR alone, no nowcasting   CSI 0.650 at 1 mm/h
  satellite advection         CSI 0.186 at +30 min
  persistence                 CSI 0.184 at +30 min

So HRRR is a strong prior and the model should lean on it rather than relearn it. It
is concatenated with the radar history at the input, and the head predicts a residual
on top of HRRR rather than the field from scratch - the same identity-at-init trick
ml/model.py uses, which means an untrained model scores exactly HRRR's 0.650 instead
of noise, and training can only add to that.

Small on purpose: this has to run on CPU in the hourly job, where the budget is about
a second a frame.
"""
import torch
import torch.nn as nn

IN_FRAMES = 4
OUT_FRAMES = 6
U8_LO, U8_HI = -30.0, 60.0
POOL_DIVISOR = 8  # three 2x pooling levels


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
    )


class NowcastUNet(nn.Module):
    """3-level UNet over stacked frames. Input H, W must be multiples of 8.

    Channels in: IN_FRAMES radar history + OUT_FRAMES HRRR (one per target lead), so
    the model can see what the model forecast expects at each step it predicts.
    """

    def __init__(self, base=32, in_frames=IN_FRAMES, out_frames=OUT_FRAMES,
                 use_hrrr=True):
        super().__init__()
        self.out_frames = out_frames
        self.use_hrrr = use_hrrr
        cin = in_frames + (out_frames if use_hrrr else 0)

        self.enc1 = _block(cin, base)
        self.enc2 = _block(base, base * 2)
        self.enc3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = _block(base * 4, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = _block(base * 8, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, out_frames, 1)
        # Start as a pass-through of HRRR: an untrained model then scores whatever
        # HRRR scores, and training strictly adds.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, radar, hrrr=None):
        """radar (B, IN, H, W) dBZ, hrrr (B, OUT, H, W) dBZ -> (B, OUT, H, W) dBZ.

        H and W are padded up to a multiple of 8 (three pooling levels) and cropped
        back, so the 100x100 tiles the dataset ships work unchanged. Same approach as
        pipeline/correct.py.
        """
        h0, w0 = radar.shape[-2:]
        ph, pw = (-h0) % POOL_DIVISOR, (-w0) % POOL_DIVISOR
        if ph or pw:
            radar = nn.functional.pad(radar, (0, pw, 0, ph), mode="reflect")
            if hrrr is not None:
                hrrr = nn.functional.pad(hrrr, (0, pw, 0, ph), mode="reflect")

        x = radar if hrrr is None else torch.cat([radar, hrrr], 1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        residual = self.head(d1)
        if hrrr is not None and self.use_hrrr:
            out = hrrr + residual
        else:
            # Without HRRR the reference is the most recent radar frame, i.e. the
            # model is learning a correction to persistence.
            out = radar[:, -1:].expand_as(residual) + residual
        return out[..., :h0, :w0]


def weighted_l1(pred, target, wet_weight=8.0, wet_dbz=23.0):
    """L1, weighted towards raining cells.

    Rain covers a few percent of a typical tile, so an unweighted loss is dominated by
    correctly predicting dry sky and the model converges to forecasting nothing. The
    weight is what stops that; 8x is a starting point, not a tuned value.
    """
    w = torch.where(target >= wet_dbz, wet_weight, 1.0)
    return (w * (pred - target).abs()).mean()


def csi_torch(pred, target, thresh):
    """CSI at one threshold, for logging during training."""
    p, t = pred >= thresh, target >= thresh
    hits = (p & t).sum().float()
    denom = (p | t).sum().float()
    return hits / denom if denom > 0 else torch.tensor(float("nan"))
