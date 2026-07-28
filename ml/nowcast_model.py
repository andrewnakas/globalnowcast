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


class GatedFusion(nn.Module):
    """Learned gate on a skip connection, instead of a plain concatenate.

    The model is fed two very different things - observed radar history and an HRRR
    forecast - and how much to trust each varies with lead time and with the weather.
    Persistence wins at +1 h and HRRR wins from +2 h out, so a fixed combination is
    wrong at one end or the other. A gate lets the network learn that weighting per
    cell rather than having it baked in.

    Ablations in the nowcasting literature (e.g. SwinNowcast's GAFFU) find gated
    attention fusion to be the component that matters most, and it is far cheaper than
    the transformer blocks that usually accompany it.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, up, skip):
        g = self.gate(torch.cat([up, skip], 1))
        return torch.cat([up, g * skip], 1)


class NowcastUNet(nn.Module):
    """3-level UNet over stacked frames. Input H, W must be multiples of 8.

    Channels in: IN_FRAMES radar history + OUT_FRAMES HRRR (one per target lead), so
    the model can see what the model forecast expects at each step it predicts.
    """

    def __init__(self, base=32, in_frames=IN_FRAMES, out_frames=OUT_FRAMES,
                 use_hrrr=True, gated=False):
        super().__init__()
        self.out_frames = out_frames
        self.use_hrrr = use_hrrr
        self.gated = gated
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
        if gated:
            self.g3 = GatedFusion(base * 4)
            self.g2 = GatedFusion(base * 2)
            self.g1 = GatedFusion(base)
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
        if self.gated:
            d3 = self.dec3(self.g3(self.up3(b), e3))
            d2 = self.dec2(self.g2(self.up2(d3), e2))
            d1 = self.dec1(self.g1(self.up1(d2), e1))
        else:
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


def probability_match(pred, reference):
    """Rewrite `pred`'s intensity distribution to match `reference`, keeping its order.

    A weighted loss buys light-rain skill by over-painting rain: the trained model
    scores +31% against HRRR at 1 mm/h but runs a frequency bias of 1.34, and at
    8 mm/h that costs more than it gains. Probability matching separates the two
    questions. The model decides *where* it rains, which is what it is good at; the
    intensity distribution is then replaced wholesale by the reference climatology, so
    the bias is 1.0 by construction at every threshold.

    This is the classical PM used with ensemble QPF, and the approach the 2025 GRL
    probability-matching-loss paper builds on. Applied post hoc it needs no retraining.

    `reference` supplies the target distribution - the observed field when calibrating,
    or a stored quantile table at inference.
    """
    flat = pred.reshape(-1)
    order = torch.argsort(torch.argsort(flat))  # rank of each cell within pred
    ref = torch.sort(reference.reshape(-1)).values
    if ref.numel() != flat.numel():  # resample the reference onto pred's length
        idx = torch.linspace(0, ref.numel() - 1, flat.numel(), device=ref.device)
        ref = ref[idx.long().clamp_(max=ref.numel() - 1)]
    return ref[order].reshape(pred.shape)


def quantile_table(field, n=1024):
    """Quantiles of `field`, as a compact reference distribution for probability_match."""
    flat = torch.sort(field.reshape(-1)).values
    idx = torch.linspace(0, flat.numel() - 1, n, device=flat.device).round().long()
    return flat[idx]
