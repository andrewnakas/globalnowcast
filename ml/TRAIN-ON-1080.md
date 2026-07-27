# Training on a local GTX 1080

A 1080 is a better fit for this model than Kaggle's TPUs: no queue, no 20 GB dataset
limit, no session cap, and the model is small enough that the GPU is barely stressed.

| model | params | activations @ batch 64 | 50 epochs on ~20k samples |
|-------|--------|------------------------|---------------------------|
| base=32 | 0.93 M | ~1.2 GB | ~6 min |
| **base=64** | **3.71 M** | **~2.5 GB** | ~15-20 min |
| base=96 | 8.34 M | ~3.7 GB | ~40 min |

`base=64` is the recommendation: four times the capacity of the default, still under a
third of the 1080's 8 GB, and a full run is minutes. There is no reason to leave that
headroom unused — the earlier sizing was written for a shared TPU budget, not a
dedicated card.

## Setup

```bash
git clone https://github.com/andrewnakas/globalnowcast.git
cd globalnowcast
python3.12 -m venv .venv          # 3.12: pygrib has no 3.14 wheel
.venv/bin/pip install -r requirements-ml.txt
```

Torch must be the CUDA build, not the CPU wheel that comes in by default:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Get the data

Either copy `ml/tiles/` across (~40 MB a shard, uint8), or rebuild it on the 1080 box
straight from the dynamical.org catalog — it is anonymous, so nothing needs
credentials:

```bash
.venv/bin/python ml/data/build_tiles.py --tiles 220 --out ml/tiles
```

Expect roughly 2.5 sequences a second; the cost is network round-trips to S3, not CPU,
so it will not compete with anything else on the machine for compute.

## Train

```bash
.venv/bin/python ml/train_nowcast.py --tiles ml/tiles --base 64 --batch 64 --epochs 50
```

Watch the first line of output. Because the head is zero-initialised the model *starts*
as a pass-through of HRRR, so epoch 0 prints the HRRR baseline and every later epoch
has to beat it. If validation CSI sits at the starting value, the model is learning
nothing and something is wrong — that is the point of initialising it that way.

## What the numbers should look like

Measured on the first shard, radar truth, 1 mm/h:

| lead | HRRR | persistence |
|------|------|-------------|
| +1h | 0.424 | **0.450** |
| +2h | 0.413 | 0.309 |
| +3h | 0.409 | 0.241 |
| +6h | 0.422 | 0.139 |

The interesting structure is the crossover: recent radar beats the model at +1h, and
the model wins from +2h out, by a widening margin. A useful nowcast has to follow
persistence early and hand over to HRRR later — which is the same shape as the
satellite/GFS blend already shipping, and exactly what the model gets both inputs to
learn.

Beating HRRR's ~0.41 average is the bar. Persistence is the easier one and should fall
immediately.

## Then

```bash
.venv/bin/python verify/run_radar.py     # scores against the advection baseline
```

Copy `ml/model/nowcast.pt` back. Note the hourly site job runs CPU-only on GitHub
Actions with a budget near a second a frame, so anything larger than base=64 should be
checked for inference cost before it ships, and exported to ONNX the way `ml/train.py`
already does for the GFS correction model.
