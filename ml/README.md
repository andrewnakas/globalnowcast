# ML REFC correction

A small residual UNet that nudges GFS composite reflectivity toward observed
precipitation. It's an **optional post-processing step** in the main pipeline:
until a trained model is committed at `ml/model/refc_correction.onnx`, the pipeline
runs exactly as before (identity passthrough). Once weights exist, every rendered
frame is corrected on CPU inside the normal hourly GitHub Actions job — no GPU, no
extra workflow.

```
GFS REFC (dBZ) ──▶ RefcUNet residual ──▶ GFS + Δ ──▶ render PNG
                     (learned toward IMERG / MRMS observations)
```

## Why this and not a from-scratch forecast model

GFS has already done the expensive physics to reach 48 h. Training a standalone
global forecast model would mean out-forecasting GFS — GPU-weeks and tens of TB.
Correcting GFS toward observations is 100–1000× smaller: a few GB of paired data
and hours on one free GPU, because the model only learns GFS's *systematic bias*
against what's actually observed.

## Layout

| File | Role |
|------|------|
| `data/common.py` | Marshall–Palmer Z–R (`Z=200·R^1.6`, `dBZ=10·log10 Z`) + regrid to the GFS 0.25° grid |
| `data/imerg.py` | GPM IMERG global precip (GES DISC, needs Earthdata token) → dBZ |
| `data/mrms.py` | MRMS CONUS composite reflectivity (anonymous S3) → dBZ |
| `data/gfs_label.py` | GFS REFC analysis (f000), reusing the production `pipeline/gfs.py` |
| `data/build_dataset.py` | pairs GFS + obs at matching valid times into `.npz` shards |
| `model.py` | the residual UNet (~0.5 M params, fully convolutional) |
| `train.py` | patch training + masked loss + ONNX export |
| `../pipeline/correct.py` | CPU inference hook used by the live pipeline |

## Train it (free GPU)

Best target: **Kaggle** (30 GPU-hr/week guaranteed, P100/T4). Colab works too.
Open `notebooks/train_kaggle.ipynb`, or run the steps manually:

1. **Get a NASA Earthdata token** (only needed for the global IMERG target):
   create a free account at <https://urs.earthdata.nasa.gov>, approve the
   *"NASA GESDISC DATA ARCHIVE"* application in your profile, then generate a
   User Token. Set it as `EARTHDATA_TOKEN`. (MRMS needs no credentials.)

2. **Build a dataset** (a year of 6-hourly global pairs ≈ a few GB):
   ```bash
   pip install -r requirements-ml.txt
   # global (IMERG):
   EARTHDATA_TOKEN=... python ml/data/build_dataset.py \
       --source imerg --start 2023-06-01 --days 365 --every 6 --out ml/shards
   # optional CONUS refinement (MRMS, no token):
   python ml/data/build_dataset.py \
       --source mrms --start 2023-06-01 --days 180 --every 6 --out ml/shards
   ```
   MRMS data starts 2020-10-14; pick start dates after that.

3. **Train + export**:
   ```bash
   python ml/train.py --shards ml/shards --epochs 20 \
       --out ml/model/refc_correction.onnx
   ```

4. **Ship it**: trained weights are gitignored by default, so force-add the model
   you trained and commit it:
   ```bash
   git add -f ml/model/refc_correction.onnx
   git commit -m "Add trained REFC correction model"
   git push
   ```
   The next pipeline run picks it up automatically and sets `"corrected": true`
   in `manifest.json` (the viewer then shows an "ML-corrected" badge).

## Toggle

`NOWCAST_CORRECT=0` in the workflow env forces the correction off even if weights
are present (handy for A/B checks).
