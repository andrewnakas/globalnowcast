# Global Nowcast

A global precipitation nowcast: observed satellite rain rates extrapolated forward and
blended into **NOAA GFS composite reflectivity (REFC)**, rendered as an animated radar
composite on an interactive world map. Runs entirely on free GitHub Actions and
publishes to GitHub Pages.

**Live:** https://andrewnakas.github.io/globalnowcast/

## Products

| Product      | Horizon | Source                                    | Refresh |
|--------------|---------|-------------------------------------------|---------|
| **Nowcast**  | 4 h     | observations, advected and blended into GFS | hourly |
| **Rapid**    | 18 h    | GFS, re-anchored to now                   | hourly  |
| **Extended** | 48 h    | GFS, f000–f048 of the cycle               | every 6 h |

Each hour, the pipeline finds the freshest complete GFS cycle on the
[AWS Open Data mirror](https://registry.opendata.aws/noaa-gfs-bdp-pds/), downloads
**only the REFC GRIB message** for each forecast hour via HTTP byte-range requests
(~1 MB per frame, no credentials), and renders transparent PNGs using the classic
NWS reflectivity palette.

It then fetches the two most recent frames of NOAA's blended geostationary rain-rate
product (GOES-19, GOES-18, Himawari-9, MSG2, MSG3 — global except the poles, about 20
minutes behind real time), estimates motion between them with optical flow, and
transports the observed field forward. Near term that beats the model by a wide margin,
so the nowcast is observation-dominated and hands over to GFS around +4.5 h. See
[`verify/README.md`](verify/README.md) for the measured skill and how to reproduce it.

## How it works

```
pipeline/gfs.py      cycle discovery + .idx byte-range fetch of REFC
pipeline/obs.py      satellite rain rate → area-mean regrid onto the GFS grid
pipeline/nowcast.py  optical flow, advection, and the observation/GFS blend
pipeline/render.py   GRIB → dBZ array → colormapped RGBA PNG
pipeline/main.py     orchestration, manifest.json
verify/              skill scoring against later observations (CSI/POD/FAR/FSS)
site/                Leaflet viewer (dark basemap + PNG imageOverlay animation)
.github/workflows/   hourly build + GitHub Pages deploy
```

## Run locally

```bash
pip install -r requirements.txt   # pygrib wheels bundle eccodes; no system libs needed
python pipeline/main.py           # writes site/data/frames/*.png + manifest.json
python -m http.server -d site 8000
open http://localhost:8000
```

## Notes

- GitHub disables scheduled workflows after 60 days of repo inactivity; push a
  commit or re-run manually to keep it alive.
- REFC is a diagnostic reflectivity field from a global NWP model, not observed
  radar. It's a physically-based precipitation forecast, best read at synoptic scale.
- The nowcast is satellite-derived rain rate converted to dBZ, not radar either. It
  covers 70N–60S — about 90% of the globe by area — and the polar caps fall back to
  GFS alone.
- Observations lag real time by ~20 minutes, so the first nowcast frame is already
  that old when it publishes; frames are labelled by their true valid time.

## Future work

- Re-fit the observation/GFS handover across more dates and seasons; the current
  crossover comes from a small sample (see `verify/README.md`).
- ML post-processing (fine-tuned UNet on GFS→observed residuals) trained off-platform.
  The target is now quantified: GFS rains over ~1.6x the area actually observed.
