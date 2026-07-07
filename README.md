# Global Nowcast

A global precipitation nowcast built from **NOAA GFS composite reflectivity (REFC)**,
rendered as an animated radar composite on an interactive world map. Runs entirely on
free GitHub Actions and publishes to GitHub Pages.

**Live:** https://andrewnakas.github.io/globalnowcast/

## Products

| Product      | Horizon | Refresh                          |
|--------------|---------|----------------------------------|
| **Rapid**    | 18 h    | hourly (frames re-anchored to now) |
| **Extended** | 48 h    | every 6 h (new GFS cycle)        |

Each hour, the pipeline finds the freshest complete GFS cycle on the
[AWS Open Data mirror](https://registry.opendata.aws/noaa-gfs-bdp-pds/), downloads
**only the REFC GRIB message** for each forecast hour via HTTP byte-range requests
(~1 MB per frame, no credentials), and renders transparent PNGs using the classic
NWS reflectivity palette. The rapid product shows lead hours starting from "now";
the extended product covers f000–f048 of that cycle.

## How it works

```
pipeline/gfs.py      cycle discovery + .idx byte-range fetch of REFC
pipeline/render.py   GRIB → dBZ array → colormapped RGBA PNG
pipeline/main.py     orchestration, manifest.json
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

## Future work

- Blend observed satellite precipitation (GPM IMERG) for the first hours.
- ML post-processing (fine-tuned UNet on GFS→observed residuals) trained off-platform.
