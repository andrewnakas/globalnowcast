"use strict";

const BOUNDS = [[-90, -180], [90, 180]];
const NWS_COLORS = [
  "#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
  "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
  "#f800fd", "#9854c6",
];

const map = L.map("map", {
  center: [20, 0],
  zoom: 3,
  minZoom: 2,
  maxZoom: 7,
  worldCopyJump: true,
  zoomControl: false,
});
L.control.zoom({ position: "topleft" }).addTo(map);
L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  {
    attribution:
      '&copy; <a href="https://carto.com/">CARTO</a> · Data: NOAA GFS + ' +
      'GOES/Himawari/Meteosat rain rate · ' +
      '<a href="https://github.com/andrewnakas/globalnowcast">source</a>',
    subdomains: "abcd",
    maxZoom: 8,
  }
).addTo(map);

const state = {
  manifest: null,
  product: "rapid",
  frames: [],
  images: [],
  index: 0,
  playing: false,
  timer: null,
  opacity: 0.8,
};

const overlay = L.imageOverlay("", BOUNDS, { opacity: state.opacity, interactive: false }).addTo(map);
// CONUS radar-model layer, drawn above the global field where radar exists. Its
// PNGs carry their own feathered alpha, so it just stacks: no client-side blending.
const conusOverlay = L.imageOverlay("", BOUNDS, { opacity: state.opacity, interactive: false });

const el = (id) => document.getElementById(id);
// Manifest uses "YYYY-MM-DDTHH:00Z"; normalize to a form every browser parses.
const parseUTC = (s) => new Date(s.replace(/T(\d\d):(\d\d)Z$/, "T$1:$2:00Z"));
const fmt = new Intl.DateTimeFormat(undefined, {
  weekday: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short",
});

function buildLegend() {
  el("legend-bar").style.background =
    `linear-gradient(90deg, ${NWS_COLORS.join(",")})`;
}

function preload(frames) {
  return frames.map((f) => {
    const img = new Image();
    img.src = `data/frames/${f.file}`;
    return img;
  });
}

function loadProduct(product) {
  const frames = state.manifest.products[product];
  if (!frames || !frames.length) return; // e.g. nowcast absent after an obs outage
  state.product = product;
  state.frames = frames;
  state.images = preload(state.frames);
  state.index = 0;
  el("scrub").max = String(Math.max(0, state.frames.length - 1));
  if (state.playing) startTimer(); // this product may animate at a different rate
  document.querySelectorAll(".products button").forEach((b) =>
    b.classList.toggle("active", b.dataset.product === product)
  );
  show(0);
}

function loadTimeline() {
  // One seamless 0-48h sequence: 15-minute satellite nowcast, then hourly GFS,
  // with the CONUS radar-model layer stacked wherever the manifest carries it.
  state.product = "timeline";
  state.frames = state.manifest.timeline;
  state.images = preload(state.frames);
  state.frames.forEach((f) => {
    if (f.conus) {
      const img = new Image();
      img.src = `data/frames/${f.conus}`;
    }
  });
  if (state.manifest.conus) {
    conusOverlay.setBounds(L.latLngBounds(state.manifest.conus.bounds));
    conusOverlay.addTo(map);
  }
  state.index = 0;
  el("scrub").max = String(Math.max(0, state.frames.length - 1));
  document.querySelector(".products").hidden = true;
  show(0);
}

function show(i) {
  if (!state.frames.length) return;
  state.index = (i + state.frames.length) % state.frames.length;
  const frame = state.frames[state.index];
  overlay.setUrl(`data/frames/${frame.file}`);
  if (state.product === "timeline" && state.manifest.conus) {
    if (frame.conus) {
      conusOverlay.setUrl(`data/frames/${frame.conus}`);
      conusOverlay.setOpacity(state.opacity);
    } else {
      conusOverlay.setOpacity(0);
    }
  }
  el("scrub").value = String(state.index);

  // Nowcast-anchored frames measure lead from the observation time, not the model
  // cycle — measuring against the cycle would overstate it by hours.
  const d = parseUTC(frame.valid);
  const timeline = state.product === "timeline";
  const nowcasting = (timeline || state.product === "nowcast") && state.manifest.obs_time;
  const ref = parseUTC(nowcasting ? state.manifest.obs_time : state.manifest.cycle);
  const mins = Math.round((d - ref) / 6e4);
  const lead = mins <= 0 ? "now"
    : mins < 120 ? `+${mins}m`
    : `+${Math.round(mins / 60)}h`;
  el("valid-time").textContent = `${fmt.format(d)}  ·  ${lead}`;

  const ml = state.manifest.corrected ? " · ML-corrected" : "";
  const label = { obs: "satellite obs", blend: "obs + GFS blend", gfs: "GFS" };
  const conus = timeline && frame.conus ? " · CONUS: radar+HRRR" : "";
  el("cycle-info").textContent = nowcasting
    ? `${label[frame.source] || "GFS"}${conus} · obs ${state.manifest.obs_time}${ml}`
    : `GFS ${state.manifest.cycle}${conus} · built ${state.manifest.generated_at.slice(11, 16)}Z${ml}`;
}

function frameDelay() {
  // 15-minute steps need a quicker cadence than hourly ones to read as motion;
  // on the unified timeline the spacing changes mid-sequence, so it is computed
  // from the actual gap to the next frame rather than fixed per product.
  if (state.product === "nowcast") return 280;
  if (state.product !== "timeline") return 450;
  const next = state.frames[(state.index + 1) % state.frames.length];
  const gap = parseUTC(next.valid) - parseUTC(state.frames[state.index].valid);
  return gap > 0 && gap <= 20 * 6e4 ? 280 : 450;
}

function startTimer() {
  clearInterval(state.timer);
  clearTimeout(state.timer);
  const tick = () => {
    show(state.index + 1);
    state.timer = setTimeout(tick, frameDelay());
  };
  state.timer = setTimeout(tick, frameDelay());
}

function play() {
  state.playing = !state.playing;
  el("play").textContent = state.playing ? "⏸" : "▶";
  if (state.playing) startTimer();
  else clearInterval(state.timer);
}

function wire() {
  el("play").onclick = play;
  el("step-back").onclick = () => show(state.index - 1);
  el("step-fwd").onclick = () => show(state.index + 1);
  el("scrub").oninput = (e) => show(Number(e.target.value));
  el("opacity").oninput = (e) => {
    state.opacity = Number(e.target.value) / 100;
    overlay.setOpacity(state.opacity);
    if (state.frames[state.index]?.conus) conusOverlay.setOpacity(state.opacity);
  };
  document.querySelectorAll(".products button").forEach((b) => {
    b.onclick = () => loadProduct(b.dataset.product);
  });
}

async function init() {
  buildLegend();
  wire();
  try {
    const res = await fetch(`data/manifest.json?t=${Date.now()}`);
    if (!res.ok) throw new Error(res.status);
    state.manifest = await res.json();
    if (state.manifest.timeline?.length) {
      loadTimeline();
    } else {
      // Older manifest: hide products this run didn't produce, then open the most
      // skilful one there is.
      let first = null;
      document.querySelectorAll(".products button").forEach((b) => {
        const has = Boolean(state.manifest.products[b.dataset.product]?.length);
        b.hidden = !has;
        if (has && !first) first = b.dataset.product;
      });
      loadProduct(first || "rapid");
    }
    el("status").classList.add("hidden");
  } catch (e) {
    el("status").textContent =
      "No forecast data yet — the first GitHub Actions run must finish. Check back shortly.";
  }
}

init();
