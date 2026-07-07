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
      '&copy; <a href="https://carto.com/">CARTO</a> · Data: NOAA GFS · ' +
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
  state.product = product;
  state.frames = state.manifest.products[product];
  state.images = preload(state.frames);
  state.index = 0;
  el("scrub").max = String(Math.max(0, state.frames.length - 1));
  document.querySelectorAll(".products button").forEach((b) =>
    b.classList.toggle("active", b.dataset.product === product)
  );
  show(0);
}

function show(i) {
  if (!state.frames.length) return;
  state.index = (i + state.frames.length) % state.frames.length;
  const frame = state.frames[state.index];
  overlay.setUrl(`data/frames/${frame.file}`);
  el("scrub").value = String(state.index);

  const d = parseUTC(frame.valid);
  const leadH = Math.round((d - parseUTC(state.manifest.cycle)) / 3.6e6);
  el("valid-time").textContent = `${fmt.format(d)}  ·  +${leadH}h`;
  el("cycle-info").textContent =
    `GFS ${state.manifest.cycle} · built ${state.manifest.generated_at.slice(11, 16)}Z`;
}

function play() {
  state.playing = !state.playing;
  el("play").textContent = state.playing ? "⏸" : "▶";
  if (state.playing) {
    state.timer = setInterval(() => show(state.index + 1), 450);
  } else {
    clearInterval(state.timer);
  }
}

function wire() {
  el("play").onclick = play;
  el("step-back").onclick = () => show(state.index - 1);
  el("step-fwd").onclick = () => show(state.index + 1);
  el("scrub").oninput = (e) => show(Number(e.target.value));
  el("opacity").oninput = (e) => {
    state.opacity = Number(e.target.value) / 100;
    overlay.setOpacity(state.opacity);
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
    loadProduct("rapid");
    el("status").classList.add("hidden");
  } catch (e) {
    el("status").textContent =
      "No forecast data yet — the first GitHub Actions run must finish. Check back shortly.";
  }
}

init();
