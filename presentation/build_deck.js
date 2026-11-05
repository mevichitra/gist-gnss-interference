// Builds presentation/GIST_DL_Project.pptx.
// Result numbers are read from presentation/data/runs/ (see UPDATE_NUMBERS.md and sync_results.sh).
// Run: NODE_PATH=<dir with pptxgenjs, react, react-dom, react-icons, sharp> node presentation/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const HERE = __dirname;
const RUNS = path.join(HERE, "data", "runs");
const OUT = path.join(HERE, "GIST_DL_Project.pptx");

// ---------------------------------------------------------------- theme (same palette as the team's major-project deck)
const THEME = {
  name: "GIST",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A2230", lt1: "FFFFFF", dk2: "12304A", lt2: "EEF2F6",
    accent1: "0F766E", accent2: "B42318", accent3: "B45309",
    accent4: "1D4ED8", accent5: "6D28D9", accent6: "15803D",
    hlink: "1D4ED8", folHlink: "6D28D9",
  },
};
const HEX = { ...THEME.colors, muted: "5B6675", grid: "D5DCE4", tealTint: "E3F0EE", redTint: "F6E4E2", amberTint: "F5EAE0", blueTint: "E4EAFB", purpleTint: "ECE6F8", greenTint: "E3F1E7" };
const ACC = ["0F766E", "1D4ED8", "B45309", "B42318", "15803D", "6D28D9"];
const TINT = { "0F766E": HEX.tealTint, "1D4ED8": HEX.blueTint, "B45309": HEX.amberTint, "B42318": HEX.redTint, "15803D": HEX.greenTint, "6D28D9": HEX.purpleTint };

// ---------------------------------------------------------------- results
const readJSON = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const std = (a) => (a.length < 2 ? 0 : Math.sqrt(a.reduce((s, x) => s + (x - mean(a)) ** 2, 0) / (a.length - 1)));
const get = (o, k) => k.split(".").reduce((v, p) => (v == null ? undefined : v[p]), o);

function loadResults() {
  const base = readJSON(path.join(RUNS, "baselines_test.json"));
  const baseD = readJSON(path.join(RUNS, "baselines_test_distract.json"));
  const finals = [0, 1, 2].map((s) => readJSON(path.join(RUNS, `gist_final_s${s}`, "eval_test.json"))).filter(Boolean);
  const finalsD = [0, 1, 2].map((s) => readJSON(path.join(RUNS, `gist_final_s${s}`, "eval_test_distract.json"))).filter(Boolean);
  const s0Test = readJSON(path.join(RUNS, "gist_s0", "eval_test.json"));
  const s0TestD = readJSON(path.join(RUNS, "gist_s0", "eval_test_distract.json"));
  const abl = {
    full: finals[0] || s0Test || null,
    nograph: readJSON(path.join(RUNS, "gist_nograph_s0", "eval_test.json")),
    integ: readJSON(path.join(RUNS, "gist_integ_s0", "eval_test.json")),
    nosim: readJSON(path.join(RUNS, "gist_nosim_s0", "eval_test.json")),
  };
  let gist, gistD, status;
  if (finals.length) {
    const agg = (runs) => (k) => { const v = runs.map((r) => get(r, k)).filter((x) => x != null); return v.length ? { m: mean(v), s: std(v), n: v.length } : null; };
    gist = agg(finals); gistD = finalsD.length ? agg(finalsD) : () => null;
    status = { final: true, tag: `FINAL · ${finals.length} SEED${finals.length > 1 ? "S" : ""} · TEST`, split: "test (2026 days)" };
  } else {
    // Interim: seed-0 model scored on the same 2026 test days as the baselines (like-for-like).
    const one = (v) => (k) => { const x = get(v, k); return x == null ? null : { m: x, s: 0, n: 1 }; };
    const v = s0Test || readJSON(path.join(RUNS, "gist_s0", "best_val.json"));
    gist = one(v);
    gistD = s0TestD ? one(s0TestD) : () => null;
    status = s0Test
      ? { final: false, tag: "INTERIM · SEED 0 · TEST", split: "test (2026 days)" }
      : { final: false, tag: "INTERIM · SEED 0 · VALIDATION", split: "validation (2025-07..12 days)" };
  }
  return { base, baseD, gist, gistD, abl, status };
}
const R = loadResults();
// Pixel size of assets/case_BAL_2026-01-01_interim.png (cropped from runs/gist_s0/fig/), for its aspect ratio.
const CASE_W = 1436, CASE_H = 436;
const f2 = (x) => (x == null ? "–" : x.toFixed(2));
const f1 = (x) => (x == null ? "–" : x.toFixed(1));
const g2 = (k) => { const r = R.gist(k); return r == null ? "pending" : r.n > 1 ? `${r.m.toFixed(2)} ± ${r.s.toFixed(2)}` : r.m.toFixed(2); };
const g1 = (k, src = R.gist) => { const r = src(k); return r == null ? "pending" : r.n > 1 ? `${r.m.toFixed(1)} ± ${r.s.toFixed(1)}` : r.m.toFixed(1); };
const gv = (k, src = R.gist) => { const r = src(k); return r == null ? null : r.m; };

// ---------------------------------------------------------------- icons
async function icon(Comp, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------------------------------------------------------------- deck
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in, same as the reference deck
pres.author = "Vansh Agarwal, Vichitra Attri, Vidya Bai K";
pres.company = "Bangalore Institute of Technology";
pres.title = "GIST: GNSS Interference Spatio-Temporal Network";
pres.subject = "Deep Learning course project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.5;
const FOOT = "GIST  ·  Deep Learning Project  ·  Department of ISE, Bangalore Institute of Technology";

pres.defineSlideMaster({
  title: "CONTENT_PH",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.32, w: 10, h: 0.3, fontSize: 10, bold: true, color: C.accent1, charSpacing: 3, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.62, w: 12.3, h: 0.7, fontSize: 32, bold: true, color: C.text2, fontFace: THEME.headFontFace, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.02, w: 9, h: 0.3, fontSize: 9, color: HEX.muted, margin: 0 } } },
  ],
  slideNumber: { x: 12.33, y: 7.02, w: 0.5, h: 0.3, fontSize: 9, color: HEX.muted, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: HEX.dk2 }, objects: [] });
pres.defineSlideMaster({ title: "TITLE", background: { color: "FFFFFF" }, objects: [] });

let sectionNo = 0;
function content(section, title, { kicker } = {}) {
  const s = pres.addSlide({ masterName: "CONTENT_PH", sectionTitle: section });
  s.addText(kicker || section.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function section(title) { pres.addSection({ title }); }

// small building blocks ---------------------------------------------------------
function card(s, x, y, w, h, fill = HEX.lt2, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.06, objectName: name });
}
function txt(s, text, o) { s.addText(text, { isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: C.text1, ...o }); }
function circleIcon(s, img, x, y, d, color) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color } });
  const p = d * 0.25;
  s.addImage({ data: img, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}
function numBadge(s, n, x, y, d, color) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color } });
  txt(s, String(n), { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, color: "FFFFFF", fontSize: d > 0.5 ? 16 : 12 });
}
function pill(s, text, x, y, w, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, fill: { color }, line: { color }, rectRadius: 0.16 });
  txt(s, text, { x, y, w, h: 0.32, align: "center", valign: "middle", fontSize: 10, bold: true, color: "FFFFFF", charSpacing: 1 });
}
function statusPill(s) { pill(s, R.status.tag, W - MX - 3.6, 0.3, 3.6, R.status.final ? HEX.accent6 : HEX.accent3); }
function box(s, x, y, w, h, title, sub, color, opts = {}) {
  const fill = opts.fill || TINT[color] || HEX.lt2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color, width: 1.25 }, rectRadius: 0.08 });
  txt(s, [
    { text: title, options: { bold: true, color, fontSize: opts.tsize || 14, breakLine: !!sub } },
    ...(sub ? [{ text: sub, options: { fontSize: opts.ssize || 11, color: C.text1 } }] : []),
  ], { x: x + 0.1, y: y + 0.06, w: w - 0.2, h: h - 0.12, align: "center", valign: "middle", paraSpaceAfter: 2 });
}
function arrow(s, x1, y1, x2, y2, color = HEX.muted) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.0001, h: Math.abs(y2 - y1) || 0.0001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 1.5, endArrowType: "triangle" } });
}
function bullets(items, o = {}) {
  return items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap ?? 6 } }));
}
function table(s, rows, o) {
  const head = rows[0].map((c) => ({ text: c, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 } } }));
  const body = rows.slice(1).map((r, i) => r.map((c) => (typeof c === "object" ? c : { text: String(c), options: { fill: { color: i % 2 ? "FFFFFF" : "F6F8FA" } } })));
  s.addTable([head, ...body], { fontSize: 12, color: HEX.dk1, border: { type: "solid", pt: 0.5, color: HEX.grid }, margin: [3, 6, 3, 6], valign: "middle", fontFace: "Calibri", ...o });
}
const chartBase = (title) => ({
  showTitle: !!title, title, titleFontSize: 12, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelFontSize: 10, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
});

async function build() {
  const I = {};
  for (const [k, comp] of Object.entries({
    plane: fa.FaPlane, sat: fa.FaSatellite, tower: fa.FaBroadcastTower, map: fa.FaMapMarkedAlt, tag: fa.FaTags, brain: fa.FaBrain,
    search: fa.FaSearch, cross: fa.FaCrosshairs, warn: fa.FaExclamationTriangle, eye: fa.FaEyeSlash, clock: fa.FaClock, ban: fa.FaBan,
    db: fa.FaDatabase, balance: fa.FaBalanceScale, flask: fa.FaFlask, chart: fa.FaChartLine, shield: fa.FaShieldAlt, memory: fa.FaMemory,
    bug: fa.FaBug, cloud: fa.FaCloudDownloadAlt, wind: fa.FaWind, check: fa.FaCheckCircle, layer: fa.FaLayerGroup, globe: fa.FaGlobeEurope,
    users: fa.FaUserFriends, signal: fa.FaSignal,
  })) I[k] = await icon(comp);

  // ============================================================ 1. Title (layout of the team's major-project title page)
  section("Title");
  {
    const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Title" });
    s.addImage({ path: path.join(HERE, "assets", "vtu_banner.png"), x: 0, y: 0, w: W, h: W * 572 / 4000, altText: "Visvesvaraya Technological University, Belagavi" });
    s.addImage({ path: path.join(HERE, "assets", "bit_logo.png"), x: 0.42, y: 2.12, w: 1.42, h: 1.39, altText: "Bangalore Institute of Technology logo" });
    const G = "1E5631", T = "Times New Roman";
    txt(s, "BANGALORE INSTITUTE OF TECHNOLOGY", { x: 1.9, y: 2.1, w: 10.5, h: 0.42, align: "center", fontFace: T, bold: true, fontSize: 22, color: G });
    txt(s, "DEPARTMENT OF INFORMATION SCIENCE AND ENGINEERING", { x: 1.9, y: 2.58, w: 10.5, h: 0.42, align: "center", fontFace: T, bold: true, fontSize: 20, color: G });
    txt(s, "K.R.ROAD, V.V.PURAM, BANGALORE - 560004", { x: 1.9, y: 3.05, w: 10.5, h: 0.32, align: "center", fontFace: T, fontSize: 14, color: G });
    txt(s, "Deep Learning Project", { x: 1.9, y: 3.42, w: 9.5, h: 0.34, align: "center", fontFace: T, bold: true, fontSize: 17, color: "000000" });
    txt(s, "GIST: Weakly-Supervised Detection, Classification and Localization of GNSS Jamming and Spoofing from Crowdsourced ADS-B",
      { x: 1.6, y: 3.82, w: 10.1, h: 0.95, align: "center", valign: "middle", fontFace: T, bold: true, fontSize: 21, color: "000000" });
    txt(s, "Presented by:", { x: 0.95, y: 4.95, w: 3, h: 0.36, fontFace: T, bold: true, fontSize: 17, color: G });
    const team = [["Vansh Agarwal", "1BI23IS133"], ["Vichitra Attri", "1BI23IS135"], ["Vidya Bai K", "1BI23IS137"]];
    txt(s, [{ text: "NAME", options: { bold: true, color: G, breakLine: true } }, ...team.map((t, i) => ({ text: t[0], options: { breakLine: i < 2 } }))],
      { x: 0.95, y: 5.38, w: 2.6, h: 1.5, fontFace: T, fontSize: 16, color: "000000", paraSpaceAfter: 2 });
    txt(s, [{ text: "USN", options: { bold: true, color: G, breakLine: true } }, ...team.map((t, i) => ({ text: t[1], options: { breakLine: i < 2 } }))],
      { x: 3.4, y: 5.38, w: 2.2, h: 1.5, fontFace: T, fontSize: 16, color: "000000", paraSpaceAfter: 2 });
    txt(s, "Under the guidance of:", { x: 8.9, y: 5.0, w: 4, h: 0.4, fontFace: T, bold: true, fontSize: 19, color: G });
    txt(s, [{ text: "Dr. Asha T.", options: { breakLine: true } }, { text: "Deep Learning course" }],
      { x: 8.9, y: 5.5, w: 4, h: 0.9, fontFace: T, fontSize: 17, color: "000000", paraSpaceAfter: 4 });
    s.addNotes("Title page in the same layout as our major-project deck. GIST stands for GNSS Interference Spatio-Temporal network. The same team (Vansh, Vichitra, Vidya) built it for the Deep Learning course under Dr. Asha T.");
  }

  // ============================================================ 2. Outline
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Title" });
    txt(s, "GIST", { x: 0.7, y: 2.2, w: 5, h: 1.0, fontFace: THEME.headFontFace, bold: true, fontSize: 54, color: "FFFFFF" });
    txt(s, "Every aircraft as a remote GNSS sensor: detecting, classifying and localizing GPS interference from crowdsourced ADS-B, with no manual labels",
      { x: 0.7, y: 3.35, w: 5.2, h: 1.2, fontFace: THEME.headFontFace, italic: true, fontSize: 16, color: "CADCFC" });
    txt(s, "Transformer  ·  Graph attention  ·  Weak supervision  ·  Differentiable physics", { x: 0.7, y: 4.75, w: 5.6, h: 0.3, fontSize: 12, color: "CADCFC" });
    s.addShape(pres.shapes.LINE, { x: 6.55, y: 1.0, w: 0, h: 5.6, line: { color: "2E5373", width: 1 } });
    txt(s, "OUTLINE", { x: 7.0, y: 0.85, w: 4, h: 0.3, fontSize: 11, bold: true, color: "CADCFC", charSpacing: 3 });
    const items = ["Abstract and motivation", "Literature survey and research gap", "Problem statement and objectives", "Dataset and licensing",
      "System architecture", "Physics features and weak supervision", "GIST model and localization head", "Simulator and experimental setup",
      "Results, ablations and case study", "Challenges, limitations, conclusion and future work"];
    items.forEach((t, i) => {
      txt(s, String(i + 1).padStart(2, "0"), { x: 7.0, y: 1.3 + i * 0.53, w: 0.5, h: 0.4, fontSize: 14, color: "5EC4B6", valign: "middle" });
      txt(s, t, { x: 7.6, y: 1.3 + i * 0.53, w: 5.3, h: 0.4, fontSize: 14, color: "FFFFFF", valign: "middle" });
    });
    s.addNotes("Roadmap of the talk.");
  }

  // ============================================================ 3. Abstract
  section("Introduction");
  {
    const s = content("Introduction", "Abstract", { kicker: "01 · ABSTRACT" });
    txt(s, [
      { text: "GIST", options: { bold: true } },
      { text: " is a deep-learning system that detects GNSS (GPS) interference affecting civil aircraft, tells jamming apart from spoofing, and estimates where a jammer is. It uses only what aircraft already broadcast over ADS-B, collected by volunteer receivers and published as open data (adsb.lol, ODbL).", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "Public maps threshold one integrity field (NIC) per hexagon per day. That catches jamming but misses spoofing, because a spoofed receiver reports high integrity. GIST instead computes physics-consistency features a spoofer cannot fake cheaply (barometric vs GNSS altitude against neighbours, wind consensus, position jumps, co-located aircraft at different altitudes), encodes each aircraft's 10-minute track with a Transformer, shares evidence between nearby aircraft with spatial graph attention, and localizes jammers with a differentiable Friis / radio-horizon likelihood.", options: { breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "No labels are drawn by hand: ten physics-based labeling functions are combined by a Dawid–Skene label model, and synthetic jammers and spoofers injected into real traffic provide exact targets." },
    ], { x: MX, y: 1.6, w: 7.9, h: 5.2, fontSize: 15 });
    const cards = [[I.search, "Detects", "Interference per aircraft and 10-minute window, minutes after it starts.", ACC[0]],
      [I.tag, "Classifies", "Normal, jammed or spoofed, including spoofing that NIC thresholds cannot see.", ACC[1]],
      [I.cross, "Localizes", "Jammer position from which aircraft are affected, through a differentiable physics layer.", ACC[2]]];
    cards.forEach(([ic, h, b, col], i) => {
      const y = 1.6 + i * 1.75;
      card(s, 8.85, y, 3.98, 1.55);
      circleIcon(s, ic, 9.05, y + 0.25, 0.62, col);
      txt(s, h, { x: 9.85, y: y + 0.2, w: 2.85, h: 0.4, bold: true, fontSize: 16, color: col, fontFace: THEME.headFontFace });
      txt(s, b, { x: 9.85, y: y + 0.62, w: 2.85, h: 0.85, fontSize: 12 });
    });
    s.addNotes("One-paragraph summary. Key idea: aircraft are sensors; physics redundancy (barometric altitude, air data, many aircraft) exposes spoofing that integrity fields hide.");
  }

  // ============================================================ 4. Motivation: jamming vs spoofing
  {
    const s = content("Introduction", "Jamming vs Spoofing: Why NIC Maps Miss Half the Problem", { kicker: "02 · INTRODUCTION" });
    const col = (x, color, ic, head, sub, rows) => {
      card(s, x, 1.6, 5.95, 3.55, TINT[color]);
      circleIcon(s, ic, x + 0.25, 1.8, 0.62, color);
      txt(s, head, { x: x + 1.05, y: 1.78, w: 4.7, h: 0.38, bold: true, fontSize: 18, color, fontFace: THEME.headFontFace });
      txt(s, sub, { x: x + 1.05, y: 2.16, w: 4.7, h: 0.32, italic: true, fontSize: 12, color: HEX.muted });
      txt(s, bullets(rows), { x: x + 0.3, y: 2.65, w: 5.4, h: 2.4, fontSize: 14 });
    };
    col(MX, ACC[2], I.signal, "Jamming", "Signal denied by RF noise", ["Receiver loses lock; broadcast NIC collapses toward 0", "Position reports drop out while altitude and track continue",
      "Visible in integrity fields, so NIC-threshold maps (gpsjam.org) catch it", "Location of the emitter is still unknown"]);
    col(6.88, ACC[3], I.eye, "Spoofing", "Counterfeit signal, confident lie", ["Receiver locks onto fake satellites and reports high integrity (NIC ≥ 8)",
      "Positions jump, circle or drift; GNSS altitude disagrees with barometric altitude", "Integrity-threshold maps are structurally blind to it", "More dangerous: the crew may trust the wrong position"]);
    card(s, MX, 5.4, 12.33, 1.4, HEX.lt2);
    txt(s, [
      { text: "Why now. ", options: { bold: true, color: C.accent1 } },
      { text: "Interference affecting aviation has grown sharply since 2022 around the Baltic, Eastern Mediterranean, Black Sea, Middle East and India–Pakistan border. In our first day of data, the share of points with NIC ≤ 6 was 15.8 % over the Baltic versus 1.7 % over SE Australia. Barometric altitude and air data do not depend on GNSS, and a spoofer cannot make hundreds of aircraft mutually consistent: that redundancy is what GIST learns from." },
    ], { x: MX + 0.3, y: 5.5, w: 11.7, h: 1.2, fontSize: 14, valign: "middle" });
    s.addNotes("NIC = Navigation Integrity Category broadcast in ADS-B. Numbers from LOG.md Phase 2 (2026-10-25 first signal check).");
  }

  // ============================================================ 5. Literature survey
  section("Literature survey");
  {
    const s = content("Literature survey", "Literature Survey", { kicker: "03 · LITERATURE SURVEY" });
    table(s, [
      ["#", "Work", "Approach", "Gap for our problem"],
      ["[1]", "Liu, Lo, Walter — ION ITM 2022", "ADS-B jamming simulator; Friis model; tolerable power ≈ −115 dBW", "Jamming only; needs clean affected / unaffected labels"],
      ["[2]", "Liu et al. — NAVIGATION 72(3), 2025", "Non-linear least squares on affected aircraft; ~4 km on Denver 2022", "No spoofing; sensitive to benign low-NIC aircraft (we measure this)"],
      ["[3]", "Schäfer et al. — Jammer on the Horizon", "Radio-horizon intersection from ADS-B coverage", "Localization only; assumes jamming is already detected"],
      ["[4]", "Wiseman — gpsjam.org", "Daily H3 hexagon map, share of aircraft with low NIC", "Day granularity; cannot see spoofing; no emitter location"],
      ["[5]", "Eichstaedt et al. (DLR) — IEEE SHaRC 2026", "Global interference measurement from space-based ADS-B", "Measurement, not learned classification or localization"],
      ["[6]", "Aireon — Spoofing white paper, 2026", "NIC and position cross-checks on space-based ADS-B", "Industry rules; no learning across aircraft"],
      ["[7,8]", "Ratner et al. 2017; Dawid & Skene 1979", "Data programming; EM label model over noisy labelers", "Method we adopt to build labels without annotation"],
    ], { x: MX, y: 1.55, w: 12.33, colW: [0.6, 3.5, 4.2, 4.03], fontSize: 12, rowH: 0.56 });
    txt(s, "Re-checked against 2025–2026 literature in Phase 1 of the project (LOG.md). Full references on the References slide.", { x: MX, y: 6.55, w: 12, h: 0.3, fontSize: 10, italic: true, color: HEX.muted });
    s.addNotes("Liu et al. and Schäfer et al. are our localization baselines (NLS and radio horizon, re-implemented). gpsjam is our detection baseline (re-implemented).");
  }

  // ============================================================ 6. Existing systems / gap
  {
    const s = content("Literature survey", "Disadvantages of Current Systems", { kicker: "04 · EXISTING SYSTEMS" });
    txt(s, "No prior work jointly separates jamming from spoofing with physics invariants, learns across aircraft, localizes with a learned + physics hybrid, and needs no manual labels.",
      { x: MX, y: 1.5, w: 12.3, h: 0.5, italic: true, fontSize: 14, color: HEX.muted });
    const items = [[I.eye, "Blind to spoofing", "Integrity-threshold maps flag low NIC; spoofed aircraft report high NIC.", ACC[3]],
      [I.clock, "Day-level latency", "Public maps aggregate per day; an event is visible only afterwards.", ACC[2]],
      [I.map, "No emitter location", "Detection maps say where aircraft were affected, not where the source is.", ACC[1]],
      [I.tag, "Labels are scarce", "Ground-truth events with known emitter positions are rare and not public.", ACC[5]],
      [I.users, "Aircraft treated alone", "Per-aircraft rules ignore that neighbours see the same emitter.", ACC[0]],
      [I.warn, "Fragile localizers", "Least-squares needs clean affected sets; benign low-NIC aircraft corrupt it.", ACC[4]]];
    items.forEach(([ic, h, b, col], i) => {
      const x = MX + (i % 2) * 6.25, y = 2.15 + Math.floor(i / 2) * 1.55;
      card(s, x, y, 6.08, 1.35);
      circleIcon(s, ic, x + 0.25, y + 0.33, 0.68, col);
      txt(s, h, { x: x + 1.15, y: y + 0.2, w: 4.75, h: 0.38, bold: true, fontSize: 16, color: col, fontFace: THEME.headFontFace });
      txt(s, b, { x: x + 1.15, y: y + 0.62, w: 4.75, h: 0.62, fontSize: 13 });
    });
    s.addNotes("These six shortcomings map one-to-one onto our objectives on the next slides.");
  }

  // ============================================================ 7. Problem statement
  section("Problem and objectives");
  {
    const s = content("Problem and objectives", "Problem Statement", { kicker: "05 · PROBLEM STATEMENT" });
    card(s, MX, 1.55, 12.33, 1.35, HEX.amberTint);
    txt(s, "“", { x: MX + 0.25, y: 1.55, w: 0.6, h: 0.8, fontSize: 48, bold: true, color: C.accent3, fontFace: THEME.headFontFace });
    txt(s, "Given only crowdsourced ADS-B broadcasts and no manual labels, detect GNSS interference per aircraft within minutes, classify it as jamming or spoofing, and estimate the jammer's location, robustly to aircraft whose low integrity has nothing to do with the emitter.",
      { x: MX + 0.9, y: 1.68, w: 11.2, h: 1.1, italic: true, fontSize: 16, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, "Inputs and outputs", { x: MX, y: 3.15, w: 6, h: 0.35, bold: true, fontSize: 15, color: C.text2 });
    box(s, MX, 3.6, 3.6, 1.35, "Input", "Per-aircraft ADS-B traces: position, baro and GNSS altitude, speed, track, NIC / NACp / SIL, readsb wind", ACC[1], { ssize: 12 });
    arrow(s, MX + 3.65, 4.27, MX + 4.25, 4.27);
    box(s, MX + 4.3, 3.6, 3.6, 1.35, "Scene", "All aircraft in a ±3° latitude box over one 10-minute window (≤ 256 aircraft, 20 steps of 30 s)", ACC[0], { ssize: 12 });
    arrow(s, MX + 7.95, 4.27, MX + 8.55, 4.27);
    box(s, MX + 8.6, 3.6, 3.73, 1.35, "Output", "Class per aircraft-window and per step (normal / jammed / spoofed) + jammer location", ACC[2], { ssize: 12 });
    const ks = [["No ground truth", "labels must come from physics and simulation"], ["Rare spoofing", "few spoofed windows per day"], ["8 GB laptop", "Apple M3; every model sized to fit"]];
    ks.forEach(([h, b], i) => {
      const x = MX + i * 4.15;
      card(s, x, 5.3, 4.0, 1.4);
      numBadge(s, i + 1, x + 0.25, 5.5, 0.45, ACC[[3, 4, 5][i]]);
      txt(s, h, { x: x + 0.85, y: 5.48, w: 3.0, h: 0.35, bold: true, fontSize: 15 });
      txt(s, b, { x: x + 0.85, y: 5.85, w: 3.0, h: 0.7, fontSize: 13, color: HEX.muted });
    });
    s.addNotes("Constraints: no annotation budget, class imbalance, and a laptop with 8 GB of RAM.");
  }

  // ============================================================ 8. Objectives
  {
    const s = content("Problem and objectives", "Objectives", { kicker: "06 · OBJECTIVES" });
    const objs = ["Build an openly licensed, weakly-supervised dataset of jamming / spoofing labels from physics-based labeling functions and a label model.",
      "Design spoofing-sensitive features that do not rely on the integrity fields a spoofer controls.",
      "Train GIST: a trajectory Transformer with spatial graph attention for 3-way classification per aircraft and per step.",
      "Localize jammers with a differentiable Friis / radio-horizon head trained sim-to-real on synthetic emitters in real traffic.",
      "Compare fairly against re-implemented baselines (gpsjam threshold, Isolation Forest, LSTM-AE, NLS, radio horizon) on day-split data.",
      "Report honestly: 3 seeds, ablations, independent MLAT check, and the limits of weak labels and simulation."];
    objs.forEach((t, i) => {
      const x = MX + (i % 3) * 4.15, y = 1.6 + Math.floor(i / 3) * 2.62;
      card(s, x, y, 4.0, 2.42);
      txt(s, String(i + 1).padStart(2, "0"), { x: x + 0.3, y: y + 0.2, w: 1.2, h: 0.6, fontSize: 30, bold: true, color: ACC[i], fontFace: THEME.headFontFace });
      txt(s, t, { x: x + 0.3, y: y + 0.88, w: 3.45, h: 1.45, fontSize: 14 });
    });
    s.addNotes("Objectives follow the paper's contribution list (PLAN.md section 2).");
  }

  // ============================================================ 9. Dataset
  section("Dataset");
  {
    const s = content("Dataset", "Dataset: Open, Crowdsourced ADS-B (adsb.lol, ODbL)", { kicker: "07 · DATASET" });
    const stats = [["67 M", "airborne trace points", "in study regions (interim, 5 days)"], ["1.22 M", "aircraft-windows", "10 min, 20 × 30 s steps"], ["67,333", "aircraft in one day", "global, 2026-10-25"], ["114 s", "to parse one day", "6 workers, 340 MB RAM"]];
    stats.forEach(([n, l, sub], i) => {
      const x = MX + i * 3.1;
      card(s, x, 1.55, 2.95, 1.45);
      txt(s, n, { x: x + 0.2, y: 1.62, w: 2.6, h: 0.65, fontSize: 30, bold: true, color: ACC[i], fontFace: THEME.headFontFace });
      txt(s, l, { x: x + 0.2, y: 2.27, w: 2.6, h: 0.3, fontSize: 13, bold: true });
      txt(s, sub, { x: x + 0.2, y: 2.57, w: 2.6, h: 0.3, fontSize: 11, color: HEX.muted });
    });
    s.addChart(pres.charts.BAR, [{ name: "Points with NIC ≤ 6 (%)", labels: ["BAL", "MEA", "BLK", "CUS", "EMED", "INPK", "CEU", "AUS"], values: [15.8, 9.1, 8.5, 5.4, 4.7, 4.7, 4.1, 1.7] }],
      { ...chartBase("Share of points with NIC ≤ 6, first day (%)"), x: MX, y: 3.2, w: 6.1, h: 3.65, barDir: "col", chartColors: [HEX.accent1], showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelFormatCode: "0.0", objectName: "NIC share chart" });
    txt(s, [
      { text: "Source and licence", options: { bold: true, color: C.accent1, breakLine: true } },
      { text: "adsb.lol globe_history: daily readsb trace dumps, ODbL-1.0; attribution “ADSB.lol and contributors”. ADS-B Exchange samples rejected (evaluation-only terms bar derivative works).", options: { breakLine: true } },
      { text: " ", options: { fontSize: 6, breakLine: true } },
      { text: "Regions", options: { bold: true, color: C.accent1, breakLine: true } },
      { text: "Interference: BAL (Baltic), EMED, BLK (Black Sea), MEA (Iraq / Iran / Gulf), INPK (India–Pakistan), MYN. Comparison: CEU (low-interference Europe), clean controls CUS (US Midwest) and AUS (SE Australia).", options: { breakLine: true } },
      { text: " ", options: { fontSize: 6, breakLine: true } },
      { text: "Days", options: { bold: true, color: C.accent1, breakLine: true } },
      { text: "1st of each month, Jan 2024 → Oct 2026, plus 2026-10-25; the final pipeline is still downloading them. Point mix: ADS-B 95 %, ADS-R 2 %, MLAT 2 %, TIS-B < 1 %." },
    ], { x: 6.9, y: 3.25, w: 5.93, h: 3.6, fontSize: 12.5 });
    s.addNotes("Counts from paper/tables/macros.tex (interim: 5 days) and LOG.md Phase 2. CEU showed 1.2% jammed labels traced to real carry-over and a real cluster, so it is reported as low-interference, not as a clean control.");
  }

  // ============================================================ 10. System architecture
  section("System design");
  {
    const s = content("System design", "System Architecture", { kicker: "08 · SYSTEM DESIGN" });
    const y1 = 1.75, bh = 1.05, bw = 2.2, gap = 0.33;
    const row1 = [["adsb.lol archive", "daily trace dumps (ODbL)", ACC[1]], ["Parse + bin", "airborne, 30 s steps, 10-min windows", ACC[1]], ["Physics features", "23 per step incl. missingness", ACC[0]], ["10 labeling functions", "vote N / J / S / abstain", ACC[5]], ["Dawid–Skene", "EM label model → soft labels", ACC[5]]];
    row1.forEach(([t, sub, c], i) => {
      const x = MX + i * (bw + gap);
      box(s, x, y1, bw, bh, t, sub, c, { tsize: 13, ssize: 11 });
      if (i < row1.length - 1) arrow(s, x + bw + 0.03, y1 + bh / 2, x + bw + gap - 0.03, y1 + bh / 2);
    });
    // simulator branch
    const y2 = 3.55;
    box(s, MX, y2, bw, bh, "Clean real traffic", "control regions / quiet days", ACC[1], { tsize: 13, ssize: 11 });
    arrow(s, MX + bw + 0.03, y2 + bh / 2, MX + bw + gap - 0.03, y2 + bh / 2);
    box(s, MX + (bw + gap), y2, bw, bh, "Emitter simulator", "inject jammers / spoofers, exact truth", ACC[2], { tsize: 13, ssize: 11 });
    arrow(s, MX + 2 * bw + gap + 0.03, y2 + bh / 2, MX + 2 * (bw + gap) - 0.03, y2 + bh / 2);
    box(s, MX + 2 * (bw + gap), y2, bw, bh, "Scene dataset", "real (soft labels) + sim (exact)", ACC[0], { tsize: 13, ssize: 11 });
    arrow(s, MX + 2 * (bw + gap) + bw / 2, y1 + bh + 0.03, MX + 2 * (bw + gap) + bw / 2, y2 - 0.03);
    // model
    const mx = MX + 3 * (bw + gap), mw = 2 * bw + gap;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: mx, y: y2 - 0.15, w: mw, h: 2.35, fill: { color: HEX.dk2 }, line: { color: HEX.dk2 }, rectRadius: 0.08 });
    txt(s, "GIST model", { x: mx + 0.2, y: y2 - 0.07, w: mw - 0.4, h: 0.35, bold: true, color: "FFFFFF", fontSize: 15, fontFace: THEME.headFontFace });
    ["Trajectory Transformer (per aircraft)", "Spatial graph attention (across aircraft)", "Node + step classification heads", "Physics-guided localization head"].forEach((t, i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: mx + 0.2, y: y2 + 0.35 + i * 0.44, w: mw - 0.4, h: 0.36, fill: { color: "1F4A6E" }, line: { color: "1F4A6E" }, rectRadius: 0.05 });
      txt(s, t, { x: mx + 0.3, y: y2 + 0.35 + i * 0.44, w: mw - 0.6, h: 0.36, color: "FFFFFF", fontSize: 12, valign: "middle" });
    });
    arrow(s, MX + 2 * (bw + gap) + bw + 0.03, y2 + bh / 2, mx - 0.03, y2 + bh / 2);
    arrow(s, MX + 4 * (bw + gap) + bw / 2, y1 + bh + 0.03, MX + 4 * (bw + gap) + bw / 2, y2 - 0.18);
    txt(s, "soft labels = targets", { x: MX + 4 * (bw + gap) + bw / 2 + 0.08, y: y1 + bh + 0.12, w: 1.8, h: 0.3, fontSize: 10, italic: true, color: HEX.muted });
    // outputs
    const y3 = 6.05;
    txt(s, "OUTPUTS", { x: MX, y: y3 + 0.12, w: 1.5, h: 0.3, fontSize: 10, bold: true, color: HEX.muted, charSpacing: 3 });
    ["P(normal / jammed / spoofed) per aircraft-window", "Per-step class (when it started)", "Jammer location estimate", "Hex maps + case-study figures"].forEach((t, i) => {
      const x = MX + 1.5 + i * 2.72;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y3, w: 2.6, h: 0.55, fill: { color: HEX.accent1 }, line: { color: HEX.accent1 }, rectRadius: 0.06 });
      txt(s, t, { x: x + 0.08, y: y3, w: 2.44, h: 0.55, color: "FFFFFF", fontSize: 11, bold: true, align: "center", valign: "middle" });
    });
    arrow(s, mx + mw / 2, y2 + 2.22, mx + mw / 2, y3 - 0.03);
    s.addNotes("Top row: weak-supervision pipeline on real traffic. Bottom row: simulator gives exact targets, including the emitter position for localization. Both feed scene datasets; GIST is trained on the mix.");
  }

  // ============================================================ 11. Physics features
  {
    const s = content("System design", "Physics-Consistency Features", { kicker: "09 · HOW IT WORKS" });
    table(s, [
      ["Feature (per 30 s step)", "What it catches", "Can a spoofer fake it?"],
      ["NIC / NACp / SIL / GVA and their drops", "integrity collapse = jamming", { text: "Yes", options: { color: HEX.accent2, bold: true } }],
      ["Kinematic residual |‖Δp‖ − v·Δt|", "position jumps", { text: "Partly", options: { color: HEX.accent3, bold: true } }],
      ["GNSS − baro altitude vs neighbour median", "GNSS altitude drift", { text: "No (baro is independent)", options: { color: HEX.accent6, bold: true } }],
      ["readsb wind vs neighbour consensus", "broken wind triangle", { text: "No", options: { color: HEX.accent6, bold: true } }],
      ["Spoof cluster: ≥ 3 aircraft in one 0.005° cell, ≥ 5000 ft apart", "“spoof point” sharing", { text: "No", options: { color: HEX.accent6, bold: true } }],
      ["ADS-B gaps, stale flags, position loss", "jamming drop-outs", { text: "—", options: {} }],
    ], { x: MX, y: 1.55, w: 7.3, colW: [3.3, 2.1, 1.9], fontSize: 12, rowH: 0.55 });
    txt(s, "Neighbour consensus = trusted aircraft (NIC ≥ 7) in the same H3 res-3 cell, 5-min slot and 6000 ft band.", { x: MX, y: 5.55, w: 7.3, h: 0.5, fontSize: 11, italic: true, color: HEX.muted });
    s.addChart(pres.charts.BAR, [{ name: "per 10k steps", labels: ["INPK", "MEA", "EMED", "BLK", "CUS", "CEU", "AUS"], values: [23.2, 7.7, 1.4, 1.1, 0.06, 0.02, 0] }],
      { ...chartBase("Altitude-mismatch rate with NIC ≥ 7 (per 10k steps)"), x: 8.1, y: 1.5, w: 4.73, h: 3.6, barDir: "bar", chartColors: [HEX.accent3], showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", dataLabelFormatCode: "0.0#", objectName: "gdiff chart" });
    card(s, 8.1, 5.3, 4.73, 1.5, HEX.tealTint);
    txt(s, [{ text: "Validated before use. ", options: { bold: true, color: C.accent1 } },
      { text: "Spoof-sensitive features fire in spoofing regions while integrity looks fine, and almost never in controls. Naive wind from tas/heading was useless (p99 of several hundred kt) and is kept as a model feature only, not a labeling function." }],
      { x: 8.3, y: 5.38, w: 4.35, h: 1.35, fontSize: 12, valign: "middle" });
    s.addNotes("Rates from LOG.md Phase 3 (2026-10-25). The 0.05° grid gave false clusters at airports, so 0.005° is used.");
  }

  // ============================================================ 12. Weak supervision
  {
    const s = content("System design", "Weak Supervision: Labeling Functions + Dawid–Skene", { kicker: "10 · HOW IT WORKS" });
    table(s, [
      ["LF", "Vote", "Condition (10-minute window)"],
      ["nic_zero", "JAM", "modern, cruise, ≥ 30 % steps NIC ≤ 1"],
      ["nic_drop", "JAM", "cruise, NIC ≥ 8 seen and ≥ 30 % steps NIC ≤ 6"],
      ["pos_loss", "JAM", "cruise, ADS-B gap ≥ 90 s, ≥ 10 % steps NIC ≤ 6"],
      ["cell_jam", "JAM", "≥ 30 % of cell aircraft and ≥ 30 % own steps NIC ≤ 6"],
      ["teleport", "SPOOF", "position jump > 10 km with NIC ≥ 7"],
      ["gdiff", "SPOOF", "≥ 2 steps |Δh geo−baro − nbr| > 1000 ft, NIC ≥ 7"],
      ["cluster", "SPOOF", "member of co-located, altitude-spread cluster"],
      ["mlat †", "SPOOF", "MLAT vs ADS-B disagreement > 10 km"],
      ["clean", "NORMAL", "all NIC ≥ 8, no jump / gdiff / cluster, ≥ 10 steps"],
      ["clean_lowcell", "NORMAL", "modern, cell NIC-low share < 5 %, own NIC never low"],
    ], { x: MX, y: 1.5, w: 7.0, colW: [1.45, 0.95, 4.6], fontSize: 11, rowH: 0.42 });
    txt(s, "† held out of training labels; evaluation only. Region names are never used inside LFs.", { x: MX, y: 6.2, w: 7, h: 0.5, fontSize: 11, italic: true, color: HEX.muted });
    s.addChart(pres.charts.BAR, [{ name: "accuracy", labels: ["nic_zero", "cell_jam", "nic_drop", "gdiff", "teleport", "pos_loss", "cluster"], values: [0.998, 0.995, 0.94, 0.80, 0.79, 0.68, 0.30] }],
      { ...chartBase("LF accuracy estimated by the label model"), x: 7.8, y: 1.45, w: 5.03, h: 3.2, barDir: "bar", chartColors: [HEX.accent5], showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", valAxisMaxVal: 1, dataLabelFormatCode: "0.00", objectName: "LF accuracy chart" });
    card(s, 7.8, 4.8, 5.03, 2.0);
    txt(s, bullets([
      "EM estimates each LF's confusion matrix per class; output = soft labels for soft cross-entropy.",
      "Coverage 77–92 % of windows in every region; controls CUS 99.9 % / AUS 99.6 % normal.",
      "Bug found and fixed: all-abstain windows inherited a “spoofed” posterior (12.8 % in AUS) → now left unlabelled.",
    ], { gap: 4 }), { x: 8.0, y: 4.92, w: 4.7, h: 1.8, fontSize: 12 });
    s.addNotes("cluster has low estimated accuracy because it has tiny coverage and no co-firing LF. Numbers: LOG.md Phase 3 and Phase 5; LF table: paper/tables/lfs.tex.");
  }

  // ============================================================ 13. Model architecture
  section("Model");
  {
    const s = content("Model", "GIST Model Architecture", { kicker: "11 · MODEL" });
    // left: per-aircraft encoder
    const lx = MX, ly = 1.6;
    txt(s, "PER AIRCRAFT", { x: lx, y: ly, w: 3, h: 0.3, fontSize: 10, bold: true, color: HEX.muted, charSpacing: 3 });
    box(s, lx, ly + 0.4, 3.0, 0.8, "20 steps × 23 features", "30 s bins, missingness flags", ACC[1], { tsize: 13, ssize: 11 });
    arrow(s, lx + 1.5, ly + 1.23, lx + 1.5, ly + 1.5);
    box(s, lx, ly + 1.53, 3.0, 0.7, "MLP embed + [CLS] + pos. emb.", null, ACC[1], { tsize: 12 });
    arrow(s, lx + 1.5, ly + 2.26, lx + 1.5, ly + 2.53);
    box(s, lx, ly + 2.56, 3.0, 1.0, "Transformer encoder", "3 layers, pre-norm, d = 96, 4 heads", ACC[0], { tsize: 13, ssize: 11 });
    arrow(s, lx + 1.5, ly + 3.59, lx + 1.5, ly + 3.86);
    box(s, lx, ly + 3.89, 3.0, 0.7, "node token + 20 step tokens", null, ACC[0], { tsize: 12 });
    arrow(s, lx + 3.05, ly + 3.06, 3.95, ly + 3.06);
    // middle: graph
    const gx = 4.0;
    txt(s, "ACROSS AIRCRAFT (SCENE)", { x: gx, y: ly, w: 4.2, h: 0.3, fontSize: 10, bold: true, color: HEX.muted, charSpacing: 3 });
    card(s, gx, ly + 0.4, 4.2, 4.2, HEX.tealTint);
    // small graph drawing
    const nodes = [[0.7, 0.75], [1.7, 0.5], [2.9, 0.8], [1.2, 1.5], [2.3, 1.65], [3.4, 1.55], [0.8, 2.3], [2.0, 2.45], [3.1, 2.4]];
    const edges = [[0, 1], [1, 2], [0, 3], [1, 3], [1, 4], [2, 4], [2, 5], [4, 5], [3, 6], [3, 7], [4, 7], [5, 8], [7, 8], [6, 7]];
    const ox = gx + 0.1, oy = ly + 0.45;
    edges.forEach(([a, b]) => { const [x1, y1] = nodes[a], [x2, y2] = nodes[b];
      s.addShape(pres.shapes.LINE, { x: ox + Math.min(x1, x2), y: oy + Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001, flipV: (x2 - x1) * (y2 - y1) < 0, line: { color: "7FB8B1", width: 1.25 } }); });
    nodes.forEach(([x, y], i) => circleIcon(s, I.plane, ox + x - 0.2, oy + y - 0.2, 0.4, i === 4 || i === 5 ? HEX.accent3 : HEX.accent1));
    txt(s, [{ text: "Spatial graph attention × 3", options: { bold: true, color: C.accent1, breakLine: true } },
      { text: "Aircraft within 400 km attend to each other (≤ 256 per scene). Logits get a learned bias from edge features (Δx, Δy, d, Δh), GATv2-style, in plain PyTorch.", options: { fontSize: 11 } }],
      { x: gx + 0.2, y: ly + 3.3, w: 3.8, h: 1.25, fontSize: 13 });
    arrow(s, gx + 4.25, ly + 2.5, 8.75, ly + 1.0);
    arrow(s, gx + 4.25, ly + 2.5, 8.75, ly + 2.45);
    arrow(s, gx + 4.25, ly + 2.5, 8.75, ly + 3.9);
    // right: heads
    const hx = 8.8;
    txt(s, "HEADS", { x: hx, y: ly, w: 3, h: 0.3, fontSize: 10, bold: true, color: HEX.muted, charSpacing: 3 });
    box(s, hx, ly + 0.45, 4.03, 1.1, "Node head", "window class: normal / jammed / spoofed (soft cross-entropy on label-model targets)", ACC[1], { tsize: 13, ssize: 11 });
    box(s, hx, ly + 1.9, 4.03, 1.1, "Step head", "per-step class from step token ⊕ node embedding (exact sim labels)", ACC[5], { tsize: 13, ssize: 11 });
    box(s, hx, ly + 3.35, 4.03, 1.1, "Localization head", "physics likelihood + learned heatmap → soft-argmax (next slide)", ACC[2], { tsize: 13, ssize: 11 });
    card(s, MX, 6.3, 12.33, 0.55, HEX.lt2);
    txt(s, [{ text: "Training: ", options: { bold: true } }, { text: "AdamW, lr 3e-4, batch 4 scenes, 15 epochs (final runs: 8), fp32 on Apple MPS. Loss = soft CE (node) + CE (step) + 0.5 × (Gaussian-target CE + L1) for location. Checkpoint 2.0 MB." }],
      { x: MX + 0.2, y: 6.3, w: 12, h: 0.55, fontSize: 12, valign: "middle" });
    s.addNotes("Architecture from src/gist/models/gist.py and paper/main.tex. Orange nodes in the sketch are affected aircraft. No PyTorch Geometric; attention over padded neighbour tensors keeps it MPS-friendly.");
  }

  // ============================================================ 14. Localization head
  {
    const s = content("Model", "Differentiable Physics-Guided Localization", { kicker: "12 · MODEL" });
    const steps = [["1", "Step jam probabilities", "p_it from the step head at aircraft positions x_it and heights h_it", ACC[5]],
      ["2", "Physics likelihood", "for every grid cell g (21 × 21, 40 km) and EIRP e: received power P_r by Friis at L1, antenna gain vs elevation, 4/3-Earth horizon", ACC[2]],
      ["3", "Score", "ℓ(g,e) = Σ p log σ(z) + (1−p) log σ(−z),  z = (P_r − τ) / T,  τ = −115 dBW + learned offset", ACC[3]],
      ["4", "Fuse + soft-argmax", "logsumexp over e, plus a learned heatmap from grid-to-aircraft cross-attention; location = soft-argmax", ACC[0]],
      ["5", "Refine", "physics-guided coarse-to-fine search around the coarse estimate", ACC[1]]];
    steps.forEach(([n, h, b, col], i) => {
      const y = 1.55 + i * 1.0;
      card(s, MX, y, 7.6, 0.88);
      numBadge(s, n, MX + 0.2, y + 0.2, 0.48, col);
      txt(s, h, { x: MX + 0.9, y: y + 0.08, w: 6.5, h: 0.32, bold: true, fontSize: 14, color: col });
      txt(s, b, { x: MX + 0.9, y: y + 0.4, w: 6.55, h: 0.45, fontSize: 12 });
    });
    const dloc = R.gist("loc.median_km"), rloc = R.gist("loc_refined.median_km");
    card(s, 8.4, 1.55, 4.43, 1.95, HEX.amberTint);
    txt(s, "Effect of refinement (" + (R.status.final ? "test" : "val, seed 0") + ")", { x: 8.6, y: 1.65, w: 4.1, h: 0.35, bold: true, fontSize: 13, color: C.accent3 });
    txt(s, [{ text: f1(dloc && dloc.m) + " km", options: { fontSize: 30, bold: true, color: HEX.muted, breakLine: true } }, { text: "coarse soft-argmax median", options: { fontSize: 12 } }], { x: 8.6, y: 2.05, w: 2.0, h: 1.4, fontFace: THEME.headFontFace });
    txt(s, [{ text: f1(rloc && rloc.m) + " km", options: { fontSize: 30, bold: true, color: HEX.accent3, breakLine: true } }, { text: "after physics refinement", options: { fontSize: 12 } }], { x: 10.7, y: 2.05, w: 2.0, h: 1.4, fontFace: THEME.headFontFace });
    card(s, 8.4, 3.7, 4.43, 3.1);
    txt(s, [{ text: "Why a physics layer?", options: { bold: true, color: C.text2, breakLine: true } },
      ...bullets(["Only simulated emitters have known positions; physics keeps the head sensible on real traffic.",
        "Uses soft probabilities, so uncertain aircraft count less than with a hard affected set.",
        "Memory: 41 × 41 grid needed 2.3 GB autograd; 21 × 21 + refinement needs 0.36 GB."], { gap: 4 })],
      { x: 8.6, y: 3.85, w: 4.1, h: 2.85, fontSize: 13 });
    s.addNotes("Equation from paper/main.tex. Coarse vs refined medians read from the results JSON (loc.median_km vs loc_refined.median_km).");
  }

  // ============================================================ 15. Simulator
  section("Experiments");
  {
    const s = content("Experiments", "Emitter Simulator: Exact Targets in Real Traffic", { kicker: "13 · SIMULATION" });
    const colS = (x, color, ic, head, share, rows) => {
      card(s, x, 1.55, 6.05, 3.85);
      circleIcon(s, ic, x + 0.25, 1.75, 0.62, color);
      txt(s, head, { x: x + 1.05, y: 1.72, w: 3.5, h: 0.4, bold: true, fontSize: 18, color, fontFace: THEME.headFontFace });
      txt(s, share, { x: x + 1.05, y: 2.1, w: 4.5, h: 0.3, fontSize: 12, color: HEX.muted, italic: true });
      txt(s, bullets(rows, { gap: 7 }), { x: x + 0.3, y: 2.6, w: 5.5, h: 2.7, fontSize: 14 });
    };
    colS(MX, ACC[2], I.tower, "Jammer", "60 % of simulated scenes · EIRP 20–50 dBW", ["Ground (2–50 m) or airborne (1–10 km)", "Friis + 4/3-Earth line of sight + elevation-dependent antenna gain",
      "Receiver tolerance N(−115, 3²) dBW per aircraft", "Above tolerance: NIC → 0, GNSS altitude lost, positions drop with prob. σ(m/3)", "Within 6 dB below: NIC in 1–6"]);
    colS(6.78, ACC[3], I.sat, "Spoofer", "40 % of simulated scenes · EIRP 10–40 dBW", ["Capture when margin > −5 dB; capture is sticky", "Fake track: static point, circle or drift",
      "NIC forced ≥ 8 (realistic), GNSS altitude offset", "Ground speed / track / wind made consistent with the fake track", "Neighbour and cluster features recomputed after injection"]);
    card(s, MX, 5.6, 12.33, 1.2, HEX.lt2);
    txt(s, [{ text: "Calibration. ", options: { bold: true, color: C.accent1 } },
      { text: "A first 1–100 W prior affected a median 0.4 % of aircraft per scene, which is unrealistic; with the final priors a median 41 % are affected (q10 5 %, q90 82 %). Sanity gate: our NLS re-implementation reaches 2.4 km median on noise-free ground jammers (11 of 12 under 6 km)." }],
      { x: MX + 0.25, y: 5.65, w: 11.8, h: 1.1, fontSize: 13, valign: "middle" });
    s.addNotes("Simulator constants from Liu et al. 2022 (tolerable power about −115 dBW, C/N0 threshold 25 dB-Hz). Numbers from LOG.md Phase 4.");
  }

  // ============================================================ 16. Experimental setup
  {
    const s = content("Experiments", "Experimental Setup", { kicker: "14 · EXPERIMENTAL SETUP" });
    // timeline of splits
    txt(s, "DAY-BASED TEMPORAL SPLITS (NO DAY IN TWO SPLITS)", { x: MX, y: 1.5, w: 8, h: 0.3, fontSize: 10, bold: true, color: HEX.muted, charSpacing: 3 });
    const segs = [["Train", "Jan 2024 → Jun 2025", 5.2, ACC[1]], ["Validation", "Jul → Dec 2025", 2.9, ACC[2]], ["Test", "Jan → Oct 2026", 4.23, ACC[0]]];
    let x = MX;
    segs.forEach(([h, b, w, c]) => {
      s.addShape(pres.shapes.RECTANGLE, { x, y: 1.9, w: w - 0.05, h: 0.85, fill: { color: c }, line: { color: c } });
      txt(s, [{ text: h, options: { bold: true, breakLine: true, fontSize: 15 } }, { text: b, options: { fontSize: 12 } }], { x: x + 0.15, y: 1.9, w: w - 0.3, h: 0.85, color: "FFFFFF", valign: "middle" });
      x += w;
    });
    const blocks = [
      [I.balance, "Baselines (re-implemented)", ["gpsjam-style cell threshold (NIC share)", "Isolation Forest on window features", "LSTM autoencoder (reconstruction error)", "Localization: centroid, radio horizon, NLS (Liu et al.)"], ACC[1]],
      [I.chart, "Metrics", ["Macro-F1 over 3 classes", "Average precision for jam and spoof", "Spoofing recall at 1 % false-positive rate", "Localization median and p95 error (km)"], ACC[0]],
      [I.memory, "Compute", ["Apple M3, 8 GB RAM, PyTorch on MPS", "MPS allocator watermarks + cache clearing", "Final: 3 seeds + 3 ablations, unattended", "≤ 256 aircraft per scene, 21 × 21 grid"], ACC[2]],
    ];
    blocks.forEach(([ic, h, rows, c], i) => {
      const bx = MX + i * 4.15;
      card(s, bx, 3.05, 4.0, 3.75);
      circleIcon(s, ic, bx + 0.25, 3.25, 0.55, c);
      txt(s, h, { x: bx + 0.95, y: 3.3, w: 2.95, h: 0.45, bold: true, fontSize: 15, color: c, valign: "middle" });
      txt(s, bullets(rows, { gap: 8 }), { x: bx + 0.3, y: 4.0, w: 3.5, h: 2.7, fontSize: 14 });
    });
    s.addNotes("Splits from README / PLAN 8.3. Real-traffic metrics are scored against label-model targets; simulated metrics against exact truth. Single-score baselines use the same anomaly score for jam and spoof.");
  }

  // ============================================================ 17. Detection results
  section("Results");
  {
    const s = content("Results", "Results: Detection and Jam-vs-Spoof Classification", { kicker: "15 · RESULTS" });
    statusPill(s);
    const B = R.base;
    const row = (name, o) => [name, f2(o.real.macro_f1), f2(o.real.jam_ap), f2(o.real.spoof_ap), f2(o.real.spoof_recall_at_1fpr), f2(o.sim.macro_f1), f2(o.sim.spoof_recall_at_1fpr)];
    const gb = (t) => ({ text: t, options: { bold: true, color: HEX.accent1, fill: { color: HEX.tealTint } } });
    table(s, [
      ["Method", "Real F1", "Real Jam AP", "Real Spoof AP", "Real Spoof R@1%", "Sim F1", "Sim Spoof R@1%"],
      row("gpsjam-style threshold", B.gpsjam), row("Isolation Forest", B.iforest), row("LSTM autoencoder", B.lstm_ae),
      [gb("GIST" + (R.status.final ? "" : " *")), gb(g2("real.macro_f1")), gb(g2("real.jam_ap")), gb(g2("real.spoof_ap")), gb(g2("real.spoof_recall_at_1fpr")), gb(g2("sim.macro_f1")), gb(g2("sim.spoof_recall_at_1fpr"))],
    ], { x: MX, y: 1.55, w: 12.33, colW: [2.85, 1.38, 1.6, 1.6, 1.75, 1.38, 1.77], fontSize: 13, rowH: 0.48, align: "center" });
    const names = ["gpsjam", "iForest", "LSTM-AE", "GIST"];
    const sv = [B.gpsjam.real.spoof_recall_at_1fpr, B.iforest.real.spoof_recall_at_1fpr, B.lstm_ae.real.spoof_recall_at_1fpr, gv("real.spoof_recall_at_1fpr") || 0];
    const fv = [B.gpsjam.real.macro_f1, B.iforest.real.macro_f1, B.lstm_ae.real.macro_f1, gv("real.macro_f1") || 0];
    s.addChart(pres.charts.BAR, [{ name: "Real macro-F1", labels: names, values: fv }, { name: "Real spoof recall @1% FPR", labels: names, values: sv }],
      { ...chartBase(), x: MX, y: 4.15, w: 7.2, h: 2.75, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.dk2, HEX.accent3], showLegend: true, legendPos: "t", valAxisMaxVal: 1, valAxisMinVal: 0, valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.00", objectName: "detection chart" });
    card(s, 7.95, 4.15, 4.88, 2.75, HEX.lt2);
    txt(s, [{ text: "Reading the table", options: { bold: true, color: C.text2, breakLine: true } },
      ...bullets([
        "The integrity threshold finds jamming (AP " + f2(B.gpsjam.real.jam_ap) + ") but spoof recall is " + f2(B.gpsjam.real.spoof_recall_at_1fpr) + ": it cannot see spoofing.",
        "GIST separates the two classes; spoof recall @1% FPR " + g2("real.spoof_recall_at_1fpr") + " on real traffic.",
        R.status.final ? "Mean ± std over " + R.gist("real.macro_f1").n + " seeds on the 2026 test days." : "* GIST = interim seed-0 model on the same 2026 test days as the baselines (n = " + Math.round(gv("real.n")).toLocaleString("en-US") + " windows, " + Math.round(gv("real.n_spoof")) + " spoofed). Final 3-seed numbers will replace this row.",
      ], { gap: 4 })], { x: 8.15, y: 4.25, w: 4.5, h: 2.6, fontSize: 12 });
    s.addNotes("Real = scored against label-model targets (weak labels, not ground truth). Sim = exact simulated truth. Values read from data/runs/*.json by build_deck.js.");
  }

  // ============================================================ 18. Localization results
  {
    const s = content("Results", "Results: Jammer Localization and Distractor Robustness", { kicker: "16 · RESULTS" });
    statusPill(s);
    const L = R.base.localization, LD = R.baseD.localization;
    const gc = gv("loc_refined.median_km"), gd = gv("loc_refined.median_km", R.gistD);
    const labs = ["Centroid", "Radio horizon", "NLS (Liu)", "GIST"];
    s.addChart(pres.charts.BAR, [{ name: "median error (km)", labels: labs, values: [L.centroid.median_km, L.horizon.median_km, L.nls.median_km, gc || 0] }],
      { ...chartBase("Median error on simulated jammers, clean scenes (km, lower is better)"), x: MX, y: 1.5, w: 6.4, h: 3.5, barDir: "col", chartColors: [HEX.accent1], showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, dataLabelFormatCode: "0.0", objectName: "localization chart" });
    table(s, [
      ["Method [km]", "Clean median", "Clean p95", "+5 % benign low-NIC median", "p95"],
      ["Centroid of affected", f1(L.centroid.median_km), f1(L.centroid.p95_km), f1(LD.centroid.median_km), f1(LD.centroid.p95_km)],
      ["Radio-horizon score", f1(L.horizon.median_km), f1(L.horizon.p95_km), f1(LD.horizon.median_km), f1(LD.horizon.p95_km)],
      ["NLS (Liu et al.)", f1(L.nls.median_km), f1(L.nls.p95_km), f1(LD.nls.median_km), f1(LD.nls.p95_km)],
      [{ text: "GIST" + (R.status.final ? "" : " *"), options: { bold: true, color: HEX.accent1, fill: { color: HEX.tealTint } } },
        ...[g1("loc_refined.median_km"), g1("loc_refined.p95_km"), g1("loc_refined.median_km", R.gistD), g1("loc_refined.p95_km", R.gistD)].map((t) => ({ text: t, options: { bold: true, color: HEX.accent1, fill: { color: HEX.tealTint } } }))],
    ], { x: 7.15, y: 1.55, w: 5.68, colW: [1.75, 0.95, 0.85, 1.38, 0.75], fontSize: 11, rowH: 0.5, align: "center" });
    const nlsUp = Math.round((LD.nls.median_km / L.nls.median_km - 1) * 100);
    card(s, 7.15, 4.35, 5.68, 2.5, HEX.amberTint);
    txt(s, [{ text: "Distractor robustness", options: { bold: true, color: C.accent3, breakLine: true } },
      ...bullets([
        "Adding 5 % benign low-NIC aircraft (affected by something else) moves NLS from " + f1(L.nls.median_km) + " to " + f1(LD.nls.median_km) + " km median (+" + nlsUp + " %).",
        gd == null ? "GIST on the distractor split: pending the final pipeline (eval_test_distract.json)." : "GIST on the same scenes: " + g1("loc_refined.median_km", R.gistD) + " km median (p95 " + g1("loc_refined.p95_km", R.gistD) + "): it learns to discount aircraft whose low NIC is unrelated to the emitter.",
        "Simulated jammers only; real-world localization is not yet validated.",
      ], { gap: 4 })], { x: 7.35, y: 4.45, w: 5.3, h: 2.35, fontSize: 12 });
    txt(s, R.status.final ? "GIST: mean ± std over seeds, test split." : "* GIST interim: seed 0, on the same 2026 test scenes as the baselines (n = " + Math.round(gv("loc_refined.n")) + " clean, " + (gv("loc_refined.n", R.gistD) == null ? "–" : Math.round(gv("loc_refined.n", R.gistD))) + " distractor).",
      { x: MX, y: 5.2, w: 6.4, h: 0.6, fontSize: 11, italic: true, color: HEX.muted });
    card(s, MX, 5.9, 6.4, 0.95, HEX.lt2);
    txt(s, [{ text: "Information limit. ", options: { bold: true, color: C.accent1 } }, { text: "Even oracle step labels reach only 17.2 km median on the dev sims: receiver tolerance spread and unknown emitter height bound accuracy." }],
      { x: MX + 0.2, y: 5.93, w: 6.0, h: 0.9, fontSize: 12, valign: "middle" });
    s.addNotes("GIST row: loc_refined from the results JSON. Oracle 17.2 km from LOG.md Phase 5 dev diagnostics.");
  }

  // ============================================================ 19. Ablations
  {
    const s = content("Results", "Ablations: What Each Component Buys", { kicker: "17 · RESULTS" });
    const anyAbl = R.abl.nograph || R.abl.integ || R.abl.nosim;
    pill(s, !anyAbl ? "ABLATIONS RUNNING" : R.status.final ? "FINAL · SEED 0 · TEST" : "INTERIM · SEED 0 · TEST", W - MX - 3.6, 0.3, 3.6, anyAbl ? HEX.accent6 : HEX.muted);
    const cell = (r, k, d = 2) => { const v = r ? get(r, k) : null; return v == null ? "pending" : v.toFixed(d); };
    const rows = [["GIST (full)", R.abl.full, "—"], ["w/o spatial graph attention", R.abl.nograph, "does sharing evidence across aircraft help?"],
      ["integrity fields only", R.abl.integ, "are physics features what catches spoofing?"], ["w/o simulated training scenes", R.abl.nosim, "is the simulator needed for localization?"]];
    table(s, [["Variant", "Question", "Real F1", "Real Spoof R@1%", "Sim F1", "Loc. median km"],
      ...rows.map(([n, r, q]) => [n, q, cell(r, "real.macro_f1"), cell(r, "real.spoof_recall_at_1fpr"), cell(r, "sim.macro_f1"), cell(r, "loc_refined.median_km", 1)])],
      { x: MX, y: 1.6, w: 12.33, colW: [2.9, 4.13, 1.2, 1.6, 1.1, 1.4], fontSize: 13, rowH: 0.6 });
    // Each card states the measured effect when the run exists, otherwise the hypothesis being tested.
    const delta = (r, k, d, unit = "") => {
      const a = R.abl.full ? get(R.abl.full, k) : null, b = r ? get(r, k) : null;
      return a == null || b == null ? null : a.toFixed(d) + unit + " → " + b.toFixed(d) + unit;
    };
    const ins = [
      [I.layer, "Graph", R.abl.nograph, [["Real spoof R@1%", delta(R.abl.nograph, "real.spoof_recall_at_1fpr", 2)], ["Loc. p95", delta(R.abl.nograph, "loc_refined.p95_km", 0, " km")]],
        "Without neighbours, a single aircraft cannot tell a local receiver fault from a regional emitter.", ACC[0], "Neighbours catch spoofing and fix the worst localization misses."],
      [I.shield, "Physics features", R.abl.integ, [["Real spoof R@1%", delta(R.abl.integ, "real.spoof_recall_at_1fpr", 2)], ["Sim F1", delta(R.abl.integ, "sim.macro_f1", 2)]],
        "The integrity-only variant mirrors what NIC maps see; spoof recall should collapse toward the threshold baseline.", ACC[3], "Spoofers fake integrity fields; physics features catch them."],
      [I.flask, "Simulation", R.abl.nosim, [["Loc.", delta(R.abl.nosim, "loc_refined.median_km", 1, " km")], ["Sim F1", delta(R.abl.nosim, "sim.macro_f1", 2)]],
        "Only simulated scenes carry emitter positions and step labels; without them the localization head has no target.", ACC[2], "Localization is learned almost entirely from simulation."]];
    ins.forEach(([ic, h, run, deltas, hyp, c, take], i) => {
      const x = MX + i * 4.15;
      card(s, x, 4.95, 4.0, 1.85);
      circleIcon(s, ic, x + 0.25, 5.15, 0.55, c);
      txt(s, h + (run ? ": full → ablated" : " (hypothesis)"), { x: x + 0.95, y: 5.15, w: 2.95, h: 0.55, bold: true, fontSize: 14, color: c, valign: "middle" });
      const body = run
        ? [...deltas.filter(([, v]) => v).map(([k, v]) => ({ text: k + ": " + v, options: { breakLine: true } })),
           { text: take, options: { bold: true, fontSize: 11, color: c } }]
        : [{ text: hyp }];
      txt(s, body, { x: x + 0.3, y: 5.75, w: 3.5, h: 1.0, fontSize: 12, valign: "top" });
    });
    s.addNotes("Each ablation retrains GIST with one component removed, using the same data, seed and 15 epochs as the interim model, then scores it on the 2026 test days. Cards show full → ablated. Integrity-only: real spoof recall falls from 0.83 to 0.29, so the physics features are what catch spoofing. No graph: spoof recall drops 0.83 to 0.74 and the 95th-percentile localization error grows, though the median is similar. No simulation: localization error goes from about 18 km to about 178 km, because only simulated scenes carry emitter positions; real-data classification is unchanged.");
  }

  // ============================================================ 20. Case study
  {
    const s = content("Results", "Case Study: Baltic Sea, 1 January 2026", { kicker: "18 · CASE STUDY" });
    pill(s, "INTERIM MODEL · HELD-OUT TEST DAY", W - MX - 3.6, 0.3, 3.6, HEX.accent6);
    s.addImage({ path: path.join(HERE, "assets", "case_BAL_2026-01-01_interim.png"), x: MX, y: 1.5, w: 12.33, h: 12.33 * CASE_H / CASE_W, altText: "Three hexagon maps of the Baltic on 2026-01-01: baseline share of NIC ≤ 6, GIST P(jammed) with a jamming footprint centred on the reported Kaliningrad source and 169 jammer estimates, and GIST P(spoofed), which is near zero" });
    const notes = [["Baseline", "Share of aircraft with NIC ≤ 6 per hexagon: the gpsjam-style view.", ACC[1]],
      ["GIST P(jammed)", "The jamming footprint is centred on the reported Kaliningrad source (±150 km). The 169 jammer estimates are spread out: real-world localization is not validated yet.", ACC[2]],
      ["GIST P(spoofed)", "Near zero everywhere: consistent with the Baltic being a jamming region, not spoofing.", ACC[5]]];
    notes.forEach(([h, b, c], i) => {
      const x = MX + i * 4.15;
      card(s, x, 5.55, 4.0, 1.3);
      txt(s, [{ text: h, options: { bold: true, color: c, breakLine: true } }, { text: b, options: { fontSize: 12 } }], { x: x + 0.2, y: 5.62, w: 3.6, h: 1.18, fontSize: 14 });
    });
    s.addNotes("Figure from runs/gist_s0/fig/case_BAL_2026-01-01.png: the interim seed-0 model run over every aircraft on a held-out test day (every third 10-minute window). Point out that the orange footprint sits around Kaliningrad while the simple map shows similar coverage but cannot separate spoofing. Be honest that the individual jammer estimates scatter; the Denver 2022 ground truth (OpenSky) is the planned real-world test. The reported source has ±150 km uncertainty, so this is qualitative.");
  }

  // ============================================================ 21. Challenges and lessons
  section("Conclusion");
  {
    const s = content("Conclusion", "Challenges and Lessons Learned", { kicker: "19 · CHALLENGES" });
    const ch = [[I.balance, "Data licence", "ADS-B Exchange samples forbid derivative works → switched to adsb.lol (ODbL).", ACC[1]],
      [I.bug, "Label-model artefact", "All-abstain windows became “spoofed” (12.8 % in AUS) → left unlabelled.", ACC[3]],
      [I.wind, "Noisy wind", "Naive wind triangle useless → fresh readsb wind, consensus per cell; feature, not LF.", ACC[0]],
      [I.memory, "8 GB RAM", "MPS cache grew to 7.6 GB (epochs 2 → 36 min) → allocator watermarks; 21 × 21 grid.", ACC[2]],
      [I.cloud, "Download throttling", "GitHub assets fell to bytes/s after ~7 GB → resumable per-part fetch with back-off.", ACC[5]],
      [I.cross, "Flattering baselines", "Clean-scene sampling helped NLS → added a benign low-NIC distractor split.", ACC[4]]];
    ch.forEach(([ic, h, b, c], i) => {
      const x = MX + (i % 3) * 4.15, y = 1.55 + Math.floor(i / 3) * 2.15;
      card(s, x, y, 4.0, 1.98);
      circleIcon(s, ic, x + 0.25, y + 0.25, 0.58, c);
      txt(s, h, { x: x + 1.0, y: y + 0.25, w: 2.9, h: 0.58, bold: true, fontSize: 15, color: c, valign: "middle" });
      txt(s, b, { x: x + 0.3, y: y + 0.98, w: 3.5, h: 0.95, fontSize: 13 });
    });
    card(s, MX, 5.95, 12.33, 0.88, HEX.tealTint);
    txt(s, [{ text: "Lesson: ", options: { bold: true, color: C.accent1 } }, { text: "check the data and the baseline before the model. Most of the time went into licences, labels and fair evaluation, and each fix changed the numbers more than any architecture change." }],
      { x: MX + 0.25, y: 5.95, w: 11.8, h: 0.88, fontSize: 14, valign: "middle" });
    s.addNotes("All items are recorded in LOG.md with the date and the fix.");
  }

  // ============================================================ 22. Limitations
  {
    const s = content("Conclusion", "Limitations (Stated Honestly)", { kicker: "20 · LIMITATIONS" });
    const lim = [["Weak labels are not ground truth", "Real-traffic scores measure agreement with our physics assumptions. The MLAT LF is held out as an independent check; spoofed windows are few (374 in the 2026 test days)."],
      ["Simulation is simplified", "Free-space Friis loss, a coarse antenna model and a Gaussian receiver tolerance. Sim scores near 1.0 show the model learns the simulator, not that it is solved."],
      ["Real-world localization unvalidated", "Localization numbers are on simulated jammers. The only real anchor so far (Kaliningrad) has ±150 km uncertainty; the Denver 2022 event with a known site needs OpenSky data."],
      ["Uneven receiver coverage", "Crowdsourced receivers are sparse over parts of the Middle East and South Asia, so detection there is biased toward covered airspace."],
      ["Interim numbers", "Results come from one seed trained on two days of data. The 34-day, 3-seed run is pending."]];
    lim.forEach(([h, b], i) => {
      const y = 1.55 + i * 1.04;
      card(s, MX, y, 12.33, 0.92);
      numBadge(s, i + 1, MX + 0.22, y + 0.2, 0.5, i === 2 ? HEX.accent2 : HEX.accent3);
      txt(s, h, { x: MX + 0.95, y: y + 0.08, w: 3.6, h: 0.76, bold: true, fontSize: 14, valign: "middle" });
      txt(s, b, { x: 4.6, y: y + 0.08, w: 8.05, h: 0.76, fontSize: 12.5, valign: "middle" });
    });
    s.addNotes("Ethics: all data are public broadcasts; we publish aggregates and events, not per-aircraft tracks. The work is defensive.");
  }

  // ============================================================ 23. Conclusion
  {
    const s = content("Conclusion", "Conclusion", { kicker: "21 · CONCLUSION" });
    card(s, MX, 1.55, 12.33, 1.75);
    txt(s, "GIST shows that crowdsourced ADS-B carries enough physical redundancy to separate GNSS spoofing from jamming without a single hand-drawn label. Physics-consistency features expose spoofed aircraft that report high integrity; a Transformer with spatial graph attention turns them into per-aircraft classes; and a differentiable Friis / radio-horizon layer localizes simulated jammers more accurately than least squares (" + g1("loc_refined.median_km") + " vs " + f1(R.base.localization.nls.median_km) + " km median on the same test scenes) and degrades less when unrelated low-integrity aircraft are present (" + g1("loc_refined.median_km", R.gistD) + " vs " + f1(R.baseD.localization.nls.median_km) + " km).",
      { x: MX + 0.3, y: 1.65, w: 11.7, h: 1.55, fontSize: 14, valign: "middle" });
    const kp = [[I.eye, "Sees spoofing", "Spoof recall @1% FPR " + g2("real.spoof_recall_at_1fpr") + " vs " + f2(R.base.gpsjam.real.spoof_recall_at_1fpr) + " for the NIC threshold.", ACC[3]],
      [I.tag, "No manual labels", "10 physics LFs + Dawid–Skene; simulator for exact targets.", ACC[5]],
      [I.cross, "Physics-guided localization", "Median " + g1("loc_refined.median_km") + " km on simulated jammers.", ACC[2]]];
    kp.forEach(([ic, h, b, c], i) => {
      const x = MX + i * 4.15;
      card(s, x, 3.55, 4.0, 2.2, TINT[c]);
      circleIcon(s, ic, x + 0.25, 3.75, 0.62, c);
      txt(s, h, { x: x + 0.25, y: 4.5, w: 3.5, h: 0.4, bold: true, fontSize: 16, color: c, fontFace: THEME.headFontFace });
      txt(s, b, { x: x + 0.25, y: 4.92, w: 3.5, h: 0.75, fontSize: 13 });
    });
    txt(s, R.status.final ? "Numbers: mean over seeds on the 2026 test days." : "Numbers: interim seed-0 model on the 2026 test days; final 3-seed results pending.",
      { x: MX, y: 5.95, w: 12, h: 0.3, fontSize: 11, italic: true, color: HEX.muted });
    s.addNotes("Close on the main message: the redundancy between independent sensors on hundreds of aircraft is what makes spoofing detectable.");
  }

  // ============================================================ 24. Future work
  {
    const s = content("Conclusion", "Future Work", { kicker: "22 · FUTURE ENHANCEMENT" });
    const fw = [["Real localization ground truth", "Denver KDEN, Jan 2022 (FAA-documented jammer) via OpenSky Trino; access requested 2026-10-27."],
      ["More real events", "Event catalogue for BAL, EMED, INPK with reported source areas and uncertainty."],
      ["Independent spoof checks", "MLAT-only release and receiver radio-horizon plausibility as held-out evaluation signals."],
      ["Self-supervised pretraining", "Masked-step reconstruction on unlabelled windows (head already in the model)."],
      ["Live monitor", "Minute-level latency evaluation and a static hex-map demo for a chosen date."]];
    const cx = (i) => MX + 0.55 + i * 2.47;
    s.addShape(pres.shapes.LINE, { x: cx(0) + 0.4, y: 1.95, w: cx(4) - cx(0), h: 0, line: { color: HEX.grid, width: 2 } });
    fw.forEach(([h, b], i) => {
      numBadge(s, String(i + 1).padStart(2, "0"), cx(i), 1.55, 0.8, ACC[i]);
      card(s, MX + i * 2.47, 2.65, 2.33, 3.3);
      txt(s, h, { x: MX + i * 2.47 + 0.18, y: 2.8, w: 1.97, h: 0.85, bold: true, fontSize: 14, color: ACC[i], fontFace: THEME.headFontFace });
      txt(s, b, { x: MX + i * 2.47 + 0.18, y: 3.75, w: 1.97, h: 2.1, fontSize: 14 });
    });
    txt(s, "Source: PLAN.md sections 5.3, 6.3 and 8; DATA_LICENSE.md.", { x: MX, y: 6.15, w: 10, h: 0.3, fontSize: 10, italic: true, color: HEX.muted });
    s.addNotes("Item 1 is the most important: it would give the first real-world localization error. adsb.lol archives start in 2023, so the 2022 Denver event needs OpenSky.");
  }

  // ============================================================ 25. References
  {
    const s = content("Conclusion", "References", { kicker: "23 · REFERENCES" });
    const refs = [
      "Z. Liu, S. Lo and T. Walter, “GNSS interference source localization using ADS-B data,” Proc. ION ITM, 2022.",
      "Z. Liu, S. Lo, J. Blanch, Y.-H. Chen and T. Walter, “Locating GNSS interference sources using ADS-B with non-linear least squares,” NAVIGATION, vol. 72, no. 3, 2025.",
      "M. Schäfer et al., “Jammer on the Horizon: A robust method for GPS jammer localization using ADS-B data,” disco.cs.rptu.de.",
      "J. Wiseman, “GPSJAM GPS/GNSS interference map,” gpsjam.org.",
      "F. Eichstaedt, J. Budroweit and F. Stehle, “Detecting and measuring global GNSS signal interference via space-based ADS-B surveillance,” IEEE SHaRC, 2026.",
      "Aireon, “GNSS spoofing detection in aviation,” white paper v1.0, 2026.",
      "A. Ratner et al., “Snorkel: Rapid training data creation with weak supervision,” Proc. VLDB Endowment, vol. 11, no. 3, 2017.",
      "A. P. Dawid and A. M. Skene, “Maximum likelihood estimation of observer error-rates using the EM algorithm,” Applied Statistics, vol. 28, no. 1, pp. 20–28, 1979.",
      "S. Brody, U. Alon and E. Yahav, “How attentive are graph attention networks?,” ICLR, 2022.",
      "A. Vaswani et al., “Attention is all you need,” NeurIPS, 2017.",
      "ADSB.lol contributors, “adsb.lol globe_history open data (ODbL),” github.com/adsblol; Wiedehopf, “readsb,” github.com/wiedehopf/readsb.",
    ];
    refs.forEach((r, i) => {
      const y = 1.5 + i * 0.48;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 12.33, h: 0.42, fill: { color: HEX.lt2 }, line: { color: HEX.lt2 }, rectRadius: 0.04 });
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 0.6, h: 0.42, fill: { color: ACC[i % 6] }, line: { color: ACC[i % 6] } });
      txt(s, `[${i + 1}]`, { x: MX, y, w: 0.6, h: 0.42, align: "center", valign: "middle", fontSize: 11, bold: true, color: "FFFFFF" });
      txt(s, r, { x: MX + 0.75, y, w: 11.45, h: 0.42, valign: "middle", fontSize: 11 });
    });
    s.addNotes("From paper/related.bib.");
  }

  // ============================================================ 26. Thank you
  {
    const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
    circleIcon(s, I.sat, W / 2 - 0.45, 1.5, 0.9, HEX.accent1);
    txt(s, "Thank You", { x: 0, y: 2.65, w: W, h: 1.0, align: "center", fontFace: THEME.headFontFace, bold: true, fontSize: 48, color: "FFFFFF" });
    txt(s, "Questions and discussion", { x: 0, y: 3.65, w: W, h: 0.5, align: "center", fontFace: THEME.headFontFace, italic: true, fontSize: 20, color: "CADCFC" });
    txt(s, "Data: ADSB.lol and contributors (ODbL-1.0)  ·  Code: MIT", { x: 0, y: 4.45, w: W, h: 0.4, align: "center", fontSize: 14, color: "5EC4B6" });
    txt(s, "Vansh Agarwal (1BI23IS133)  ·  Vichitra Attri (1BI23IS135)  ·  Vidya Bai K (1BI23IS137)", { x: 0, y: 5.6, w: W, h: 0.35, align: "center", fontSize: 13, color: "FFFFFF" });
    txt(s, "Faculty: Dr. Asha T., Deep Learning course, Department of ISE, Bangalore Institute of Technology", { x: 0, y: 5.95, w: W, h: 0.35, align: "center", fontSize: 13, color: "FFFFFF" });
  }

  await pres.writeFile({ fileName: OUT });
  const SKILL = process.env.PPTX_SKILL;
  if (SKILL) { const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js")); await applyTheme(OUT, THEME); }
  console.log("wrote", OUT, "|", R.status.tag);
}
build().catch((e) => { console.error(e); process.exit(1); });
