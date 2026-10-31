const scenarioValues = {
  normal: { speed: 60, brake: 20, drowsy: 15, gap: 60, rain: 10 },
  drowsy: { speed: 95, brake: 25, drowsy: 80, gap: 35, rain: 10 },
  wet: { speed: 115, brake: 55, drowsy: 30, gap: 12, rain: 75 },
};
const sliderDefinitions = [
  { key: "speed", label: "Speed", unit: "km/h", min: 20, max: 160 },
  { key: "brake", label: "Hard-braking intensity", unit: "%", min: 0, max: 100 },
  { key: "drowsy", label: "Eye closure / drowsiness", unit: "%", min: 0, max: 100 },
  { key: "gap", label: "Following distance", unit: "m", min: 5, max: 80 },
  { key: "rain", label: "Rain / low road grip", unit: "%", min: 0, max: 100 },
];
const viewCopy = {
  dashboard: ["Safety overview", "Live GPS and hardware readings. No fabricated values are substituted for missing sensors."],
  sensors: ["Live sensor stream", "Inspect timestamped GPS, motion, and device sensor events."],
  model: ["Crash severity model", "Train and evaluate a historical severity classifier using your labeled dataset."],
  monitor: ["What-if scenario lab", "Test alert behavior separately from real sensor data."],
  sos: ["SOS & alerts", "Manage emergency escalation and review stored alert activity."],
  hq: ["Incident console", "Review persisted event coordinates, update local status, and capture outcomes."],
  system: ["System design", "Inspect data flow, device integration, and deployment boundaries."],
  trips: ["Trip history", "Review monitoring sessions and observed sensor activity."],
  settings: ["Monitoring settings", "Configure alert sensitivity and refresh cadence."],
};
const state = {
  values: { ...scenarioValues.normal },
  sensorEvents: [],
  sensors: { gps: null, motion: null, obd: null, vision: null },
  incidents: [],
  alerts: [],
  trips: [],
  threshold: 60,
  interval: 3,
  refreshTimer: null,
  autoTimer: null,
  countdownTimer: null,
  gpsWatchId: null,
  motionEnabled: false,
  lastMotionSent: 0,
  activeTrip: false,
  lastRisk: 0,
  sendingSos: false,
  sosSubmitting: false,
  datasetFile: null,
  datasetColumns: [],
  modelMetadata: null,
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const timeLabel = (value) => value ? new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";
const dateLabel = (value) => value ? new Date(value).toLocaleString() : "—";
const sourceLabel = (source) => ({ legacy: "Legacy prototype", model: "Risk model", manual_sos: "Manual SOS", scenario_demo: "Scenario demo", local_operator: "Operator action" })[source] || "Device / operator";

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = response.headers.get("content-type")?.includes("application/json") ? await response.json() : {};
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function uploadDataset(path, file) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
  const data = response.headers.get("content-type")?.includes("application/json") ? await response.json() : {};
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function renderFeatureOptions() {
  const target = $("#targetColumn").value;
  $("#featureColumns").innerHTML = state.datasetColumns
    .filter((column) => column.name !== target)
    .map((column) => `
      <label class="feature-option">
        <input type="checkbox" value="${escapeHtml(column.name)}">
        <span>${escapeHtml(column.name)}</span>
        <small>${escapeHtml(column.kind)} · ${column.non_empty} present · ${column.unique} unique</small>
      </label>`).join("");
  $("#trainModel").disabled = true;
}

function renderModelResults(metadata) {
  state.modelMetadata = metadata;
  $("#modelResults").hidden = false;
  $("#modelTrainingTime").textContent = `Trained ${dateLabel(metadata.trained_at)} · ${metadata.rows_used} labeled rows`;
  $("#modelProvenance").textContent = `Dataset: ${metadata.dataset_source}. Permission basis: ${metadata.dataset_permission}.`;
  $("#modelMetricCards").innerHTML = [
    ["Accuracy", metadata.metrics.accuracy],
    ["Majority baseline", metadata.metrics.majority_baseline_accuracy],
    ["Balanced accuracy", metadata.metrics.balanced_accuracy],
    ["Macro F1", metadata.metrics.macro_f1],
    ["Evaluation split", `${metadata.test_rows} / ${metadata.rows_used}`],
  ].map(([label, value]) => `<div class="metric"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b><small>Holdout set</small></div>`).join("");
  $("#classMetrics").innerHTML = metadata.classes.map((label) => {
    const report = metadata.class_report[label] || {};
    return `<tr><td>${escapeHtml(label)}</td><td>${report.precision ?? "—"}</td><td>${report.recall ?? "—"}</td><td>${report["f1-score"] ?? "—"}</td><td>${report.support ?? 0}</td></tr>`;
  }).join("");
  const heading = metadata.classes.map((label) => `<th>${escapeHtml(label)}</th>`).join("");
  const rows = metadata.confusion_matrix.map((row, index) => `
    <tr><th>${escapeHtml(metadata.classes[index])}</th>${row.map((count) => `<td>${count}</td>`).join("")}</tr>`).join("");
  $("#confusionMatrix").innerHTML = `<table><thead><tr><th>Actual \\ Predicted</th>${heading}</tr></thead><tbody>${rows}</tbody></table>`;
  $("#modelWarnings").innerHTML = metadata.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  $("#severityAssessment").hidden = false;
  $("#assessmentResult").textContent = "";
  $("#assessmentFields").innerHTML = metadata.features.map((feature) => {
    const numeric = metadata.feature_schema[feature] === "numeric";
    return `<label>${escapeHtml(feature)}<input data-feature="${escapeHtml(feature)}" type="${numeric ? "number" : "text"}" ${numeric ? 'step="any"' : ""}></label>`;
  }).join("");
}

async function loadModelStatus() {
  const status = await api("/api/model/status");
  if (status.trained) {
    renderModelResults(status);
    $("#datasetStatus").textContent = `Saved model ready: ${status.classes.join(", ")} using ${status.features.length} selected features.`;
  } else {
    $("#datasetStatus").textContent = "No trained severity model. Upload a labeled CSV or Excel file to begin.";
  }
}

async function inspectDataset() {
  const file = $("#datasetFile").files[0];
  const status = $("#datasetStatus");
  if (!file) {
    status.textContent = "Choose a CSV or Excel file first.";
    return;
  }
  $("#inspectDataset").disabled = true;
  status.textContent = "Inspecting the selected file…";
  $("#datasetControls").hidden = true;
  $("#modelResults").hidden = true;
  $("#severityAssessment").hidden = true;
  state.datasetFile = file;
  try {
    const query = new URLSearchParams({ filename: file.name });
    const dataset = await uploadDataset(`/api/dataset/inspect?${query}`, file);
    state.datasetColumns = dataset.columns;
    $("#targetColumn").innerHTML = dataset.columns.map((column) =>
      `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)}</option>`).join("");
    $("#datasetControls").hidden = false;
    renderFeatureOptions();
    status.textContent = `File checked: ${dataset.rows.toLocaleString()} rows, ${dataset.columns.length} columns. Raw rows were not saved. Select the severity label and input features.`;
  } catch (error) {
    state.datasetFile = null;
    status.textContent = error.message;
  } finally {
    $("#inspectDataset").disabled = false;
  }
}

async function trainSeverityModel() {
  const file = state.datasetFile;
  const target = $("#targetColumn").value;
  const features = [...$("#featureColumns").querySelectorAll("input:checked")].map((input) => input.value);
  const datasetSource = $("#datasetSource").value.trim();
  const datasetPermission = $("#datasetPermission").value.trim();
  const status = $("#datasetStatus");
  if (!file || !target || !features.length || !datasetSource || !datasetPermission) {
    status.textContent = "Provide the dataset source, permission basis, severity label, and at least one feature.";
    return;
  }
  $("#trainModel").disabled = true;
  status.textContent = "Training and evaluating on a stratified holdout split…";
  try {
    const query = new URLSearchParams({
      filename: file.name,
      target,
      dataset_source: datasetSource,
      dataset_permission: datasetPermission,
    });
    for (const feature of features) query.append("feature", feature);
    const metadata = await uploadDataset(`/api/model/train?${query}`, file);
    renderModelResults(metadata);
    status.textContent = `Model trained on ${metadata.rows_used} labeled rows. See holdout metrics and limitations below.`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    $("#trainModel").disabled = false;
  }
}

async function assessSeverity() {
  const features = {};
  for (const input of $("#assessmentFields").querySelectorAll("[data-feature]")) {
    const value = input.value.trim();
    features[input.dataset.feature] = value === "" ? null : input.type === "number" ? Number(value) : value;
  }
  const resultElement = $("#assessmentResult");
  resultElement.textContent = "Estimating severity class…";
  try {
    const result = await api("/api/model/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    });
    const probabilities = Object.entries(result.probabilities)
      .map(([label, probability]) => `${label}: ${(probability * 100).toFixed(1)}%`).join(" · ");
    resultElement.textContent = `Exploratory estimate: ${result.severity}. ${probabilities}. ${result.disclaimer} ${result.warnings.join(" ")}`;
  } catch (error) {
    resultElement.textContent = error.message;
  }
}

function riskFactors() {
  const values = state.values;
  const normalized = [
    ["Speed", clamp((values.speed - 40) / 100, 0, 1), 0.28],
    ["Hard braking", values.brake / 100, 0.18],
    ["Drowsiness", values.drowsy / 100, 0.26],
    ["Short following distance", 1 - clamp((values.gap - 5) / 75, 0, 1), 0.16],
    ["Wet / low-grip road", values.rain / 100, 0.12],
  ];
  return normalized.map(([name, value, weight]) => ({ name, value, weight, contribution: value * weight }));
}

function calculateRisk() {
  const factors = riskFactors();
  const elevatedFactors = factors.filter((factor) => factor.value > 0.6).length;
  const score = Math.round(factors.reduce((total, factor) => total + factor.contribution, 0) * 100 * (1 + 0.15 * elevatedFactors));
  return { score: clamp(score, 0, 100), factors };
}

function riskBand(score) {
  if (score < 35) return { name: "Low", color: "low", action: "Logged silently." };
  if (score < 65) return { name: "Elevated", color: "elevated", action: "Driver warning: reduce speed and stay alert." };
  if (score < 85) return { name: "High", color: "high", action: "Driver warning and HQ advisory required." };
  return { name: "Critical", color: "critical", action: "Critical risk. SOS countdown initiated." };
}

function buildSliders() {
  $("#sliders").innerHTML = sliderDefinitions.map(({ key, label, unit, min, max }) => `
    <label class="slider-control" for="slider-${key}">
      <span>${escapeHtml(label)}</span><output id="output-${key}"></output>
      <input id="slider-${key}" type="range" min="${min}" max="${max}" value="${state.values[key]}">
    </label>`).join("");
  for (const { key } of sliderDefinitions) {
    $(`#slider-${key}`).addEventListener("input", (event) => {
      state.values[key] = Number(event.target.value);
      renderRisk();
    });
  }
}

function renderRisk() {
  const { score, factors } = calculateRisk();
  const band = riskBand(score);
  const arc = $("#gaugeArc");
  arc.setAttribute("stroke-dasharray", `${(score / 100) * 282.7} 283`);
  arc.style.stroke = `var(--risk-${band.color})`;
  $("#monitorScore").textContent = score;
  $("#riskTier").textContent = band.name;
  $("#riskTier").className = `tier ${band.color}`;
  $("#recommendedAction").textContent = band.action;
  $("#riskFactors").innerHTML = factors.map((factor) => `
    <div class="factor">
      <div><span>${escapeHtml(factor.name)}</span><b>${Math.round(factor.value * 100)}%</b></div>
      <div class="factor-track"><i style="width:${factor.value * 100}%"></i></div>
    </div>`).join("");
  for (const { key, unit } of sliderDefinitions) {
    $(`#slider-${key}`).value = state.values[key];
    $(`#output-${key}`).textContent = `${state.values[key]} ${unit}`;
  }
  const warning = $("#warningBanner");
  warning.hidden = score < 35;
  warning.className = `warning-banner ${band.color}`;
  warning.textContent = `${band.name} risk · ${score}/100. ${band.action}`;
  if (score >= 85 && state.lastRisk < 85 && !state.sendingSos) startSos("Accident predicted · simulated scenario");
  state.lastRisk = score;
}

function renderSpeedChart(events) {
  const canvas = $("#riskChart");
  if (!canvas) return;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = bounds.width * ratio;
  canvas.height = bounds.height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const pad = 8;
  const width = bounds.width;
  const height = bounds.height;
  context.clearRect(0, 0, width, height);
  const points = events.filter((event) => Number.isFinite(Number(event.speed_kmh)));
  if (!points.length) return;
  context.strokeStyle = "#1e9a84";
  context.lineWidth = 2.5;
  context.beginPath();
  points.forEach((point, index) => {
    const x = pad + (index / Math.max(points.length - 1, 1)) * (width - pad * 2);
    const y = height - pad - (clamp(Number(point.speed_kmh), 0, 120) / 120) * (height - pad * 2);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function renderDashboard(sensors) {
  const gps = sensors.gps;
  const speed = gps?.speed_kmh;
  const gpsFresh = gps !== null && Date.now() - new Date(gps.observed_at).getTime() <= 15000;
  const motion = sensors.motion?.payload;
  const coordinates = gps ? `${Number(gps.latitude).toFixed(6)}, ${Number(gps.longitude).toFixed(6)}` : "Waiting for a GPS fix";
  $("#locationLabel").textContent = coordinates;
  $("#gpsStatus").textContent = gpsFresh ? "GPS · LIVE" : gps ? "GPS · STALE" : state.gpsWatchId === null ? "GPS · NOT STARTED" : "GPS · SEARCHING";
  $("#score").textContent = "—";
  $("#riskLabel").textContent = "MODEL NOT CONNECTED";
  $("#riskLabel").className = "risk-label";
  $("#confidence").textContent = "Accident prediction unavailable";
  $("#meter").style.width = "0%";
  $("#assessmentBasis").textContent = "Risk model not connected";
  $("#sampleTime").textContent = gps ? `${gpsFresh ? "GPS live" : "Last GPS fix"} · ${timeLabel(gps.observed_at)}` : "Waiting for GPS permission";
  $("#time").textContent = gps ? timeLabel(gps.observed_at) : "Waiting for GPS";
  $("#speed").textContent = speed !== null && speed !== undefined && Number.isFinite(Number(speed)) ? Number(speed).toFixed(1) : "—";
  $("#speedSource").textContent = speed == null ? "km/h · no speed fix" : `${gpsFresh ? "" : "last fix · "}${gps?.speed_method === "gps_receiver" ? "GPS receiver" : "GPS position delta"}`;
    $("#locationMeta").textContent = gps
      ? `${gps.accuracy_m == null ? "Accuracy unavailable" : `Accuracy ±${Number(gps.accuracy_m).toFixed(1)} m`} · ${gpsFresh ? "live fix" : "last known fix"}`
      : "Location permission has not been granted or no fix is available";
    $("#locationHeading").textContent = gps?.heading_deg == null ? "--" : `${Number(gps.heading_deg).toFixed(0)}°`;
    $("#alertHeading").textContent = "Live sensor status";
    $("#alertText").textContent = gps
      ? "GPS fix stored. No collision-risk model is connected, so this system cannot assess accident risk or declare the drive safe."
      : "No GPS fix yet. Motion, OBD-II, and camera sensors are not connected. Accident-risk scoring is unavailable.";
    $("#alertBadge").textContent = gpsFresh ? "GPS LIVE" : gps ? "GPS STALE" : "WAITING";
  if (motion) {
    const linearAcceleration = motion.acceleration_x_g !== undefined || motion.acceleration_y_g !== undefined || motion.acceleration_z_g !== undefined;
    const prefix = linearAcceleration ? "acceleration" : "acceleration_including_gravity";
    const x = motion[`${prefix}_x_g`];
    const y = motion[`${prefix}_y_g`];
    const z = motion[`${prefix}_z_g`];
    const available = [x, y, z].filter((value) => value !== undefined);
    $("#accel").textContent = available.length ? Math.sqrt(available.reduce((total, value) => total + value ** 2, 0)).toFixed(2) : "—";
    $("#accelSource").textContent = available.length ? linearAcceleration ? "g · linear acceleration" : "g · includes gravity" : "g · not available";
  } else {
    $("#accel").textContent = "—";
    $("#accelSource").textContent = "g · not connected";
  }
  $("#laneStability").textContent = sensors.vision?.payload?.lane_offset_m === undefined ? "—" : Number(sensors.vision.payload.lane_offset_m).toFixed(2);
  $("#laneState").textContent = sensors.vision ? "m · camera / ADAS" : "m · camera required";
  $("#driverState").textContent = sensors.vision?.payload?.drowsiness_pct === undefined ? "—" : sensors.vision.payload.drowsiness_pct >= 65 ? "Drowsy" : "Monitoring";
  $("#driverDetail").textContent = sensors.vision ? "camera / ADAS" : "camera required";
  $("#gpsSystemState").textContent = gpsFresh ? "Live GPS" : gps ? "Stale fix" : state.gpsWatchId === null ? "Not started" : "Searching";
  $("#motionSystemState").textContent = motion ? "Live device motion" : state.motionEnabled ? "Waiting for event" : "Not connected";
  $("#obdSystemState").textContent = sensors.obd ? "Device connected" : "Not connected";
  $("#visionSystemState").textContent = sensors.vision ? "Device connected" : "Not connected";
  $("#sensorStreamState").textContent = state.sensorEvents.length ? `${state.sensorEvents.length} recent events` : "Waiting for sensor data";
  $("#gpsFixState").textContent = gps ? "Live fix" : "No fix";
  $("#gpsCoordinates").textContent = gps ? coordinates : "—";
  $("#gpsAccuracy").textContent = gps?.accuracy_m == null ? "—" : `${Number(gps.accuracy_m).toFixed(1)} m`;
  $("#gpsSpeed").textContent = speed !== null && speed !== undefined && Number.isFinite(Number(speed)) ? `${Number(speed).toFixed(2)} km/h` : "—";
  $("#gpsSpeedMethod").textContent = gps?.speed_method || "Unavailable";
  $("#gpsHeading").textContent = gps?.heading_deg == null ? "—" : `${Number(gps.heading_deg).toFixed(1)}°`;
  $("#gpsObservedAt").textContent = gps ? dateLabel(gps.observed_at) : "—";
  $("#motionFixState").textContent = motion ? "Live event" : "Not connected";
  const accelerationPrefix = motion && (motion.acceleration_x_g !== undefined || motion.acceleration_y_g !== undefined || motion.acceleration_z_g !== undefined) ? "acceleration" : "acceleration_including_gravity";
  $("#motionAcceleration").textContent = motion ? ["x", "y", "z"].map((axis) => formatReading(motion[`${accelerationPrefix}_${axis}_g`])).join(" / ") : "—";
  $("#motionRotation").textContent = motion ? [motion.rotation_rate_x_dps, motion.rotation_rate_y_dps, motion.rotation_rate_z_dps].map(formatReading).join(" / ") : "—";
  $("#motionObservedAt").textContent = sensors.motion ? dateLabel(sensors.motion.observed_at) : "—";
  $("#sensorRecordCount").textContent = `${state.sensorEvents.length} recent sensor events shown`;
  $("#sensorRowsEmpty").hidden = state.sensorEvents.length > 0;
  $("#sensorRows").innerHTML = [...state.sensorEvents].reverse().map((event) => `
    <tr><td>${escapeHtml(timeLabel(event.observed_at))}</td><td>${escapeHtml(event.source)}</td><td>${event.latitude == null ? "—" : `${Number(event.latitude).toFixed(5)}, ${Number(event.longitude).toFixed(5)}`}</td><td>${event.speed_kmh == null ? "—" : Number(event.speed_kmh).toFixed(1)}</td><td>${event.accuracy_m == null ? "—" : `${Number(event.accuracy_m).toFixed(1)} m`}</td><td><code>${escapeHtml(JSON.stringify(event.payload))}</code></td></tr>`).join("");
  renderSpeedChart(state.sensorEvents.filter((event) => event.source === "gps"));
}

function formatReading(value) {
  return value === undefined || value === null ? "—" : Number(value).toFixed(3);
}

async function refreshSensors() {
  try {
    const [sensors, events, health] = await Promise.all([
      api("/api/sensors/latest"),
      api("/api/sensors/history?limit=100"),
      api("/api/health"),
    ]);
    state.sensors = sensors;
    state.sensorEvents = events;
    renderDashboard(sensors);
    $("#connectionState").textContent = "Network online";
    $("#connectionDetail").textContent = "SQLite connected";
    $("#sensorSummary").textContent = `${health.sensor_readings} recorded sensor events`;
    $("#feedMode").textContent = state.gpsWatchId === null ? "GPS NOT STARTED" : sensors.gps ? "LIVE GPS" : "GPS SEARCHING";
  } catch (error) {
    $("#connectionState").textContent = "Network offline";
    $("#connectionDetail").textContent = "Start server.py";
    $("#sensorSummary").textContent = "Service unavailable";
    $("#sensorMessage").textContent = error.message;
  }
}

async function startGps() {
  if (!window.isSecureContext) {
    $("#sensorMessage").textContent = "Browser GPS requires HTTPS or localhost. Open the app securely, or send GPS fixes to /api/sensors/gps from a device client.";
    return;
  }
  if (!navigator.geolocation) {
    $("#sensorMessage").textContent = "This browser does not expose a GPS/location provider.";
    return;
  }
  if (state.gpsWatchId !== null) {
    navigator.geolocation.clearWatch(state.gpsWatchId);
    state.gpsWatchId = null;
    $("#gpsToggle").textContent = "Enable GPS tracking";
    $("#sensorMessage").textContent = "GPS tracking stopped. Existing readings remain saved.";
    await refreshSensors();
    return;
  }
  $("#sensorMessage").textContent = "Requesting location permission and waiting for a GPS fix…";
  state.gpsWatchId = navigator.geolocation.watchPosition(async (position) => {
    const coords = position.coords;
    const payload = {
      observed_at: new Date(position.timestamp).toISOString(),
      latitude: coords.latitude,
      longitude: coords.longitude,
      accuracy_m: coords.accuracy,
    };
    if (coords.altitude !== null) payload.altitude_m = coords.altitude;
    if (coords.heading !== null) payload.heading_deg = coords.heading;
    if (coords.speed !== null && coords.speed >= 0) payload.speed_mps = coords.speed;
    try {
      await api("/api/sensors/gps", { method: "POST", body: JSON.stringify(payload) });
      $("#gpsToggle").textContent = "Stop GPS tracking";
      $("#sensorMessage").textContent = `GPS fix stored · accuracy ${coords.accuracy.toFixed(1)} m`;
      await refreshSensors();
    } catch (error) {
      $("#sensorMessage").textContent = `Could not store GPS fix: ${error.message}`;
    }
  }, (error) => {
    if (state.gpsWatchId !== null) navigator.geolocation.clearWatch(state.gpsWatchId);
    state.gpsWatchId = null;
    $("#gpsToggle").textContent = "Enable GPS tracking";
    const messages = {
      1: "Location permission was denied. Allow location access in browser settings and retry.",
      2: "The device could not determine its location. Check its GPS/location service.",
      3: "GPS request timed out. Try again outdoors with a clear sky view.",
    };
    $("#sensorMessage").textContent = messages[error.code] || error.message;
    $("#gpsStatus").textContent = "GPS · UNAVAILABLE";
  }, { enableHighAccuracy: true, maximumAge: 1000, timeout: 20000 });
  $("#gpsToggle").textContent = "Stop GPS tracking";
  $("#gpsStatus").textContent = "GPS · SEARCHING";
  $("#gpsSystemState").textContent = "Searching";
  $("#feedMode").textContent = "GPS SEARCHING";
}

async function startMotion() {
  if (!("DeviceMotionEvent" in window)) {
    $("#sensorMessage").textContent = "This device/browser does not provide a motion sensor.";
    return;
  }
  try {
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      const permission = await DeviceMotionEvent.requestPermission();
      if (permission !== "granted") throw new Error("Motion sensor permission was not granted.");
    }
    if (!state.motionEnabled) {
      window.addEventListener("devicemotion", onMotion);
      state.motionEnabled = true;
    }
    $("#motionToggle").textContent = "Stop motion sensor";
    $("#sensorMessage").textContent = "Motion sensor enabled. Readings are sampled at most once per second.";
  } catch (error) {
    $("#sensorMessage").textContent = error.message || "Motion sensor permission failed.";
  }
}

async function onMotion(event) {
  const now = Date.now();
  if (now - state.lastMotionSent < 1000) return;
  state.lastMotionSent = now;
  const acceleration = event.acceleration;
  const gravity = event.accelerationIncludingGravity;
  const sourceAcceleration = acceleration || gravity;
  if (!sourceAcceleration && !event.rotationRate) return;
  const payload = { observed_at: new Date().toISOString() };
  if (sourceAcceleration) {
    for (const [axis, value] of [["x", sourceAcceleration.x], ["y", sourceAcceleration.y], ["z", sourceAcceleration.z]]) {
      if (value !== null && value !== undefined) {
        const field = acceleration ? `acceleration_${axis}_g` : `acceleration_including_gravity_${axis}_g`;
        payload[field] = value / 9.80665;
      }
    }
  }
  if (event.rotationRate) {
    for (const [axis, value] of [["x", event.rotationRate.beta], ["y", event.rotationRate.gamma], ["z", event.rotationRate.alpha]]) {
      if (value !== null && value !== undefined) payload[`rotation_rate_${axis}_dps`] = value;
    }
  }
  if (Object.keys(payload).length === 1) return;
  try {
    await api("/api/sensors/motion", { method: "POST", body: JSON.stringify(payload) });
    await refreshSensors();
  } catch (error) {
    $("#sensorMessage").textContent = `Could not store motion reading: ${error.message}`;
  }
}

async function loadAlerts() {
  state.alerts = await api("/api/alerts");
  const unacknowledged = state.alerts.filter((alert) => !alert.acknowledged).length;
  $("#navBadge").textContent = unacknowledged;
  $("#navBadge").hidden = unacknowledged === 0;
  $("#alertCount").textContent = `${state.alerts.length} events`;
  $("#alertsEmpty").hidden = state.alerts.length > 0;
  $("#alertRows").innerHTML = state.alerts.map((alert) => `
    <tr><td>${escapeHtml(timeLabel(alert.created_at))}</td><td><span class="severity ${escapeHtml(alert.severity)}">${escapeHtml(alert.severity)}</span></td><td>${escapeHtml(alert.category)}<small class="source-note">${escapeHtml(sourceLabel(alert.source))}</small></td><td>${escapeHtml(alert.message)}</td><td>${alert.acknowledged ? "Acknowledged" : "Open"}</td><td>${alert.acknowledged ? "" : `<button class="table-action" data-ack="${alert.id}">Acknowledge</button>`}</td></tr>`).join("");
  $$('[data-ack]').forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/alerts/${button.dataset.ack}/ack`, { method: "POST", body: "{}" });
    await loadAlerts();
  }));
}

async function loadIncidents() {
  state.incidents = await api("/api/incidents");
  const open = state.incidents.filter((incident) => incident.status < 3).length;
  $("#activeIncidentCount").textContent = `${open} open incidents`;
  $("#incidentsEmpty").hidden = state.incidents.length > 0;
  const newCount = state.incidents.filter((incident) => incident.status === 0).length;
  const feedbackCount = state.incidents.filter((incident) => incident.confirmed === 0).length;
  const resolved = state.incidents.filter((incident) => incident.status === 3).length;
  $("#hqKpis").innerHTML = [["Active incidents", open], ["New", newCount], ["Resolved", resolved], ["False-alarm feedback", feedbackCount]]
    .map(([label, value]) => `<div class="metric"><span>${label}</span><b>${value}</b><small>persisted records</small></div>`).join("");
  $("#incidentList").innerHTML = state.incidents.map((incident) => `
    <article class="incident-item"><div class="incident-heading"><b>INC-${String(incident.id).padStart(4, "0")} · ${escapeHtml(incident.incident_type)}</b><span class="severity ${escapeHtml(incident.severity)}">${escapeHtml(incident.severity)}</span></div><small class="source-note">Origin: ${escapeHtml(sourceLabel(incident.source))}</small>
      <p>${escapeHtml(dateLabel(incident.created_at))} · ${escapeHtml(incident.location)} · risk ${incident.risk}</p>
      <div class="incident-state">${["New", "Acknowledged", "Dispatch request logged locally", "Resolved locally"][incident.status]}${incident.confirmed === null ? " · outcome not labeled" : incident.confirmed ? " · confirmed" : " · false alarm"}</div>
      <div class="incident-actions">${incident.status < 3 ? `<button class="table-action" data-status="${incident.id}" data-next="${incident.status + 1}">${["Acknowledge", "Log dispatch request", "Resolve locally"][incident.status]}</button>` : ""}${incident.confirmed === null ? `<button class="table-action" data-feedback="${incident.id}" data-confirmed="true">Confirm</button><button class="table-action" data-feedback="${incident.id}" data-confirmed="false">False alarm</button>` : ""}</div></article>`).join("");
  $("#incidentPins").innerHTML = state.incidents.map((incident) => `<div class="incident-location"><b>INC-${String(incident.id).padStart(4, "0")}</b><span>${escapeHtml(incident.location)}</span><small>${escapeHtml(sourceLabel(incident.source))}</small></div>`).join("");
  $$('[data-status]').forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/incidents/${button.dataset.status}/status`, { method: "POST", body: JSON.stringify({ status: Number(button.dataset.next) }) });
    await loadIncidents();
  }));
  $$('[data-feedback]').forEach((button) => button.addEventListener("click", async () => {
    await api(`/api/incidents/${button.dataset.feedback}/feedback`, { method: "POST", body: JSON.stringify({ confirmed: button.dataset.confirmed === "true" }) });
    await loadIncidents();
  }));
}

async function loadTrips() {
  state.trips = await api("/api/trips");
  state.activeTrip = state.trips.some((trip) => trip.status === "active");
  $("#startTrip").disabled = state.activeTrip;
  $("#stopTrip").disabled = !state.activeTrip;
  $("#tripCount").textContent = `${state.trips.length} trips`;
  $("#tripsEmpty").hidden = state.trips.length > 0;
  $("#tripRows").innerHTML = state.trips.map((trip) => `<tr><td>TR-${String(trip.id).padStart(4, "0")}</td><td>${escapeHtml(dateLabel(trip.started_at))}</td><td>${escapeHtml(dateLabel(trip.ended_at))}</td><td>${Number(trip.distance_km).toFixed(3)} km</td><td>${trip.gps_fixes}</td><td>${escapeHtml(trip.status)}</td></tr>`).join("");
}

async function loadSettings() {
  const settings = await api("/api/settings");
  state.threshold = settings.risk_threshold;
  state.interval = settings.sampling_interval;
  $("#thresholdInput").value = state.threshold;
  $("#intervalInput").value = state.interval;
}

function resetRefresh() {
  clearInterval(state.refreshTimer);
  state.refreshTimer = setInterval(refreshSensors, state.interval * 1000);
}

function showPage(name) {
  if (!viewCopy[name]) return;
  $$('[data-page]').forEach((page) => page.classList.toggle("on", page.dataset.page === name));
  $$('[data-view]').forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  $("#pageTitle").textContent = viewCopy[name][0];
  $("#pageDescription").textContent = viewCopy[name][1];
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "monitor") renderRisk();
  if (name === "hq") loadIncidents();
  if (name === "sos") loadAlerts();
  if (name === "trips") loadTrips();
  if (name === "model") loadModelStatus().catch((error) => { $("#datasetStatus").textContent = error.message; });
  if (name === "sensors" || name === "dashboard") requestAnimationFrame(() => renderDashboard(state.sensors));
}

function startSos(type) {
  if (state.sendingSos) return;
  state.sendingSos = true;
  state.sosSubmitting = false;
  let remaining = 10;
  $("#sosModal").hidden = false;
  // User-friendly title: map internal type strings to readable headings
  const titleMap = {
    "Manual SOS": "Emergency SOS",
    "Accident occurred · simulated scenario": "Simulated Crash Impact",
    "Accident predicted · simulated scenario": "High-Risk Alert (Simulated)",
  };
  $("#modalTitle").textContent = titleMap[type] || type;
  $("#modalMessage").textContent = type.toLowerCase().includes("simulated")
    ? "Recording this simulated alert in"
    : "Sending emergency alert in";
  $("#countdown").hidden = false;
  $("#countdown").textContent = remaining;
  $("#sendNow").hidden = false;
  $("#cancelSos").hidden = false;
  $("#closeModal").hidden = true;
  $("#modalStatus").textContent = "This local demo does not contact emergency services.";
  $("#cancelSos").focus();
  clearInterval(state.countdownTimer);
  state.countdownTimer = setInterval(() => {
    remaining -= 1;
    $("#countdown").textContent = remaining;
    if (remaining <= 0) sendSos(type);
  }, 1000);
}

async function sendSos(type) {
  if (!state.sendingSos || state.sosSubmitting) return;
  state.sosSubmitting = true;
  clearInterval(state.countdownTimer);
  $("#sendNow").disabled = true;
  $("#cancelSos").disabled = true;
  $("#modalMessage").textContent = "Recording incident in local network";
  try {
    const { score } = calculateRisk();
    await api("/api/alerts/sos", { method: "POST", body: JSON.stringify({ type, risk: type.startsWith("Accident occurred") ? 100 : score, location: state.sensors.gps ? `${state.sensors.gps.latitude}, ${state.sensors.gps.longitude}` : "GPS unavailable", message: `${type} · local demo only; no emergency services contacted` }) });
    $("#modalStatus").textContent = "Incident persisted to SQLite and added to the HQ queue. No external service was contacted.";
    $("#closeModal").hidden = false;
    await Promise.all([loadAlerts(), loadIncidents()]);
  } catch (error) {
    $("#modalStatus").textContent = error.message;
    $("#cancelSos").hidden = true;
    $("#closeModal").hidden = false;
  } finally {
    $("#countdown").hidden = true;
    $("#sendNow").hidden = true;
  }
}

async function cancelSos() {
  if (!state.sendingSos || state.sosSubmitting) return;
  state.sosSubmitting = true;
  clearInterval(state.countdownTimer);
  $("#cancelSos").disabled = true;
  $("#sendNow").disabled = true;
  try {
    await api("/api/alerts/sos/cancel", { method: "POST", body: JSON.stringify({ message: "SOS countdown canceled; no dispatch was made." }) });
    await loadAlerts();
  } catch (error) {
    $("#modalStatus").textContent = error.message;
  }
  closeSosModal();
}

function closeSosModal() {
  clearInterval(state.countdownTimer);
  state.sendingSos = false;
  state.sosSubmitting = false;
  $("#sosModal").hidden = true;
  $("#sendNow").disabled = false;
  $("#cancelSos").disabled = false;
}

function bindEvents() {
  $$('[data-view]').forEach((button) => button.addEventListener("click", () => showPage(button.dataset.view)));
  $("#featureColumns").addEventListener("change", () => {
    $("#trainModel").disabled = $("#featureColumns").querySelectorAll("input:checked").length === 0;
  });
  $("#inspectDataset").addEventListener("click", inspectDataset);
  $("#targetColumn").addEventListener("change", renderFeatureOptions);
  $("#trainModel").addEventListener("click", trainSeverityModel);
  $("#assessSeverity").addEventListener("click", assessSeverity);
  $("#datasetFile").addEventListener("change", () => {
    state.datasetFile = null;
    state.datasetColumns = [];
    $("#datasetSource").value = "";
    $("#datasetPermission").value = "";
    $("#datasetControls").hidden = true;
    $("#datasetStatus").textContent = "File changed. Inspect it before training.";
  });
  $$('[data-scenario]').forEach((button) => button.addEventListener("click", () => {
    state.values = { ...scenarioValues[button.dataset.scenario] };
    renderRisk();
  }));
  $("#gpsToggle").addEventListener("click", startGps);
  $("#motionToggle").addEventListener("click", async () => {
    if (state.motionEnabled) {
      window.removeEventListener("devicemotion", onMotion);
      state.motionEnabled = false;
      $("#motionToggle").textContent = "Enable motion sensor";
      $("#sensorMessage").textContent = "Motion sensor stopped. Stored readings remain available.";
      await refreshSensors();
    } else {
      await startMotion();
    }
  });
  $("#autoDrive").addEventListener("click", (event) => {
    if (state.autoTimer) {
      clearInterval(state.autoTimer);
      state.autoTimer = null;
      event.currentTarget.textContent = "Auto-drive: off";
    } else {
      event.currentTarget.textContent = "Auto-drive: on";
      state.autoTimer = setInterval(() => {
        for (const { key, min, max } of sliderDefinitions) state.values[key] = Math.round(clamp(state.values[key] + (Math.random() - 0.45) * (max - min) * 0.08, min, max));
        renderRisk();
      }, 1000);
    }
  });
  $("#crashButton").addEventListener("click", () => startSos("Accident occurred · simulated scenario"));
  $("#sos").addEventListener("click", () => startSos("Manual SOS"));
  $("#sosButton").addEventListener("click", () => startSos("Manual SOS"));
  $("#sendNow").addEventListener("click", () => sendSos($("#modalTitle").textContent));
  $("#cancelSos").addEventListener("click", cancelSos);
  $("#closeModal").addEventListener("click", closeSosModal);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#sosModal").hidden && !$("#cancelSos").hidden) cancelSos();
  });
  $("#startTrip").addEventListener("click", async () => {
    const feedback = $("#tripFeedback");
    try {
      await api("/api/trips/start", { method: "POST", body: "{}" });
      if (feedback) feedback.textContent = "";
      await loadTrips();
    } catch (error) {
      if (feedback) { feedback.textContent = error.message; feedback.style.color = "var(--danger)"; }
    }
  });
  $("#stopTrip").addEventListener("click", async () => {
    const feedback = $("#tripFeedback");
    try {
      await api("/api/trips/stop", { method: "POST", body: "{}" });
      if (feedback) feedback.textContent = "";
      await loadTrips();
    } catch (error) {
      if (feedback) { feedback.textContent = error.message; feedback.style.color = "var(--danger)"; }
    }
  });
  $("#settingsForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const feedback = $("#settingsFeedback");
    try {
      const settings = await api("/api/settings", { method: "POST", body: JSON.stringify({ risk_threshold: Number($("#thresholdInput").value), sampling_interval: Number($("#intervalInput").value) }) });
      state.threshold = settings.risk_threshold;
      state.interval = settings.sampling_interval;
      resetRefresh();
      feedback.textContent = "Settings saved to SQLite.";
    } catch (error) { feedback.textContent = error.message; }
  });
  $("#themeToggle").addEventListener("click", () => {
    const isDark = document.documentElement.dataset.theme === "dark";
    const next = isDark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("safedrive-theme", next);
    $("#themeToggle").textContent = next === "dark" ? "Light mode" : "Dark mode";
  });
  window.addEventListener("resize", () => renderSpeedChart(state.sensorEvents));
}

function applyTheme() {
  const saved = localStorage.getItem("safedrive-theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  const toggle = $("#themeToggle");
  if (toggle) toggle.textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

async function initialize() {
  applyTheme();
  buildSliders();
  bindEvents();
  renderRisk();
  try {
    await loadSettings();
    await refreshSensors();
  } catch (error) {
    $("#connectionState").textContent = "Network offline";
    $("#connectionDetail").textContent = "Start server.py";
    $("#sensorMessage").textContent = error.message;
  }
  resetRefresh();
}

initialize();
