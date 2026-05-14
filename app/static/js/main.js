// Three operating regimes mapped to representative C-MAPSS sensor values.
const SENSORS = [
  "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
  "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
  "sensor_15", "sensor_17", "sensor_20", "sensor_21",
];

const PRESETS = {
  healthy: {
    sensor_2: 641.8,  sensor_3: 1589.0, sensor_4: 1399.0, sensor_7: 554.8,
    sensor_8: 2388.0, sensor_9: 9050.0, sensor_11: 47.5,  sensor_12: 521.9,
    sensor_13: 2388.0, sensor_14: 8140.0, sensor_15: 8.4, sensor_17: 392.0,
    sensor_20: 39.0,  sensor_21: 23.4,
  },
  degrading: {
    sensor_2: 643.2,  sensor_3: 1601.0, sensor_4: 1410.0, sensor_7: 553.4,
    sensor_8: 2392.0, sensor_9: 9080.0, sensor_11: 47.7,  sensor_12: 520.4,
    sensor_13: 2392.0, sensor_14: 8160.0, sensor_15: 8.35, sensor_17: 395.0,
    sensor_20: 38.5,  sensor_21: 22.9,
  },
  critical: {
    sensor_2: 645.5,  sensor_3: 1618.0, sensor_4: 1424.0, sensor_7: 551.0,
    sensor_8: 2399.0, sensor_9: 9120.0, sensor_11: 47.9,  sensor_12: 517.8,
    sensor_13: 2399.0, sensor_14: 8190.0, sensor_15: 8.25, sensor_17: 400.5,
    sensor_20: 37.7,  sensor_21: 22.0,
  },
};

const grid = document.getElementById("sensor-grid");
const presetEl = document.getElementById("preset");
const form = document.getElementById("predict-form");
const resultEl = document.getElementById("predict-result");

function buildInputs(preset) {
  grid.innerHTML = "";
  for (const s of SENSORS) {
    const wrap = document.createElement("label");
    wrap.innerHTML = `<span>${s}</span><input type="number" step="0.01" name="${s}" value="${preset[s]}" />`;
    grid.appendChild(wrap);
  }
}

function applyPreset(name) {
  const preset = PRESETS[name];
  for (const s of SENSORS) {
    const el = grid.querySelector(`input[name="${s}"]`);
    if (el) el.value = preset[s];
  }
}

buildInputs(PRESETS[presetEl.value]);
presetEl.addEventListener("change", (e) => applyPreset(e.target.value));

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const cycle = Number(form.cycle.value || 100);
  const sensors = {};
  for (const s of SENSORS) sensors[s] = Number(form[s].value);

  resultEl.innerHTML = `<h3>Prediction result</h3><p class="muted">Running ensemble + ARIMA forecast…</p>`;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cycle, sensors }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    resultEl.innerHTML = `<h3>Prediction result</h3>
      <p class="risk-pill critical">Error</p>
      <p class="muted">${err.message}</p>
      <p class="muted small">Make sure models are trained: <code>python -m src.train</code></p>`;
  }
});

function renderResult(data) {
  const band = (data.risk_band || "healthy").toLowerCase();
  const rul = Number(data.predicted_rul).toFixed(1);
  const risk = (Number(data.failure_risk) * 100).toFixed(1);
  resultEl.innerHTML = `
    <h3>Prediction result</h3>
    <div class="risk-pill ${band}">
      <span class="dot"></span>
      ${band.toUpperCase()}
    </div>
    <div class="result-meta">
      <div><div class="lbl">Predicted RUL</div><div class="val">${rul}<span class="muted small"> cycles</span></div></div>
      <div><div class="lbl">Failure risk</div><div class="val">${risk}<span class="muted small">%</span></div></div>
    </div>
    <div>
      <h4 style="margin-bottom:8px">Recommended actions</h4>
      <ul class="action-list">${data.actions.map((a) => `<li>${a}</li>`).join("")}</ul>
    </div>
    <div class="forecast-chart" id="forecast-chart"></div>
  `;
  const forecast = data.forecast_rul_next_20 || [];
  const x = forecast.map((_, i) => `+${i + 1}`);
  Plotly.newPlot(
    "forecast-chart",
    [
      {
        x,
        y: forecast,
        mode: "lines+markers",
        line: { color: "#22d3ee", width: 3, shape: "spline" },
        marker: { size: 6, color: "#6366f1" },
        fill: "tozeroy",
        fillcolor: "rgba(99, 102, 241, 0.15)",
        name: "RUL forecast",
      },
    ],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#98a2c1", family: "Inter" },
      margin: { l: 40, r: 16, t: 30, b: 40 },
      height: 220,
      xaxis: { title: "Cycles ahead", gridcolor: "rgba(255,255,255,0.06)" },
      yaxis: { title: "RUL", gridcolor: "rgba(255,255,255,0.06)" },
      title: { text: "ARIMA forecast, next 20 cycles", font: { size: 13 } },
    },
    { displayModeBar: false, responsive: true }
  );
}
