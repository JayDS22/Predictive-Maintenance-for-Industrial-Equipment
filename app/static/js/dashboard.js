const FLEET_URL = "/api/fleet";
const UNIT_URL = (id) => `/api/unit/${id}`;

const tbody = document.querySelector("#fleet-table tbody");
const refreshBtn = document.getElementById("refresh-btn");
const updatedEl = document.getElementById("last-updated");

const layoutBase = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#98a2c1", family: "Inter" },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: { gridcolor: "rgba(255,255,255,0.06)" },
  yaxis: { gridcolor: "rgba(255,255,255,0.06)" },
};

async function loadFleet() {
  refreshBtn.disabled = true;
  refreshBtn.textContent = "Refreshing…";
  try {
    const res = await fetch(FLEET_URL);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const { units, summary } = await res.json();
    renderSummary(summary);
    renderTable(units);
    if (units.length) loadUnit(units[0].unit_id);
    updatedEl.textContent = `Last updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:#fda4af; padding:18px">
      ${err.message} · Train models first: <code>python -m src.train</code>
    </td></tr>`;
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = "Refresh";
  }
}

function renderSummary(summary) {
  document.getElementById("stat-total").textContent = summary.n_units;
  document.getElementById("stat-critical").textContent = summary.n_critical;
  document.getElementById("stat-watch").textContent = summary.n_watch;
  document.getElementById("stat-healthy").textContent = summary.n_healthy;
}

function renderTable(units) {
  tbody.innerHTML = units
    .map((u) => {
      const risk = (u.failure_risk * 100).toFixed(1);
      const rul = u.current_rul.toFixed(1);
      return `
        <tr data-unit="${u.unit_id}">
          <td>#${u.unit_id}</td>
          <td>${u.latest_cycle}</td>
          <td>${rul}</td>
          <td>${risk}%</td>
          <td><span class="risk-pill ${u.risk_band}">${u.risk_band}</span></td>
        </tr>`;
    })
    .join("");
  tbody.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      tbody.querySelectorAll("tr").forEach((r) => r.classList.remove("active"));
      row.classList.add("active");
      loadUnit(Number(row.dataset.unit));
    });
  });
  if (tbody.firstElementChild) tbody.firstElementChild.classList.add("active");
}

async function loadUnit(unitId) {
  document.getElementById("unit-title").textContent = `· unit #${unitId}`;
  try {
    const res = await fetch(UNIT_URL(unitId));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderRulChart(data);
    renderRiskChart(data);
    renderSensorChart(data);
    renderDecompChart(data);
  } catch (err) {
    console.error(err);
  }
}

function renderRulChart(d) {
  const traces = [];
  if (d.actual_rul) {
    traces.push({
      x: d.cycles, y: d.actual_rul,
      mode: "lines", name: "Actual RUL",
      line: { color: "rgba(152, 162, 193, 0.65)", width: 2, dash: "dot" },
    });
  }
  traces.push({
    x: d.cycles, y: d.predicted_rul,
    mode: "lines", name: "Predicted RUL",
    line: { color: "#6366f1", width: 3 },
  });
  const lastCycle = d.cycles[d.cycles.length - 1];
  const forecastX = d.forecast_rul_next_20.map((_, i) => lastCycle + i + 1);
  traces.push({
    x: forecastX, y: d.forecast_rul_next_20,
    mode: "lines", name: "ARIMA forecast",
    line: { color: "#22d3ee", width: 3, dash: "dash" },
  });
  Plotly.newPlot("rul-chart", traces, {
    ...layoutBase,
    height: 300,
    xaxis: { ...layoutBase.xaxis, title: "Cycle" },
    yaxis: { ...layoutBase.yaxis, title: "RUL (cycles)" },
    legend: { orientation: "h", y: -0.2 },
  }, { displayModeBar: false, responsive: true });
}

function renderRiskChart(d) {
  Plotly.newPlot("risk-chart", [
    {
      x: d.cycles, y: d.failure_risk.map((r) => r * 100),
      mode: "lines", fill: "tozeroy",
      line: { color: "#f43f5e", width: 2, shape: "spline" },
      fillcolor: "rgba(244, 63, 94, 0.18)",
      name: "Failure risk %",
    },
  ], {
    ...layoutBase,
    height: 300,
    xaxis: { ...layoutBase.xaxis, title: "Cycle" },
    yaxis: { ...layoutBase.yaxis, title: "Risk (%)", range: [0, 100] },
    shapes: [{
      type: "line", xref: "paper", x0: 0, x1: 1,
      y0: 60, y1: 60, line: { color: "#fbbf24", dash: "dot", width: 1 },
    }],
  }, { displayModeBar: false, responsive: true });
}

function renderSensorChart(d) {
  const traces = Object.entries(d.sensor_traces).map(([name, values]) => ({
    x: d.cycles,
    y: values,
    mode: "lines",
    name,
    line: { width: 2 },
  }));
  Plotly.newPlot("sensor-chart", traces, {
    ...layoutBase,
    height: 300,
    xaxis: { ...layoutBase.xaxis, title: "Cycle" },
    yaxis: { ...layoutBase.yaxis, title: "Sensor reading" },
    legend: { orientation: "h", y: -0.2 },
  }, { displayModeBar: false, responsive: true });
}

function renderDecompChart(d) {
  const decomp = d.decomposition_sensor_3;
  const x = d.cycles;
  Plotly.newPlot("decomp-chart", [
    { x, y: decomp.observed, name: "Observed", line: { color: "#e8ecf7", width: 1.5 } },
    { x, y: decomp.trend,    name: "Trend",    line: { color: "#6366f1", width: 2 } },
    { x, y: decomp.seasonal, name: "Seasonal", line: { color: "#22d3ee", width: 1.5 } },
    { x, y: decomp.resid,    name: "Residual", line: { color: "#fbbf24", width: 1 } },
  ], {
    ...layoutBase,
    height: 300,
    xaxis: { ...layoutBase.xaxis, title: "Cycle" },
    yaxis: { ...layoutBase.yaxis, title: "Sensor_3" },
    legend: { orientation: "h", y: -0.2 },
  }, { displayModeBar: false, responsive: true });
}

refreshBtn.addEventListener("click", loadFleet);
loadFleet();
