/* Vanilla replay UI. Pure helpers are exported for tests without a browser. */
(function (root) {
  "use strict";

  const percentage = value => {
    const number = Number(value);
    return value == null || !Number.isFinite(number) ? "Unavailable" : `${(number * 100).toFixed(1)}%`;
  };
  const number = (value, digits = 1, unit = "") => {
    const parsed = Number(value);
    return value == null || !Number.isFinite(parsed) ? "Unavailable" : `${parsed.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
  };
  // Replay frames are 15 minutes apart; a longer jump between consecutive plotted
  // points is a recording gap and must not be bridged by a line.
  const GAP_MS = 20 * 60 * 1000;
  // Number(null) is 0, so missing values must be mapped to null before any numeric test.
  const finiteOrNull = value => {
    if (value == null || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const segmentSeries = (rows, key, gapMs = GAP_MS) => {
    const segments = [];
    let current = null;
    let previous = null;
    (Array.isArray(rows) ? rows : []).forEach(row => {
      const value = finiteOrNull(row && row[key]);
      const time = new Date(row && row.timestamp).getTime();
      if (value === null || !Number.isFinite(time)) { current = null; previous = null; return; }
      if (current === null || time - previous > gapMs) { current = []; segments.push(current); }
      current.push({ timestamp: row.timestamp, time, value });
      previous = time;
    });
    return segments;
  };
  const resetReplayState = () => ({ elapsed: 0, loading: false, error: null, response: null });
  const selectReplayState = (participants, index) => {
    const list = Array.isArray(participants) ? participants : [];
    const selected = list.length ? list[Math.max(0, Math.min(Number(index) || 0, list.length - 1))] : null;
    return { participant: selected, ...resetReplayState() };
  };

  function init() {
    const byId = id => document.getElementById(id);
    const picker = byId("patient");
    const source = byId("source");
    const slider = byId("time");
    const label = byId("timeLabel");
    const app = byId("app");
    const list = byId("worklist");
    let data = null;
    let participant = null;
    let state = resetReplayState();
    let requestVersion = 0;

    const esc = value => {
      const element = document.createElement("span");
      element.textContent = value == null ? "Unavailable" : String(value);
      return element.innerHTML;
    };
    const date = value => value ? new Date(value).toLocaleString() : "Unavailable";
    const query = (response, elapsed) => `/api/replay?participant=${encodeURIComponent(participant.id)}&timestamp=${encodeURIComponent(response)}&source=${encodeURIComponent(source.value)}&elapsed=${elapsed}`;

    function drawTimeline(response) {
      const canvas = byId("timeline");
      if (!canvas) return;
      const box = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(320, box.width || 640);
      const height = 405;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const rows = response.timeline || [];
      const left = 58, right = 16, top = 18, panelHeight = 62, gap = 14;
      ctx.clearRect(0, 0, width, height);
      if (!rows.length) {
        ctx.fillStyle = "#5B6B76";
        ctx.fillText("No synchronized feature snapshots are available.", left, 40);
        return;
      }
      const start = new Date(rows[0].timestamp).getTime();
      const end = new Date(response.target_timestamp).getTime();
      const x = stamp => left + ((new Date(stamp).getTime() - start) / Math.max(1, end - start)) * (width - left - right);
      const series = [
        ["Glucose", "glucose", "mg/dL", "#0F8B8D", 40, 260],
        ["Heart rate", "heart_rate", "bpm", "#5369A8", 40, 160],
        ["Activity", "activity", "METs", "#D68A2D", 0, 12],
        ["Carbohydrate summary", "carbs_2h", "g", "#B64A52", 0, Math.max(40, ...rows.map(r => Number(r.carbs_2h) || 0))],
        ["Meal-event summary", "meal_events_2h", "events", "#855B9A", 0, Math.max(1, ...rows.map(r => Number(r.meal_events_2h) || 0))],
      ];
      ctx.font = "11px var(--font-mono)";
      series.forEach(([title, key, unit, color, lo, hi], index) => {
        const y0 = top + index * (panelHeight + gap);
        ctx.strokeStyle = "#D5DEE2";
        ctx.beginPath(); ctx.moveTo(left, y0 + panelHeight); ctx.lineTo(width - right, y0 + panelHeight); ctx.stroke();
        ctx.fillStyle = color; ctx.fillText(`${title} (${unit})`, 4, y0 + 13);
        const toY = value => y0 + panelHeight - ((value - lo) / Math.max(1, hi - lo)) * (panelHeight - 18) - 3;
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
        segmentSeries(rows, key).forEach(segment => {
          if (segment.length === 1) {
            ctx.beginPath(); ctx.arc(x(segment[0].timestamp), toY(segment[0].value), 2, 0, 2 * Math.PI); ctx.fill();
            return;
          }
          ctx.beginPath();
          segment.forEach((point, index) => {
            if (index === 0) ctx.moveTo(x(point.timestamp), toY(point.value)); else ctx.lineTo(x(point.timestamp), toY(point.value));
          });
          ctx.stroke();
        });
      });
      const forecastX = x(response.forecast_timestamp);
      const targetX = x(response.target_timestamp);
      [[forecastX, "#0F8B8D", "forecast t"], [targetX, "#F2A33A", "target t+120"]].forEach(([px, color, text]) => {
        ctx.strokeStyle = color; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, height - 8); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = color; ctx.fillText(text, Math.min(width - 80, Math.max(left, px + 4)), height - 2);
      });
    }

    function renderWorklist(response) {
      const items = response.worklist || [];
      list.className = "card worklist";
      list.innerHTML = `<h2>Eligible retrospective review list · not triage</h2><table><thead><tr><th>Participant</th><th>Exact-time score</th><th>Current glucose</th><th>Timestamp match</th></tr></thead><tbody>${items.map(item => `<tr class="${item.id === participant.id ? "selected" : ""}"><td>Participant ${esc(item.id)}</td><td>${percentage(item.score)}</td><td>${number(item.glucose, 0, "mg/dL")}</td><td>${item.timestamp ? "Exact" : "Unavailable"}</td></tr>`).join("")}</tbody></table><p class="muted">Ranking uses an exact shared replay timestamp. Missing timestamps are not substituted with nearest observations.</p>`;
    }

    function render(response) {
      const forecast = response.forecast || {};
      const context = response.context || {};
      const quality = response.quality || {};
      const outcome = response.outcome || {};
      label.textContent = date(response.forecast_timestamp);
      renderWorklist(response);
      const profile = participant.profile || {};
      const fields = [["Age", "ehr_age_years", "years"], ["Gender", "ehr_gender_female", ""], ["BMI", "ehr_bmi", "kg/m²"], ["HbA1c", "ehr_hba1c_pct", "%"], ["Fasting glucose", "ehr_fasting_glucose_mg_dl", "mg/dL"], ["Fasting insulin", "ehr_fasting_insulin", ""], ["Triglycerides", "ehr_triglycerides_mg_dl", "mg/dL"], ["Total cholesterol", "ehr_total_cholesterol_mg_dl", "mg/dL"], ["HDL", "ehr_hdl_mg_dl", "mg/dL"], ["LDL", "ehr_ldl_mg_dl", "mg/dL"]];
      const profileHtml = fields.map(([title, key, unit]) => {
        const value = key === "ehr_gender_female" ? (profile[key] === 1 ? "Female" : profile[key] === 0 ? "Male" : "Unavailable") : number(profile[key], 1, unit);
        return `<div class="item"><span>${title}</span><strong>${esc(value)}</strong></div>`;
      }).join("");
      const streamHtml = Object.entries(quality.streams || {}).map(([key, status]) => `<li><strong>${esc(key)}</strong>: ${status === "available" ? "Available" : "Missing"}</li>`).join("");
      const outcomeText = outcome.status === "available" ? `Observed exact +120-minute glucose: ${number(outcome.glucose_mg_dl, 0, "mg/dL")} · absolute error: ${number(outcome.absolute_error_mg_dl, 1, "mg/dL")}` : esc(outcome.reason || "Observed outcome unavailable");
      app.className = "layout";
      app.innerHTML = `<section class="card"><h2>Synchronized feature snapshots</h2><div class="chart"><canvas id="timeline" role="img" aria-label="Synchronized historical glucose, heart-rate, activity, and meal summary timeline"></canvas></div><p class="muted">Real timestamp axis. Forecast marker is at t; observed target marker is at t+120 and is revealed only after advancing. These are model-input snapshots, not raw measurements or exact meal events.</p><div class="stats"><div class="stat"><label>Glucose at t</label><strong>${number(context.glucose, 0, "mg/dL")}</strong></div><div class="stat"><label>Heart rate at t</label><strong>${number(context.heart_rate, 0, "bpm")}</strong></div><div class="stat"><label>Activity at t</label><strong>${number(context.activity, 1, "METs")}</strong></div><div class="stat"><label>Carbs / meals, past 2h</label><strong>${number(context.carbs_2h, 0, "g")} / ${number(context.meal_events_2h, 1)}</strong></div></div><details><summary>Accessible synchronized data table</summary><table><thead><tr><th>Timestamp</th><th>Glucose</th><th>Heart rate</th><th>Activity</th><th>Carbs 2h</th><th>Meals 2h</th></tr></thead><tbody>${(response.timeline || []).map(row => `<tr><td>${esc(date(row.timestamp))}</td><td>${number(row.glucose, 1, "mg/dL")}</td><td>${number(row.heart_rate, 1, "bpm")}</td><td>${number(row.activity, 1, "METs")}</td><td>${number(row.carbs_2h, 1, "g")}</td><td>${number(row.meal_events_2h, 1)}</td></tr>`).join("")}</tbody></table></details></section><section class="card"><div class="forecast"><label>Probability of exact glucose ≥180 mg/dL at +120 minutes · ${esc(forecast.source || source.value)}</label><strong>${percentage(forecast.score)}</strong><div>${forecast.glucose_mg_dl == null ? "Glucose regression estimate unavailable for this source" : `${number(forecast.glucose_mg_dl, 1, "mg/dL")} estimated glucose`}</div><small>Fold ${esc(forecast.fold)} · ${esc(forecast.calibration || "Calibration unavailable")}<br>${esc(forecast.provenance || "Provenance unavailable")}</small></div><div class="replay-actions"><button id="advance" type="button" ${state.elapsed ? "disabled" : ""}>Advance to observed +120 minutes</button><button id="reset" type="button" ${state.elapsed ? "" : "disabled"}>Reset to forecast time</button></div><div class="outcome"><h2>Forecast versus observed</h2><p>${outcomeText}</p>${outcome.status === "hidden" ? "<p class=\"muted\">The observed future measurement is intentionally hidden until the retrospective replay advances.</p>" : ""}</div><div class="quality"><h2>Data quality and eligibility</h2><p>${quality.missing_feature_count || 0} of ${quality.feature_count || 0} displayed stream summaries missing.</p><ul>${streamHtml}</ul><p class="muted">${esc(quality.note || "No quality note available.")}</p></div><h2>Static EHR summary · recorded baseline context</h2><div class="profile">${profileHtml}</div><div class="quality"><h2>Personalization replay</h2><p><strong>Unavailable.</strong> ${esc(response.personalization?.reason || "No supported comparison artifact.")}</p><p class="muted">The fixed 72-hour adaptation idea is not a physiology simulator. Historical aggregate claims remain in the evaluation documentation and are not presented as this individual replay.</p></div></section>`;
      const advance = byId("advance");
      const reset = byId("reset");
      if (advance) advance.onclick = () => { state.elapsed = 120; fetchReplay(); };
      if (reset) reset.onclick = () => { state.elapsed = 0; fetchReplay(); };
      drawTimeline(response);
    }

    function fetchReplay() {
      if (!participant) return;
      const token = ++requestVersion;
      const timestamp = participant.timestamps[Number(slider.value)];
      state = { ...state, loading: true, error: null, response: null };
      app.className = "loading";
      app.textContent = "Loading replay frame…";
      fetch(query(timestamp, state.elapsed)).then(response => {
        if (!response.ok) throw new Error("Replay frame unavailable.");
        return response.json();
      }).then(response => {
        if (token !== requestVersion) return;
        state = { ...state, loading: false, response };
        render(response);
      }).catch(error => {
        if (token !== requestVersion) return;
        state = { ...state, loading: false, error, response: null };
        app.className = "loading";
        app.textContent = error.message;
      });
    }

    function chooseParticipant() {
      participant = data.participants.find(item => String(item.id) === String(picker.value));
      slider.max = Math.max(0, (participant.timestamps || []).length - 1);
      slider.value = 0;
      state = resetReplayState();
      fetchReplay();
    }

    fetch("/api/data").then(response => {
      if (!response.ok) throw new Error("Dashboard data unavailable.");
      return response.json();
    }).then(payload => {
      data = payload;
      if (!Array.isArray(data.participants) || !data.participants.length) throw new Error("No eligible participants found.");
      picker.innerHTML = data.participants.map(item => `<option value="${esc(item.id)}">Participant ${esc(item.id)}</option>`).join("");
      source.innerHTML = (data.sources || []).filter(item => item.available !== false).map(item => `<option value="${esc(item.id)}">${esc(item.label)}</option>`).join("");
      picker.disabled = source.disabled = slider.disabled = false;
      picker.onchange = chooseParticipant;
      source.onchange = () => { state = resetReplayState(); fetchReplay(); };
      slider.oninput = () => { state = resetReplayState(); fetchReplay(); };
      byId("prev").onclick = () => { slider.value = Math.max(0, Number(slider.value) - 1); state = resetReplayState(); fetchReplay(); };
      byId("next").onclick = () => { slider.value = Math.min(Number(slider.max), Number(slider.value) + 1); state = resetReplayState(); fetchReplay(); };
      byId("prev").disabled = byId("next").disabled = false;
      window.addEventListener("resize", () => { if (state.response) drawTimeline(state.response); });
      chooseParticipant();
    }).catch(error => {
      list.className = "card loading";
      list.textContent = error.message;
      app.className = "loading";
      app.textContent = "No retrospective replay data loaded.";
    });
  }

  if (typeof module !== "undefined") module.exports = { percentage, resetReplayState, selectReplayState, finiteOrNull, segmentSeries, GAP_MS };
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", init);
})(typeof window !== "undefined" ? window : globalThis);
