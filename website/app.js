/* Static site: Python produces the data; this file displays it. */
const $ = (id) => document.getElementById(id);
const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const shortName = (name) =>
  name
    .replace(/ FC$| AFC$/, "")
    .replace("Brighton & Hove Albion", "Brighton")
    .replace("Manchester United", "Man United")
    .replace("Manchester City", "Man City")
    .replace("Tottenham Hotspur", "Tottenham")
    .replace("Nottingham Forest", "Nott’m Forest");
const initials = (name) =>
  shortName(name)
    .split(" ")
    .map((s) => s[0])
    .join("")
    .slice(0, 3);
const percent = (value) => `${(value * 100).toFixed(1)}%`;
const dateLabel = (date) =>
  new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(date));
const palette = [
  ["#aa8cde", "#f3effa"],
  ["#70b9ad", "#edf8f4"],
  ["#e9af79", "#fff4e9"],
  ["#d992b7", "#fcf0f7"],
  ["#92a7e0", "#f0f3fc"],
  ["#c5b668", "#faf8e9"],
];
function probabilities(fixture) {
  const p = fixture.probabilities;
  const values = [p?.home_win, p?.draw, p?.away_win];
  if (
    !values.every(
      (v) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1,
    ) ||
    Math.abs(values.reduce((a, b) => a + b, 0) - 1) > 0.001
  )
    throw new Error("Invalid probabilities in prediction data.");
  return values;
}
async function readJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path}.`);
  return response.json();
}
function flipCard(card, open) {
  card.classList.toggle("is-flipped", open);
  const front = card.querySelector(".card-front");
  const back = card.querySelector(".card-back");
  front.inert = open;
  back.inert = !open;
  front.setAttribute("aria-hidden", String(open));
  back.setAttribute("aria-hidden", String(!open));
  card
    .querySelector(".flip-button")
    .setAttribute("aria-expanded", String(open));
  (open
    ? card.querySelector(".close-card")
    : card.querySelector(".flip-button")
  ).focus({ preventScroll: true });
}
function fixtureCard(fixture, index) {
  const p = probabilities(fixture);
  const h = escapeHTML(shortName(fixture.home_team));
  const a = escapeHTML(shortName(fixture.away_team));
  const f = fixture.features || {};
  const number = (key, digits = 1) =>
    typeof f[key] === "number" && Number.isFinite(f[key])
      ? f[key].toFixed(digits)
      : "—";
  const stats = [
    ["Elo rating", "elo_before", 0],
    ["Points / match · last 5", "form_5", 1],
    ["Goals scored · last 5 avg.", "goals_scored_5", 1],
    ["Goals conceded · last 5 avg.", "goals_conceded_5", 1],
  ];
  const card = document.createElement("article");
  card.className = "fixture-card";
  card.style.setProperty("--accent", palette[index % palette.length][0]);
  card.style.setProperty("--tint", palette[index % palette.length][1]);
  card.setAttribute(
    "aria-label",
    `${fixture.home_team} versus ${fixture.away_team}`,
  );
  card.innerHTML = `<div class="card-inner">
    <div class="card-face card-front"><div class="card-meta"><time datetime="${escapeHTML(fixture.date)}">${dateLabel(fixture.date)}</time><span class="card-index">MATCH ${String(index + 1).padStart(2, "0")}</span></div>
      <div class="teams"><div class="team"><span class="team-badge" aria-hidden="true">${escapeHTML(initials(fixture.home_team))}</span><span class="team-name">${h}</span><span class="venue-label">Home</span></div><span class="versus">vs</span><div class="team"><span class="team-badge" aria-hidden="true">${escapeHTML(initials(fixture.away_team))}</span><span class="team-name">${a}</span><span class="venue-label">Away</span></div></div>
      <div class="probability-bar" aria-hidden="true">${p.map((v) => `<span style="width:${v * 100}%"></span>`).join("")}</div>
      <div class="probability-labels">${p.map((v, i) => `<div><strong>${percent(v)}</strong><span>${["Home win", "Draw", "Away win"][i]}</span></div>`).join("")}</div>
      <button class="flip-button" aria-expanded="false" aria-controls="stats-${index}" aria-label="View stats for ${h} versus ${a}">Behind the prediction <b aria-hidden="true">↗</b></button>
    </div>
    <div class="card-face card-back" id="stats-${index}" aria-hidden="true" inert>
      <div class="back-heading"><h3>The match in numbers</h3><button class="close-card" aria-label="Back to ${h} versus ${a}">↶</button></div>
      <table class="stats-table" aria-label="Prematch features for ${h} versus ${a}"><thead><tr><th scope="col">Prematch stats</th><th scope="col">Home</th><th scope="col">Away</th></tr></thead><tbody>${stats.map(([label, key, digits]) => `<tr><th scope="row">${label}</th><td>${number("home_" + key, digits)}</td><td>${number("away_" + key, digits)}</td></tr>`).join("")}</tbody></table>
      <div class="elo-diff"><span>Elo difference · home − away</span><strong>${number("elo_difference", 0)}</strong></div>
      <table class="outcomes-table" aria-label="Outcome probabilities"><thead><tr><th scope="col">Team</th><th scope="col">Win</th><th scope="col">Draw</th><th scope="col">Lose</th></tr></thead><tbody><tr><th scope="row">${h}</th><td>${percent(p[0])}</td><td>${percent(p[1])}</td><td>${percent(p[2])}</td></tr><tr><th scope="row">${a}</th><td>${percent(p[2])}</td><td>${percent(p[1])}</td><td>${percent(p[0])}</td></tr></tbody></table>
      <p class="back-note">${Object.keys(f).length ? "Form and goals use each team’s last five available matches. Elo uses available prior results." : "Stats are not included in this snapshot yet. Regenerate predictions to add them."}</p>
    </div></div>`;
  card
    .querySelector(".flip-button")
    .addEventListener("click", () => flipCard(card, true));
  card
    .querySelector(".close-card")
    .addEventListener("click", () => flipCard(card, false));
  card.querySelector(".card-front").addEventListener("click", (event) => {
    if (!event.target.closest("button")) flipCard(card, true);
  });
  card.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && card.classList.contains("is-flipped"))
      flipCard(card, false);
  });
  return card;
}
function renderPredictions(data) {
  if (!Array.isArray(data.fixtures)) throw new Error("Missing fixtures.");
  const fixtures = [...data.fixtures]
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(0, 10);
  const cards = fixtures.map(fixtureCard);
  $("fixtures-grid").replaceChildren(...cards);
  $("round-label").textContent = data.round || "Upcoming fixtures";
  $("data-status").textContent =
    `Snapshot: ${dateLabel(data.generated_at)} · Results through ${data.history_through ? dateLabel(data.history_through) : "unknown"}`;
  if (!fixtures.length)
    $("fixtures-grid").innerHTML =
      '<p class="loading-message">No upcoming fixtures in this snapshot. Check back after the next data update.</p>';
  $("data-note").textContent =
    "Saved prediction snapshot · Automatic updates are not enabled yet. Percentages may differ slightly from 100% due to rounding.";
}
function renderHistory(data) {
  if (!Array.isArray(data.fixtures))
    throw new Error("Missing history fixtures.");
  // Display only completed results backed by a timestamped pre-kickoff forecast.
  const fixtures = data.fixtures
    .filter(
      (f) =>
        Number.isInteger(f.home_goals) &&
        Number.isInteger(f.away_goals) &&
        f.home_goals >= 0 &&
        f.away_goals >= 0 &&
        Date.parse(f.recorded_at) < Date.parse(f.kickoff_at),
    )
    .sort((a, b) => Date.parse(b.kickoff_at) - Date.parse(a.kickoff_at))
    .slice(0, 30);
  $("history-count").textContent = `${fixtures.length} / 30 results`;
  if (!fixtures.length) {
    $("history-content").innerHTML =
      '<div class="empty-history"><div class="empty-symbol" aria-hidden="true">↗</div><h3>Every track record starts somewhere.</h3><p>No settled public predictions yet. This space will fill with predictions saved before kickoff and the results that followed. The results update process is coming next.</p></div>';
    return;
  }
  const rows = fixtures
    .map((f) => {
      const p = probabilities(f),
        pick = p.indexOf(Math.max(...p));
      const actual =
        f.home_goals > f.away_goals ? 0 : f.home_goals === f.away_goals ? 1 : 2;
      const names = [shortName(f.home_team), "Draw", shortName(f.away_team)];
      return `<tr><td>${dateLabel(f.kickoff_at)}</td><td>${escapeHTML(shortName(f.home_team))}<br>${escapeHTML(shortName(f.away_team))}</td><td>${escapeHTML(names[pick])}<br><small>${p.map(percent).join(" / ")} H/D/A</small></td><td>${f.home_goals}–${f.away_goals}</td><td><span class="result-tag ${pick === actual ? "correct" : ""}">${pick === actual ? "Correct" : "Missed"}</span></td></tr>`;
    })
    .join("");
  $("history-content").innerHTML =
    `<div class="history-scroll" tabindex="0" role="region" aria-label="Recent prediction results"><table class="history-table"><thead><tr><th scope="col">Date</th><th scope="col">Fixture</th><th scope="col">Model pick & probabilities</th><th scope="col">Score</th><th scope="col">Result</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
readJSON("predictions.json")
  .then(renderPredictions)
  .catch((error) => {
    console.error(error);
    $("data-status").textContent = "Prediction data unavailable";
    $("fixtures-grid").innerHTML =
      '<p class="error-message">The predictions could not be loaded. Try refreshing. If you’re previewing locally, serve the website through a local web server instead of opening the HTML file directly.</p>';
  });
readJSON("history.json")
  .then(renderHistory)
  .catch((error) => {
    console.error(error);
    $("history-count").textContent = "History unavailable";
    $("history-content").innerHTML =
      '<p class="error-message">The prediction history could not be loaded. Please try again later.</p>';
  });
