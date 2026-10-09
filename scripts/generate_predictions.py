from pathlib import Path
import sys
import joblib
import pandas as pd
import json

# This file is inside scripts/, one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soccer_predictor.predict import predict_fixture

model_path = PROJECT_ROOT / "models" / "calibrated_logistic.joblib"
model_bundle = joblib.load(model_path)

print("Model loaded successfully.")

from soccer_predictor.data import load_matches

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

# Historical seasons used by the saved model.
historical_matches = load_matches(
    DATA_DIRECTORY,
    model_bundle["training_seasons"],
)

# Current season includes completed and unplayed fixtures.
current_matches = load_matches(
    DATA_DIRECTORY,
    ["2026_27"],
)

completed_mask = current_matches[
    ["home_goals", "away_goals"]
].notna().all(axis=1)

completed_matches = current_matches.loc[completed_mask].copy()
pending_matches = current_matches.loc[~completed_mask].copy()

print("Historical matches:", len(historical_matches))
print("Completed current-season matches:", len(completed_matches))
print("Pending fixtures:", len(pending_matches))

# Completed results provide the history used for predictions.
match_history = pd.concat(
    [historical_matches, completed_matches],
    ignore_index=True
).sort_values("date", kind="stable").reset_index(drop=True)

# Exclude past unscored fixtures and fixtures with unknown dates.
today = pd.Timestamp.now(tz="Europe/London").date()

upcoming_matches = pending_matches.loc[
    pending_matches["date"].notna()
    & (pending_matches["date"].dt.date >= today)
].sort_values("date", kind="stable")

if upcoming_matches.empty:
    print("No upcoming fixtures found.")
    raise SystemExit(0)

# Select the round containing the earliest upcoming fixture.
next_round = upcoming_matches.iloc[0]["round"]

next_fixtures = upcoming_matches.loc[
    upcoming_matches["round"] == next_round
].copy()

print("\nSelected round:", next_round)
print(next_fixtures[["date", "team1", "team2"]].to_string(index=False))

predictions = []

for fixture in next_fixtures.itertuples(index=False):
    probabilities = predict_fixture(
        model_bundle=model_bundle,
        completed_matches=match_history,
        home_team=fixture.team1,
        away_team=fixture.team2,
        prediction_date=fixture.date,
    )

    predictions.append({
        "date": fixture.date.strftime("%Y-%m-%d"),
        "home_team": fixture.team1,
        "away_team": fixture.team2,
        "probabilities": probabilities,
    })

output = {
    "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    "history_through": match_history["date"].max().strftime("%Y-%m-%d"),
    "round": next_round,
    "fixtures": predictions,
}

output_path = PROJECT_ROOT / "website" / "predictions.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(output, file, indent=2, ensure_ascii=False, allow_nan=False)

print(f"\nSaved {len(predictions)} predictions to {output_path}")