import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccer_predictor.data import load_matches
from soccer_predictor.features import build_team_state, make_fixture_features
from soccer_predictor.predict import predict_fixture
from soccer_predictor.live_data import fetch_current_season

SEASON = "2026_27"
SOURCE = (
    "https://raw.githubusercontent.com/openfootball/"
    "football.json/master/2026-27/en.1.json"
)
LONDON = ZoneInfo("Europe/London")


def read_json(path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def fixture_id(home, away):
    # Dates can change when a match is postponed.
    return json.dumps([SEASON, home, away], ensure_ascii=False)


def full_time(match):
    score = match.get("score")

    if isinstance(score, dict):
        score = score.get("ft")

    if score is not None and (
        not isinstance(score, list)
        or len(score) != 2
        or any(type(goal) is not int or goal < 0 for goal in score)
    ):
        raise ValueError("Invalid full-time score in downloaded data.")

    return score


def midnight(date):
    return datetime.fromisoformat(date).replace(tzinfo=LONDON)


def main():
    now = datetime.now(timezone.utc)
    stamp = now.isoformat()

    raw_path = (
        ROOT / "data" / "raw"
        / f"premier_league_{SEASON}.json"
    )
    archive_path = ROOT / "data" / "prediction_archive.json"

    previous = read_json(raw_path, {"matches": []})

    # Download fixtures and final results from the API.
    fresh = fetch_current_season(
        SEASON,
        previous["matches"],
    )

    matches = fresh["matches"]

    ids = [
        fixture_id(match["team1"], match["team2"])
        for match in matches
    ]

    if len(matches) != 380 or len(set(ids)) != 380:
        raise ValueError(
            "Expected 380 unique Premier League fixtures."
        )

    old_ids = {
        fixture_id(match["team1"], match["team2"])
        for match in previous["matches"]
    }

    if old_ids and set(ids) != old_ids:
        raise ValueError(
            "Source fixture/team names changed; "
            "review before updating."
        )

    old_completed = {
        fixture_id(match["team1"], match["team2"])
        for match in previous["matches"]
        if full_time(match) is not None
    }

    new_completed = {
        key
        for key, match in zip(ids, matches)
        if full_time(match) is not None
    }

    if not old_completed <= new_completed:
        raise ValueError(
            "Downloaded data lost completed results; "
            "keeping existing files."
        )

    for match in matches:
        if match.get("date"):
            datetime.strptime(match["date"], "%Y-%m-%d")

        # Normalize both supported score formats.
        match["score"] = {"ft": full_time(match)}

    save_json(raw_path, fresh)

    # Load the existing model and completed match history.
    bundle = joblib.load(
        ROOT / "models" / "calibrated_logistic.joblib"
    )

    historical = load_matches(
        ROOT / "data" / "raw",
        bundle["training_seasons"],
    )

    current = load_matches(
        ROOT / "data" / "raw",
        [SEASON],
    )

    completed = current[
        ["home_goals", "away_goals"]
    ].notna().all(axis=1)

    history = pd.concat(
        [historical, current.loc[completed]],
        ignore_index=True,
    )

    history = (
        history.sort_values("date", kind="stable")
        .reset_index(drop=True)
    )

    archive = read_json(
        archive_path,
        {"fixtures": {}},
    )["fixtures"]

        # Recover the forecasts published before Matchday 6.
    original_snapshot = read_json(
        ROOT / "data" / "predictions_matchday6.json",
        {"fixtures": []},
    )

    # Verified timestamp of the Git commit containing this snapshot.
    published_at = "2026-10-09T05:34:21+00:00"

    for fixture in original_snapshot["fixtures"]:
        key = fixture_id(
            fixture["home_team"],
            fixture["away_team"],
        )

        if datetime.fromisoformat(
            published_at
        ) < midnight(fixture["date"]):
            archive.setdefault(key, {
                **fixture,
                "season": SEASON,
                "recorded_at": published_at,
            })

    pending = current.loc[
        ~completed
        & current["date"].notna()
        & current["status"].isin(["SCHEDULED", "TIMED"])
    ].sort_values("date", kind="stable")

    pending = pending.loc[
        pending["date"].dt.date
        >= now.astimezone(LONDON).date()
    ]

    fixtures = []

    for match in pending.itertuples(index=False):
        key = fixture_id(match.team1, match.team2)
        date = match.date.strftime("%Y-%m-%d")
        cutoff = midnight(date)
        saved = archive.get(key)

        # Refresh forecasts only before matchday begins.
        # This also works when the kickoff time is unknown.
        if now < cutoff:
            state = build_team_state(
                history,
                date,
                k=bundle["elo_k"],
                home_advantage=bundle["home_advantage"],
            )

            features = make_fixture_features(
                state,
                match.team1,
                match.team2,
            ).iloc[0]

            candidate = {
                "season": SEASON,
                "date": date,
                "home_team": match.team1,
                "away_team": match.team2,
                "probabilities": predict_fixture(
                    bundle,
                    history,
                    match.team1,
                    match.team2,
                    date,
                ),
                "features": {
                    name: (
                        float(value)
                        if math.isfinite(value)
                        else None
                    )
                    for name, value in features.items()
                },
            }

            old = {
                name: value
                for name, value in (saved or {}).items()
                if name != "recorded_at"
            }

            # Preserve the timestamp when nothing changed.
            if old != candidate:
                saved = {
                    **candidate,
                    "recorded_at": stamp,
                }
                archive[key] = saved

        # Never create a retrospective forecast.
        if (
            saved
            and saved["date"] == date
            and datetime.fromisoformat(
                saved["recorded_at"]
            ) < cutoff
        ):
            fixtures.append(saved)

        if len(fixtures) == 10:
            break

    # Attach actual scores to previously recorded forecasts.
    results = []

    for match in matches:
        saved = archive.get(
            fixture_id(match["team1"], match["team2"])
        )
        score = full_time(match)

        if (
            saved is None
            or score is None
            or not match.get("date")
        ):
            continue

        if saved["date"] != match["date"]:
            continue

        if datetime.fromisoformat(
            saved["recorded_at"]
        ) >= midnight(match["date"]):
            continue

        kickoff = midnight(match["date"])

        if match.get("time"):
            kickoff = datetime.fromisoformat(
                f'{match["date"]}T{match["time"]}'
            ).replace(tzinfo=LONDON)

        if kickoff >= now:
            continue

        results.append({
            **saved,
            "kickoff_at": kickoff.isoformat(),
            "kickoff_time_known": bool(match.get("time")),
            "home_goals": score[0],
            "away_goals": score[1],
        })

    save_json(archive_path, {"fixtures": archive})

    outputs = {
        "predictions.json": {
            "history_through": (
                history["date"].max().strftime("%Y-%m-%d")
            ),
            "round": "Next 10 fixtures",
            "fixtures": fixtures,
        },
        "history.json": {
            "fixtures": sorted(
                results,
                key=lambda match: match["kickoff_at"],
                reverse=True,
            ),
        },
    }

    # Avoid new commits when the displayed data is unchanged.
    for filename, payload in outputs.items():
        path = ROOT / "website" / filename
        previous_output = read_json(path, {})
        previous_output.pop("generated_at", None)

        if previous_output != payload:
            save_json(path, {
                "generated_at": stamp,
                **payload,
            })

    print(
        f"Updated: {len(fixtures)} upcoming predictions; "
        f"{len(results)} settled forecasts."
    )


if __name__ == "__main__":
    main()