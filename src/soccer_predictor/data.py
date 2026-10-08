import json
from pathlib import Path

import pandas as pd


TRAINING_SEASONS = [
    "2021_22",
    "2022_23",
    "2023_24",
    "2024_25",
    "2025_26",
]

ORIGINAL_SEASONS = [
    "2023_24",
    "2024_25",
    "2025_26",
]


def load_matches(data_directory, seasons):
    """Load season files and arrange their matches chronologically."""
    data_directory = Path(data_directory)
    season_tables = []

    for season in seasons:
        filename = data_directory / f"premier_league_{season}.json"

        with filename.open(encoding="utf-8") as file:
            data = json.load(file)

        season_matches = pd.json_normalize(data["matches"])
        season_matches["season"] = season
        season_tables.append(season_matches)

    matches = pd.concat(season_tables, ignore_index=True)

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    # Support files containing either or both score columns.
    if "score.ft" not in matches:
        matches["score.ft"] = matches["score"]

    elif "score" in matches:
        matches["score.ft"] = matches["score.ft"].fillna(
            matches["score"]
        )

    matches["home_goals"] = matches["score.ft"].str[0]
    matches["away_goals"] = matches["score.ft"].str[1]

    return matches