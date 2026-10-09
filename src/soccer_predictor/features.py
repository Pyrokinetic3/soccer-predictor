import numpy as np
import pandas as pd


INPUT_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_difference",
    "home_form_5",
    "away_form_5",
    "home_goals_scored_5",
    "home_goals_conceded_5",
    "away_goals_scored_5",
    "away_goals_conceded_5",
]


def expected_result(home_rating, away_rating, home_advantage=0):
    """Return the home team's expected Elo score."""
    return 1 / (
        1 + 10 ** (
            (away_rating - (home_rating + home_advantage)) / 400
        )
    )


def match_result(home_goals, away_goals):
    """Return 0 for home win, 1 for draw, or 2 for away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def points_won(home_goals, away_goals):
    """Return the league points earned by each team."""
    result = match_result(home_goals, away_goals)

    if result == 0:
        return 3, 0
    if result == 1:
        return 1, 1
    return 0, 3


def update_ratings(
    home_rating,
    away_rating,
    home_goals,
    away_goals,
    k=40,
    home_advantage=0,
):
    """Return both teams' updated Elo ratings."""
    expected = expected_result(
        home_rating, away_rating, home_advantage
    )

    result = match_result(home_goals, away_goals)
    actual = {0: 1, 1: 0.5, 2: 0}[result]

    change = k * (actual - expected)

    return home_rating + change, away_rating - change


def recent_average(history, window=5):
    """Average the previous matches, or return NaN with no history."""
    return np.mean(history[-window:]) if history else np.nan


def build_features(matches, k=40, home_advantage=50):
    """Build prematch features for chronologically ordered results."""
    matches = matches.copy()

    if not matches["date"].is_monotonic_increasing:
        raise ValueError("Matches must be ordered chronologically.")

    if matches[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError(
            "Feature rebuilding requires completed match scores."
        )

    teams = pd.concat(
        [matches["team1"], matches["team2"]]
    ).unique()

    ratings = {team: 1500.0 for team in teams}
    points_history = {team: [] for team in teams}
    scored_history = {team: [] for team in teams}
    conceded_history = {team: [] for team in teams}

    feature_rows = []
    results = []

    for match in matches.itertuples(index=False):
        home = match.team1
        away = match.team2

        # Record features using only earlier results.
        feature_rows.append({
            "home_elo_before": ratings[home],
            "away_elo_before": ratings[away],
            "elo_difference": ratings[home] - ratings[away],
            "home_form_5": recent_average(points_history[home]),
            "away_form_5": recent_average(points_history[away]),
            "home_goals_scored_5": recent_average(
                scored_history[home]
            ),
            "home_goals_conceded_5": recent_average(
                conceded_history[home]
            ),
            "away_goals_scored_5": recent_average(
                scored_history[away]
            ),
            "away_goals_conceded_5": recent_average(
                conceded_history[away]
            ),
        })

        results.append(
            match_result(match.home_goals, match.away_goals)
        )

        # Update histories only after recording prematch features.
        ratings[home], ratings[away] = update_ratings(
            ratings[home],
            ratings[away],
            match.home_goals,
            match.away_goals,
            k=k,
            home_advantage=home_advantage,
        )

        home_points, away_points = points_won(
            match.home_goals, match.away_goals
        )

        points_history[home].append(home_points)
        points_history[away].append(away_points)

        scored_history[home].append(match.home_goals)
        conceded_history[home].append(match.away_goals)

        scored_history[away].append(match.away_goals)
        conceded_history[away].append(match.home_goals)

    features = pd.DataFrame(feature_rows, index=matches.index)
    matches[INPUT_COLUMNS] = features[INPUT_COLUMNS]
    matches["result"] = results

    return matches


def build_team_state(matches, prediction_date, k=40, home_advantage=50):
    # Only use results from dates before the fixture being predicted.
    earlier_matches = matches.loc[
        matches["date"] < pd.Timestamp(prediction_date)
    ].sort_values("date", kind="stable")

    if earlier_matches[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Team history must contain completed matches.")

    teams = pd.concat([
        earlier_matches["team1"],
        earlier_matches["team2"]
    ]).unique()

    state = {
        team: {
            "rating": 1500.0,
            "points": [],
            "scored": [],
            "conceded": []
        }
        for team in teams
    }

    for match in earlier_matches.itertuples(index=False):
        home = state[match.team1]
        away = state[match.team2]

        home["rating"], away["rating"] = update_ratings(
            home["rating"],
            away["rating"],
            match.home_goals,
            match.away_goals,
            k=k,
            home_advantage=home_advantage
        )

        home_points, away_points = points_won(
            match.home_goals, match.away_goals
        )

        home["points"].append(home_points)
        away["points"].append(away_points)

        home["scored"].append(match.home_goals)
        home["conceded"].append(match.away_goals)

        away["scored"].append(match.away_goals)
        away["conceded"].append(match.home_goals)

    return state


def make_fixture_features(state, home_team, away_team):
    if home_team == away_team:
        raise ValueError("Choose two different teams.")

    for team in [home_team, away_team]:
        if team not in state:
            raise ValueError(f"No match history found for {team}.")

    home = state[home_team]
    away = state[away_team]

    features = {
        "home_elo_before": home["rating"],
        "away_elo_before": away["rating"],
        "elo_difference": home["rating"] - away["rating"],
        "home_form_5": recent_average(home["points"]),
        "away_form_5": recent_average(away["points"]),
        "home_goals_scored_5": recent_average(home["scored"]),
        "home_goals_conceded_5": recent_average(home["conceded"]),
        "away_goals_scored_5": recent_average(away["scored"]),
        "away_goals_conceded_5": recent_average(away["conceded"])
    }

    return pd.DataFrame([features], columns=INPUT_COLUMNS)