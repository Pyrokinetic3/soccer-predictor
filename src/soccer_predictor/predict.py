from .features import build_team_state, make_fixture_features
from .models import predict_calibrated


def predict_fixture(
    model_bundle,
    completed_matches,
    home_team,
    away_team,
    prediction_date,
):
    # Rebuild history using only results before the fixture date.
    team_state = build_team_state(
        completed_matches,
        prediction_date,
        k=model_bundle["elo_k"],
        home_advantage=model_bundle["home_advantage"],
    )

    # Create one row of prematch features.
    features = make_fixture_features(
        team_state,
        home_team,
        away_team,
    )

    probabilities = predict_calibrated(
        model_bundle["base_model"],
        model_bundle["calibrator"],
        features[model_bundle["input_columns"]],
    )[0]

    # Return named probabilities for easy use by the website.
    outcome_names = {
        0: "home_win",
        1: "draw",
        2: "away_win",
    }

    return {
        outcome_names[int(outcome)]: float(probability)
        for outcome, probability in zip(
            model_bundle["classes"],
            probabilities,
        )
    }