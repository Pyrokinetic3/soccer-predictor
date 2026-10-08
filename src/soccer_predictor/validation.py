import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


FIXTURE_COLUMNS = ["season", "team1", "team2"]


def make_validation_fixtures(original_matches, n_splits=3):
    """Preserve the original chronological validation fixtures."""
    validation_fixtures = []

    for _, valid_indices in TimeSeriesSplit(
        n_splits=n_splits
    ).split(original_matches):
        validation_fixtures.append(
            original_matches.iloc[valid_indices][
                FIXTURE_COLUMNS
            ].copy()
        )

    return validation_fixtures


def make_cv_splits(matches, validation_fixtures):
    """Map validation fixtures to positions in the expanded data."""
    if matches.duplicated(FIXTURE_COLUMNS).any():
        raise ValueError("Duplicate fixture identities found.")

    match_ids = pd.MultiIndex.from_frame(
        matches[FIXTURE_COLUMNS]
    )

    cv_splits = []

    for fixtures in validation_fixtures:
        fixture_ids = pd.MultiIndex.from_frame(
            fixtures[FIXTURE_COLUMNS]
        )

        valid_indices = np.flatnonzero(
            match_ids.isin(fixture_ids)
        )

        if len(valid_indices) != len(fixtures):
            raise ValueError(
                "Some validation fixtures are missing or duplicated."
            )

        valid_date = matches["date"].iloc[valid_indices].min()

        train_indices = np.flatnonzero(
            matches["date"] < valid_date
        )

        if len(train_indices) == 0:
            raise ValueError("A fold has no earlier training matches.")

        cv_splits.append((train_indices, valid_indices))

    return cv_splits


def split_calibration_indices(
    matches, train_indices, fit_fraction=0.8
):
    """Split training chronologically without dividing a calendar day."""
    cut_position = int(len(train_indices) * fit_fraction)

    calibration_start_date = matches["date"].iloc[
        train_indices[cut_position]
    ]

    training_dates = matches["date"].iloc[train_indices]

    fit_indices = train_indices[
        training_dates < calibration_start_date
    ]

    calibration_indices = train_indices[
        training_dates >= calibration_start_date
    ]

    if len(fit_indices) == 0 or len(calibration_indices) == 0:
        raise ValueError("The calibration split produced an empty set.")

    return fit_indices, calibration_indices