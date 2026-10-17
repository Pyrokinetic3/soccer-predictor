import json
import os
import re
from datetime import datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


def team_key(name):
    """Normalize spelling while preserving the existing team names."""
    name = name.casefold().replace("&", "and")
    name = re.sub(r"\b(?:fc|afc)\b", "", name)
    return re.sub(r"[^a-z0-9]", "", name)


def convert_matches(payload, season, existing_matches):
    year = int(season.split("_")[0])

    known_teams = {
        name
        for match in existing_matches
        for name in (match["team1"], match["team2"])
    }

    names = {
        team_key(name): name
        for name in known_teams
    }

    if len(names) != len(known_teams):
        raise ValueError("Ambiguous team names in existing data.")

    def resolve(team):
        for field in ("name", "shortName"):
            if team.get(field):
                key = team_key(team[field])
                if key in names:
                    return names[key]

        raise ValueError(
            f'Unmatched API team: {team.get("name")}. '
            "Review team names."
        )

    converted = []

    for match in payload["matches"]:
        if str(match["season"]["startDate"])[:4] != str(year):
            raise ValueError("API returned the wrong season.")

        kickoff = None

        if match.get("utcDate"):
            kickoff = datetime.fromisoformat(
                match["utcDate"].replace("Z", "+00:00")
            ).astimezone(ZoneInfo("Europe/London"))

        score = None

        # In-progress scores must never be treated as final results.
        if match["status"] == "FINISHED":
            goals = match["score"]["fullTime"]
            score = [goals["home"], goals["away"]]

            if any(
                type(goal) is not int or goal < 0
                for goal in score
            ):
                raise ValueError(
                    "Finished API match is missing a valid final score."
                )

            if kickoff is None:
                raise ValueError(
                    "Finished API match is missing its date."
                )

        converted.append({
            "round": f'Matchday {match["matchday"]}',
            "date": (
                kickoff.strftime("%Y-%m-%d")
                if kickoff else None
            ),
            "time": (
                kickoff.strftime("%H:%M")
                if kickoff else None
            ),
            "team1": resolve(match["homeTeam"]),
            "team2": resolve(match["awayTeam"]),
            "score": {"ft": score},
            "status": match["status"],
            "source_id": match["id"],
        })

    return {
        "name": (
            f"English Premier League "
            f"{season.replace('_', '/')}"
        ),
        "matches": converted,
    }


def fetch_current_season(season, existing_matches):
    token = os.environ.get(
        "FOOTBALL_DATA_TOKEN", ""
    ).strip()

    if not token:
        raise RuntimeError(
            "Set the FOOTBALL_DATA_TOKEN GitHub Actions secret."
        )

    year = int(season.split("_")[0])

    url = (
        "https://api.football-data.org/v4/"
        f"competitions/PL/matches?season={year}"
    )

    request = Request(
        url,
        headers={"X-Auth-Token": token},
    )

    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"Results API returned HTTP {error.code}. "
            "Check token, season access, or request limit."
        ) from None

    return convert_matches(
        payload,
        season,
        existing_matches,
    )