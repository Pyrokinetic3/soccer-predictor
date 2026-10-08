# Premier League Predictor

A Python project exploring how historical match results can be used to estimate team strength, predict Premier League match outcomes, and simulate the title race throughout a season.

The project starts with an Elo rating system and data preparation, with plans to build an XGBoost classifier and an interactive web application. Development takes place in a Jupyter notebook in VS Code.

**Status: In development.** Data loading, score-format cleaning, and basic Elo functions are implemented. Historical Elo processing, machine learning, season simulations, and the website are planned. No predictive performance results have been established yet.

## Goals

- Estimate team strength using ratings updated after each match.
- Predict probabilities of a home win, draw, or away win using information available before kickoff.
- Compare XGBoost with simpler baseline models.
- Simulate remaining fixtures to estimate each team's chance of winning the league.
- Present ratings, predictions, and title probabilities through an interactive dashboard.

## Data

The initial dataset contains **1,140 Premier League match records** across three seasons:

| Season | Matches |
| --- | ---: |
| 2023–24 | 380 |
| 2024–25 | 380 |
| 2025–26 | 380 |

Source: [OpenFootball's football.json repository](https://github.com/openfootball/football.json), which provides public-domain football fixtures and results under CC0-1.0.

The JSON files include match dates, home and away teams, and scores. They are loaded into pandas DataFrames and combined into a chronological match table.

Data preparation currently includes:

- Adding season labels and combining the three datasets.
- Converting dates to datetime values and sorting matches chronologically.
- Standardizing inconsistent score formats: 27 records in the 2025–26 file store scores directly as lists rather than under the nested `score.ft` field.
- Extracting home and away goals into separate columns.

The score-format correction uses recorded results rather than imputing outcomes. These files provide a starting dataset; further checks for duplicate fixtures, consistent team names, and valid results are planned.

## Elo Rating System

Each team starts at **1500 Elo** in the initial implementation. Two functions have been written:

- `expected_result(home_rating, away_rating)` calculates the home team's expected result value from the rating difference.
- `update_ratings(home_rating, away_rating, home_goals, away_goals, k=20)` returns updated ratings after a match.

Actual results are encoded as 1 for a home win, 0.5 for a draw, and 0 for a home loss. The rating change is:

```text
change = K × (actual result − expected result)
```

The home team gains this amount and the away team loses the same amount. Unexpected wins earn more points, while unexpected losses cost more. The current version uses `K = 20` and treats all winning margins equally.

The Elo expected result represents `P(home win) + 0.5 × P(draw)`; it is not a standalone home-win probability.

The next step is to process all matches chronologically, carry ratings forward between seasons, and save each team's rating before every match. Home advantage, offseason adjustments, and initialization rules for promoted teams are possible later improvements.

## Planned Machine Learning and Evaluation

An XGBoost classifier will predict three outcomes: **home win, draw, and away win**. Candidate features include pre-match Elo differences, recent goals scored and conceded, recent results, and days of rest.

The evaluation plan includes:

- Comparing XGBoost with simple baselines, including an Elo-based probability model.
- Using chronological training and validation periods rather than random match splits.
- Tuning hyperparameters on training and validation data only.
- Evaluating probability quality with log loss and calibration, alongside classification accuracy.
- Examining errors by season and matchup type.

### Preventing data leakage

Data cleaning and leakage prevention are separate parts of the workflow. Cleaning makes records consistent; leakage prevention ensures predictions only use information available at prediction time.

The modeling pipeline will:

- Record pre-match Elo before updating ratings with that match's outcome.
- Calculate rolling statistics from previous matches only.
- Exclude the target match's goals and other post-match information from input features.
- Fit learned preprocessing steps only on the training portion of each split.
- Reserve a later period for final evaluation, without using its results to select features or tune parameters.

The three seasons will support chronological Elo calculations. End-of-2025–26 ratings will not be used to predict earlier matches. A final evaluation split will be established before model tuning begins.

## Planned Season Simulator and Website

The season simulator will combine the actual league table with predicted outcomes for remaining fixtures. Repeating the simulation will produce estimated title probabilities that can be updated after each matchday.

The simulator will need to handle league tie-breaking rules and document its assumptions about future team strength. Title probabilities will be conditional on the underlying match model and those assumptions.

The planned website will provide:

- Team-strength rankings and rating histories.
- Matchup selection with win, draw, and loss probabilities.
- Title-probability charts across matchdays.
- Historical evaluation results and explanations of model limitations.

Streamlit is the intended starting point for a Python-based interactive application. Modeling and evaluation will be developed before expanding the interface.

## Technology

| Tool | Role | Status |
| --- | --- | --- |
| Python | Data processing and rating calculations | In use |
| pandas | Loading, cleaning, and organizing match data | In use |
| Jupyter / VS Code | Development and exploration | In use |
| NumPy / Matplotlib | Numerical work and visualization | Installed for upcoming work |
| scikit-learn / XGBoost | Modeling, preprocessing, and evaluation | Planned |
| Streamlit | Interactive web application | Planned |

## Running the Current Notebook

The current development environment uses Windows, Python, VS Code, and Microsoft's Python and Jupyter extensions.

1. Download or clone this repository and open its folder in VS Code.
2. Create a virtual environment from the project terminal:

   ```powershell
   python -m venv .venv
   ```

3. Install the packages used by the notebook:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib ipykernel
   ```

4. Open `exploration.ipynb` and select `.venv` as its kernel.
5. Keep the three JSON files alongside the notebook, then run the cells from top to bottom.

A dependency file with tested package versions will be added as the project develops. The `.venv` directory is excluded from version control.

## Roadmap

- [x] Set up a local Python environment and notebook.
- [x] Load three seasons of Premier League match records.
- [x] Standardize score formats and extract goal totals.
- [x] Implement expected-result and Elo-update functions.
- [x] Generate historical pre-match Elo ratings and team rankings.
- [x] Add data validation and checks for rating calculations.
- [x] Build features using only previously available information.
- [x] Establish chronological evaluation and baseline models.
- [x] Train and evaluate XGBoost.
- [ ] Implement and evaluate season simulations.
- [ ] Build and deploy an interactive dashboard.
- [ ] Explore expansion to other domestic leagues and the Champions League.

## Limitations

The current rating functions do not account for home advantage, goal margin, injuries, transfers, or lineup changes. Match results alone cannot fully describe team strength. Three seasons also provide only three title races, so match-level evaluation will be the primary measure of predictive quality. There are currently no claims of betting profitability or demonstrated predictive accuracy.
