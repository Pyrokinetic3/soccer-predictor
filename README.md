# Premier League Predictor

**Football predictions, built from match history.**

I’m Aren Khachikian, a lifelong soccer fan learning machine learning. I built this project to combine the two: estimate team strength, turn historical results into match probabilities, and track how those predictions compare with what actually happens.

The project uses five seasons of Premier League results, nine prematch features, chronological validation, and probability calibration. It compares a historical-frequency baseline, logistic regression, and XGBoost. **Calibrated logistic regression powers the current predictions.**

The website displays ten upcoming fixtures, animated cards with prematch statistics, and home-win, draw, and away-win probabilities. Its public results-history interface is built; automatic data updates and prediction archiving are still planned.

## Results

### Later-season evaluation

The evaluation notebook tests the models on the **first 50 completed matches in the local 2026/27 dataset**, after fitting on data from 2021/22–2025/26.

| Model | Test log loss ↓ | Test accuracy ↑ |
| --- | ---: | ---: |
| **Calibrated logistic regression** | **1.029980** | **46% — 23/50** |
| Logistic regression, full historical training set | 1.043127 | — |
| Tuned XGBoost | 1.065769 | — |
| Historical-frequency baseline | 1.117493 | 36% — 18/50 |

The calibrated model reduced log loss by **7.83% against the baseline**, with a **10 percentage-point increase in accuracy**. Accuracy is reported only where it was recorded in the notebook.

Log loss evaluates the probability assigned to the outcome that actually occurred. It penalizes confidently wrong predictions more heavily; lower is better. Accuracy counts how often the highest-probability outcome was correct. A predicted 60% chance of winning is a probability estimate, not a guaranteed result.

**These are initial results from a small sample.** They are separate from the website’s public prediction history and do not establish how well the model will perform over a full future season.

### Chronological validation

The original validation fixtures from the three-season dataset were preserved when two earlier seasons were added. This lets the models use more training history while being compared on the same validation matches.

| Fold | Training matches | Validation matches | First validation date |
| --- | ---: | ---: | --- |
| 1 | 1,043 | 285 | 2024-03-30 |
| 2 | 1,329 | 285 | 2025-01-04 |
| 3 | 1,610 | 285 | 2025-11-01 |

| Model / experiment | Mean validation log loss ↓ |
| --- | ---: |
| Historical-frequency baseline | 1.081514 |
| XGBoost, first tuning stage | 1.008032 |
| XGBoost, second tuning stage | 1.007953 |
| Logistic regression, `C=0.1` | 0.998849 |
| Calibrated logistic regression | **0.996776** |

These folds were used for model development and hyperparameter selection. Their scores are development results, not an independent final test. Recorded outputs and experiment details are in [exploration.ipynb](notebooks/exploration.ipynb); later-season results are in [evaluation.ipynb](notebooks/evaluation.ipynb).

## Data and features

The local JSON files contain **1,900 completed historical matches** across 2021/22–2025/26. The current 2026/27 snapshot contains 50 completed matches and 330 unplayed fixtures. These counts describe the checked-in snapshot and will change as results are updated.

Data source: [OpenFootball’s football.json repository](https://github.com/openfootball/football.json).

The loader combines season files, adds season labels, sorts by date, and extracts goal totals. It handles both nested `score.ft` values and alternative score-list records. Unplayed fixtures are excluded from training and completed-match feature construction; missing scores are not treated as draws.

The model uses nine inputs:

| Columns | Meaning |
| --- | --- |
| `home_elo_before`, `away_elo_before` | Each team’s rating before the match |
| `elo_difference` | Home Elo minus away Elo |
| `home_form_5`, `away_form_5` | Average points earned over each team’s last five available matches |
| `home_goals_scored_5`, `away_goals_scored_5` | Average goals scored over those matches |
| `home_goals_conceded_5`, `away_goals_conceded_5` | Average goals conceded over those matches |

Form uses three points for a win, one for a draw, and zero for a loss. When fewer than five previous matches are available, averages use the available history. With no previous matches, rolling features are missing; the logistic pipeline handles them with training-fitted median imputation.

Teams start at **1500 Elo**. The current prediction configuration uses **`K=40`** and a **50-point home advantage** when calculating the expected result. Ratings and rolling history carry across season boundaries. Elo is updated after each result; there is no offseason reset or goal-margin adjustment.

Outcome labels are `0 = home win`, `1 = draw`, and `2 = away win`. A home loss and an away win describe the same outcome, so only three probabilities are needed.

## Modeling decisions

**Baseline.** `DummyClassifier(strategy="prior")` assigns every fixture the outcome frequencies observed in its training set. It provides a reference for whether match-specific features add value.

**Logistic regression.** A pipeline combines median imputation, standardization, and logistic regression. The selected regularization setting is `C=0.1`, with `max_iter=1000`. Preprocessing is fitted within each training split.

**XGBoost.** A two-stage grid search first explores tree count, learning rate, depth, minimum child weight, and L2 regularization; a second stage explores row/column sampling, L1 regularization, and `gamma`. The full grids contain 432 and 36 candidates respectively, each evaluated on three folds. The selected configuration is:

```python
n_estimators=50
learning_rate=0.1
max_depth=1
min_child_weight=5
reg_lambda=1
subsample=1.0
colsample_bytree=0.8
reg_alpha=0.1
gamma=0
```

The small improvement from the second search is documented rather than treated as a major gain. The current search uses fixed tree counts, not early stopping.

**Calibration.** Initial checks compared predicted probabilities with observed frequencies for all three outcomes, plus home wins by fold. They showed overestimation in the 60–80% home-win bin and underestimation of draws.

For each calibration experiment, the historical training window is split chronologically at approximately 80/20, keeping calendar days together. The base pipeline fits on the earlier portion. A second logistic regression learns an adjustment from the log-transformed, clipped probabilities on the later portion. Both models are then evaluated on subsequent validation matches.

Calibration reduced mean validation log loss from **1.003321 to 0.996776** for the same reduced-training base model. On the 50-match test set, that base model scored **1.051749 before calibration** and **1.029980 after calibration**. This comparison differs from the uncalibrated model trained on all historical matches in the main results table.

Additional team-strength indicators were considered but deferred because of possible overlap with existing strength and form inputs. Their redundancy has not been established through an ablation experiment.

## Chronology and leakage prevention

- Features are recorded before a match’s result updates Elo, points, or goal history.
- Cross-validation training dates are strictly earlier than the first validation date.
- Validation fixtures are identified by season, home team, and away team so they survive changes to row positions.
- The calibration set is separate from the data used to fit its base model and precedes evaluation.
- The saved model is fitted using historical seasons; the 2026/27 test labels are not used to fit it.
- Later evaluation fixtures can use results from earlier evaluation fixtures to update team state. This is sequential evaluation with fixed model parameters, not a forecast of the entire season from a single starting date.
- Single-fixture predictions rebuild state using completed results from dates strictly before the requested fixture date.

Updating team history is different from retraining the model. New scores can update Elo and rolling features while the fitted classifier and calibrator remain unchanged.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Historical and current-season JSON files |
| `notebooks/exploration.ipynb` | Feature development, chronological validation, tuning, and calibration experiments |
| `notebooks/evaluation.ipynb` | Later-season evaluation, model export, reload verification, and individual fixture predictions |
| `src/soccer_predictor/data.py` | Data loading and season definitions |
| `src/soccer_predictor/features.py` | Elo, rolling features, team state, and fixture inputs |
| `src/soccer_predictor/validation.py` | Validation-fixture mapping and chronological calibration splits |
| `src/soccer_predictor/models.py` | Model factories and calibration helpers |
| `src/soccer_predictor/predict.py` | Reusable single-fixture prediction function |
| `models/calibrated_logistic.joblib` | Saved base model, calibrator, feature order, class labels, and configuration |
| `scripts/generate_predictions.py` | Exports probabilities and features for up to ten upcoming fixtures |
| `scripts/train.py`, `scripts/evaluate.py` | Placeholders for future command-line workflows; currently empty |
| `website/` | Static HTML, CSS, JavaScript, prediction data, and history data |
| `requirements.txt` | Python dependencies |

The notebooks preserve the experimentation and learning process. Reusable Python modules support evaluation and prediction outside the exploratory notebook.

## Run locally

### Python setup

Clone the repository and open a terminal in its root:

```bash
git clone https://github.com/Pyrokinetic3/soccer-predictor.git
cd soccer-predictor
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter and notebook kernel. Open notebooks from the project root or `notebooks/` directory and run their cells in order. Their setup cells resolve the `data/`, `src/`, and `models/` paths.

Dependencies are currently unpinned. Exact numerical reproduction and loading the saved model may depend on matching the original package versions; environment pinning remains a follow-up task.

### Explore, evaluate, and save the model

1. Run `notebooks/exploration.ipynb` to follow feature construction and model comparisons. Leave `RUN_FULL_SEARCH = False` for quick reevaluation of the selected XGBoost settings. Set it to `True` to rerun both full XGBoost grids. The logistic regression search runs separately.
2. Run `notebooks/evaluation.ipynb` for the later-season comparison. Its export cell writes `models/calibrated_logistic.joblib`, and its reload check confirms that saved predictions match the original ones. Rerunning the export replaces the existing bundle.
3. Use the final cells in that notebook to predict an individual fixture using its date and exact team names.

### Preview the website

The site requires no frontend build or Node installation. From the project root:

```bash
python -m http.server 8080 --directory website
```

Open [http://localhost:8080](http://localhost:8080). Use this local server instead of opening `index.html` directly so the browser can load the JSON files.

### Generate a new prediction snapshot

After updating the local season data, run the exporter with the project environment. On Windows:

```powershell
.\.venv\Scripts\python.exe scripts/generate_predictions.py
```

On macOS or Linux:

```bash
.venv/bin/python scripts/generate_predictions.py
```

The exporter loads the saved model and local results, selects the next ten dated, unplayed fixtures, and writes `website/predictions.json`. It includes the nine input features used by the website cards. The current season is configured as `2026_27`; upcoming-date selection uses London time.

This command does **not** download new results, retrain the model, archive predictions, or deploy the website.

## Website

The interface uses a white background with rainbow accents, a personal introduction, and selectable fixture cards. Cards flip to show Elo, recent form, scoring/conceding averages, and each team’s win/draw/loss probabilities.

| File | Role |
| --- | --- |
| `website/index.html` | Page structure and project copy |
| `website/styles.css` | Responsive styling and flip animations |
| `website/app.js` | Loads data, renders cards, and displays results history |
| `website/predictions.json` | Saved forecast snapshot, generation time, history cutoff, and fixture features |
| `website/history.json` | Public prediction history; initially empty |

Keyboard users can open cards with Enter and return with Escape or the back button. Reduced-motion preferences disable the flip transition. Google Fonts are optional; local sans-serif fonts provide a fallback.

The site currently displays a **saved snapshot**, not a live feed. The contents of `website/` can be served by a static host. Python runs when preparing the data, not in visitors’ browsers.

### Public prediction history

The history interface shows the latest 30 completed matches with recorded pre-kickoff predictions. It is intentionally empty until forecasts are archived and results added. Historical test results are shown separately and are not presented as published forecasts.

Each entry under `history.json` → `fixtures` needs:

| Field | Content |
| --- | --- |
| `home_team`, `away_team` | Team names |
| `recorded_at` | Original forecast timestamp, ISO format with timezone |
| `kickoff_at` | Verified kickoff timestamp, ISO format with timezone |
| `probabilities` | Decimal `home_win`, `draw`, and `away_win` values summing to one |
| `home_goals`, `away_goals` | Completed scores as integers |

The renderer requires completed scores and a recording timestamp before kickoff. The future archive process must preserve authentic timestamps and original probabilities, then attach results without recalculating the old forecast. The full archive should be retained even though the interface displays only 30 entries.

## Limitations and next steps

The model uses match results only: no expected goals, injuries, lineups, transfers, player-level information, or betting odds. New teams start at the same Elo rating, and predictions require available team history. Current data uses dates rather than verified kickoff times, so same-day availability and postponed fixtures need care when adding automation.

Repeated validation experiments can overfit development choices. The 50-match evaluation is small and has now been inspected; future improvements should also be assessed on newly arriving, untouched matches.

- [x] Expand historical data from three seasons to five.
- [x] Build nine chronological prematch features.
- [x] Compare a baseline, logistic regression, and tuned XGBoost.
- [x] Inspect probability calibration and evaluate an adjustment.
- [x] Evaluate on 50 later-season matches.
- [x] Save and reload the calibrated model with matching predictions.
- [x] Export upcoming fixture probabilities and stats.
- [x] Build the responsive website and results-history interface.
- [ ] Deploy the updated website.
- [ ] Automate result retrieval, forecast generation, and site updates.
- [ ] Archive forecasts before kickoff and attach actual results afterward.
- [ ] Track log loss, calibration, and accuracy over a larger future sample.
- [ ] Implement the standalone training/evaluation scripts and pin the environment.
- [ ] Explore season simulations and league-title probabilities.

## Tools

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · joblib · Jupyter · HTML · CSS · JavaScript

Built by **Aren Khachikian** as a learning project connecting machine learning with a lifelong interest in soccer.
