# League of Legends Esports Match Predictor

A logistic regression model that predicts professional League of Legends match outcomes using historical team performance data from OraclesElixir.

## How It Works

1. **Team profiles** — Raw match data is aggregated into per-team stat profiles (gold diff, XP diff, CS diff, kill diff, objective rates, game length, CKPM). Recent games are weighted more heavily using exponential decay with a 180-day half-life, and international games are boosted 3x when predicting cross-region matchups.

2. **Region weights** — Each region's win rate at international events (Worlds, MSI, EWC) is used to derive a normalized strength multiplier applied to every prediction.

3. **Prediction** — For each matchup, 10,000 Monte Carlo simulations sample from each team's stat distributions (mean ± std), scale the features, and feed them into the logistic regression model via `predict_proba`. Simulations are run twice (swapping blue/red side) and averaged to remove side bias. The result is a win probability for each team.

4. **Win-rate control** — A rudimentary baseline that predicts the team with the higher cumulative pre-game win rate (computed chronologically with no future leakage). Used to benchmark how much `predict_matchup` improves over "pick the team that's been winning more."

## Data

- **Source:** [OraclesElixir](https://oracleselixir.com/)
- **Coverage:** January 2025 – June 21, 2026
- **Refresh:** Re-download the yearly CSVs from OraclesElixir and replace the files in `data/` to update the model with newer results.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Backtest predict_matchup vs win-rate control, then predict a sample matchup
# (edit sample_a / sample_b at the bottom of main.py to change teams)
python main.py
```

Running `main.py` prints region weights, win-rate control accuracy, `predict_matchup` accuracy (with per-game international scaling on the test set), a side-by-side summary, and sample live matchup predictions from both. Test evaluation uses an 80/20 random holdout split with profiles, weights, and model fit on training games only.

### Model vs control (80/20 holdout)

| Predictor | Accuracy |
|---|---|
| Win-rate control | 59.89% |
| predict_matchup | 61.13% |
| Delta | +1.25 pp |

International test games use intl-boosted profiles and region weights; domestic test games do not.

```bash
# Run the MSI 2026 Play-in bracket predictor
python predict_playin.py

# Run the MSI 2026 Main Event bracket predictor
python predict_main_event.py
```

See [PLAYIN_RESULTS.md](PLAYIN_RESULTS.md) for MSI 2026 Play-in predictions and results.

## MSI 2026 Main Event Predictions

### Upper Bracket Round 1

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| Bilibili Gaming vs T1 | Bilibili Gaming | 53.2% |
| LYON vs FURIA | FURIA | 59.8% |
| Hanwha Life Esports vs Team Secret Whales | Hanwha Life Esports | 63.6% |
| G2 Esports vs Top Esports | G2 Esports | 67.7% |

### Lower Bracket Round 1

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| T1 vs LYON | T1 | 66.0% |
| Team Secret Whales vs Top Esports | Team Secret Whales | 65.2% |

### Upper Bracket Round 2

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| Bilibili Gaming vs FURIA | Bilibili Gaming | 62.2% |
| Hanwha Life Esports vs G2 Esports | Hanwha Life Esports | 61.8% |

### Lower Bracket Round 2

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| FURIA vs Team Secret Whales | FURIA | 50.2% |
| T1 vs G2 Esports | T1 | 56.0% |

### Upper Bracket Round 3

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| Hanwha Life Esports vs Bilibili Gaming | Hanwha Life Esports | 52.7% |

### Lower Bracket Round 3

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| T1 vs FURIA | T1 | 57.5% |

### Lower Bracket Finals

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| Bilibili Gaming vs T1 | Bilibili Gaming | 53.5% |

### Grand Finals

| Matchup | Predicted Winner | Confidence |
|---|---|---|
| Hanwha Life Esports vs Bilibili Gaming | Hanwha Life Esports | 52.9% |
