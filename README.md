# Can Machine Learning Find the Best Move?

Predicts the correct move in tactical chess puzzles from the board position
and the candidate move alone, and compares logistic regression and a small
neural network with simple chess heuristics. Data: Lichess puzzle database.

## Setup
pip install -r requirements.txt

## How to run (in this order)
1. `01_get_data.ipynb`: samples 5,000 puzzles (skips the download if
   `data/puzzles_sample.csv` exists; the committed CSV is the one used in the paper)
2. `02_build_candidates.ipynb`: generates all legal moves per puzzle
   -> `data/candidates.csv`
3. `03_models.ipynb`: trains and evaluates the models -> `results/`
4. `04_figures_tables.ipynb`: produces the figures -> `figures/`

Shared code is in `src/chess_features.py`.
Random seeds are fixed. Approximate running time: [X minutes] on [your machine].
No resource beyond the standard workspace is needed.

## Data
Lichess puzzle database (https://database.lichess.org/#puzzles),
released under CC0. A 5,000-puzzle random sample is stored in `data/`.
Cleaning steps are described in the paper (Section 5).

## Approximate running time
Approximate running time (MacBook [M1], [8] GB): the whole pipeline (notebooks
02-04) runs in under one minute. Notebook 01 skips the download if
`data/puzzles_sample.csv` is present; a download from scratch depends on the network
and can take several minutes, and yields a different sample (the committed CSV is the
one used in the paper). No resource beyond the standard workspace is needed.