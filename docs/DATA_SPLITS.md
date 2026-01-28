# Data Splits and Evaluation

## Why two split views

There is no universal rule that you must use two splits. In applied ML, it is common to report:

- Cross-validation for stability across many train/test partitions
- A single holdout split for a clean, easy-to-explain train/test example

This repo keeps GroupKFold as the primary evaluation and adds a 70/30 holdout for Stage 3/4 to satisfy supervisor guidance.

## Strict evaluation (primary)

- Method: GroupKFold by race
- Folds: 5 (approx 80/20 per fold)
- Goal: prevent leakage across races and get stable metrics

## Holdout evaluation (secondary, Stage 3/4)

- Method: GroupShuffleSplit by race
- Split: 70/30 (train/test)
- Goal: one clean, explainable split on unseen races

## Notes

- Group-based splits are required because rows within a race are correlated.
- Using random row splits would leak information across laps in the same race.
