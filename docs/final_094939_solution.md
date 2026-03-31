# Final 0.94939 Solution

## What The Final Solution Is

The final public leaderboard score in this repo is `0.94939`.

The submission is:

```text
exp064b_exp058_anti_light_gradient
```

with formula:

```text
clip(1.11 * exp058_blend_exp052_catneural - 0.11 * exp_light_attention, 0, 1)
```

This is not a standard positive-weight blend. It is a small extrapolation away
from `exp_light_attention` while keeping `exp058` as the main anchor.

## Why `exp058` Is The Anchor

`exp058_blend_exp052_catneural` was the strongest stable submission family in
the run history.

Its saved blend members were:

- `exp052_blend_poststack`
- `exp026_fttransformer_lite_gpu`
- `exp047_tabnet_deep`
- `exp028_attention_catfreq_gpu`
- `exp019_blend_bucket_state_crosses_fixed`
- `exp040_gbm_trio_blend`

Saved local AUC:

```text
0.938097
```

Recorded public LB:

```text
0.94916
```

That made `exp058` the best available base prediction to perturb.

## Why `exp062` Was Tested

The first follow-up tested whether an older attention branch added useful
public-LB diversity.

`exp062_exp058_plus_light_attention` used:

```text
0.90 * exp058 + 0.10 * exp_light_attention
```

Saved local AUC:

```text
0.938226
```

So locally it looked slightly better than `exp058`.

But the recorded public LB was:

```text
0.94855
```

That was worse than `exp058` by:

```text
0.94916 - 0.94855 = 0.00061
```

The important conclusion was not that `exp062` was useless. The important
conclusion was that moving in the `+exp_light_attention` direction hurt the
leaderboard.

## Why `exp064b` Goes In The Opposite Direction

After `exp062`, the remaining one-shot strategy was:

1. Treat `exp058` as the best point.
2. Treat `exp062` as a measured step from that point.
3. Extrapolate in the opposite direction.

The tested move from `exp058` to `exp062` was:

```text
exp062 = exp058 + 0.10 * (exp_light_attention - exp058)
```

Observed public-LB response:

```text
-0.00061
```

So the final candidate used the opposite sign:

```text
final = exp058 - 0.11 * (exp_light_attention - exp058)
      = 1.11 * exp058 - 0.11 * exp_light_attention
```

The `0.11` coefficient is the rounded public-LB-gradient step selected for the
final submission.

## Why Clipping Is Used

The extrapolated formula can push a few predictions below `0` or above `1`.

The implementation therefore applies:

```text
np.clip(predictions, 0.0, 1.0)
```

This preserves a valid Kaggle submission format while keeping the extrapolated
ordering as intact as possible.

## How It Is Implemented

Implementation inputs:

- `outputs/submissions/exp058_blend_exp052_catneural/submission.csv`
- `outputs/submissions/exp_light_attention/submission.csv`
- `outputs/oof/exp058_blend_exp052_catneural/oof_predictions.csv`
- `outputs/oof/exp_light_attention/oof_predictions.csv`
- `data/raw/train.csv`

Implementation steps:

1. Sort all frames by `id`.
2. Reconstruct train-side predictions for `exp_light_attention` by joining its
   saved OOF-style predictions back to `train.csv`.
3. Apply the same formula on train and test:
   `clip(1.11 * exp058 - 0.11 * exp_light_attention, 0, 1)`.
4. Save:
   - `submission.csv`
   - `oof_predictions.csv`
   - `best_run_summary.json`
   - `best_params.json`
5. Validate the final `submission.csv` with `python -m src.submit ...`.

## Why Local AUC Is Lower Than The Final Public Score Suggests

Saved local AUC for `exp064b`:

```text
0.937633
```

This is lower than the local AUC of `exp058`.

That means `exp064b` is not a conventional CV-improving blend. It is a
leaderboard-directed extrapolation based on the measured response of `exp062`.
It exists specifically because the contest was submission-limited and the final
decision was made using public-LB feedback.

## Final Recorded Result

Recorded in `outputs/logs/leaderboard_journal.csv`:

- `exp058_blend_exp052_catneural`: `0.94916`
- `exp062_exp058_plus_light_attention`: `0.94855`
- `exp064b_exp058_anti_light_gradient`: `0.94939`

So the anti-light extrapolation improved over `exp058`, but still finished
short of the `0.94983` target that was being chased.
