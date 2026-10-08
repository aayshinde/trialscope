# The easy way to get 0.82 AUROC predicting clinical-trial failure (and why it's wrong)

**TL;DR.** On 30,000 real ClinicalTrials.gov trials, a model that only uses information known at a trial's start reaches
**0.71 AUROC** on trials it never saw. Add six fields from the same public record and the score jumps to **0.82**, with
precision-recall more than doubling (AUPRC 0.22 → 0.48). Those six fields are only knowable after the trial has run. A pipeline that includes them
would look excellent and would be useless for deciding anything at launch.

## Why registry data leaks

ClinicalTrials.gov serves only the **current** version of each record. When a trial finishes or stops, the registry overwrites
what was planned with what happened:

| Field | At registration | In today's record |
|---|---|---|
| Enrollment | planned (ESTIMATED) | what actually enrolled (ACTUAL), tiny for failed trials |
| Sites | planned list | final list; sites that never opened or closed early are gone |
| Results posted | no | yes, for trials that ran to completion |
| Publications | none | accumulate for trials that ran long enough |

A model trained on today's record is, in effect, told the outcome in the features. `enrollment_count` alone makes a
terminated trial easy to spot, because it terminated with few patients.

## The experiment

Same cohorts (train ≤ 2015, validate 2016, test 2017-2019), same XGBoost settings, same split. Only the feature set changes.

| Model | AUROC | AUPRC | Failures caught in top 10% of risk |
|---|---|---|---|
| Phase × therapeutic-area historical rate | 0.644 | 0.192 | 24.8% |
| Logistic regression, structured fields | 0.656 | 0.203 | 23.9% |
| **XGBoost, start-time features only** | **0.712** | **0.221** | **27.1%** |
| XGBoost **with six post-start fields** | 0.821 | 0.480 | 49.4% |

The leaky model's top driver after the text embedding is `LEAK:enrollment_count`. The inflated score is the leakage, not skill.
The honest model still beats both baselines overall, and its AUROC is stable across the three test years (0.708 / 0.741 / 0.690 for 2017 / 2018 / 2019).

## How the pipeline stays honest

1. **A feature registry** marks every feature as known-at-start or not. The default model can only see the former.
2. **Out-of-time evaluation.** Train on earlier start years, test on later ones. Recent years are excluded from testing because
   trials that start recently *and already ended* are mostly the ones that stopped early.
3. **Sponsor track record as of each trial's start**, over a trailing 5-year window (DuckDB ASOF joins). A dbt test recomputes it by brute
   force and fails the build if the two disagree.
4. **A dbt snapshot (SCD type 2)** records every change to a trial going forward, since the API has no history endpoint.
5. **The ablation above runs on every training run**, so a leak that sneaks in shows up as a number rather than a surprise.

## Other things that would have flattered the results

- **Overconfident calibration.** Isotonic regression fitted on a 1,000-trial validation year produced a step at exactly 1.0, so unusual
  trials were scored "99% risk" while the highest observed termination rate in any test-set risk bin was about 27%. Platt (sigmoid) scaling fixed it:
  Brier 0.089 → 0.086, highest score 100% → 41%, and predicted vs observed rates now track across all eight risk bins.
- **Demo data in the warehouse.** A synthetic dataset left in the landing zone was loaded next to the real data, which hid a broken
  real-data path (the API had dropped the MeSH branch field the therapeutic-area logic depended on). The fix is in the README.
- **A constant column** (a rare-disease flag that became all-false) made the Cox model's matrix singular. It is dropped when it has no variance.

## What the model can and can't do

It **ranks**: the top 10% of scored trials contain 27% of the eventual failures (2.7× random), and the top 20% contain 47%. It is weaker for
Phase 3/4 trials and for metabolic and mental-health areas (AUROC between 0.48 and 0.57 in those slices). Termination is a rare event driven by
funding, strategy and safety signals that never appear in a registration record. A split-conformal check holds its 90% coverage but returns
"will complete" for almost every trial, so the guarantee is valid and not very informative here. The estimate is a prioritisation aid for
where to look first, not a forecast for any single trial.

## Reproduce

```bash
pip install -r requirements-pipeline.txt
python run_pipeline.py --limit 30000
open results/classifier_report.md      # the table above, plus calibration, backtest and slices
```
