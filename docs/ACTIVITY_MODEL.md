# Activity recognition baseline

This pipeline learns an activity label from wearable accelerometers. It is separate from `MuscleEngine`, which still calculates the existing physiological metrics. The model cannot identify biceps-curl compensation, fatigue, or injury: its dataset does not provide those labels. No unsupported labels are synthesized.

## Data and permission

Source: [UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset).
Attribution: Banos, O., Garcia, R., & Saez, A. (2014). MHEALTH [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5TW22](https://doi.org/10.24432/C5TW22).

UCI publishes the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain attribution, this license link, and descriptions of transformations when sharing derived data. The repository's software license does not replace the data license. The user authorized public-data download and baseline training in this session.

The download is cached locally, excluded from Git, and identified by SHA-256 in each report. Transformations: select accelerometer channels, create non-overlapping windows, exclude unlabeled/mixed windows, calculate features. No raw participant data or trained binaries are committed by default.

## Run locally

```bash
python -m pip install -r requirements-ml.txt
python -m bioteq.ml.activity
python -m bioteq.ml.activity --predict artifacts/activity/example_window.json
```

The first run downloads about 72 MB and evaluates ten participant folds, then fits a final model using all participants. It writes:

- `reports/activity_metrics.json`: out-of-fold metrics, per-class precision/recall/F1/support, confusion matrix, individual participant scores, source attribution/checksum, and package versions.
- `artifacts/activity/model.joblib`: complete preprocessing/classifier bundle.
- `artifacts/activity/example_window.json` and `example_prediction.json`: inference example. This example is from the final model's training data, so its prediction is not evaluation evidence.

Load only trusted joblib files; they use pickle. Regenerate models with the same package versions recorded in the report.

## Input and evaluation contract

Inference expects a finite 100 × 9 array: 2 seconds at 50 Hz, with chest xyz, left-ankle xyz, then right-lower-arm xyz accelerations in m/s². Placement, order, units, and sample rate must match. One sensor or the existing simulated bicep CSV is insufficient; there is no automatic conversion that invents missing sensors. The model returns a label distribution for each window. Forest vote probabilities are uncalibrated and do not represent medical risk. Unseen activities still receive a known label; no out-of-distribution rejection is implemented.

Each channel contributes mean, standard deviation, RMS, and range. A fixed random forest is compared with a training-fold majority baseline. Every participant is held out once, including preprocessing. Neither participant IDs, timestamps, nor labels enter the feature vector. No hyperparameters are selected on test participants.

Accuracy measures the fraction of correctly classified eligible windows. Balanced accuracy averages recall across classes; macro F1 weights classes equally. Log loss and multiclass Brier score assess probability errors (lower is better). The report also includes participant mean and spread; these are not confidence intervals. Pooled windows from one participant are correlated.

This is a small-dataset baseline, not external validation. Excluding transitions and unknown activities makes evaluation easier than continuous deployment. Final-model training scores are not reported as test accuracy. Repetition error and compensation detection metrics remain unavailable because this task does not supply the required reference labels.

## First recorded run

The checked-in `reports/activity_metrics.json` records 3,313 eligible windows from ten participants. Pooled held-out accuracy is 96.23%, balanced accuracy 96.34%, and macro F1 96.40%; the fold-specific majority baseline achieves 8.99% accuracy. Participant accuracy ranges from 85.92% to 100%. These numbers apply only to this activity-recognition protocol. They do not measure curl technique, repetition counting, or physiological assessment.
