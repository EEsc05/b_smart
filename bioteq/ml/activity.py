"""MHEALTH accelerometer baseline with leave-one-participant-out evaluation.

Run: python -m bioteq.ml.activity --help
"""
import argparse
import hashlib
import json
import platform
import re
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

URL = 'https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip'
SOURCE = 'https://archive.ics.uci.edu/dataset/319/mhealth+dataset'
LABELS = {
    1: 'standing', 2: 'sitting', 3: 'lying', 4: 'walking',
    5: 'climbing_stairs', 6: 'waist_bends', 7: 'arm_elevation',
    8: 'knees_bending', 9: 'cycling', 10: 'jogging', 11: 'running',
    12: 'jumping',
}
CLASSES = list(LABELS)
# Zero-based columns: chest, left ankle, right lower arm acceleration.
COLUMNS = [0, 1, 2, 5, 6, 7, 14, 15, 16]
WINDOW = 100  # 2 seconds at the dataset's 50 Hz sampling rate.
FEATURES = [f'{stat}_{site}_{axis}' for stat in ('mean', 'std', 'rms', 'range')
            for site in ('chest', 'left_ankle', 'right_lower_arm') for axis in 'xyz']


def download(destination):
    """Cache the source ZIP atomically; never extract untrusted archive paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'source.zip'
    if not target.exists():
        temporary = destination / 'source.zip.part'
        try:
            with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Corrupt dataset archive')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def features(acceleration):
    """Fixed ordered features; accepts exactly one finite 100x9 SI-unit window."""
    values = np.asarray(acceleration, dtype=float)
    if values.shape != (WINDOW, 9) or not np.isfinite(values).all():
        raise ValueError('Expected finite acceleration with shape (100, 9), at 50 Hz')
    return np.concatenate([values.mean(axis=0), values.std(axis=0),
                           np.sqrt((values**2).mean(axis=0)), np.ptp(values, axis=0)])


def windows(rows):
    """Non-overlapping windows; discard null/mixed-label and nonfinite windows.

    Labels are used only to select clean evaluation windows, never as features.
    At inference, callers supply a complete window without labels.
    """
    if rows.ndim != 2 or rows.shape[1] != 24:
        raise ValueError('Expected MHEALTH rows with 24 columns')
    for start in range(0, len(rows) - WINDOW + 1, WINDOW):
        block = rows[start:start + WINDOW]
        labels = block[:, 23]
        if labels[0] not in LABELS or not np.all(labels == labels[0]):
            continue
        acceleration = block[:, COLUMNS]
        if np.isfinite(acceleration).all():
            yield features(acceleration), int(labels[0]), acceleration


def load_dataset(archive_path):
    x, y, groups = [], [], []
    example = None
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted((int(m.group(1)), name) for name in archive.namelist()
                         if (m := re.search(r'(?:^|/)mHealth_subject(\d+)\.log$', name)))
        if [subject for subject, _ in members] != list(range(1, 11)):
            raise ValueError('Expected exactly subjects 1 through 10')
        for subject, name in members:
            with archive.open(name) as stream:
                rows = np.loadtxt(stream)
            for vector, label, acceleration in windows(rows):
                x.append(vector)
                y.append(label)
                groups.append(subject)
                if example is None:
                    example = acceleration.tolist()
    if not x:
        raise ValueError('No usable windows')
    return np.asarray(x), np.asarray(y), np.asarray(groups), example


def new_model():
    # Fixed before evaluation. No hyperparameter search on held-out participants.
    return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=2,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1))


def scores(y, predictions, probabilities):
    return {
        'accuracy': accuracy_score(y, predictions),
        'balanced_accuracy': balanced_accuracy_score(y, predictions),
        'macro_f1': f1_score(y, predictions, labels=CLASSES, average='macro', zero_division=0),
        'log_loss': log_loss(y, probabilities, labels=CLASSES),
        'multiclass_brier': float(np.mean(np.sum(
            (probabilities - (y[:, None] == np.asarray(CLASSES)))**2, axis=1))),
    }


def evaluate(x, y, groups):
    """Each participant is held out once; preprocessing is fit on training only."""
    probability = np.zeros((len(y), len(CLASSES)))
    baseline = np.zeros_like(y)
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Participant leakage')
        if set(y[train]) != set(CLASSES):
            raise ValueError('Training fold lacks one or more classes')
        model = new_model().fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])
        predicted = np.asarray(CLASSES)[probability[test].argmax(axis=1)]
        baseline[test] = DummyClassifier(strategy='most_frequent').fit(
            x[train], y[train]).predict(x[test])
        folds.append({'test_subject': int(groups[test][0]),
                      'train_subjects': sorted(map(int, set(groups[train]))),
                      'test_windows': len(test), **scores(y[test], predicted, probability[test])})
        print(f"Held-out participant {groups[test][0]}: accuracy {folds[-1]['accuracy']:.3f}", flush=True)
    predicted = np.asarray(CLASSES)[probability.argmax(axis=1)]
    return {
        'protocol': 'leave_one_subject_out; fixed model; non-overlapping 2-second pure-label windows',
        'pooled_out_of_fold': scores(y, predicted, probability),
        'subject_mean_accuracy': float(np.mean([f['accuracy'] for f in folds])),
        'subject_std_accuracy': float(np.std([f['accuracy'] for f in folds])),
        'majority_baseline_accuracy': accuracy_score(y, baseline),
        'per_class': classification_report(y, predicted, labels=CLASSES,
            target_names=list(LABELS.values()), output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y, predicted, labels=CLASSES).tolist(),
        'class_order': list(LABELS.values()), 'folds': folds,
    }


def predict_window(bundle, acceleration):
    if bundle.get('schema_version') != 1 or bundle.get('feature_names') != FEATURES:
        raise ValueError('Incompatible model feature schema')
    probabilities = bundle['model'].predict_proba([features(acceleration)])[0]
    labels = bundle['model'].classes_
    best = int(labels[np.argmax(probabilities)])
    return {'task': 'mhealth_activity_recognition', 'activity': LABELS[best],
            'probabilities': {LABELS[int(c)]: float(p) for c, p in zip(labels, probabilities)},
            'probability_status': 'uncalibrated_random_forest_votes',
            'scope': 'MHEALTH sensor placement at 50 Hz; no injury, compensation or fatigue inference'}


def train(data_dir='data/mhealth', output_dir='artifacts/activity', report_path='reports/activity_metrics.json'):
    archive = download(data_dir)
    x, y, groups, example = load_dataset(archive)
    report = evaluate(x, y, groups)
    with archive.open('rb') as source_file:
        source_hash = hashlib.file_digest(source_file, 'sha256').hexdigest()
    provenance = {
        'dataset': 'MHEALTH', 'source': SOURCE, 'download_url': URL,
        'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'citation': 'Banos, O., Garcia, R., & Saez, A. (2014). MHEALTH. UCI. DOI:10.24432/C5TW22',
        'source_sha256': source_hash,
        'transformations': '9 accelerometer channels; 100-row non-overlapping windows; exclude null, mixed-label, nonfinite windows; mean/std/RMS/range',
    }
    report.update({'provenance': provenance, 'windows': len(y),
        'windows_per_class': {LABELS[c]: int(sum(y == c)) for c in CLASSES},
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
        'limitations': ['Only 10 participants; no independent external validation.',
            'Pure-label windows exclude transitions and null activity; deployment may perform worse.',
            'Probabilities are uncalibrated, not medical confidence or risk.',
            'No compensation, fatigue, injury, EMG or NIRS labels; no such claims supported.',
            'Out-of-fold scores evaluate the training procedure; final model is refit on all participants.']})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {'schema_version': 1, 'model': new_model().fit(x, y),
              'feature_names': FEATURES, 'provenance': provenance}
    joblib.dump(bundle, output / 'model.joblib')
    (output / 'example_window.json').write_text(json.dumps(example))
    (output / 'example_prediction.json').write_text(json.dumps(predict_window(bundle, example), indent=2)+'\n')
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['pooled_out_of_fold'], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/mhealth')
    parser.add_argument('--output-dir', default='artifacts/activity')
    parser.add_argument('--report', default='reports/activity_metrics.json')
    parser.add_argument('--predict', type=Path, help='JSON 100x9 acceleration array; needs a locally trained model')
    args = parser.parse_args()
    if args.predict:
        # joblib/pickle can execute code: load only your own locally trained artifact.
        bundle = joblib.load(Path(args.output_dir) / 'model.joblib')
        print(json.dumps(predict_window(bundle, json.loads(args.predict.read_text())), indent=2))
    else:
        train(args.data_dir, args.output_dir, args.report)


if __name__ == '__main__':
    main()
