import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from bioteq.ml.activity import features, windows, predict_window, FEATURES


def test_windows_exclude_transitions_null_and_nonfinite():
    rows = np.ones((500, 24))
    rows[:, 23] = 4
    rows[100:200, 23] = 0
    rows[250:300, 23] = 5
    rows[300, 0] = np.nan
    result = list(windows(rows))
    assert len(result) == 2
    assert [label for _, label, _ in result] == [4, 4]
    assert result[0][0].shape == (36,)


def test_feature_contract_and_probabilities(tmp_path):
    import joblib
    first = np.ones((100, 9))
    second = first * 2
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        [features(first), features(second)], [1, 2])
    bundle = {'schema_version': 1, 'model': model, 'feature_names': FEATURES}
    joblib.dump(bundle, tmp_path / 'model.joblib')
    output = predict_window(joblib.load(tmp_path / 'model.joblib'), first)
    assert sum(output['probabilities'].values()) == pytest.approx(1)
    assert set(output['probabilities']) == {'standing', 'sitting'}
    assert output['probability_status'].startswith('uncalibrated')
    for bad in (np.ones((99, 9)), np.full((100, 9), np.nan)):
        with pytest.raises(ValueError):
            predict_window(bundle, bad)


def test_evaluation_keeps_participants_separate(monkeypatch):
    import bioteq.ml.activity as activity
    from sklearn.dummy import DummyClassifier
    # Cheap test of the real splitting/reporting path, not model performance.
    monkeypatch.setattr(activity, 'new_model', lambda: DummyClassifier(strategy='prior'))
    y = np.tile(np.arange(1, 13), 3)
    groups = np.repeat([1, 2, 3], 12)
    result = activity.evaluate(np.ones((36, 36)), y, groups)
    for fold in result['folds']:
        assert fold['test_subject'] not in fold['train_subjects']
        assert fold['test_windows'] == 12
    assert np.asarray(result['confusion_matrix']).sum() == 36
