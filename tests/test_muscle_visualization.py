import numpy as np
import pytest
pytest.importorskip("matplotlib")
from bioteq.schemas.sensor import SensorReading
from bioteq.visualization.muscles import emg_windows


def row(t, value):
    return SensorReading(timestamp_ms=t, session_id='s', exercise='bicep_curl', channel='emg',
                         value=value, unit='mV', muscle_site='biceps', side='right', source='simulated')


def test_rms_gaps_and_reference_normalization():
    rows = [row(0,3), row(500,-4), row(2000,0),row(3000,None)]
    keys, edges, matrix = emg_windows(rows)
    assert keys == [('biceps','right')]
    assert matrix[0,0] == pytest.approx(np.sqrt(12.5))
    assert np.isnan(matrix[0,1])
    assert matrix[0,2] == 0  # Zero is a measurement, not missing data.
    assert np.isnan(matrix[0,3])
    assert edges.tolist() == [0,1000,2000,3000,4000]
    _,_,normalized = emg_windows(rows,references={keys[0]:5},normalized=True)
    assert normalized[0,0] == pytest.approx(20*np.sqrt(12.5))
    assert np.isnan(emg_windows(rows,normalized=True)[2]).all()
    with pytest.raises(ValueError):
        emg_windows(rows, references={keys[0]:float('nan')})
    with pytest.raises(ValueError):
        emg_windows(rows, bin_ms=0)
