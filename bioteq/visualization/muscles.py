"""Schematic measured-muscle maps and time-binned EMG RMS, without interpolation."""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, Ellipse, Polygon


def emg_windows(readings, bin_ms=1000, references=None, normalized=False):
    """Return muscle keys, bin edges (ms), and muscle-by-time RMS values.

    References are RMS mV by (muscle_site, side); normalization is percent of
    reference, not automatically percent MVC. Missing bins/references stay NaN.
    """
    if not isinstance(bin_ms, int) or bin_ms <= 0:
        raise ValueError('bin_ms must be a positive integer')
    references = references or {}
    if any(not np.isfinite(v) or v <= 0 for v in references.values()):
        raise ValueError('References must be finite positive RMS mV')
    rows = [r for r in readings if r.channel == 'emg' and r.unit == 'mV']
    if not rows:
        raise ValueError('No EMG readings in mV available')
    keys = sorted({(r.muscle_site, r.side) for r in rows}, key=str)
    start = min(r.timestamp_ms for r in rows)
    stop = max(r.timestamp_ms for r in rows)
    edges = start + np.arange((stop-start)//bin_ms + 2)*bin_ms
    grouped = defaultdict(list)
    for r in rows:
        if r.value is not None and np.isfinite(r.value):
            grouped[((r.muscle_site, r.side), (r.timestamp_ms-start)//bin_ms)].append(r.value)
    matrix = np.full((len(keys), len(edges)-1), np.nan)
    for i, key in enumerate(keys):
        for j in range(len(edges)-1):
            values = grouped.get((key, j), [])
            if values and (not normalized or key in references):
                rms = np.sqrt(np.mean(np.square(values)))
                matrix[i, j] = 100*rms/references[key] if normalized else rms
    return keys, edges, matrix


def plot_muscles(readings, bin_ms=1000, references=None, normalized=False):
    """Return body-map and time-heatmap figures using one shared color scale.

    The schematic supports biceps/trapezius only; other sites remain in the time
    chart and summary. Whole-session RMS weights samples, not bins.
    """
    readings = list(readings)
    keys, edges, matrix = emg_windows(readings, bin_ms, references, normalized)
    references = references or {}
    summaries = {}
    for key in keys:
        values = [r.value for r in readings if r.channel == 'emg' and r.unit == 'mV'
                  and (r.muscle_site, r.side) == key and r.value is not None and np.isfinite(r.value)]
        if values and (not normalized or key in references):
            rms = float(np.sqrt(np.mean(np.square(values))))
            summaries[key] = 100*rms/references[key] if normalized else rms
    finite = matrix[np.isfinite(matrix)]
    maximum = max([*finite, *summaries.values(), 0.0]) or 1.0
    norm = colors.Normalize(0, maximum)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('#d1d5db')
    unit = '% of supplied RMS reference' if normalized else 'EMG RMS (mV)'
    fig, axes = plt.subplots(1, 2, figsize=(9, 7), layout='constrained')
    mapped = set()
    for ax, view in zip(axes, ['Front', 'Back']):
        ax.set(xlim=(-1.7, 1.7), ylim=(-0.3, 6.6), aspect='equal')
        ax.axis('off')
        ax.set_title(view)
        gray = '#d1d5db'
        ax.add_patch(Circle((0, 5.85), .42, facecolor=gray, edgecolor='white'))
        ax.add_patch(Polygon([(-.7,5.2),(.7,5.2),(.5,3.1),(-.5,3.1)], facecolor=gray, edgecolor='white'))
        for sign in [-1,1]:
            ax.add_patch(Ellipse((sign*.95,4.25), .48, 2.0, angle=sign*12, facecolor=gray, edgecolor='white'))
            ax.add_patch(Ellipse((sign*.32,1.65), .55, 3.0, facecolor=gray, edgecolor='white'))
        # Front view: subject's right is on viewer's left. Back view is reversed.
        for side, sign in [('right', -1 if view == 'Front' else 1), ('left', 1 if view == 'Front' else -1)]:
            site = 'biceps' if view == 'Front' else 'trapezius'
            key = (site, side)
            value = summaries.get(key)
            color = cmap(norm(value)) if value is not None else gray
            if site == 'biceps':
                patch = Ellipse((sign*.9,4.6), .35, .8, angle=sign*12, facecolor=color, edgecolor='#6b7280')
            else:
                patch = Polygon([(sign*.05,5.25),(sign*.7,5.05),(sign*.45,4.65),(sign*.05,4.4)], facecolor=color, edgecolor='#6b7280')
            ax.add_patch(patch)
            label = f'{side} {site}\n' + (f'{value:.2f}' if value is not None else 'unavailable')
            ax.text(sign*.8, 2.9 if site=='biceps' else 3.9, label, ha='center', fontsize=9)
            mapped.add(key)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), ax=axes, shrink=.6, label=unit)
    fig.suptitle('Measured muscle signals • schematic, whole-session RMS')
    fig.text(.5,.01,'Gray = unmeasured / unavailable; position is schematic, not anatomical localization.',ha='center',fontsize=9)
    heat, ax = plt.subplots(figsize=(12, max(3, .5*len(keys)+1.5)), layout='constrained')
    mesh = ax.pcolormesh(edges/1000, np.arange(len(keys)+1), np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, shading='flat')
    ax.set(yticks=np.arange(len(keys))+.5, yticklabels=[f'{site} ({side or "unspecified side"})' for site,side in keys],
           xlabel='Timestamp (seconds)', title=f'Muscle signals over time • {bin_ms/1000:g}-second RMS bins')
    ax.invert_yaxis()
    heat.colorbar(mesh, ax=ax, label=unit)
    unsupported = [key for key in keys if key not in mapped]
    return fig, heat, {'unit': unit, 'session_rms': summaries, 'unmapped_sites': unsupported,
                      'note': 'Raw amplitudes are not comparable effort across muscles. Gray bins are unavailable; no interpolation.'}
