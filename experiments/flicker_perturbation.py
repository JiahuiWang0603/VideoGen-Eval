"""Deterministic brightness perturbations; no video encoding or weight downloads."""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from metrics.temporal_flickering import get_frames, mae_seq, score_frames

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'Original': 0, 'Mild Flicker': 8, 'Strong Flicker': 32, 'Control': 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rgb(path):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in get_frames(path)]


def perturb(frames, amplitude):
    if amplitude < 0 or amplitude > 255:
        raise ValueError('amplitude must be between 0 and 255')
    # Modify only odd-index frames; all channels receive the same additive offset.
    return [np.clip(f.astype(np.int16) + (amplitude if i % 2 else 0), 0, 255).astype(np.uint8)
            for i, f in enumerate(frames)]


def evaluate(frames):
    records, curves = [], []
    for group, amplitude in GROUPS.items():
        start = time.perf_counter()
        changed = perturb(frames, amplitude)
        errors = mae_seq(changed)
        score = score_frames(changed)
        clipped = sum(int(np.count_nonzero(f.astype(np.int16) + amplitude > 255))
                      for i, f in enumerate(frames) if i % 2)
        eligible = sum(f.size for i, f in enumerate(frames) if i % 2)
        records.append(dict(group=group, amplitude=amplitude, score=score,
            mean_mae=float(errors.mean()), elapsed_seconds=time.perf_counter()-start,
            status='ok', pixels_unchanged=all(np.array_equal(a,b) for a,b in zip(frames,changed)),
            clipped_channel_fraction=clipped/eligible if eligible else 0))
        curves.extend(dict(group=group, frame_from=i, frame_to=i+1, mae=float(v))
                      for i,v in enumerate(errors))
    scores = {r['group']: r['score'] for r in records}
    checks = dict(control_equals_original=scores['Control']==scores['Original'],
        mild_lower_than_original=scores['Mild Flicker']<scores['Original'],
        strong_lower_than_mild=scores['Strong Flicker']<scores['Mild Flicker'])
    return records, curves, checks


def save_tables(output, results, curves):
    # Exclusive creation makes accidental overwriting an error.
    pd.DataFrame(results).to_csv(output/'perturbation_results.csv',index=False, mode='x')
    pd.DataFrame(curves).to_csv(output/'frame_mae_curves.csv',index=False, mode='x')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261109)
    args = parser.parse_args()
    np.random.seed(args.seed)
    protected = [p for folder in ['videos','data','results','metrics']
                 for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    protected.append(ROOT/'evaluate_video.py')
    before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    run_id = datetime.now(timezone.utc).strftime('flicker_perturbation_%Y%m%dT%H%M%S%fZ')
    output = ROOT/'results'/run_id
    output.mkdir(exist_ok=False)
    metadata = pd.read_csv(ROOT/'data/metadata.csv')
    selected = metadata[(metadata.dimension=='temporal_flickering') &
                        (metadata.prompt=='In a still frame, a stop sign')].drop_duplicates('video_id')
    if set(selected.model_name) != {'lavie','modelscope'} or len(selected)!=2:
        raise ValueError('Expected one stop-sign video for each of lavie and modelscope')
    results, curves, validations, inputs = [], [], {}, []
    for r in selected.itertuples():
        start = time.perf_counter()
        frames = read_rgb(ROOT/r.video_path)
        decode_seconds = time.perf_counter()-start
        rows, series, checks = evaluate(frames)
        for row in rows:
            row.update(run_id=run_id,video_id=r.video_id,model_name=r.model_name,
                       metric_name='temporal_flickering',decode_seconds=decode_seconds)
        for row in series:
            row.update(video_id=r.video_id,model_name=r.model_name)
        results.extend(rows); curves.extend(series); validations[r.model_name]=checks
        inputs.append(dict(video_id=r.video_id,path=r.video_path,sha256=sha(ROOT/r.video_path),
                           frames=len(frames),shape=list(frames[0].shape)))
    save_tables(output,results,curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(results); curve=pd.DataFrame(curves)
    fig,ax=plt.subplots(figsize=(8,4))
    pivot=table.pivot(index='group',columns='model_name',values='score').reindex(GROUPS)
    pivot.plot.bar(ax=ax,rot=0)
    ax.set(ylabel='Temporal Flickering score (higher = less change)',ylim=(0,1.02))
    fig.tight_layout(); fig.savefig(output/'score_comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,sharey=True)
    for ax,model in zip(axes,['lavie','modelscope']):
        for group in GROUPS:
            d=curve[(curve.model_name==model)&(curve.group==group)]
            ax.plot(d.frame_to,d.mae,label=group,linestyle='--' if group=='Control' else '-')
        ax.set(title=model,ylabel='Color MAE (0–255)'); ax.legend(); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Destination frame index (zero-based)')
    fig.tight_layout(); fig.savefig(output/'frame_mae_curves.png',dpi=160); plt.close(fig)
    manifest=dict(run_id=run_id,seed=args.seed,randomness='none; deterministic perturbation, seed reserved',
        parameters=dict(groups=GROUPS,operation='odd zero-based frames: RGB + amplitude; clip [0,255]; uint8',
                        original='copy unchanged',control='same copy/clip/cast path with amplitude zero',
                        transcode=False,resize=False,crop=False,frame_sampling='all decoded frames'),
        environment=dict(python=sys.version,platform=platform.platform(),opencv=cv2.__version__,
                         numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        timing='elapsed_seconds: perturbation and metric; decode_seconds: shared per-video decode',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'metrics/temporal_flickering.py',ROOT/'tests/test_flicker_perturbation.py']},
        upstream=json.loads((ROOT/'vendor/temporal_flickering_source.json').read_text()),
        inputs=inputs,validation=validations,existing_files_sha256=before)
    assert all(sha(ROOT/p)==h for p,h in before.items()), 'Existing artifact changed'
    manifest['existing_artifacts_unchanged']=True
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Automated Temporal Flickering perturbation experiment','',
        'No manual frame inspection or automatic static filter was used in this experiment.',
        'Existing static-prompt selection is held fixed. Perturbations are in-memory RGB only.',
        'Odd zero-based frames receive +8 (Mild) or +32 (Strong) on all channels; values saturate at 255.',
        'Control follows the same transformation path with amplitude zero. No random perturbation is used.', '',
        '| Model | Group | Score | Mean MAE | Clipped channel fraction |', '|---|---|---:|---:|---:|']
    lines += [f"| {r['model_name']} | {r['group']} | {r['score']:.10f} | {r['mean_mae']:.6f} | {r['clipped_channel_fraction']:.6f} |" for r in results]
    lines += ['', 'Validation: '+json.dumps(validations), '',
        'If an ordering check fails, inspect saturation, existing alternating brightness, or cancellation with the applied offset. Do not label it passed.',
        'This experiment tests formula sensitivity to these additive alternating offsets and zero-offset invariance.',
        'It does not validate perceptual quality, motion robustness, official static filtering, full VBench reproduction, or overall model ranking.',
        'Clipping and existing motion/compression can affect the response. These are two fixed clips, not independent statistical replicates.',
        'All pre-existing data, metrics and results were hash-checked unchanged. See manifest.json for environment, inputs and code hashes.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    loaded=pd.read_csv(output/'perturbation_results.csv')
    assert len(loaded)==8 and np.allclose(loaded.score,table.score,rtol=0,atol=1e-15)
    print(table[['model_name','group','score','mean_mae','clipped_channel_fraction']].to_string(index=False))
    print(json.dumps(validations)); print(output)
    if not all(all(c.values()) for c in validations.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
