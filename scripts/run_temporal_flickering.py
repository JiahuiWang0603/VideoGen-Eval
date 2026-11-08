"""Run only the two selected static-prompt videos; no weight downloads."""
import ast
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics.temporal_flickering import get_frames, mae_seq, score_frames


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_core():
    # Execute only five audited numerical/decode functions, excluding imports
    # of VBench distributed infrastructure and its optional ML dependencies.
    tree = ast.parse((ROOT / 'vendor/temporal_flickering.py.txt').read_text())
    names = {'get_frames', 'mae_seq', 'calculate_mae', 'cal_score'}
    subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[])
    env = {'cv2': cv2, 'np': np}
    exec(compile(subset, 'official_temporal_flickering_core', 'exec'), env)
    return env


def main():
    protected = [ROOT / 'data/metadata.csv', ROOT / 'evaluate_video.py']
    before = {str(p): digest(p) for p in protected}
    metadata = pd.read_csv(ROOT / 'data/metadata.csv')
    videos = metadata.drop_duplicates('video_id').copy()
    provenance, tasks = [], []
    for r in videos.itertuples():
        static = r.prompt == 'In a still frame, a stop sign'
        provenance.append(dict(video_id=r.video_id, model_name=r.model_name,
            video_path=r.video_path, source_archive_url=r.source, source_suite='per_dimension',
            source_subfolder='temporal_flickering' if static else 'subject_consistency',
            provenance_evidence='user_manual_extraction_context; archive_member_path_not_independently_verified',
            video_sha256=digest(ROOT / r.video_path)))
        for metric in (['temporal_flickering'] if static else ['subject_consistency', 'motion_smoothness']):
            tasks.append(dict(video_id=r.video_id, metric_name=metric,
                protocol_id='vbench_color_mae_static_manual_v1' if static else 'planned_not_executed',
                selection_reason='static_prompt_and_manual_frame_review' if static else 'visible_swimmer_and_water_motion'))
    pd.DataFrame(provenance).to_csv(ROOT / 'data/video_provenance.csv', index=False)
    pd.DataFrame(tasks).to_csv(ROOT / 'data/evaluation_tasks.csv', index=False)
    now = datetime.now(timezone.utc)
    run_id = now.strftime('tf_%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'results' / run_id
    output.mkdir(parents=True)
    reference = official_core()
    records = []
    for r in videos.itertuples():
        if r.prompt != 'In a still frame, a stop sign':
            continue
        started = time.perf_counter()
        record = dict(run_id=run_id, video_id=r.video_id, model_name=r.model_name,
                      metric_name='temporal_flickering', score=None, status='error', error_message='')
        try:
            path = ROOT / r.video_path
            frames = get_frames(path)
            sequence = mae_seq(frames)
            score = score_frames(frames)
            official_score = reference['cal_score'](str(path))
            official_sequence = reference['mae_seq'](reference['get_frames'](str(path)))
            np.testing.assert_array_equal(sequence, official_sequence)
            if score != official_score:
                raise AssertionError('Official scalar mismatch')
            fps_cap = cv2.VideoCapture(str(path))
            fps = fps_cap.get(cv2.CAP_PROP_FPS)
            fps_cap.release()
            curve = pd.DataFrame(dict(frame_from=np.arange(len(sequence)),
                frame_to=np.arange(1, len(sequence)+1), time_seconds=np.arange(1,len(sequence)+1)/fps,
                mae_color_0_255=sequence, normalized_change=sequence/255.0))
            curve.to_csv(output / f'{r.video_id}_curve.csv', index=False)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(curve.frame_to, sequence, marker='o')
            ax.set(xlabel='Destination frame index (zero-based)', ylabel='Color MAE (0–255)',
                   title=f'{r.model_name}: stop sign; score={score:.8f}')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(output / f'{r.video_id}_curve.png', dpi=160)
            plt.close(fig)
            record.update(score=score, status='ok', mean_mae=float(sequence.mean()),
                          official_score=official_score, official_abs_difference=abs(score-official_score),
                          frames=len(frames), fps=fps, width=frames[0].shape[1], height=frames[0].shape[0])
        except Exception as exc:
            record['error_message'] = f'{type(exc).__name__}: {exc}'
        record['elapsed_seconds'] = time.perf_counter()-started
        records.append(record)
    scores_path = ROOT / 'results/metric_scores.csv'
    scores = pd.DataFrame(records)
    previous = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    pd.concat([previous, scores], ignore_index=True).to_csv(scores_path,index=False)
    run = dict(run_id=run_id, started_at_utc=now.isoformat(),
        environment=dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                         opencv=cv2.__version__, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
        upstream=json.loads((ROOT / 'vendor/temporal_flickering_source.json').read_text()),
        preprocessing=dict(decoder='OpenCV VideoCapture', channels='BGR color, all three channels',
            subtraction_dtype='float32', resize=False, crop=False, resample_fps=False,
            frame_selection='all decoded frames in order', normalization='(255 - mean(pairwise_color_MAE)) / 255',
            static_filter='manual prompt/frame review only; official automated static filter NOT run'),
        timing_scope='decode, metric, upstream parity check, curve CSV and PNG; excludes setup',
        weights='none', artifact_directory=str(output.relative_to(ROOT)),
        implementation_sha256=digest(ROOT/'metrics/temporal_flickering.py'),
        metadata_sha256=before[str(ROOT/'data/metadata.csv')],
        validation='exact scalar and per-transition equality with AST-extracted upstream core; not full VBench pipeline')
    runs_path = ROOT/'results/runs.json'
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append(run)
    runs_path.write_text(json.dumps(runs,indent=2,ensure_ascii=False)+'\n')
    assert all(digest(p)==before[str(p)] for p in protected)
    print(scores.to_string(index=False))
    if any(r['status'] != 'ok' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
