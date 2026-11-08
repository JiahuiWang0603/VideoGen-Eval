"""Evaluate two real swimmer clips and compare with unmodified VBench classes."""
import json
import platform
import resource
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from metrics.motion_smoothness import load_model,evaluate_video,official_reference
from pipeline import ROOT,sha,append_history,inspect_video


def main():
    old_paths=[p for folder in ['data','videos','results','metrics'] for p in (ROOT/folder).rglob('*')
               if p.is_file() and '__pycache__' not in str(p)]
    before={str(p.relative_to(ROOT)):sha(p) for p in old_paths}
    history=ROOT/'results/metric_scores.csv'
    history_prefix=history.read_bytes()
    torch.manual_seed(20261109);torch.set_num_threads(4)
    run_id=datetime.now(timezone.utc).strftime('motion_%Y%m%dT%H%M%S%fZ')
    out=ROOT/'results'/run_id;out.mkdir(exist_ok=False)
    start=time.perf_counter();model,source=load_model();load_seconds=time.perf_counter()-start
    reference=official_reference()
    videos=pd.read_csv(ROOT/'data/metadata.csv')
    videos=videos[(videos.dimension=='motion_smoothness') & (videos.prompt=='a person swimming in ocean')].drop_duplicates('video_id')
    if len(videos)!=2 or set(videos.model_name)!={'lavie','modelscope'}:
        raise ValueError('Expected two existing swimmer videos')
    records=[];details=[]
    for r in videos.itertuples():
        print('Evaluating',r.model_name,flush=True)
        row=dict(run_id=run_id,video_id=r.video_id,model_name=r.model_name,metric_name='motion_smoothness',
                 score=None,status='error',error_message='')
        started=time.perf_counter()
        try:
            info=inspect_video(ROOT/r.video_path)
            start=time.perf_counter();score,curve,stats=evaluate_video(model,ROOT/r.video_path);inference_seconds=time.perf_counter()-start
            captured=[]
            original_diff=reference.get_diff
            def capture(a,b):
                value=original_diff(a,b);captured.append(float(value));return value
            reference.get_diff=capture
            start=time.perf_counter();official_score=float(reference.motion_score(str(ROOT/r.video_path)));reference_seconds=time.perf_counter()-start
            reference.get_diff=original_diff
            errors=[x['reconstruction_mae'] for x in curve]
            np.testing.assert_array_equal(errors,captured)
            if score!=official_score:raise AssertionError(f'Official scalar mismatch: {score} vs {official_score}')
            frame_table=pd.DataFrame(curve)
            frame_table['official_mae']=captured
            frame_table['absolute_difference']=np.abs(np.array(errors)-np.array(captured))
            frame_table.to_csv(out/f'{r.video_id}_reconstruction.csv',index=False)
            fig=Figure(figsize=(8,4));ax=fig.subplots()
            ax.plot(frame_table.frame_id,frame_table.reconstruction_mae,marker='o')
            ax.set(title=f'{r.model_name} Motion Smoothness: {score:.8f}',xlabel='Reconstructed original frame index',ylabel='RGB reconstruction MAE (0–255)')
            ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/f'{r.video_id}_reconstruction.png',dpi=160)
            row.update(score=score,status='ok',mean_mae=float(np.mean(errors)),official_score=official_score,
                       official_abs_difference=abs(score-official_score),frames=info['frames'],fps=info['fps'],width=info['width'],height=info['height'])
            details.append(dict(video_id=r.video_id,video_path=r.video_path,video_sha256=sha(ROOT/r.video_path),
                stats=stats,inference_seconds=inference_seconds,reference_seconds=reference_seconds,
                pairwise_mae_exact_parity=True))
        except Exception as exc:
            row['error_message']=f'{type(exc).__name__}: {exc}'
        row['elapsed_seconds']=time.perf_counter()-started
        records.append(row);print(row,flush=True)
    scores=pd.DataFrame(records);scores.to_csv(out/'metric_scores.csv',index=False)
    saved=pd.read_csv(out/'metric_scores.csv')
    assert len(saved)==2
    for a,b in zip(records,saved.itertuples()):
        if a['status']=='ok':assert abs(a['score']-b.score)<1e-15
    assert all(sha(ROOT/p)==h for p,h in before.items()),'Preexisting artifact modified'
    manifest=dict(run_id=run_id,environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
        opencv=cv2.__version__,numpy=np.__version__,pandas=pd.__version__,mps_available=torch.backends.mps.is_available()),
        device='cpu',threads=4,seed=20261109,model=source,model_load_seconds=load_seconds,
        peak_process_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        preprocessing=dict(channels='RGB',range='float32 [0,1]',frames='all decoded; even-index endpoints',
            padding='replicate to int(16/scale) divisor',cpu_scale='exact upstream formula; 1.0 for both current clips',
            output='unpad; multiply 255; clip; astype uint8',fps_resampling=False,crop=False),
        score_formula='(255 - mean RGB uint8 reconstruction MAE over odd interpolated frames) / 255',
        niters=1,interpolation_time=.5,details=details,
        timing='elapsed_seconds includes validation, own inference, official second inference and artifact output',
        reference_validation='AST-extracted unmodified upstream FrameProcess/MotionSmoothness classes; separate official model instance; exact per-pair MAE and scalar parity',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'metrics/motion_smoothness.py',Path(__file__).resolve(),ROOT/'tests/test_motion_smoothness.py']},
        preserved_artifact_sha256=before,limitations=['CPU only; MPS/CUDA numerical parity not tested','Full VBench CLI/distributed aggregation not tested','Two videos do not establish model ranking'])
    (out/'runs.json').write_text(json.dumps([manifest],indent=2)+'\n')
    append_history(history,records)
    assert history.read_bytes().startswith(history_prefix)
    report=['# Motion Smoothness official AMT-S reproduction','',
            'CPU, unmodified pinned AMT-S network, official weight SHA-256 verified. No video modifications.',
            '| Model | Score | Mean reconstruction MAE | Official difference | Status |',
            '|---|---:|---:|---:|---|']
    for row in records:
        report.append(f"| {row['model_name']} | {row.get('score')} | {row.get('mean_mae')} | {row.get('official_abs_difference')} | {row['status']} |")
    report += ['', 'Each 16-frame clip uses endpoints 0,2,...,14 and compares predictions at frames 1,3,...,13 (7 pairs). Frame 15 is not scored.',
               'Per-pair RGB uint8 MAEs and final scalars match official classes exactly when status is ok.',
               'This is interpolation reconstruction consistency, not action realism, identity accuracy or overall model quality.',
               'CPU parity established; GPU/MPS, full CLI and distributed pipeline remain unverified. Historical score bytes were preserved; new rows only appended.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print('Outputs:',out)
    if any(r['status']!='ok' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
