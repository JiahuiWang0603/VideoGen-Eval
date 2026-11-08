"""Unified local MP4 evaluation. Existing metric functions are reused unchanged."""
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from metrics import temporal_flickering as tf
from metrics import subject_consistency as sc
from metrics import motion_smoothness as ms

ROOT = Path(__file__).resolve().parent
METRICS = ('temporal_flickering', 'subject_consistency', 'motion_smoothness')
MODEL_LOCK = threading.Lock()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    count = 0
    shapes = set()
    try:
        if not cap.isOpened():
            raise ValueError('无法打开视频，请检查文件和编解码器。')
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError('解码得到空帧。')
            count += 1
            shapes.add(frame.shape)
    finally:
        cap.release()
    if count < 2:
        raise ValueError('至少需要两帧可解码画面。')
    if expected > 0 and count < expected:
        raise ValueError(f'解码不完整：读到 {count} 帧，容器报告 {expected} 帧。')
    if len(shapes) != 1:
        raise ValueError('帧尺寸不一致。')
    height, width, _ = shapes.pop()
    return dict(frames=count, fps=fps, width=width, height=height,
                reported_frames=expected, duration_seconds=count/fps if fps > 0 else None)


@lru_cache(maxsize=1)
def get_subject_model():
    weights = ROOT / 'models/dino/dino_vitbase16_pretrain.pth'
    source = json.loads((ROOT / 'models/dino/source.json').read_text())
    if not weights.is_file():
        raise FileNotFoundError('缺少本地 DINO 权重；不会自动下载。')
    if sha(weights) != source['weight_sha256']:
        raise ValueError('DINO 权重 SHA-256 与记录不一致。')
    torch.set_num_threads(4)
    return sc.load_model(ROOT/'vendor/dino', weights), source


@lru_cache(maxsize=1)
def get_motion_model():
    torch.set_num_threads(4)
    return ms.load_model()


def append_history(path, rows):
    """Append within existing CSV schema, preserving every historical byte."""
    path = Path(path)
    with path.open('a+', newline='', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            header = next(csv.reader(handle), None)
            if header is None:
                header = list(rows[0])
                handle.seek(0, os.SEEK_END)
                csv.DictWriter(handle, fieldnames=header).writeheader()
            else:
                handle.seek(0, os.SEEK_END)
                if handle.tell():
                    # Existing project CSVs end in newline; also handle external CSVs.
                    with path.open('rb') as check:
                        check.seek(-1, os.SEEK_END)
                        if check.read(1) not in (b'\n', b'\r'):
                            handle.write('\n')
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction='ignore')
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluate(video_path, model_name='', prompt='', video_id='', metrics=METRICS,
             results_root=None):
    path = Path(video_path).expanduser().resolve()
    if path.suffix.lower() != '.mp4' or not path.is_file():
        raise ValueError('请选择一个存在的本地 MP4 文件。')
    metrics = list(dict.fromkeys(metrics))
    if not metrics or set(metrics) - set(METRICS):
        raise ValueError('请选择至少一个受支持的指标。')
    root = Path(results_root) if results_root is not None else ROOT/'results'
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('pipeline_%Y%m%dT%H%M%S%fZ_') + uuid4().hex[:8]
    out = root/run_id
    out.mkdir(exist_ok=False)
    snapshot = out/'input.mp4'
    shutil.copyfile(path, snapshot)
    input_hash = sha(snapshot)
    video_id = str(video_id).strip() or input_hash[:16]
    warnings = []
    if 'temporal_flickering' in metrics:
        warnings.append('Temporal Flickering 适用于静态内容。本次未运行官方静态筛选；动态视频的正常运动也会降低得分，不能据此确定存在闪烁缺陷。')
    records, plots, curve_paths = [], [], []
    info, decode_error = {}, None
    try:
        info = inspect_video(snapshot)
    except Exception as exc:
        decode_error = f'{type(exc).__name__}: {exc}'
    model_info = {}
    metric_parameters = {}
    for metric in metrics:
        started = time.perf_counter()
        row = dict(run_id=run_id, video_id=video_id, model_name=str(model_name),
                   prompt=str(prompt), metric_name=metric, score=None, status='error', error_message='',
                   frames=info.get('frames'), fps=info.get('fps'), width=info.get('width'), height=info.get('height'))
        try:
            if decode_error:
                raise ValueError(decode_error)
            if metric == 'temporal_flickering':
                frames = tf.get_frames(snapshot)
                values = tf.mae_seq(frames)
                row['score'] = tf.score_frames(frames)
                row['mean_mae'] = float(values.mean())
                curve = pd.DataFrame({'frame_id': np.arange(1,len(frames)), 'color_mae': values})
                ylabel = 'Color MAE (0–255)'
            elif metric == 'subject_consistency':
                with MODEL_LOCK:
                    model, source = get_subject_model()
                    model_info[metric] = source
                    score, values, _ = sc.evaluate_video(model, snapshot)
                row['score'] = score
                curve = pd.DataFrame(values)
                ylabel = 'Feature cosine similarity'
            else:
                with MODEL_LOCK:
                    model, source = get_motion_model()
                    model_info[metric] = source
                    score, values, stats = ms.evaluate_video(model, snapshot)
                row['score'] = score
                row['mean_mae'] = float(np.mean([v['reconstruction_mae'] for v in values]))
                metric_parameters[metric] = stats
                curve = pd.DataFrame(values)
                ylabel = 'RGB reconstruction MAE (0–255)'
            if not np.isfinite(row['score']):
                raise ValueError('指标产生非有限数值。')
            csv_path = out/f'{metric}_curve.csv'
            curve.to_csv(csv_path,index=False)
            curve_paths.append(str(csv_path))
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8,4))
            ax = fig.subplots()
            plot_columns = ['reconstruction_mae'] if metric == 'motion_smoothness' else list(curve.columns[1:])
            for column in plot_columns:
                ax.plot(curve.frame_id, curve[column], label=column)
            ax.set(xlabel='Frame index (zero-based)',ylabel=ylabel,title=metric)
            ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout()
            png = out/f'{metric}_curve.png'
            fig.savefig(png,dpi=150)
            plots.append(str(png))
            row['status'] = 'ok'
        except Exception as exc:
            row.update(score=None, status='error',error_message=f'{type(exc).__name__}: {exc}')
        row['elapsed_seconds'] = time.perf_counter()-started
        records.append(row)
    scores = pd.DataFrame(records)
    scores.to_csv(out/'metric_scores.csv',index=False)
    run = dict(run_id=run_id, created_at_utc=datetime.now(timezone.utc).isoformat(),
        input=dict(original_path=str(path),snapshot='input.mp4',sha256=input_hash,
                   model_name=str(model_name),prompt=str(prompt),video_id=video_id),
        video=info, decode_error=decode_error, metrics=metrics, warnings=warnings,
        environment=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
                         torchvision=torchvision.__version__,opencv=cv2.__version__,numpy=np.__version__,pandas=pd.__version__),
        device='cpu', model=model_info, metric_parameters=metric_parameters,
        installed_packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        preprocessing=dict(decode='OpenCV; all frames; BGR for TF, RGB for DINO/AMT-S',
            temporal_flickering='float32 color MAE; (255-mean_MAE)/255; no resize/crop/resampling; no static filter',
            subject_consistency='float32 RGB; bilinear Resize short edge 224 antialias=False; /255; ImageNet mean/std; normalized CLS; nonnegative first/previous cosine average',
            motion_smoothness='official AMT-S CPU; even endpoints; midpoint t=0.5; niters=1; official pad/scale; unpad, quantize uint8; (255-mean reconstruction MAE)/255'),
        timing='per-metric decode, compute, curve output; includes initial model load on first use; excludes initial validation/copy',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'app.py',ROOT/'metrics/subject_consistency.py',ROOT/'metrics/temporal_flickering.py',ROOT/'metrics/motion_smoothness.py']},
        limitations=['OpenCV/decord pixel parity unverified', 'Subject score is not identity accuracy',
                     'OpenCV end-of-stream detection is not a byte-level integrity check'])
    (out/'runs.json').write_text(json.dumps([run],indent=2,ensure_ascii=False)+'\n')
    append_history(root/'metric_scores.csv',records)
    return dict(run_id=run_id, directory=str(out), scores=scores, plots=plots,
                csv=str(out/'metric_scores.csv'), curves=curve_paths, warnings=warnings, video=info)
