"""Decode videos and report cross-model coverage; does not compute VBench scores."""
import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from prepare_metadata import COLUMNS, DIMENSIONS, ROOT, SOURCES


def inspect_video(path):
    if not path.is_file():
        return {'status': 'missing', 'decoded_frames': 0}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {'status': 'cannot_open', 'decoded_frames': 0}
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                break
            count += 1
        status = 'ok' if count >= 2 else 'too_few_frames'
        if expected > 0 and count < expected:
            status = 'incomplete_decode'
        return dict(status=status, decoded_frames=count, reported_frames=expected, fps=fps)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT / 'data/metadata.csv')
    args = parser.parse_args()
    df = pd.read_csv(args.metadata, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        parser.error(f'缺少列：{sorted(missing)}')
    if df.empty:
        print('metadata.csv 暂无真实视频记录。请按 data/download_plan.csv 手动下载后重新建立 metadata。')
        return 1
    info = {x['prompt_en']: x['dimension'] for x in json.loads(
        (ROOT / 'data/VBench_full_info.json').read_text(encoding='utf-8'))}
    cache, records = {}, []
    duplicates = df.duplicated(['video_id', 'dimension'], keep=False)
    for index, row in df.iterrows():
        errors = []
        if any(not str(row[c]).strip() for c in COLUMNS):
            errors.append('empty_field')
        if row.dimension not in DIMENSIONS or row.dimension not in info.get(row.prompt, []):
            errors.append('invalid_prompt_dimension')
        if row.model_name not in SOURCES or row.source != SOURCES.get(row.model_name):
            errors.append('unknown_model_or_source')
        if duplicates.iloc[index]:
            errors.append('duplicate_video_dimension')
        path = (ROOT / row.video_path).resolve()
        if not path.is_relative_to(ROOT / 'videos' / row.model_name):
            errors.append('path_outside_model_directory')
        if path not in cache:
            cache[path] = inspect_video(path)
        result = cache[path]
        records.append({**row.to_dict(), **result, 'metadata_errors': ';'.join(errors)})
    report = pd.DataFrame(records)
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    report.to_csv(output / 'data_check.csv', index=False)
    valid = report[(report.status == 'ok') & (report.metadata_errors == '')]
    coverage = df[['prompt', 'dimension']].drop_duplicates().copy()
    counts = valid.groupby(['prompt', 'dimension']).model_name.nunique()
    coverage['readable_model_count'] = [int(counts.get((r.prompt, r.dimension), 0))
                                        for r in coverage.itertuples()]
    coverage['has_cross_model'] = coverage.readable_model_count >= 2
    coverage.to_csv(output / 'prompt_coverage.csv', index=False)
    print(report.status.value_counts().to_string())
    print(f'可读取跨模型 Prompt/维度组合：{coverage.has_cross_model.sum()}/{len(coverage)}')
    missing_dims = set(DIMENSIONS) - set(valid.dimension)
    print(f'缺少可读取样本的维度：{sorted(missing_dims)}')
    return int(len(valid) != len(df) or not coverage.has_cross_model.all() or bool(missing_dims))


if __name__ == '__main__':
    raise SystemExit(main())
