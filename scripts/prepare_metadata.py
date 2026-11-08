"""Index only real local videos matching the official VBench prompt mapping."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['video_id', 'model_name', 'prompt', 'dimension', 'video_path', 'source']
DIMENSIONS = ['temporal_flickering', 'motion_smoothness', 'subject_consistency']
SOURCES = {
    'lavie': 'https://drive.google.com/file/d/1hviZzsInIgJA96ppVj4B2DHhTZWeM4nc/view',
    'modelscope': 'https://drive.google.com/file/d/1UH2-lALFShjBywyImjDPPHTpE43eoMQE/view',
}


def main():
    info = {x['prompt_en']: x['dimension'] for x in json.loads(
        (ROOT / 'data/VBench_full_info.json').read_text(encoding='utf-8'))}
    planned = []
    for suite in ['subject_consistency', 'temporal_flickering']:
        prompt = (ROOT / f'data/prompts/{suite}.txt').read_text(encoding='utf-8').splitlines()[0]
        for model, source in SOURCES.items():
            for dimension in DIMENSIONS:
                if dimension in info[prompt]:
                    planned.append(dict(model_name=model, prompt=prompt, dimension=dimension,
                                        expected_filename=f'{prompt}-0.mp4',
                                        video_path=f'videos/{model}/{prompt}-0.mp4',
                                        source=source, status='pending_download'))
    pd.DataFrame(planned).to_csv(ROOT / 'data/download_plan.csv', index=False)
    rows = []
    for model, source in SOURCES.items():
        folder = ROOT / 'videos' / model
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() != '.mp4':
                continue
            match = re.fullmatch(r'(.+)-(\d+)', path.stem)
            prompt = match[1] if match else None
            if prompt not in info:
                print(f'跳过：无法精确匹配官方 Prompt：{path.relative_to(ROOT)}')
                continue
            relative = path.relative_to(ROOT).as_posix()
            video_id = hashlib.sha256(relative.encode()).hexdigest()[:16]
            for dimension in DIMENSIONS:
                if dimension in info[prompt]:
                    rows.append(dict(video_id=video_id, model_name=model, prompt=prompt,
                                     dimension=dimension, video_path=relative, source=source))
    pd.DataFrame(rows, columns=COLUMNS).to_csv(ROOT / 'data/metadata.csv', index=False)
    print(f'已登记 {len(rows)} 条维度记录；下载计划独立保存在 data/download_plan.csv。')


if __name__ == '__main__':
    main()
