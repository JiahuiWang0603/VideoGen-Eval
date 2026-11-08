"""Unified CLI for arbitrary local MP4 videos."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import evaluate, METRICS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video')
    p.add_argument('--model-name',default='')
    p.add_argument('--prompt',default='')
    p.add_argument('--video-id',default='')
    p.add_argument('--metrics',nargs='+',choices=METRICS,default=list(METRICS))
    a=p.parse_args()
    result=evaluate(a.video,a.model_name,a.prompt,a.video_id,a.metrics)
    print(result['scores'].to_string(index=False))
    for warning in result['warnings']: print(warning)
    print('Outputs:',result['directory'])
    raise SystemExit(int(any(result['scores'].status!='ok')))
