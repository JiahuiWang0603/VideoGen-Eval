"""Real-data regression against frozen historical metric scores."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from pipeline import ROOT,evaluate,sha


def main():
    history=ROOT/'results/metric_scores.csv'
    prefix=history.read_bytes()
    protected={str(p.relative_to(ROOT)):sha(p) for folder in ['data','videos','metrics','results']
               for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)
               and p != history}
    old=pd.read_csv(history)
    selected=pd.read_csv(ROOT/'data/metadata.csv').drop_duplicates('video_id')
    checks=[]
    for r in selected.itertuples():
        metric='temporal_flickering' if r.prompt=='In a still frame, a stop sign' else 'subject_consistency'
        out=evaluate(ROOT/r.video_path,r.model_name,r.prompt,r.video_id,[metric])
        row=out['scores'].iloc[0]
        expected=old[(old.video_id==r.video_id)&(old.metric_name==metric)].iloc[0].score
        assert row.status=='ok',row.error_message
        delta=abs(row.score-expected)
        assert delta<1e-12,(r.video_id,delta)
        assert len(pd.read_csv(out['curves'][0]))==15
        checks.append(dict(video_id=r.video_id,model=r.model_name,metric=metric,
                           score=row.score,historical_score=expected,abs_difference=delta,run_id=out['run_id']))
    assert history.read_bytes().startswith(prefix)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    report=dict(checks=checks,historical_csv_byte_prefix_preserved=True,
                existing_artifacts_unchanged=True,existing_artifact_sha256=protected)
    target=ROOT/'results'/out['run_id']/'regression_validation.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(checks).to_string(index=False)); print('Validation:',target)


if __name__=='__main__': main()
