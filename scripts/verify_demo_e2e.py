"""Live HTTP upload/inference/download regression. Start app.py before running."""
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from gradio_client import Client, handle_file
from pipeline import ROOT, METRICS, sha


def main():
    history_path=ROOT/'results/metric_scores.csv'
    prefix=history_path.read_bytes()
    history=pd.read_csv(history_path)
    protected={str(p.relative_to(ROOT)):sha(p) for folder in ['data','videos','metrics','results']
               for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and p!=history_path}
    url=os.environ.get('DEMO_URL','http://127.0.0.1:7860')
    with urllib.request.urlopen(url+'/',timeout=10) as response:
        assert response.status==200
    client=Client(url,verbose=False)
    checks=[]; comparison_ids=[]
    for content,metrics in [('a person swimming in ocean',list(METRICS)),('In a still frame, a stop sign',['temporal_flickering'])]:
        paths=[ROOT/'videos'/model/f'{content}-0.mp4' for model in ['lavie','modelscope']]
        result=client.predict(handle_file(str(paths[0])),'lavie',handle_file(str(paths[1])),'modelscope',content,metrics,api_name='/evaluate')
        scores=pd.DataFrame(result[0]['data'],columns=result[0]['headers'])
        assert len(scores)==2*len(metrics) and (scores.status=='ok').all()
        assert len(result[1])==1+2*(1+len(metrics))
        for file in result[1]:assert Path(file).is_file(),file
        for index,metric in enumerate(METRICS):
            for slot_index in range(2):
                image=result[3+index*4+slot_index*2+1]
                assert (image is not None)==(metric in metrics),(metric,image)
        metadata=pd.read_csv(ROOT/'data/metadata.csv')
        for row in scores.itertuples():
            video_id=metadata[(metadata.model_name==row.model_name)&(metadata.prompt==content)].iloc[0].video_id
            previous=history[(history.video_id==video_id)&(history.metric_name==row.metric_name)].iloc[0].score
            delta=abs(float(row.score)-float(previous))
            assert delta<1e-12,(row.model_name,row.metric_name,delta)
            checks.append(dict(model=row.model_name,prompt=content,metric=row.metric_name,score=float(row.score),historical_difference=delta))
        comparison_ids.append(result[2].splitlines()[0].split(': ',1)[1])
    with tempfile.TemporaryDirectory() as tmp:
        broken=Path(tmp)/'broken.mp4';broken.write_bytes(b'invalid mp4 test input')
        result=client.predict(handle_file(str(broken)),'broken',None,'','invalid',list(METRICS),api_name='/evaluate')
        errors=pd.DataFrame(result[0]['data'],columns=result[0]['headers'])
        assert (errors.status=='error').all() and all(v=='' for v in errors.score)
        assert all(errors.error_message.str.len()>0)
        assert all(result[i] is None for i in [4,6,8,10,12,14])
        invalid_comparison=result[2].splitlines()[0].split(': ',1)[1]
    assert history_path.read_bytes().startswith(prefix)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    report=dict(homepage_http_status=200,comparison_ids=comparison_ids,checks=checks,
        invalid_video_comparison=invalid_comparison,invalid_video_errors=errors.error_message.tolist(),
        downloads_verified=True,curves_verified=True,historical_bytes_preserved=True,existing_artifacts_unchanged=True)
    path=ROOT/'results'/comparison_ids[0]/'http_e2e_validation.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(pd.DataFrame(checks).to_string(index=False));print('Validation:',path)


if __name__=='__main__':main()
