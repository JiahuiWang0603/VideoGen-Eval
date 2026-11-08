"""CPU reproduction with reference numerical parity and reproducibility records."""
import ast
import hashlib
import json
import platform
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import torchvision
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics.subject_consistency import load_model,evaluate_video,load_video,dino_transform


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    protected=[p for folder in ['data','videos','metrics','results'] for p in (ROOT/folder).rglob('*')
               if p.is_file() and '__pycache__' not in str(p) and p.name not in ['metric_scores.csv','runs.json']]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    scores_path=ROOT/'results/metric_scores.csv'; prior=pd.read_csv(scores_path)
    runs_path=ROOT/'results/runs.json'; old_runs=json.loads(runs_path.read_text())
    torch.manual_seed(20261109); torch.set_num_threads(4)
    weights=ROOT/'models/dino/dino_vitbase16_pretrain.pth'
    source=json.loads((ROOT/'models/dino/source.json').read_text())
    assert sha(weights)==source['weight_sha256']
    started=time.perf_counter(); model=load_model(ROOT/'vendor/dino',weights)
    model_seconds=time.perf_counter()-started
    run_id=datetime.now(timezone.utc).strftime('subject_%Y%m%dT%H%M%S%fZ')
    out=ROOT/'results'/run_id; out.mkdir(exist_ok=False)
    tree=ast.parse((ROOT/'vendor/subject_consistency.py.txt').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='subject_consistency')
    env=dict(torch=torch,F=F,dino_transform=dino_transform,load_video=load_video,tqdm=tqdm,get_rank=lambda:0)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'upstream_subject_core','exec'),env)
    selected=pd.read_csv(ROOT/'data/metadata.csv')
    selected=selected[selected.dimension=='subject_consistency'].drop_duplicates('video_id')
    records=[]
    for r in selected.itertuples():
        print('Evaluating',r.model_name,flush=True)
        start=time.perf_counter()
        score,curve,features=evaluate_video(model,ROOT/r.video_path)
        original_score,detail=env['subject_consistency'](model,[str(ROOT/r.video_path)],'cpu',False)
        assert score==detail[0]['video_results'],(score,detail)
        pd.DataFrame(curve).to_csv(out/f'{r.video_id}_similarities.csv',index=False)
        np.save(out/f'{r.video_id}_features.npy',features.numpy())
        fig,ax=plt.subplots(figsize=(8,4))
        for key in ['similarity_previous','similarity_first','similarity_mean']:
            ax.plot([x['frame_id'] for x in curve],[x[key] for x in curve],label=key)
        ax.set(title=f'{r.model_name} Subject Consistency: {score:.8f}',xlabel='Frame index',ylabel='Feature similarity',ylim=(0,1.01))
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/f'{r.video_id}_similarities.png',dpi=160); plt.close(fig)
        records.append(dict(run_id=run_id,video_id=r.video_id,model_name=r.model_name,
            metric_name='subject_consistency',score=score,status='ok',error_message='',
            elapsed_seconds=time.perf_counter()-start,frames=len(curve)+1,
            official_score=original_score,official_abs_difference=abs(score-original_score)))
        print(records[-1],flush=True)
    # Extra real-model identical-image check, without replacing ordinary features.
    from metrics.subject_consistency import extract_features,feature_similarity
    image=dino_transform()(load_video(ROOT/selected.iloc[0].video_path))[:1]
    same=extract_features(model,image.repeat(2,1,1,1))
    identical_score=feature_similarity(same)[0]
    assert abs(identical_score-1)<1e-6
    assert all(sha(ROOT/p)==h for p,h in before.items())
    run=dict(run_id=run_id,device='cpu',seed=20261109,threads=4,
        environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,torchvision=torchvision.__version__,opencv=cv2.__version__,numpy=np.__version__,pandas=pd.__version__,mps_available=torch.backends.mps.is_available()),
        model=source,model_load_seconds=model_seconds,identical_image_score=identical_score,
        preprocessing=dict(decoder='OpenCV, RGB; decord pixel parity NOT verified',all_frames=True,resize_short_edge=224,interpolation='bilinear tensor float32',antialias=False,crop=False,divide=255,mean=[.485,.456,.406],std=[.229,.224,.225]),
        feature='DINO ViT-B/16 CLS 768d, eval/no_grad, L2 normalized',
        reference='AST extracted upstream subject_consistency using same OpenCV adapter and transform; exact scalar equality; full decord pipeline not verified',
        timing_scope='decode, inference, second upstream-reference inference, curves; excludes model load',
        existing_artifacts_sha256=before,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'metrics/subject_consistency.py',Path(__file__).resolve()]},artifact_directory=str(out.relative_to(ROOT)))
    (out/'manifest.json').write_text(json.dumps(run,indent=2)+'\n')
    pd.concat([prior,pd.DataFrame(records)],ignore_index=True).to_csv(scores_path,index=False)
    # Confirm prior rows survived serialization unchanged in their original columns.
    pd.testing.assert_frame_equal(pd.read_csv(scores_path).iloc[:len(prior)][prior.columns].reset_index(drop=True),prior,check_dtype=False)
    runs_path.write_text(json.dumps(old_runs+[run],indent=2)+'\n')
    (out/'report.md').write_text('Subject Consistency CPU reproduction\n\n'+pd.DataFrame(records).to_csv(index=False)+'\nExact upstream core score parity with shared OpenCV decoder. Full decord decode and distributed pipeline unverified. Whole-frame feature similarity is not identity accuracy; two clips cannot rank overall models.\n')
    print('Outputs:',out)


if __name__=='__main__': main()
