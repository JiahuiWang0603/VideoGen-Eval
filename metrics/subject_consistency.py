"""DINO ViT-B/16 VBench subject consistency; explicit OpenCV decode adapter."""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


def dino_transform(size=224):
    return transforms.Compose([transforms.Resize(size=size, antialias=False),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])


def load_video(path):
    cap=cv2.VideoCapture(str(path)); frames=[]
    try:
        while cap.isOpened():
            ok,frame=cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally: cap.release()
    if len(frames)<2: raise ValueError('At least two decoded frames required')
    # Official load_video converts uint8 buffer to float Tensor before transforms.
    return torch.Tensor(np.array(frames,dtype=np.uint8)).permute(0,3,1,2)


def load_model(repo, weights):
    repo=Path(repo); weights=Path(weights)
    if not weights.is_file(): raise FileNotFoundError(weights)
    sys.path.insert(0,str(repo.resolve()))
    import vision_transformer
    model=vision_transformer.vit_base(patch_size=16,num_classes=0)
    state=torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.eval()


def feature_similarity(features):
    if len(features)<2: raise ValueError('At least two feature vectors required')
    features=F.normalize(features,dim=-1,p=2)
    rows=[]
    for i in range(1,len(features)):
        prev=max(0.,F.cosine_similarity(features[i-1:i],features[i:i+1]).item())
        first=max(0.,F.cosine_similarity(features[0:1],features[i:i+1]).item())
        rows.append(dict(frame_id=i,similarity_previous=prev,similarity_first=first,
                         similarity_mean=(prev+first)/2))
    return sum(r['similarity_mean'] for r in rows)/len(rows),rows


def extract_features(model,images):
    features=[]
    with torch.no_grad():
        for image in images:
            features.append(F.normalize(model(image.unsqueeze(0)),dim=-1,p=2))
    return torch.cat(features)


def evaluate_video(model,path):
    images=dino_transform()(load_video(path))
    features=extract_features(model,images)
    # Already normalized exactly once, as upstream; helper normalizes again, so
    # use direct upstream-compatible comparison below to avoid numeric drift.
    rows=[]
    for i in range(1,len(features)):
        prev=max(0.,F.cosine_similarity(features[i-1:i],features[i:i+1]).item())
        first=max(0.,F.cosine_similarity(features[:1],features[i:i+1]).item())
        rows.append(dict(frame_id=i,similarity_previous=prev,similarity_first=first,
                         similarity_mean=(prev+first)/2))
    return sum(r['similarity_mean'] for r in rows)/len(rows),rows,features
