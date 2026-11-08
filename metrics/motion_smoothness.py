"""VBench AMT-S motion smoothness on CPU, using unmodified upstream network code."""
import ast
import glob
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'vendor/amt_reference'


def upstream_tools():
    sys.path.insert(0, str(REFERENCE)) if str(REFERENCE) not in sys.path else None
    return importlib.import_module('vbench.third_party.amt.utils.utils')


def read_rgb(path):
    video = cv2.VideoCapture(str(path))
    frames = []
    try:
        expected = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        video.release()
    if len(frames) < 3:
        raise ValueError('AMT-S requires at least three decoded frames')
    if expected > len(frames):
        raise ValueError('Incomplete video decode')
    if len({f.shape for f in frames}) != 1:
        raise ValueError('Inconsistent frame sizes')
    return frames


def score_errors(errors):
    if not len(errors):
        raise ValueError('No interpolated frames to compare')
    return float((255.0 - np.mean(np.array(errors))) / 255.0)


def prepare_inputs(frames):
    tools = upstream_tools()
    inputs = [tools.img2tensor(frame) for frame in frames[::2]]
    if len(inputs) < 2:
        raise ValueError('At least two interpolation endpoints required')
    inputs = tools.check_dim_and_resize(inputs)
    # These are the exact upstream CPU branch constants and scale formula.
    h, w = inputs[0].shape[-2:]
    scale = (8192*8192) / (h*w) * np.sqrt((1-0)/1)
    scale = 1 if scale > 1 else scale
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    padder = tools.InputPadder(inputs[0].shape, int(16/scale))
    return padder.pad(*inputs), padder, scale


def load_model():
    from omegaconf import OmegaConf
    weights = ROOT/'models/amt/amt-s.pth'
    source = json.loads((ROOT/'models/amt/source.json').read_text())
    if not weights.is_file():
        raise FileNotFoundError('Missing AMT-S weights; no automatic download')
    if hashlib.sha256(weights.read_bytes()).hexdigest() != source['weight_sha256']:
        raise ValueError('AMT-S weight SHA-256 mismatch')
    for name, expected in source['source_sha256'].items():
        if hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Upstream source changed: {name}')
    upstream_tools()
    config = REFERENCE/'vbench/third_party/amt/cfgs/AMT-S.yaml'
    cfg = OmegaConf.load(config).network
    builder = importlib.import_module('vbench.third_party.amt.utils.build_utils')
    model = builder.build_from_cfg(cfg)
    # Match upstream legacy checkpoint loading, only after pinned SHA-256 verification.
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.cpu().eval(), source


def evaluate_frames(model, frames):
    inputs, padder, scale = prepare_inputs(frames)
    tools = upstream_tools()
    embt = torch.tensor(1/2).float().view(1,1,1,1)
    outputs = [inputs[0]]
    with torch.no_grad():
        for a, b in zip(inputs[:-1], inputs[1:]):
            pred = model(a, b, embt, scale_factor=scale, eval=True)['imgt_pred']
            outputs += [pred.cpu(), b.cpu()]
    # Preserve upstream multiply/clip/uint8 quantization before pixel MAE.
    images = [tools.tensor2img(out) for out in padder.unpad(*outputs)]
    rows = []
    for i, prediction in enumerate(images[1::2]):
        index = 2*i+1
        error = float(np.mean(cv2.absdiff(frames[index], prediction)))
        rows.append(dict(frame_id=index, endpoint_left=index-1, endpoint_right=index+1,
                         reconstruction_mae=error, normalized_error=error/255.0))
    return score_errors([r['reconstruction_mae'] for r in rows]), rows, dict(
        input_frames=len(frames), endpoint_indices=list(range(0,len(frames),2)),
        compared_indices=[r['frame_id'] for r in rows], scale_factor=float(scale),
        padded_shape=list(inputs[0].shape), niters=1, interpolation_time=.5)


def evaluate_video(model, path):
    return evaluate_frames(model, read_rgb(path))


def official_reference():
    """Run the upstream class unchanged, bypassing unrelated VBench imports."""
    from omegaconf import OmegaConf
    tools = upstream_tools()
    builder = importlib.import_module('vbench.third_party.amt.utils.build_utils')
    tree = ast.parse((REFERENCE/'vbench/motion_smoothness.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('FrameProcess','MotionSmoothness')]
    env = dict(os=os,glob=glob,cv2=cv2,np=np,torch=torch,OmegaConf=OmegaConf,
               build_from_cfg=builder.build_from_cfg)
    env.update({name:getattr(tools,name) for name in ['img2tensor','tensor2img','check_dim_and_resize','InputPadder']})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_motion_classes','exec'),env)
    return env['MotionSmoothness'](str(REFERENCE/'vbench/third_party/amt/cfgs/AMT-S.yaml'),
                                str(ROOT/'models/amt/amt-s.pth'),'cpu')
