"""VBench color-frame temporal flickering, without static filtering."""
import cv2
import numpy as np


def get_frames(video_path):
    frames = []
    video = cv2.VideoCapture(str(video_path))
    try:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frames.append(frame)
    finally:
        video.release()
    if len(frames) < 2:
        raise ValueError('Temporal flickering requires at least two readable frames')
    return frames


def calculate_mae(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError('Frames must have identical shapes')
    return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32),
                               np.array(img2, dtype=np.float32)))


def mae_seq(frames):
    return np.array([calculate_mae(a, b) for a, b in zip(frames[:-1], frames[1:])])


def score_frames(frames):
    if len(frames) < 2:
        raise ValueError('At least two frames required')
    return (255.0 - np.mean(mae_seq(frames)).item()) / 255.0


def cal_score(video_path):
    return score_frames(get_frames(video_path))
