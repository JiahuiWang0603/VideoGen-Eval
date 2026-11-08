import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from metrics.motion_smoothness import read_rgb,prepare_inputs,score_errors,evaluate_frames


class Midpoint(torch.nn.Module):
    # Test double for sampling/quantization only, never used in real evaluation.
    def forward(self,a,b,t,**kwargs):
        return {'imgt_pred':(a+b)/2}


class MotionTests(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(score_errors([0,0]),1.)
        self.assertEqual(score_errors([255]),0.)
        self.assertEqual(score_errors([0,255]),.5)
        with self.assertRaises(ValueError):score_errors([])

    def test_preprocessing_pad_and_quantization(self):
        f=np.full((33,35,3),128,np.uint8)
        inputs,padder,scale=prepare_inputs([f]*3)
        self.assertEqual(scale,1.)
        self.assertEqual(tuple(inputs[0].shape),(1,3,48,48))
        self.assertAlmostEqual(inputs[0][0,0,0,0].item(),128/255,places=6)
        self.assertEqual(tuple(padder.unpad(*inputs)[0].shape),(1,3,33,35))

    def test_sampling_and_uint8_error(self):
        # Prediction 0.5 intensity is quantized to 0; comparison against 1 yields MAE=1.
        frames=[np.full((32,32,3),x,np.uint8) for x in [0,1,1,255]]
        score,rows,info=evaluate_frames(Midpoint(),frames)
        self.assertEqual(info['compared_indices'],[1])
        self.assertEqual(rows[0]['reconstruction_mae'],1.)
        self.assertEqual(score,1-1/255)

    def test_decode_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sample.avi'
            writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),8,(32,32))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):writer.write(np.full((32,32,3),(10,80,180),np.uint8))
            finally:writer.release()
            frames=read_rgb(p)
            self.assertEqual(len(frames),3)
            self.assertGreater(frames[0][:,:,0].mean(),frames[0][:,:,2].mean())
            with self.assertRaises(ValueError):read_rgb(Path(tmp)/'missing.mp4')


if __name__=='__main__':unittest.main()
