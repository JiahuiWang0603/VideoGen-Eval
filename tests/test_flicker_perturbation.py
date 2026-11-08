import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from experiments.flicker_perturbation import read_rgb, perturb, evaluate, save_tables


class PerturbationTests(unittest.TestCase):
    def test_control_and_analytic_scores(self):
        frames=[np.full((4,4,3),100,np.uint8) for _ in range(4)]
        rows,curves,checks=evaluate(frames)
        scores={r['group']:r['score'] for r in rows}
        self.assertEqual(scores['Original'],1)
        self.assertEqual(scores['Control'],1)
        self.assertAlmostEqual(scores['Mild Flicker'],1-8/255)
        self.assertAlmostEqual(scores['Strong Flicker'],1-32/255)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(curves),12)
        self.assertTrue(all((f==100).all() for f in frames))

    def test_clipping_and_copy(self):
        frames=[np.full((2,2,3),250,np.uint8) for _ in range(2)]
        out=perturb(frames,32)
        self.assertTrue((out[0]==250).all()); self.assertTrue((out[1]==255).all())
        self.assertFalse(np.shares_memory(out[0],frames[0]))

    def test_expected_order_can_fail(self):
        frames=[np.zeros((2,2,3),np.uint8),np.full((2,2,3),255,np.uint8)]
        _,_,checks=evaluate(frames)
        self.assertFalse(checks['mild_lower_than_original'])

    def test_decode_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'video.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),8,(32,32))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3): writer.write(np.full((32,32,3),(10,80,180),np.uint8))
            finally: writer.release()
            frames=read_rgb(path)
            self.assertEqual(len(frames),3)
            self.assertGreater(float(frames[0][:,:,0].mean()),float(frames[0][:,:,2].mean()))
            with self.assertRaises(ValueError): read_rgb(Path(tmp)/'missing.mp4')

    def test_save_roundtrip_and_no_overwrite(self):
        frames=[np.full((2,2,3),100,np.uint8) for _ in range(3)]
        rows,curves,_=evaluate(frames)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); save_tables(out,rows,curves)
            result=pd.read_csv(out/'perturbation_results.csv')
            self.assertEqual(len(result),4)
            np.testing.assert_allclose(result.score,[r['score'] for r in rows],rtol=0,atol=1e-15)
            self.assertEqual(len(pd.read_csv(out/'frame_mae_curves.csv')),8)
            with self.assertRaises(FileExistsError): save_tables(out,rows,curves)


if __name__=='__main__': unittest.main()
