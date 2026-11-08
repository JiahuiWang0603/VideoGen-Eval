import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
import pandas as pd
from pipeline import evaluate, append_history, inspect_video, ROOT


class PipelineTests(unittest.TestCase):
    def test_history_byte_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metric_scores.csv'
            old=b'run_id,score,status\nold,0.12345678912345678,ok\n'
            path.write_bytes(old)
            append_history(path,[dict(run_id='new',score=.5,status='ok',extra='ignored')])
            self.assertTrue(path.read_bytes().startswith(old))
            self.assertEqual(len(pd.read_csv(path)),2)

    def test_invalid_input(self):
        with self.assertRaises(ValueError): evaluate('missing.mp4')
        path=ROOT/'videos/lavie/In a still frame, a stop sign-0.mp4'
        with self.assertRaises(ValueError): evaluate(path,metrics=[])

    def test_invalid_decode_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'broken.mp4'; path.write_bytes(b'broken')
            out=evaluate(path,metrics=['temporal_flickering'],results_root=Path(tmp)/'out')
            self.assertEqual(out['scores'].iloc[0].status,'error')
            self.assertTrue(pd.isna(pd.read_csv(out['csv']).iloc[0].score))
            self.assertTrue(out['scores'].iloc[0].error_message)

    def test_partial_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('pipeline.get_subject_model',side_effect=FileNotFoundError('missing weights')), patch('pipeline.get_motion_model',side_effect=FileNotFoundError('missing AMT-S weights')):
                out=evaluate(ROOT/'videos/lavie/In a still frame, a stop sign-0.mp4',
                             results_root=tmp)
            self.assertEqual(list(out['scores'].status),['ok','error','error'])
            self.assertEqual(out['video']['frames'],16)
            self.assertTrue(out['warnings'])
            self.assertEqual(len(out['plots']),1)
            curves=pd.read_csv(out['curves'][0]);self.assertEqual(len(curves),15)
            self.assertTrue(Path(out['directory'],'input.mp4').exists())
            self.assertEqual(len(pd.read_csv(Path(tmp)/'metric_scores.csv')),3)

    def test_actual_missing_weight_loaders(self):
        from pipeline import get_subject_model,get_motion_model
        original = Path.is_file
        get_subject_model.cache_clear()
        get_motion_model.cache_clear()
        def exists(path):
            if path.name in ['dino_vitbase16_pretrain.pth','amt-s.pth']:
                return False
            return original(path)
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path,'is_file',exists):
            out=evaluate(ROOT/'videos/lavie/a person swimming in ocean-0.mp4',results_root=tmp)
        self.assertEqual(list(out['scores'].status),['ok','error','error'])
        self.assertTrue(out['scores'].iloc[1].error_message)
        self.assertIn('AMT-S',out['scores'].iloc[2].error_message)
        self.assertTrue(out['scores'].iloc[1:]['score'].isna().all())


if __name__=='__main__': unittest.main()
