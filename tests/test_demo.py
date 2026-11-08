import unittest
from unittest.mock import patch
import gradio as gr
from app import build_demo,run_demo,run_compare


class DemoTests(unittest.TestCase):
    def test_schema_builds(self):
        demo=build_demo()
        info=demo.get_api_info()
        self.assertIn('/evaluate',info['named_endpoints'])

    def test_missing_upload(self):
        with self.assertRaises(gr.Error): run_demo(None,'','','',['temporal_flickering'])

    def test_missing_weights_ui_clears_score_and_curve(self):
        import tempfile
        from pathlib import Path
        from pipeline import ROOT,evaluate,get_motion_model
        original=Path.is_file
        get_motion_model.cache_clear()
        def exists(path):
            return False if path.name=='amt-s.pth' else original(path)
        with tempfile.TemporaryDirectory() as tmp:
            def run(*args,**kwargs):return evaluate(*args,**kwargs,results_root=tmp)
            with patch('app.ROOT',Path(tmp)),patch('app.evaluate',run),patch.object(Path,'is_file',exists):
                output=run_compare(str(ROOT/'videos/lavie/a person swimming in ocean-0.mp4'),'lavie',None,'','swimmer',['temporal_flickering','motion_smoothness'])
            self.assertIn('AMT-S',output[2])
            self.assertIn('计算失败',output[11])
            self.assertIsNone(output[12])
            self.assertIsNotNone(output[4])


if __name__=='__main__': unittest.main()
