# Subject Consistency reproduction

Use the separate `.venv-subject` environment; the existing `.venv` is unchanged.
Installed versions are frozen in `requirements-subject.lock.txt`.

```bash
.venv-subject/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python scripts/run_subject_consistency.py
```

The runner requires the existing local `models/dino/dino_vitbase16_pretrain.pth`;
it never downloads missing weights. Source URL, DINO commit, size and weight SHA-256
are in `models/dino/source.json`. Official source snapshots are in `vendor/`.

Algorithm: original DINO ViT-B/16, 768-dimensional whole-frame CLS features.
Load strict official state dict, run CPU eval/no_grad. Decode all frames via OpenCV,
convert BGR to RGB, form float32 TCHW tensor (same dtype as official load_video).
Resize short edge to 224 with tensor bilinear and antialias=False, then divide by
255 and apply ImageNet normalization. No cropping, frame sampling or FPS change.
For each frame after the first, average nonnegative cosine similarity against the
previous and first frame, then average those values across the video.

Per-video feature arrays, all 15 comparison points, PNG curves and an independent
manifest/report are saved in the new run directory. Unified metric_scores.csv and
runs.json retain prior records and append the new run. Other existing data/results
are checked by SHA-256. No existing Temporal Flickering code or data was changed.

Reference check executes the saved upstream subject_consistency function with the
same OpenCV adapter and preprocessing. Exact scalar parity is required. This does
not validate default decord decoding, distributed aggregation, or a full official
CLI run. Runtime includes two inference passes for reference comparison and plots.
Whole-frame feature similarity is not identity accuracy. The background and water
can influence the feature. Two clips do not support an overall model ranking.
