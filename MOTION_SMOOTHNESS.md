# Motion Smoothness — VBench AMT-S CPU reproduction

Use the existing `.venv-subject` environment. Added dependencies: OmegaConf and imageio.
The complete current environment is frozen in `requirements-motion.lock.txt`.

```bash
cd /Users/wangjiahui/Desktop/VideoGen-Eval
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python scripts/run_motion_smoothness.py
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python -m unittest discover -s tests -v
```

No weights are downloaded by the runner. The approved official `amt-s.pth` is in
`models/amt/`; source URL, byte count, SHA-256 and fixed VBench commit are in
`models/amt/source.json`. The unmodified upstream network/config/tool sources are
in `vendor/amt_reference`. Source hashes and weight hash are checked before loading.
The official legacy checkpoint contains pickle globals unsupported by the restricted
PyTorch loader; after SHA-256 verification, loading uses `weights_only=False`, matching
upstream. The model has 2,990,543 parameters; no architecture substitutions are made.

The independent implementation is `metrics/motion_smoothness.py`:
OpenCV full decode -> RGB -> even frames -> float32 /255 -> upstream CPU scaling
and replicate padding -> official AMT-S midpoint prediction -> unpad -> multiply 255,
clip and uint8 conversion -> RGB MAE against corresponding odd original frames ->
`(255 - mean(MAE)) / 255`. One interpolation iteration, t=0.5, CPU float32.
Both current clips use scale=1 with no padding needed. Each has 16 frames and seven
comparisons at frames 1,3,5,7,9,11,13. Original frame 15 has no right endpoint and is
not scored. Neither source video is changed. FPS is not resampled.

The runner loads a separate official MotionSmoothness instance from AST-extracted,
unmodified upstream classes and compares each reconstruction MAE and final score.
Two actual clips have exact pairwise and final parity. This bypasses unrelated
VBench imports, not AMT-S operations. Full CLI, distributed evaluation, MPS and CUDA
numerical parity remain unverified. CPU grid_sample and complete real-video inference
have succeeded on this Mac ARM64; MPS is unavailable in the current execution context.

Each run creates `results/motion_<UTC timestamp>/` with per-video reconstruction CSV,
PNG, score CSV, report and runs.json. The manifest records code/source/weight hashes,
input hashes, environment, CPU threads, timings, peak process memory and preprocessing.
Elapsed time includes validation, our inference, the official second inference and
curve output. Per-inference timings are separately available in the manifest.
Scores append to the existing unified metric_scores.csv using its existing schema
and a file lock; historical bytes are preserved. Existing metric modules, data,
Temporal Flickering and Subject Consistency artifacts are hash-checked unchanged.
The old global runs.json remains unchanged; this run has its own runs.json.

Tests cover RGB decoding, invalid input, scaling and padding, /255 normalization,
frame pairing, uint8 quantization and score formula. The actual runner additionally
reads saved CSV back, checks scores and pairwise official parity, and verifies
historical artifact preservation. Test interpolation doubles are only used in unit
tests; all reported real-video scores use the approved official AMT-S weights.

This score measures interpolation reconstruction consistency. It is not a direct
measure of action realism and two videos cannot establish overall model superiority.
The current unified Pipeline/Gradio still exposes the previous two metrics; this
stage adds the independently runnable official third metric without changing them.
