# 主体一致性（Subject Consistency）指标复现

使用独立的 `.venv-subject` 环境，原有 `.venv` 环境保持不变。
本阶段安装的依赖版本记录在 `requirements-subject.lock.txt` 中。

```bash
.venv-subject/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python scripts/run_subject_consistency.py
```

运行脚本需要本地已有的 `models/dino/dino_vitbase16_pretrain.pth` 权重文件，
不会自动下载缺失的权重。下载来源、DINO 源码提交版本、文件大小和权重 SHA-256
记录在 `models/dino/source.json` 中，官方源码快照保存在 `vendor/` 中。

## 算法与预处理

采用第一代 DINO ViT-B/16，提取 768 维全画面 CLS 特征。
严格加载官方权重参数，在 CPU 上使用评估模式（eval）和不计算梯度的模式（no_grad）。
通过 OpenCV 解码全部帧，将 BGR 转为 RGB，再构造 float32 类型的 TCHW 张量，
数据类型与官方 `load_video` 一致。

使用张量双线性插值将短边缩放至 224，设置 `antialias=False`；随后除以 255，
并应用 ImageNet 均值与标准差进行归一化。不裁剪、不抽帧、不改变帧率。
对首帧之后的每一帧，分别计算其与前一帧、首帧的余弦相似度，
将负值截断为零后取二者均值，再对整段视频的这些均值取平均，得到视频得分。

## 输出与记录

每段视频的特征数组、全部 15 个比较点、PNG 曲线及独立的运行清单和报告，
保存在新的运行目录中。统一的 `metric_scores.csv` 和 `runs.json` 保留旧记录并追加本次记录。
其他已有数据和结果通过 SHA-256 核对，原有 Temporal Flickering 代码与数据未修改。

## 官方对照与解释边界

对照验证执行已保存的上游 `subject_consistency` 函数，使用相同的 OpenCV 解码适配器和预处理，
要求最终标量得分完全一致。这项验证不涵盖默认 decord 解码、分布式汇总或完整官方命令行流程。
运行耗时包含本地实现与官方对照的两次推理，以及曲线输出。

全画面特征相似度不是主体身份识别准确率，背景和水面同样可能影响特征。
仅凭两段视频不能判断模型整体排名。
