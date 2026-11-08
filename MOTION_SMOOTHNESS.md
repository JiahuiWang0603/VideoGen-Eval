# 运动平滑度（Motion Smoothness）：VBench AMT-S CPU 复现

使用已有的 `.venv-subject` 环境，新增依赖为 OmegaConf 和 imageio。
本阶段的完整环境版本记录在 `requirements-motion.lock.txt` 中。

```bash
cd /Users/wangjiahui/Desktop/VideoGen-Eval
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python scripts/run_motion_smoothness.py
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python -m unittest discover -s tests -v
```

## 模型来源与加载

运行脚本不会自动下载权重。已确认下载的官方 `amt-s.pth` 位于 `models/amt/`；
来源地址、文件字节数、SHA-256 和固定的 VBench 提交版本记录在 `models/amt/source.json` 中。
未经修改的上游网络、配置和工具源码位于 `vendor/amt_reference`，加载前会核对源码与权重哈希。

官方旧版检查点包含受限 PyTorch 加载器不支持的 pickle 全局对象，因此在确认 SHA-256 后，
使用与上游一致的 `weights_only=False` 加载方式。模型共有 2,990,543 个参数，没有替换模型结构。

## 计算流程

独立实现位于 `metrics/motion_smoothness.py`，流程如下：

1. 使用 OpenCV 完整解码视频并转为 RGB。
2. 取偶数索引帧作为插值端点，转换为 float32 并除以 255。
3. 使用官方 CPU 缩放规则和边界复制填充。
4. 使用官方 AMT-S 预测中间帧。
5. 移除填充，乘以 255，截断后转换为 uint8。
6. 与对应的原始奇数索引帧计算 RGB 平均绝对误差（MAE）。
7. 按 `(255 - mean(MAE)) / 255` 得到视频得分。

仅执行一轮插值，时间位置为 `t=0.5`，采用 CPU float32 推理。
当前两段视频均使用 `scale=1`，无需实际填充。每段均为 16 帧，
在索引 1、3、5、7、9、11、13 处进行七次比较。
原始第 15 帧缺少右侧端点，不参与评分。不修改源视频，不重采样帧率。

## 官方对照与运行兼容性

运行脚本通过抽象语法树（AST）提取未经修改的上游类，
加载独立的官方 `MotionSmoothness` 实例，逐项比较重建 MAE 和最终得分。
两段真实视频的各项误差与最终得分均完全一致。
此方式绕过了无关的 VBench 导入，没有绕过或替换 AMT-S 运算。

完整命令行流程、分布式评测，以及 MPS 和 CUDA 的数值一致性仍未验证。
本机 Mac ARM64 已成功运行 CPU `grid_sample` 和完整真实视频推理；
当前执行环境中 MPS 不可用。

## 结果保存与复现记录

每次运行新建 `results/motion_<UTC时间戳>/`，保存各视频的重建误差 CSV、
PNG 曲线、得分 CSV、报告和 `runs.json`。
运行清单记录代码、源码、权重和输入文件哈希，以及环境、CPU 线程数、耗时、
进程峰值内存和预处理参数。

总耗时包含检查、本地实现推理、官方对照的第二次推理和曲线输出，
每次推理的独立耗时另外记录在运行清单中。
得分按原有列结构、通过文件锁追加到统一 `metric_scores.csv`，保留历史字节。
已有指标模块、数据以及 Temporal Flickering、Subject Consistency 的结果，均通过哈希核对未发生变化。
旧的全局 `runs.json` 不变，本次运行使用独立的 `runs.json`。

## 测试与解释边界

测试覆盖 RGB 解码、非法输入、缩放和填充、除以 255 的归一化、帧配对、
uint8 量化及评分公式。真实运行脚本还会回读保存的 CSV，核对得分与官方逐帧误差，
并检查历史文件是否保留。
插值测试替身仅用于单元测试；所有已报告的真实视频分数均使用已确认的官方 AMT-S 权重。

该得分衡量插值重建一致性，不直接衡量动作真实性，也不能根据两段视频判断模型整体优劣。

本说明记录独立复现阶段的实现。三个指标现已接入统一 Pipeline 和 Gradio，
当前 Demo 的启动与使用方法见 `README.md`。
