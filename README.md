# 三指标测评

已接入 Temporal Flickering、Subject Consistency、Motion Smoothness，分别复用现有
RGB MAE、DINO ViT-B/16、AMT-S 模块，不更改官方公式。支持单视频或双视频同指标并排比较。

## 本地启动

```bash
cd /Users/wangjiahui/Desktop/VideoGen-Eval
GRADIO_SERVER_PORT=7861 MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python app.py
```

1. 上传视频 A；需要比较时再上传 B。两者必须为本地 MP4。
2. 填写各自模型名称和共同 Prompt；Prompt 是用户标签，不是验证过的生成来源。
3. 选择一个或多个指标，点击 Evaluate；A、B 顺序运行相同指标。
4. 在各指标页并排查看得分、耗时、逐帧曲线；在「汇总与下载」下载 CSV。
5. 「运行记录」展示子 run_id、视频规格、适用性和错误详情。

每个视频生成独立 `pipeline_*` run_id，保存输入原始字节副本、内容哈希、分数和曲线，
`runs.json` 包含依赖、模型权重版本、代码哈希与处理参数。
每次按钮操作另建 `comparison_*` 目录，保存 `comparison_scores.csv` 和引用子运行的
`runs.json`。共享 metric_scores.csv 仅追加，历史字节和旧目录不覆盖。
模型加载缓存会让首个请求比后续请求慢；耗时包含解码、初次加载和曲线生成，不能当纯推理基准。

Motion Smoothness 曲线显示中间帧重建 MAE；其端点索引等中间统计在逐帧 CSV，
实际 pad/scale/采样参数在子 runs.json。只画重建误差，不混画端点索引或归一化误差。

Temporal Flickering 未执行官方静态筛选，正常运动会降低分数，不能确定为闪烁缺陷。
Subject Consistency 是全画面特征相似度，不是身份识别准确率。
Motion Smoothness 衡量 AMT-S 插值重建一致性，不直接代表动作真实性。
三者不合成总分；两个不同规格样本不足以判定模型总体优劣。

## 测试与真实验证

```bash
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python -m unittest discover -s tests -v
# 另一个终端启动上述 Demo 后执行真实 HTTP 上传/推理/下载回归：
DEMO_URL=http://127.0.0.1:7861 .venv-subject/bin/python scripts/verify_demo_e2e.py
```

已通过 29 项自动化测试。HTTP 回归覆盖两模型游泳视频的全部三个指标、两段停止标志的
Temporal Flickering、双视频曲线/CSV 下载，以及单个损坏 MP4 的失败分数与清空曲线。
缺失 DINO/AMT-S 权重测试通过拦截文件存在性检查触发真实加载器错误路径，
未删除、移动或替换实际权重；UI 验证失败分数留空，其他指标继续执行。
每次回归会新增真实运行记录；现有结果不会覆盖。

真实 HTTP 验证记录：
`results/comparison_20261108T160643265971Z_7a343e9e/http_e2e_validation.json`。
与历史得分的最大差异约 1.1e-16，属于 CSV 浮点读取精度。

---

以下为之前版本的运行和数据准备记录；如与上文冲突，以本节三指标说明为准。

# VideoGen-Eval：统一评测与本地 Demo

当前支持 Temporal Flickering 和 Subject Consistency；复用 `metrics/` 中已有实现。
原始视频、数据表、既有指标和实验结果保留。Motion Smoothness 尚未开放。

## 启动 Gradio Demo（Mac）

在终端完整执行：

```bash
cd /Users/wangjiahui/Desktop/VideoGen-Eval
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python app.py
```

当前 `.venv-subject` 已安装完整依赖。新建环境可使用 Python 3.9 安装已冻结依赖：

```bash
python3 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-demo.lock.txt
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-demo/bin/python app.py
```

本地 DINO 文件须位于 `models/dino/dino_vitbase16_pretrain.pth`，来源与 SHA-256 在同目录 JSON。
系统只读取既有权重，不自动下载；缺失或校验失败时 Subject Consistency 显示错误、分数留空，
另一个指标仍会执行。无需安装 AMT-S。Gradio 固定 4.44.1、Pydantic 2.10.6，以兼容当前 Python 3.9。

## 统一 CLI / Python Pipeline

替换下方 MP4 路径即可评测任意本地 MP4，无需修改 metadata.csv：

```bash
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python scripts/evaluate.py \
  "videos/lavie/In a still frame, a stop sign-0.mp4" \
  --model-name lavie --prompt "In a still frame, a stop sign" \
  --metrics temporal_flickering subject_consistency
```

`--metrics` 可只选一个；`--video-id` 可选，留空则取视频内容 SHA-256 的前 16 位。
Python 入口是 `pipeline.evaluate(video_path, model_name, prompt, video_id, metrics)`。
模型名和 Prompt 是用户提供的标签，不会自动变成官方数据来源证明。

每次生成独立 `results/pipeline_<时间>_<随机标识>/`，包含：

- `input.mp4`：原始字节副本及记录的 SHA-256，避免上传临时文件失效后无法复现。
- `metric_scores.csv`：本次完整结果，含输入标签、状态、错误原因和耗时。
- `<metric>_curve.csv` / `.png`：逐帧曲线。
- `runs.json`：视频帧数、FPS、分辨率、环境依赖、代码哈希、权重版本及预处理。

共享 `results/metric_scores.csv` 按原有列结构追加，加文件锁，保留历史字节；Prompt 等扩展信息
保存在本次结果表和 runs.json 中。旧的全局 runs.json 和历史运行目录不改写。
损坏视频会为所选指标保存失败记录；输入路径不存在或未选择指标时直接提示输入错误。
单项失败不会伪造分数或阻止另一个指标执行。
CPU 推理，缓存 DINO；Demo 队列逐请求执行。当前实现会全帧读入内存，优先使用短片，
长视频资源消耗与帧数增长；完整性检查不能保证发现所有容器尾部损坏。

Temporal Flickering 使用全部彩色通道的官方 MAE 公式，未运行官方静态筛选。
动态视频可以计算，但正常运动同样降低分数，页面会明确提示，不能解释为确定的闪烁缺陷。
Subject Consistency 保留现有 DINO / 预处理 / 首帧与前帧相似度逻辑；OpenCV 与官方 decord
逐像素一致性仍未验证。主体一致性不是身份识别准确率，少量视频不能支持模型总体排名。

## 验证

```bash
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv-subject/bin/python scripts/verify_pipeline.py
```

第二条使用现有四个视频与历史分数做回归比较，并创建新的真实运行结果。
历史一致性、曲线长度和旧文件哈希检查记录在新目录的 regression_validation.json。
已通过真实 Gradio HTTP 上传、双指标计算、两张曲线及三个 CSV 下载测试；
记录见 `results/pipeline_20261108T153726736954Z_ac8ba194/demo_http_validation.json`。

---

以下保留早期数据准备和单指标运行记录；其中“初始状态”等描述为历史状态。

# VideoGen-Eval 小型数据准备

现有 `evaluate_video.py` 原样保留：它读取 `videos/test.mp4`，计算相邻帧灰度像素差。
这不是 VBench 的正式 temporal_flickering、motion_smoothness 或 subject_consistency 分数。
本次新增脚本仅整理数据和检查可读性，不生成评测结果。

## 已核实的数据组织

- [官方视频说明](https://github.com/Vchitect/VBench/blob/master/sampled_videos/README.md)：视频按 `per_dimension` / `per_category` 和模型打包。
- [官方 Prompt 说明](https://github.com/Vchitect/VBench/blob/master/prompts/README.md)：视频名为 `<完整 Prompt>-<序号>.mp4`。
- motion_smoothness 和 subject_consistency 使用同一个 subject_consistency Prompt 集；temporal_flickering 使用独立 Prompt 集。
- 实际维度归属以下载的 `data/VBench_full_info.json` 为准。两个 Prompt 文本和映射的 URL、内容 SHA-256 记录在 `data/sources.json`。
- temporal_flickering 正式评测涉及静态筛选及更多采样。本项目的 4 个候选视频只用于验证小型流程，不代表完整 VBench 或具有统计代表性的模型比较。

## 手动下载哪些文件

提供的 [Drive 文件夹](https://drive.google.com/drive/folders/1bm7zSu2wxT7wY8IpitmJc_7qi8EYMCim) 当前无法通过访问工具读取。
官方说明现列出的 [Drive 根目录](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X) 与上述 ID 不同。
未下载任何视频，未核验压缩包内部目录，不能保证候选文件实际存在。

优先在 `per_dimension` 中查找：

| 模型目录 | 官方文件 | 官方链接 |
| --- | --- | --- |
| `videos/lavie/` | `lavie.zip` | [LaVie per-dimension](https://drive.google.com/file/d/1hviZzsInIgJA96ppVj4B2DHhTZWeM4nc/view) |
| `videos/modelscope/` | `modelscope.zip` | [ModelScope per-dimension](https://drive.google.com/file/d/1UH2-lALFShjBywyImjDPPHTpE43eoMQE/view) |

每个模型仅需要这两个视频，文件名依据官方命名规则推导，尚未核实：

```text
a person swimming in ocean-0.mp4
In a still frame, a stop sign-0.mp4
```

完整目标清单在 `data/download_plan.csv`，状态为 `pending_download`。
优先在 Drive 预览中查找并单独下载这四个文件。如果 Drive 仅提供 ZIP、不能逐文件下载，
则手动下载上述两个模型压缩包，解压后仅复制目标文件到相应模型目录；不要下载整个文件夹或其他模型/类别数据。
压缩包大小尚未核实。如果不希望下载模型级压缩包，应等待可单文件下载的来源。
如果序号 0 不存在，可取另一个实际存在的数字序号，保留原文件名；不要凭空创建文件或改写 Prompt。
脚本递归扫描这两个模型目录，可以保留解压后的子目录。其他文件名规则会被明确跳过，需另行核实后适配。
两个模型有相同 Prompt 即为配对；相同序号不意味着相同随机种子。

## Mac 本地运行

当前 `.venv` 中 OpenCV 和 Pandas 已可导入。从项目根目录执行：

```bash
# 新环境才需要安装
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 下载并放入真实视频后执行；从项目之外运行脚本也能正确解析路径
.venv/bin/python scripts/prepare_metadata.py
.venv/bin/python scripts/check_data.py
```

Python 至少 3.9。无需 GPU、PyTorch 或完整 VBench 安装。
`prepare_metadata.py` 会重新生成 metadata.csv，因此不要手动在该生成文件中维护额外记录。
它只登记两个已核实模型目录中真实存在、文件名 Prompt 精确匹配官方映射的 MP4；能否解码由检查脚本判断。
每行是一个视频在一个维度中的记录，同一视频可对应多个维度并共享 video_id。
source 是官方模型压缩包链接；目录名是人工声明的模型来源，脚本无法从视频像素鉴定模型。

`check_data.py` 检查必填字段、官方 Prompt/维度对应、模型来源、重复记录、路径、文件存在及全片逐帧解码。
同一路径只解码一次；少于两帧、无法打开或解码帧数少于容器报告帧数都会报错。
OpenCV 在部分编解码器上不能区分正常结束和尾部损坏，检查通过不能保证字节级完整性。
跨模型覆盖仅统计元数据有效且可读取的视频，按 `(prompt, dimension)` 分组。
输出 `results/data_check.csv` 和 `results/prompt_coverage.csv`。
无数据、数据错误、缺少任一目标维度或任何 Prompt/维度无跨模型配对时，退出码为 1；就绪为 0。
无数据时只打印提示，不生成检查报告。

初始状态：metadata.csv 仅表头，尚无实际视频和跨模型配对，未计算任何模型分数。

## Temporal Flickering 第一阶段复现

独立模块：`metrics/temporal_flickering.py`；原始灰度基线不变。
固定的官方源码在 `vendor/temporal_flickering.py.txt`，提交与 SHA-256 见旁边的 JSON。
只用 OpenCV 全帧解码，保留 BGR 三通道和原始尺寸、FPS；float32 做绝对差，
逐对求全部像素及通道的 MAE，再取时间平均，得分为 `(255 - mean_MAE) / 255`。
BGR 与 RGB 的通道排列不影响三通道整体均值。

```bash
.venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/videogen-eval-matplotlib .venv/bin/python scripts/run_temporal_flickering.py
```

结果追加到 `results/metric_scores.csv`；环境、参数、源码版本保存在 `results/runs.json`。
每次运行的独立目录内含两段视频的帧间 MAE CSV 与 PNG 曲线。
耗时含解码、计算、官方核心函数核对及曲线输出，不是纯指标推理耗时。

`data/video_provenance.csv` 单独登记源 Prompt 子目录、压缩包 URL 与视频哈希。
子目录依据手动解压上下文，未独立读取压缩包成员清单；不把任务名称当作源路径。
`data/evaluation_tasks.csv` 登记六项计划任务，本阶段只执行两个 temporal_flickering 任务。
运行器重建这两张派生表，保留原始 metadata.csv、evaluate_video.py 与视频。

六项测试覆盖完全相同帧、全亮度突变、规则平移、彩色通道、时间均值和非法输入。
规则运动也会降低该指标，说明必须使用静态内容，不能把运动像素差都解释为闪烁。
运行器用 AST 提取固定官方源码的解码和数值函数，以同一 OpenCV/NumPy 环境核对
每个视频的全部帧间 MAE 和视频得分；没有导入完整 VBench 或分布式模块。
对正常输入已验证完全一致；非法输入采取显式报错，不复现官方空序列的 NaN 行为。
仅人工审阅停止标志片段及 Prompt，尚未运行官方自动静态过滤，
也未验证跨解码器、完整 CLI、分布式汇总或完整 Benchmark 的可比性。
不得据这两个视频判断模型整体优劣；分辨率和水印等也会影响像素指标。
