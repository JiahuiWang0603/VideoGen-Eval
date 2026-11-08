"""Local three-metric Gradio demo with optional side-by-side video comparison."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
os.environ.setdefault('MPLCONFIGDIR','/tmp/videogen-eval-matplotlib')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
import gradio as gr
import pandas as pd
from pipeline import evaluate, METRICS, ROOT, sha

TITLES = dict(temporal_flickering='Temporal Flickering · 时间闪烁',
              subject_consistency='Subject Consistency · 主体一致性',
              motion_smoothness='Motion Smoothness · 运动平滑度')
EXPLANATIONS = dict(
    temporal_flickering='相邻彩色帧变化越小，得分越高。动态画面的正常运动也会降低得分；本次未执行官方静态筛选。',
    subject_consistency='与前帧、首帧的 DINO 全画面特征越相似，得分越高。不是主体身份识别准确率。',
    motion_smoothness='AMT-S 对中间帧的重建误差越小，得分越高。不直接衡量动作真实性。')
CURVE_HELP = dict(
    temporal_flickering='横轴：目标帧索引。纵轴：与前一帧的彩色 MAE。峰值表示画面变化增大，可能来自运动、镜头或亮度变化，不能直接认定为闪烁缺陷。',
    subject_consistency='previous：与前帧相似度；first：与首帧相似度；mean：二者均值。最终分数为 mean 曲线的平均。主体、背景、姿态和构图均可能影响结果。',
    motion_smoothness='横轴：被重建的原始奇数帧索引。纵轴：AMT-S 预测帧与原帧的 RGB MAE。分数为 1 − 平均 MAE / 255；末帧缺少右端点时不参与比较。')
CSS = """
.gradio-container {max-width: 1360px !important;}
#hero {padding: 24px; border-radius: 18px; background: linear-gradient(120deg,#172554,#134e4a); margin-bottom: 16px;}
#hero h1, #hero p {color:white !important;}
.metric-card {padding: 18px; border: 1px solid var(--border-color-primary); border-radius: 14px; min-height: 170px;}
.metric-card h2 {font-size: 30px !important;}
"""


def metric_card(scores, name):
    title = TITLES[name]
    selected = scores[scores.metric_name == name]
    if selected.empty:
        return f'### {title}\n\n## —\n\n未选择此指标或未上传该视频。'
    row = selected.iloc[0]
    if row.status != 'ok':
        return f'### {title}\n\n## 计算失败\n\n分数留空。错误原因见「运行记录」。'
    return f'### {title}\n\n## {row.score:.6f}\n\n耗时 **{row.elapsed_seconds:.3f} 秒**\n\n{EXPLANATIONS[name]}'


def run_compare(video_a, model_a, video_b, model_b, prompt, metrics):
    if not video_a:
        raise gr.Error('请先上传视频 A；视频 B 可选。')
    if not metrics or set(metrics)-set(METRICS):
        raise gr.Error('请选择至少一个受支持的指标。')
    results = []
    for slot, video, model in [('A',video_a,model_a),('B',video_b,model_b)]:
        if video:
            try:
                result = evaluate(video,model,prompt,metrics=metrics)
            except Exception as exc:
                raise gr.Error(f'视频 {slot}: {exc}') from exc
            result['slot'] = slot
            results.append(result)
    run_id = datetime.now(timezone.utc).strftime('comparison_%Y%m%dT%H%M%S%fZ_')+uuid4().hex[:8]
    out = ROOT/'results'/run_id
    out.mkdir(parents=True,exist_ok=False)
    frames = [r['scores'].assign(comparison_slot=r['slot']) for r in results]
    scores = pd.concat(frames,ignore_index=True)
    csv = out/'comparison_scores.csv'
    scores.to_csv(csv,index=False)
    files = [str(csv)]
    messages = [f'Comparison run: {run_id}']
    for r in results:
        files += [r['csv']]+r['curves']
        messages += [f"视频 {r['slot']} Run: {r['run_id']}",f"视频规格：{r['video']}"]+r['warnings']
        messages += [f"视频 {r['slot']} / {v.metric_name}: {v.error_message}" for v in r['scores'].itertuples() if v.status!='ok']
    if len(results)==2:
        messages.append('并排结果仅描述这两个样本；分辨率、内容和视频长度差异可能影响比较，不据此判断模型总体优劣。')
    manifest = dict(run_id=run_id,created_at_utc=datetime.now(timezone.utc).isoformat(),
        prompt=prompt,metrics=list(metrics),app_sha256=sha(Path(__file__)),
        children=[dict(slot=r['slot'],run_id=r['run_id'],directory=r['directory']) for r in results],
        note='Each child runs.json contains its input hash, environment, model hashes and preprocessing.')
    (out/'runs.json').write_text(json.dumps([manifest],indent=2,ensure_ascii=False)+'\n')
    components = []
    for metric in METRICS:
        for slot in ['A','B']:
            selected = next((r for r in results if r['slot']==slot),None)
            if selected:
                images = {Path(p).name:p for p in selected['plots']}
                components += [metric_card(selected['scores'],metric),images.get(f'{metric}_curve.png')]
            else:
                components += [f'### {TITLES[metric]}\n\n## —\n\n未上传视频 B。',None]
    display_cols=['comparison_slot','model_name','metric_name','score','elapsed_seconds','status','error_message']
    return (scores[display_cols].fillna(''),files,'\n\n'.join(messages),*components)


def run_demo(video, model, prompt, video_id, metrics):
    # Compatibility wrapper for existing callers; the UI uses the comparison API.
    if not video:
        raise gr.Error('请先上传 MP4 视频。')
    return run_compare(video,model,None,'',prompt,metrics)


def build_demo():
    with gr.Blocks(title='VideoGen-Eval',theme=gr.themes.Soft(primary_hue='teal',neutral_hue='slate'),css=CSS) as demo:
        gr.Markdown('# VideoGen-Eval\n三个官方指标 · 单视频评测 / 双视频并排比较',elem_id='hero')
        with gr.Row():
            with gr.Column():
                gr.Markdown('### 视频 A · 必选')
                a=gr.File(label='上传原始 MP4 · A',file_types=['.mp4'],type='filepath')
                pa=gr.Video(label='视频 A 预览',interactive=False,height=220)
                ma=gr.Textbox(label='模型名称 A',placeholder='例如 LaVie')
            with gr.Column():
                gr.Markdown('### 视频 B · 可选')
                b=gr.File(label='上传原始 MP4 · B',file_types=['.mp4'],type='filepath')
                pb=gr.Video(label='视频 B 预览',interactive=False,height=220)
                mb=gr.Textbox(label='模型名称 B',placeholder='例如 ModelScope')
        a.change(lambda value:value,a,pa,api_name=False)
        b.change(lambda value:value,b,pb,api_name=False)
        prompt=gr.Textbox(label='共同 Prompt / 样本说明',lines=2)
        metrics=gr.CheckboxGroup(choices=[(TITLES[m],m) for m in METRICS],value=list(METRICS),label='选择评测指标（A / B 使用相同选择）')
        gr.Markdown('Temporal Flickering 适用于静态内容，正常运动也会降低其得分。原始 MP4 直接参与推理；浏览器预览编码兼容性不影响 OpenCV 读取。')
        button=gr.Button('Evaluate · 开始评测 / 比较',variant='primary')
        outputs=[]
        with gr.Tabs():
            for metric in METRICS:
                with gr.Tab(TITLES[metric]):
                    gr.Markdown(CURVE_HELP[metric])
                    with gr.Row():
                        for slot in ['A','B']:
                            with gr.Column():
                                gr.Markdown(f'### 视频 {slot}')
                                card=gr.Markdown('等待评测',elem_classes='metric-card')
                                image=gr.Image(label=f'{slot} · 逐帧曲线',type='filepath',interactive=False)
                                outputs += [card,image]
            with gr.Tab('汇总与下载'):
                table=gr.Dataframe(label='实际分数、耗时与状态',interactive=False)
                files=gr.File(label='下载：对比 CSV、各视频得分及逐帧 CSV',file_count='multiple')
            with gr.Tab('运行记录'):
                status=gr.Textbox(label='Run ID / 视频规格 / 适用性 / 错误原因',lines=12,interactive=False)
        gr.Markdown('三个指标独立展示，不合成为总分，也不设置未经验证的“优秀/合格”阈值。CPU 逐视频运行；首次运行包含模型加载，耗时不等于纯推理基准。')
        args=[a,ma,b,mb,prompt,metrics]
        callback_outputs=[table,files,status]+outputs
        button.click(run_compare,args,callback_outputs,api_name='evaluate',concurrency_limit=1)
    return demo


if __name__=='__main__':
    build_demo().queue().launch(server_name='127.0.0.1',server_port=int(os.environ.get('GRADIO_SERVER_PORT','7860')),share=False,inbrowser=False)
