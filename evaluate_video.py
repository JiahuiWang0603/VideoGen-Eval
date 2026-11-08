import cv2
import numpy as np
import pandas as pd
from pathlib import Path

video_path = "videos/test.mp4"
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError("视频无法打开，请检查路径")

results = []
prev_gray = None
frame_id = 0

# 逐帧处理
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    if prev_gray is not None:
        # 计算相邻帧像素差异
        diff = cv2.absdiff(prev_gray, gray)

        # 归一化变化得分
        score = np.mean(diff) / 255.0

        results.append({
            "frame_id": frame_id,
            "change_score": float(score)
        })

    prev_gray = gray
    frame_id += 1

cap.release()

# 保存评测结果
df = pd.DataFrame(results)
df.to_csv(
    output_dir / "frame_scores.csv",
    index=False
)

print(f"视频总帧数：{frame_id}")

if not df.empty:
    print(f"平均变化得分：{df['change_score'].mean():.4f}")

print("评测完成！")
