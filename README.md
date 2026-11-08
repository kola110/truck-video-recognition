# 货车视频识别：带框 RTSP 流

用 YOLO 给 RTSP 视频流加上目标框，再推流出去（或只在本地预览）。

## 功能

- `result.plot()` 画目标框、类别和置信度
- RTSP 强制走 TCP，断线自动重连
- 读帧线程 + 队列，丢弃过期帧
- `--skip N` 跳帧推理，间隔帧复用上一轮的框
- 可选 ffmpeg 推流（RTMP）到 ZLMediaKit，前端用 FLV/HLS 观看

## 环境（conda）

本项目 Python 部分统一用 conda 环境运行：

```bash
# 创建环境（Python 3.11，ultralytics、OpenCV、ffmpeg 都走 conda-forge）
conda create -n truck-video python=3.11 -y
conda activate truck-video
conda install -c conda-forge ultralytics opencv ffmpeg -y
```

以后每次运行前先激活环境：

```bash
conda activate truck-video
```

## 用法

```bash
conda activate truck-video

# 推流到 ZLMediaKit
python rtsp_yolo_annotated.py --source "rtsp://..." --weights yolo26n.pt --push "rtmp://127.0.0.1:1935/live/yolo_annotated"

# 只在本地预览
python rtsp_yolo_annotated.py --source "rtsp://..." --no-push --show

# 先用本地视频文件验证（不用摄像头和推流）
python rtsp_yolo_annotated.py --source ./truck.mp4 --weights yolo26n.pt --no-push --show
```

ZLMediaKit 默认观看地址（以你自己的配置为准）：

- `http://<zlm-ip>:80/live/yolo_annotated.live.flv`
- `http://<zlm-ip>:80/live/yolo_annotated/hls.m3u8`

## 状态说明

- 默认权重 `yolo26n.pt` 首次运行时自动下载；也可以换成自己的货车模型权重。
- 带框流只证明车辆检测：车牌和车门文字识别是第二阶段，需要专门的检测 + OCR。
- 本脚本目前只做了语法检查，尚未在真实摄像头或推流环境里完整跑通过。
