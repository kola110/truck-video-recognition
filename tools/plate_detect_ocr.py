"""专用车牌检测器实验：用 CCPD 训练的 YOLO26 车牌检测模型定位车牌框，
裁出车牌区域、放大 2 倍再调边车 OCR，与整图 OCR 基线对比。

模型：puhaiyang/OpenTrafficFlow 的 weights/best.pt（YOLO26，CCPD 训练，
15.5 MB）。下载：
    curl -L -o /data/models/plate_best.pt \
      https://raw.githubusercontent.com/puhaiyang/OpenTrafficFlow/945486e02c083f0dfb42f2514b11c84092e8d952/weights/best.pt
    # 核对大小应为 15532512 字节
该仓库的完整管线还用 LPRNet 做车牌识别；本实验只用其检测器，识别
仍走 PaddleOCR 边车，以便单独衡量检测器的作用。

用法（free-machine，truck-video 环境跑 YOLO，边车已启动）：
    # 烟雾测试（单张整帧）
    /data/miniconda3/envs/truck-video/bin/python tools/plate_detect_ocr.py \
        --single /tmp/probe60.jpg --model /data/models/plate_best.pt \
        --out-dir /data/plate_detections --plate MD598
    # 全量实验（113 张最佳帧截图）
    /data/miniconda3/envs/truck-video/bin/python tools/plate_detect_ocr.py \
        --crops /data/best_frames --model /data/models/plate_best.pt \
        --out-dir /data/plate_detections --results /data/ocr_run5.txt \
        --detections /data/plate_detections.json --plate MD598 \
        --plate-exact "浙M·MD598" --conf 0.3

输出：
- 车牌裁剪图写到 --out-dir（<原名>_p<序号>.jpg）；
- --results：每行一张原图：文件名<TAB>检测数<TAB>识别文本（多块用 | 连接）；
- --detections：JSON，含每张图的检测框、置信度与识别文本；
- 最后打印统计：有检测的图数、总检测数、车牌命中图数、精确命中图数。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO


def ocr_file(path: Path, sidecar: str) -> str:
    """调边车 /ocr，返回识别文本（失败返回空字符串并打印警告）。"""
    try:
        out = subprocess.run(
            ["curl", "-s", "-F", f"image=@{path}", f"{sidecar}/ocr"],
            capture_output=True, text=True, timeout=120,
        )
        return json.loads(out.stdout).get("text", "")
    except Exception as exc:  # noqa: BLE001 - 实验脚本，单张失败不中断
        print(f"[warn] OCR failed for {path.name}: {exc}", file=sys.stderr)
        return ""


def process_image(model, path, out_dir, sidecar, conf, scale, pad_frac):
    """检测一张图里的车牌，裁剪放大后逐块 OCR，返回检测记录列表。"""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[warn] cannot read {path}", file=sys.stderr)
        return []
    results = model(img, conf=conf, verbose=False)
    records = []
    h, w = img.shape[:2]
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad_frac), int(bh * pad_frac)
        x1c, y1c = max(0, x1 - px), max(0, y1 - py)
        x2c, y2c = min(w, x2 + px), min(h, y2 + py)
        crop = Image.fromarray(cv2.cvtColor(img[y1c:y2c, x1c:x2c],
                                            cv2.COLOR_BGR2RGB))
        big = crop.resize((crop.width * scale, crop.height * scale),
                          Image.LANCZOS)
        out_path = out_dir / f"{path.stem}_p{i}.jpg"
        big.save(out_path, quality=92)
        text = ocr_file(out_path, sidecar)
        records.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "conf": round(float(box.conf[0]), 3),
            "crop": out_path.name,
            "text": text,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="专用车牌检测器 OCR 实验")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", help="只处理单张图片（烟雾测试）")
    group.add_argument("--crops", help="最佳帧截图目录（全量实验）")
    parser.add_argument("--model", required=True, help="车牌检测权重 .pt 路径")
    parser.add_argument("--out-dir", required=True, help="车牌裁剪图输出目录")
    parser.add_argument("--results", help="识别结果输出文件（每行一张原图）")
    parser.add_argument("--detections", help="检测详情 JSON 输出文件")
    parser.add_argument("--sidecar", default="http://127.0.0.1:8000",
                        help="边车地址")
    parser.add_argument("--conf", type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--scale", type=int, default=2, help="裁剪放大倍数")
    parser.add_argument("--pad-frac", type=float, default=0.05,
                        help="裁剪时框外扩比例（相对框宽高）")
    parser.add_argument("--plate", default="",
                        help="用于统计的车牌关键字符（如 MD598）")
    parser.add_argument("--plate-exact", default="",
                        help="用于统计的精确车牌字符串（如 浙M·MD598）")
    args = parser.parse_args()

    if args.single:
        images = [Path(args.single)]
    else:
        images = sorted(Path(args.crops).glob("track_*.jpg"))
    if not images:
        print("[error] no images found", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading plate model: {args.model}", flush=True)
    model = YOLO(args.model)

    t0 = time.time()
    n_with_det = 0
    n_dets = 0
    plate_hits = 0
    exact_hits = 0
    result_lines = []
    all_dets = {}
    for path in images:
        records = process_image(model, path, out_dir, args.sidecar,
                                args.conf, args.scale, args.pad_frac)
        all_dets[path.name] = records
        if records:
            n_with_det += 1
            n_dets += len(records)
        texts = [r["text"] for r in records if r["text"].strip()]
        if args.plate and any(args.plate in t for t in texts):
            plate_hits += 1
        if args.plate_exact and any(args.plate_exact in t for t in texts):
            exact_hits += 1
        line = f"{path.name}\t{len(records)}\t{' | '.join(texts)}"
        result_lines.append(line)
        print(line, flush=True)

    if args.results:
        Path(args.results).write_text("\n".join(result_lines) + "\n",
                                      encoding="utf-8")
    if args.detections:
        Path(args.detections).write_text(
            json.dumps(all_dets, ensure_ascii=False, indent=2),
            encoding="utf-8")
    elapsed = time.time() - t0
    print("\n=== summary ===", flush=True)
    print(f"total images: {len(images)}", flush=True)
    print(f"images with >=1 detection: {n_with_det}", flush=True)
    print(f"total detections: {n_dets}", flush=True)
    if args.plate:
        print(f"plate hits ({args.plate}): {plate_hits}", flush=True)
    if args.plate_exact:
        print(f"exact plate hits ({args.plate_exact}): {exact_hits}",
              flush=True)
    print(f"elapsed: {elapsed:.1f}s, avg {elapsed / len(images):.2f}s/image",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
