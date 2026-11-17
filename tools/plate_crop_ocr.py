"""车牌区域裁剪放大实验：把最佳帧截图的下半部分裁出来、放大 2 倍再 OCR，
与整图 OCR 的基线对比。

背景：货车后视的车牌与尾门喷涂字都在车辆框的下半部分；整图 OCR 时
车牌只占一小块，检测器容易漏检（基线：113 张截图仅 33 张返回文本、
7 张命中车牌）。裁出下半部分并放大后，文字相对尺寸变大，检测与
识别都更容易。

用法（free-machine，video-ocr 环境，边车已启动）：
    /data/miniconda3/envs/video-ocr/bin/python tools/plate_crop_ocr.py \
        --crops /data/best_frames --out-dir /data/plate_crops \
        --results /data/ocr_run4.txt --plate MD598

输出：
- 放大后的裁剪图写到 --out-dir（与原图同名）；
- 识别结果写到 --results（每行：文件名<TAB>识别文本）；
- 最后打印统计：总数、非空数、含 --plate 的张数。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


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


def main() -> int:
    parser = argparse.ArgumentParser(description="车牌区域裁剪放大 OCR 实验")
    parser.add_argument("--crops", required=True, help="最佳帧截图目录")
    parser.add_argument("--out-dir", required=True, help="裁剪放大图的输出目录")
    parser.add_argument("--results", required=True, help="识别结果输出文件")
    parser.add_argument("--sidecar", default="http://127.0.0.1:8000", help="边车地址")
    parser.add_argument("--bottom-frac", type=float, default=0.5,
                        help="裁剪下半部分的比例（默认 0.5，即下半部分）")
    parser.add_argument("--scale", type=float, default=2.0, help="放大倍数（默认 2）")
    parser.add_argument("--plate", default="",
                        help="用于统计的车牌关键字符（如 MD598）；不给则不统计命中")
    args = parser.parse_args()

    crops = sorted(Path(args.crops).glob("track_*.jpg"))
    if not crops:
        print(f"[error] no track_*.jpg found in {args.crops}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nonempty = 0
    plate_hits = 0
    lines = []
    for crop in crops:
        img = Image.open(crop).convert("RGB")
        w, h = img.size
        y0 = int(h * (1 - args.bottom_frac))
        region = img.crop((0, y0, w, h))
        big = region.resize((int(w * args.scale), int((h - y0) * args.scale)),
                            Image.LANCZOS)
        out_path = out_dir / crop.name
        big.save(out_path, quality=92)
        text = ocr_file(out_path, args.sidecar)
        if text.strip():
            nonempty += 1
        if args.plate and args.plate in text:
            plate_hits += 1
        lines.append(f"{crop.name}\t{text}")
        print(f"{crop.name}\t{text}", flush=True)

    Path(args.results).write_text("\n".join(lines) + "\n", encoding="utf-8")
    elapsed = time.time() - t0
    print("\n=== summary ===", flush=True)
    print(f"total: {len(crops)}", flush=True)
    print(f"non-empty: {nonempty} ({nonempty / len(crops) * 100:.1f}%)", flush=True)
    if args.plate:
        print(f"plate hits ({args.plate}): {plate_hits}", flush=True)
    print(f"elapsed: {elapsed:.1f}s, avg {elapsed / len(crops):.2f}s/call", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
