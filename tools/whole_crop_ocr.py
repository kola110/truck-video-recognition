"""整图 OCR：把最佳帧截图发给 PaddleOCR 边车，逐张读出全部文字。

用途：验证车门文字（公司名、总质量、限座人数等印在车身侧面的文字）
以及补查车牌检测器漏掉的车牌。检测器只在找到车牌框时才 OCR，
整图 OCR 不依赖检测框，能读出图里所有文字——代价是输出更杂，
需要人工或后续步骤挑出想要的字段。

用法：
    python3 tools/whole_crop_ocr.py --crops /data/best_frames_street \
        --results /data/ocr_street_run2.txt

输出文件每行一张图，Tab 分隔：文件名<TAB>识别文本（边车把各行
文字用空格连成一段）。空识别的行文本为空，但仍保留文件名，
便于和检测器轮的结果逐行对齐。格式和裁剪轮（plate_crop_ocr.py）
一致，plate_vote.py 可以直接解析。

速度：街景大截图（最大 2.6M 像素）整图 OCR 约 13 秒/张，91 张
约 22 分钟。两个提速选项（可同时用）：
- --crop-frac x0 y0 x1 y1：按比例裁一块区域再识别。车门文字只在
  车身侧面中段，截图上沿的时间戳水印和下沿的地面都是干扰；
  街景视频上标定的默认值是 0 0.15 1 0.95（去顶部水印、留全部
  车门文字，留有余量）。车头朝向不同时文字带左右位置会变，
  所以默认不裁左右，只裁上下。
- --max-side N：把长边缩到 N 像素再识别（如 1280）。车门字本身
  够大，缩小后一般还认得出；像素少了，检测和识别都快。

本脚本用 PIL 做裁剪和缩小（需 Pillow），请用 video-ocr 的 conda
环境跑：/data/miniconda3/envs/video-ocr/bin/python。
"""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image


def prepare_image(path: Path, crop_frac: tuple | None, max_side: int) -> bytes:
    """打开一张截图，按需裁剪、缩小，返回 JPEG 字节串。

    crop_frac 是 (x0, y0, x1, y1) 四个 0~1 的比例，表示在原图里
    截取的区域；为 None 时不裁剪。max_side > 0 时，若图片长边
    超过 max_side，等比缩小到长边等于 max_side；否则不缩小。
    """
    img = Image.open(path).convert("RGB")
    width, height = img.size
    if crop_frac is not None:
        x0, y0, x1, y1 = crop_frac
        # 比例转像素，并夹到图片范围内，防止越界。
        left = max(0, int(width * x0))
        top = max(0, int(height * y0))
        right = min(width, int(width * x1))
        bottom = min(height, int(height * y1))
        if right <= left or bottom <= top:
            raise ValueError(f"invalid crop region {crop_frac} for {width}x{height}")
        img = img.crop((left, top, right, bottom))
        width, height = img.size
    if max_side > 0:
        long_side = max(width, height)
        if long_side > max_side:
            scale = max_side / long_side
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def ocr_image_bytes(data: bytes, sidecar: str) -> str:
    """把一张图的字节串以 multipart 表单发给边车，返回识别出的文字。

    边车 /ocr 接口的字段名是 image（见 ocr-service/server.py），
    返回 {"text": "各行用空格连接", "lines": [...]}，这里只取 text。
    """
    # 手工拼 multipart 请求体：只用标准库，不依赖 requests。
    boundary = "----wholecropocr" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="image"; '
        + b'filename="crop.jpg"\r\n'
        + b"Content-Type: image/jpeg\r\n\r\n"
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{sidecar}/ocr",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("text", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="整图 OCR：逐张读出最佳帧截图里的全部文字")
    parser.add_argument("--crops", required=True, help="最佳帧截图目录")
    parser.add_argument("--results", required=True, help="识别结果输出文件（每行一张图）")
    parser.add_argument("--sidecar", default="http://127.0.0.1:8000", help="边车地址")
    parser.add_argument("--crop-frac", nargs=4, type=float, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="按比例裁取区域再识别，如 0 0.15 1 0.95；不给则整图识别")
    parser.add_argument("--max-side", type=int, default=0,
                        help="长边超过此值时等比缩小，如 1280；0 表示不缩小")
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    if not crops_dir.is_dir():
        print(f"[error] not a directory: {crops_dir}", file=sys.stderr)
        return 1
    # 只取 jpg，跳过 best.json / merged_best.json 等清单文件。
    images = sorted(crops_dir.glob("*.jpg"))
    if not images:
        print(f"[error] no .jpg files in {crops_dir}", file=sys.stderr)
        return 1
    if args.crop_frac is not None:
        x0, y0, x1, y1 = args.crop_frac
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            print(f"[error] invalid --crop-frac {args.crop_frac}, "
                  "need 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1", file=sys.stderr)
            return 1

    # 先探一张：如果边车没启动，立刻报错退出，不写一堆空结果。
    try:
        first_data = prepare_image(images[0], args.crop_frac, args.max_side)
        first_text = ocr_image_bytes(first_data, args.sidecar)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] cannot reach sidecar at {args.sidecar}: {exc}", file=sys.stderr)
        return 1

    non_empty = 0
    start = time.perf_counter()
    with open(args.results, "w", encoding="utf-8") as out:
        for index, image in enumerate(images):
            # 第一张已经识别过，直接用探针结果，不重复调边车。
            if index == 0:
                text = first_text
            else:
                try:
                    data = prepare_image(image, args.crop_frac, args.max_side)
                    text = ocr_image_bytes(data, args.sidecar)
                except Exception as exc:  # noqa: BLE001 - 单张失败不中断整批
                    print(f"[warn] {image.name} failed: {exc}", file=sys.stderr)
                    text = ""
            if text.strip():
                non_empty += 1
            out.write(f"{image.name}\t{text}\n")
            print(f"[{index + 1}/{len(images)}] {image.name}: {text[:60]}")

    elapsed = time.perf_counter() - start
    avg = elapsed / len(images)
    print(f"\ndone: {len(images)} images, {non_empty} non-empty, "
          f"{elapsed:.1f}s total, {avg:.2f}s/image, results in {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
