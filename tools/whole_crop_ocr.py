"""整图 OCR：把最佳帧截图原样发给 PaddleOCR 边车，逐张读出全部文字。

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

只依赖标准库（urllib），系统 python3 就能跑；边车地址默认
http://127.0.0.1:8000，可用 --sidecar 改。
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def ocr_image(path: Path, sidecar: str) -> str:
    """把一张图以 multipart 表单发给边车，返回识别出的文字。

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
        + path.read_bytes()
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

    # 先探一张：如果边车没启动，立刻报错退出，不写一堆空结果。
    try:
        first_text = ocr_image(images[0], args.sidecar)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[error] cannot reach sidecar at {args.sidecar}: {exc}", file=sys.stderr)
        return 1

    non_empty = 0
    with open(args.results, "w", encoding="utf-8") as out:
        for index, image in enumerate(images):
            # 第一张已经识别过，直接用探针结果，不重复调边车。
            if index == 0:
                text = first_text
            else:
                try:
                    text = ocr_image(image, args.sidecar)
                except Exception as exc:  # noqa: BLE001 - 单张失败不中断整批
                    print(f"[warn] {image.name} failed: {exc}", file=sys.stderr)
                    text = ""
            if text.strip():
                non_empty += 1
            out.write(f"{image.name}\t{text}\n")
            print(f"[{index + 1}/{len(images)}] {image.name}: {text[:60]}")

    print(f"\ndone: {len(images)} images, {non_empty} non-empty, results in {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
