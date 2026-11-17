"""车牌格式校验：从 OCR 文本里提取符合中国车牌格式的车牌，滤掉误读行。

背景：OCR 常常一图返回多行文本，其中混着误读（如把车牌读成
“浙M·IMD5”）和文字碎片。格式校验是架构评审里列的车牌后处理项：
只有符合车牌格式的才进入后续流程。

格式（GB/T《机动车号牌》）：
- 普通车牌：省份简称（1 个汉字）+ 城市字母（1 位）+ 圆点 + 5 位序号
  例：浙M·MD598
- 新能源车牌：省份简称 + 城市字母 + 圆点 + 6 位序号
  例：浙A·D12345
- 序号字母不含 I 和 O（易与 1、0 混淆）；OCR 可能把圆点读成 . 或漏掉，
  也可能在字符间插入空格，提取时先做归一化。

诚实说明：格式校验只能滤掉“格式不对”的行，拦不住“格式对但字错”
的读法——比如把 8 读成 9（浙M·MD5198）格式完全合法，校验会放行。
这类错误需要多截图投票来纠正，不在本模块范围。

用法：
    from plate_format import extract_plate, is_valid_plate
    extract_plate("浙M·IMD5 浙M·MD598")  # -> "浙M·MD598"
    is_valid_plate("浙M·MD598")          # -> True
    is_valid_plate("浙M·IMD5")           # -> False（只有 4 位序号，且含 I）

命令行：对 OCR 结果文件逐行提取车牌（逐行扫描，不依赖列格式，
因为文件名和组号里不会出现省份汉字，不会误匹配）：
    python tools/plate_format.py --results /data/ocr_run6.txt
    # 每行输出：原行内容<TAB>提取到的车牌（无则为 -）
    # 最后打印统计：有车牌的行数、去重后的车牌列表
"""

import argparse
import re
import sys
from pathlib import Path

# 31 个省份简称
PROVINCES = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
# 车牌字母表：不含 I、O
LETTERS = "A-HJ-NP-Z"
# 序号字符：字母（不含 I、O）+ 数字
SEQ = f"[{LETTERS}0-9]"

# 提取用：新能源（6 位序号）优先于普通（5 位序号）
_PLATE_RE = re.compile(
    f"([{PROVINCES}])([{LETTERS}])[·.•]?({SEQ}{{6}}|{SEQ}{{5}})"
)
# 严格校验用：标准形式 省份+城市+圆点+5/6 位序号
_VALID_RE = re.compile(
    f"^[{PROVINCES}][{LETTERS}]·{SEQ}{{5,6}}$"
)


def _normalize(text: str) -> str:
    """归一化：去空白、ASCII 转大写（省份汉字不受影响）。"""
    return re.sub(r"\s+", "", text).upper()


def extract_plate(text: str) -> str | None:
    """从一段文本里提取第一个符合车牌格式的车牌，返回标准形式。

    标准形式：省份+城市+圆点+序号，如“浙M·MD598”。
    找不到返回 None。

    两遍匹配：先在原文上找，再试去空格后的文本。去空格会把相邻
    碎片拼成假车牌（如“浙M·MD5 ME 浙M·MD598”去空格后会先匹配到
    “浙M·MD5ME”，把后面真正的车牌挤掉），所以原文优先；去空格
    只用于恢复“浙M·MD5 98”这类字符间带空格的真车牌。
    """
    if not text:
        return None
    match = _PLATE_RE.search(text.upper())
    if match is None:
        match = _PLATE_RE.search(_normalize(text))
    if match is None:
        return None
    province, city, seq = match.groups()
    return f"{province}{city}·{seq}"


def extract_plate_from_lines(lines) -> str | None:
    """从多行 OCR 文本里提取车牌：逐行找，返回第一个命中的。"""
    for line in lines:
        plate = extract_plate(line)
        if plate is not None:
            return plate
    return None


def is_valid_plate(text: str) -> bool:
    """严格校验：文本本身（归一化后）是否就是一个标准格式的车牌。"""
    if not text:
        return False
    return bool(_VALID_RE.match(_normalize(text)))


def main() -> int:
    parser = argparse.ArgumentParser(description="车牌格式校验（结果文件过滤）")
    parser.add_argument("--results", required=True, help="OCR 结果文件路径")
    parser.add_argument("--out", help="过滤结果输出文件（默认只打印）")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"[error] not found: {args.results}", file=sys.stderr)
        return 1

    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    plates = []
    for line in lines:
        plate = extract_plate(line)
        out_lines.append(f"{line}\t{plate if plate else '-'}")
        if plate:
            plates.append(plate)

    distinct = sorted(set(plates))
    summary = [
        f"total lines: {len(lines)}",
        f"lines with a valid plate: {len(plates)}",
        f"distinct plates: {len(distinct)}",
    ]
    for plate in distinct:
        summary.append(f"  {plate}: {plates.count(plate)} line(s)")

    text = "\n".join(out_lines + ["", "=== summary ==="] + summary) + "\n"
    print(text, end="", flush=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
