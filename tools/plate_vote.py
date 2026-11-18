"""多截图投票：同一辆车的多个截图各读一次，投票定出车牌。

背景：车牌格式校验（tools/plate_format.py）只能滤掉“格式不对”的行，
拦不住“格式对但字错”的读法——2026-11-17 三轮真实数据重验证后还剩
字符级误读：末位 8 读成 9 有 3 条、省份字读错 1 条、多余字符拼成新
能源格式 1 条。这些错误是单张截图的偶发误读，同一辆车在多张截图
里多数时候读对了，所以按车汇总投票可以纠正它们。

输入（全部是已有的文件，不需要重新跑推理、不用重新调 OCR）：
- 合并清单 merged_best.json（tools/merge_tracks.py 的输出）：每组一条，
  含 group_id、成员 track_id 列表、代表截图；
- 三轮 OCR 结果文件（--results 可重复给，每次都要带方法名）：
  整图 OCR、裁剪放大 OCR、专用检测器 OCR。三轮文件的分隔符不统一：
  整图轮用空格分隔（文本里还含空格），裁剪轮和检测器轮用 Tab 分隔，
  检测器轮还多一列检测数。解析时按第一个空白字符切分，文件名是
  第一段，其余全当识别文本（文件名里不含空白，所以切分是安全的）。

投票规则（两层，先看整体再看逐字）：
1. 整串投票：同一组里所有成员截图、三轮方法提取到的车牌按整串计数，
   票数最多者为该组车牌；票数相同时，先看支持的方法数（三种方法都
   读到的比只一种方法读到的更可信）。票数和方法数都相同就是平票，
   脚本会标出 TIE 交给人工复核，不强行定一个。
2. 逐字投票：把同一组里长度相同的候选车牌按位置逐字投票，拼出共识
   车牌，并给出每一位的赞成比例。逐字投票能纠正“整串投票里票数
   分散”的情况：比如同一辆车 6 张截图读对、3 张把末位 8 读成 9，
   整串投票已经能赢，但逐字投票还能把每一位的可信度算出来，方便
   人工复核；如果某一位赞成比例低于一半，说明这一位不可信，要标
   出来而不是悄悄定下来。

输出：
- 打印每组的投票表：候选车牌、票数、支持方法、逐字共识及每位赞成比；
- --out 写一份 JSON：每组一条，含 group_id、成员数、候选票数、胜出
  车牌、逐字共识、每位赞成比、是否平票、是否无票。

用法（free-machine，数据都在机器上，文件名以机器实际文件为准）：
    python tools/plate_vote.py \\
        --merged /data/best_frames/merged_best.json \\
        --results whole:/data/ocr_run3.txt \\
        --results crop:/data/ocr_run4.txt \\
        --results detect:/data/ocr_run5.txt \\
        --out /data/plate_votes.json

注意：三轮结果文件的实际路径要先在机器上核对（整图轮的文件名不在
仓库里写死），路径不对脚本会报错退出，不会静默出空结果。
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# plate_format.py 与本脚本在同一个 tools 目录下，直接复用它的车牌正则，
# 保证“什么是车牌”的定义与格式校验完全一致，不在这里另写一份。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_format import _PLATE_RE  # noqa: E402


def _format_match(match: re.Match) -> str:
    """把一条正则匹配结果拼成标准形式：省份 + 城市 + 圆点 + 序号。"""
    province, city, seq = match.groups()
    return f"{province}{city}·{seq}"


def extract_all_plates(text: str) -> list[str]:
    """从一段 OCR 文本里提取全部符合车牌格式的车牌（按出现顺序）。

    两遍匹配，与 plate_format.extract_plate 的策略一致：先在原文上找，
    找不到再试去空格后的文本。去空格会把相邻碎片拼成假车牌，所以原文
    优先；去空格只用于恢复“浙M·MD5 98”这类字符间带空格的真车牌。
    同一段文本里出现多次的同一车牌只保留一次，避免同一行重复计票。
    """
    if not text:
        return []
    found: list[str] = []
    for source in (text.upper(), re.sub(r"\s+", "", text).upper()):
        for match in _PLATE_RE.finditer(source):
            plate = _format_match(match)
            if plate not in found:
                found.append(plate)
        if found:
            break
    return found


def parse_results_file(path: Path) -> dict[str, list[str]]:
    """解析一份 OCR 结果文件，返回 {截图文件名: [提取到的车牌, ...]}。

    行格式：第一段是文件名，其余都是识别文本。三轮文件的分隔符不统一：
    整图轮（ocr_run3.txt）用空格分隔，裁剪轮和检测器轮用 Tab 分隔，
    检测器轮还多一列检测数。这里按“第一个空白字符”切分——文件名里
    不含空白，所以第一段一定是文件名，剩下的全当识别文本；检测器轮
    的数字列混在文本里，但它是纯数字，正则不会把它当成车牌。
    空行跳过；文件不存在时报错退出，不静默返回空结果。
    """
    if not path.exists():
        print(f"[error] not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    votes: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        # 按第一个空白字符切分：兼容空格分隔（整图轮）和 Tab 分隔
        # （裁剪轮、检测器轮）；文本里可能还有空格或 Tab，全部保留。
        fields = line.split(None, 1)
        filename = fields[0]
        text = fields[1] if len(fields) > 1 else ""
        # 检测器轮的多块文本用 " | " 连接，先拆块再逐块提取，避免跨块
        # 拼出假车牌；整图/裁剪轮没有 " | "，拆分后只有一块，不受影响。
        plates: list[str] = []
        for chunk in text.split(" | "):
            for plate in extract_all_plates(chunk):
                if plate not in plates:
                    plates.append(plate)
        if plates:
            votes.setdefault(filename, []).extend(plates)
    return votes


def vote_group(plates: list[tuple[str, str]]) -> dict:
    """对一组（车牌，方法名）投票，返回整串胜者与逐字共识。

    plates 里的每一项是一次“一张截图、一种方法读到的一个车牌”，
    同一张截图在三轮里各读到一次就算三票——三种独立方法都读到同一
    串，比单一方法重复读到更可信，所以平票时先比支持的方法数。
    """
    if not plates:
        return {"winner": None, "votes": {}, "consensus": None,
                "position_agreement": [], "tie": False}
    counts: Counter = Counter(p for p, _ in plates)
    methods: dict[str, set] = defaultdict(set)
    for plate, method in plates:
        methods[plate].add(method)
    # 排序：票数优先，其次支持的方法数（三种方法都读到的更可信）。
    ranked = sorted(counts.items(),
                    key=lambda kv: (kv[1], len(methods[kv[0]])), reverse=True)
    top_votes = ranked[0][1]
    tie = len(ranked) > 1 and ranked[1][1] == top_votes
    winner = ranked[0][0]

    # 逐字投票：只拿与胜者等长的候选车牌参与，避免普通车牌（8 位：
    # 省份 + 城市 + 圆点 + 5 位序号）与新能源车牌（9 位）混在一起逐字比。
    same_length = [p for p, _ in plates if len(p) == len(winner)]
    consensus_chars: list[str] = []
    agreement: list[float] = []
    for pos in range(len(winner)):
        char_counts: Counter = Counter(p[pos] for p in same_length)
        best_char, best_n = char_counts.most_common(1)[0]
        consensus_chars.append(best_char)
        agreement.append(round(best_n / len(same_length), 2))
    consensus = "".join(consensus_chars)
    return {
        "winner": winner,
        "votes": {plate: counts[plate] for plate, _ in ranked},
        "vote_methods": {plate: sorted(methods[plate])
                         for plate, _ in ranked},
        "consensus": consensus,
        "position_agreement": agreement,
        "tie": tie,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="多截图投票定车牌")
    parser.add_argument("--merged", required=True,
                        help="合并清单 merged_best.json 路径")
    parser.add_argument("--results", action="append", required=True,
                        help="OCR 结果文件，格式 方法名:路径，可重复给"
                             "（如 whole:/data/ocr_run3.txt）")
    parser.add_argument("--out", help="投票结果 JSON 输出文件")
    args = parser.parse_args()

    merged_path = Path(args.merged)
    if not merged_path.exists():
        print(f"[error] not found: {args.merged}", file=sys.stderr)
        return 1
    groups = json.loads(merged_path.read_text(encoding="utf-8"))
    if not groups:
        print("[error] merged manifest is empty", file=sys.stderr)
        return 1

    # 先把三轮结果文件都读进来：{方法名: {截图文件名: [车牌, ...]}}。
    # 方法名来自 --results 的冒号前半段，文件名以冒号后的路径为准。
    all_votes: dict[str, dict[str, list[str]]] = {}
    for spec in args.results:
        if ":" not in spec:
            print(f"[error] bad --results spec (want method:path): {spec}",
                  file=sys.stderr)
            return 1
        method, _, path = spec.partition(":")
        all_votes[method] = parse_results_file(Path(path))

    results = []
    for group in groups:
        # 成员截图文件名按 track_<id>.jpg 推导，与 rtsp_yolo_annotated.py
        # 的 --save-best 命名一致；代表截图本身也在成员里，不用单独加。
        member_files = [f"track_{tid}.jpg" for tid in group["members"]]
        plates: list[tuple[str, str]] = []
        for method, votes in all_votes.items():
            for filename in member_files:
                for plate in votes.get(filename, []):
                    plates.append((plate, method))
        outcome = vote_group(plates)
        record = {
            "group_id": group["group_id"],
            "member_count": group["member_count"],
            "representative": group["file"],
            "total_votes": len(plates),
            **outcome,
        }
        results.append(record)
        votes_text = ", ".join(
            f"{plate}×{n}" for plate, n in outcome["votes"].items())
        print(f"group {group['group_id']} "
              f"({group['member_count']} members, {len(plates)} votes): "
              f"{votes_text or 'no plate'}", flush=True)
        if outcome["winner"]:
            print(f"  winner: {outcome['winner']}"
                  + (" (TIE — needs review)" if outcome["tie"] else ""),
                  flush=True)
            print(f"  consensus: {outcome['consensus']} "
                  f"agreement={outcome['position_agreement']}", flush=True)
            if outcome["consensus"] != outcome["winner"]:
                print("  [note] consensus differs from winner — "
                      "check the low-agreement position(s)", flush=True)

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nvotes written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
