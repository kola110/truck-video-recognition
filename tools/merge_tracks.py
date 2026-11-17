"""事后轨道合并：把同一辆车的碎片轨道合并成一组。

背景：CPU 慢速推理下 ByteTrack 容易断裂，113 个跟踪 ID 对应约 2–3 辆
真实车辆，每个碎片轨道都存了一张“最佳帧”截图，大多是垃圾截图。
本脚本在 best.json 上做事后合并，不需要重新跑推理。

合并规则：两个轨道的最佳帧框 IoU 超过阈值（默认 0.5）即视为同一辆车，
用并查集合并。停着的车辆各碎片的最佳框高度重叠，基本能合到一起；
行驶中的车辆在不同位置的最佳框可能合并不了——这是已知局限，
行驶段的碎片需要靠调优跟踪器（trackers/bytetrack_cpu.yaml）减少。

用法：
    python tools/merge_tracks.py --manifest /data/best_frames/best.json \
        --out /data/best_frames/merged_best.json

输出 merged_best.json：每组一条，含 group_id、成员 track_id 列表、
代表截图（组内面积最大的那张，即最清晰的一张）及其 frame/conf/box/area。
同时打印统计：轨道数 → 组数、各组大小、面积分布（供后续定最小面积
门槛参考）。
"""

import argparse
import json
from pathlib import Path


def iou(box_a, box_b) -> float:
    """两框交并比，box 为 [x1, y1, x2, y2]。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    if inter == 0:
        return 0.0
    area_a = max(box_a[2] - box_a[0], 0) * max(box_a[3] - box_a[1], 0)
    area_b = max(box_b[2] - box_b[0], 0) * max(box_b[3] - box_b[1], 0)
    return inter / (area_a + area_b - inter)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> int:
    parser = argparse.ArgumentParser(description="事后轨道合并（IoU）")
    parser.add_argument("--manifest", required=True, help="best.json 路径")
    parser.add_argument("--out", required=True, help="合并后清单输出路径")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="合并的 IoU 阈值（默认 0.5）")
    args = parser.parse_args()

    entries = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    n = len(entries)
    if n == 0:
        print("[error] manifest is empty", flush=True)
        return 1

    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if iou(entries[i]["box"], entries[j]["box"]) >= args.iou_thresh:
                uf.union(i, j)

    groups: dict[int, list] = {}
    for idx, entry in enumerate(entries):
        groups.setdefault(uf.find(idx), []).append(entry)

    merged = []
    for gid, (root, members) in enumerate(sorted(groups.items()), start=1):
        best = max(members, key=lambda e: e["area"])
        merged.append({
            "group_id": gid,
            "members": sorted(e["track_id"] for e in members),
            "member_count": len(members),
            "file": best["file"],
            "frame": best["frame"],
            "conf": best["conf"],
            "box": best["box"],
            "area": best["area"],
        })
    merged.sort(key=lambda g: g["area"], reverse=True)
    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    areas = sorted(e["area"] for e in entries)
    print(f"tracks: {n} -> groups: {len(merged)}", flush=True)
    print("group sizes (largest first): "
          f"{sorted((g['member_count'] for g in merged), reverse=True)}",
          flush=True)
    print(f"areas: min={areas[0]} median={areas[n // 2]} max={areas[-1]}",
          flush=True)
    for thresh in (100_000, 300_000, 600_000):
        count = sum(1 for a in areas if a >= thresh)
        print(f"tracks with area >= {thresh}: {count}", flush=True)
    print(f"merged manifest: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
