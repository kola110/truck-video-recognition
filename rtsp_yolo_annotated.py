"""
RTSP + YOLO: read an RTSP stream, draw boxes, push the annotated stream out.

This is the "push it out" follow-on to the quick local-only version:
  - local-only: result.plot() + cv2.imshow()
  - this file: same plot() result, but piped to ffmpeg -> RTMP -> ZLMediaKit,
    so a Vue frontend (or anyone else) can watch the annotated stream
    without running Python locally.

Usage (replace the two URLs with yours):
    python rtsp_yolo_annotated.py \
        --source "rtsp://user:pass@camera-ip:554/stream" \
        --weights yolo26n.pt \
        --push "rtmp://127.0.0.1:1935/live/yolo_annotated"

    # Local preview only, no push:
    python rtsp_yolo_annotated.py --source "rtsp://..." --no-push --show

    # With ByteTrack tracking + best-frame crops for the OCR stage:
    python rtsp_yolo_annotated.py --source "./truck.mp4" --no-push --show \
        --track --save-best ./best_frames

Watch the pushed stream (ZLMediaKit defaults; check your own ZLM config):
    FLV: http://<zlm-ip>:80/live/yolo_annotated.live.flv
    HLS: http://<zlm-ip>:80/live/yolo_annotated/hls.m3u8

Notes:
  - RTSP is forced over TCP to reduce tearing / disconnects.
  - A reader thread + queue(maxsize=1) drops stale frames instead of
    letting RTSP lag grow. If the camera drops, it reconnects.
  - --skip 2 means: run the model every 2nd frame, redraw the previous
    boxes on the frames in between. Use 1 for every frame.
  - --device auto: CUDA + half precision when available, else CPU.
  - --track: use model.track() with ByteTrack (persist=True) so each
    vehicle keeps one ID across frames; plot() draws the ID on the box.
    With --save-best DIR, the largest (best-view) crop of each vehicle
    is saved once as track_<id>.jpg, plus a best.json manifest. That is
    the "one OCR trigger per vehicle" input for stage 2 -- run OCR on
    these crops instead of on every frame.
  - With --track and --skip > 1, tracking persists across skipped
    frames, but IDs may switch more often; use --skip 1 when the IDs
    matter (e.g. when saving best frames for OCR).
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class RTSPReader(threading.Thread):
    """Keep only the newest frame; drop the rest so lag cannot accumulate."""

    def __init__(self, source: str):
        super().__init__(daemon=True)
        self.source = source
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stop_flag = threading.Event()

    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def run(self):
        while not self.stop_flag.is_set():
            cap = self._open()
            if cap is None:
                time.sleep(2)
                continue
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break  # camera dropped -> outer loop reconnects
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.q.put_nowait(frame)
                    except queue.Full:
                        pass
            cap.release()
            time.sleep(1)

    def read(self, timeout: float = 5.0):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None


def draw_last_boxes(frame, boxes, names):
    """Redraw previous detections on a new frame (used between inferences)."""
    if boxes is None:
        return frame
    out = frame
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cls = int(box.cls[0]) if box.cls is not None else -1
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        base = f"{names.get(cls, cls)} {conf:.2f}" if names else f"{cls} {conf:.2f}"
        tid = int(box.id[0]) if box.id is not None else None
        label = f"{base} id:{tid}" if tid is not None else base
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


class BestFrameSaver:
    """Keep the single best (largest-box) crop per tracked vehicle.

    OCR is expensive, so stage 2 should run on one good crop per vehicle,
    not on every frame. For each track id we keep only the frame where
    the box is largest; the crop is overwritten whenever a better frame
    appears. A best.json manifest records what was saved.
    """

    def __init__(self, outdir: str, pad: int = 10):
        self.outdir = outdir
        self.pad = pad
        self.best: dict = {}
        os.makedirs(outdir, exist_ok=True)

    def consider(self, frame, boxes, frame_idx: int) -> None:
        if boxes is None or len(boxes) == 0:
            return
        h, w = frame.shape[:2]
        for box in boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if area <= 0:
                continue
            prev = self.best.get(tid)
            if prev is not None and area <= prev["area"]:
                continue
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            px1 = max(x1 - self.pad, 0)
            py1 = max(y1 - self.pad, 0)
            px2 = min(x2 + self.pad, w)
            py2 = min(y2 + self.pad, h)
            path = os.path.join(self.outdir, f"track_{tid}.jpg")
            cv2.imwrite(path, frame[py1:py2, px1:px2])
            self.best[tid] = {
                "track_id": tid,
                "file": path,
                "frame": frame_idx,
                "conf": round(conf, 3),
                "box": [x1, y1, x2, y2],
                "area": area,
                "time": time.strftime("%H:%M:%S"),
            }
            print(f"[track] id={tid} new best frame={frame_idx} "
                  f"area={area} conf={conf:.2f} -> {path}", flush=True)

    def write_manifest(self) -> None:
        if not self.best:
            return
        path = os.path.join(self.outdir, "best.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.best.values(), key=lambda r: r["track_id"]),
                      f, ensure_ascii=False, indent=2)
        print(f"[track] manifest: {len(self.best)} vehicle(s) -> {path}",
              flush=True)


def start_ffmpeg(push_url: str, width: int, height: int, fps: float):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-f", "flv", push_url,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="RTSP URL of the camera")
    ap.add_argument("--weights", default="yolo26n.pt",
                    help="YOLO weights; swap in your truck-model weights here")
    ap.add_argument("--push", default="rtmp://127.0.0.1:1935/live/yolo_annotated",
                    help="ZLMediaKit RTMP push URL")
    ap.add_argument("--no-push", action="store_true", help="preview only, do not push")
    ap.add_argument("--show", action="store_true", help="also show a local window")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip", type=int, default=1, help="infer every Nth frame")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--track", action="store_true",
                    help="enable ByteTrack tracking (stable id per vehicle)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="tracker config for model.track()")
    ap.add_argument("--save-best", default=None, metavar="DIR",
                    help="with --track, save the best crop per vehicle here")
    args = ap.parse_args()

    device = pick_device(args.device)
    use_half = device != "cpu"
    print(f"[info] device={device} half={use_half} skip={args.skip} "
          f"track={args.track}", flush=True)
    model = YOLO(args.weights)
    names = model.names

    saver = None
    if args.save_best:
        if not args.track:
            print("[warn] --save-best needs --track; enabling tracking",
                  flush=True)
            args.track = True
        saver = BestFrameSaver(args.save_best)

    reader = RTSPReader(args.source)
    reader.start()

    ffmpeg_proc = None
    last_boxes = None
    frame_idx = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                print("[warn] no frame for 5s, still waiting / reconnecting...",
                      flush=True)
                continue
            frame_idx += 1

            if frame_idx % max(args.skip, 1) == 1:
                if args.track:
                    result = model.track(frame, persist=True,
                                         tracker=args.tracker,
                                         conf=args.conf, imgsz=args.imgsz,
                                         device=device, half=use_half,
                                         verbose=False)[0]
                else:
                    result = model(frame, conf=args.conf, imgsz=args.imgsz,
                                   device=device, half=use_half,
                                   verbose=False)[0]
                last_boxes = result.boxes
                annotated = result.plot()  # boxes + class + conf (+ id), BGR
                if saver is not None:
                    saver.consider(frame, result.boxes, frame_idx)
            else:
                annotated = draw_last_boxes(frame.copy(), last_boxes, names)

            if ffmpeg_proc is None and not args.no_push:
                h, w = annotated.shape[:2]
                ffmpeg_proc = start_ffmpeg(args.push, w, h, args.fps)
                print(f"[info] pushing annotated stream to {args.push}", flush=True)
            if ffmpeg_proc is not None:
                try:
                    ffmpeg_proc.stdin.write(annotated.tobytes())
                except BrokenPipeError:
                    print("[warn] ffmpeg pipe broke; will restart on next frame",
                          flush=True)
                    ffmpeg_proc = None

            if args.show:
                cv2.imshow("RTSP + YOLO (annotated)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[info] stopping", flush=True)
    finally:
        if saver is not None:
            saver.write_manifest()
        reader.stop_flag.set()
        if ffmpeg_proc is not None and ffmpeg_proc.stdin:
            try:
                ffmpeg_proc.stdin.close()
            except BrokenPipeError:
                pass
            ffmpeg_proc.wait(timeout=5)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
