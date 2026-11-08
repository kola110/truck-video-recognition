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
"""
import argparse
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
        label = f"{names.get(cls, cls)} {conf:.2f}" if names else f"{cls} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


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
    args = ap.parse_args()

    device = pick_device(args.device)
    use_half = device != "cpu"
    print(f"[info] device={device} half={use_half} skip={args.skip}", flush=True)
    model = YOLO(args.weights)
    names = model.names

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
                result = model(frame, conf=args.conf, imgsz=args.imgsz,
                               device=device, half=use_half, verbose=False)[0]
                last_boxes = result.boxes
                annotated = result.plot()  # boxes + class + confidence, BGR
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
