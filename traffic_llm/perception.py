"""영상 인지: 프레임 샘플링 → 객체 검출 → 다중객체 추적.

두 가지 구현을 제공한다.
  - YoloPerception : ultralytics + ByteTrack (실제 영상 처리)
  - JsonPerception : 사전 계산된 검출 결과(JSON/MOT) 로딩 — 오프라인 재현·평가용

새 검출기를 쓰려면 PerceptionBackend 를 상속해 run() 만 구현하면 된다.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

from .config import PerceptionConfig
from .schemas import Detection


class PerceptionBackend:
    """검출·추적 백엔드 인터페이스."""

    def run(self, video_path: str, t0: float) -> List[Detection]:
        """영상 전체를 처리하여 시각순 Detection 리스트 반환.

        t0: 영상 첫 프레임의 절대 시각 [s]. 차량 간 시간 동기화에 사용.
        """
        raise NotImplementedError


class YoloPerception(PerceptionBackend):
    """ultralytics YOLO + ByteTrack. `pip install ultralytics opencv-python` 필요."""

    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        self._model = None

    def _lazy_model(self):
        if self._model is None:
            from ultralytics import YOLO  # 지연 임포트

            self._model = YOLO(self.cfg.model_name)
        return self._model

    def run(self, video_path: str, t0: float) -> List[Detection]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"영상을 열 수 없습니다: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(int(round(fps / self.cfg.sample_hz)), 1)
        model = self._lazy_model()

        dets: List[Detection] = []
        frame_idx = 0
        keep = set(self.cfg.vehicle_classes) | {"person"}
        try:
            while True:
                ok = cap.grab()  # 디코딩 없이 스킵 → 샘플링 비용 절감
                if not ok:
                    break
                if frame_idx % stride == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    res = model.track(
                        frame,
                        persist=True,
                        tracker=self.cfg.tracker,
                        conf=self.cfg.min_conf,
                        verbose=False,
                    )[0]
                    t = t0 + frame_idx / fps
                    if res.boxes is not None and res.boxes.id is not None:
                        names = res.names
                        for box, tid, cid, conf in zip(
                            res.boxes.xyxy.tolist(),
                            res.boxes.id.int().tolist(),
                            res.boxes.cls.int().tolist(),
                            res.boxes.conf.tolist(),
                        ):
                            cls = names[cid]
                            if cls not in keep:
                                continue
                            dets.append(
                                Detection(
                                    t=t,
                                    frame_idx=frame_idx,
                                    track_id=int(tid),
                                    cls=cls,
                                    conf=float(conf),
                                    bbox=tuple(box),
                                )
                            )
                frame_idx += 1
        finally:
            cap.release()
        return dets


class JsonPerception(PerceptionBackend):
    """사전 계산된 검출 결과 로딩.

    포맷: {"fps": 30, "detections": [{"frame": 0, "track_id": 3, "cls": "car",
           "conf": 0.9, "bbox": [x1, y1, x2, y2]}, ...]}
    또는 위 dict 의 "detections" 리스트만 담긴 JSON 배열.
    """

    def __init__(self, cfg: PerceptionConfig, det_paths: Dict[str, str]):
        self.cfg = cfg
        self.det_paths = det_paths  # video_path → json_path

    def run(self, video_path: str, t0: float) -> List[Detection]:
        path = self.det_paths.get(video_path)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"검출 결과 파일이 없습니다: {video_path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items: Iterable[dict]
        if isinstance(raw, dict):
            fps = float(raw.get("fps", 30.0))
            items = raw.get("detections", [])
        else:
            fps, items = 30.0, raw

        out: List[Detection] = []
        for d in items:
            if float(d.get("conf", 1.0)) < self.cfg.min_conf:
                continue
            frame = int(d.get("frame", 0))
            out.append(
                Detection(
                    t=float(d["t"]) if "t" in d else t0 + frame / fps,
                    frame_idx=frame,
                    track_id=int(d["track_id"]),
                    cls=str(d.get("cls", "car")),
                    conf=float(d.get("conf", 1.0)),
                    bbox=tuple(float(v) for v in d["bbox"]),
                )
            )
        out.sort(key=lambda x: (x.t, x.track_id))
        return out


def build_backend(
    cfg: PerceptionConfig, det_paths: Optional[Dict[str, str]] = None
) -> PerceptionBackend:
    """검출 결과가 주어지면 JSON 백엔드, 아니면 YOLO 백엔드."""
    if det_paths:
        return JsonPerception(cfg, det_paths)
    return YoloPerception(cfg)


def index_by_time(dets: List[Detection]) -> Dict[float, List[Detection]]:
    """시각별 그룹화 (동일 프레임 검출을 한 번에 처리하기 위함)."""
    out: Dict[float, List[Detection]] = {}
    for d in dets:
        out.setdefault(round(d.t, 3), []).append(d)
    return out
