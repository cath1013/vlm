"""traffic_llm — 다중 차량 전면 카메라 영상 + 도로 지도 → LLM 입력 변환 파이프라인."""

from .config import (
    CameraConfig,
    ClassSize,
    DeepAccidentConfig,
    FusionConfig,
    LaneConfig,
    PerceptionConfig,
    PipelineConfig,
    RoadGenConfig,
    SerializeConfig,
)
from .geometry import CameraModel, LocalENU
from .perception import JsonPerception, PerceptionBackend, YoloPerception
from .pipeline import TrafficSceneConverter, load_telemetry
from .roadgen import synthesize_road_network
from .roadmap import RoadNetwork
from .schemas import (
    ActorState,
    Detection,
    EgoSample,
    InfraState,
    Interaction,
    Observation,
    PredictedPath,
    RoadPlacement,
    ScenarioContext,
    SceneSnapshot,
)
from .serialize import (
    build_messages,
    to_bev_ascii,
    to_evaluation_record,
    to_json,
    to_text,
)

__version__ = "0.1.0"

__all__ = [
    "ActorState",
    "CameraConfig",
    "CameraModel",
    "ClassSize",
    "DeepAccidentConfig",
    "Detection",
    "EgoSample",
    "FusionConfig",
    "InfraState",
    "Interaction",
    "JsonPerception",
    "LaneConfig",
    "LocalENU",
    "Observation",
    "PerceptionBackend",
    "PerceptionConfig",
    "PipelineConfig",
    "PredictedPath",
    "RoadGenConfig",
    "RoadNetwork",
    "RoadPlacement",
    "ScenarioContext",
    "SceneSnapshot",
    "SerializeConfig",
    "TrafficSceneConverter",
    "YoloPerception",
    "build_messages",
    "load_telemetry",
    "synthesize_road_network",
    "to_bev_ascii",
    "to_evaluation_record",
    "to_json",
    "to_text",
]


def __getattr__(name):
    """DeepAccident 관련 심볼은 지연 로딩 (numpy 외 추가 의존성 없음)."""
    da = {
        "DeepAccidentRunner": ("da_runner", "DeepAccidentRunner"),
        "DeepAccidentPerception": ("deepaccident", "DeepAccidentPerception"),
        "scan_scenarios": ("deepaccident", "scan_scenarios"),
        "find_scenario": ("deepaccident", "find_scenario"),
        "synthesize_telemetry": ("deepaccident", "synthesize_telemetry"),
        "evaluate_scenario": ("da_eval", "evaluate_scenario"),
        "load_opendrive": ("carla_map", "load_opendrive"),
    }
    if name in da:
        import importlib

        mod, attr = da[name]
        return getattr(importlib.import_module(f".{mod}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
