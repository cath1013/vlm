"""다중 차량 관측 융합.

여러 관측자의 여러 카메라가 같은 물리 차량을 동시에 볼 수 있으므로,
월드 좌표 근접성 + 클래스 일치도로 관측을 병합하여 중복을 제거한다.
관측 객체가 다른 관측차량(ego) 본체인 경우도 판별하여 ego 트랙에 흡수시킨다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .config import FusionConfig
from .schemas import ActorState, EgoSample, InfraState, Observation

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# 동역학이 비슷한 클래스끼리 묶은 그룹. 서로 다른 그룹의 관측은 같은 물체로
# 보지 않는다 — 보행자 관측이 근접만으로 오토바이 트랙을 물려받으면 그 트랙의
# 속도 상한이 오토바이 기준이 되어 100km/h 로 달리는 보행자가 만들어진다.
CLASS_GROUP: Dict[str, str] = {
    "car": "vehicle",
    "van": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",  # 차량 흐름 속도로 달린다
    "bicycle": "vru",
    "person": "vru",
}


def class_group(cls: str) -> str:
    """클래스의 동역학 그룹. 모르는 클래스는 그룹 제약을 걸지 않는다."""
    return CLASS_GROUP.get(cls, "")


def class_compatible(a: str, b: str) -> bool:
    """두 클래스가 같은 물체일 수 있는가."""
    ga, gb = class_group(a), class_group(b)
    return not (ga and gb and ga != gb)


def gate_m(
    cfg: FusionConfig, *ranges: float, quality: float = 1.0
) -> float:
    """관측거리·신뢰도에 따른 매칭 허용 반경.

    거리에 비례해 넓히고, 거리 추정 신뢰도가 낮으면 더 넓힌다. 극근접에서
    bbox 가 절단되면 접지점을 쓸 수 없어 오차가 수 m 가 되는데, 거리 항만으로는
    (거리가 작아) 게이트가 좁아져 같은 차량 관측이 병합에 실패한다.
    """
    r = min(ranges) if ranges else 0.0
    q = max(0.0, min(1.0, quality))
    g = (
        cfg.max_match_dist_m
        + cfg.range_gate_coeff * r
        + (1.0 - q) * cfg.quality_gate_m
    )
    return min(g, cfg.max_gate_m)


class GlobalTrackRegistry:
    """전역 actor_id 부여 및 프레임 간 연속성 유지.

    (관측차량 id, 로컬 트랙 id) → 전역 actor_id 매핑을 유지하고,
    서로 다른 관측차량이 같은 물체를 보면 두 매핑이 같은 전역 id 를 가리키게 한다.
    """

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self._map: Dict[Tuple[str, int], str] = {}
        self._last_xy: Dict[str, Tuple[float, float]] = {}
        self._last_t: Dict[str, float] = {}
        # actor id → 최근 클래스. 신원 재사용 시 동역학 그룹 일치를 확인한다.
        self._cls: Dict[str, str] = {}
        # 로컬 트랙 id → actor id. global_track_ids 일 때만 쓴다 (관측자와
        # 무관하게 같은 id = 같은 물체이므로 관측자가 바뀌어도 신원이 이어진다).
        self._by_track: Dict[int, str] = {}
        # 같은 트랙 id 가 한 물체일 수 없을 만큼 흩어져 나타난 횟수.
        # 0 이 아니면 "트랙 id 가 전역 유일하다"는 가정이 깨진 것이다
        # (카메라별 독립 추적기를 쓰면서 global_track_ids=True 로 둔 경우).
        self.id_collisions: int = 0
        # 관측자 본체용으로 예약된 actor id. 주변차량 클러스터가 이 id 를
        # 가져가면 ego 액터를 덮어써서 관측차량이 주변차량으로 뒤바뀐다.
        self._reserved: set = set()
        self._counter = 0

    # 클래스별 actor id 접두어 — BEV/텍스트에서 종류를 바로 읽을 수 있게
    ID_PREFIX = {
        "car": "V",
        "van": "N",
        "truck": "T",
        "bus": "B",
        "motorcycle": "M",
        "bicycle": "C",
        "person": "P",
    }

    def _new_id(self, cls: str) -> str:
        self._counter += 1
        return f"{self.ID_PREFIX.get(cls, 'O')}{self._counter:03d}"

    def _max_speed(self, cls: str) -> float:
        return self.cfg.max_speed_by_class.get(cls, self.cfg.max_speed_mps)

    def _class_ok(self, actor_id: str, cls: str) -> bool:
        return class_compatible(self._cls.get(actor_id, ""), cls)

    def resolve(
        self,
        t: float,
        key: Tuple[str, int],
        world_xy: Tuple[float, float],
        cls: str,
        obs_range_m: float = 0.0,
    ) -> str:
        """관측 하나에 전역 actor_id 부여 (단일 키 편의 함수)."""
        return self.resolve_cluster(t, [key], world_xy, cls, obs_range_m)

    def resolve_cluster(
        self,
        t: float,
        keys: List[Tuple[str, int]],
        world_xy: Tuple[float, float],
        cls: str,
        obs_range_m: float = 0.0,
        exclude: Optional[Set[str]] = None,
    ) -> str:
        """클러스터(같은 물체로 판정된 관측 묶음)에 전역 actor_id 부여.

        클러스터를 이루는 **모든** (관측자, 로컬트랙) 키로 기존 액터를 찾는다.
        대표 관측 하나로만 조회하면, 프레임마다 가장 가까운 관측자가 바뀔 때
        키가 달라져 같은 차량에 새 id 가 계속 발급된다 (신원 불안정).

        exclude: 이 시각에 이미 다른 클러스터가 가져간 actor id. 같은 스냅샷의
          두 클러스터가 같은 id 로 해석되면 뒤엣것이 앞엣것을 덮어써 액터가
          순간이동한다 (같은 시각이라 dt=0 이므로 운동 관문이 걸러내지 못한다).
        """
        exclude = exclude or set()
        if self.cfg.global_track_ids:
            # 관측자를 무시하고 트랙 id 로만 조회한다. (관측자, 트랙) 쌍으로
            # 조회하면 다른 관측자가 같은 차량을 처음 볼 때 키가 없어 새 액터가
            # 발급되고, 한 물체가 관측자 수만큼 쪼개진다.
            mapped = {self._by_track.get(lid) for _, lid in keys}
        else:
            mapped = {self._map.get(k) for k in keys if k in self._map}
        if self.cfg.global_track_ids:
            # 트랙 id 가 전역 유일하면 키 일치는 곧 신원 일치다. 관측이 가려짐
            # 등으로 track_timeout_s 보다 길게 끊겼다가 돌아와도 같은 물체이므로
            # 시간·운동 관문을 걸지 않는다 — 걸면 같은 차량이 여러 액터로 쪼개져
            # 궤적 이력이 끊긴다.
            fresh = [aid for aid in mapped if aid and aid not in exclude]
        else:
            fresh = [
                aid
                for aid in mapped
                if aid
                and aid not in exclude
                and t - self._last_t.get(aid, t) <= self.cfg.track_timeout_s
                and self._class_ok(aid, cls)
                and self._motion_plausible(aid, world_xy, t, cls)
            ]
        if fresh:
            # 여러 액터가 걸리면 마지막 위치가 가장 가까운 쪽을 쓴다.
            # 다른 액터의 과거 매핑까지 이 id 로 다시 쓰지는 않는다 — 한 번의
            # 오연관이 두 차량의 신원을 영구히 결합시켜 순간이동을 만든다.
            aid = min(
                fresh,
                key=lambda a: math.dist(self._last_xy.get(a, world_xy), world_xy),
            )
            self._register(keys, aid, world_xy, t, cls)
            return aid

        # 다른 관측자가 이미 등록한 동일 물체인지 근접 탐색
        observers = {k[0] for k in keys}
        best, best_d = None, gate_m(self.cfg, obs_range_m)
        for aid, xy in self._last_xy.items():
            if aid in self._reserved or aid in exclude:
                continue  # 관측자 본체 id, 이미 배정된 id 는 재사용 금지
            if t - self._last_t.get(aid, -1e9) > self.cfg.track_timeout_s:
                continue
            # 같은 관측자가 이미 **다른** 로컬트랙을 이 액터에 매핑했다면
            # 별개 물체다 (한 관측자의 서로 다른 트랙은 다른 객체)
            if any(
                obs in observers and (obs, lid) not in keys
                for (obs, lid), a in self._map.items()
                if a == aid
            ):
                continue
            if not self._class_ok(aid, cls):
                continue  # 보행자↔차량 등 동역학이 다른 물체는 병합 금지
            if not self._motion_plausible(aid, world_xy, t, cls):
                continue
            d = math.dist(xy, world_xy)
            if d < best_d:
                best, best_d = aid, d
        aid = best or self._new_id(cls)
        self._register(keys, aid, world_xy, t, cls)
        return aid

    def _motion_plausible(
        self,
        actor_id: str,
        world_xy: Tuple[float, float],
        t: float,
        cls: str = "",
    ) -> bool:
        """기존 액터 id 재사용이 물리적으로 가능한지.

        직전 위치에서 지금 위치까지 필요한 속도가 클래스 상한을 넘으면 다른
        물체다. 이 관문이 없으면 오연관 한 번으로 액터가 수십 미터를
        순간이동하고, 보행자가 100km/h 로 달리는 결과가 나온다.
        """
        last_t = self._last_t.get(actor_id)
        last_xy = self._last_xy.get(actor_id)
        if last_t is None or last_xy is None:
            return True
        dt = t - last_t
        if dt <= 1e-6:
            return True
        return math.dist(last_xy, world_xy) / dt <= self._max_speed(cls)

    def _register(
        self,
        keys: List[Tuple[str, int]],
        aid: str,
        world_xy: Tuple[float, float],
        t: float,
        cls: str = "",
    ) -> None:
        for k in keys:
            self._map[k] = aid
            self._by_track[k[1]] = aid
        self._last_xy[aid] = world_xy
        self._last_t[aid] = t
        if cls:
            self._cls[aid] = cls

    def local_ids_of(self, actor_id: str) -> List[int]:
        """전역 actor_id 에 매핑된 로컬 트랙 id 목록.

        DeepAccident 처럼 로컬 트랙 id 가 전역적으로 유일한(CARLA actor id)
        데이터에서는 이것으로 정답과 신원 기반 매칭을 할 수 있다.
        """
        return sorted({lid for (_, lid), aid in self._map.items() if aid == actor_id})

    def actor_of_local_id(self, local_id: int) -> Optional[str]:
        for (_, lid), aid in self._map.items():
            if lid == local_id:
                return aid
        return None

    def _owner_of(self, actor_id: str) -> Optional[str]:
        for (obs_id, _), aid in self._map.items():
            if aid == actor_id:
                return obs_id
        return None

    def register_ego(
        self,
        observer_id: str,
        xy: Tuple[float, float],
        t: float,
        cls: str = "car",
    ) -> str:
        """Reserve an observer actor id with its supplied semantic class."""
        aid = f"EGO_{observer_id}"
        self._reserved.add(aid)
        self._cls[aid] = cls
        self._last_xy[aid] = xy
        self._last_t[aid] = t
        return aid


def _cost(obs_a: Observation, obs_b: Observation, cfg: FusionConfig) -> float:
    """정규화 거리 비용. 1.0 미만이면 동일 물체 후보."""
    d = math.dist(obs_a.world_xy, obs_b.world_xy)
    if obs_a.cls != obs_b.cls:
        d += cfg.class_mismatch_penalty_m
    return d / gate_m(
        cfg,
        obs_a.distance_m,
        obs_b.distance_m,
        quality=min(obs_a.range_quality, obs_b.range_quality),
    )


def _freshest(
    obs: List[Observation], t: float
) -> Optional[Observation]:
    """대표 관측 선택: 스냅샷 시각에 **가장 가까운 검출**, 동시각이면 근거리.

    거리만으로 고르면 안 된다 — 가까운 관측자가 이 시각에 표본이 없어 이웃
    프레임 검출을 쓴 경우, 0.1초 낡은 값이 최신 값을 이긴다 (8m/s 차량에 0.8m).
    """
    if not obs:
        return None
    return min(
        obs, key=lambda o: (abs((o.measured_t if o.measured_t is not None else o.t) - t), o.distance_m)
    )


def _split_if_id_collision(
    group: List[Observation],
    registry: "GlobalTrackRegistry",
    cfg: FusionConfig,
) -> List[List[Observation]]:
    """같은 트랙 id 로 묶인 관측이 **한 물체일 수 없을 만큼** 흩어져 있으면 쪼갠다.

    `global_track_ids=True` 는 "같은 트랙 id = 같은 물체"라는 가정이다. V2X 로
    객체 id 를 공유하는 시스템에서는 맞지만, 카메라별 독립 추적기(YOLO+ByteTrack
    등)는 각자 1번부터 번호를 매기므로 서로 다른 차량이 같은 id 를 갖는다.
    그 상태로 id 만 믿으면 **수백 m 떨어진 두 차량이 한 액터로 병합되고 하나가
    조용히 사라진다** (실측: 360m 떨어진 두 차량 → 액터 1개).

    관측자 간 위치 불일치는 단안 측거 오차 때문에 원거리에서 20~25m 까지 벌어질
    수 있다. 그보다 훨씬 큰 `id_collision_dist_m`(기본 40m)을 넘으면 오차로
    설명할 수 없으므로 가정 위반으로 보고, 위치 기준으로 쪼갠 뒤 카운터를 올린다.
    조용히 병합하는 것보다 드러내는 편이 낫다.
    """
    if len(group) < 2 or cfg.id_collision_dist_m <= 0:
        return [group]
    spread = max(
        math.dist(a.world_xy, b.world_xy)
        for i, a in enumerate(group)
        for b in group[i + 1 :]
    )
    if spread <= cfg.id_collision_dist_m:
        return [group]

    registry.id_collisions += 1
    # 단순 응집: 게이트 안에 드는 것끼리만 같은 클러스터로 둔다
    out: List[List[Observation]] = []
    for obs in sorted(group, key=lambda o: o.distance_m):
        for cl in out:
            if any(
                math.dist(obs.world_xy, m.world_xy)
                <= gate_m(cfg, min(obs.distance_m, m.distance_m))
                for m in cl
            ):
                cl.append(obs)
                break
        else:
            out.append([obs])
    return out


def fuse_observations(
    t: float,
    observations: List[Observation],
    ego_states: Dict[str, EgoSample],
    ego_world: Dict[str, Tuple[float, float]],
    registry: GlobalTrackRegistry,
    cfg: FusionConfig,
    observer_roles: Optional[Dict[str, str]] = None,
    observer_self_ids: Optional[Dict[str, int]] = None,
    observer_self_classes: Optional[Dict[str, str]] = None,
) -> List[ActorState]:
    """한 시각의 모든 관측 → 중복 제거된 ActorState 리스트.

    1) 관측 **차량** 본체를 ego actor 로 등록 (노변 인프라는 교통 참여자가
       아니므로 actor 로 만들지 않는다)
    2) 각 관측이 다른 관측차량을 가리키면 해당 ego 에 흡수
    3) 남은 관측을 관측자 쌍 간에 매칭하여 클러스터로 병합
    """
    roles = observer_roles or {}
    self_classes = observer_self_classes or {}
    actors: Dict[str, ActorState] = {}
    vehicle_world = {
        oid: xy
        for oid, xy in ego_world.items()
        if roles.get(oid, "vehicle") == "vehicle"
    }

    # 1) ego actors — 차량 관측자만
    for oid, samp in ego_states.items():
        if roles.get(oid, "vehicle") != "vehicle":
            continue
        xy = ego_world[oid]
        ego_cls = self_classes.get(oid) or "car"
        aid = registry.register_ego(oid, xy, t, ego_cls)
        actors[aid] = ActorState(
            actor_id=aid,
            kind="ego",
            cls=ego_cls,
            world_xy=xy,
            heading_deg=samp.heading_deg,
            speed_mps=samp.speed_mps,
            accel_mps2=None,
            observed_by=[oid],
            confidence=1.0,
            heading_source="telemetry",
        )

    # 2) 관측차량 본체를 가리키는 관측 흡수.
    #
    # 단순히 "게이트 안에 있으면 흡수"하면 안 된다. 게이트는 관측거리에 따라
    # 최대 25m 까지 넓어지므로, 관측차량 근처를 지나는 **다른** 차량이 삼켜져
    # 장면에서 사라지고 신원 추적도 오염된다. 두 조건을 추가한다.
    #   - 최근접 우선: 거리가 가까운 (관측, 관측차량) 짝부터 배정
    #   - 관측자별 1대 1: 한 관측자의 관측 여러 개가 같은 관측차량으로 흡수될
    #     수 없다 (관측자는 특정 차량을 한 프레임에 한 번만 본다)
    absorbed_idx: Set[int] = set()
    taken: Set[Tuple[str, str]] = set()  # (관측자, 관측차량) 짝은 1회만

    # 2-a) 트랙 id 가 전역 유일하면 신원을 **정확히** 알 수 있다.
    # V2X 로 객체 id 를 공유하는 시스템에서는 어떤 트랙 id 가 어느 관측 차량의
    # 본체인지 알려줄 수 있다. 이때는 거리 게이트에 의존하지 않는다 — 단안
    # 측거 오차가 게이트를 넘으면 관측차량의 복제 유령이 별도 차량으로 등장한다.
    if cfg.global_track_ids and observer_self_ids:
        id_to_observer = {
            tid: oid for oid, tid in observer_self_ids.items() if tid is not None
        }
        for i, obs in enumerate(observations):
            oid = id_to_observer.get(obs.local_track_id)
            if oid is None or oid == obs.observer_id:
                continue
            if oid not in vehicle_world:
                continue
            absorbed_idx.add(i)
            taken.add((obs.observer_id, oid))
            aid = f"EGO_{oid}"
            if aid in actors:
                if obs.observer_id not in actors[aid].observed_by:
                    actors[aid].observed_by.append(obs.observer_id)
                if obs.local_track_id not in actors[aid].source_track_ids:
                    actors[aid].source_track_ids.append(obs.local_track_id)

    # 2-b) 나머지는 거리 기반 최근접 배정.
    #
    # 단, 모든 차량 관측자의 본체 트랙 id 를 알고 있으면 2-a 가 이미 신원으로
    # 정확히 판정했으므로 거리 추정은 쓰지 않는다. 여기서 거리 게이트를 또
    # 돌리면 관측차량 옆을 지나는 다른 차량(트랙 2674)이 본체로 흡수되어
    # 같은 차량이 ego 와 별도 액터로 이중 계수된다.
    authoritative = bool(
        cfg.global_track_ids
        and observer_self_ids
        and vehicle_world
        and all(observer_self_ids.get(oid) is not None for oid in vehicle_world)
    )
    cand: List[Tuple[float, int, str]] = []
    for i, obs in enumerate(observations):
        if authoritative or i in absorbed_idx:
            continue
        # 관측차량 본체는 차량이다. 보행자·자전거 관측을 흡수하면 실제 교통
        # 참여자가 장면에서 사라진다 (인프라 카메라가 본 인도의 보행자가
        # 8m 게이트 안에 들어오는 일이 흔하다).
        if not class_compatible(obs.cls, "car"):
            continue
        ego_gate = min(
            cfg.ego_match_dist_m + cfg.range_gate_coeff * obs.distance_m,
            cfg.max_gate_m,
        )
        for oid, xy in vehicle_world.items():
            if oid == obs.observer_id:
                continue
            d = math.dist(obs.world_xy, xy)
            if d <= ego_gate:
                cand.append((d, i, oid))
    cand.sort(key=lambda z: z[0])

    for d, i, oid in cand:
        obs = observations[i]
        key = (obs.observer_id, oid)
        if i in absorbed_idx or key in taken:
            continue
        absorbed_idx.add(i)
        taken.add(key)
        aid = f"EGO_{oid}"
        if aid in actors:
            if obs.observer_id not in actors[aid].observed_by:
                actors[aid].observed_by.append(obs.observer_id)
            # 흡수된 검출도 출처로 기록해 원시 검출까지 역추적 가능하게
            if obs.local_track_id not in actors[aid].source_track_ids:
                actors[aid].source_track_ids.append(obs.local_track_id)

    remaining = [o for i, o in enumerate(observations) if i not in absorbed_idx]

    # 3) 관측을 클러스터로 묶는다
    clusters: List[List[Observation]] = []
    if cfg.global_track_ids:
        # 트랙 id 가 전역 유일하면 같은 id = 같은 물체다. 위치 게이트에 의존하지
        # 않으므로 극근거리 측거 오차로 신원이 갈라지지 않는다.
        by_id: Dict[int, List[Observation]] = {}
        for obs in remaining:
            by_id.setdefault(obs.local_track_id, []).append(obs)
        for k in sorted(by_id):
            clusters.extend(_split_if_id_collision(by_id[k], registry, cfg))
    else:
        # 관측자별로 그룹화 후 그룹 간 위치 매칭
        by_observer: Dict[str, List[Observation]] = {}
        for obs in remaining:
            by_observer.setdefault(obs.observer_id, []).append(obs)
        for oid in sorted(by_observer):
            group = by_observer[oid]
            if not clusters:
                clusters = [[o] for o in group]
                continue
            matches = _match_groups(clusters, group, cfg)
            used = set()
            for ci, gi in matches:
                clusters[ci].append(group[gi])
                used.add(gi)
            for gi, o in enumerate(group):
                if gi not in used:
                    clusters.append([o])

    # 4) 클러스터 → ActorState (신뢰도 가중 평균 위치)
    # 이 시각에 배정된 actor id 를 추적해 두 클러스터가 같은 id 를 쓰지 못하게 한다
    assigned: Set[str] = set(actors)
    for cluster in clusters:
        rep = min(cluster, key=lambda o: o.distance_m)  # 가장 가까운 관측을 대표로
        # 위치 이상치 제거. 트랙 id 로 클러스터를 만들면(global_track_ids) 위치
        # 게이트를 거치지 않으므로, 원거리에서 지평선 근처로 역투영된 관측 하나가
        # 수십 m 벗어난 채 섞여 들어와 평균을 끌고 간다. 최근접 관측 기준
        # 물리적 허용 반경을 넘는 관측은 위치 평균에서 뺀다 (신원은 유지).
        inliers = [
            o
            for o in cluster
            if math.dist(o.world_xy, rep.world_xy)
            <= gate_m(cfg, rep.distance_m, o.distance_m,
                      quality=min(rep.range_quality, o.range_quality))
        ] or [rep]
        wsum = sum(max(o.range_quality * o.conf, 1e-3) for o in inliers)
        e = sum(o.world_xy[0] * max(o.range_quality * o.conf, 1e-3) for o in inliers) / wsum
        n = sum(o.world_xy[1] * max(o.range_quality * o.conf, 1e-3) for o in inliers) / wsum
        # 3D 센서 관측은 값이 정확하므로 **평균하지 않는다**. 평균은 어떤 센서도
        # 보고하지 않은 값을 만들고, 센서 간 불일치가 있으면 그것을 조용히
        # 뭉갠다. 대표 하나를 골라 그 값을 그대로 쓴다.
        exact = [o for o in cluster if o.from_3d_sensor]
        best = _freshest(exact, t) if exact else None
        if best is not None:
            e, n = best.world_xy
        # 클러스터의 모든 트랙 키로 신원을 조회한다 (대표 관측자가 프레임마다
        # 바뀌어도 actor id 가 유지되도록)
        aid = registry.resolve_cluster(
            t,
            [(o.observer_id, o.local_track_id) for o in cluster],
            (e, n),
            rep.cls,
            obs_range_m=rep.distance_m,
            exclude=assigned,
        )
        assigned.add(aid)
        # 존재 확신도: 검출기 신뢰도 + 다중 관측자 일치 보너스
        conf = rep.conf * (1.0 + 0.2 * (len(cluster) - 1))
        # 위치 정확도: 관측들의 거리추정 신뢰도 가중 평균 (별도 값)
        pq = sum(o.range_quality for o in inliers) / len(inliers)
        if len(inliers) > 1:
            # 여러 관측자가 독립적으로 본 위치는 평균으로 오차가 줄어든다
            pq = min(1.0, pq * (1.0 + 0.15 * (len(inliers) - 1)))

        # 3D 센서(라이다/스테레오)가 방위·속도를 직접 준 경우 그 값을 쓴다.
        # 궤적 미분보다 정확하고, 정지·저속에서도 유효하다.
        if best is not None and best.heading_deg is not None:
            head_obs = [best]
        else:
            head_obs = [o for o in cluster if o.heading_deg is not None]
        spd_obs = [o for o in cluster if o.speed_mps is not None]
        heading = None
        if head_obs:
            best = min(head_obs, key=lambda o: o.distance_m)
            heading = best.heading_deg
        speed = None
        if spd_obs:
            fresh = (
                _freshest([o for o in exact if o.speed_mps is not None], t)
                if exact
                else None
            )
            if fresh is not None:
                speed = float(fresh.speed_mps)  # 대표 센서 값 그대로 (평균 금지)
            else:
                speed = float(sum(o.speed_mps for o in spd_obs) / len(spd_obs))

        actors[aid] = ActorState(
            actor_id=aid,
            kind="observed",
            cls=rep.cls,
            # 방위·속도가 없으면 운동학 추정기가 궤적으로 채운다
            world_xy=(e, n),
            heading_deg=heading,
            heading_source="measured" if heading is not None else None,
            speed_mps=speed,
            accel_mps2=None,
            observed_by=sorted({o.observer_id for o in cluster}),
            confidence=min(1.0, conf),
            position_quality=round(pq, 3),
            observed_range_m=rep.distance_m,
            source_track_ids=sorted({o.local_track_id for o in cluster}),
        )
    return list(actors.values())


def _match_groups(
    clusters: List[List[Observation]],
    group: List[Observation],
    cfg: FusionConfig,
) -> List[Tuple[int, int]]:
    """기존 클러스터 ↔ 새 관측 그룹 매칭. scipy 있으면 최적, 없으면 탐욕."""
    if not clusters or not group:
        return []

    def cluster_rep(c: List[Observation]) -> Observation:
        return min(c, key=lambda o: o.distance_m)

    costs = [
        [_cost(cluster_rep(c), o, cfg) for o in group] for c in clusters
    ]

    # _cost 는 게이트로 정규화된 값이므로 임계값은 항상 1.0
    pairs: List[Tuple[int, int]] = []
    if _HAS_SCIPY:
        import numpy as np

        m = np.array(costs, dtype=float)
        m[m >= 1.0] = 1e6  # 불가능한 매칭
        rows, cols = linear_sum_assignment(m)
        for r, c in zip(rows, cols):
            if m[r, c] < 1.0:
                pairs.append((int(r), int(c)))
    else:
        taken_c, taken_g = set(), set()
        flat = sorted(
            (
                (costs[i][j], i, j)
                for i in range(len(clusters))
                for j in range(len(group))
            ),
        )
        for cost, i, j in flat:
            if cost >= 1.0:
                break
            if i in taken_c or j in taken_g:
                continue
            taken_c.add(i)
            taken_g.add(j)
            pairs.append((i, j))
    return pairs
