"""DeepAccident 운동·사고 예측의 재구현 — 실제로 돌아가는 비교 대상.

**왜 재구현인가.** 공개 저장소는 이 머신에서 돌지 않는다. 두 가지가 각각 독립적으로
막는다.

    가중치 없음   `README.md` 의 Model Zoo 절이 통째로 주석 처리돼 있고
                 (`[//]: # (## Model Zoo)`), 살아 있는 링크 둘은 nuScenes 용
                 BEVDet 가중치다. DeepAccident 학습 결과는 배포된 적이 없다.
    빌드 불가     원본은 Python 3.7 / PyTorch 1.10.2 / **CUDA 10.2** / mmcv-full
                 1.3.14 를 요구한다. CUDA 10.2 가 지원하는 최대 아키텍처는 sm_75 인데
                 이 머신의 GPU 는 **sm_120**(Blackwell) 이다. mmcv 의 커스텀 CUDA
                 커널을 sm_120 으로 빌드할 수 없다.

그래서 **방법을 다시 구현한다.** 논문 저자들도 사고 예측은 후처리로 정의했으므로
(운동 예측 → 인스턴스 접촉 판정), 재구현 대상은 명확하다.

**무엇을 맞췄나.**

| 항목 | 값 | 출처 |
|---|---|---|
| BEV 격자 | ±50.0 m @ 0.5 m/px → 200×200 | `configs/DeepAccident_tiny.py` motion grid |
| 관측 | 과거 3 프레임 @ 2 Hz (1초) | 같은 config |
| 예측 | 미래 4 프레임 @ 2 Hz (2초) | 같은 config |
| 표본 | 학습 분포에서 **5개 + 평균 = 6개** | `_base_motion_head.py:315-320` |
| 사고 판정 | 표본 **아무거나** 접촉하면 사고 | 논문 §"prioritizing safety" |
| 접촉 임계 | 5 px × 0.5 m/px = 2.5 m | `multi_gpu_test.py:527` |
| 관측 범위 | ego 중심 ±50 m 밖은 **보이지 않는다** | BEV 격자의 정의 |

**무엇이 다른가 (논문에 반드시 적어야 한다).** 입력이 카메라가 아니라 **라벨에서
래스터화한 BEV** 다. 즉 인지 단계를 우회했다. 이것은 의도적이다 — 우리 파이프라인도
같은 라벨을 "각 차량이 센서로 취득한 정보" 로 쓰므로, 두 방법에 **같은 인지**를 주고
예측·판정 방법만 다르게 하는 통제 비교가 된다. 카메라 프론트엔드는 이 위에 따로
얹는다 (motion head 와 사고 후처리는 그대로 재사용된다).

**운동 예측의 형태도 맞췄다.** 좌표를 액터별로 따로 회귀하는 것이 아니라(그것은
`predict_nets.WaypointNet` 이다), 장면 전체를 BEV 래스터로 보고 **픽셀별 미래 흐름
(flow)** 을 낸다 — FIERY/IterativeFlow 계열의 방식이고 DeepAccident 의 motion head 가
쓰는 것이다. 액터의 미래 위치는 그 액터 픽셀에서 흐름을 샘플링해 얻는다. 이 차이가
중요하다: 장면을 함께 보므로 차들이 서로를 통과하는 예측이 억제된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- 격자


@dataclass(frozen=True)
class BevGrid:
    """DeepAccident 의 motion BEV 격자.

    `configs/DeepAccident_tiny.py` 의 motion grid_conf:
        xbound = [-50.0, 50.0, 0.5], ybound = [-50.0, 50.0, 0.5]
    """

    range_m: float = 50.0
    res_m: float = 0.5

    @property
    def size(self) -> int:
        return int(round(2 * self.range_m / self.res_m))

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        """자기 중심 좌표 [m] → 픽셀 (col, row). 격자 밖도 그대로 돌려준다."""
        return (
            (x + self.range_m) / self.res_m,
            (y + self.range_m) / self.res_m,
        )

    def in_range(self, x: float, y: float) -> bool:
        return abs(x) < self.range_m and abs(y) < self.range_m


#: 래스터 채널. 순서를 바꾸면 학습된 가중치와 어긋난다.
CHANNELS = ("occupancy", "vx", "vy", "cos_yaw", "sin_yaw")
N_CHANNELS = len(CHANNELS)

#: DeepAccident 의 config 값
N_PAST = 3
N_FUTURE = 4
FRAME_DT_S = 0.5  # 2 Hz
N_SAMPLES = 6  # 5 표본 + 분포 평균


def _rect_corners(
    x: float, y: float, yaw_rad: float, length: float, width: float
) -> np.ndarray:
    """중심·방위·크기 → 회전 사각형 꼭짓점 4개 (m 단위, 자기 중심 좌표)."""
    hl, hw = length / 2.0, width / 2.0
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out = np.empty((4, 2), dtype=np.float32)
    for i, (dx, dy) in enumerate(((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))):
        out[i, 0] = x + dx * c - dy * s
        out[i, 1] = y + dx * s + dy * c
    return out


def _fill_rect(grid_shape: Tuple[int, int], corners_px: np.ndarray) -> np.ndarray:
    """볼록 사각형 내부 픽셀 마스크. numpy 만 쓴다 (cv2 의존성을 만들지 않는다)."""
    h, w = grid_shape
    mask = np.zeros((h, w), dtype=bool)
    c0 = np.floor(corners_px.min(axis=0)).astype(int)
    c1 = np.ceil(corners_px.max(axis=0)).astype(int)
    x0, y0 = max(0, c0[0]), max(0, c0[1])
    x1, y1 = min(w, c1[0] + 1), min(h, c1[1] + 1)
    if x1 <= x0 or y1 <= y0:
        return mask
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    inside = np.ones(gx.shape, dtype=bool)
    for i in range(4):
        ax, ay = corners_px[i]
        bx, by = corners_px[(i + 1) % 4]
        # 볼록 다각형: 모든 변에 대해 같은 부호여야 내부다
        cross = (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)
        inside &= cross >= 0
    if not inside.any():
        # 꼭짓점 순서가 반대 방향이면 부호가 뒤집힌다
        inside = np.ones(gx.shape, dtype=bool)
        for i in range(4):
            ax, ay = corners_px[i]
            bx, by = corners_px[(i + 1) % 4]
            cross = (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)
            inside &= cross <= 0
    mask[y0:y1, x0:x1] = inside
    return mask


# ---------------------------------------------------------------- 래스터화

#: DeepAccident 의 motion 라벨은 `only_vehicle=True` 다 (`ConvertMotionLabels`).
#: 보행자·자전거는 운동 예측 대상에서 빠진다 — 그 필터를 그대로 쓴다.
VEHICLE_CLASSES = frozenset({"car", "van", "truck", "bus", "motorcycle"})


def _extent_of(cls: str, class_sizes) -> Tuple[float, float]:
    sz = class_sizes.get(cls) if class_sizes else None
    return (sz.length_m, sz.width_m) if sz else (4.6, 1.93)


@dataclass
class FrameActors:
    """한 프레임의 액터들을 **ego 중심 좌표**로 옮긴 것.

    `ids` 는 액터 id, `xy`/`yaw`/`vel`/`size` 는 같은 순서의 배열이다. **`yaw` 는
    수학 각도(rad)** 다 — 진북 기준 방위각이 아니다 (`ego_frame` 이 변환한다).
    ego 중심 ±range_m 밖은 이미 걸러져 있다 — DeepAccident 의 BEV 격자가
    그만큼만 보기 때문이다.
    """

    ids: List[str]
    xy: np.ndarray  # (N, 2) [m]
    yaw: np.ndarray  # (N,) [rad]
    vel: np.ndarray  # (N, 2) [m/s]
    size: np.ndarray  # (N, 2) (길이, 폭) [m]

    def __len__(self) -> int:
        return len(self.ids)


def ego_frame(snapshot, grid: BevGrid, class_sizes=None,
              ego_id: Optional[str] = None) -> Optional[FrameActors]:
    """스냅샷 → ego 중심 좌표의 차량들.

    원점은 ego 차량의 위치다. **축은 회전시키지 않는다** — 회전까지 맞추면 과거
    프레임끼리 격자가 어긋나 흐름(flow) 라벨이 자기 회전과 뒤섞인다. DeepAccident 는
    `FeatureWarper` 로 자기 운동을 보정해 같은 효과를 낸다.

    ego 를 찾지 못하면 None 을 돌려준다 (그 프레임은 표본에서 뺀다).
    """
    ego_xy = None
    cands = [a for a in snapshot.actors if a.world_xy is not None]
    if ego_id:
        for a in cands:
            if a.actor_id == ego_id:
                ego_xy = a.world_xy
                break
    if ego_xy is None:
        for a in cands:
            if a.actor_id == "EGO_ego_vehicle":
                ego_xy = a.world_xy
                break
    if ego_xy is None:
        for a in cands:
            if a.kind == "ego":
                ego_xy = a.world_xy
                break
    if ego_xy is None:
        return None

    ids: List[str] = []
    xy: List[Tuple[float, float]] = []
    yaw: List[float] = []
    vel: List[Tuple[float, float]] = []
    size: List[Tuple[float, float]] = []
    for a in cands:
        if a.cls not in VEHICLE_CLASSES:
            continue
        dx = a.world_xy[0] - ego_xy[0]
        dy = a.world_xy[1] - ego_xy[1]
        if not grid.in_range(dx, dy):
            continue
        # `heading_deg` 는 **진북 기준 시계방향 방위각**이다 (`geometry.heading_to_unit`:
        # 단위벡터 = (sin h, cos h)). 사각형을 그리는 `_rect_corners` 는 수학 각도
        # (cos θ, sin θ) 를 쓰므로 θ = π/2 − h 로 바꿔 담는다. 이걸 빠뜨리면 모든
        # 차체가 90° 어긋나 접촉 판정이 통째로 틀린다.
        th = (math.pi / 2 - math.radians(a.heading_deg)
              if a.heading_deg is not None else math.pi / 2)
        sp = a.speed_mps or 0.0
        ids.append(a.actor_id)
        xy.append((dx, dy))
        yaw.append(th)
        vel.append((sp * math.cos(th), sp * math.sin(th)))
        size.append(_extent_of(a.cls, class_sizes))
    if not ids:
        return None
    return FrameActors(
        ids=ids,
        xy=np.asarray(xy, dtype=np.float32).reshape(-1, 2),
        yaw=np.asarray(yaw, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32).reshape(-1, 2),
        size=np.asarray(size, dtype=np.float32).reshape(-1, 2),
    )


def rasterize(fr: FrameActors, grid: BevGrid) -> np.ndarray:
    """FrameActors → (C, H, W) 채널 래스터.

    채널은 `CHANNELS` 순서다. 점유 픽셀에만 속도·방위를 채운다 — 빈 공간의 0 과
    "정지한 차" 의 0 을 구별해야 하므로 점유 채널이 먼저 온다.
    """
    n = grid.size
    out = np.zeros((N_CHANNELS, n, n), dtype=np.float32)
    for i in range(len(fr)):
        corners = _rect_corners(
            float(fr.xy[i, 0]), float(fr.xy[i, 1]), float(fr.yaw[i]),
            float(fr.size[i, 0]), float(fr.size[i, 1]),
        )
        px = np.stack(
            [(corners[:, 0] + grid.range_m) / grid.res_m,
             (corners[:, 1] + grid.range_m) / grid.res_m], axis=1
        )
        m = _fill_rect((n, n), px)
        if not m.any():
            continue
        out[0][m] = 1.0
        out[1][m] = fr.vel[i, 0]
        out[2][m] = fr.vel[i, 1]
        out[3][m] = math.cos(float(fr.yaw[i]))
        out[4][m] = math.sin(float(fr.yaw[i]))
    return out


def instance_masks(fr: FrameActors, grid: BevGrid) -> Dict[str, np.ndarray]:
    """액터 id → 그 액터의 점유 마스크. 흐름 라벨을 만들 때 쓴다."""
    n = grid.size
    out: Dict[str, np.ndarray] = {}
    for i, aid in enumerate(fr.ids):
        corners = _rect_corners(
            float(fr.xy[i, 0]), float(fr.xy[i, 1]), float(fr.yaw[i]),
            float(fr.size[i, 0]), float(fr.size[i, 1]),
        )
        px = np.stack(
            [(corners[:, 0] + grid.range_m) / grid.res_m,
             (corners[:, 1] + grid.range_m) / grid.res_m], axis=1
        )
        out[aid] = _fill_rect((n, n), px)
    return out


def flow_label(
    cur: FrameActors, fut: FrameActors, grid: BevGrid
) -> Tuple[np.ndarray, np.ndarray]:
    """현재 프레임 픽셀에 **그 픽셀 차량이 이 미래 프레임까지 움직인 변위**를 적는다.

    돌려주는 것은 `(flow (2,H,W) [m], valid (H,W))` 다. 미래 프레임에 없는 차량은
    valid 가 0 이라 손실에서 빠진다 — 시야 밖으로 나간 차를 억지로 맞히게 하면
    학습이 망가진다.

    이것이 FIERY/IterativeFlow 계열의 라벨 형태이고, DeepAccident 의 motion head 가
    회귀하는 대상이다.
    """
    n = grid.size
    flow = np.zeros((2, n, n), dtype=np.float32)
    valid = np.zeros((n, n), dtype=np.float32)
    fut_xy = {aid: fut.xy[i] for i, aid in enumerate(fut.ids)}
    masks = instance_masks(cur, grid)
    for i, aid in enumerate(cur.ids):
        tgt = fut_xy.get(aid)
        if tgt is None:
            continue
        m = masks[aid]
        if not m.any():
            continue
        flow[0][m] = float(tgt[0] - cur.xy[i, 0])
        flow[1][m] = float(tgt[1] - cur.xy[i, 1])
        valid[m] = 1.0
    return flow, valid


# ---------------------------------------------------------------- 신경망


def _torch():
    """torch 를 늦게 불러온다 — 래스터화·판정만 쓰는 경로는 torch 없이 돈다."""
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401

    return torch, nn, F


def build_net(n_past: int = N_PAST, n_future: int = N_FUTURE, width: int = 64,
              latent_dim: int = 32):
    """DeepAccident motion head 의 구조를 옮긴 신경망.

    네 부분이 원본과 대응한다.

        BEV 인코더      프레임별 2D conv (원본은 카메라 → LSS 로 BEV 특징을 만들지만,
                        우리는 라벨 래스터가 곧 BEV 특징이다)
        시간 융합       과거 프레임 위의 3D conv — 원본의 `TemporalBlock` 자리
        확률 분포       현재 상태에서 (mu, logvar) 를 내고 표본을 뽑는다.
                        원본은 표본 5개 + 분포 평균 = 6개를 쓴다
        IterativeFlow   미래 프레임마다 픽셀별 흐름을 하나씩 내놓는다. 이전 단계의
                        흐름을 다음 단계 입력에 넣어 **누적**한다 — 이름 그대로다

    폭(width)은 원본(BEVerse-tiny)보다 작다. 입력이 카메라 특징이 아니라 5채널
    래스터라 표현력이 그만큼 덜 필요하고, 이 비교의 관심사는 백본 용량이 아니라
    "장면을 함께 보는 흐름 예측" 이라는 형태다.
    """
    torch, nn, F = _torch()

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class Encoder(nn.Module):
        """(B*T, C, H, W) → (B*T, width, H/8, W/8)"""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                block(N_CHANNELS, width // 2),
                block(width // 2, width, stride=2),
                block(width, width),
                block(width, width, stride=2),
                block(width, width),
                block(width, width, stride=2),
                block(width, width),
            )

        def forward(self, x):
            return self.net(x)

    class DAMotionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_past = n_past
            self.n_future = n_future
            self.latent_dim = latent_dim
            self.encoder = Encoder()
            # 시간 융합 — 원본의 TemporalBlock 자리
            self.temporal = nn.Sequential(
                nn.Conv3d(width, width, (n_past, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(width),
                nn.ReLU(inplace=True),
            )
            # 확률 분포 (mu, logvar). 원본은 학습 시 미래를 본 사후 분포도 쓰지만,
            # 추론 경로는 현재까지만 보는 사전 분포다 — 그쪽을 구현한다.
            self.dist = nn.Conv2d(width, 2 * latent_dim, 1)
            # IterativeFlow: 이전 흐름을 함께 받는다. 여기까지는 1/8 해상도에서
            # 장면 맥락을 본다.
            self.flow_head = nn.Sequential(
                block(width + latent_dim + 2, width),
                block(width, width),
            )
            self.flow_out = nn.Conv2d(width, 2, 1)
            # 1/8 (4 m/셀) 에서 그대로 읽으면 차 한 대가 한 셀보다 작아 접촉 판정이
            # 불가능하다. FIERY 처럼 **업샘플 디코더**로 원해상도(0.5 m/px)까지
            # 되돌린 뒤 흐름·점유를 낸다.
            self.up = nn.ModuleList([
                nn.Sequential(block(width, width // 2), block(width // 2, width // 2)),
                nn.Sequential(block(width // 2, width // 4), block(width // 4, width // 4)),
                nn.Sequential(block(width // 4, width // 4), block(width // 4, width // 4)),
            ])
            self.head_flow = nn.Conv2d(width // 4, 2, 1)
            self.head_occ = nn.Conv2d(width // 4, 1, 1)
            # 사고 헤드. 미래 프레임마다 "여기서 사고가 난다" 를 BEV 열지도로 낸다.
            # 모든 쌍의 최소 간격에 임계를 거는 후처리는 이 데이터에서 포화된다
            # (기록된 미래에서도 89 % 가 2.5 m 안에 든다). 모델이 **어느 자리**를
            # 사고로 지목하게 하면 그 문제를 피한다 — 원본의 MultiTaskHead 에
            # 헤드를 하나 더 다는 것과 같은 구조다.
            self.head_acc = nn.Conv2d(width // 4, 1, 1)

        def encode(self, past):
            """past: (B, T, C, H, W) → 장면 특징 (B, width, h, w)"""
            b, t = past.shape[:2]
            f = self.encoder(past.flatten(0, 1))
            f = f.view(b, t, *f.shape[1:]).permute(0, 2, 1, 3, 4)  # B,C,T,h,w
            return self.temporal(f).squeeze(2)

        def sample_latents(self, feat, n_samples: int, generator=None):
            """(B, S, latent_dim, h, w). **마지막 표본이 분포 평균**이다.

            원본이 `samples[-1] = mu` 로 두고 평균만 운동 지표에 쓰므로, 그 규약을
            그대로 따른다.
            """
            stats = self.dist(feat)
            mu, logvar = stats[:, : self.latent_dim], stats[:, self.latent_dim:]
            logvar = logvar.clamp(-8.0, 8.0)
            std = (0.5 * logvar).exp()
            outs = []
            for i in range(n_samples):
                if i == n_samples - 1:
                    outs.append(mu)
                else:
                    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype,
                                      generator=generator)
                    outs.append(mu + eps * std)
            return torch.stack(outs, dim=1), mu, logvar

        def decode(self, feat, z):
            """feat (B,C,h,w), z (B,L,h,w) → 흐름 (B,F,2,h,w), 점유 (B,F,1,h,w).

            흐름은 **누적**이다: k번째 출력은 현재에서 k번째 미래 프레임까지의 변위.
            """
            b, _, h, w = feat.shape
            prev = feat.new_zeros(b, 2, h, w)
            flows, occs, accs = [], [], []
            for _ in range(self.n_future):
                x = torch.cat([feat, z, prev], dim=1)
                f = self.flow_head(x)
                prev = prev + self.flow_out(f)  # IterativeFlow — 이전 위에 쌓는다
                # 원해상도로 되돌린다 (1/8 → 1/4 → 1/2 → 1/1)
                u = f
                for up in self.up:
                    u = up(F.interpolate(u, scale_factor=2, mode="bilinear",
                                         align_corners=False))
                flows.append(
                    F.interpolate(prev, size=u.shape[-2:], mode="bilinear",
                                  align_corners=False) + self.head_flow(u)
                )
                occs.append(self.head_occ(u))
                accs.append(self.head_acc(u))
            return torch.stack(flows, 1), torch.stack(occs, 1), torch.stack(accs, 1)

        def forward(self, past, n_samples: int = 1, generator=None):
            feat = self.encode(past)
            zs, mu, logvar = self.sample_latents(feat, n_samples, generator)
            flows, occs, accs = [], [], []
            for i in range(n_samples):
                f, o, a = self.decode(feat, zs[:, i])
                flows.append(f)
                occs.append(o)
                accs.append(a)
            # (B, S, F, 2, H, W) / (B, S, F, 1, H, W) / (B, S, F, 1, H, W)
            return (torch.stack(flows, 1), torch.stack(occs, 1),
                    torch.stack(accs, 1), mu, logvar)

    return DAMotionNet()


# ---------------------------------------------------------------- 판독
#
# 예측된 흐름장 → 액터별 미래 위치 → 인스턴스 폴리곤 사이 거리 → 사고 판정.
# 마지막 단계는 `da_baseline` 의 이식된 규칙이 그대로 받는다.


def _separated(A: np.ndarray, B: np.ndarray) -> bool:
    """분리축 정리로 두 볼록 사각형이 떨어져 있는지."""
    for P in (A, B):
        n = len(P)
        for i in range(n):
            ax, ay = P[i]
            bx, by = P[(i + 1) % n]
            nx, ny = -(by - ay), (bx - ax)
            pa = A @ np.array([nx, ny])
            pb = B @ np.array([nx, ny])
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


def _seg_point_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    t = 0.0 if dd == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / dd))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def poly_gap(A: np.ndarray, B: np.ndarray) -> float:
    """두 사각형 사이 최단거리 [m]. 겹치면 0.

    DeepAccident 의 `poly_distance` (shapely `Polygon.distance`) 와 같은 양이다.
    """
    if not _separated(A, B):
        return 0.0
    best = float("inf")
    for P, Q in ((A, B), (B, A)):
        n = len(Q)
        for px, py in P:
            for i in range(n):
                ax, ay = Q[i]
                bx, by = Q[(i + 1) % n]
                d = _seg_point_dist(px, py, ax, ay, bx, by)
                if d < best:
                    best = d
    return best


def read_tracks(
    flow: np.ndarray, fr: FrameActors, grid: BevGrid
) -> np.ndarray:
    """흐름장에서 액터별 미래 위치를 읽는다 → (F, N, 2) [m].

    `flow` 는 (F, 2, H, W) 이고 값은 **현재에서 그 미래 프레임까지의 변위** 다.
    액터의 변위는 그 액터 점유 마스크 위 흐름의 **평균**으로 잡는다 — 중심 픽셀
    하나만 읽으면 흐름장의 국소 잡음에 흔들린다. 이것이 픽셀별 흐름을 인스턴스
    운동으로 바꾸는 FIERY 계열의 표준 판독이다.
    """
    n_future = flow.shape[0]
    masks = instance_masks(fr, grid)
    out = np.zeros((n_future, len(fr), 2), dtype=np.float32)
    for i, aid in enumerate(fr.ids):
        m = masks[aid]
        for k in range(n_future):
            if m.any():
                dx = float(flow[k, 0][m].mean())
                dy = float(flow[k, 1][m].mean())
            else:
                dx = dy = 0.0
            out[k, i, 0] = fr.xy[i, 0] + dx
            out[k, i, 1] = fr.xy[i, 1] + dy
    return out


#: 이 속도 이하는 정지로 본다 [m/s]. 라벨 속도의 잡음보다 크고 서행보다 작다.
STATIONARY_MPS = 0.5


def track_gaps(
    tracks: np.ndarray, fr: FrameActors, exclude_touching_now: bool = True,
    exclude_static_pairs: bool = True, stationary_mps: float = STATIONARY_MPS,
) -> List[Tuple[float, Optional[Tuple[str, str]]]]:
    """미래 프레임마다 (모든 쌍 최소 간격, 그 쌍). 길이는 미래 프레임 수.

    방위는 변위 방향에서 다시 잡는다 — 흐름 예측은 위치만 주므로, 차체 사각형을
    그리려면 진행 방향이 필요하다. 변위가 거의 없으면 현재 방위를 쓴다.

    두 필터가 후보 쌍을 줄인다. 둘 다 "예측할 사고가 아닌 것" 을 뺀다.

    `exclude_static_pairs`
        **둘 다 정지한 쌍**을 뺀다. 주차된 차 두 대는 사고를 낼 수 없다.
        DeepAccident 라벨은 주차 차량을 줄지어 놓는데 그 박스들이 서로 겹쳐 있어
        (중심거리 3~4 m 인데 차 길이 4.5~5.4 m) 최소 간격이 영구히 0 이 된다.
        val 라벨 400프레임 실측: 겹치는 쌍 537개 중 **526개가 정지-정지**,
        정지-이동은 0개, 이동-이동은 11개(실제 충돌)였다. 이 필터 하나로 겹치는
        프레임이 **31 % → 2.8 %** 로 떨어진다. 한쪽만 정지한 쌍은 남긴다 —
        신호 대기 중 추돌당하는 것은 사고다.

    `exclude_touching_now`
        관측 시점에 이미 붙어 있는 쌍을 뺀다. 사고 예측은 **벌어져 있다가 닫히는**
        쌍의 문제다.

    치수를 객체별 라벨 값으로 바꿔도 겹침은 거의 그대로다 (0.310 → 0.305). 즉
    문제는 치수 출처가 아니라 정지 차량 쌍이다.
    """
    n_future, n_actor, _ = tracks.shape
    if n_actor < 2:
        return [(float("inf"), None)] * n_future

    now_corners = [
        _rect_corners(float(fr.xy[i, 0]), float(fr.xy[i, 1]), float(fr.yaw[i]),
                      float(fr.size[i, 0]), float(fr.size[i, 1]))
        for i in range(n_actor)
    ]
    speed = np.hypot(fr.vel[:, 0], fr.vel[:, 1])
    moving = speed > stationary_mps
    banned = set()
    for i in range(n_actor):
        for j in range(i + 1, n_actor):
            if exclude_static_pairs and not (moving[i] or moving[j]):
                banned.add((i, j))
            elif exclude_touching_now and poly_gap(now_corners[i], now_corners[j]) <= 0.0:
                banned.add((i, j))

    out: List[Tuple[float, Optional[Tuple[str, str]]]] = []
    prev = fr.xy
    for k in range(n_future):
        pos = tracks[k]
        corners = []
        for i in range(n_actor):
            dx, dy = pos[i, 0] - prev[i, 0], pos[i, 1] - prev[i, 1]
            yaw = (math.atan2(dy, dx) if math.hypot(dx, dy) > 0.05
                   else float(fr.yaw[i]))
            corners.append(
                _rect_corners(float(pos[i, 0]), float(pos[i, 1]), yaw,
                              float(fr.size[i, 0]), float(fr.size[i, 1]))
            )
        best, pair = float("inf"), None
        for i in range(n_actor):
            for j in range(i + 1, n_actor):
                if (i, j) in banned:
                    continue
                g = poly_gap(corners[i], corners[j])
                if g < best:
                    best, pair = g, (fr.ids[i], fr.ids[j])
        out.append((best, pair))
        prev = pos
    return out


def frames_to_buckets(
    frame_gaps: Sequence[Sequence[Tuple[float, Optional[Tuple[str, str]]]]],
    n_buckets: int,
    dt_s: float = FRAME_DT_S,
) -> List[Tuple[float, Optional[Tuple[str, str]]]]:
    """미래 **프레임**별 간격(표본 여러 개) → 우리 **1초 구간**별 간격.

    두 가지를 동시에 접는다.

        표본     DeepAccident 는 표본 6개 중 **아무거나** 사고를 가리키면 사고로
                 본다 ("prioritizing safety"). 그러므로 표본 축은 **최솟값**이다.
        프레임   구간 k = (k-1, k] 초에 드는 프레임들의 최솟값.

    예측 지평(2초) 밖 구간은 무한대다 — 그 규칙은 거기를 보지 않는다.
    """
    out: List[Tuple[float, Optional[Tuple[str, str]]]] = []
    n_frames = len(frame_gaps[0]) if frame_gaps else 0
    for k in range(1, n_buckets + 1):
        lo, hi = (k - 1), k
        best, pair = float("inf"), None
        for f in range(n_frames):
            t = (f + 1) * dt_s
            if not (lo < t <= hi + 1e-9):
                continue
            for sample in frame_gaps:
                g, p = sample[f]
                if g < best:
                    best, pair = g, p
        out.append((best, pair))
    return out


# ---------------------------------------------------------------- 사고 헤드
#
# "모든 쌍의 최소 간격 < 임계" 후처리는 이 데이터에서 판정을 못 한다. val 816창
# 임계값 스윕에서 균형정확도 최대가 0.541(우연 0.500)이었고, 원인은 모델이 아니라
# 양(quantity)이다 — 기록된 실제 미래에서도 표본의 89 % 가 어떤 쌍이든 2.5 m 안에
# 든다(중앙값 간격 1.34 m). 2.5 m 차체 간격은 교차로의 정상 주행 거리다.
#
# 그래서 **모델이 사고 자리를 직접 지목하게** 한다. 원본도 사고를 "예측된 인스턴스
# 중 가장 가까운 쌍" 으로 정의하는데, 그 인스턴스 집합이 자기 BEV 모델이 내놓은
# 소수라서 성립한다. 장면의 모든 차량 쌍에 같은 규칙을 씌우면 무너진다.


def collision_heatmap_label(
    grid: BevGrid, xy: Optional[Tuple[float, float]], sigma_m: float = 3.0
) -> np.ndarray:
    """사고 지점 하나 → (H, W) 가우시안 열지도. 사고가 없으면 0 이다.

    점 하나를 그대로 쓰면 양성 픽셀이 4만분의 1이라 학습이 안 된다. 반경
    `sigma_m` 의 가우시안으로 퍼뜨린다 — CenterPoint 계열의 중심 열지도와 같은
    방식이다.
    """
    n = grid.size
    out = np.zeros((n, n), dtype=np.float32)
    if xy is None or not grid.in_range(xy[0], xy[1]):
        return out
    cx, cy = grid.to_px(xy[0], xy[1])
    s = sigma_m / grid.res_m
    r = int(math.ceil(3 * s))
    x0, x1 = max(0, int(cx) - r), min(n, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(n, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return out
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    out[y0:y1, x0:x1] = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * s * s))
    return out


def read_accident(
    acc_logits: np.ndarray, grid: BevGrid, threshold: float = 0.5
) -> dict:
    """사고 헤드 출력 → 판정 1건.

    `acc_logits` 는 (S, F, H, W) 로짓이다. 표본 축은 **최댓값**으로 접는다 —
    원본이 "표본 아무거나 사고를 가리키면 사고" 규칙을 쓰기 때문이다
    (논문 "prioritizing safety").

    돌려주는 것: `{accident, frame, score, xy}`. `frame` 은 0-기반 미래 프레임
    번호이고 `xy` 는 그 프레임에서 가장 강한 픽셀의 자기 중심 좌표 [m] 다.
    """
    prob = 1.0 / (1.0 + np.exp(-acc_logits))
    per_frame = prob.max(axis=0)  # (F, H, W) — 표본 최댓값
    scores = per_frame.reshape(per_frame.shape[0], -1).max(axis=1)
    f = int(scores.argmax())
    score = float(scores[f])
    flat = int(per_frame[f].argmax())
    row, col = divmod(flat, grid.size)
    xy = (
        (col + 0.5) * grid.res_m - grid.range_m,
        (row + 0.5) * grid.res_m - grid.range_m,
    )
    return {
        "accident": bool(score >= threshold),
        "frame": f,
        "score": round(score, 4),
        "xy": (round(xy[0], 2), round(xy[1], 2)),
        "per_frame_score": [round(float(x), 4) for x in scores],
    }


def head_response(
    t_end: float,
    n_buckets: int,
    acc_logits: np.ndarray,
    grid: BevGrid,
    fr: Optional[FrameActors] = None,
    threshold: float = 0.5,
    dt_s: float = FRAME_DT_S,
) -> dict:
    """사고 헤드 판정을 `PREDICTION_SCHEMA` 형태의 응답으로.

    LLM 응답과 **같은 채점기**(`score_modes`)를 통과해야 같은 표에 오른다.
    지목한 프레임이 속한 1초 구간 하나만 True 로 둔다.

    `fr` 을 주면 지목한 자리에서 가장 가까운 차량 둘을 관련 차량으로 적는다.
    """
    v = read_accident(acc_logits, grid, threshold)
    n_future = acc_logits.shape[1]
    hit_k = int(math.ceil((v["frame"] + 1) * dt_s - 1e-9)) if v["accident"] else None
    horizon_k = int(math.ceil(n_future * dt_s - 1e-9))

    ids: List[str] = []
    if v["accident"] and fr is not None and len(fr):
        d = np.hypot(fr.xy[:, 0] - v["xy"][0], fr.xy[:, 1] - v["xy"][1])
        ids = [fr.ids[i] for i in np.argsort(d)[:2]]

    preds: List[dict] = []
    for k in range(1, n_buckets + 1):
        hit = k == hit_k
        if k > horizon_k:
            why = f"지평 {n_future * dt_s:g}초 밖 — 이 모델은 보지 않는다"
        elif hit:
            why = f"사고 열지도 최대 {v['score']:.3f} @ {v['xy']} (자기 중심 좌표)"
        else:
            why = f"사고 열지도가 임계 {threshold:g} 미만"
        preds.append({
            "k": k,
            "interval_s": f"({t_end + k - 1:.1f}, {t_end + k:.1f}]",
            "accident_expected": hit,
            "involved_actor_ids": list(ids) if hit else [],
            "reason": why,
            "confidence": "high" if v["score"] > 0.8 else "medium",
        })
    return {
        "predictions": preds,
        "overall_assessment": (
            "DeepAccident 재구현 사고 헤드: "
            + ("사고 예측" if v["accident"] else "사고 없음")
            + f" (최대 점수 {v['score']:.3f})"
        ),
        "data_limitations": (
            "학습된 인지(카메라→BEV)가 아니라 라벨 래스터를 입력으로 쓴다. "
            "공개 저장소에 DeepAccident 학습 가중치가 없다."
        ),
        "deepaccident_head": v,
    }
