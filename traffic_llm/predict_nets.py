"""경로 예측용 PyTorch 신경망.

**이 모듈은 import 시점에 torch 를 요구한다.** `predict_model.py` 는 torch 없이도
동작해야 하므로 신경망 정의를 여기로 분리했다. 규칙 기반만 쓰는 쪽은 이 모듈을
import 하지 않는다.

왜 패키지 안에 두는가
    `torch.save(model, path)` 는 클래스를 **경로로** 저장한다. 학습 스크립트 안에
    정의하면 `__main__.RankNet` 으로 저장되어 다른 프로세스에서 불러올 때
    `AttributeError: Can't get attribute 'RankNet' on <module '__main__'>` 가 난다.
    패키지 모듈에 두면 `traffic_llm.predict_nets.RankNet` 으로 저장되어 어디서든
    불러온다.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .predict_model import (
    IDX_HEADING_COS,
    IDX_HEADING_SIN,
    IDX_SPEED,
    MANEUVER_CLASSES,
    N_CANDIDATE_FEATURES,
    N_GLOBAL_FEATURES,
    N_HISTORY_FEATURES,
    N_HISTORY_STEPS,
    N_INTERACTION_FEATURES,
)

# 마스킹용 큰 음수. `-inf` 를 쓰지 않는다 — 한 행이 전부 패딩이면 softmax 가 NaN 이
# 되고 그 NaN 이 하류 전체를 오염시킨다. 유한한 값이면 최악의 경우도 큰 음수로 끝난다.
# 어텐션·최대풀링·`masked_scores` 가 **같은 값**을 쓴다 (같은 목적에 다른 상수를 쓰면
# 읽는 쪽이 다른 의도로 오해한다).
MASK_NEG = -1e9


class _Normalized(nn.Module):
    """입력 표준화를 모델 **안에** 넣는다.

    특징의 눈금이 크게 다르다 — 속도 0~30, 교차로 거리 0~200, 후보 길이 0~150,
    원-핫 0/1. 그대로 넣으면 첫 선형층이 큰 눈금 쪽에 지배되어 학습이 거의 진행되지
    않는다 (실측: 정규화 없이 30 epoch 손실 10.28→10.03, 등속을 못 넘었다).

    평균·표준편차를 **버퍼로 저장**하므로 `torch.save(model, path)` 에 함께 실려
    추론 시 같은 변환이 적용된다. 스크립트 밖에서 모델을 불러 쓸 때 전처리를 따로
    맞춰야 하는 문제가 생기지 않는다.
    """

    def _init_norm(self, n_global: int, n_cand: int):
        self.register_buffer("g_mu", torch.zeros(n_global))
        self.register_buffer("g_sd", torch.ones(n_global))
        self.register_buffer("c_mu", torch.zeros(n_cand))
        self.register_buffer("c_sd", torch.ones(n_cand))

    @torch.no_grad()
    def fit_normalizer(self, g: torch.Tensor, c: torch.Tensor,
                       mask: torch.Tensor):
        """학습셋으로 평균·표준편차를 정한다. **학습셋만 써야 한다.**

        `mask` 로 패딩 후보를 뺀다. 패딩은 0 으로 채운 자리이므로 통계에 넣으면
        평균이 0 쪽으로, 표준편차가 실제보다 크게 끌려간다 — 후보 수가 표본마다
        다르므로 그 왜곡의 크기도 일정하지 않다.

        (패딩 판정 자체는 `forward` 가 **정규화 전** 값으로 하고, 정규화 뒤 다시
        0 으로 덮는다. 정규화하면 0 이 0 으로 남지 않으므로 그 순서가 중요하다.)
        """
        self.g_mu.copy_(g.mean(0))
        self.g_sd.copy_(g.std(0, unbiased=False).clamp(min=1e-3))
        flat = c[mask]                       # [유효 후보 수, n_cand]
        if flat.numel():
            self.c_mu.copy_(flat.mean(0))
            self.c_sd.copy_(flat.std(0, unbiased=False).clamp(min=1e-3))

    def _norm(self, g: torch.Tensor, c: torch.Tensor):
        return (g - self.g_mu) / self.g_sd, (c - self.c_mu) / self.c_sd


class RankNet(_Normalized):
    """후보마다 점수를 내는 MLP. `TorchPredictor(mode="rank")` 가 쓰는 형태.

    전역 특징을 후보마다 이어 붙여 점수를 낸다 — 후보 수가 표본마다 달라도 되고,
    후보끼리의 상호작용을 보지 않으므로 파라미터가 후보 수와 무관하다.

    **두 가지 입력 모양을 모두 받는다.**

        추론 (TorchPredictor):  g [22]      c [N, 12]     → [N]
        학습 (배치, 패딩):      g [B, 22]   c [B, N, 12]  → [B, N]

    추론 쪽 모양을 반드시 지원해야 한다 — 학습만 되는 모델은 파이프라인에 꽂히지
    않는다.
    """

    def __init__(
        self,
        n_global: int = N_GLOBAL_FEATURES,
        n_cand: int = N_CANDIDATE_FEATURES,
        hidden: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [n_global + n_cand] + [hidden] * depth
        layers: list = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.mlp = nn.Sequential(*layers)
        self._init_norm(n_global, n_cand)
        # 저장된 모델을 나중에 읽을 때 무엇으로 학습했는지 알 수 있게 남긴다.
        self.spec = {
            "n_global": n_global, "n_cand": n_cand,
            "hidden": hidden, "depth": depth, "dropout": dropout,
        }

    def forward(self, g: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 패딩 판정(prior > 0)은 **정규화 전** 값으로 해야 한다
        keep = candidate_mask(c)
        g, c = self._norm(g, c)
        c = c * keep.unsqueeze(-1)            # 패딩 자리를 다시 0 으로
        if c.dim() == 2:                      # 추론: g[22], c[N,12]
            g = g.reshape(1, -1).expand(c.shape[0], -1)
        else:                                 # 학습: g[B,22], c[B,N,12]
            g = g.unsqueeze(1).expand(-1, c.shape[1], -1)
        return self.mlp(torch.cat([g, c], dim=-1)).squeeze(-1)


def candidate_mask(c: torch.Tensor) -> torch.Tensor:
    """패딩이 아닌 후보 위치. `forward(g, c)` 계약에 마스크 인자가 없어 유도한다.

    후보 특징 0번 칸은 사전확률이고 **항상 양수**다 (`downstream_paths` 의 가중치를
    정규화한 값이라 0 이 될 수 없다). 패딩은 0 으로 채우므로 그것으로 구별된다.
    """
    return c[..., 0] > 0.0


class HistoryEncoder(nn.Module):
    """과거 궤적 [B, T, 5] → 고정 길이 벡터 [B, H].

    `kind` 로 구조를 고른다.

    `transformer`  (기본) 학습 위치 임베딩 + self-attention 1층 + 마스킹 평균 풀링.
           시드 3회 실측에서 시험 ADE 가 일관되게 가장 낮았다 (3.003m).
    `gru`  단방향 GRU 의 **마지막 유효 단계** 은닉상태. 파라미터가 더 적다 (3.023m).
    `mlp`  평탄화 후 MLP. 순서 정보를 구조로 주지 않는 **대조군**이다 — 계열 구조가
           실제로 기여하는지 재려면 이것과 비교해야 한다.

    수치와 이력 보유량별 분해는 `docs/waypointnet.md` §2.5.

    관측이 없는 단계는 `valid`(마지막 칸)가 0 이다. GRU 는 그 행이 0 이라 자연히
    영향이 작지만, transformer 는 **명시적으로 마스킹**해야 패딩을 평균에 섞지 않는다.
    """

    KINDS = ("gru", "transformer", "mlp")

    def __init__(
        self,
        n_feat: int = N_HISTORY_FEATURES,
        n_steps: int = N_HISTORY_STEPS,
        hidden: int = 64,
        kind: str = "transformer",
        dropout: float = 0.1,
    ):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"history_encoder 는 {self.KINDS} 중 하나여야 합니다: {kind}")
        self.kind = kind
        self.n_steps = n_steps
        self.out_dim = hidden
        if kind == "gru":
            self.rnn = nn.GRU(n_feat, hidden, batch_first=True)
        elif kind == "transformer":
            self.proj = nn.Linear(n_feat, hidden)
            self.pos = nn.Parameter(torch.zeros(1, n_steps, hidden))
            self.enc = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_feat * n_steps, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, n_feat] — 마지막 칸이 valid
        valid = h[..., -1] > 0.5                        # [B, T]
        if self.kind == "gru":
            out, _ = self.rnn(h)
            # 마지막 은닉상태를 쓰되, **마지막 유효 단계**의 것을 고른다.
            # 최근 관측이 끊긴 트랙에서 그냥 [-1] 을 쓰면 0 행을 통과한 상태를 쓴다.
            idx = (
                valid.float()
                * torch.arange(h.shape[1], device=h.device, dtype=torch.float32)
            ).argmax(dim=1)
            picked = out[torch.arange(h.shape[0], device=h.device), idx]
            # 유효 단계가 아예 없으면(새 트랙) 0 을 낸다 — "이력 없음"을 0 으로
            # 표현하고, 아래 `has_history` 플래그로 구별한다.
            return picked * valid.any(dim=1, keepdim=True).float()
        if self.kind == "transformer":
            z = self.proj(h) + self.pos
            # **전부 패딩인 행을 만들면 안 된다.** 모든 위치를 마스킹하면 어텐션
            # softmax 가 분모 0 이 되어 NaN 이 나오고, 그 NaN 이 손실 전체를
            # 오염시킨다 (실측으로 이력 없는 새 트랙에서 그렇게 됐다).
            # 그런 행은 한 자리를 열어 두고 **출력을 0 으로 덮는다** — GRU 쪽과
            # 같은 규약("이력 없음"은 0)이다.
            any_valid = valid.any(dim=1, keepdim=True)
            keep = valid.clone()
            keep[:, -1] |= ~any_valid.squeeze(1)
            z = self.enc(z, src_key_padding_mask=~keep)
            # 패딩을 평균에 섞지 않는다
            m = valid.unsqueeze(-1).float()
            pooled = (z * m).sum(1) / m.sum(1).clamp(min=1.0)
            return pooled * any_valid.float()
        return self.mlp(h.reshape(h.shape[0], -1))


class WaypointNet(_Normalized):
    """좌표를 직접 내는 모델. `TorchPredictor(mode="waypoints")` 가 쓰는 형태.

    후보 중 하나를 고르는 것이 아니라 **미래 좌표를 회귀한다.** 후보 집합에 정답이
    없는 경우(실측 21.4%)에도 정답을 낼 수 있는 것이 이 방식의 이유다. 대신 지도
    제약이 없으므로 **원리적으로** 노면을 벗어날 수 있다 (Town05 한 시나리오에서는
    예측 웨이포인트 2,035개 중 도로 미매칭 0개였다 — 다른 타운은 확인하지 않았다).

        추론:  g [22]     c [N, 12]     → [K, 2]
        학습:  g [B, 22]  c [B, N, 12]  → [B, K, 2]

    출력은 **현재 위치 기준 상대 좌표**이고 t = 1..K초에 해당한다 (t=0 은 항상
    원점이므로 내지 않는다 — 상수를 회귀하게 만들 이유가 없다).

    후보는 개수가 표본마다 다르므로 **집계해서** 쓴다. 후보별로 인코딩한 뒤 학습된
    가중치로 가중합하고(어텐션), 평균·최대와 함께 이어 붙인다. 후보를 무시하면
    회전을 알 수 없고, 순서에 의존하면 목록 순서를 외운다.

    `predict_maneuver=True`(기본)면 **좌표와 기동 로짓 두 개를 낸다.** 반환형이
    두 가지이므로 `forward` 에 단일 텐서 주석을 달지 않는다.

    구조·설계 근거·실측치는 `docs/waypointnet.md` 에 있다.
    """

    def __init__(
        self,
        n_global: int = N_GLOBAL_FEATURES,
        n_cand: int = N_CANDIDATE_FEATURES,
        hidden: int = 128,
        n_steps: int = 5,
        dropout: float = 0.1,
        residual: bool = True,
        predict_maneuver: bool = True,
        # 과거 궤적 인코더. "none" 이면 이력을 아예 쓰지 않는다 (예전 동작).
        # 기본은 transformer — 시드 3회 실측에서 시험 ADE 가 일관되게 가장 낮았다
        # (transformer 3.003 / gru 3.023 / none 3.040m, docs/waypointnet.md §2.5).
        history_encoder: str = "transformer",
        history_hidden: int = 64,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.residual = residual
        self.predict_maneuver = predict_maneuver
        self.hist = (
            None if history_encoder == "none"
            else HistoryEncoder(hidden=history_hidden, kind=history_encoder,
                                dropout=dropout)
        )
        # `TorchPredictor` 가 3번째 인자를 넘길지 판정할 때 쓴다. 서명만 봐도
        # 알 수 있지만, 이력을 안 쓰는 설정("none")까지 구별하려면 명시해야 한다.
        self.accepts_history = self.hist is not None
        hist_dim = 0 if self.hist is None else self.hist.out_dim + 1
        self.cand_enc = nn.Sequential(
            nn.Linear(n_cand, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.attn = nn.Linear(hidden, 1)
        # `trunk` 를 좌표 헤드와 기동 헤드가 공유한다 — 같은 궤적을 설명하는 두
        # 출력이 서로 모순되지 않게 하려면 같은 표현에서 나와야 한다.
        self.trunk = nn.Sequential(
            nn.Linear(n_global + hidden * 3 + hist_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.wp_head = nn.Linear(hidden, n_steps * 2)
        # 기동 분류 헤드. **좌표에서 사후에 되짚지 않고 모델이 직접 낸다.**
        # 근거는 "규칙 기반을 이긴다"가 아니다 — 후보를 따라가는 표본에서는 지도에서
        # 읽는 쪽이 여전히 낫다 (98.6% 대 92.2%). 근거는 두 가지다.
        #   (1) 전체 정확도가 높다 (83.9% 대 77.0%) — "후보 밖"을 표현할 수 있어서다.
        #   (2) 학습되는 출력이어야 정확도를 **잴 수 있다.** 사후 주석은 검증 대상이
        #       없고, 실제로 오래 재지 않은 채로 남아 있었다.
        # 자세한 비교는 docs/predict_model_io.md §5.8.
        self.man_head = (
            nn.Linear(hidden, len(MANEUVER_CLASSES)) if predict_maneuver else None
        )
        self._init_norm(n_global, n_cand)
        self.spec = {
            "n_global": n_global, "n_cand": n_cand, "hidden": hidden,
            "n_steps": n_steps, "dropout": dropout, "residual": residual,
            "predict_maneuver": predict_maneuver,
            "maneuver_classes": list(MANEUVER_CLASSES),
            "history_encoder": history_encoder,
            "history_hidden": history_hidden,
            "n_history_steps": N_HISTORY_STEPS,
            "n_history_features": N_HISTORY_FEATURES,
        }
        if residual:
            # 마지막 층을 0 으로 시작해 **학습 전 출력이 정확히 등속**이 되게 한다.
            # 안전한 기준선에서 출발하므로 초기 손실이 작고, 모델은 보정만 배운다.
            nn.init.zeros_(self.wp_head.weight)
            nn.init.zeros_(self.wp_head.bias)

    def forward(self, g: torch.Tensor, c: torch.Tensor,
                h: Optional[torch.Tensor] = None):
        """`predict_maneuver` 면 `(좌표, 기동 로짓)`, 아니면 좌표만 낸다.

        반환형이 두 가지이므로 단일 텐서로 주석을 달 수 없다. 받는 쪽
        (`TorchPredictor._from_waypoints`)이 tuple 여부로 갈라 처리한다.

        `h` 는 과거 궤적 `[T, 5]`(추론) 또는 `[B, T, 5]`(학습)다. 이력 인코더를
        쓰는 설정인데 `h` 가 없으면 **조용히 0 으로 채우지 않고 예외를 던진다** —
        학습 때 본 입력과 추론 입력이 어긋나면 성능이 조용히 떨어진다.
        """
        single = c.dim() == 2
        if single:
            g = g.reshape(1, -1)
            c = c.unsqueeze(0)
            if h is not None and h.dim() == 2:
                h = h.unsqueeze(0)
        if self.hist is not None and h is None:
            raise ValueError(
                "이 모델은 과거 궤적(h)을 받도록 학습됐습니다 "
                f"(history_encoder={self.spec['history_encoder']}). "
                "forward(g, c, h) 로 호출하십시오."
            )
        mask = candidate_mask(c)                       # 정규화 전에 판정
        cv = self.constant_velocity(g)                 # 원 눈금에서 만든다
        gn, cn = self._norm(g, c)
        cn = cn * mask.unsqueeze(-1)
        # 후보 은닉은 `ch` 다 — `h` 는 **과거 궤적 인자**이므로 이름을 겹치면
        # 이력 인코더에 후보 텐서가 들어간다 (실제로 그렇게 터졌다).
        ch = self.cand_enc(cn)                         # [B, N, H]
        a = torch.softmax(
            self.attn(ch).squeeze(-1).masked_fill(~mask, MASK_NEG), -1)
        pooled = torch.cat([
            (ch * a.unsqueeze(-1)).sum(1),                     # 어텐션 가중합
            (ch * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1, keepdim=True).clamp(min=1),           # 평균
            ch.masked_fill(~mask.unsqueeze(-1), MASK_NEG).max(1).values,  # 최대
        ], dim=-1)
        parts = [gn, pooled]
        if self.hist is not None:
            # 이력 벡터와 함께 **이력이 있었는지**를 1칸으로 준다. 0 벡터만 주면
            # 모델이 "이력 없음"과 "이력이 전부 0(정지)"을 구별할 수 없다.
            parts.append(self.hist(h))
            parts.append((h[..., -1] > 0.5).any(dim=1, keepdim=True).float())
        z = torch.cat(parts, dim=-1)
        # `trunk` 를 **한 번만** 부른다. 두 번 부르면 학습 모드에서 dropout 이 서로
        # 다르게 걸려 두 헤드가 다른 표현을 보게 된다 (실측: 같은 입력에 대한 두
        # 호출의 최대 차이 0.25). 그러면 "같은 표현을 공유한다"는 설계 의도가
        # 학습 중에 — 즉 정작 중요한 때에 — 성립하지 않는다.
        t = self.trunk(z)
        out = self.wp_head(t).reshape(-1, self.n_steps, 2)
        if self.residual:
            out = out + cv
        if self.man_head is None:
            return out.squeeze(0) if single else out
        logits = self.man_head(t)
        if single:
            return out.squeeze(0), logits.squeeze(0)
        return out, logits

    def constant_velocity(self, g: torch.Tensor) -> torch.Tensor:
        """속도·방위로 만든 등속 직진 궤적 [B, K, 2].

        이것을 기준선으로 깔고 보정만 학습한다. 5초 예측의 상당 부분이 "직진 유지"
        이므로, 그것을 처음부터 맞히고 시작하는 것이 훨씬 쉽다.

        전체 시험셋에서 등속 ADE 는 4.13m, 규칙 기반은 4.01m 다 — **등속이 규칙보다
        약간 나쁘다.** 그래도 기준선으로 쓸 값어치가 있는 것은 학습 전 출력이 도로
        위 어딘가에서 시작하게 만들고(무작위 초기화는 수십 m 를 벗어난다), 학습이
        실패해도 그 근처로 떨어지기 때문이다. 최종 모델은 3.06m 로 둘 다 넘는다.
        """
        v = g[:, IDX_SPEED]
        step = torch.stack([v * g[:, IDX_HEADING_SIN],
                            v * g[:, IDX_HEADING_COS]], dim=-1)   # [B,2]
        ks = torch.arange(1, self.n_steps + 1, device=g.device,
                          dtype=step.dtype).reshape(1, -1, 1)
        return step.unsqueeze(1) * ks


class InteractionWaypointNet(WaypointNet):
    """WaypointNet에 주변 actor attention을 더한 경로 회귀 모델.

    target과 주변 actor의 미래를 순서대로 예측하지 않는다. 동일 시점의 관측 상태만
    받아 target별 미래 경로를 한 번에 조건부로 회귀하므로 actor 처리 순서가 결과를
    바꾸지 않는다. 관측 관계도 규칙으로 제동을 강제하지 않고 feature로만 제공한다.
    """

    def __init__(self, *args, interaction_hidden: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.spec["hidden"]
        self.accepts_interactions = True
        self.interaction_hidden = interaction_hidden
        self.register_buffer("i_mu", torch.zeros(N_INTERACTION_FEATURES))
        self.register_buffer("i_sd", torch.ones(N_INTERACTION_FEATURES))
        self.interaction_enc = nn.Sequential(
            nn.Linear(N_INTERACTION_FEATURES, interaction_hidden), nn.ReLU(),
            nn.Linear(interaction_hidden, interaction_hidden), nn.ReLU(),
        )
        self.interaction_query = nn.Linear(self.spec["n_global"], interaction_hidden)
        # WaypointNet trunk is replaced; the rest (history, candidate pool,
        # residual output and maneuver head) stays exactly the same.
        hist_dim = 0 if self.hist is None else self.hist.out_dim + 1
        self.trunk = nn.Sequential(
            nn.Linear(self.spec["n_global"] + hidden * 3 + hist_dim
                      + interaction_hidden * 3 + 1, hidden), nn.ReLU(),
            nn.Dropout(self.spec["dropout"]),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(self.spec["dropout"]),
        )
        self.spec.update({
            "interaction": True,
            "n_interaction_features": N_INTERACTION_FEATURES,
            "interaction_hidden": interaction_hidden,
        })

    @torch.no_grad()
    def fit_normalizer(self, g, c, mask, interaction=None, interaction_mask=None):
        super().fit_normalizer(g, c, mask)
        if interaction is None or interaction_mask is None:
            raise ValueError("InteractionWaypointNet 학습에는 interaction tensor와 mask가 필요합니다")
        flat = interaction[interaction_mask]
        if flat.numel():
            self.i_mu.copy_(flat.mean(0))
            # 이웃이 한 개뿐인 소형 pilot에서도 unbiased std는 NaN이 된다.
            self.i_sd.copy_(flat.std(0, unbiased=False).clamp(min=1e-3))

    def forward(self, g, c, h=None, interaction=None, interaction_mask=None):
        single = c.dim() == 2
        if single:
            g = g.reshape(1, -1)
            c = c.unsqueeze(0)
            if h is not None and h.dim() == 2:
                h = h.unsqueeze(0)
            if interaction is not None and interaction.dim() == 2:
                interaction = interaction.unsqueeze(0)
        if self.hist is not None and h is None:
            raise ValueError("이 모델은 과거 궤적(h)을 받도록 학습됐습니다")
        if interaction is None:
            raise ValueError("InteractionWaypointNet은 주변 actor interaction 입력이 필요합니다")
        if interaction.numel() == 0:
            interaction = interaction.reshape(g.shape[0], 0, N_INTERACTION_FEATURES)
        if interaction.dim() != 3 or interaction.shape[-1] != N_INTERACTION_FEATURES:
            raise ValueError(
                f"interaction shape must be [B,N,{N_INTERACTION_FEATURES}], got {tuple(interaction.shape)}"
            )
        if interaction_mask is None:
            interaction_mask = interaction[..., -1] > 0.5
        # max/softmax pooling needs one physical row even when this snapshot has
        # no other actors. It remains masked, so the interaction summary is 0.
        if interaction.shape[1] == 0:
            interaction = torch.zeros(
                interaction.shape[0], 1, N_INTERACTION_FEATURES,
                dtype=interaction.dtype, device=interaction.device,
            )
            interaction_mask = torch.zeros(
                interaction.shape[0], 1, dtype=torch.bool,
                device=interaction.device,
            )
        if interaction_mask.dim() == 1:
            interaction_mask = interaction_mask.unsqueeze(0)
        mask = candidate_mask(c)
        cv = self.constant_velocity(g)
        gn, cn = self._norm(g, c)
        cn = cn * mask.unsqueeze(-1)
        ch = self.cand_enc(cn)
        ca = torch.softmax(self.attn(ch).squeeze(-1).masked_fill(~mask, MASK_NEG), -1)
        candidate_pool = torch.cat([
            (ch * ca.unsqueeze(-1)).sum(1),
            (ch * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1),
            ch.masked_fill(~mask.unsqueeze(-1), MASK_NEG).max(1).values,
        ], dim=-1)

        inn = (interaction - self.i_mu) / self.i_sd
        inn = inn * interaction_mask.unsqueeze(-1)
        ih = self.interaction_enc(inn)
        query = self.interaction_query(gn).unsqueeze(1)
        scores = (ih * query).sum(-1) / (self.interaction_hidden ** 0.5)
        ia = torch.softmax(scores.masked_fill(~interaction_mask, MASK_NEG), -1)
        imax = ih.masked_fill(~interaction_mask.unsqueeze(-1), MASK_NEG).max(1).values
        imax = torch.where(interaction_mask.any(1, keepdim=True), imax,
                           torch.zeros_like(imax))
        interaction_pool = torch.cat([
            (ih * ia.unsqueeze(-1)).sum(1),
            (ih * interaction_mask.unsqueeze(-1)).sum(1)
            / interaction_mask.sum(1, keepdim=True).clamp(min=1),
            imax,
            interaction_mask.any(1, keepdim=True).float(),
        ], dim=-1)
        parts = [gn, candidate_pool]
        if self.hist is not None:
            parts.append(self.hist(h))
            parts.append((h[..., -1] > 0.5).any(dim=1, keepdim=True).float())
        parts.append(interaction_pool)
        t = self.trunk(torch.cat(parts, dim=-1))
        out = self.wp_head(t).reshape(-1, self.n_steps, 2)
        if self.residual:
            out = out + cv
        if self.man_head is None:
            return out.squeeze(0) if single else out
        logits = self.man_head(t)
        return (out.squeeze(0), logits.squeeze(0)) if single else (out, logits)


def masked_scores(
    model: RankNet, g: torch.Tensor, c: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """패딩 자리를 softmax 에서 배제한 점수. `mask` 는 유효 후보 위치가 True.

    `-inf` 대신 큰 음수(`MASK_NEG`)를 쓴다 — 전부 패딩인 행이 생기면 `-inf` 는
    NaN 을 만든다.
    """
    return model(g, c).masked_fill(~mask, MASK_NEG)
