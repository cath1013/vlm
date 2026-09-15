"""미팅용 표를 xlsx 로 만든다 — 조건·payload 내용·결과를 한 파일에. 한/영 두 판.

payload 발췌는 **손으로 옮기지 않고** `dump_prompts.py` 가 코드 경로에서 뽑은
JSON 에서 가져온다. 결과 수치도 저장된 sweep 파일에서 다시 채점해 넣는다.

    ~/vlm/.venv-vlm/bin/python reports/_gen/dump_prompts.py > /tmp/prompts.json
    .venv/bin/python reports/_gen/make_xlsx.py /tmp/prompts.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
SWEEP = Path("/home/sryu/vlm/outputs/eval/bev_sweep/gemini-2.5-flash")

HEAD = PatternFill("solid", fgColor="1F3B57")
HEADF = Font(color="FFFFFF", bold=True, size=10)
HL = PatternFill("solid", fgColor="FFF2CC")
DIM = Font(color="808080", size=9)
MONO = Font(name="Consolas", size=9)
THIN = Side(style="thin", color="D0D0D0")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")

# 저장된 조건 이름 ↔ 프롬프트 dump 의 키. 프로덕션 두 줄은 표를 그 자리에서
# 다시 계산하므로 길이를 미리 뽑을 수 없다.
LEN_KEY = {"D_reports_plus_geometry": "D_reports_plus_geom",
           "L_iitp_predicted_bev": "L_iitp_pred_bev",
           "G_geometry_plus_iitp_scene": "G_iitp_scene",
           "H_geometry_plus_iitp_numeric": "H_iitp_numeric",
           "J_geometry_plus_iitp_numeric_plus_bev": "J_iitp_numeric_bev"}


def sec(t, head, upto=None, limit=None):
    i = t.find(head)
    if i < 0:
        return ""
    j = t.find(upto, i + len(head)) if upto else -1
    s = t[i: j if j > 0 else len(t)].rstrip()
    if limit and len(s) > limit:
        s = s[:limit].rstrip() + f"\n… ({len(s) - limit:,} more chars)"
    return s


def score_all() -> dict:
    """저장된 sweep 을 다시 채점한다. 빈 답 비율도 함께 센다 — 차종쌍 정답률이
    낮은 것이 '틀렸다' 인지 '답을 안 했다' 인지 구분해야 하기 때문이다."""
    spec = importlib.util.spec_from_file_location(
        "_scoring", "/home/sryu/vlm/src/deepaccident_vlm/scoring.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_scoring"] = mod          # dataclass 가 등록을 요구한다
    spec.loader.exec_module(mod)

    recs = [json.loads(f.read_text()) for f in sorted(SWEEP.glob("*.json"))]
    out = {}
    for c in {k for r in recs for k in r["predictions"]}:
        rs = [{"gt": r["gt"], "pred": r["predictions"][c], "parse_ok": True}
              for r in recs if c in r["predictions"]]
        if len(rs) < 50:
            continue
        s = mod.score(rs)
        b, cc = s["collision_detection"], s["conditional_on_both_collision"]
        said = [r["pred"] for r in rs if r["pred"].get("collision")]
        blank = sum(1 for p in said if not (p.get("involved") or []))
        out[c] = {**b, "n_scenarios": s["n"], "cls": cc.get("class_pair_exact"),
                  "cls_n": cc.get("n"),
                  "blank": blank / len(said) if said else None}
    return out


def style_header(ws) -> None:
    for c in ws[1]:
        if c.value is not None:
            c.fill, c.font, c.alignment, c.border = HEAD, HEADF, WRAP, BOX


# (저장 키, 한국어 이름, 영어 이름, 한국어 설명, 영어 설명,
#  기하표, 그림, 관측자글, iitp설명, 표 만든 방식, 강조)
ROWS = [
 ("E_geometry_only",
  "기하 표만", "Geometry table only",
  "차량 쌍마다 '지금 간격'·'앞으로 1초 안에 가장 가까워지는 간격'·'그게 몇 초 뒤인지' "
  "세 숫자를 적은 표 하나. 그 표 말고는 아무것도 주지 않는다.\n\n"
  "표를 만든 방법 — LiDAR 점을 덩어리로 묶어 차를 세고, 각 차가 지금 속도로 계속 "
  "직진한다고 보고 0.05초마다 밀면서 차체(회전 사각형) 사이 간격을 잰다.\n\n"
  "차 이름은 1, 2, 3 … 익명 번호라 그게 승용차인지 트럭인지 알 수 없다.",
  "One table. For every pair of road users: the gap right now, the smallest gap they "
  "reach within the next 1.0 s, and when that happens. Nothing else is given.\n\n"
  "How the table is built — LiDAR points are clustered into objects, each is pushed "
  "forward assuming it holds its current speed and heading, and the gap between the "
  "rotated rectangles of their bodies is measured every 0.05 s.\n\n"
  "Objects are named 1, 2, 3 … The model cannot tell a car from a truck.",
  "vlm", "", "", "", "등속|constant velocity", False),

 ("F_geometry_plus_bev",
  "기하 표 + 조감도 그림", "Geometry table + bird's-eye image",
  "위의 표에 그림 1장을 더한 것.\n\n"
  "그림은 다섯 관측자의 LiDAR 를 합쳐 위에서 내려다본 것이다. 색이 시간을 뜻한다 — "
  "파랑 1초 전, 주황 0.5초 전, 빨강 지금. 움직이는 차는 파랑-주황-빨강 줄무늬로 보인다.\n\n"
  "그림에 덩어리 번호가 적혀 있어서, 표의 '2번'이 어느 차인지 그림에서 찾아볼 수 있다.",
  "The table above plus one image.\n\n"
  "The image fuses the LiDAR of all five platforms into a single top-down view. Colour "
  "is time — blue 1.0 s ago, amber 0.5 s ago, red now — so a moving vehicle appears as "
  "a blue-amber-red streak.\n\n"
  "Cluster numbers are drawn on the image, so the model can look up which object the "
  "table's \"2\" refers to.",
  "vlm", "O", "", "", "등속|constant velocity", False),

 ("D_reports_plus_geometry",
  "관측자가 쓴 글 + 기하 표", "Observer write-ups + geometry table",
  "위의 표에 관측자가 쓴 글을 더한 것.\n\n"
  "자차와 노변 센서가 각각 자기가 본 것을 LLM 으로 글로 쓴 것이다. "
  "\"1번과 2번이 수렴 중, 위험 높음\" 같은 판정이 들어 있다.\n\n"
  "차 번호가 관측자마다 따로 매겨져서, 한쪽의 2번과 다른 쪽의 2번이 같은 차가 아니다.",
  "The table above plus prose written by the sensor platforms.\n\n"
  "The ego vehicle and the roadside unit each described what they saw, written by an "
  "LLM. It carries verdicts — \"1 and 2 converging, risk high\".\n\n"
  "Object numbers are local to each platform: one platform's 2 is not the other's 2.",
  "vlm", "", "O", "", "등속|constant velocity", False),

 ("P_production",
  "프로덕션 구성 (이전 최고)", "Production shape (previous best)",
  "기하 표 + 조감도 그림. 관측자가 쓴 글은 주지 않는다.\n\n"
  "위의 '기하 표 + 조감도 그림'과 한 가지가 다르다 — 표를 만들 때 검출기가 찾은 덩어리를 "
  "걸러내지 않고 전부 쓴다. 원래는 관측자가 LLM 으로 '이게 진짜 차 맞나' 확인해 걸렀는데, "
  "재보니 그 필터가 오히려 손해였다.\n\n"
  "관측자 단계를 통째로 없애서 호출이 시나리오당 11회 → 1회, 비용이 $5.86 → $0.46 이 됐다.",
  "Geometry table plus the bird's-eye image. No observer prose.\n\n"
  "One thing differs from \"table + image\" above — the table is computed over every "
  "cluster the detector produced, rather than the subset an observer call kept. "
  "Measured, that filter cost more than it saved.\n\n"
  "Dropping the observer stage took the run from 11 calls per scenario to 1, and the "
  "cost from $5.86 to $0.46.",
  "vlm(all)", "O", "", "", "등속|constant velocity", False),

 ("P_production_qwen32b",
  "프로덕션 구성 · 모델만 Qwen 으로", "Production shape · Qwen instead of Gemini",
  "바로 위와 완전히 같은 것을 gemini-2.5-flash 대신 Qwen3-VL-32B 에 보냈다.\n\n"
  "103개 중 92개를 '충돌'이라 답했다. 오탐이 11 → 43 으로 늘었다.",
  "Exactly the same input as the row above, sent to Qwen3-VL-32B instead of "
  "gemini-2.5-flash.\n\n"
  "It answered \"collision\" on 92 of 103 scenarios. False positives went 11 → 43.",
  "vlm(all)", "O", "", "", "등속|constant velocity", False),

 ("G_geometry_plus_iitp_scene",
  "기하 표 + iitp 가 쓴 장면 설명 (전체)", "Geometry table + traffic_llm scene write-up (full)",
  "맨 위의 기하 표(vlm, 등속)에 traffic_llm 이 만든 글 설명 전부를 더한 것.\n\n"
  "설명에 들어 있는 것 — 차마다 어느 도로 몇 차선인지, 다음 교차로까지 몇 m 인지, "
  "제한속도, 현재 속도, 가속도, 앞으로 갈 것으로 본 경로. 그리고 차량 쌍마다 충돌까지 "
  "남은 시간(TTC), 차간 시간(headway), 교차로 도달 시간차.\n\n"
  "여기에 '차선 유지', '정지', '선행차 추종', '직교 진입 상충 예상' 같은 판정 성격의 "
  "표현도 함께 들어 있다. 전부 사람이 아니라 코드가 계산한 값이다.",
  "The first table (vlm, constant velocity) plus everything traffic_llm writes about "
  "the scene.\n\n"
  "What it contains — for each vehicle: which road and lane, metres to the next "
  "junction, posted limit, current speed, acceleration, and the path it is predicted "
  "to take. For each pair: time-to-collision, headway, and the difference in junction "
  "arrival times.\n\n"
  "It also contains verdict-like phrases — \"keeping lane\", \"stopped\", \"car "
  "following\", \"orthogonal entry conflict expected\". All of it is computed by "
  "deterministic code, not written by a model.",
  "vlm", "", "", "전체|full", "등속|constant velocity", False),

 ("H_geometry_plus_iitp_numeric",
  "기하 표 + iitp 장면 설명 (판정 표현 삭제)",
  "Geometry table + traffic_llm scene write-up (verdict phrases removed)",
  "바로 위와 딱 한 가지만 다르다 — 판정 성격의 표현을 지웠다.\n\n"
  "지운 것: 예상 경로(2,753) · 차선 유지(1,495) · 정지(1,099) · 불확실하다는 훈수(914) · "
  "선행차 추종(518) · 직교 진입 상충 예상(119) · 가속 중(111) · 감속 중(33) · 좌회전(9) · "
  "우회전(6). 10종 7,057개.\n\n"
  "숫자는 97.8% 그대로 남겼다 — 도로·차선 번호, 거리, 속도, TTC, 헤드웨이 전부.",
  "Exactly one thing differs from the row above — the verdict-like phrases are stripped.\n\n"
  "Removed: predicted path (2,753) · keeping lane (1,495) · stopped (1,099) · "
  "uncertainty notes (914) · car following (518) · orthogonal entry conflict expected "
  "(119) · accelerating (111) · decelerating (33) · turning left (9) · turning right "
  "(6). Ten kinds, 7,057 occurrences.\n\n"
  "97.8% of the numbers survive — road and lane numbers, distances, speeds, TTC, headway.",
  "vlm", "", "", "수치만|numbers only", "등속|constant velocity", False),

 ("J_geometry_plus_iitp_numeric_plus_bev",
  "기하 표 + iitp 장면 설명(판정 삭제) + 조감도 그림",
  "Geometry table + scene write-up (stripped) + bird's-eye image",
  "바로 위(판정 표현을 지운 설명)에 조감도 그림 1장을 더한 것.\n\n"
  "설명이 없을 때는 그림이 차종 응답률을 크게 올린다. 설명이 있으면 거의 못 올린다.",
  "The row above plus one bird's-eye image.\n\n"
  "Without the write-up the image lifts vehicle-class answers sharply. With the "
  "write-up present, the same image barely moves them.",
  "vlm", "O", "", "수치만|numbers only", "등속|constant velocity", False),

 ("K_iitp_cv",
  "기하 표를 iitp 가 만든 것 (등속)",
  "Geometry table rebuilt by traffic_llm (constant velocity)",
  "표만 준다. 글 설명도 그림도 없다. 맨 위 조건과 표를 만든 재료가 다르다.\n\n"
  "vlm 은 LiDAR 점을 덩어리로 묶어 차를 세는데, traffic_llm 은 다섯 관측자가 본 것을 "
  "합쳐 차 목록을 확정한다. 앞으로 미는 방식은 등속으로 똑같다.\n\n"
  "차 이름이 종류를 담는다 — V=승용차, M=오토바이, N=승합차, T=트럭, P=사람, "
  "EGO_*=센서를 단 차량 자신. 다만 프롬프트에 그 규칙을 알려주지는 않는다.",
  "The table only — no write-up, no image. What differs from the first row is the "
  "material the table is built from.\n\n"
  "vlm clusters LiDAR points into objects; traffic_llm fuses what all five platforms "
  "saw into one confirmed list of road users. The way each is pushed forward is the "
  "same: constant velocity.\n\n"
  "Object names carry the class — V=car, M=motorcycle, N=van, T=truck, P=pedestrian, "
  "EGO_* = a sensor platform itself. The prompt never states this convention.",
  "iitp", "", "", "", "등속|constant velocity", False),

 ("K_iitp_predicted",
  "기하 표를 iitp 가 만든 것 (WaypointNet)",
  "Geometry table rebuilt by traffic_llm (WaypointNet)",
  "바로 위와 차 목록·좌표·시점이 전부 같고, 앞으로 미는 방식만 다르다.\n\n"
  "등속은 '지금 속도로 계속 직진한다'고 치는 것이고, 여기서는 학습된 신경망 WaypointNet 이 "
  "찍은 궤적을 따라간다. 이 프로젝트에서 직접 만들어 15만 표본으로 학습시킨 것이다 "
  "(DeepAccident 논문의 모델이 아니다).\n\n"
  "표 형식과 길이는 위와 같다. 값만 달라진다.",
  "Same road users, same coordinates, same moment as the row above. Only the way they "
  "are pushed forward differs.\n\n"
  "Constant velocity assumes each vehicle holds its current speed and heading. Here the "
  "trajectory comes from WaypointNet, a network built for this project and trained on "
  "150k samples (it is not the model from the DeepAccident paper).\n\n"
  "The table's format and length are identical. Only the numbers change.",
  "iitp", "", "", "", "WaypointNet", True),

 ("L_iitp_predicted_bev",
  "기하 표를 iitp 가 만든 것 (WaypointNet) + 조감도 그림",
  "Geometry table rebuilt by traffic_llm (WaypointNet) + bird's-eye image",
  "바로 위(WaypointNet 으로 만든 표)에 조감도 그림 1장을 더한 것.\n\n"
  "표는 '충돌이 일어나는가'를, 그림은 '누가 부딪히는가'를 맡는다. 둘을 같이 준 유일한 조건이다.",
  "The row above plus one bird's-eye image.\n\n"
  "The table drives whether a collision happens; the image drives what is involved. "
  "This is the only condition that has both.",
  "iitp", "O", "", "", "WaypointNet", True),
]

COLS = {
 "ko": ["조건", "모델이 받은 것 — 풀어 쓰면", "기하 표", "조감도\n그림", "관측자\n글",
        "iitp\n장면설명", "표를 만든\n방식", "보낸 글\n길이(자)", "정확도", "정밀도",
        "재현율", "F1", "TP", "FP", "TN", "FN", "차종쌍\n정답률", "차종 답을\n비운 비율",
        "저장된 이름"],
 "en": ["Condition", "What the model received", "Geometry\ntable", "Bird's-eye\nimage",
        "Observer\nprose", "traffic_llm\nwrite-up", "How the table\nwas built",
        "Prompt\nlength (chars)", "Accuracy", "Precision", "Recall", "F1",
        "TP", "FP", "TN", "FN", "Class-pair\nexact", "Answers left\nempty",
        "Stored name"],
}
NOTES = {
 "ko": ["모두 gemini-2.5-flash · DeepAccident val 103 시나리오 (사고 51 / 무사고 52) "
        "· 형식 오류 0건 · 조건당 약 $0.5",
        "'차종쌍 정답률' 은 모델과 정답이 둘 다 '사고' 라 한 건에 대해서만 계산한다 "
        "(조건마다 n 이 31~46 으로 작다)",
        "'차종 답을 비운 비율' 을 같이 봐야 한다 — 표만 준 조건은 39건 중 38건이 빈 배열이라, "
        "정답률 0.032 는 '틀렸다' 가 아니라 '답을 안 했다' 는 뜻이다",
        "정답의 63% 가 car+car 라, car+car 로 찍기만 해도 0.63 이 나온다. 이 지표는 정보량이 약하다"],
 "en": ["All gemini-2.5-flash · DeepAccident val, 103 scenarios (51 collision / 52 not) "
        "· 0 format failures · about $0.5 per condition",
        "'Class-pair exact' is computed only where both the model and the ground truth "
        "say collision (n is 31-46 depending on the condition)",
        "Read it together with 'answers left empty' — with the table alone the model "
        "left 38 of 39 blank, so 0.032 means it did not answer, not that it answered wrong",
        "63% of the ground-truth pairs are car+car, so always guessing car+car already "
        "scores 0.63. This metric is weakly informative"],
}
HORIZON = {
 "ko": (["구간", "시점", "구간 수", "등속 (val)", "카메라로 위치 추정", "WaypointNet", "판정"],
        [("k=1", "0~1초", 710, 0.813, 0.575, 0.871, "풀린다"),
         ("k=2", "1~2초", 575, 0.622, 0.543, 0.617, "약하다"),
         ("k=3", "2~3초", 471, 0.536, 0.521, 0.564, "거의 우연"),
         ("k=4", "3~4초", 367, 0.472, 0.508, 0.503, "우연 이하"),
         ("k=5", "4~5초", 271, 0.423, 0.479, 0.419, "우연 이하"),
         ("전체", "", 2394, 0.616, 0.534, 0.643, "")],
        ["숫자는 AUC — 값 하나로 정답을 얼마나 가르는지. 0.5 면 정보 없음",
         "모델도 학습도 없는 순수 기하 계산이므로, 여기서 안 갈리면 어떤 방법으로도 안 갈린다",
         "train 120 시나리오에서 따로 재도 같다 — k=1 0.806 · k=3 0.556 · k=5 0.460"]),
 "en": (["Bucket", "Window", "Buckets", "Constant velocity (val)",
         "Position from camera", "WaypointNet", "Verdict"],
        [("k=1", "0-1 s", 710, 0.813, 0.575, 0.871, "solvable"),
         ("k=2", "1-2 s", 575, 0.622, 0.543, 0.617, "weak"),
         ("k=3", "2-3 s", 471, 0.536, 0.521, 0.564, "near chance"),
         ("k=4", "3-4 s", 367, 0.472, 0.508, 0.503, "below chance"),
         ("k=5", "4-5 s", 271, 0.423, 0.479, 0.419, "below chance"),
         ("all", "", 2394, 0.616, 0.534, 0.643, "")],
        ["Numbers are AUC — how well one value separates the classes. 0.5 = no information",
         "This is pure geometry, no model and no learning. If it does not separate here, "
         "nothing will",
         "Measured separately on 120 train scenarios: k=1 0.806 · k=3 0.556 · k=5 0.460"]),
}


def build(lang: str, d: dict, S: dict) -> Path:
    C = d["conditions"]
    ko = lang == "ko"
    wb = Workbook()

    ws = wb.active
    ws.title = "조건 비교" if ko else "Conditions"
    ws.append(COLS[lang])
    for key, n_ko, n_en, d_ko, d_en, gap, img, rep, scn, mode, hl in ROWS:
        v = S.get(key, {})
        m_ko, m_en = mode.split("|") if "|" in mode else (mode, mode)
        ws.append([n_ko if ko else n_en, d_ko if ko else d_en, gap, img, rep,
                   (scn.split("|")[0] if ko else scn.split("|")[-1]) if scn else "",
                   m_ko if ko else m_en,
                   len(C.get(LEN_KEY.get(key, key), "")) or None,
                   v.get("accuracy"), v.get("precision"), v.get("recall"), v.get("f1"),
                   v.get("tp"), v.get("fp"), v.get("tn"), v.get("fn"),
                   v.get("cls"), v.get("blank"), key])
        if hl:
            for c in ws[ws.max_row]:
                c.fill = HL
    style_header(ws)
    for r in ws.iter_rows(min_row=2, max_row=1 + len(ROWS)):
        for c in r:
            c.border = BOX
            c.alignment = (WRAP if c.column == 2 else
                           Alignment(vertical="center",
                                     horizontal="center" if c.column > 2 else "left"))
            if c.column in (9, 10, 11, 12, 17, 18) and isinstance(c.value, float):
                c.number_format = "0.000"
        r[-1].font, r[-1].alignment = DIM, Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[r[0].row].height = 122
    for i, w in enumerate([40, 78, 11, 11, 11, 12, 15, 12, 10, 10, 9, 8,
                           6, 6, 6, 6, 11, 12, 36], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "C2"
    ws.append([])
    for n in NOTES[lang]:
        ws.append([n])
        ws.cell(ws.max_row, 1).font = DIM

    # payload 내용
    ws2 = wb.create_sheet("payload 내용" if ko else "What was in the prompt")
    E, K = C["E_geometry_only"], C["K_iitp_predicted"]
    P = [
     ("system", "모든 조건 공통. 누구로서 답하라는 지시",
      "Shared by every condition — who the model is answering as",
      d["system"].strip()),
     ("기하 표 (vlm)" if ko else "Geometry table (vlm)",
      "쌍마다 세 숫자. gap now = 지금 간격. min gap = 앞으로 1초 동안 가장 가까워지는 "
      "간격(1초 시점이 아니라 0~1초 전체의 최솟값이다). at t+ = 그 최솟값이 몇 초 뒤인지.\n\n"
      "at t+ 가 정보를 담는다 — 0.0 이면 지금이 제일 가깝다는 뜻이라 멀어지는 중이고"
      "(전체 쌍의 54%), 1.0 이면 끝까지 계속 가까워지는 중이며(42%), 그 사이면 스쳐 지나간다(4%).\n\n"
      "중심이 아니라 차체(회전 사각형) 사이 거리다. 0.00m 면 닿는다. id 는 LiDAR 덩어리에 "
      "붙인 익명 번호라 어떤 차인지 알 수 없다.",
      "Three numbers per pair. 'gap now' is the separation right now. 'min gap' is the "
      "smallest they reach within the next 1.0 s — the minimum over the whole window, "
      "not the value at t=1.0. 'at t+' is when that minimum occurs.\n\n"
      "'at t+' carries information — 0.0 means they are closest right now, i.e. moving "
      "apart (54% of pairs); 1.0 means they keep closing to the end of the window (42%); "
      "anything between means they pass each other (4%).\n\n"
      "Distances are between vehicle bodies (rotated rectangles), not centres. 0.00 m "
      "means touching. Ids are anonymous cluster numbers.",
      sec(E, "COMPUTED PROXIMITY", "COLLISION TAXONOMY")),
     ("기하 표 (iitp · WaypointNet)" if ko else "Geometry table (traffic_llm · WaypointNet)",
      "같은 형식인데 (1) 차량 목록을 iitp 가 다섯 관측자를 합쳐 확정하고 (2) 앞으로 미는 "
      "방식이 학습된 궤적이다. id 접두어가 종류를 담는다 (V=승용차, M=오토바이, N=승합차, "
      "T=트럭, P=사람).",
      "Same format, but (1) the road-user list is fused by traffic_llm across all five "
      "platforms and (2) each is pushed along a learned trajectory instead of a straight "
      "line. The id prefix encodes the class (V=car, M=motorcycle, N=van, T=truck, "
      "P=pedestrian).",
      sec(K, "COMPUTED PROXIMITY", "COLLISION TAXONOMY")),
     ("관측자가 쓴 글" if ko else "Observer prose",
      "각 센서 플랫폼이 LLM 으로 쓴 글. 이것을 주면 정확도가 떨어진다",
      "Prose each sensor platform wrote with an LLM. Giving it lowers accuracy",
      sec(C["D_reports_plus_geom"], "REPORT FROM", "COMPUTED GEOMETRY", limit=1200)),
     ("iitp 장면 설명" if ko else "traffic_llm scene write-up",
      "도로·차선, 교차로까지 거리, 제한속도, 가속도, 예상 경로, 차량 쌍마다 충돌까지 남은 "
      "시간(TTC)·차간 시간(headway)",
      "Road and lane, distance to the junction, posted limit, acceleration, predicted "
      "path, and per pair: time-to-collision and headway",
      sec(C["G_iitp_scene"], "MAP-GROUNDED SCENE", "COLLISION TAXONOMY", limit=2500)),
     ("조감도 그림 안내" if ko else "Note that accompanies the image",
      "그림은 글이 아니라 첨부로 간다. 색이 시간을 뜻한다 (파랑 1초 전 → 주황 0.5초 전 → 빨강 지금)",
      "The image is attached, not written into the text. Colour is time "
      "(blue 1.0 s ago → amber 0.5 s ago → red now)",
      sec(C["L_iitp_pred_bev"], "You are also given", "COLLISION TAXONOMY")),
     ("충돌 분류 정의" if ko else "Collision taxonomy",
      "답할 때 쓸 말을 정해 준다. 없으면 제멋대로 표현해서 채점이 안 된다",
      "Fixes the vocabulary of the answer. Without it the model invents wording and "
      "nothing can be scored",
      sec(E, "COLLISION TAXONOMY", "TASK")),
     ("질문과 답 형식" if ko else "The question and the answer format",
      "아홉 조건이 전부 같다", "Identical across all nine conditions", sec(E, "TASK")),
    ]
    ws2.append(["구성 요소", "무엇인가", "실제로 보낸 내용 (payload 에서 그대로 추출)"] if ko
               else ["Part", "What it is", "What was actually sent (pulled from the payload)"])
    for a, b_ko, b_en, c in P:
        ws2.append([a, b_ko if ko else b_en, c])
    style_header(ws2)
    for r in ws2.iter_rows(min_row=2):
        r[0].font = Font(bold=True, size=10)
        for c in r:
            c.alignment, c.border = WRAP, BOX
        r[2].font = MONO
        ws2.row_dimensions[r[0].row].height = 210
    for i, w in enumerate([26, 46, 108], 1):
        ws2.column_dimensions[get_column_letter(i)].width = w

    # 구간별 한계
    ws3 = wb.create_sheet("구간별 한계" if ko else "How far ahead is solvable")
    cols3, rows3, notes3 = HORIZON[lang]
    ws3.append(cols3)
    for row in rows3:
        ws3.append(list(row))
    style_header(ws3)
    for r in ws3.iter_rows(min_row=2, max_row=1 + len(rows3)):
        for c in r:
            c.border, c.alignment = BOX, Alignment(horizontal="center")
            if isinstance(c.value, float):
                c.number_format = "0.000"
        if r[0].value == "k=1":
            for c in r:
                c.fill = HL
    for i, w in enumerate([9, 11, 10, 22, 22, 15, 14], 1):
        ws3.column_dimensions[get_column_letter(i)].width = w
    ws3.append([])
    for n in notes3:
        ws3.append([n])
        ws3.cell(ws3.max_row, 1).font = DIM

    out = ROOT / ("조건별_결과.xlsx" if ko else "conditions_and_results.xlsx")
    wb.save(out)
    return out


def main() -> int:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    S = score_all()
    for lang in ("ko", "en"):
        print(f"저장: {build(lang, d, S)}")
    print(f"  시트 3개 · 조건 {len(ROWS)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
