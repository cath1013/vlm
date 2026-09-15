"""`reports/04-prompts.md` 를 실제 코드 경로에서 다시 생성한다.

프롬프트를 손으로 옮겨 적으면 실제로 보낸 것과 어긋나고, 그 어긋남은 결과를
재해석할 때 드러나지 않는다. 그래서 문서를 코드에서 뽑는다.

    ~/vlm/.venv-vlm/bin/python reports/_gen/dump_prompts.py > /tmp/prompts.json
    .venv/bin/python reports/_gen/render_prompts.py /tmp/prompts.json

주의 — 앞 단계는 **vlm 의 venv** 로, 뒤 단계는 iitp 의 venv 로 돈다.
두 환경은 JSON 파일로만 주고받는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VLM = Path("/home/sryu/vlm")
IITP = Path("/home/sryu/inclab-nas/sryu/iitp/out/vlm_scene")
SWEEP = VLM / "outputs/eval/bev_sweep/gemini-2.5-flash"

sys.path.insert(0, str(VLM / "src"))
from deepaccident_vlm.bev_prompts import CENTRAL_SYSTEM, central_prompt  # noqa: E402
from deepaccident_vlm.demo import format_gaps  # noqa: E402


def main() -> int:
    full = json.loads((IITP / "val_h1.0_en.json").read_text())
    num = json.loads((IITP / "val_h1.0_en_numeric.json").read_text())
    gapt = json.loads((IITP / "gaps_val_h1.0.json").read_text())

    f = sorted(SWEEP.glob("*.json"))[0]
    d = json.loads(f.read_text())
    uid = d["uid"]
    vlm_gaps = format_gaps(d["observers"]["fused"]["computed_gaps"])
    icv = format_gaps([tuple(r) for r in gapt[uid]["cv"]])
    iwp = format_gaps([tuple(r) for r in gapt[uid]["predicted"]])
    rep = {k: d["observers"][k]["report"] for k in ("ego_vehicle", "infrastructure")}

    json.dump({
        "uid": uid, "gt": d["gt"], "system": CENTRAL_SYSTEM,
        "conditions": {
            "E_geometry_only": central_prompt({}, 1.0, gaps=vlm_gaps),
            "F_geometry_plus_bev": central_prompt({}, 1.0, gaps=vlm_gaps, with_bev=True),
            "D_reports_plus_geom": central_prompt(rep, 1.0, gaps=vlm_gaps),
            "G_iitp_scene": central_prompt({}, 1.0, gaps=vlm_gaps, scene=full[uid]["text"]),
            "H_iitp_numeric": central_prompt({}, 1.0, gaps=vlm_gaps, scene=num[uid]["text"]),
            "J_iitp_numeric_bev": central_prompt({}, 1.0, gaps=vlm_gaps,
                                                 scene=num[uid]["text"], with_bev=True),
            "K_iitp_cv": central_prompt({}, 1.0, gaps=icv),
            "K_iitp_predicted": central_prompt({}, 1.0, gaps=iwp),
            "L_iitp_pred_bev": central_prompt({}, 1.0, gaps=iwp, with_bev=True),
        },
    }, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
