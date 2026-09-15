"""`dump_prompts.py` 의 JSON 에서 `reports/04-prompts.md` 에 넣을 조각을 뽑는다.

긴 조건(G·H·J 는 17,000~19,000자)은 전문을 싣지 않는다. 대신 **조건 사이의 차이가
보이는 지점**만 꺼낸다 — 기하 표, 장면 설명 앞부분, 판정 라벨이 지워진 줄의 전후,
이미지 안내문 두 판.

조각은 `/tmp/prompt_parts.json` 에 나온다. 문서의 설명 문장은 사람이 쓰되,
```text 블록 안의 내용은 **반드시 여기서 나온 것을 그대로** 써야 한다 — 손으로
옮겨 적으면 실제로 보낸 것과 어긋난다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "04-prompts.md"
SCENE_LIMIT = 2800


def section(text: str, header: str, upto: str | None = None,
            limit: int | None = None) -> str:
    i = text.find(header)
    if i < 0:
        return ""
    j = text.find(upto, i + len(header)) if upto else -1
    end = j if j > 0 else len(text)
    s = text[i:end]
    if limit and len(s) > limit:
        s = s[:limit].rstrip() + f"\n\n  … (이하 {len(s) - limit:,}자 생략)"
    return s.rstrip()


def find_line(text: str, pattern: str) -> str:
    for ln in text.split("\n"):
        if re.search(pattern, ln):
            return ln.strip()
    return ""


def main() -> int:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    C = d["conditions"]
    E = C["E_geometry_only"]

    parts = {
        "gap": section(E, "COMPUTED PROXIMITY", "COLLISION TAXONOMY"),
        "taxonomy": section(E, "COLLISION TAXONOMY", "TASK"),
        "task": section(E, "TASK"),
        "icv": section(C["K_iitp_cv"], "COMPUTED PROXIMITY", "COLLISION TAXONOMY"),
        "iwp": section(C["K_iitp_predicted"], "COMPUTED PROXIMITY", "COLLISION TAXONOMY"),
        "scene": section(C["G_iitp_scene"], "MAP-GROUNDED SCENE", "COLLISION TAXONOMY",
                         limit=SCENE_LIMIT),
        "img_f": section(C["F_geometry_plus_bev"], "You are also given",
                         "COLLISION TAXONOMY"),
        "img_l": section(C["L_iitp_pred_bev"], "You are also given",
                         "COLLISION TAXONOMY"),
        "g_actor": find_line(C["G_iitp_scene"], r"^- M008\(")[:190],
        "h_actor": find_line(C["H_iitp_numeric"], r"^- M008\(")[:190],
        "g_int": find_line(C["G_iitp_scene"], r"orthogonal entry"),
        "h_int": find_line(C["H_iitp_numeric"], r"arrival time gap"),
        "scene_full_len": len(section(C["G_iitp_scene"], "MAP-GROUNDED SCENE",
                                      "COLLISION TAXONOMY")),
    }
    sizes = {k: len(v) for k, v in C.items()}
    print(f"조건 {len(C)}개 · 표 {len(parts['gap']):,}자 · "
          f"브리핑 {parts['scene_full_len']:,}자")
    print(f"→ {OUT} 는 이 조각들로 조립한다. 본문 틀은 문서를 직접 편집하되,")
    print("   ```text 블록 안의 내용은 반드시 여기서 나온 것을 쓸 것.")
    json.dump({"sizes": sizes, "parts": parts, "uid": d["uid"], "gt": d["gt"]},
              open("/tmp/prompt_parts.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("   조각: /tmp/prompt_parts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
