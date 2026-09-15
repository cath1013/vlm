# 이 프로젝트 실행용 환경변수.
#
# **키를 여기 적지 마십시오.** 이 폴더는 NAS(권한 777)라 접근 가능한 누구나 읽습니다.
# 키는 홈 디렉터리(로컬 디스크, 600 권한)에 두고 여기서는 읽어오기만 합니다.
#
#     source env.sh
#     .venv/bin/python examples/ask_llm.py ...
#
# 키를 새로 넣거나 바꾸려면 ~/.config/iitp/secrets.env 를 편집하십시오.

for f in "$HOME/.config/iitp/secrets.env" "$HOME/vlm/.env"; do
  if [ -f "$f" ]; then
    set -a; . "$f"; set +a
  fi
done

export DEEPACCIDENT_ROOT="${DEEPACCIDENT_ROOT:-/home/sryu/inclab-nas/DeepAccident}"
export CARLA_MAP_DIR="${CARLA_MAP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/carla_map}"

for k in GEMINI_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY; do
  v="${!k}"
  if [ -n "$v" ]; then printf '  %-20s 설정됨 (%d자)\n' "$k" "${#v}"; fi
done
