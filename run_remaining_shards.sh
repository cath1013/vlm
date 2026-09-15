#!/bin/bash
# 나머지 샤드(16~127) 생성 스크립트
# 16개씩 병렬 실행 (32코어 중 절반 사용하여 시스템 여유 확보)
set -e

PYTHON=".venv/bin/python"
SCRIPT="examples/make_predict_dataset.py"
ROOT="/home/sryu/inclab-nas/DeepAccident"
CARLA_MAPS="./carla_map"
OUT="out/predict_dataset_interaction_v1"
BATCH_SIZE=16
TOTAL_SHARDS=128
START_SHARD=16

echo "=== InteractionWaypointNet 데이터셋 생성 (샤드 $START_SHARD~$((TOTAL_SHARDS-1)) / $TOTAL_SHARDS) ==="
echo "병렬 배치 크기: $BATCH_SIZE"
echo ""

completed=0
failed=0

for batch_start in $(seq $START_SHARD $BATCH_SIZE $((TOTAL_SHARDS-1))); do
    batch_end=$((batch_start + BATCH_SIZE - 1))
    if [ $batch_end -ge $TOTAL_SHARDS ]; then
        batch_end=$((TOTAL_SHARDS - 1))
    fi
    
    echo "[$(date '+%H:%M:%S')] 배치 시작: 샤드 $batch_start ~ $batch_end"
    
    pids=()
    for i in $(seq $batch_start $batch_end); do
        $PYTHON $SCRIPT \
            --root "$ROOT" \
            --carla-maps "$CARLA_MAPS" \
            --out "$OUT" \
            --shard "$i/$TOTAL_SHARDS" \
            > /dev/null 2>&1 &
        pids+=($!)
    done
    
    # 배치 내 모든 프로세스 완료 대기
    batch_failed=0
    for pid in "${pids[@]}"; do
        if wait $pid; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
            batch_failed=$((batch_failed + 1))
        fi
    done
    
    done_shards=$((batch_end - START_SHARD + 1))
    total_remaining=$((TOTAL_SHARDS - START_SHARD))
    echo "[$(date '+%H:%M:%S')] 배치 완료 ($done_shards/$total_remaining 샤드, 실패: $batch_failed)"
done

echo ""
echo "=== 완료 ==="
echo "성공: $completed, 실패: $failed"
echo ""

# 전체 샤드가 있는지 확인
existing=$(ls "$OUT"/manifest.*of128.json 2>/dev/null | wc -l)
echo "총 매니페스트 파일 수: $existing / $TOTAL_SHARDS"

if [ "$existing" -eq "$TOTAL_SHARDS" ]; then
    echo ""
    echo "모든 샤드 완료. 병합 시작..."
    $PYTHON $SCRIPT --merge --out "$OUT"
else
    echo ""
    echo "경고: $((TOTAL_SHARDS - existing))개 샤드 누락. 확인 후 수동 병합 필요:"
    echo "  $PYTHON $SCRIPT --merge --out $OUT"
fi
