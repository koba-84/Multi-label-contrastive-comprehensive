# wait_then_run_fixed.sh
#!/usr/bin/env bash
set -euo pipefail

############################################
# ここだけ編集：終了後に実行したいコマンドを並べる
# （1つでも複数でもOK。順番に実行されます）
COMMANDS=(
  'source .venv/bin/activate'
  'cd src'
  'uv run main.py --config config/config.json'
)
############################################

log(){ echo "[$(date '+%F %T')] $*"; }

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <PID>"
  exit 1
fi

PID="$1"

# 監視開始
if ! ps -p "$PID" >/dev/null 2>&1; then
  log "PID $PID not found（既に終了か無効）→すぐ実行に移行します。"
else
  log "PID $PID の終了待ちを開始..."
  while kill -0 "$PID" >/dev/null 2>&1; do
    sleep 300
  done
  log "PID $PID が終了。"
fi

# 固定コマンド群を順番に実行
overall_rc=0
for ((i=0; i<${#COMMANDS[@]}; i++)); do
  cmd="${COMMANDS[$i]}"
  log "[$((i+1))/${#COMMANDS[@]}] START: $cmd"
  bash -lc "$cmd"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    log "[$((i+1))/${#COMMANDS[@]}] FAIL (rc=$rc)"
    overall_rc=$rc
    # 失敗で止めたい場合は次の行のコメントを外す
    # exit $rc
  else
    log "[$((i+1))/${#COMMANDS[@]}] DONE (rc=$rc)"
  fi
done

log "ALL DONE (last rc=$overall_rc)"
exit $overall_rc
