#!/usr/bin/env bash
#
# Mancini bot deploy script — laptop → VM.
#
# Usage:
#   deploy/deploy.sh            # interactive, asks before destructive steps
#   deploy/deploy.sh --yes      # skip confirmations (for known-good runs)
#   deploy/deploy.sh --dry-run  # show what would happen, do nothing
#
# Pre-conditions:
#   - You're on a clean main with the PR merged
#   - You have ssh access via ~/.ssh/oracle_bullmachine
#   - There are no open positions (script checks this — hard gate)
#
# What it does (5 phases, each gated):
#   1. Pre-flight: verify git state + open-position check on VM
#   2. Rsync source dirs (NEVER touches data/, logs/, .env)
#   3. Verify code landed via grep checks on VM
#   4. docker compose build --no-cache mancini-bot
#   5. docker compose up -d --no-deps mancini-bot, then smoke + config verify
#
# Rollback hint at the end if anything fails.

set -Eeuo pipefail

# ---------- config ----------
VM_USER="ubuntu"
VM_HOST="152.70.113.24"
VM_KEY="$HOME/.ssh/oracle_bullmachine"
VM_PATH="/home/ubuntu/mancini"
# Compose v2 dash naming — the old underscore name made post-deploy
# verification report failure after a successful recreate (2026-06-09).
CONTAINER="mancini-mancini-bot-1"
EXPECTED_BRANCH="main"

SSH_OPTS=(-i "$VM_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ---------- helpers ----------
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

step() { echo; bold "=== $* ==="; }

confirm() {
  local prompt="$1"
  if [[ $ASSUME_YES -eq 1 ]]; then
    yellow "[--yes] auto-confirming: $prompt"
    return 0
  fi
  read -r -p "$prompt [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

ssh_vm() { ssh "${SSH_OPTS[@]}" "$VM_USER@$VM_HOST" "$@"; }

run_or_dry() {
  if [[ $DRY_RUN -eq 1 ]]; then
    yellow "[dry-run] $*"
  else
    # "$@" not eval: eval re-splits quoted args, so rsync's -e "ssh -i <key>"
    # became -e ssh + stray args and rsync connected without the key
    # (publickey denied, deploy aborted 2026-06-09).
    "$@"
  fi
}

# Trap to show rollback hint on failure
on_err() {
  echo
  red "❌ Deploy aborted."
  yellow "Rollback hint:"
  cat <<EOF
  ssh -i $VM_KEY $VM_USER@$VM_HOST
  cd $VM_PATH
  docker images mancini-bot --format '{{.ID}} {{.CreatedAt}}'  # find prior image
  # docker tag <prior-image-id> mancini-bot:latest
  # docker compose up -d --no-deps mancini-bot
EOF
}
trap on_err ERR

# ---------- phase 1: pre-flight ----------
step "1/5 Pre-flight"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$EXPECTED_BRANCH" ]]; then
  red "Current branch is '$current_branch', expected '$EXPECTED_BRANCH'."
  yellow "Run 'git checkout $EXPECTED_BRANCH && git pull' first."
  exit 1
fi

if ! git diff-index --quiet HEAD --; then
  red "Working tree has uncommitted changes."
  git status --short
  exit 1
fi

if ! git diff --quiet @{u}..HEAD 2>/dev/null && ! git diff --quiet HEAD..@{u} 2>/dev/null; then
  yellow "Local $EXPECTED_BRANCH differs from origin. Run 'git pull' first?"
  exit 1
fi

green "✓ On clean $EXPECTED_BRANCH, in sync with origin"
git log --oneline -3

echo
yellow "Checking for open positions on VM…"

# HARD gate: the bot writes a full position snapshot every bar. If it shows
# an open position, STOP — no --yes bypass. (2026-07-20: a --yes deploy
# restarted the bot mid-trade; recovery dropped fields and the runner was
# mis-trailed. Mechanical check, not a log-grep + human shrug.)
snap_open="$(ssh_vm "python3 -c \"
import json
try:
    d = json.load(open('$VM_PATH/data/position_snapshot.json'))
    print('OPEN' if d.get('position') else 'FLAT')
except Exception:
    print('UNKNOWN')
\"" || echo UNKNOWN)"
if [[ "$snap_open" == "OPEN" ]]; then
  red "❌ Position snapshot shows an OPEN position. Deploy refused (no --yes override)."
  yellow "Wait for the position to close, or flatten first."
  exit 1
fi
yellow "Snapshot position check: $snap_open"
position_log="$(ssh_vm "docker logs --tail 800 $CONTAINER 2>&1 | grep -E 'ENTRY|FILLED|STOP_FILLED|T1_FILLED|T2_FILLED|RUNNER|flatten' | tail -25" || true)"

if [[ -z "$position_log" ]]; then
  yellow "(no recent ENTRY/FILL events in last 800 log lines — likely no active position)"
else
  echo "$position_log"
  echo
  yellow "Review the last lines above. A bare ENTRY without a matching STOP_FILLED/T2_FILLED/flatten means a position is OPEN."
  if ! confirm "Are you sure NO position is open?"; then
    red "Aborting. Wait for the position to close, or flatten manually first."
    exit 1
  fi
fi

# Trading-window safety check
et_now="$(TZ=America/New_York date +'%H:%M %A')"
et_dow="$(TZ=America/New_York date +'%u')"  # 1=Mon … 7=Sun
et_h="$(TZ=America/New_York date +'%H')"
in_safe_window=0
# Fri 17:00+ … Sun 18:00 = no Globex
if [[ ($et_dow == "5" && $et_h -ge 17) || $et_dow == "6" || ($et_dow == "7" && $et_h -lt 18) ]]; then
  in_safe_window=1
fi
# Daily break 17:00-17:59 ET
if [[ $et_h == "17" ]]; then
  in_safe_window=1
fi

yellow "Now in ET: $et_now"
if [[ $in_safe_window -eq 1 ]]; then
  green "✓ In safe deploy window (weekend or daily break)"
else
  red "⚠ Markets are OPEN. Recreating the container will drop the IB connection mid-session."
  if ! confirm "Proceed anyway?"; then
    exit 1
  fi
fi

# ---------- phase 2: rsync code ----------
step "2/5 Sync code to VM"

RSYNC_DIRS=(config core strategy backtest live tests Dockerfile docker-compose.yml)
yellow "Files being synced: ${RSYNC_DIRS[*]}"
yellow "NEVER synced: data/, logs/, .env, *.parquet, __pycache__/, .venv*/"

if confirm "Run rsync now?"; then
  run_or_dry rsync -avz --delete \
    -e "ssh ${SSH_OPTS[*]}" \
    --exclude='__pycache__' --exclude='.venv*' --exclude='.pytest_cache' \
    --exclude='data/' --exclude='logs/' --exclude='.env' \
    --exclude='*.parquet' --exclude='.git/' --exclude='*.pyc' \
    "${RSYNC_DIRS[@]}" \
    "$VM_USER@$VM_HOST:$VM_PATH/"
  green "✓ rsync complete"
else
  exit 1
fi

# ---------- phase 3: verify code landed ----------
step "3/5 Verify code landed on VM"

verify_out="$(ssh_vm "
  cd $VM_PATH
  echo '--- eod_flatten_enabled ---'
  grep -n 'eod_flatten_enabled' config/settings.py | head -3 || echo MISSING
  echo '--- _compute_globex_trading_date ---'
  grep -n '_compute_globex_trading_date' live/ib_runner.py | head -3 || echo MISSING
  echo '--- structure_trail_enabled ---'
  grep -n 'structure_trail_enabled' config/settings.py | head -2 || echo MISSING
  echo '--- multi_session_runner ---'
  grep -n 'multi_session_runner' config/settings.py | head -2 || echo MISSING
")"
echo "$verify_out"

if echo "$verify_out" | grep -q "MISSING"; then
  red "One or more expected markers missing — rsync did not deliver the new code."
  exit 1
fi
green "✓ All four markers present on VM"

# ---------- phase 4: docker build ----------
step "4/5 Rebuild image (docker compose build --no-cache mancini-bot)"

if confirm "Proceed with --no-cache rebuild (~3-5 min)?"; then
  if [[ $DRY_RUN -eq 1 ]]; then
    yellow "[dry-run] would build"
  else
    ssh_vm "cd $VM_PATH && docker compose build --no-cache mancini-bot"
  fi
  green "✓ Build complete"
else
  exit 1
fi

# ---------- phase 5: recreate container + verify ----------
step "5/5 Recreate bot container"

if confirm "Recreate mancini-bot container now? (drops IB connection briefly)"; then
  if [[ $DRY_RUN -eq 1 ]]; then
    yellow "[dry-run] would recreate"
  else
    ssh_vm "cd $VM_PATH && docker compose up -d --no-deps mancini-bot"
    sleep 6
    status="$(ssh_vm "docker ps --filter name=$CONTAINER --format '{{.Status}}'")"
    echo "Container status: $status"
    if [[ ! "$status" =~ ^Up ]]; then
      red "Container is not running."
      ssh_vm "docker logs --tail 80 $CONTAINER"
      exit 1
    fi
  fi
  green "✓ Container running"
else
  exit 1
fi

# Verify runtime config matches expected new defaults
yellow "Verifying runtime config…"
config_out="$(ssh_vm "docker exec $CONTAINER python3 -c '
from live.ib_runner import PRODUCTION_EXIT
print(f\"eod_flatten_enabled={PRODUCTION_EXIT.eod_flatten_enabled}\")
print(f\"multi_session_runner={PRODUCTION_EXIT.multi_session_runner}\")
print(f\"structure_trail_enabled={PRODUCTION_EXIT.structure_trail_enabled}\")
print(f\"splits={PRODUCTION_EXIT.t1_exit_fraction}/{PRODUCTION_EXIT.t2_exit_fraction}/{PRODUCTION_EXIT.runner_fraction}\")
'")"
echo "$config_out"

fail=0
echo "$config_out" | grep -q "eod_flatten_enabled=False" || { red "✗ eod_flatten_enabled != False"; fail=1; }
echo "$config_out" | grep -q "multi_session_runner=True" || { red "✗ multi_session_runner != True"; fail=1; }
echo "$config_out" | grep -q "structure_trail_enabled=True" || { red "✗ structure_trail_enabled != True"; fail=1; }
echo "$config_out" | grep -q "splits=0.75/0.15/0.1" || { red "✗ splits != 0.75/0.15/0.1"; fail=1; }
[[ $fail -eq 1 ]] && exit 1
green "✓ Runtime config matches expected new defaults"

# 60-second log tail so caller can spot tracebacks
echo
yellow "Streaming first 60s of logs (Ctrl-C anytime)…"
if [[ $DRY_RUN -eq 0 ]]; then
  timeout 60 ssh "${SSH_OPTS[@]}" "$VM_USER@$VM_HOST" "docker logs -f --tail 30 $CONTAINER" || true
fi

trap - ERR
echo
green "✅ Deploy complete."
yellow "Watch for the next Mancini plan load to confirm Globex-aware date:"
echo "  ssh -i $VM_KEY $VM_USER@$VM_HOST 'docker logs -f $CONTAINER | grep -i plan'"
