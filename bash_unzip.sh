RUN_ID=$(date +%Y%m%d_%H%M%S)
BASE=/home/lgz/dataset/carla_length
NPROC=12
LOG=/tmp/unzip_carla_length_parallel_${RUN_ID}.log
RES=/tmp/unzip_carla_length_parallel_${RUN_ID}.results.tsv
NOHUP_OUT=/tmp/unzip_carla_length_parallel_${RUN_ID}.nohup.out

nohup bash -s "$BASE" "$LOG" "$RES" "$NPROC" >"$NOHUP_OUT" 2>&1 <<'BASH' &
set -euo pipefail

BASE="$1"
LOG="$2"
RES="$3"
NPROC="$4"

: > "$LOG"
: > "$RES"

echo "START $(date '+%F %T')" | tee -a "$LOG"
echo "BASE=$BASE NPROC=$NPROC" | tee -a "$LOG"

process_one() {
  local z="$1"
  local d="$2"
  local split="$3"
  local res="$4"
  local name mode dest status

  name="$(basename "$z" .zip)"

  mode="$(zipinfo -1 "$z" | awk -F/ '
    BEGIN { top=""; multi=0; root=0; seen=0 }
    {
      line=$0
      sub(/^\.\//, "", line)
      if (line=="" || line ~ /^__MACOSX\//) next
      seen=1
      n=split(line, a, "/")
      if (n==1 && line !~ /\/$/) root=1
      t=a[1]
      if (top=="") top=t
      else if (t!=top) multi=1
    }
    END {
      if (seen==1 && multi==0 && root==0) print "keep"
      else print "wrap"
    }
  ')"

  if [[ "$mode" == "keep" ]]; then
    dest="$d"
  else
    dest="$d/$name"
    mkdir -p "$dest"
  fi

  if unzip -oq "$z" -d "$dest"; then
    status="ok"
  else
    status="fail"
  fi

  printf "%s\t%s\t%s\t%s\n" "$split" "$name" "$mode" "$status" >> "$res"
}

process_split() {
  local split="$1"
  local d="$BASE/$split"
  [[ -d "$d" ]] || return 0

  echo "== SPLIT $split ==" | tee -a "$LOG"
  mapfile -d '' zips < <(find "$d" -maxdepth 1 -type f -name '*.zip' -print0 | sort -z)
  echo "ZIPS ${#zips[@]}" | tee -a "$LOG"

  local running=0
  for z in "${zips[@]}"; do
    process_one "$z" "$d" "$split" "$RES" &
    ((running+=1))
    if (( running >= NPROC )); then
      wait -n || true
      ((running-=1))
    fi
  done
  wait || true
}

process_split val
# process_split train


echo "DONE $(date '+%F %T')" | tee -a "$LOG"
awk -F '\t' '
  BEGIN {total=0;wrap=0;keep=0;ok=0;fail=0}
  {total++; if($3=="wrap") wrap++; if($3=="keep") keep++; if($4=="ok") ok++; if($4=="fail") fail++}
  END {printf("TOTAL=%d WRAP=%d KEEP_PARENT=%d OK=%d FAIL=%d\n", total, wrap, keep, ok, fail)}
' "$RES" | tee -a "$LOG"
BASH

echo "PID=$!"
echo "LOG=$LOG"
echo "RES=$RES"
echo "NOHUP_OUT=$NOHUP_OUT"