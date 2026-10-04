#!/bin/bash
# pe-tune-v2 hardened bootstrap — paste into the pod console.
# FAILS FAST: any error halts immediately (no silent GPU burn).
# TIMEBOXED: whole job dies at 9h (~$4.41) no matter what.
# SECRET: set HF_TOKEN in this console first (RunPod dashboard > pod > env
#   is better). It never leaves the pod. Do NOT paste it into any chat.
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN not set in pod. export HF_TOKEN=... first. Stopping.}"
TIMEBOX=${TIMEBOX:-32400}  # 9h default; override per run

trap 'echo "BOOTSTRAP-FAILED at line $LINENO. Run: runpodctl stop pod $POD_ID (or stop from dashboard). No further billing after stop." >&2' ERR

pip -q install transformers datasets soundfile torchcodec timm huggingface_hub
git clone https://github.com/shadyuwugurl/pe-single-L14 && cd pe-single-L14

timeout "$TIMEBOX" python3 pe_single_github/tune_pe.py \
  --model MC7ever/pe-single-L14 \
  --out ./pe-single-L14-tuned-v2 --n-coco-train 4000 --n-esc-train 1600 \
  --audio-every 4 --epochs 2 --batch 32 --accum 1 --lr 5e-6 \
  --device cuda --amp --ckpt-every 50 2>&1 | tee tune_v2.log
test -f pe-single-L14-tuned-v2/model.safetensors || { echo "NO WEIGHTS PRODUCED. Stopping."; exit 1; }

python3 eval_pe_bench.py --models ./pe-single-L14-tuned-v2 \
  --n-coco 150 --n-esc 200 --device cuda --dtype fp32 --out /tmp/v2.json
tail -n 3 /tmp/v2.json

python3 - ./pe-single-L14-tuned-v2 <<'EOF'
import sys, time
from huggingface_hub import upload_folder
from huggingface_hub.utils import HfHubHTTPError
folder, repo = sys.argv[1], "MC7ever/pe-single-L14"
for attempt in range(5):
    try:
        print(upload_folder(folder_path=folder, repo_id=repo,
              commit_message="v4: audio-text tune v2 (RunPod A40)"))
        break
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0)
        if code in (401, 403):
            print(f"AUTH {code}: bad HF_TOKEN. Fix token, rerun upload only. Stopping."); break
        if code == 429:
            wait = int(e.response.headers.get("Retry-After", 30))
            print(f"429: backing off {wait}s"); time.sleep(wait); continue
        if 500 <= code < 600:
            print(f"{code}: backing off 60s"); time.sleep(60); continue
        raise
EOF
echo "DONE. Stop the pod now (dashboard stop, or ask your agent to terminate it) to end billing."
