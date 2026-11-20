#!/usr/bin/env bash
# Runs on the GPU machine after bootstrap: one job, results to the Hugging Face
# dataset under runs/<RUN_ID>/, then the machine tries to delete itself (the
# launcher deletes it too - this is the backstop).
#
# Env: RUN_ID, JOB, JOB_ARGS, HF_REPO, HF_TOKEN, MAX_MINUTES, RUNPOD_POD_ID
set -uo pipefail
OUT=/workspace/out
mkdir -p "$OUT"
note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/log.txt"; }

note "job $JOB (${JOB_ARGS:-no args}), limit ${MAX_MINUTES} min"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1 || true
note "gpu: $(cat "$OUT/gpu.txt")"
start=$(date +%s)
# shellcheck disable=SC2086
timeout "${MAX_MINUTES}m" bash "scripts/cloud/jobs/${JOB}.sh" "$OUT" ${JOB_ARGS:-} >> "$OUT/log.txt" 2>&1
code=$?
[ "$code" = 124 ] && note "job hit the ${MAX_MINUTES} min limit"
note "job finished with exit code $code after $(( $(date +%s) - start )) s"
python3 - "$code" "$(( $(date +%s) - start ))" <<'PY'
import json, os, sys
gpu = open("/workspace/out/gpu.txt").read().strip()
json.dump({"exit_code": int(sys.argv[1]), "seconds": int(sys.argv[2]), "gpu": gpu,
           "self_delete_key_present": bool(os.environ.get("RUNPOD_API_KEY"))},
          open("/workspace/out/DONE.json", "w"))
PY
python3 - <<'PY' || note "upload to Hugging Face failed"
import os
from huggingface_hub import upload_folder
upload_folder(repo_id=os.environ["HF_REPO"], repo_type="dataset", folder_path="/workspace/out",
              path_in_repo=f"runs/{os.environ['RUN_ID']}", commit_message=f"run {os.environ['RUN_ID']} ({os.environ['JOB']})")
PY
note "results uploaded to $HF_REPO/runs/$RUN_ID"

# Backstop: delete this machine. RunPod may or may not give the machine a key
# for this; the gpu_check job reports which.
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    curl -s -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" > /dev/null || true
fi
sleep 3600  # if still alive, the launcher (or the next launcher run) deletes it
