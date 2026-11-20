#!/usr/bin/env bash
# First thing a GPU machine runs: get the code at the requested commit, then
# hand over to the job script that lives in the repository (so job logic can
# change without rebuilding the image).
#
# Env (set by the launcher): REPO, GIT_REF, RUN_ID, HF_REPO, GITHUB_TOKEN, HF_TOKEN
set -uo pipefail
mkdir -p /workspace/out
note() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a /workspace/out/log.txt; }
fail() {  # report the failure where the launcher looks for results, then stop
    note "BOOTSTRAP FAILED: $1"
    echo "{\"exit_code\": 99, \"error\": \"$1\"}" > /workspace/out/DONE.json
    python3 - <<'PY' || true
import os
from huggingface_hub import upload_folder
upload_folder(repo_id=os.environ["HF_REPO"], repo_type="dataset", folder_path="/workspace/out",
              path_in_repo=f"runs/{os.environ['RUN_ID']}", commit_message=f"run {os.environ['RUN_ID']} bootstrap failed")
PY
    sleep 600; exit 1
}
note "bootstrap: run $RUN_ID, $REPO @ $GIT_REF"
git init -q /workspace/code && cd /workspace/code || fail "git init"
git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git"
git fetch -q --depth 1 origin "$GIT_REF" || fail "could not fetch $REPO@$GIT_REF (check the github_token secret)"
git checkout -q FETCH_HEAD || fail "checkout"
git remote remove origin  # don't keep the token in the checkout
exec bash scripts/cloud/job_entry.sh
