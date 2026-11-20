#!/usr/bin/env python
"""Run one job on a rented RunPod GPU and bring the results back.

    python scripts/cloud/runpod_launch.py run gpu_check --gpus debug
    python scripts/cloud/runpod_launch.py run unpile --gpus batch --args "--seed 3" --max-minutes 60
    python scripts/cloud/runpod_launch.py status      # list our machines
    python scripts/cloud/runpod_launch.py cleanup     # delete all our machines

What `run` does:
  1. deletes leftover laundry-bot machines that outlived their time limit
  2. starts a machine with the first available GPU type from the chosen list
  3. the machine fetches the code at your current commit (must be pushed),
     runs scripts/cloud/jobs/<job>.sh, uploads results to the Hugging Face
     dataset under runs/<run id>/
  4. waits for those results, downloads them to output/cloud/<run id>/
  5. ALWAYS deletes the machine - on success, failure, timeout or Ctrl-C

Keys: RunPod and Hugging Face keys are injected by this session's network
proxy, so no key is ever in a file or variable here. Elsewhere, set
RUNPOD_API_KEY and HF_TOKEN. On RunPod, the machines read two secrets you
create in RunPod's settings: `github_token` (read this repo) and
`laundry-bot-runs-token` (write the results dataset).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://rest.runpod.io/v1"
IMAGE = "ghcr.io/shachar-roth/laundry-bot-sim:latest"
REPO = "shachar-roth/laundry-bot"
HF_REPO = "Shachar-roth/laundry-bot-runs"
NAME_PREFIX = "laundry-bot-"

# GPU lists, in order of preference (RunPod's exact names). All are NVIDIA with
# ray-tracing cores and >= 16 GB memory.
GPU_LISTS = {
    "debug": ["NVIDIA RTX A4000", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX A4500", "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX A5000"],
    "batch": ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"],
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ RunPod API
def api(method: str, path: str, body: dict | None = None) -> object:
    key = os.environ.get("RUNPOD_API_KEY", "proxy-injected")
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RunPod {method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}") from None


def our_pods() -> list[dict]:
    return [p for p in api("GET", "/pods") or [] if str(p.get("name", "")).startswith(NAME_PREFIX)]


def delete_pod(pod_id: str) -> None:
    for attempt in range(5):
        try:
            api("DELETE", f"/pods/{pod_id}")
            log(f"machine {pod_id} deleted")
            return
        except RuntimeError as e:
            if "404" in str(e):
                log(f"machine {pod_id} already gone")
                return
            log(f"delete failed ({e}); retrying")
            time.sleep(5 * (attempt + 1))
    log(f"!!! could not delete machine {pod_id} - delete it in the RunPod console")


def reap_stale(grace_minutes: int = 20) -> None:
    """Delete our machines that outlived their own time limit (e.g. a launcher
    that was interrupted)."""
    now = datetime.now(timezone.utc)
    for p in our_pods():
        started = p.get("lastStartedAt") or p.get("lastStatusChange")
        limit = int((p.get("env") or {}).get("MAX_MINUTES", 60))
        try:
            age = (now - datetime.fromisoformat(str(started).replace("Z", "+00:00"))).total_seconds() / 60
        except ValueError:
            age = None
        if age is None or age > limit + grace_minutes:
            log(f"leftover machine {p['id']} ({p.get('name')}, age {age if age is None else round(age)} min) - deleting")
            delete_pod(p["id"])


# ------------------------------------------------------------- Hugging Face
def hf_files(run_id: str) -> list[str]:
    key = os.environ.get("HF_TOKEN", "proxy-injected")
    url = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/runs/{run_id}?recursive=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return [f["path"] for f in json.loads(r.read()) if f.get("type") == "file"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def hf_download(run_id: str, dest: Path) -> list[Path]:
    os.environ.setdefault("HF_TOKEN", "proxy-injected")
    from huggingface_hub import snapshot_download

    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=[f"runs/{run_id}/*", f"runs/{run_id}/**"], local_dir=dest.parent / "_hf")
    src = dest.parent / "_hf" / "runs" / run_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    src.rename(dest)
    return sorted(p for p in dest.rglob("*") if p.is_file())


# --------------------------------------------------------------------- run
def current_commit() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        log("warning: you have uncommitted changes - the machine runs the last pushed commit, not them")
    pushed = subprocess.run(["git", "branch", "-r", "--contains", sha], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not pushed:
        sys.exit(f"commit {sha[:8]} is not pushed to GitHub - push it first (the machine downloads it from there)")
    return sha


def run(args) -> int:
    sha = current_commit()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{args.job}"
    gpus = GPU_LISTS.get(args.gpus) or [g.strip() for g in args.gpus.split(",")]
    reap_stale()

    body = {
        "name": NAME_PREFIX + run_id,
        "imageName": args.image,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": gpus,
        "gpuTypePriority": "custom",  # try them in our order (cheapest suitable first)
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "containerDiskInGb": 40,
        "volumeInGb": 0,  # no persistent storage: results go to Hugging Face
        "interruptible": args.spot,
        "env": {
            "RUN_ID": run_id,
            "JOB": args.job,
            "JOB_ARGS": args.args,
            "MAX_MINUTES": str(args.max_minutes),
            "REPO": REPO,
            "GIT_REF": sha,
            "HF_REPO": HF_REPO,
            # RunPod fills these in from the secrets you saved in its settings
            "GITHUB_TOKEN": "{{ RUNPOD_SECRET_github_token }}",
            "HF_TOKEN": "{{ RUNPOD_SECRET_laundry-bot-runs-token }}",
        },
    }
    if args.registry_auth:
        body["containerRegistryAuthId"] = args.registry_auth

    pod = None
    for cloud in (["SECURE", "COMMUNITY"] if args.cloud == "any" else [args.cloud.upper()]):
        try:
            pod = api("POST", "/pods", {**body, "cloudType": cloud})
            break
        except RuntimeError as e:
            log(f"{cloud.lower()} cloud: {e}")
    if not pod:
        log("no machine available with those GPUs right now - try another --gpus list, or later")
        return 2
    pod_id = pod["id"]
    gpu = (pod.get("machine") or {}).get("gpuTypeId") or (pod.get("gpu") or {}).get("id") or "?"
    log(f"machine {pod_id} starting: {gpu}, ${pod.get('costPerHr', '?')}/hour, run {run_id} @ {sha[:8]}")

    t0 = time.time()
    deadline = t0 + 60 * (args.max_minutes + 15)  # + time to download the image and start
    status = "timeout"
    try:
        last_state = None
        while time.time() < deadline:
            time.sleep(20)
            try:
                p = api("GET", f"/pods/{pod_id}")
            except RuntimeError as e:
                log(f"status check failed: {e}")
                p = {}
            state = p.get("desiredStatus")
            if state != last_state:
                log(f"machine state: {state}")
                last_state = state
            if state in ("TERMINATED", "EXITED") and not hf_files(run_id):
                status = f"machine {state.lower()} before results arrived"
                break
            if f"runs/{run_id}/DONE.json" in hf_files(run_id):
                status = "done"
                break
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        cost_per_hr = float(pod.get("costPerHr") or 0)
        delete_pod(pod_id)
        minutes = (time.time() - t0) / 60
        log(f"{status} after {minutes:.1f} min, machine cost about ${cost_per_hr * minutes / 60:.2f}")

    if hf_files(run_id):
        dest = ROOT / "output" / "cloud" / run_id
        files = hf_download(run_id, dest)
        log(f"results in {dest.relative_to(ROOT)}: {', '.join(str(f.relative_to(dest)) for f in files)}")
        done = dest / "DONE.json"
        if done.exists():
            log("DONE.json: " + done.read_text().strip())
    return 0 if status == "done" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a job on a GPU machine")
    r.add_argument("job", help="name of a script in scripts/cloud/jobs/ (without .sh)")
    r.add_argument("--gpus", default="debug", help=f"one of {list(GPU_LISTS)} or a comma-separated list of RunPod GPU names")
    r.add_argument("--args", default="", help="arguments passed to the job script")
    r.add_argument("--max-minutes", type=int, default=30)
    r.add_argument("--cloud", default="any", choices=["any", "secure", "community"])
    r.add_argument("--spot", action="store_true", help="interruptible machine: cheaper, may be stopped at any time")
    r.add_argument("--image", default=IMAGE)
    r.add_argument("--registry-auth", default="", help="RunPod registry-credential id, if the image is private")
    sub.add_parser("status", help="list our machines")
    sub.add_parser("cleanup", help="delete all our machines")
    args = ap.parse_args()

    if args.cmd == "status":
        pods = our_pods()
        for p in pods:
            print(p["id"], p.get("name"), p.get("desiredStatus"), f"${p.get('costPerHr')}/h")
        print(f"{len(pods)} laundry-bot machine(s)")
        return 0
    if args.cmd == "cleanup":
        for p in our_pods():
            delete_pod(p["id"])
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
