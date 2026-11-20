#!/usr/bin/env bash
# First job on a new kind of machine: does everything work, and how fast is it?
#   1. GPU visible, 2. GPU rendering (EGL) of the pile camera,
#   3. Newton cloth-pile benchmark with real cloth-cloth contact (1 and many piles),
#   4. one of our existing fold demos end to end (CPU physics, GPU rendering).
set -uo pipefail
OUT="$1"
python3 -c "import warp as wp; wp.init(); print('warp devices:', [str(d) for d in wp.get_devices()])"
python3 scripts/cloud/render_check.py > "$OUT/render_check.json" || echo '{"ok": false}' > "$OUT/render_check.json"
cat "$OUT/render_check.json"
for piles in 1 16 64; do
    python3 scripts/cloud/newton_cloth_bench.py --device cuda --piles "$piles" --seconds 3 | tail -1 | tee -a "$OUT/newton_bench.jsonl"
done
python3 scripts/run_fold_demo.py --garment towel --fast --seed 1 --out "$OUT/towel_fold"
