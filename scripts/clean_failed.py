#!/usr/bin/env python
"""Remove all failed experiments from manifest and disk."""
import json
import shutil
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")
MANIFEST_FILE = EXPERIMENTS_DIR / "manifest.json"

def main():
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    kept = []
    removed = []

    for exp in manifest["experiments"]:
        if exp["status"] == "failed":
            exp_dir = EXPERIMENTS_DIR / exp["id"]
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
            removed.append(exp["id"])
            print(f"  ✗ Removed: {exp['id']} ({exp['model_type']})")
        else:
            kept.append(exp)
            print(f"  ✓ Kept:    {exp['id']} ({exp['model_type']}) - {exp['status']}")

    manifest["experiments"] = kept
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: removed {len(removed)}, kept {len(kept)}")

if __name__ == "__main__":
    main()
