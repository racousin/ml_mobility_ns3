import json
import shutil
from pathlib import Path

def cleanup_failed_experiments():
    manifest_path = Path("experiments/manifest.json")
    if not manifest_path.exists():
        print("Manifest not found.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    original_count = len(manifest["experiments"])
    failed_ids = [exp["id"] for exp in manifest["experiments"] if exp["status"] == "failed"]
    
    if not failed_ids:
        print("No failed experiments found.")
        return

    print(f"Found {len(failed_ids)} failed experiments. Cleaning up...")

    # Keep only non-failed experiments
    manifest["experiments"] = [exp for exp in manifest["experiments"] if exp["status"] != "failed"]

    # Delete folders
    for exp_id in failed_ids:
        exp_dir = Path("experiments") / exp_id
        if exp_dir.exists():
            print(f"Deleting {exp_dir}...")
            shutil.rmtree(exp_dir)
        else:
            print(f"Folder {exp_dir} already gone or doesn't exist.")

    # Write updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Cleanup complete. Removed {len(failed_ids)} entries. Remaining: {len(manifest['experiments'])}")

if __name__ == "__main__":
    cleanup_failed_experiments()
