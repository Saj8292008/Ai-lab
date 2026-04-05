from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_lab.loop_runner import run_phase_one_loops


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sydney AI Lab phase-one research loops.")
    parser.add_argument("--config", default="ai_lab/configs/qwen3_text_lora.yaml")
    parser.add_argument("--manifest", default="runs/qwen3-1.7b-lora/manifest.json")
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    result = run_phase_one_loops(
        config_path=args.config,
        manifest_path=args.manifest,
        report_dir=args.report_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
