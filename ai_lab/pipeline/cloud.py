from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


def build_cloud_training_bundle(
    *,
    output_dir: str | Path,
    training_config: dict[str, Any],
    cloud_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cloud_config = dict(cloud_config or {})
    output = Path(output_dir) / "cloud"
    output.mkdir(parents=True, exist_ok=True)

    provider = str(cloud_config.get("provider", "ssh")).strip().lower() or "ssh"
    host = str(cloud_config.get("host", "training-host")).strip()
    user = str(cloud_config.get("user", "ubuntu")).strip()
    remote_root = str(cloud_config.get("remote_root", "/workspace/ai-lab")).rstrip("/")
    remote_python = str(cloud_config.get("python", "python3")).strip() or "python3"
    default_bundle_name = f"{Path(str(training_config.get('output_dir', 'artifacts'))).name}-cloud"
    bundle_name = str(cloud_config.get("bundle_name", default_bundle_name)).strip()
    bundle_name = bundle_name or "cloud-run"
    remote_bundle_dir = f"{remote_root}/{bundle_name}"
    remote_user_host = f"{user}@{host}" if user else host

    dataset_source = Path(str(training_config.get("dataset_path", output / "train.jsonl")))
    dataset_target = output / "train.jsonl"
    if dataset_source.exists() and dataset_source.resolve() != dataset_target.resolve():
        dataset_target.write_text(dataset_source.read_text())

    cloud_training_config = dict(training_config)
    cloud_training_config["dataset_path"] = f"{remote_bundle_dir}/train.jsonl"
    cloud_training_config["output_dir"] = f"{remote_bundle_dir}/artifacts"
    cloud_training_config["cloud"] = {
        "provider": provider,
        "host": host,
        "user": user,
        "remote_root": remote_root,
        "bundle_name": bundle_name,
    }

    cloud_training_config_path = output / "training_config.json"
    cloud_training_config_path.write_text(json.dumps(cloud_training_config, indent=2, sort_keys=True))

    sync_command = f'rsync -av --delete {shlex.quote(str(output))}/ {shlex.quote(remote_user_host)}:{shlex.quote(remote_bundle_dir)}/'
    launch_command = (
        f"ssh {shlex.quote(remote_user_host)} "
        f"'cd {shlex.quote(remote_root)} && {remote_python} -m ai_lab.scripts.train_peft_sft "
        f"--config {shlex.quote(f'{remote_bundle_dir}/training_config.json')}'"
    )

    sync_script = output / "sync.sh"
    sync_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{sync_command}\n"
    )
    sync_script.chmod(0o755)

    launch_script = output / "launch.sh"
    launch_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{launch_command}\n"
    )
    launch_script.chmod(0o755)

    manifest = {
        "provider": provider,
        "host": host,
        "user": user,
        "remote_root": remote_root,
        "remote_bundle_dir": remote_bundle_dir,
        "dataset_path": str(dataset_target),
        "cloud_training_config_path": str(cloud_training_config_path),
        "sync_command": sync_command,
        "launch_command": launch_command,
        "sync_script_path": str(sync_script),
        "launch_script_path": str(launch_script),
    }
    manifest_path = output / "cloud_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "cloud_training_config": cloud_training_config,
        "cloud_training_config_path": str(cloud_training_config_path),
        "sync_script_path": str(sync_script),
        "launch_script_path": str(launch_script),
        "dataset_path": str(dataset_target),
    }
