from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.config import load_lab_config
from ai_lab.dashboard import build_lab_state, render_dashboard_html
from ai_lab.paths import repo_root


class LabHandler(BaseHTTPRequestHandler):
    config_path: Path
    manifest_path: Path

    def do_GET(self) -> None:  # noqa: N802
        state = build_lab_state(self.config_path, self.manifest_path)

        if self.path == "/api/state":
            body = json.dumps(state, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = render_dashboard_html(state).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sydney AI Lab dashboard.")
    parser.add_argument("--config", default="config/lab.sample.yaml")
    parser.add_argument("--manifest", default="storage/runs/latest-manifest.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root() / manifest_path

    LabHandler.config_path = config_path
    LabHandler.manifest_path = manifest_path

    server = ThreadingHTTPServer((args.host, args.port), LabHandler)
    print(f"Sydney AI Lab running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

