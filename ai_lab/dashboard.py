from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ai_lab.config import parse_simple_yaml
from ai_lab.paths import repo_root
from ai_lab.research_ops import build_default_loops, load_reports
from ai_lab.storage import LabStore


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _resolve_project_root(config_file: Path) -> Path:
    resolved = config_file.resolve()
    parent = resolved.parent

    if parent.name == "configs" and parent.parent.name == "ai_lab":
        return parent.parent.parent
    if parent.name == "config":
        return parent.parent
    if parent.name == "ai_lab" and (parent / "configs").exists():
        return parent.parent
    return parent


def _load_manifest(manifest_file: Path) -> dict[str, Any]:
    if not manifest_file.exists():
        return {}
    return json.loads(manifest_file.read_text())


def build_lab_state(
    config_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    config_file = Path(config_path)
    project_root = _resolve_project_root(config_file)
    config = parse_simple_yaml(config_file) if config_file.exists() else {}

    manifest_file = Path(manifest_path)
    last_run = _load_manifest(manifest_file)
    storage_dir = project_root / "storage"
    store = LabStore(storage_dir)

    train_file = project_root / str(config.get("train_file", "storage/datasets/train.jsonl"))
    eval_file = project_root / str(config.get("eval_file", "storage/datasets/evals.jsonl"))

    recent_runs = store.recent_runs(limit=5)
    latest_stored_run = store.latest_run() or {}
    latest_metrics = json.loads(latest_stored_run.get("metrics_json", "{}")) if latest_stored_run else {}
    latest_model = latest_metrics.get("model", {}) if isinstance(latest_metrics, dict) else {}
    latest_cache = latest_metrics.get("cache", {}) if isinstance(latest_metrics, dict) else {}
    if latest_stored_run:
        last_run = {
            **last_run,
            "created_at": latest_stored_run.get("created_at", last_run.get("created_at")),
            "name": latest_stored_run.get("name"),
            "mode": latest_stored_run.get("mode"),
            "status": latest_stored_run.get("status"),
            "artifact_path": latest_stored_run.get("artifact_path"),
        }

    latest_eval_results = (
        store.eval_results_for_run(str(latest_stored_run.get("run_id"))) if latest_stored_run else []
    )

    reports = load_reports(project_root / "reports")
    loops = build_default_loops(config.get("project_name", "sydney-ai-lab"))
    comparisons = _build_run_comparisons(recent_runs)
    task_diffs = _build_task_diffs(latest_eval_results)

    return {
        "project_name": config.get("project_name", "sydney-ai-lab"),
        "base_model": config.get("base_model", "unknown"),
        "hardware_profile": config.get("hardware_profile", "unknown"),
        "train_file": str(config.get("train_file", train_file)),
        "eval_file": str(config.get("eval_file", eval_file)),
        "docs_dir": str(config.get("docs_dir", project_root / "storage" / "docs")),
        "output_dir": str(config.get("output_dir", "storage/runs")),
        "database_path": str(storage_dir / "lab.sqlite3"),
        "model_backend": f"{latest_model.get('provider', 'auto')}:{latest_model.get('name', config.get('base_model', 'unknown'))}",
        "cache_state": "hit" if latest_cache.get("hit") else "miss",
        "dataset_counts": {
            "train": _count_jsonl_rows(train_file),
            "eval": _count_jsonl_rows(eval_file),
        },
        "last_run": last_run,
        "recent_runs": recent_runs,
        "status": "ready" if train_file.exists() and eval_file.exists() else "missing-data",
        "loops": loops,
        "reports": reports,
        "experiments": _build_experiments(recent_runs, loops, reports),
        "comparisons": comparisons,
        "task_diffs": task_diffs,
    }


def _build_experiments(
    recent_runs: list[dict[str, Any]],
    loops: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    for run in recent_runs:
        metrics = json.loads(run.get("metrics_json", "{}"))
        experiments.append(
            {
                "name": run.get("name", "experiment"),
                "mode": run.get("mode", "unknown"),
                "status": run.get("status", "unknown"),
                "created_at": run.get("created_at", "unknown"),
                "avg_score": metrics.get("variants", {}).get("rag", {}).get("avg_score", 0.0),
                "best_score": metrics.get("variants", {}).get("rag", {}).get("best_score", 0.0),
                "avg_latency_seconds": metrics.get("variants", {}).get("rag", {}).get("avg_latency_seconds", 0.0),
                "estimated_cost_usd": metrics.get("variants", {}).get("rag", {}).get("estimated_cost_usd", 0.0),
                "docs": metrics.get("docs", 0),
                "chunks": metrics.get("chunks", 0),
            }
        )

    if experiments:
        return experiments

    for loop in loops[:3]:
        experiments.append(
            {
                "name": loop["name"],
                "mode": loop["id"],
                "status": loop["status"],
                "created_at": "not-run-yet",
                "avg_score": 0.0,
                "best_score": 0.0,
                "avg_latency_seconds": 0.0,
                "estimated_cost_usd": 0.0,
                "docs": 0,
                "chunks": 0,
            }
        )

    for report in reports[:2]:
        experiments.append(
            {
                "name": report.get("title", "report"),
                "mode": report.get("loop_id", "report"),
                "status": "report",
                "created_at": report.get("created_at", "unknown"),
                "avg_score": 0.0,
                "best_score": 0.0,
                "avg_latency_seconds": 0.0,
                "estimated_cost_usd": 0.0,
                "docs": 0,
                "chunks": 0,
            }
        )

    return experiments


def _build_run_comparisons(recent_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    previous_rag = None

    for run in reversed(recent_runs):
        metrics = json.loads(run.get("metrics_json", "{}"))
        variants = metrics.get("variants", {})
        rag = variants.get("rag", {})
        baseline = variants.get("baseline", {})
        rag_avg = float(rag.get("avg_score", 0.0))
        baseline_avg = float(baseline.get("avg_score", 0.0))
        delta = rag_avg - previous_rag if previous_rag is not None else 0.0
        previous_rag = rag_avg
        comparisons.append(
            {
                "run_id": run.get("run_id", "unknown"),
                "created_at": run.get("created_at", "unknown"),
                "name": run.get("name", "experiment"),
                "docs": metrics.get("docs", 0),
                "chunks": metrics.get("chunks", 0),
                "baseline_avg": baseline_avg,
                "rag_avg": rag_avg,
                "pass_rate": float(rag.get("pass_rate", 0.0)),
                "avg_latency_seconds": float(rag.get("avg_latency_seconds", 0.0)),
                "estimated_cost_usd": float(rag.get("estimated_cost_usd", 0.0)),
                "delta": delta,
                "synthetic_examples": metrics.get("synthetic_examples", 0),
                "cache_hit": bool(metrics.get("cache", {}).get("hit", False)),
            }
        )
    return list(reversed(comparisons))


def render_dashboard_html(state: dict[str, Any]) -> str:
    dataset_counts = state.get("dataset_counts", {})
    project_name = html.escape(str(state.get("project_name", "sydney-ai-research-lab")))
    base_model = html.escape(str(state.get("base_model", "unknown")))
    hardware = html.escape(str(state.get("hardware_profile", "unknown")))
    status = html.escape(str(state.get("status", "unknown")))
    train_count = html.escape(str(dataset_counts.get("train", 0)))
    eval_count = html.escape(str(dataset_counts.get("eval", 0)))
    train_file = html.escape(str(state.get("train_file", "storage/datasets/train.jsonl")))
    eval_file = html.escape(str(state.get("eval_file", "storage/datasets/evals.jsonl")))
    docs_dir = html.escape(str(state.get("docs_dir", "storage/docs")))
    database_path = html.escape(str(state.get("database_path", "storage/lab.sqlite3")))
    output_dir = html.escape(str(state.get("output_dir", "storage/runs")))
    model_backend = html.escape(str(state.get("model_backend", "auto")))
    cache_state = html.escape(str(state.get("cache_state", "unknown")))
    created_at = html.escape(str(state.get("last_run", {}).get("created_at", "not-run-yet")))
    recent_runs = state.get("recent_runs", [])
    loops = state.get("loops", [])
    reports = state.get("reports", [])
    experiments = state.get("experiments", [])
    comparisons = state.get("comparisons", [])
    task_diffs = state.get("task_diffs", [])

    summary_cards = [
        _metric_card("Train samples", train_count, train_file),
        _metric_card("Eval samples", eval_count, eval_file),
        _metric_card("Docs source", "local-first", docs_dir),
        _metric_card("SQLite", "on", database_path),
    ]

    history_cards = "".join(
        _run_card(run) for run in recent_runs
    ) or '<article class="list-card"><p class="muted">No experiment runs yet.</p></article>'

    loop_cards = "".join(
        f'<article class="list-card"><div class="list-head"><h3>{html.escape(loop["name"])}</h3><span class="pill">{html.escape(loop["status"])}</span></div><div class="meta">Cadence: {html.escape(loop["cadence"])}</div><p>{html.escape(loop["goal"])}</p></article>'
        for loop in loops
    ) or '<article class="list-card"><p class="muted">No active loops yet.</p></article>'

    report_cards = "".join(
        f'<article class="list-card"><div class="list-head"><h3>{html.escape(report["title"])}</h3><span class="meta">{html.escape(report.get("created_at", "unknown"))}</span></div><div class="meta">Report summary</div><p>{html.escape(report["summary"])}</p></article>'
        for report in reports
    ) or '<article class="list-card"><p class="muted">No reports yet. Run the loop generator to seed reports.</p></article>'

    experiment_cards = "".join(_experiment_card(experiment) for experiment in experiments) or '<article class="list-card"><p class="muted">No experiments yet.</p></article>'
    chart_svg = _render_experiment_chart(experiments)
    comparison_table = _render_comparison_table(comparisons)
    task_diff_cards = _render_task_diff_cards(task_diffs)

    return f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sydney AI Research Lab</title>
    <style>
      :root {{
        --bg: #07111a;
        --panel: rgba(12, 20, 30, 0.92);
        --panel-2: #0f1b27;
        --border: #1e3245;
        --text: #d8e4ef;
        --muted: #7f98ad;
        --accent: #67e8f9;
        --good: #22c55e;
        --bad: #ef4444;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background: radial-gradient(circle at top, rgba(103,232,249,.12), transparent 30%), linear-gradient(180deg, #0b1622 0%, var(--bg) 100%);
        color: var(--text);
        font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      }}
      .lab-shell {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 60px; }}
      .hero {{ display: grid; grid-template-columns: 1.5fr 1fr; gap: 16px; margin-bottom: 18px; }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 14px 48px rgba(0,0,0,0.28);
        backdrop-filter: blur(10px);
      }}
      .eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: .12em; font-size: 11px; margin-bottom: 10px; }}
      h1, h2, h3 {{ margin: 0; }}
      h1 {{ font-size: 30px; letter-spacing: -0.03em; margin-bottom: 8px; }}
      h2 {{ font-size: 17px; margin-bottom: 14px; }}
      h3 {{ font-size: 15px; margin-bottom: 6px; }}
      p {{ margin: 8px 0 0; }}
      .muted, .meta, .path, .chart-label, .chart-side-label {{ color: var(--muted); }}
      .runtime-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
      .runtime-item {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; }}
      .runtime-label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
      .runtime-value {{ display: block; font-weight: 600; word-break: break-word; }}
      .metrics-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
      .metric-card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 16px; padding: 16px; min-height: 126px; display: flex; flex-direction: column; gap: 8px; }}
      .metric-label {{ color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-size: 11px; }}
      .metric-value {{ font-size: 26px; font-weight: 700; letter-spacing: -0.04em; }}
      .tabs {{ display: flex; gap: 10px; margin: 8px 0 18px; flex-wrap: wrap; }}
      .tab-button {{ appearance: none; border: 1px solid var(--border); background: #0d1824; color: var(--muted); border-radius: 999px; padding: 10px 14px; cursor: default; font-weight: 600; }}
      .tab-button.active {{ color: white; background: linear-gradient(135deg, #13283d, #102030); border-color: #2b4862; }}
      .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }}
      .list-card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 14px; padding: 14px; margin-bottom: 12px; }}
      .list-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: start; }}
      .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; border: 1px solid #28445c; color: var(--accent); background: rgba(103, 232, 249, 0.08); }}
      .pill.success {{ color: #bbf7d0; border-color: #1f6d43; background: rgba(34,197,94,.12); }}
      .pill.failure {{ color: #fecaca; border-color: #7f1d1d; background: rgba(239,68,68,.12); }}
      .pill.neutral {{ color: #dbeafe; border-color: #315275; background: rgba(59,130,246,.12); }}
      .section-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin-bottom: 10px; }}
      .topline {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }}
      .topline .pill {{ font-size: 11px; }}
      .chart-wrap {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 16px; padding: 16px; margin-top: 16px; }}
      .experiment-chart {{ width: 100%; height: auto; display: block; }}
      .axis {{ stroke: #385066; stroke-width: 1.5; }}
      .guide {{ stroke-width: 1; stroke-dasharray: 4 5; }}
      .success-guide {{ stroke: rgba(34,197,94,.35); }}
      .failure-guide {{ stroke: rgba(239,68,68,.35); }}
      .line-success, .line-failure {{ fill: none; stroke-width: 4; stroke-linecap: round; stroke-linejoin: round; }}
      .line-success {{ stroke: var(--good); }}
      .line-failure {{ stroke: var(--bad); }}
      .chart-label, .chart-side-label {{ font-size: 11px; fill: var(--muted); }}
      .legend {{ display: flex; gap: 18px; margin-bottom: 14px; flex-wrap: wrap; }}
      .legend-item {{ display: inline-flex; align-items: center; gap: 8px; color: var(--muted); }}
      .legend-swatch {{ width: 22px; height: 4px; border-radius: 999px; display: inline-block; }}
      .legend-swatch.success {{ background: var(--good); }}
      .legend-swatch.failure {{ background: var(--bad); }}
      .comparison-table-wrap {{ overflow-x: auto; }}
      .comparison-table {{ width: 100%; border-collapse: collapse; min-width: 940px; }}
      .comparison-table th, .comparison-table td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
      .comparison-table th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .1em; }}
      .task-diff-card {{ border-left: 3px solid rgba(103, 232, 249, 0.25); }}
      .diff-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
      .diff-pane {{ background: rgba(7, 17, 26, 0.7); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }}
      .diff-label {{ color: var(--accent); text-transform: uppercase; letter-spacing: .08em; font-size: 11px; margin-bottom: 8px; }}
      .diff-answer {{ margin: 0; white-space: pre-wrap; color: var(--text); font: inherit; line-height: 1.55; }}
      .diff-support {{ margin: 10px 0 0; padding-left: 18px; color: var(--muted); }}
      .diff-support li {{ margin-bottom: 6px; }}
      @media (max-width: 980px) {{
        .hero, .grid-2, .metrics-grid {{ grid-template-columns: 1fr; }}
        .diff-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="lab-shell">
      <section class="hero">
        <div class="panel">
          <div class="eyebrow">Sydney AI Research Lab</div>
          <h1>{project_name}</h1>
          <p class="muted">Local-first research loop for docs ingest, chunking, retrieval, evals, and repeatable experiments.</p>
          <div class="topline">
            <span class="pill neutral">{status}</span>
            <span class="pill">{base_model}</span>
            <span class="pill">{hardware}</span>
            <span class="pill">{model_backend}</span>
            <span class="pill">{cache_state}</span>
            <span class="pill">{created_at}</span>
          </div>
          <p class="muted">Output directory: {output_dir}</p>
        </div>
        <div class="panel">
          <h2>Runtime</h2>
          <div class="runtime-grid">
            <div class="runtime-item"><span class="runtime-label">Train file</span><span class="runtime-value">{train_file}</span></div>
            <div class="runtime-item"><span class="runtime-label">Eval file</span><span class="runtime-value">{eval_file}</span></div>
            <div class="runtime-item"><span class="runtime-label">Docs dir</span><span class="runtime-value">{docs_dir}</span></div>
            <div class="runtime-item"><span class="runtime-label">SQLite DB</span><span class="runtime-value">{database_path}</span></div>
          </div>
        </div>
      </section>

      <nav class="tabs" aria-label="Dashboard tabs">
        <button class="tab-button active">Overview</button>
        <button class="tab-button">Experiments</button>
        <button class="tab-button">Reports</button>
      </nav>

      <section class="panel">
        <h2>Workspace Overview</h2>
        <div class="metrics-grid">
          {''.join(summary_cards)}
        </div>
        <div class="chart-wrap">
          <div class="legend">
            <span class="legend-item"><span class="legend-swatch success"></span>Worked</span>
            <span class="legend-item"><span class="legend-swatch failure"></span>Did not land</span>
          </div>
          {chart_svg}
        </div>
      </section>

      <section class="grid-2">
        <div class="panel">
          <div class="section-label">Recent Runs</div>
          {history_cards}
        </div>
        <div class="panel">
          <div class="section-label">Experiment Lane</div>
          {experiment_cards}
        </div>
      </section>

      <section class="grid-2">
        <div class="panel">
          <div class="section-label">Operating Loops</div>
          {loop_cards}
        </div>
        <div class="panel">
          <div class="section-label">Saved Reports</div>
          {report_cards}
        </div>
      </section>

      <section class="panel">
        <div class="section-label">Run Comparison</div>
        {comparison_table}
      </section>

      <section class="panel">
        <div class="section-label">Task Diff View</div>
        {task_diff_cards}
      </section>
    </main>
  </body>
</html>"""


def _metric_card(label: str, value: str, path: str) -> str:
    return (
        '<div class="metric-card">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<span class="metric-value">{html.escape(value)}</span>'
        f'<span class="path">{html.escape(path)}</span>'
        "</div>"
    )


def _run_card(run: dict[str, Any]) -> str:
    metrics = json.loads(run.get("metrics_json", "{}"))
    rag = metrics.get("variants", {}).get("rag", {})
    status_class = "success" if str(run.get("status", "")).lower() == "completed" else "neutral"
    return (
        '<article class="list-card">'
        '<div class="list-head">'
        f'<h3>{html.escape(run.get("name", "run"))}</h3>'
        f'<span class="pill {status_class}">{html.escape(str(run.get("status", "unknown")))}'
        "</span></div>"
        f'<div class="meta">Created {html.escape(str(run.get("created_at", "unknown")))} · mode {html.escape(str(run.get("mode", "unknown")))} </div>'
        f'<p>RAG avg score: {rag.get("avg_score", 0.0):.3f} · latency {rag.get("avg_latency_seconds", 0.0):.2f}s · cost ${rag.get("estimated_cost_usd", 0.0):.4f}</p>'
        f'<div class="meta">docs {metrics.get("docs", 0)} · chunks {metrics.get("chunks", 0)}</div>'
        "</article>"
    )


def _experiment_card(experiment: dict[str, Any]) -> str:
    score = float(experiment.get("avg_score", 0.0))
    pill_class = "success" if score >= 0.7 else "neutral"
    return (
        '<article class="list-card">'
        '<div class="list-head">'
        f'<h3>{html.escape(str(experiment.get("name", "experiment")))}</h3>'
        f'<span class="pill {pill_class}">{score:.3f}</span>'
        "</div>"
        f'<div class="meta">{html.escape(str(experiment.get("created_at", "unknown")))} · {html.escape(str(experiment.get("mode", "unknown")))}</div>'
        f'<p>Best score {float(experiment.get("best_score", 0.0)):.3f} · latency {float(experiment.get("avg_latency_seconds", 0.0)):.2f}s · cost ${float(experiment.get("estimated_cost_usd", 0.0)):.4f}</p>'
        f'<div class="meta">docs {experiment.get("docs", 0)} · chunks {experiment.get("chunks", 0)}</div>'
        "</article>"
    )


def _render_comparison_table(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return '<p class="muted">No run comparisons yet.</p>'

    rows = []
    for comparison in comparisons[:6]:
        delta = float(comparison.get("delta", 0.0))
        delta_class = "success" if delta >= 0 else "failure"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(comparison.get('created_at', 'unknown')))}</td>"
            f"<td>{html.escape(str(comparison.get('name', 'run')))}</td>"
            f"<td>{comparison.get('docs', 0)}</td>"
            f"<td>{comparison.get('chunks', 0)}</td>"
            f"<td>{float(comparison.get('baseline_avg', 0.0)):.3f}</td>"
            f"<td>{float(comparison.get('rag_avg', 0.0)):.3f}</td>"
            f"<td><span class=\"pill {delta_class}\">{delta:+.3f}</span></td>"
            f"<td>{float(comparison.get('pass_rate', 0.0)):.2f}</td>"
            f"<td>{float(comparison.get('avg_latency_seconds', 0.0)):.2f}s</td>"
            f"<td>${float(comparison.get('estimated_cost_usd', 0.0)):.4f}</td>"
            f"<td>{comparison.get('synthetic_examples', 0)}</td>"
            "</tr>"
        )

    return (
        '<div class="comparison-table-wrap">'
        '<table class="comparison-table">'
        '<thead><tr><th>Created</th><th>Run</th><th>Docs</th><th>Chunks</th><th>Baseline</th><th>RAG</th><th>Delta</th><th>Pass Rate</th><th>Latency</th><th>Cost</th><th>Synthetic</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _build_task_diffs(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in eval_rows:
        payload = json.loads(row.get("details_json", "{}")) if row.get("details_json") else row
        task_id = str(payload.get("task_id", row.get("task_id", "unknown")))
        variant = str(payload.get("variant", row.get("variant", "unknown")))
        grouped.setdefault(task_id, {})[variant] = payload

    diffs: list[dict[str, Any]] = []
    for task_id, variants in grouped.items():
        baseline = variants.get("baseline", {})
        rag = variants.get("rag", {})
        if not baseline and not rag:
            continue
        baseline_score = float(baseline.get("score", 0.0))
        rag_score = float(rag.get("score", 0.0))
        diffs.append(
            {
                "task_id": task_id,
                "delta": rag_score - baseline_score,
                "baseline": baseline,
                "rag": rag,
                "baseline_score": baseline_score,
                "rag_score": rag_score,
                "baseline_passed": bool(baseline.get("passed", False)),
                "rag_passed": bool(rag.get("passed", False)),
                "baseline_latency_seconds": float(baseline.get("metrics", {}).get("latency_seconds", 0.0)),
                "rag_latency_seconds": float(rag.get("metrics", {}).get("latency_seconds", 0.0)),
                "baseline_cost_usd": float(baseline.get("metrics", {}).get("estimated_cost_usd", 0.0)),
                "rag_cost_usd": float(rag.get("metrics", {}).get("estimated_cost_usd", 0.0)),
                "baseline_prompt_tokens": float(baseline.get("metrics", {}).get("prompt_tokens", 0.0)),
                "rag_prompt_tokens": float(rag.get("metrics", {}).get("prompt_tokens", 0.0)),
                "baseline_completion_tokens": float(baseline.get("metrics", {}).get("completion_tokens", 0.0)),
                "rag_completion_tokens": float(rag.get("metrics", {}).get("completion_tokens", 0.0)),
                "retrieved_chunks": rag.get("retrieved_chunks", []),
            }
        )

    return sorted(diffs, key=lambda item: item["delta"], reverse=True)


def _render_task_diff_cards(task_diffs: list[dict[str, Any]]) -> str:
    if not task_diffs:
        return '<p class="muted">No task diff data yet. Run an experiment to populate baseline vs RAG comparisons.</p>'

    cards = []
    for diff in task_diffs:
        delta = float(diff.get("delta", 0.0))
        delta_class = "success" if delta >= 0 else "failure"
        retrieved_chunks = diff.get("retrieved_chunks", [])
        support = ""
        if retrieved_chunks:
            support = "<ul class=\"diff-support\">" + "".join(
                f"<li><strong>{html.escape(str(chunk.get('doc_id', 'doc')))}</strong>: {html.escape(str(chunk.get('preview', '')))}</li>"
                for chunk in retrieved_chunks[:3]
            ) + "</ul>"

        cards.append(
            '<article class="list-card task-diff-card">'
            '<div class="list-head">'
            f'<h3>{html.escape(str(diff.get("task_id", "task")))}</h3>'
            f'<span class="pill {delta_class}">{delta:+.3f}</span>'
            "</div>"
            f'<div class="meta">Baseline {float(diff.get("baseline_score", 0.0)):.3f} · RAG {float(diff.get("rag_score", 0.0)):.3f} · '
            f'latency {float(diff.get("baseline_latency_seconds", 0.0)):.2f}s → {float(diff.get("rag_latency_seconds", 0.0)):.2f}s · '
            f'cost ${float(diff.get("baseline_cost_usd", 0.0)):.4f} → ${float(diff.get("rag_cost_usd", 0.0)):.4f}</div>'
            '<div class="diff-grid">'
            f'<div class="diff-pane"><div class="diff-label">Baseline</div><pre class="diff-answer">{html.escape(str(diff.get("baseline", {}).get("answer", "")))}</pre></div>'
            f'<div class="diff-pane"><div class="diff-label">RAG</div><pre class="diff-answer">{html.escape(str(diff.get("rag", {}).get("answer", "")))}</pre></div>'
            "</div>"
            f'<div class="meta">Pass/fail: {html.escape(str(bool(diff.get("baseline_passed", False))))} → {html.escape(str(bool(diff.get("rag_passed", False))))}</div>'
            f"{support}"
            "</article>"
        )

    return "".join(cards)


def _render_experiment_chart(experiments: list[dict[str, Any]]) -> str:
    width = 760
    height = 220
    left = 36
    bottom = 28
    top = 24
    usable_w = width - left - 18
    usable_h = height - top - bottom

    if not experiments:
        return f"""<svg class="experiment-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Experiment outcomes chart">
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - 10}" y2="{height - bottom}" class="axis" />
  <line x1="{left}" y1="{top + usable_h * 0.25:.1f}" x2="{width - 10}" y2="{top + usable_h * 0.25:.1f}" class="guide success-guide" />
  <line x1="{left}" y1="{top + usable_h * 0.78:.1f}" x2="{width - 10}" y2="{top + usable_h * 0.78:.1f}" class="guide failure-guide" />
  <text x="8" y="{top + usable_h * 0.25:.1f}" class="chart-side-label">worked</text>
  <text x="8" y="{top + usable_h * 0.78:.1f}" class="chart-side-label">missed</text>
</svg>"""

    step = usable_w / max(len(experiments) - 1, 1)
    success_points: list[str] = []
    failure_points: list[str] = []
    labels: list[str] = []

    for index, experiment in enumerate(experiments[:8]):
        score = float(experiment.get("avg_score", 0.0))
        worked = score >= 0.5
        x = left + index * step
        y = top + (usable_h * (0.25 if worked else 0.78))
        target = success_points if worked else failure_points
        target.append(f"{x:.1f},{y:.1f}")
        labels.append(
            f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" class="chart-label">{html.escape(str(index + 1))}</text>'
        )

    success_poly = f'<polyline class="line-success" points="{" ".join(success_points)}" />' if success_points else ""
    failure_poly = f'<polyline class="line-failure" points="{" ".join(failure_points)}" />' if failure_points else ""

    return f"""<svg class="experiment-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Experiment outcomes chart">
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - 10}" y2="{height - bottom}" class="axis" />
  <line x1="{left}" y1="{top + usable_h * 0.25:.1f}" x2="{width - 10}" y2="{top + usable_h * 0.25:.1f}" class="guide success-guide" />
  <line x1="{left}" y1="{top + usable_h * 0.78:.1f}" x2="{width - 10}" y2="{top + usable_h * 0.78:.1f}" class="guide failure-guide" />
  {success_poly}
  {failure_poly}
  {''.join(labels)}
  <text x="8" y="{top + usable_h * 0.25:.1f}" class="chart-side-label">worked</text>
  <text x="8" y="{top + usable_h * 0.78:.1f}" class="chart-side-label">missed</text>
</svg>"""
