import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.research_ops import build_default_loops, build_report, write_report, load_reports


class BuildDefaultLoopsTests(unittest.TestCase):
    def test_build_default_loops_returns_expected_phase_one_loops(self):
        loops = build_default_loops(project_name="sydney-ai-lab")

        self.assertEqual([loop["id"] for loop in loops], [
            "prompt-optimizer",
            "rd-council",
            "ambient-research",
        ])
        self.assertEqual(loops[0]["cadence"], "every 5m")
        self.assertEqual(loops[1]["cadence"], "twice daily")
        self.assertEqual(loops[2]["status"], "active")


class ReportPersistenceTests(unittest.TestCase):
    def test_write_report_persists_structured_report_and_load_reports_returns_latest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            first = build_report(
                loop_id="prompt-optimizer",
                title="Prompt loop",
                summary="Improved the baseline prompt.",
                findings=["Variant B handled edge cases better"],
                next_actions=["Promote Variant B to staging"],
                created_at="2026-03-22T03:30:00Z",
            )
            second = build_report(
                loop_id="rd-council",
                title="R&D Council",
                summary="Council recommends eval automation.",
                findings=["Need benchmark prompts", "Need quality rubric"],
                next_actions=["Add benchmark runner"],
                created_at="2026-03-22T07:30:00Z",
            )

            write_report(report_dir, first)
            write_report(report_dir, second)

            reports = load_reports(report_dir)

            self.assertEqual(len(reports), 2)
            self.assertEqual(reports[0]["loop_id"], "rd-council")
            self.assertEqual(reports[1]["loop_id"], "prompt-optimizer")
            self.assertEqual(reports[0]["findings"], ["Need benchmark prompts", "Need quality rubric"])


if __name__ == "__main__":
    unittest.main()
