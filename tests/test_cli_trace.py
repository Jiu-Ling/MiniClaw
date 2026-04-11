import json
from pathlib import Path

from typer.testing import CliRunner

from miniclaw.cli.app import app


def test_trace_tail_reads_jsonl(tmp_path):
    trace_file = tmp_path / "trace_test.jsonl"
    records = [
        {"kind": "run_start", "trace_id": "t1", "span_id": "s1", "parent_span_id": None, "name": "graph.run"},
        {"kind": "span_start", "trace_id": "t1", "span_id": "s2", "parent_span_id": "s1", "name": "graph.classify"},
        {"kind": "span_finish", "trace_id": "t1", "span_id": "s2", "parent_span_id": "s1", "name": "graph.classify", "status": "ok"},
        {"kind": "run_finish", "trace_id": "t1", "span_id": "s1", "parent_span_id": None, "name": "graph.run", "status": "ok"},
    ]
    trace_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    runner = CliRunner()
    result = runner.invoke(app, ["trace", "tail", str(trace_file), "--no-follow"])
    assert result.exit_code == 0, f"output: {result.output}"
    assert "graph.run" in result.stdout
    assert "graph.classify" in result.stdout


def test_trace_tail_missing_file(tmp_path):
    runner = CliRunner()
    missing = tmp_path / "nope.jsonl"
    result = runner.invoke(app, ["trace", "tail", str(missing), "--no-follow"])
    assert result.exit_code != 0
