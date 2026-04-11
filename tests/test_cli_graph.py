from typer.testing import CliRunner

from miniclaw.cli.app import app


def test_graph_command_outputs_mermaid():
    runner = CliRunner()
    result = runner.invoke(app, ["graph", "--format", "mermaid"])
    assert result.exit_code == 0, f"stderr: {result.stderr if hasattr(result, 'stderr') else result.output}"
    out = result.stdout
    # The graph contains these nodes
    assert "classify" in out
    assert "agent" in out
    assert "planner" in out
    assert "error_handler" in out


def test_graph_command_default_is_mermaid():
    runner = CliRunner()
    result = runner.invoke(app, ["graph"])
    assert result.exit_code == 0
    assert "agent" in result.stdout
