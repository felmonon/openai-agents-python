from __future__ import annotations

import inspect
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from agents import Usage
from agents.exceptions import UserError
from agents.extensions.experimental.codex import (
    BuildPlan,
    BuildRoundReport,
    CodexBuilderQAToolOptions,
    CodexBuilderQAToolResult,
    EvaluationReport,
    codex_builder_qa_tool,
)
from agents.tool_context import ToolContext


class FakeRunResult:
    def __init__(self, final_output: Any, usage: Usage | None = None) -> None:
        self.final_output = final_output
        self.context_wrapper = SimpleNamespace(usage=usage or Usage())

    def final_output_as(self, cls: type[Any], raise_if_incorrect_type: bool = False) -> Any:
        if raise_if_incorrect_type and not isinstance(self.final_output, cls):
            raise TypeError(f"Final output is not of type {cls.__name__}")
        return self.final_output


def test_codex_builder_qa_tool_kw_matches_options() -> None:
    signature = inspect.signature(codex_builder_qa_tool)
    kw_only = [
        param.name
        for param in signature.parameters.values()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    option_fields = [field.name for field in fields(CodexBuilderQAToolOptions)]
    assert kw_only == option_fields


@pytest.mark.asyncio
async def test_codex_builder_qa_tool_runs_multi_round_harness(monkeypatch, tmp_path: Path) -> None:
    planner_usage = Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
    builder_usage = Usage(requests=1, input_tokens=20, output_tokens=10, total_tokens=30)
    qa_usage = Usage(requests=1, input_tokens=15, output_tokens=5, total_tokens=20)
    builder_calls = 0
    qa_calls = 0

    async def fake_run(agent: Any, input_data: str, context: Any = None) -> FakeRunResult:
        nonlocal builder_calls, qa_calls
        if agent.name == "planner_agent":
            return FakeRunResult(
                BuildPlan(
                    project_name="Demo App",
                    product_goal="Ship a demo app",
                    architecture=["CLI"],
                    milestones=["Build the CLI"],
                    acceptance_criteria=["Commands work"],
                    qa_focus=["Regression"],
                ),
                planner_usage,
            )
        if agent.name == "generator_agent":
            builder_calls += 1
            context.codex_thread_id_builder = f"builder-thread-{builder_calls}"
            return FakeRunResult(
                BuildRoundReport(
                    summary=f"builder round {builder_calls}",
                    completed_work=[f"round-{builder_calls}"],
                    validations_run=["pytest -q"],
                    remaining_risks=[] if builder_calls > 1 else ["Need QA"],
                ),
                builder_usage,
            )
        if agent.name == "evaluator_agent":
            qa_calls += 1
            context.codex_thread_id_qa = f"qa-thread-{qa_calls}"
            verdict: Literal["pass", "revise"] = "revise" if qa_calls == 1 else "pass"
            return FakeRunResult(
                EvaluationReport(
                    verdict=verdict,
                    summary=f"qa round {qa_calls}",
                    strengths=["Good structure"],
                    issues=[],
                    next_actions=[] if verdict == "pass" else ["Fix remaining bug"],
                ),
                qa_usage,
            )
        raise AssertionError(f"Unexpected agent: {agent.name}")

    monkeypatch.setattr(
        "agents.extensions.experimental.codex.builder_qa_tool.Runner.run",
        fake_run,
    )

    tool = codex_builder_qa_tool(
        working_directory=str(tmp_path / "workspace"),
        create_scratch_workspace=False,
        default_max_rounds=3,
    )
    input_json = json.dumps({"task": "Build a demo CLI"})
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    result = await tool.on_invoke_tool(context, input_json)

    assert isinstance(result, CodexBuilderQAToolResult)
    assert result.final_verdict == "pass"
    assert result.rounds_completed == 2
    assert result.builder_thread_id == "builder-thread-2"
    assert result.qa_thread_id == "qa-thread-2"
    assert context.usage.total_tokens == 15 + 30 + 20 + 30 + 20

    artifact_dir = tmp_path / "workspace" / ".codex-harness"
    assert (artifact_dir / "plan.json").exists()
    assert (artifact_dir / "build_round_1.json").exists()
    assert (artifact_dir / "build_round_2.json").exists()
    assert (artifact_dir / "qa_round_1.json").exists()
    assert (artifact_dir / "qa_round_2.json").exists()

    stringified = json.loads(str(result))
    assert stringified["final_verdict"] == "pass"
    assert stringified["rounds_completed"] == 2


@pytest.mark.asyncio
async def test_codex_builder_qa_tool_scratch_workspace_sets_src_pythonpath(
    monkeypatch,
) -> None:
    async def fake_run(agent: Any, input_data: str, context: Any = None) -> FakeRunResult:
        if agent.name == "planner_agent":
            return FakeRunResult(
                BuildPlan(
                    project_name="Scratch App",
                    product_goal="Ship scratch app",
                    architecture=[],
                    milestones=[],
                    acceptance_criteria=[],
                    qa_focus=[],
                )
            )
        if agent.name == "generator_agent":
            context.codex_thread_id_builder = "builder-thread"
            return FakeRunResult(
                BuildRoundReport(
                    summary="built",
                    completed_work=[],
                    validations_run=[],
                    remaining_risks=[],
                )
            )
        if agent.name == "evaluator_agent":
            context.codex_thread_id_qa = "qa-thread"
            return FakeRunResult(
                EvaluationReport(
                    verdict="pass",
                    summary="passed",
                    strengths=[],
                    issues=[],
                    next_actions=[],
                )
            )
        raise AssertionError(f"Unexpected agent: {agent.name}")

    monkeypatch.setattr(
        "agents.extensions.experimental.codex.builder_qa_tool.Runner.run",
        fake_run,
    )

    tool = codex_builder_qa_tool()
    input_json = json.dumps({"task": "Build a scratch app"})
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    result = await tool.on_invoke_tool(context, input_json)

    pyproject = Path(result.workspace) / "pyproject.toml"
    assert pyproject.exists()
    assert 'pythonpath = ["src"]' in pyproject.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_codex_builder_qa_tool_requires_workspace_when_scratch_disabled() -> None:
    tool = codex_builder_qa_tool(
        create_scratch_workspace=False,
        failure_error_function=None,
    )
    input_json = json.dumps({"task": "Build a demo app"})
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments=input_json,
    )

    with pytest.raises(UserError, match="requires working_directory"):
        await tool.on_invoke_tool(context, input_json)
