from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import signal
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, TypeVar, cast

from pydantic import BaseModel, Field

from agents import Agent, ModelSettings, Runner, Usage, UserError, default_tool_error_function
from agents.run_context import RunContextWrapper
from agents.tool import ComputerConfig, ComputerTool, FunctionTool, ToolErrorFunction, function_tool
from agents.tool_context import ToolContext
from agents.util._types import MaybeAwaitable

from .codex_tool import CodexToolStreamEvent, codex_tool
from .thread_options import SandboxMode, ThreadOptions, coerce_thread_options

DEFAULT_HARNESS_TOOL_NAME = "codex_builder_qa"
DEFAULT_ARTIFACT_DIR_NAME = ".codex-harness"
DEFAULT_SCRATCH_WORKSPACE_PREFIX = "codex-builder-qa-"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT: Literal["medium"] = "medium"

DEFAULT_PLANNER_INSTRUCTIONS = (
    "You are the planner in a long-running coding harness. Always use the codex_planner tool. "
    "Expand a terse product request into a concrete build plan that a coding agent and a QA agent "
    "can execute. Be ambitious enough to produce a real product, but keep the scope small enough "
    "that it can plausibly converge in a few implementation rounds. Focus on product behavior, "
    "verification criteria, and likely risks rather than low-level implementation trivia."
)
DEFAULT_BUILDER_INSTRUCTIONS = (
    "You are the generator in a planner / generator / evaluator coding harness. Always use the "
    "codex_builder tool. Work inside the target workspace, implement the highest-value remaining "
    "product slice, and avoid claiming credit for work you did not verify. Favor coherent, "
    "working behavior over broad but stubbed scope. Before you finish each round, run the most "
    "relevant local checks you can and inspect the changed files."
)
DEFAULT_QA_INSTRUCTIONS = (
    "You are the evaluator in a planner / generator / evaluator coding harness. Always use the "
    "codex_qa tool. Be skeptical. If a core requirement is stubbed, broken, or not actually "
    "verified, fail the round. If a browser tool is available, exercise the live app before "
    "deciding. Inspect the workspace, run the most relevant checks you can, and return concrete "
    "evidence plus specific next actions. Do not fix issues yourself."
)


class CodexBuilderQAToolParameters(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        description="Short product or implementation request to plan, build, and review.",
    )
    max_rounds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional maximum builder / QA rounds for this invocation. When omitted, the tool "
            "uses its configured default."
        ),
    )
    resume: bool | None = Field(
        default=None,
        description=(
            "Optional override for resuming an existing harness state from the artifact directory."
        ),
    )


class BuildPlan(BaseModel):
    project_name: str
    product_goal: str
    architecture: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    qa_focus: list[str] = Field(default_factory=list)


class BuildRoundReport(BaseModel):
    summary: str
    completed_work: list[str] = Field(default_factory=list)
    validations_run: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)


class EvaluationIssue(BaseModel):
    severity: Literal["high", "medium", "low"]
    title: str
    evidence: str
    recommendation: str


class EvaluationReport(BaseModel):
    verdict: Literal["pass", "revise"]
    summary: str
    strengths: list[str] = Field(default_factory=list)
    issues: list[EvaluationIssue] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class HarnessState(BaseModel):
    version: Literal[1] = 1
    task: str
    max_rounds: int
    status: Literal["planning", "building", "qa", "completed"]
    current_round: int | None = None
    created_scratch_workspace: bool = False
    planner_thread_id: str | None = None
    builder_thread_id: str | None = None
    qa_thread_id: str | None = None
    final_verdict: str | None = None


class _HarnessContext(BaseModel):
    codex_thread_id_planner: str | None = None
    codex_thread_id_builder: str | None = None
    codex_thread_id_qa: str | None = None


class _UnsetType:
    pass


_UNSET = _UnsetType()
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class CodexBuilderQAToolResult:
    workspace: str
    artifact_dir: str
    state_file: str
    created_scratch_workspace: bool
    rounds_completed: int
    final_verdict: str
    planner_thread_id: str | None
    builder_thread_id: str | None
    qa_thread_id: str | None
    plan: BuildPlan
    build_reports: list[BuildRoundReport]
    qa_reports: list[EvaluationReport]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "artifact_dir": self.artifact_dir,
            "state_file": self.state_file,
            "created_scratch_workspace": self.created_scratch_workspace,
            "rounds_completed": self.rounds_completed,
            "final_verdict": self.final_verdict,
            "planner_thread_id": self.planner_thread_id,
            "builder_thread_id": self.builder_thread_id,
            "qa_thread_id": self.qa_thread_id,
            "plan": self.plan.model_dump(),
            "build_reports": [report.model_dump() for report in self.build_reports],
            "qa_reports": [report.model_dump() for report in self.qa_reports],
        }

    def __str__(self) -> str:
        return json.dumps(self.as_dict())


@dataclass
class _QAServerProcess:
    process: asyncio.subprocess.Process
    log_path: Path
    log_handle: IO[bytes]


@dataclass
class CodexBuilderQAToolOptions:
    name: str | None = None
    description: str | None = None
    planner_model: str | None = None
    planner_model_settings: ModelSettings | None = None
    planner_instructions: str = DEFAULT_PLANNER_INSTRUCTIONS
    planner_thread_options: ThreadOptions | Mapping[str, Any] | None = None
    builder_instructions: str = DEFAULT_BUILDER_INSTRUCTIONS
    qa_instructions: str = DEFAULT_QA_INSTRUCTIONS
    builder_thread_options: ThreadOptions | Mapping[str, Any] | None = None
    qa_thread_options: ThreadOptions | Mapping[str, Any] | None = None
    qa_computer: ComputerConfig[Any] | None = None
    qa_start_command: str | None = None
    qa_base_url: str | None = None
    qa_ready_url: str | None = None
    qa_startup_timeout_seconds: float = 30.0
    sandbox_mode: SandboxMode | None = None
    working_directory: str | None = None
    skip_git_repo_check: bool | None = None
    resume_existing_run: bool = False
    create_scratch_workspace: bool = True
    scratch_workspace_prefix: str = DEFAULT_SCRATCH_WORKSPACE_PREFIX
    artifact_dir_name: str = DEFAULT_ARTIFACT_DIR_NAME
    default_max_rounds: int = 2
    planner_on_stream: Callable[[CodexToolStreamEvent], MaybeAwaitable[None]] | None = None
    builder_on_stream: Callable[[CodexToolStreamEvent], MaybeAwaitable[None]] | None = None
    qa_on_stream: Callable[[CodexToolStreamEvent], MaybeAwaitable[None]] | None = None
    is_enabled: bool | Callable[[RunContextWrapper[Any], Any], MaybeAwaitable[bool]] = True
    failure_error_function: ToolErrorFunction | None = default_tool_error_function


def codex_builder_qa_tool(
    options: CodexBuilderQAToolOptions | Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    planner_model: str | None = None,
    planner_model_settings: ModelSettings | None = None,
    planner_instructions: str | None = None,
    planner_thread_options: ThreadOptions | Mapping[str, Any] | None = None,
    builder_instructions: str | None = None,
    qa_instructions: str | None = None,
    builder_thread_options: ThreadOptions | Mapping[str, Any] | None = None,
    qa_thread_options: ThreadOptions | Mapping[str, Any] | None = None,
    qa_computer: ComputerConfig[Any] | None = None,
    qa_start_command: str | None = None,
    qa_base_url: str | None = None,
    qa_ready_url: str | None = None,
    qa_startup_timeout_seconds: float | None = None,
    sandbox_mode: SandboxMode | None = None,
    working_directory: str | None = None,
    skip_git_repo_check: bool | None = None,
    resume_existing_run: bool | None = None,
    create_scratch_workspace: bool | None = None,
    scratch_workspace_prefix: str | None = None,
    artifact_dir_name: str | None = None,
    default_max_rounds: int | None = None,
    planner_on_stream: Callable[[CodexToolStreamEvent], MaybeAwaitable[None]] | None = None,
    builder_on_stream: Callable[[CodexToolStreamEvent], MaybeAwaitable[None]] | None = None,
    qa_on_stream: Callable[[CodexToolStreamEvent], MaybeAwaitable[None]] | None = None,
    is_enabled: bool | Callable[[RunContextWrapper[Any], Any], MaybeAwaitable[bool]] | None = None,
    failure_error_function: ToolErrorFunction | None | _UnsetType = _UNSET,
) -> FunctionTool:
    resolved_options = _coerce_tool_options(options)
    if name is not None:
        resolved_options.name = name
    if description is not None:
        resolved_options.description = description
    if planner_model is not None:
        resolved_options.planner_model = planner_model
    if planner_model_settings is not None:
        resolved_options.planner_model_settings = planner_model_settings
    if planner_instructions is not None:
        resolved_options.planner_instructions = planner_instructions
    if planner_thread_options is not None:
        resolved_options.planner_thread_options = planner_thread_options
    if builder_instructions is not None:
        resolved_options.builder_instructions = builder_instructions
    if qa_instructions is not None:
        resolved_options.qa_instructions = qa_instructions
    if builder_thread_options is not None:
        resolved_options.builder_thread_options = builder_thread_options
    if qa_thread_options is not None:
        resolved_options.qa_thread_options = qa_thread_options
    if qa_computer is not None:
        resolved_options.qa_computer = qa_computer
    if qa_start_command is not None:
        resolved_options.qa_start_command = qa_start_command
    if qa_base_url is not None:
        resolved_options.qa_base_url = qa_base_url
    if qa_ready_url is not None:
        resolved_options.qa_ready_url = qa_ready_url
    if qa_startup_timeout_seconds is not None:
        resolved_options.qa_startup_timeout_seconds = qa_startup_timeout_seconds
    if sandbox_mode is not None:
        resolved_options.sandbox_mode = sandbox_mode
    if working_directory is not None:
        resolved_options.working_directory = working_directory
    if skip_git_repo_check is not None:
        resolved_options.skip_git_repo_check = skip_git_repo_check
    if resume_existing_run is not None:
        resolved_options.resume_existing_run = resume_existing_run
    if create_scratch_workspace is not None:
        resolved_options.create_scratch_workspace = create_scratch_workspace
    if scratch_workspace_prefix is not None:
        resolved_options.scratch_workspace_prefix = scratch_workspace_prefix
    if artifact_dir_name is not None:
        resolved_options.artifact_dir_name = artifact_dir_name
    if default_max_rounds is not None:
        resolved_options.default_max_rounds = default_max_rounds
    if planner_on_stream is not None:
        resolved_options.planner_on_stream = planner_on_stream
    if builder_on_stream is not None:
        resolved_options.builder_on_stream = builder_on_stream
    if qa_on_stream is not None:
        resolved_options.qa_on_stream = qa_on_stream
    if is_enabled is not None:
        resolved_options.is_enabled = is_enabled
    if not isinstance(failure_error_function, _UnsetType):
        resolved_options.failure_error_function = failure_error_function

    if resolved_options.default_max_rounds < 1:
        raise UserError("Codex builder/QA tool default_max_rounds must be at least 1.")
    if resolved_options.qa_start_command and not (
        resolved_options.qa_ready_url or resolved_options.qa_base_url
    ):
        raise UserError(
            "Codex builder/QA tool qa_start_command requires qa_ready_url or qa_base_url."
        )
    if resolved_options.qa_startup_timeout_seconds <= 0:
        raise UserError("Codex builder/QA tool qa_startup_timeout_seconds must be positive.")

    tool_name = resolved_options.name or DEFAULT_HARNESS_TOOL_NAME
    tool_description = resolved_options.description or (
        "Plans, builds, and QA-checks a code task with a planner plus separate Codex builder and "
        "evaluator loops."
    )

    async def _invoke(
        ctx: ToolContext[Any],
        task: str,
        max_rounds: int | None = None,
        resume: bool | None = None,
    ) -> CodexBuilderQAToolResult:
        """Run a planner, Codex builder, and separate QA loop for a coding task."""

        result = await _run_builder_qa_harness(
            ctx=ctx,
            task=task,
            max_rounds=max_rounds or resolved_options.default_max_rounds,
            resume=resolved_options.resume_existing_run if resume is None else resume,
            options=resolved_options,
        )
        return result

    return function_tool(
        _invoke,
        name_override=tool_name,
        description_override=tool_description,
        failure_error_function=resolved_options.failure_error_function,
        is_enabled=resolved_options.is_enabled,
    )


def _coerce_tool_options(
    options: CodexBuilderQAToolOptions | Mapping[str, Any] | None,
) -> CodexBuilderQAToolOptions:
    if options is None:
        resolved = CodexBuilderQAToolOptions()
    elif isinstance(options, CodexBuilderQAToolOptions):
        resolved = options
    else:
        if not isinstance(options, Mapping):
            raise UserError(
                "Codex builder/QA tool options must be a CodexBuilderQAToolOptions or a mapping."
            )
        allowed = {field.name for field in dataclasses.fields(CodexBuilderQAToolOptions)}
        unknown = set(options.keys()) - allowed
        if unknown:
            raise UserError(f"Unknown Codex builder/QA tool option(s): {sorted(unknown)}")
        resolved = CodexBuilderQAToolOptions(**dict(options))

    resolved.planner_thread_options = coerce_thread_options(resolved.planner_thread_options)
    resolved.builder_thread_options = coerce_thread_options(resolved.builder_thread_options)
    resolved.qa_thread_options = coerce_thread_options(resolved.qa_thread_options)
    return resolved


async def _run_builder_qa_harness(
    *,
    ctx: ToolContext[Any],
    task: str,
    max_rounds: int,
    resume: bool,
    options: CodexBuilderQAToolOptions,
) -> CodexBuilderQAToolResult:
    workspace, created_scratch_workspace = _prepare_workspace(options)
    artifact_dir = workspace / options.artifact_dir_name
    state_path = harness_state_path(workspace, options.artifact_dir_name)
    state = _load_existing_state(state_path)
    if state is not None:
        if not resume:
            raise UserError(
                "Existing Codex builder/QA harness state found. Re-run with resume=True or use "
                "a different artifact_dir_name."
            )
        if state.task != task:
            raise UserError(
                "Existing Codex builder/QA harness state task does not match the requested task."
            )
    state_created_scratch_workspace = (
        state.created_scratch_workspace if state is not None else created_scratch_workspace
    )
    harness_context = _HarnessContext(
        codex_thread_id_planner=state.planner_thread_id if state is not None else None,
        codex_thread_id_builder=state.builder_thread_id if state is not None else None,
        codex_thread_id_qa=state.qa_thread_id if state is not None else None,
    )
    builder_reports = _load_round_reports(
        workspace,
        options.artifact_dir_name,
        "build_round_",
        BuildRoundReport,
    )
    qa_reports = _load_round_reports(
        workspace,
        options.artifact_dir_name,
        "qa_round_",
        EvaluationReport,
    )
    latest_feedback: EvaluationReport | None = qa_reports[-1] if qa_reports else None
    plan = _load_optional_model(plan_artifact_path(workspace, options.artifact_dir_name), BuildPlan)

    if state is not None and state.status == "completed":
        if plan is None:
            raise UserError("Codex builder/QA harness state is completed but plan.json is missing.")
        return _build_result(
            workspace=workspace,
            artifact_dir=artifact_dir,
            state_path=state_path,
            created_scratch_workspace=state_created_scratch_workspace,
            harness_context=harness_context,
            plan=plan,
            build_reports=builder_reports,
            qa_reports=qa_reports,
        )

    state = state or HarnessState(
        task=task,
        max_rounds=max_rounds,
        status="planning",
        current_round=None,
        created_scratch_workspace=created_scratch_workspace,
    )
    state.max_rounds = max_rounds
    planner_agent_kwargs: dict[str, Any] = {
        "name": "planner_agent",
        "instructions": options.planner_instructions,
        "model": options.planner_model,
        "output_type": BuildPlan,
    }
    planner_thread_options = cast(ThreadOptions | None, options.planner_thread_options)
    builder_thread_options = cast(ThreadOptions | None, options.builder_thread_options)
    qa_thread_options = cast(ThreadOptions | None, options.qa_thread_options)

    planner_model_settings = (
        dataclasses.replace(options.planner_model_settings, tool_choice="required")
        if options.planner_model_settings is not None
        else ModelSettings(tool_choice="required")
    )

    planner_agent = Agent(
        **planner_agent_kwargs,
        tools=[
            codex_tool(
                name="codex_planner",
                default_thread_options=_resolve_nested_thread_options(
                    planner_thread_options,
                    workspace=workspace,
                    sandbox_mode=options.sandbox_mode,
                    skip_git_repo_check=options.skip_git_repo_check,
                ),
                on_stream=options.planner_on_stream,
                use_run_context_thread_id=True,
            )
        ],
        model_settings=planner_model_settings,
    )
    builder_agent = Agent(
        name="generator_agent",
        instructions=options.builder_instructions,
        tools=[
            codex_tool(
                name="codex_builder",
                default_thread_options=_resolve_nested_thread_options(
                    builder_thread_options,
                    workspace=workspace,
                    sandbox_mode=options.sandbox_mode,
                    skip_git_repo_check=options.skip_git_repo_check,
                ),
                on_stream=options.builder_on_stream,
                use_run_context_thread_id=True,
            )
        ],
        model_settings=ModelSettings(tool_choice="required"),
        output_type=BuildRoundReport,
    )
    qa_tools: list[Any] = [
        codex_tool(
            name="codex_qa",
            default_thread_options=_resolve_nested_thread_options(
                qa_thread_options,
                workspace=workspace,
                sandbox_mode=options.sandbox_mode,
                skip_git_repo_check=options.skip_git_repo_check,
            ),
            on_stream=options.qa_on_stream,
            use_run_context_thread_id=True,
        )
    ]
    if options.qa_computer is not None:
        qa_tools.append(ComputerTool(computer=options.qa_computer))
    qa_agent = Agent(
        name="evaluator_agent",
        instructions=options.qa_instructions,
        tools=qa_tools,
        model=DEFAULT_MODEL if options.qa_computer is not None else None,
        model_settings=ModelSettings(tool_choice="required"),
        output_type=EvaluationReport,
    )

    if plan is None:
        _persist_harness_state(
            state_path,
            state,
            status="planning",
            current_round=None,
            harness_context=harness_context,
            final_verdict=None,
        )
        plan_result = await Runner.run(
            planner_agent,
            _build_planner_prompt(
                task=task,
                workspace=workspace,
                artifact_dir_name=options.artifact_dir_name,
            ),
            context=harness_context,
        )
        _accumulate_usage(ctx, plan_result.context_wrapper.usage)
        plan = plan_result.final_output_as(BuildPlan)
        _write_json(plan_artifact_path(workspace, options.artifact_dir_name), plan)

    next_round_number, run_builder_for_next_round = _resolve_resume_position(
        max_rounds=max_rounds,
        build_reports=builder_reports,
        qa_reports=qa_reports,
    )

    for round_number in range(next_round_number, max_rounds + 1):
        if run_builder_for_next_round:
            _persist_harness_state(
                state_path,
                state,
                status="building",
                current_round=round_number,
                harness_context=harness_context,
                final_verdict=None,
            )
            build_result = await Runner.run(
                builder_agent,
                _build_generator_prompt(
                    task=task,
                    workspace=workspace,
                    artifact_dir_name=options.artifact_dir_name,
                    plan=plan,
                    round_number=round_number,
                    max_rounds=max_rounds,
                    latest_feedback=latest_feedback,
                ),
                context=harness_context,
            )
            _accumulate_usage(ctx, build_result.context_wrapper.usage)
            build_report = build_result.final_output_as(BuildRoundReport)
            if len(builder_reports) >= round_number:
                builder_reports[round_number - 1] = build_report
            else:
                builder_reports.append(build_report)
            _write_json(
                build_artifact_path(workspace, options.artifact_dir_name, round_number),
                build_report,
            )
        else:
            build_report = builder_reports[round_number - 1]

        _persist_harness_state(
            state_path,
            state,
            status="qa",
            current_round=round_number,
            harness_context=harness_context,
            final_verdict=None,
        )
        qa_server: _QAServerProcess | None = None
        if options.qa_start_command is not None:
            qa_server = await _start_qa_server(
                workspace=workspace,
                artifact_dir_name=options.artifact_dir_name,
                round_number=round_number,
                start_command=options.qa_start_command,
                ready_url=options.qa_ready_url or options.qa_base_url,
                startup_timeout_seconds=options.qa_startup_timeout_seconds,
            )
        try:
            qa_result = await Runner.run(
                qa_agent,
                _build_qa_prompt(
                    task=task,
                    workspace=workspace,
                    artifact_dir_name=options.artifact_dir_name,
                    plan=plan,
                    round_number=round_number,
                    build_report=build_report,
                    browser_enabled=options.qa_computer is not None,
                    qa_base_url=options.qa_base_url,
                    qa_server_log_file=qa_server.log_path if qa_server is not None else None,
                ),
                context=harness_context,
            )
        finally:
            if qa_server is not None:
                await _stop_qa_server(qa_server)
        _accumulate_usage(ctx, qa_result.context_wrapper.usage)
        evaluation = qa_result.final_output_as(EvaluationReport)
        if len(qa_reports) >= round_number:
            qa_reports[round_number - 1] = evaluation
        else:
            qa_reports.append(evaluation)
        latest_feedback = evaluation
        _write_json(
            qa_artifact_path(workspace, options.artifact_dir_name, round_number),
            evaluation,
        )

        if evaluation.verdict == "pass":
            _persist_harness_state(
                state_path,
                state,
                status="completed",
                current_round=round_number,
                harness_context=harness_context,
                final_verdict=evaluation.verdict,
            )
            return _build_result(
                workspace=workspace,
                artifact_dir=artifact_dir,
                state_path=state_path,
                created_scratch_workspace=state_created_scratch_workspace,
                harness_context=harness_context,
                plan=plan,
                build_reports=builder_reports,
                qa_reports=qa_reports,
            )

        run_builder_for_next_round = True

    if not qa_reports:
        raise UserError("Codex builder/QA tool did not produce a QA report.")

    _persist_harness_state(
        state_path,
        state,
        status="completed",
        current_round=len(qa_reports),
        harness_context=harness_context,
        final_verdict=qa_reports[-1].verdict,
    )
    return _build_result(
        workspace=workspace,
        artifact_dir=artifact_dir,
        state_path=state_path,
        created_scratch_workspace=state_created_scratch_workspace,
        harness_context=harness_context,
        plan=plan,
        build_reports=builder_reports,
        qa_reports=qa_reports,
    )


def _prepare_workspace(options: CodexBuilderQAToolOptions) -> tuple[Path, bool]:
    if options.working_directory:
        workspace = Path(options.working_directory).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace, False
    if not options.create_scratch_workspace:
        raise UserError(
            "Codex builder/QA tool requires working_directory when "
            "create_scratch_workspace is False."
        )
    return _create_scratch_workspace(options), True


def _create_scratch_workspace(options: CodexBuilderQAToolOptions) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix=options.scratch_workspace_prefix))
    (workspace / "src" / "scratch_app").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)

    (workspace / "README.md").write_text(
        "# Scratch Workspace\n\nThis temporary workspace is used by the Codex builder / QA tool.\n",
        encoding="utf-8",
    )
    (workspace / "AGENTS.md").write_text(
        "# Local Instructions\n\n"
        "- Keep dependencies minimal.\n"
        "- Prefer Python standard library where reasonable.\n"
        "- Add or update pytest tests for shipped behavior.\n"
        "- Use ASCII unless a file already needs Unicode.\n",
        encoding="utf-8",
    )
    (workspace / ".gitignore").write_text(
        f"__pycache__/\n.pytest_cache/\n.venv/\n*.pyc\n{options.artifact_dir_name}/\n",
        encoding="utf-8",
    )
    (workspace / "pyproject.toml").write_text(
        "[project]\n"
        'name = "scratch-app"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.10"\n'
        "dependencies = []\n\n"
        "[build-system]\n"
        'requires = ["setuptools>=68"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
        'pythonpath = ["src"]\n',
        encoding="utf-8",
    )
    (workspace / "src" / "scratch_app" / "__init__.py").write_text("", encoding="utf-8")
    return workspace


def plan_artifact_path(workspace: Path, artifact_dir_name: str) -> Path:
    return workspace / artifact_dir_name / "plan.json"


def harness_state_path(workspace: Path, artifact_dir_name: str) -> Path:
    return workspace / artifact_dir_name / "state.json"


def build_artifact_path(workspace: Path, artifact_dir_name: str, round_number: int) -> Path:
    return workspace / artifact_dir_name / f"build_round_{round_number}.json"


def qa_artifact_path(workspace: Path, artifact_dir_name: str, round_number: int) -> Path:
    return workspace / artifact_dir_name / f"qa_round_{round_number}.json"


def qa_server_log_path(workspace: Path, artifact_dir_name: str, round_number: int) -> Path:
    return workspace / artifact_dir_name / f"qa_server_round_{round_number}.log"


def _write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _load_existing_state(path: Path) -> HarnessState | None:
    return _load_optional_model(path, HarnessState)


def _load_optional_model(path: Path, model_cls: type[ModelT]) -> ModelT | None:
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def _load_round_reports(
    workspace: Path,
    artifact_dir_name: str,
    prefix: str,
    model_cls: type[ModelT],
) -> list[ModelT]:
    reports: list[ModelT] = []
    round_number = 1
    while True:
        path = workspace / artifact_dir_name / f"{prefix}{round_number}.json"
        if not path.exists():
            break
        reports.append(model_cls.model_validate_json(path.read_text(encoding="utf-8")))
        round_number += 1
    return reports


def _persist_harness_state(
    path: Path,
    state: HarnessState,
    *,
    status: Literal["planning", "building", "qa", "completed"],
    current_round: int | None,
    harness_context: _HarnessContext,
    final_verdict: str | None,
) -> None:
    state.status = status
    state.current_round = current_round
    state.planner_thread_id = harness_context.codex_thread_id_planner
    state.builder_thread_id = harness_context.codex_thread_id_builder
    state.qa_thread_id = harness_context.codex_thread_id_qa
    state.final_verdict = final_verdict
    _write_json(path, state)


def _resolve_resume_position(
    *,
    max_rounds: int,
    build_reports: list[BuildRoundReport],
    qa_reports: list[EvaluationReport],
) -> tuple[int, bool]:
    if len(qa_reports) > len(build_reports):
        raise UserError("Codex builder/QA harness artifacts are inconsistent: QA exceeds build.")
    if len(build_reports) > max_rounds or len(qa_reports) > max_rounds:
        raise UserError("Codex builder/QA harness artifacts exceed the configured max_rounds.")
    if build_reports and len(build_reports) > len(qa_reports):
        return len(build_reports), False
    return len(qa_reports) + 1, True


def _build_result(
    *,
    workspace: Path,
    artifact_dir: Path,
    state_path: Path,
    created_scratch_workspace: bool,
    harness_context: _HarnessContext,
    plan: BuildPlan,
    build_reports: list[BuildRoundReport],
    qa_reports: list[EvaluationReport],
) -> CodexBuilderQAToolResult:
    if not qa_reports:
        raise UserError("Codex builder/QA tool did not produce a QA report.")
    return CodexBuilderQAToolResult(
        workspace=str(workspace),
        artifact_dir=str(artifact_dir),
        state_file=str(state_path),
        created_scratch_workspace=created_scratch_workspace,
        rounds_completed=len(qa_reports),
        final_verdict=qa_reports[-1].verdict,
        planner_thread_id=harness_context.codex_thread_id_planner,
        builder_thread_id=harness_context.codex_thread_id_builder,
        qa_thread_id=harness_context.codex_thread_id_qa,
        plan=plan,
        build_reports=build_reports,
        qa_reports=qa_reports,
    )


async def _start_qa_server(
    *,
    workspace: Path,
    artifact_dir_name: str,
    round_number: int,
    start_command: str,
    ready_url: str | None,
    startup_timeout_seconds: float,
) -> _QAServerProcess:
    log_path = qa_server_log_path(workspace, artifact_dir_name, round_number)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    process_kwargs: dict[str, Any] = {
        "cwd": str(workspace),
        "stdout": log_handle,
        "stderr": asyncio.subprocess.STDOUT,
    }
    if os.name != "nt":
        process_kwargs["start_new_session"] = True
    process = await asyncio.create_subprocess_shell(start_command, **process_kwargs)
    server = _QAServerProcess(process=process, log_path=log_path, log_handle=log_handle)
    try:
        if ready_url is not None:
            await _wait_for_http_ready(
                process=process,
                ready_url=ready_url,
                log_path=log_path,
                timeout_seconds=startup_timeout_seconds,
            )
        else:
            await asyncio.sleep(min(startup_timeout_seconds, 2.0))
        return server
    except Exception:
        await _stop_qa_server(server)
        raise


async def _stop_qa_server(server: _QAServerProcess) -> None:
    try:
        if server.process.returncode is None:
            if os.name != "nt" and server.process.pid is not None:
                with suppress(ProcessLookupError):
                    os.killpg(server.process.pid, signal.SIGTERM)
            else:
                server.process.terminate()
            try:
                await asyncio.wait_for(server.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                if os.name != "nt" and server.process.pid is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(server.process.pid, signal.SIGKILL)
                else:
                    server.process.kill()
                await server.process.wait()
    finally:
        server.log_handle.close()


async def _wait_for_http_ready(
    *,
    process: asyncio.subprocess.Process,
    ready_url: str,
    log_path: Path,
    timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        if process.returncode is not None:
            raise UserError(
                "QA start command exited with code "
                f"{process.returncode}. See server log: {log_path}"
            )
        if await asyncio.to_thread(_probe_ready_url, ready_url):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise UserError(
                f"Timed out waiting for QA server at {ready_url}. See server log: {log_path}"
            )
        await asyncio.sleep(0.5)


def _probe_ready_url(ready_url: str) -> bool:
    request = urllib.request.Request(ready_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2):
            return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def _resolve_nested_thread_options(
    configured: ThreadOptions | None,
    *,
    workspace: Path,
    sandbox_mode: SandboxMode | None,
    skip_git_repo_check: bool | None,
) -> ThreadOptions:
    resolved_sandbox_mode = (
        sandbox_mode
        if sandbox_mode is not None
        else configured.sandbox_mode
        if configured
        else "workspace-write"
    )
    resolved_skip_git_repo_check = (
        skip_git_repo_check
        if skip_git_repo_check is not None
        else configured.skip_git_repo_check
        if configured
        else True
    )
    resolved_web_search_enabled = (
        configured.web_search_enabled
        if configured and configured.web_search_enabled is not None
        else False
    )
    return ThreadOptions(
        model=configured.model if configured and configured.model else DEFAULT_MODEL,
        sandbox_mode=resolved_sandbox_mode,
        working_directory=str(workspace),
        skip_git_repo_check=resolved_skip_git_repo_check,
        model_reasoning_effort=(
            configured.model_reasoning_effort
            if configured and configured.model_reasoning_effort
            else DEFAULT_REASONING_EFFORT
        ),
        network_access_enabled=(
            configured.network_access_enabled
            if configured and configured.network_access_enabled is not None
            else False
        ),
        web_search_mode=(
            configured.web_search_mode if configured and configured.web_search_mode else "disabled"
        ),
        web_search_enabled=resolved_web_search_enabled,
        approval_policy=(
            configured.approval_policy if configured and configured.approval_policy else "never"
        ),
        additional_directories=configured.additional_directories if configured else None,
    )


def _accumulate_usage(ctx: ToolContext[Any], usage: Usage) -> None:
    ctx.usage.add(usage)


def _build_generator_prompt(
    *,
    task: str,
    workspace: Path,
    artifact_dir_name: str,
    plan: BuildPlan,
    round_number: int,
    max_rounds: int,
    latest_feedback: EvaluationReport | None,
) -> str:
    feedback_text = (
        latest_feedback.model_dump_json(indent=2)
        if latest_feedback
        else "No previous QA feedback. This is the first implementation round."
    )
    return (
        f"User request:\n{task}\n\n"
        f"Workspace:\n{workspace}\n\n"
        f"Plan artifact:\n{plan_artifact_path(workspace, artifact_dir_name)}\n\n"
        f"Round:\n{round_number} of {max_rounds}\n\n"
        "Build plan:\n"
        f"{plan.model_dump_json(indent=2)}\n\n"
        "Latest QA feedback:\n"
        f"{feedback_text}\n\n"
        "Use the codex_builder tool to implement the next highest-value improvements now. "
        "If previous QA feedback exists, prioritize fixing those issues before expanding scope. "
        "Keep the workspace in a runnable state."
    )


def _build_planner_prompt(
    *,
    task: str,
    workspace: Path,
    artifact_dir_name: str,
) -> str:
    return (
        f"User request:\n{task}\n\n"
        f"Workspace:\n{workspace}\n\n"
        f"Plan artifact path:\n{plan_artifact_path(workspace, artifact_dir_name)}\n\n"
        "Use the codex_planner tool to inspect the workspace if needed and produce a concrete "
        "BuildPlan for the builder and QA roles."
    )


def _build_qa_prompt(
    *,
    task: str,
    workspace: Path,
    artifact_dir_name: str,
    plan: BuildPlan,
    round_number: int,
    build_report: BuildRoundReport,
    browser_enabled: bool,
    qa_base_url: str | None,
    qa_server_log_file: Path | None,
) -> str:
    runtime_notes: list[str] = []
    if qa_base_url:
        runtime_notes.append(f"QA base URL:\n{qa_base_url}\n")
    if qa_server_log_file is not None:
        runtime_notes.append(f"QA server log:\n{qa_server_log_file}\n")
    if browser_enabled:
        runtime_notes.append(
            "Live browser QA:\nA computer tool is available for this round. Use it to interact "
            "with the running app before deciding the verdict.\n"
        )
    runtime_context = "\n".join(runtime_notes)
    return (
        f"User request:\n{task}\n\n"
        f"Workspace:\n{workspace}\n\n"
        f"Plan artifact:\n{plan_artifact_path(workspace, artifact_dir_name)}\n\n"
        "Builder report artifact:\n"
        f"{build_artifact_path(workspace, artifact_dir_name, round_number)}\n\n"
        f"Round:\n{round_number}\n\n"
        "Build plan:\n"
        f"{plan.model_dump_json(indent=2)}\n\n"
        "Latest builder report:\n"
        f"{build_report.model_dump_json(indent=2)}\n\n"
        f"{runtime_context}"
        "Use the codex_qa tool to inspect the workspace and decide whether the app is good enough "
        "to stop. If a computer tool is available, use it to verify live behavior before "
        "finalizing the report. Fail if core acceptance criteria are missing, broken, or "
        "unverified. Prefer concrete evidence such as commands run, files inspected, or "
        "observable behavior."
    )
