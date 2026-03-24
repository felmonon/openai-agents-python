from __future__ import annotations

import argparse
import asyncio
from datetime import datetime

from agents import Agent, ModelSettings, Runner, gen_trace_id, trace
from agents.extensions.experimental.codex import (
    CodexToolStreamEvent,
    ThreadErrorEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
    codex_builder_qa_tool,
)
from examples.auto_mode import input_with_fallback, is_auto_mode

DEFAULT_PROMPT = (
    "Build a tiny Python CLI todo manager with add, list, complete, and search commands. "
    "Store data in a local JSON file and add pytest coverage for the core flows."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the experimental Codex builder / QA tool through an agent."
    )
    parser.add_argument("--prompt", help="Short product request to build.")
    parser.add_argument(
        "--working-directory",
        help=(
            "Target workspace for the harness. If omitted, the tool creates a scratch workspace."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum builder / QA rounds. Defaults to 2, or 1 in auto mode.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    timestamp = _timestamp()
    lines = str(message).splitlines() or [""]
    for line in lines:
        print(f"{timestamp} {line}")


def make_stream_logger(label: str):
    async def _on_stream(payload: CodexToolStreamEvent) -> None:
        event = payload.event
        if isinstance(event, ThreadStartedEvent):
            log(f"{label}: Codex thread started: {event.thread_id}")
            return
        if isinstance(event, TurnStartedEvent):
            log(f"{label}: Codex turn started")
            return
        if isinstance(event, TurnCompletedEvent):
            log(f"{label}: Codex turn completed: {event.usage}")
            return
        if isinstance(event, TurnFailedEvent):
            log(f"{label}: Codex turn failed: {event.error.message}")
            return
        if isinstance(event, ThreadErrorEvent):
            log(f"{label}: Codex stream error: {event.message}")

    return _on_stream


async def main() -> None:
    args = parse_args()
    auto_mode = is_auto_mode()
    max_rounds = args.max_rounds if args.max_rounds is not None else (1 if auto_mode else 2)
    if max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1.")

    user_prompt = args.prompt or input_with_fallback(
        "What should the harness build? ",
        DEFAULT_PROMPT,
    )

    agent = Agent(
        name="Codex Builder QA Agent",
        instructions=(
            "Always use the codex_builder_qa tool to plan, build, and review the user's request. "
            "Return a concise summary of the final verdict, workspace, and artifact directory."
        ),
        tools=[
            codex_builder_qa_tool(
                working_directory=args.working_directory,
                create_scratch_workspace=args.working_directory is None,
                default_max_rounds=max_rounds,
                builder_on_stream=make_stream_logger("builder"),
                qa_on_stream=make_stream_logger("qa"),
            )
        ],
        model_settings=ModelSettings(tool_choice="required"),
    )

    trace_id = gen_trace_id()
    log(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")

    with trace("Codex builder / QA tool example", trace_id=trace_id):
        result = await Runner.run(
            agent,
            (
                "Use the codex_builder_qa tool to build this task: "
                f"{user_prompt}\nReturn the tool result summary."
            ),
        )
        log(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
