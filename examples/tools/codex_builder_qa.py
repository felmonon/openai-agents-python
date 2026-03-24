from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime
from typing import Any, Literal

from agents import (
    Agent,
    AsyncComputer,
    Button,
    ComputerProvider,
    ModelSettings,
    RunContextWrapper,
    Runner,
    gen_trace_id,
    trace,
)
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


class PlaywrightQABrowser(AsyncComputer):
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    @property
    def environment(self) -> Literal["browser"]:
        return "browser"

    @property
    def dimensions(self) -> tuple[int, int]:
        return (1280, 800)

    async def open(self) -> PlaywrightQABrowser:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[f"--window-size={self.dimensions[0]},{self.dimensions[1]}"],
        )
        self._page = await self._browser.new_page()
        await self._page.set_viewport_size(
            {"width": self.dimensions[0], "height": self.dimensions[1]}
        )
        await self._page.goto(self.base_url)
        return self

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    @property
    def page(self) -> Any:
        if self._page is None:
            raise RuntimeError("Playwright browser page is not initialized.")
        return self._page

    async def screenshot(self) -> str:
        png_bytes = await self.page.screenshot(full_page=False)
        return base64.b64encode(png_bytes).decode("utf-8")

    async def click(self, x: int, y: int, button: Button = "left") -> None:
        mapped_button: Literal["left", "middle", "right"] = "left"
        if button == "right":
            mapped_button = "right"
        await self.page.mouse.click(x, y, button=mapped_button)

    async def double_click(self, x: int, y: int) -> None:
        await self.page.mouse.dblclick(x, y)

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self.page.mouse.move(x, y)
        await self.page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

    async def type(self, text: str) -> None:
        await self.page.keyboard.type(text)

    async def wait(self) -> None:
        await asyncio.sleep(1)

    async def move(self, x: int, y: int) -> None:
        await self.page.mouse.move(x, y)

    async def keypress(self, keys: list[str]) -> None:
        for key in keys:
            await self.page.keyboard.down(key)
        for key in reversed(keys):
            await self.page.keyboard.up(key)

    async def drag(self, path: list[tuple[int, int]]) -> None:
        if not path:
            return
        await self.page.mouse.move(path[0][0], path[0][1])
        await self.page.mouse.down()
        for px, py in path[1:]:
            await self.page.mouse.move(px, py)
        await self.page.mouse.up()


def make_browser_provider(base_url: str) -> ComputerProvider[PlaywrightQABrowser]:
    async def create_computer(*, run_context: RunContextWrapper[Any]) -> PlaywrightQABrowser:
        del run_context
        return await PlaywrightQABrowser(base_url).open()

    async def dispose_computer(
        *,
        run_context: RunContextWrapper[Any],
        computer: PlaywrightQABrowser,
    ) -> None:
        del run_context
        await computer.close()

    return ComputerProvider[PlaywrightQABrowser](
        create=create_computer,
        dispose=dispose_computer,
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
    parser.add_argument(
        "--qa-start-command",
        help="Optional command to start the app before each QA round, for example `npm run dev`.",
    )
    parser.add_argument(
        "--qa-base-url",
        help="Optional base URL for live QA, for example `http://127.0.0.1:4173`.",
    )
    parser.add_argument(
        "--browser-qa",
        action="store_true",
        help="Enable Playwright-backed browser QA against --qa-base-url.",
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
    if args.browser_qa and not args.qa_base_url:
        raise ValueError("--browser-qa requires --qa-base-url.")

    user_prompt = args.prompt or input_with_fallback(
        "What should the harness build? ",
        DEFAULT_PROMPT,
    )
    qa_computer = make_browser_provider(args.qa_base_url) if args.browser_qa else None

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
                qa_computer=qa_computer,
                qa_start_command=args.qa_start_command,
                qa_base_url=args.qa_base_url,
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
