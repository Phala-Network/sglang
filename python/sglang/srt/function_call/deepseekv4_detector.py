import logging
from typing import Any, List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import StructuralTag
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification
    """

    # Complete calls are emitted once, with both name and arguments. Serving
    # must not synthesize argument deltas from the parent's pending state.
    emits_tool_calls_atomically = True

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
        self._atomic_internal_tool_count = 0
        self._atomic_wire_tool_count = 0

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """Emit only complete DSV4 calls.

        The V3.2 parser incrementally builds DSML arguments.  DSV4 requests
        can be cut off by an output budget or cancellation, so forwarding
        those intermediate fragments would expose a name-only or malformed
        call that cannot be retracted.  Keep the parent's parsing state, but
        publish one complete item only after ``</｜DSML｜invoke>`` arrives.
        """
        parsed = super().parse_streaming_increment(new_text, tools)

        # The parent resets its internal arrays after a parse error.  Start a
        # fresh internal ordinal while preserving already published wire
        # ordinals for this response.
        if self.current_tool_id < 0:
            if not self.prev_tool_call_arr:
                self._atomic_internal_tool_count = 0
            return StreamingParseResult(normal_text=parsed.normal_text)

        completed = min(self.current_tool_id, len(self.prev_tool_call_arr))
        known_tool_names = {
            tool.function.name
            for tool in tools
            if getattr(tool, "function", None) is not None and tool.function.name
        }
        calls = []
        while self._atomic_internal_tool_count < completed:
            state = self.prev_tool_call_arr[self._atomic_internal_tool_count]
            name = state.get("name")
            parameters = state.get("arguments") or "{}"
            self._atomic_internal_tool_count += 1

            if (
                name not in known_tool_names
                and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
            ):
                logger.warning("Model attempted to call undefined function: %s", name)
                continue

            calls.append(
                ToolCallItem(
                    tool_index=self._atomic_wire_tool_count,
                    name=name,
                    parameters=parameters,
                )
            )
            self._atomic_wire_tool_count += 1

        return StreamingParseResult(normal_text=parsed.normal_text, calls=calls)

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Drop any unclosed pending call and flush only ordinary text."""
        parsed = super().finish(tools)
        self._atomic_internal_tool_count = 0
        self._atomic_wire_tool_count = 0
        return StreamingParseResult(normal_text=parsed.normal_text)

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """Build DSV4's native grammar and honor the parallel-call policy.

        xgrammar's model helper does not accept ``parallel_tool_calls`` and
        currently emits ``stop_after_first=False`` even when the OpenAI
        request explicitly disables parallel calls.  Restrict only the
        tool-invocation list nodes in DSV4's structural format; the grammar
        still owns the native DSML wrapper and argument schemas.
        """
        structural_tag = super().get_structural_tag(
            tools=tools,
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )
        if structural_tag is None or parallel_tool_calls:
            return structural_tag

        payload = structural_tag.model_dump()

        def stop_after_first_tool(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "tags_with_separator":
                    node["stop_after_first"] = True
                for child in node.values():
                    stop_after_first_tool(child)
            elif isinstance(node, list):
                for child in node:
                    stop_after_first_tool(child)

        stop_after_first_tool(payload.get("format"))
        return StructuralTag.model_validate(payload)
