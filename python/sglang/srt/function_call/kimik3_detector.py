import json
import logging
import re
from typing import List, Literal, Optional, Union

from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
    partial_suffix_len,
    strip_partial_marker_suffix,
    strip_response_wrappers,
)
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)

logger = logging.getLogger(__name__)

_CALL_OPEN_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>",
    re.DOTALL,
)
_ARG_OPEN_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>",
    re.DOTALL,
)
_CALL_BOUNDARY_RE = re.compile(
    "|".join(map(re.escape, ("<|open|>argument", CALL_CLOSE, TOOLS_CLOSE)))
)
_ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')
_JSON_DECODER = json.JSONDecoder()


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(attrs: str) -> dict:
    return {m["k"]: _unescape_attr(m["v"]) for m in _ATTR_RE.finditer(attrs)}


class KimiK3Detector(BaseFormatDetector):
    """Detector for the Kimi K3 XTML tool-call format.

    K3 emits tool calls in a ``tools`` channel built from dedicated special
    tokens; the plain reply lives in a preceding ``response`` channel:

    ```
    <|open|>response<|sep|>text<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="k" type="string"<|sep|>raw text<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>
    ```

    ``type="string"`` argument values are raw text. ``type="json-string"``
    explicitly carries a JSON string; the other native types carry JSON values.
    Attribute values reverse the template's ``&amp;``/``&quot;`` escaping.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._sent_normal_idx = 0

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Kimi K3 uses its model-native structural tag implementation"
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_kimik3_auto_tool_call_structural_tag(
            tools or [],
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:
        return get_kimik3_structural_tag(
            tools=tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _parse_argument(self, text: str, start: int) -> tuple[str, object, int] | None:
        arg = _ARG_OPEN_RE.match(text, start)
        if arg is None:
            return None
        attrs = _parse_attrs(arg["attrs"])
        value_start = arg.end()
        wire_type = attrs.get("type", "string")
        if wire_type not in {
            "string",
            "json-string",
            "number",
            "boolean",
            "array",
            "object",
            "null",
        }:
            return None
        if wire_type == "string":
            # Raw XTML strings end at the first argument delimiter. Enclosing
            # call/tools markers inside that span are ordinary string values.
            end = text.find(ARGUMENT_CLOSE, value_start)
            if end == -1:
                return None
            value = text[value_start:end]
        else:
            # Typed values are JSON. A delimiter inside a quoted nested value
            # must not end the argument before the entire JSON value completes.
            while value_start < len(text) and text[value_start] in " \t\r\n":
                value_start += 1
            try:
                value, end = _JSON_DECODER.raw_decode(text, value_start)
            except json.JSONDecodeError:
                return None
            if wire_type == "json-string" and not isinstance(value, str):
                return None
            while end < len(text) and text[end] in " \t\r\n":
                end += 1
            if not text.startswith(ARGUMENT_CLOSE, end):
                return None
        return attrs.get("key", ""), value, end + len(ARGUMENT_CLOSE)

    def _parse_calls(self, section: str) -> List[dict]:
        calls = []
        pos = 0
        while call := _CALL_OPEN_RE.search(section, pos):
            tools_end = section.find(TOOLS_CLOSE, pos)
            if tools_end != -1 and tools_end < call.start():
                break
            pos = call.end()
            arguments = {}
            while boundary := _CALL_BOUNDARY_RE.search(section, pos):
                if section[pos : boundary.start()].strip():
                    return calls
                marker = boundary.group()
                if marker == TOOLS_CLOSE:
                    return calls
                if marker == CALL_CLOSE:
                    tool_name = _parse_attrs(call["attrs"]).get("tool", "")
                    if tool_name:
                        calls.append(
                            {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            }
                        )
                    pos = boundary.end()
                    break
                argument = self._parse_argument(section, boundary.start())
                if argument is None:
                    return calls
                key, value, pos = argument
                arguments[key] = value
            else:
                break
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        open_idx = text.find(self.bot_token)
        if open_idx == -1:
            return StreamingParseResult(normal_text=strip_response_wrappers(text))
        # Computed outside the try so the error path can reuse it instead of
        # falling back to raw text, which would ship the XTML tools markup to
        # the client.
        before = strip_response_wrappers(text[:open_idx])
        try:
            section_start = open_idx + len(self.bot_token)
            section = text[section_start:]
            calls = [
                ToolCallItem(
                    tool_index=i,
                    name=call["name"],
                    parameters=call["arguments"],
                )
                for i, call in enumerate(self._parse_calls(section))
            ]
            return StreamingParseResult(normal_text=before, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi K3 detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=before)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            open_idx = self._buffer.find(self.bot_token)
            if open_idx == -1:
                return StreamingParseResult(normal_text=self._emit_normal_text())

            normal_text = self._emit_normal_text(limit=open_idx)
            section = self._buffer[open_idx + len(self.bot_token) :]
            calls = []
            parsed = self._parse_calls(section)
            for call in parsed[self.current_tool_id + 1 :]:
                self.current_tool_id += 1
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": call["name"],
                    "arguments": json.loads(call["arguments"]),
                }
                self.streamed_args_for_tool[self.current_tool_id] = call["arguments"]
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=call["name"],
                        parameters=call["arguments"],
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(
                "Error in Kimi K3 parse_streaming_increment: %s", e, exc_info=True
            )
            # _sent_normal_idx indexes into _buffer, so it must be reset with it;
            # otherwise every later _emit_normal_text sees limit <= _sent_normal_idx
            # and silently drops the rest of the response.
            self._buffer = ""
            self._sent_normal_idx = 0
            return StreamingParseResult()

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        open_idx = self._buffer.find(self.bot_token)
        if open_idx != -1:
            section = self._buffer[open_idx + len(self.bot_token) :]
            if not self._parse_calls(section):
                logger.warning(
                    "Kimi K3 tools section ended with no complete tool call; "
                    "dropping %d buffered chars",
                    len(section),
                )
            return StreamingParseResult()
        pending = self._emit_normal_text(limit=len(self._buffer))
        return StreamingParseResult(normal_text=strip_partial_marker_suffix(pending))

    def _emit_normal_text(self, limit: int | None = None) -> str:
        if limit is None:
            holdback = partial_suffix_len(
                self._buffer,
                [self.bot_token, RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE],
            )
            limit = len(self._buffer) - holdback
        if limit <= self._sent_normal_idx:
            return ""
        pending = self._buffer[self._sent_normal_idx : limit]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            if marker in pending:
                pending = pending.replace(marker, "")
        self._sent_normal_idx = limit
        return pending
