import json
import logging
import re
from typing import Any, List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector, StructuralTag
from sglang.srt.function_call.schema_argument_coercion import (
    coerce_argument_to_schema,
    get_argument_schema,
)
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    get_schema_properties,
    infer_type_from_json_schema,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)


def _align_required_tool_call_repetition(
    structural_tag: StructuralTag, parallel_tool_calls: bool
) -> StructuralTag:
    """Permit the newline separator used between trained Qwen tool blocks."""
    value = structural_tag.model_dump()
    repetitions = []

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "tags_with_separator":
                repetitions.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    if len(repetitions) != 1:
        raise ValueError("Unexpected Qwen required-tool repetition structure")
    repetitions[0]["separator"] = "\n"
    repetitions[0]["stop_after_first"] = not parallel_tool_calls
    return StructuralTag.model_validate(value)


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict):
                    properties = get_schema_properties(params)
                    if properties or "properties" in params:
                        return properties
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _get_param_type(self, param_schema: Any) -> str:
        """Infer the parser conversion type from a JSON schema parameter."""
        inferred_type = infer_type_from_json_schema(param_schema)
        if inferred_type is None:
            return "string"
        return str(inferred_type).strip().lower()

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str,
        tools: Optional[List[Tool]] = None,
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        schema = get_argument_schema(func_name, param_name, tools or [])
        if schema is not None:
            converted, valid = coerce_argument_to_schema(param_value, schema)
            # Preserve invalid model text; never invent a schema's const value
            # or silently turn an invalid boolean into false.
            return converted if valid else param_value
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        param_type = self._get_param_type(param_config[param_name])
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = safe_literal_eval(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            raw_tool_calls = self.tool_call_regex.findall(text)

            tool_idx = 0
            for tool_content in raw_tool_calls:
                if self.function_end_token not in tool_content:
                    continue
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    if tools and not any(tool.function.name == func_name for tool in tools):
                        continue
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name, tools
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Publish only complete calls, so a truncated stream cannot expose one."""
        self._buffer += new_text
        calls = []
        normal = []
        if not hasattr(self, "_next_complete_tool_index"):
            self._next_complete_tool_index = 0

        def emit_text(value):
            if value and (not self._next_complete_tool_index or value.strip()):
                normal.append(value)

        while self._buffer:
            start = self._buffer.find(self.tool_call_start_token)
            if start < 0:
                # Hold only a suffix that could be the beginning of the marker.
                keep = 0
                for size in range(1, min(len(self._buffer), len(self.tool_call_start_token) - 1) + 1):
                    if self.tool_call_start_token.startswith(self._buffer[-size:]):
                        keep = size
                emit_text(self._buffer[:-keep] if keep else self._buffer)
                self._buffer = self._buffer[-keep:] if keep else ""
                break
            if start:
                emit_text(self._buffer[:start])
                self._buffer = self._buffer[start:]
            end = self._buffer.find(self.tool_call_end_token, len(self.tool_call_start_token))
            if end < 0:
                break
            boundary = end + len(self.tool_call_end_token)
            block = self._buffer[:boundary]
            self._buffer = self._buffer[boundary:]
            parsed = self.detect_and_parse(block, tools)
            for call in parsed.calls:
                calls.append(ToolCallItem(
                    tool_index=self._next_complete_tool_index,
                    name=call.name,
                    parameters=call.parameters,
                ))
                self._next_complete_tool_index += 1

        return StreamingParseResult(calls=calls, normal_text="".join(normal))

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        tail = self._buffer
        self._buffer = ""
        if tail.startswith(self.tool_call_start_token):
            return StreamingParseResult()
        return StreamingParseResult(normal_text=tail)

    def supports_structural_tag(self) -> bool:
        return True

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        tag = super().get_structural_tag(
            tools=tools, tool_choice=tool_choice, thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )
        if tag is None or tool_choice != "required":
            return tag
        return _align_required_tool_call_repetition(tag, parallel_tool_calls)

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
