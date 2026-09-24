import json
import logging
import re
from typing import Any, List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import (
    BaseFormatDetector,
    StructuralTag,
)
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    coerce_argument_to_schema,
    get_argument_schema,
    get_schema_properties,
    infer_type_from_json_schema,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)


def _align_required_tool_call_repetition(
    structural_tag: StructuralTag, parallel_tool_calls: bool
) -> StructuralTag:
    """Match Qwen's trained separator and enforce the request's call limit."""
    value = structural_tag.model_dump()
    repetitions = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") in ("tags_with_separator", "triggered_tags"):
                repetitions.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    if len(repetitions) != 1:
        raise ValueError(
            "Qwen3 Coder required structural tag must contain one repeated-tag format"
        )

    repetition = repetitions[0]
    if repetition.get("type") == "triggered_tags":
        if not repetition.get("at_least_one") or repetition.get("triggers") != [
            "<tool_call>\n<function="
        ]:
            raise ValueError("Unexpected Qwen3 Coder required tool-call trigger")
        tags = repetition["tags"]
        repetition.clear()
        repetition.update(
            type="tags_with_separator",
            tags=tags,
            at_least_one=True,
        )
    repetition["separator"] = "\n"
    repetition["stop_after_first"] = not parallel_tool_calls
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

        # Only closed function bodies inside closed tool blocks are executable.
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=([^<>]+)>(.*?)</function>", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

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
        logger.warning("Tool '<redacted>' is not defined in the tools list.")
        return {}

    def _get_param_type(self, param_schema: Any) -> str:
        """Infer the parser conversion type from a JSON schema parameter."""
        # Const intersects union types; infer conversion from its JSON type,
        # without substituting the constant for the model's actual value.
        type_schema = (
            {"enum": [param_schema["const"]]}
            if isinstance(param_schema, dict) and "const" in param_schema
            else param_schema
        )
        inferred_type = infer_type_from_json_schema(type_schema)
        if inferred_type is None:
            return "string"
        return str(inferred_type).strip().lower()

    def _convert_param_value(
        self,
        param_value: str,
        param_name: str,
        param_config: dict,
        func_name: str,
        tools: Optional[List[Tool]] = None,
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        argument_schema = get_argument_schema(func_name, param_name, tools or [])
        if argument_schema is None:
            for tool in tools or []:
                if tool.function.name != func_name:
                    continue
                schema = tool.function.parameters
                if not isinstance(schema, dict) or schema.get("patternProperties"):
                    break
                if param_name in get_schema_properties(schema):
                    break
                additional = schema.get(
                    "additionalProperties", schema.get("unevaluatedProperties")
                )
                if isinstance(additional, dict):
                    argument_schema = dict(additional)
                    for key in ("$defs", "definitions"):
                        if key in schema and key not in argument_schema:
                            argument_schema[key] = schema[key]
                break
        if argument_schema is not None:
            converted, schema_valid = coerce_argument_to_schema(
                param_value, argument_schema
            )
            if schema_valid:
                return converted

        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    "Parsed parameter '<redacted>' is not defined in the tool parameters for tool '<redacted>', directly returning the string value."
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
                    "Parsed value '<redacted>' of parameter '<redacted>' is not an integer in tool '<redacted>', degenerating to string."
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
                    "Parsed value '<redacted>' of parameter '<redacted>' is not a float in tool '<redacted>', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    "Parsed value '<redacted>' of parameter '<redacted>' is not a boolean (`true` of `false`) in tool '<redacted>', degenerating to false."
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
                        "Parsed value '<redacted>' of parameter '<redacted>' cannot be parsed with json.loads in tool '<redacted>', will try other methods to parse it."
                    )
            try:
                param_value = safe_literal_eval(param_value)
            except Exception:
                logger.warning(
                    "Parsed value '<redacted>' of parameter '<redacted>' cannot be converted via Python `ast.literal_eval()` in tool '<redacted>', degenerating to string."
                )
            return param_value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Use the streaming block boundary rules without retaining request state."""
        detector = type(self)()
        parsed = detector.parse_streaming_increment(text, tools)
        tail = detector.finish(tools)
        return StreamingParseResult(
            normal_text=parsed.normal_text + tail.normal_text, calls=parsed.calls
        )

    def _parse_complete_block(self, text: str, tools: List[Tool]) -> List[ToolCallItem]:
        """Parse function bodies only after the outer block has closed."""
        calls = []
        try:
            raw_tool_calls = self.tool_call_regex.findall(text)
            known_names = {tool.function.name for tool in tools or []}

            tool_idx = 0
            for tool_content in raw_tool_calls:
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_name, params_str in funcs:
                    if (
                        func_name not in known_names
                        and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
                    ):
                        continue

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

        except Exception as e:
            logger.error("Error parsing complete Qwen tool block: <redacted>")
            return []
        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Publish complete blocks atomically; incomplete calls are never emitted."""
        self._buffer += new_text
        calls = []
        normal_text_chunks = []

        while self._buffer:
            start = self._buffer.find(self.tool_call_start_token)
            if start < 0:
                # Retain a suffix only if it could become the opening marker.
                keep = 0
                for size in range(
                    1, min(len(self._buffer), len(self.tool_call_start_token) - 1) + 1
                ):
                    if self.tool_call_start_token.startswith(self._buffer[-size:]):
                        keep = size
                normal_text_chunks.append(
                    self._buffer[:-keep] if keep else self._buffer
                )
                self._buffer = self._buffer[-keep:] if keep else ""
                break
            if start:
                normal_text_chunks.append(self._buffer[:start])
                self._buffer = self._buffer[start:]
            end = self._buffer.find(
                self.tool_call_end_token, len(self.tool_call_start_token)
            )
            if end < 0:
                break
            boundary = end + len(self.tool_call_end_token)
            block = self._buffer[:boundary]
            self._buffer = self._buffer[boundary:]
            for call in self._parse_complete_block(block, tools):
                self.current_tool_id += 1
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=call.name,
                        parameters=call.parameters,
                    )
                )

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

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
        structural_tag = super().get_structural_tag(
            tools=tools,
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )
        if structural_tag is None or tool_choice != "required":
            return structural_tag
        return _align_required_tool_call_repetition(
            structural_tag, parallel_tool_calls=parallel_tool_calls
        )

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
