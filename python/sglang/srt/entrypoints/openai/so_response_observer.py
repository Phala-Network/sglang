"""Opt-in, bounded final OpenAI SO observations; never changes wire output."""

import codecs
import copy
import io
import json
import math
from contextvars import ContextVar

from jsonschema import Draft202012Validator, validators
from referencing import Registry
from referencing.exceptions import NoSuchResource

MAX_CONTENT_CHARS = 65536  # Total across all choices, not per choice.
MAX_SCHEMA_CHARS = 16384
MAX_NODES = 2048
MAX_DEPTH = 32
MAX_CHOICES = 8
MAX_EVENTS = 8192
MAX_FRAME_CHARS = 131072
MAX_OPERATIONS = 8192

FORMATS = frozenset(("json_schema", "json_object"))
OUTCOMES = frozenset(
    ("valid", "json_invalid", "schema_invalid", "truncated", "unknown", "aborted")
)
FINISHES = frozenset(("stop", "length", "tool_calls", "abort", "other", "missing", "mixed"))
_budget = ContextVar("openai_so_validation_budget", default=None)


class _Limit(Exception):
    pass


def _bounded(value, max_chars):
    stack = [(value, 0)]
    nodes = chars = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise _Limit()
        if isinstance(item, str):
            chars += len(item)
        elif isinstance(item, dict):
            if nodes + len(stack) + len(item) > MAX_NODES:
                raise _Limit()
            chars += sum(len(k) for k in item)
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            if nodes + len(stack) + len(item) > MAX_NODES:
                raise _Limit()
            stack.extend((v, depth + 1) for v in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise _Limit()
        if chars > max_chars:
            raise _Limit()


# Regex, format, uniqueItems, dynamic refs and unevaluated* are deliberately
# unknown. There is no thread timeout pretending to interrupt atomic validators.
_CHILD_MAPS = frozenset(("properties", "$defs", "dependentSchemas"))
_CHILD_LISTS = frozenset(("allOf", "anyOf", "oneOf", "prefixItems"))
_CHILD_SINGLE = frozenset(
    ("additionalProperties", "items", "not", "if", "then", "else", "propertyNames")
)
_SUPPORTED = (
    _CHILD_MAPS | _CHILD_LISTS | _CHILD_SINGLE
    | frozenset((
        "$schema", "$ref", "type", "enum", "const", "required",
        "minProperties", "maxProperties", "minItems", "maxItems",
        "minLength", "maxLength", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "dependentRequired",
        "title", "description", "$comment", "default", "examples",
        "readOnly", "writeOnly", "deprecated",
    ))
)


def _prepare_schema(schema):
    _bounded(schema, MAX_SCHEMA_CHARS)
    schema = copy.deepcopy(schema)
    stack = [schema]
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if not isinstance(item, dict) or not set(item) <= _SUPPORTED:
            raise _Limit()
        if "$schema" in item and item["$schema"] not in (
            "https://json-schema.org/draft/2020-12/schema",
            "https://json-schema.org/draft/2020-12/schema#",
        ):
            raise _Limit()
        # Keep jsonschema.evolve on the budgeted class. The only admitted
        # dialect is the same Draft 2020-12; removing its annotation on this
        # private copy changes no validation assertion or response byte.
        item.pop("$schema", None)
        ref = item.get("$ref")
        if ref is not None and (
            not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/"))
        ):
            raise _Limit()
        for key in _CHILD_MAPS & item.keys():
            stack.extend(item[key].values())
        for key in _CHILD_LISTS & item.keys():
            stack.extend(item[key])
        stack.extend(item[k] for k in _CHILD_SINGLE & item.keys())
    Draft202012Validator.check_schema(schema)
    return schema


class _Work:
    def __init__(self):
        self.left = MAX_OPERATIONS
        self.depth = 0

    def charge(self, amount=1):
        self.left -= amount
        if self.left < 0 or self.depth > MAX_DEPTH:
            raise _Limit()


def _guard(original):
    def check(validator, constraint, instance, schema):
        work = _budget.get()
        work.charge()
        # Charge atomic collection scans too; no regex-bearing keyword is used.
        if isinstance(constraint, (list, dict)):
            work.charge(len(constraint))
        if isinstance(instance, (list, dict)) and isinstance(constraint, (list, dict)):
            work.charge(len(instance) * len(constraint))
        work.depth += 1
        try:
            work.charge()
            yield from original(validator, constraint, instance, schema)
        finally:
            work.depth -= 1
    return check


_Validator = validators.extend(
    Draft202012Validator,
    {name: _guard(fn) for name, fn in Draft202012Validator.VALIDATORS.items()},
)


def _deny_resource(uri):
    raise NoSuchResource(ref=uri)


def _reject_constant(value):
    raise ValueError("Non-JSON numeric constant")


class SOResponseObserver:
    """One request, one fixed-label observation, no retained payload after finish."""

    def __init__(self, request, emit):
        self.format = request.response_format.type
        self.streaming = bool(request.stream)
        self.emit = emit
        self.n = request.n
        self.unknown = not (type(self.n) is int and 1 <= self.n <= MAX_CHOICES)
        if self.unknown:
            self.n = 0
        self.schema = None
        if self.format == "json_schema":
            try:
                self.schema = _prepare_schema(request.response_format.json_schema.schema_)
            except Exception:
                self.unknown = True
        self.choices = {}
        self.content_chars = 0
        self.events = 0
        self.chunks = 0
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.data_lines = []
        self.frame_chars = 0
        self.event_type = ""
        self.done = False
        self.finished = False

    @classmethod
    def create(cls, request, emit):
        response_format = getattr(request, "response_format", None)
        if getattr(response_format, "type", None) not in FORMATS:
            return None
        return cls(request, emit)

    def _choice(self, row, streaming):
        index = row.get("index")
        if type(index) is not int or not 0 <= index < min(self.n, MAX_CHOICES):
            self.unknown = True
            return
        state = self.choices.setdefault(index, {"text": io.StringIO(), "finish": None})
        message = row.get("delta" if streaming else "message") or {}
        if not isinstance(message, dict):
            self.unknown = True
            return
        content = message.get("content")
        if message.get("tool_calls") or message.get("refusal"):
            self.unknown = True
        if content is not None:
            if not isinstance(content, str) or state["finish"] is not None:
                self.unknown = True
            elif not self.unknown:
                if self.content_chars + len(content) > MAX_CONTENT_CHARS:
                    self.unknown = True
                    self._clear_content()
                else:
                    self.content_chars += len(content)
                    state["text"].write(content)
        finish = row.get("finish_reason")
        if finish is not None:
            if state["finish"] is not None:
                self.unknown = True
            state["finish"] = finish if finish in FINISHES - {"mixed", "missing"} else "other"

    def _clear_content(self):
        for state in self.choices.values():
            state["text"].close()
            state["text"] = io.StringIO()

    def response(self, response):
        try:
            choices = getattr(response, "choices", None)
            if choices is None:
                self.unknown = True
            else:
                for row in choices[:MAX_CHOICES + 1]:
                    self._choice(
                        {"index": row.index, "message": {
                            "content": row.message.content,
                            "tool_calls": row.message.tool_calls,
                            "refusal": getattr(row.message, "refusal", None),
                        }, "finish_reason": row.finish_reason},
                        False,
                    )
                if len(choices) != self.n:
                    self.unknown = True
            self.done = True
        except Exception:
            self.unknown = True
        self.finish()

    def feed(self, chunk):
        self.chunks += 1
        if self.events > MAX_EVENTS or self.chunks > MAX_EVENTS:
            self.unknown = True
            self.pending = ""
            self.data_lines.clear()
            return
        try:
            if isinstance(chunk, bytes) and len(chunk) <= MAX_FRAME_CHARS:
                chunk = self.decoder.decode(chunk)
            if not isinstance(chunk, str) or len(chunk) > MAX_FRAME_CHARS:
                self.unknown = True
                return
            self.pending += chunk
            if len(self.pending) > MAX_FRAME_CHARS:
                self.unknown = True
                self.pending = ""
                return
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                line = line.removesuffix("\r")
                if not line:
                    self._event()
                    if self.events > MAX_EVENTS:
                        self.pending = ""
                        return
                elif line.startswith("data:"):
                    value = line[5:].removeprefix(" ")
                    self.frame_chars += len(value) + 1
                    if self.frame_chars > MAX_FRAME_CHARS:
                        self.unknown = True
                        self.data_lines.clear()
                    else:
                        self.data_lines.append(value)
                elif line.startswith("event:"):
                    self.event_type = line[6:].strip()
        except Exception:
            self.unknown = True

    def _event(self):
        self.events += 1
        data = "\n".join(self.data_lines)
        event_type = self.event_type
        self.data_lines.clear()
        self.frame_chars = 0
        self.event_type = ""
        if self.events > MAX_EVENTS:
            self.unknown = True
            return
        if not data or event_type not in ("", "message"):
            return
        if data == "[DONE]":
            if self.done:
                self.unknown = True
            self.done = True
            return
        if self.done:
            self.unknown = True
            return
        try:
            obj = json.loads(data, parse_constant=_reject_constant)
            if "error" in obj:
                self.unknown = True
            choices = obj.get("choices", [])
            if not isinstance(choices, list) or len(choices) > MAX_CHOICES:
                self.unknown = True
                return
            for row in choices:
                self._choice(row, True)
        except Exception:
            self.unknown = True

    def _validate(self, content):
        try:
            value = json.loads(content, parse_constant=_reject_constant)
        except (ValueError, TypeError):
            return "json_invalid"
        except Exception:
            return "unknown"
        try:
            _bounded(value, MAX_CONTENT_CHARS)
            if self.format == "json_object":
                return "valid" if isinstance(value, dict) else "schema_invalid"
            token = _budget.set(_Work())
            try:
                validator = _Validator(self.schema, registry=Registry(retrieve=_deny_resource))
                return "schema_invalid" if next(validator.iter_errors(value), None) else "valid"
            finally:
                _budget.reset(token)
        except Exception:
            return "unknown"

    def finish(self, aborted=False):
        if self.finished:
            return
        self.finished = True
        finish_values = {s["finish"] or "missing" for s in self.choices.values()}
        finish = next(iter(finish_values)) if len(finish_values) == 1 else (
            "mixed" if finish_values else "missing"
        )
        if aborted or "abort" in finish_values:
            outcome = "aborted"
        elif "length" in finish_values:
            outcome = "truncated"
        elif self.unknown or self.decoder.getstate()[0] or not self.done or self.pending or self.data_lines or len(self.choices) != self.n or finish != "stop":
            outcome = "unknown"
        else:
            try:
                results = {self._validate(s["text"].getvalue()) for s in self.choices.values()}
                outcome = next((v for v in ("unknown", "json_invalid", "schema_invalid") if v in results), "valid")
            except Exception:
                outcome = "unknown"
        self._clear_content()
        self.schema = None
        self.pending = ""
        self.data_lines.clear()
        try:
            self.emit(self.format, self.streaming, outcome, finish)
        except Exception:
            # Metrics emission must not turn successful serving into a failure.
            pass


class ObservedSOStream:
    """Transparent owner whose aclose also handles a never-started body iterator."""

    def __init__(self, source, observer):
        self.source = source.__aiter__()
        self.observer = observer

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.source.__anext__()
        except StopAsyncIteration:
            self.observer.finish()
            raise
        except BaseException:
            self.observer.finish(aborted=True)
            raise
        self.observer.feed(chunk)
        return chunk

    async def aclose(self):
        try:
            await self.source.aclose()
        except BaseException:
            self.observer.finish(aborted=True)
            raise
        finally:
            self.observer.finish(aborted=not self.observer.done)
