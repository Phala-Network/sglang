"""Print a frozen finite protocol manifest; never sends requests or reads credentials."""

import base64
import copy
import hashlib
import json
import random
import struct
import sys
import zlib

MODEL = "deepseek/deepseek-v4.1-flash"
ENDPOINT = "http://127.0.0.1:18301/v1/chat/completions"
SEED = 20260919
CITIES = "Osaka Cordoba Tallinn Utrecht Katowice Bergen Cebu Hobart Lyon Kigali Bilbao Nagoya".split()
TEMPLATE = "Think about it carefully, step by step, and reason through what you know about {city} before answering. Give facts about {city}. [r{number}]"
CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "population": {"type": "integer"},
        "notable": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["city", "country", "population", "notable"],
    "additionalProperties": False,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def wire(value):
    """Preserve caller ordering on the wire; canonical() is only for hashes."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict(name, schema):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def body(prompt, *, budget=768, stream=False, **kwargs):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": budget,
        "stream": stream,
        "reasoning_effort": "none",
        **kwargs,
    }


def tool(name, schema, description):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": schema,
        },
    }


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def red_png():
    def chunk(kind, value):
        return (
            struct.pack(">I", len(value))
            + kind
            + value
            + struct.pack(">I", zlib.crc32(kind + value) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + b"\xff\0\0" * 64 for _ in range(64))
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(image).decode()


def manifest():
    cases = []

    def add(case_id, group, request, expect, source, **meta):
        cases.append(
            {
                "case_id": case_id,
                "group": group,
                "body": request,
                "body_sha256": sha(request),
                "body_wire_sha256": hashlib.sha256(wire(request).encode()).hexdigest(),
                "expect": expect,
                "source": source,
                **meta,
            }
        )

    base = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 2000,
        "reasoning_effort": "none",
        "response_format": strict("city_facts", CITY_SCHEMA),
    }
    for city, number in (("Osaka", 217058), ("Cordoba", 594224), ("Tallinn", 646439)):
        for repeat in range(1, 6):
            req = copy.deepcopy(base)
            req["messages"] = [
                {"role": "user", "content": TEMPLATE.format(city=city, number=number)}
            ]
            add(
                f"fixed-{city.lower()}-{repeat}",
                "conflict35",
                req,
                {"kind": "json", "schema": CITY_SCHEMA, "reasoning": "off"},
                "reasoning-off-prompt-conflict/fixed",
                city=city,
                suffix=f"r{number}",
                cohort="fixed15",
            )
    rng = random.Random(SEED)
    for index in range(1, 21):
        city, number = rng.choice(CITIES), rng.randint(1, 999999)
        req = copy.deepcopy(base)
        req["messages"] = [
            {"role": "user", "content": TEMPLATE.format(city=city, number=number)}
        ]
        add(
            f"random-{index:02d}",
            "conflict35",
            req,
            {"kind": "json", "schema": CITY_SCHEMA, "reasoning": "off"},
            "reasoning-off-prompt-conflict/random",
            city=city,
            suffix=f"r{number}",
            cohort="random20",
        )

    add(
        "chat-basic",
        "p0",
        body("Reply with exactly READY.", budget=64),
        {"kind": "text", "text": "READY", "reasoning": "off"},
        "corpus/Tier0",
    )
    add(
        "chat-sse",
        "p0",
        body("Reply with exactly STREAM_READY.", budget=64, stream=True),
        {"kind": "text", "text": "STREAM_READY", "reasoning": "off"},
        "corpus/Tier0",
    )
    weather_schema = object_schema(
        {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        }
    )
    weather = tool("get_weather", weather_schema, "Get current weather for a city")
    weather_expect = {
        "kind": "tools",
        "calls": [
            {
                "name": "get_weather",
                "arguments": {"location": "Paris", "unit": "celsius"},
            }
        ],
        "reasoning": "off",
    }
    for choice in ("auto", "required"):
        req = body(
            "Use get_weather to obtain the current weather in Paris using celsius.",
            tools=[weather],
            tool_choice=choice,
        )
        req["messages"].insert(
            0,
            {
                "role": "system",
                "content": "Use the supplied tool. Do not invent tool names.",
            },
        )
        add("weather-" + choice, "p0", req, weather_expect, "corpus/TC-P0-01,02")
    req = copy.deepcopy(cases[-2]["body"])
    req["stream"] = True
    add("weather-sse", "p0", req, weather_expect, "corpus/TC-P0-05")
    coordinate_schema = {
        "$defs": {
            "Location": object_schema(
                {"lat": {"type": "number"}, "lon": {"type": "number"}}
            )
        },
        **object_schema({"location": {"$ref": "#/$defs/Location"}}),
    }
    coordinate = tool(
        "get_weather_by_coordinate",
        coordinate_schema,
        "Get weather at the supplied coordinates",
    )
    add(
        "named-defs",
        "p0",
        body(
            "Call get_weather_by_coordinate with location lat=48.8566 and lon=2.3522.",
            tools=[coordinate],
            tool_choice={
                "type": "function",
                "function": {"name": coordinate["function"]["name"]},
            },
        ),
        {
            "kind": "tools",
            "calls": [
                {
                    "name": coordinate["function"]["name"],
                    "arguments": {"location": {"lat": 48.8566, "lon": 2.3522}},
                }
            ],
            "reasoning": "off",
        },
        "corpus/TC-P0-03",
    )
    search = tool(
        "search_catalog",
        object_schema(
            {"query": {"type": "string"}, "max_results": {"type": "integer"}}
        ),
        "Search the product catalog",
    )
    add(
        "parallel-catalog",
        "p0",
        body(
            'Make two independent catalog searches in parallel: one for "wireless keyboard" with 3 results and one for "USB-C dock" with 5 results.',
            budget=1024,
            tools=[search],
            tool_choice="required",
            parallel_tool_calls=True,
        ),
        {
            "kind": "tools",
            "calls": [
                {
                    "name": "search_catalog",
                    "arguments": {"query": "wireless keyboard", "max_results": 3},
                },
                {
                    "name": "search_catalog",
                    "arguments": {"query": "USB-C dock", "max_results": 5},
                },
            ],
            "reasoning": "off",
        },
        "corpus/TC-P0-04",
    )
    history = [
        {"role": "user", "content": "What is the current weather in Paris in celsius?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": canonical(
                            {"location": "Paris", "unit": "celsius"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_1",
            "content": canonical(
                {
                    "location": "Paris",
                    "temperature": 21,
                    "unit": "celsius",
                    "conditions": "sunny",
                }
            ),
        },
    ]
    req = body("unused", tools=[weather], tool_choice="auto")
    req["messages"] = [
        {
            "role": "system",
            "content": "Use available tool results to answer. Call tools only when information is missing.",
        }
    ] + history
    add(
        "tool-result-final",
        "p0",
        req,
        {"kind": "text", "contains": ["21"], "reasoning": "off"},
        "protocol-repairs/tool-result-history",
    )
    req = copy.deepcopy(req)
    req["messages"].append(
        {
            "role": "user",
            "content": "Now get the current weather in London in celsius; the Paris result does not provide London's weather.",
        }
    )
    add(
        "legitimate-second-tool",
        "p0",
        req,
        {
            "kind": "tools",
            "calls": [
                {
                    "name": "get_weather",
                    "arguments": {"location": "London", "unit": "celsius"},
                }
            ],
            "reasoning": "off",
        },
        "protocol-repairs/legitimate-second-call",
    )
    union_schema = object_schema(
        {
            "retry": {"type": ["integer", "null"], "const": None},
            "payload": {"oneOf": [{"type": "string"}, {"type": "integer"}], "const": 7},
            "state": {"const": "ready"},
            "selector": {"enum": ["alpha", 2, False], "const": False},
            "mode": {"enum": ["safe", "fast"], "const": "safe"},
        }
    )
    union_expected = {
        "retry": None,
        "payload": 7,
        "state": "ready",
        "selector": False,
        "mode": "safe",
    }
    union_tool = tool(
        "record_union_values", union_schema, "Record values with mixed JSON types"
    )
    add(
        "tool-union-types",
        "p0",
        body(
            "Call record_union_values with these exact JSON values and types: "
            + canonical(union_expected),
            tools=[union_tool],
            tool_choice="required",
        ),
        {
            "kind": "tools",
            "calls": [{"name": "record_union_values", "arguments": union_expected}],
            "reasoning": "off",
        },
        "corpus/TC-P0-08",
    )
    basic_schema = object_schema(
        {
            "language": {"type": "string", "enum": ["typescript"]},
            "steps": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "string"},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
    )
    basic_expected = {
        "language": "typescript",
        "steps": ["plan", "build"],
        "confidence": 0.91,
    }
    for name, stream, n in (
        ("strict-basic", False, 1),
        ("strict-sse", True, 1),
        ("strict-n2", False, 2),
    ):
        req = body(
            "Return language=typescript, exactly two steps [plan, build], and confidence=0.91.",
            stream=stream,
            n=n,
            response_format=strict("basic_json_schema_default", basic_schema),
        )
        req["messages"].insert(
            0,
            {
                "role": "system",
                "content": "Return only JSON conforming to the supplied schema. Do not use markdown or prose.",
            },
        )
        add(
            name,
            "p0",
            req,
            {
                "kind": "json",
                "schema": basic_schema,
                "value": basic_expected,
                "reasoning": "off",
            },
            "corpus/SO-P0-01,10; finite n2-off control",
        )
    req = copy.deepcopy(cases[-3]["body"])
    req["response_format"] = {"type": "json_object"}
    add(
        "legacy-json-object",
        "p0",
        req,
        {
            "kind": "json",
            "schema": basic_schema,
            "value": basic_expected,
            "reasoning": "off",
        },
        "corpus/SO-P0-02; local shape separately recorded",
    )
    for name, excluded in (("reasoning-on", False), ("reasoning-exclude", True)):
        req = body("Calculate 37*43 and reply with the final number.", budget=2000)
        req.pop("reasoning_effort")
        req["reasoning"] = {"effort": "low", "exclude": excluded}
        add(
            name,
            "p0",
            req,
            {
                "kind": "text",
                "contains": ["1591"],
                "reasoning": "exclude" if excluded else "on",
            },
            "protocol-repairs/reasoning-on-vs-output-excluded; finite 2000-token control",
        )
    req = body("unused", budget=256)
    req["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the dominant color in this image? Reply with one English color word.",
                },
                {"type": "image_url", "image_url": {"url": red_png()}},
            ],
        }
    ]
    add(
        "image-red",
        "p0",
        req,
        {"kind": "text", "text": "red", "casefold": True, "reasoning": "off"},
        "finite vision capability control/generated 64x64 RGB PNG",
    )
    add(
        "cancel-sse",
        "p0",
        body(
            "Write the integers from 1 to 1000, one per line, without commentary.",
            budget=1024,
            stream=True,
        ),
        {"kind": "cancel", "cancel_after_semantic_events": 3},
        "finite cancellation mechanism; backend-release proof external",
    )
    add(
        "post-cancel-chat",
        "p0",
        body("Reply with exactly RECOVERED.", budget=64),
        {"kind": "text", "text": "RECOVERED", "reasoning": "off"},
        "post-cancel service control",
    )
    value = {
        "schema": "phala.dsv41.protocol105.v2",
        "wire_serialization": "UTF-8 compact JSON, insertion order preserved; canonical hashes remain order-insensitive",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "headers_without_auth": {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        "generation": {
            "seed": SEED,
            "generator": "Python random.Random.choice/randint",
            "python_version": sys.version.split()[0],
        },
        "adaptations": [
            "Original GLM-5.3 conflict fixture changes only model and endpoint; all 35 request constraints/prompts/schema preserved.",
            "P0 is a finite selected subset, one attempt per case, with declared reasoning-off control; not a full 22-case/repetition/cross-model corpus pass.",
            "No inference model seed is added; no retry, no budget escalation, no answer repair.",
            "Direct localhost backend proves only this boundary; external logs/metrics must prove cancellation cleanup and absence of engine restart.",
        ],
        "request_order": [case["case_id"] for case in cases],
        "cases": cases,
    }
    value["payload_sha256"] = sha(value)
    return value


if __name__ == "__main__":
    serialized = json.dumps(manifest(), ensure_ascii=False, indent=2) + "\n"
    if len(sys.argv) == 2:
        # Write the generated artifact directly; terminal output may truncate it.
        with open(sys.argv[1], "x", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
        print("Generated ordered-wire manifest")
    else:
        print(serialized, end="")
